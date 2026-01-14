import math
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import tqdm
from safetensors.torch import load_file as load_safetensors_file
from safetensors.torch import save_file as safetensors_save_file
from traitlets import HasTraits

from . import quant_utils, utils
import torch.nn.functional as F

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


class GPTQ:
    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0
        # self.scales = torch.zeros(self.rows, int(self.columns/64)).to(self.dev)
        self.W_int = None
        self.quantizer = None
        self.damp_auto_increment = 0.01

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        inp = inp.t()
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        # inp = inp.float()
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        # self.H += 2 / self.nsamples * inp.matmul(inp.t())
        self.H += inp.matmul(inp.t())

    def fasterquant(
        self,
        blocksize=128,
        percdamp=0.01,
        groupsize=-1,
        actorder=False,
        static_groups=False,
        use_hession_mse=True,
    ):
        W = self.layer.weight.data.clone()
        W = W.float()

        if not self.quantizer.ready():
            self.quantizer.find_params(W)

        H = self.H
        # del self.H
        # torch.nan_to_num_(H, nan=0.)
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        if static_groups:
            import copy

            groups = []
            for i in range(0, self.columns, groupsize):
                quantizer = copy.deepcopy(self.quantizer)
                quantizer.find_params(W[:, i : (i + groupsize)])
                groups.append(quantizer)

        if actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]
            invperm = torch.argsort(perm)

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        mean = torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)

        while 0 < percdamp < 1:
            try:
                H2 = H.clone()
                H2[diag, diag] += percdamp * mean
                H2 = torch.linalg.cholesky(H2)
                H2 = torch.cholesky_inverse(H2)
                Hinv = torch.linalg.cholesky(H2, upper=True)
                # del H, H2
                break
            except Exception as e:
                percdamp += self.damp_auto_increment

        if not (0 < percdamp < 1):
            raise ValueError(f"Failed to find Hinv with percdamp {percdamp}")

        # scales_list = []
        W_int = torch.zeros_like(W)
        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if groupsize != -1:
                    if not static_groups:
                        if (i1 + i) % groupsize == 0:
                            if use_hession_mse:
                                self.quantizer.find_params(W[:, (i1 + i) : (i1 + i + groupsize)], H = self.H[(i1 + i) : (i1 + i + groupsize), (i1 + i) : (i1 + i + groupsize)])
                            else:
                                self.quantizer.find_params(W[:, (i1 + i) : (i1 + i + groupsize)])
                    else:
                        idx = i1 + i
                        if actorder:
                            idx = perm[idx]
                        self.quantizer = groups[idx // groupsize]
                # scales_list.append(self.quantizer.scale)
                w_int = torch.clamp(
                    torch.round(w.unsqueeze(1) / self.quantizer.scale.to(w.device)),
                    -(self.quantizer.maxq + 1),
                    self.quantizer.maxq,
                )
                W_int[:, i1 + i] = w_int.squeeze(1)
                q = self.quantizer.quantize(w.unsqueeze(1)).flatten()
                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d**2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        torch.cuda.synchronize()
        # reset the scale
        # for col in range(0,self.columns,groupsize):
        #     self.scales[:,int(col//groupsize)] = scales_list[col].squeeze(1)
        self.W_int = W_int

        if actorder:
            Q = Q[:, invperm]

        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        if torch.any(torch.isnan(self.layer.weight.data)):
            print("NaN in weights")
            import pprint

            pprint.pprint(self.quantizer.bits, self.quantizer.scale, self.quantizer.zero_point)
            raise ValueError("NaN in weights")

    def free(self):
        self.H = None
        self.Losses = None
        self.Trace = None
        torch.cuda.empty_cache()
        utils.cleanup_memory(verbose=False)


def load_layer_cache(cache_file: str, layer: nn.Module) -> None:
    # cache_state_dict = torch.load(cache_file, weights_only=True, map_location="cpu")
    cache_state_dict = load_safetensors_file(cache_file, device="cpu")
    for k in cache_state_dict:
        paths = k.split(".")
        if paths[-1] == "quant_weight":
            submodule_name = ".".join(paths[:-1])
            submodule = layer.get_submodule(submodule_name)
            submodule.register_parameter("quant_weight", nn.Parameter(cache_state_dict[k], requires_grad=False))

    layer.load_state_dict(cache_state_dict)
    del cache_state_dict
    cache_state_dict = None


def process_qwen2_5_vl_batch(batch, device, processor):
    from qwen_vl_utils import process_vision_info

    text = processor.apply_chat_template(batch, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(batch)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(device)
    return inputs

class Qwen3VLMoeTextExperts_Linearized(nn.Module):
    def __init__(self, ori_experts):
        super().__init__()
        self.num_experts = ori_experts.num_experts
        self.intermediate_size = ori_experts.intermediate_size
        self.hidden_size = ori_experts.hidden_size
        self.expert_dim = ori_experts.expert_dim
        self.act_fn =ori_experts.act_fn

        self.gate_up_proj = nn.ModuleList([
            nn.Linear(self.hidden_size, 2 * self.expert_dim, bias=False)
            for _ in range(self.num_experts)
        ])
        self.down_proj = nn.ModuleList([
            nn.Linear(self.expert_dim, self.hidden_size, bias=False)
            for _ in range(self.num_experts)
        ])
        if ori_experts is not None:
            for i in range(self.num_experts):
                # self.gate_up_proj[i].weight.copy_(ori_experts.gate_up_proj[i].T.contiguous())
                # self.down_proj[i].weight.copy_(ori_experts.down_proj[i].T.contiguous())
                self.gate_up_proj[i].weight.data = ori_experts.gate_up_proj[i].T.contiguous().data
                self.down_proj[i].weight.data = ori_experts.down_proj[i].T.contiguous().data


    def forward(
        self, hidden_states: torch.Tensor, routing_weights: torch.Tensor, router_indices: torch.Tensor
    ) -> torch.Tensor:
        batch_size = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(-1, self.hidden_size)
        # 推理模式：并行执行所有 expert
        hidden_states = hidden_states.repeat(self.num_experts, 1)
        hidden_states = hidden_states.view(self.num_experts, -1, self.hidden_size)

        outs = []
        for i in range(self.num_experts):
            gate_up = self.gate_up_proj[i](hidden_states[i])
            gate, up = gate_up.chunk(2, dim=-1)
            out = self.down_proj[i](up * self.act_fn(gate))
            outs.append(out)

        next_states = torch.stack(outs, dim=0)
        next_states = next_states.reshape(self.num_experts, batch_size, -1, self.hidden_size)
        next_states = (
            next_states * routing_weights.transpose(0, 1).view(self.num_experts, batch_size, -1)[..., None]
        )
        next_states = next_states.sum(dim=0)

        return next_states

@torch.no_grad()
def gptq_fwrd(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    nsamples: int = 128,
    seqlen: int = 2048,
    w_clip: bool = True,
    w_bits: int = 4,
    w_head_bits: int = 8,
    w_asym: bool = False,
    w_groupsize: int = 64,
    percdamp: float = 0.01,
    act_order: bool = False,
    int8_down_proj: bool = False,
    heading_gptq: bool = False,
    device: Optional[torch.device] = None,
    ddevice: Optional[torch.device] = None,
    layers_cache_dir: Optional[str] = None,
    is_qwen2_5_vl: bool = False,
    processor=None,
    tokenizer=None,
    is_qwen3_vl=False,
    is_moe=False,
    use_hession_mse=False,
    self_attn_weight = None
) -> Dict[str, Any]:
    """
    From GPTQ repo
    TODO: Make this function general to support both OPT and LLaMA models
    """
    if device is None:
        device = utils.DEV

    if is_qwen2_5_vl or is_qwen3_vl:
        use_cache = model.config.text_config.use_cache
        model.config.text_config.use_cache = False
        layers = model.model.language_model.layers
    else:
        use_cache = model.config.use_cache
        model.config.use_cache = False
        layers = model.model.layers

    if is_qwen2_5_vl or is_qwen3_vl:
        model.model.visual = model.model.visual.to(device)

    if is_qwen2_5_vl or is_qwen3_vl:
        model.model.language_model.embed_tokens = model.model.language_model.embed_tokens.to(device)
        model.model.language_model.norm = model.model.language_model.norm.to(device)
        model.model.language_model.rotary_emb = model.model.language_model.rotary_emb.to(device)
    else:
        model.model.embed_tokens = model.model.embed_tokens.to(device)
        model.model.norm = model.model.norm.to(device)
        model.model.rotary_emb = model.model.rotary_emb.to(device)

    layers[0] = layers[0].to(device)
    dtype = model.lm_head.weight.dtype
    # dtype = next(iter(model.parameters())).dtype

    if ddevice is None:
        ddevice = torch.device("cpu")
    nsamples = len(dataloader)
    if is_qwen2_5_vl or is_qwen3_vl:
        inps = list()
        attention_mask = list()
        position_ids = list()
        position_embeddings = list()
    else:
        inps = torch.zeros((nsamples, seqlen, model.config.hidden_size), dtype=dtype, device=ddevice)
    cache = {"i": 0, "attention_mask": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            if is_qwen2_5_vl:
                self.attention_type = module.attention_type
            if hasattr(self.module, "attention_type"):
                self.attention_type = module.attention_type

        def forward(self, inp, **kwargs):
            if is_qwen2_5_vl or is_qwen3_vl:
                inps.append(inp.to(ddevice))
                attention_mask.append(kwargs["attention_mask"])
                position_ids.append(kwargs["position_ids"])
                position_embeddings.append(kwargs["position_embeddings"])
            else:
                inps[cache["i"]] = inp.to(ddevice)
                cache["attention_mask"] = kwargs["attention_mask"]
                cache["position_ids"] = kwargs["position_ids"]
                if "position_embeddings" in kwargs:
                    cache["position_embeddings"] = kwargs["position_embeddings"]
                else:
                    cache["position_embeddings"] = None
            cache["i"] += 1
            raise ValueError

    # 获取第一个layer的输入用于标定样本
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            if is_qwen2_5_vl or is_qwen3_vl:
                inputs = process_qwen2_5_vl_batch(batch, device, processor)
                model.generate(**inputs, max_new_tokens=1)
            else:
                if tokenizer is not None:
                    texts = batch["text"]
                    queries = [query for query in texts]
                    inputs = tokenizer(queries, return_tensors="pt", truncation=True, max_length=seqlen,padding=True).to(device)
                    inputs['input_ids'] =  F.pad(inputs['input_ids'],(0,seqlen-inputs['input_ids'].shape[1]))
                    model(inputs['input_ids'])
                else:
                    model(batch[0].to(device))
        except ValueError:
            pass
    
    layers[0] = layers[0].module.cpu()
    if is_qwen2_5_vl or is_qwen3_vl:
        model.model.language_model.embed_tokens = model.model.language_model.embed_tokens.cpu()
        model.model.language_model.norm = model.model.language_model.norm.cpu()
        model.model.language_model.rotary_emb = model.model.language_model.rotary_emb.cpu()
        model.model.visual.cpu()
    else:
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        model.model.norm = model.model.norm.cpu()
        model.model.rotary_emb = model.model.rotary_emb.cpu()
    torch.cuda.empty_cache()

    if is_qwen2_5_vl or is_qwen3_vl:
        outs = list()
    else:
        outs = torch.zeros_like(inps)
        attention_mask = cache["attention_mask"]
        position_ids = cache["position_ids"]
        position_embeddings = cache["position_embeddings"]

    quantizers: Dict[str, Any] = {}
    if is_moe:
        names_gate_up = []
        names_down = []
        for i in range(128):
            name_gate = 'mlp.experts.gate_up_proj.'+str(i)
            name_down =  'mlp.experts.down_proj.'+str(i)
            names_gate_up.append(name_gate)
            names_down.append(name_down)
        sequential =  [
        [
            "self_attn.k_proj.module",
            "self_attn.v_proj.module",
            "self_attn.q_proj.module",
        ],
        ["self_attn.o_proj.module"],
        ["mlp.gate.module"]+names_gate_up,
        names_down,
    ]
    else:
        sequential = [
            [
                "self_attn.k_proj.module",
                "self_attn.v_proj.module",
                "self_attn.q_proj.module",
            ],
            ["self_attn.o_proj.module"],
            ["mlp.up_proj.module", "mlp.gate_proj.module"],
            ["mlp.down_proj.module"],
        ]

    # 加载输入cache
    inps_cache_file: Optional[str] = None
    if layers_cache_dir is not None:
        inps_cache_file = str(Path(layers_cache_dir) / f"inps.pt")
        # if Path(inps_cache_file).exists():
        #     inps = torch.load(inps_cache_file, map_location=inps.device, weights_only=True)

    for i in range(len(layers)):
        print(f"\nLayer {i}:", flush=True, end=" ")

        layer_cache_file: Optional[str] = None
        if layers_cache_dir is not None:
            layer_cache_file = str(Path(layers_cache_dir) / f"layer_{i}.safetensors")

        if layer_cache_file is not None and Path(layer_cache_file).exists():
            load_layer_cache(layer_cache_file, layers[i])
            print(f"from cache: {layer_cache_file}", flush=True, end=" ")
            try:
                import os

                import psutil

                process = psutil.Process(os.getpid())
                mem_info = process.memory_info()
                print(f"mem: {mem_info.rss / 1024 / 1024 / 1024:.2f} GB", flush=True, end=" ")
            except Exception:
                pass
            continue
        if is_moe:
            layers[i].mlp.experts = Qwen3VLMoeTextExperts_Linearized(layers[i].mlp.experts)
        if is_qwen2_5_vl or is_qwen3_vl:
            layer = layers[i].to(device=device, dtype=torch.float16) # use flash attention 2
        else:
            layer = layers[i].to(device=device, dtype=torch.float32)

        full = quant_utils.find_qlayers(layer, layers=[torch.nn.Linear])

        for names in sequential:
            subset = {n: full[n.replace('.module', '', 1)] for n in names}
            gptq: Dict[str, GPTQ] = {}
            for name in subset:
                print(f"{name}", end="  ", flush=True)
                layer_weight_bits = w_bits
                layer_weight_sym = not (w_asym)
                if self_attn_weight is not None and "self_attn" in name:
                    layer_weight_bits = self_attn_weight
                if "lm_head" in name:
                    layer_weight_bits = 16
                    continue
                if int8_down_proj and "down_proj" in name:
                    layer_weight_bits = 8
                gptq[name] = GPTQ(subset[name])
                gptq[name].quantizer = quant_utils.WeightQuantizer()
                gptq[name].quantizer.configure(
                    layer_weight_bits,
                    perchannel=True,
                    sym=layer_weight_sym,
                    mse=w_clip,
                )

            def add_batch(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)

                return tmp

            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))

            if is_qwen2_5_vl or is_qwen3_vl:
                outs = list()
            for j in range(nsamples):
                if is_qwen2_5_vl:
                    outs.append(
                        layer(
                            inps[j].to(device=device),
                            attention_mask=attention_mask[j],
                            position_ids=position_ids[j],
                            position_embeddings=position_embeddings[j],
                        )[0].to(device=ddevice, dtype=dtype)
                    )
                elif is_qwen3_vl:
                    outs.append(
                        layer(
                            inps[j].to(device=device),
                            attention_mask=attention_mask[j],
                            position_ids=position_ids[j],
                            position_embeddings=position_embeddings[j],
                        ).to(device=ddevice, dtype=dtype)
                    )
                else:
                    outs[j] = layer(
                        inps[j].unsqueeze(0).to(device=device, dtype=torch.float32),
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        position_embeddings=position_embeddings,
                    )[0].to(device=ddevice, dtype=dtype)

            for h in handles:
                h.remove()

            for name in subset:
                layer_w_groupsize = w_groupsize
                gptq[name].fasterquant(
                    percdamp=percdamp,
                    groupsize=layer_w_groupsize,
                    actorder=act_order,
                    static_groups=False,
                    use_hession_mse=use_hession_mse,
                )

                quant_value = gptq[name].W_int
                if quant_value.max().item() < pow(2, 7) and quant_value.min().item() >= -pow(2, 7):
                    quant_value = quant_value.to(torch.int8)
                elif quant_value.max().item() < pow(2, 15) and quant_value.min().item() >= -pow(2, 15):
                    quant_value = quant_value.to(torch.int16)
                subset[name].register_parameter("quant_weight", nn.Parameter(quant_value, requires_grad=False))
                gptq[name].free()

        if True:
            for j in range(nsamples):
                if is_qwen2_5_vl:
                    outs[j] = layer(
                        inps[j].to(device=device, dtype=torch.float16),
                        attention_mask=attention_mask[j],
                        position_ids=position_ids[j],
                        position_embeddings=position_embeddings[j],
                    )[0].to(device=ddevice, dtype=dtype)
                elif is_qwen3_vl:
                    outs[j] = layer(
                        inps[j].to(device=device, dtype=torch.float16),
                        attention_mask=attention_mask[j],
                        position_ids=position_ids[j],
                        position_embeddings=position_embeddings[j],
                    ).to(device=ddevice, dtype=dtype)
                else:
                    outs[j] = layer(
                        inps[j].unsqueeze(0).to(device=device, dtype=torch.float32),
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        position_embeddings=position_embeddings,
                    )[0].to(device=ddevice, dtype=dtype)

        inps, outs = outs, inps
        if layer_cache_file is not None:
            layer_state_dict = layer.cpu().state_dict()
            safetensors_save_file(layer_state_dict, layer_cache_file)
        # 保存下一个layer的输入
        if inps_cache_file is not None:
            torch.save(inps, inps_cache_file)

        layers[i] = layer.cpu()
        del layer
        del gptq
        torch.cuda.empty_cache()

    # head gptq
    if heading_gptq:
        print(f"\nhead:", flush=True, end=" ")
        head_layer_cache_file: Optional[str] = None
        if layers_cache_dir is not None:
            head_layer_cache_file = str(Path(layers_cache_dir) / f"lm_head.safetensors")

        if head_layer_cache_file is not None and Path(head_layer_cache_file).exists():
            load_layer_cache(head_layer_cache_file, model.lm_head)
            print(f"from cache: {head_layer_cache_file}")
        else:
            # Convert to module and move to CUDA
            if is_qwen2_5_vl or is_qwen3_vl:
                model.lm_head = model.lm_head.to(device=device, dtype=torch.float16)
            else:
                model.lm_head = model.lm_head.to(device=device, dtype=torch.float32)
            gptq = {}
            name = "lm_head"
            layer_weight_bits = w_head_bits
            layer_weight_sym = not (w_asym)
            gptq[name] = GPTQ(model.lm_head)
            gptq[name].quantizer = quant_utils.WeightQuantizer()
            gptq[name].quantizer.configure(layer_weight_bits, perchannel=True, sym=layer_weight_sym, mse=w_clip)

            def add_batch(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)

                return tmp

            handles = []
            handles.append(model.lm_head.register_forward_hook(add_batch("lm_head")))
            for j in range(nsamples):
                if is_qwen2_5_vl or is_qwen3_vl:
                    model.lm_head(inps[j].to(device=device, dtype=torch.float16))
                else:
                    model.lm_head(inps[j].unsqueeze(0).to(device=device, dtype=torch.float32))
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            for h in handles:
                h.remove()

            layer_w_groupsize = w_groupsize
            gptq[name].fasterquant(
                percdamp=percdamp,
                groupsize=layer_w_groupsize,
                actorder=act_order,
                static_groups=False,
                use_hession_mse=False,
            )

            quant_value = gptq[name].W_int
            if quant_value.max().item() < pow(2, 7) and quant_value.min().item() >= -pow(2, 7):
                quant_value = quant_value.to(torch.int8)
            elif quant_value.max().item() < pow(2, 15) and quant_value.min().item() >= -pow(2, 15):
                quant_value = quant_value.to(torch.int16)
            model.lm_head.register_parameter("quant_weight", nn.Parameter(quant_value, requires_grad=False))
            gptq[name].free()
            model.lm_head = model.lm_head.cpu()

            if layers_cache_dir is not None and head_layer_cache_file is not None:
                head_state_dict = model.lm_head.state_dict()
                safetensors_save_file(head_state_dict, str(head_layer_cache_file))  # Ensure string path
            del gptq
            torch.cuda.empty_cache()

    model.config.use_cache = use_cache
    utils.cleanup_memory(verbose=True)
    return quantizers


@torch.no_grad()
def rtn_fwrd(model: nn.Module, dev: torch.device, args: Any) -> Dict[str, Any]:
    """
    From GPTQ repo
    TODO: Make this function general to support both OPT and LLaMA models
    """
    assert args.w_groupsize == -1, "Groupsize not supported in RTN!"
    layers = model.model.layers
    torch.cuda.empty_cache()

    quantizers = {}

    for i in tqdm.tqdm(range(len(layers)), desc="(RtN Quant.) Layers"):
        layer = layers[i].to(dev)

        subset = quant_utils.find_qlayers(layer, layers=[torch.nn.Linear])

        for name in subset:
            layer_weight_bits = args.w_bits
            if "lm_head" in name:
                layer_weight_bits = 16
                continue
            if args.int8_down_proj and "down_proj" in name:
                layer_weight_bits = 8

            quantizer = quant_utils.WeightQuantizer()
            quantizer.configure(
                layer_weight_bits,
                perchannel=True,
                sym=not (args.w_asym),
                mse=args.w_clip,
            )
            W = subset[name].weight.data
            quantizer.find_params(W)
            subset[name].weight.data = quantizer.quantize(W).to(layer.self_attn.q_proj.weight.dtype)
            quantizers["model.layers.%d.%s" % (i, name)] = quantizer.cpu()
        layers[i] = layer.cpu()
        torch.cuda.empty_cache()
        del layer

    utils.cleanup_memory(verbose=True)
    return quantizers
