import logging
import os

import psutil
import torch
import transformers
from mpmath import isint

from . import utils

OPT_MODEL = transformers.models.opt.modeling_opt.OPTForCausalLM
OPT_LAYER = transformers.models.opt.modeling_opt.OPTDecoderLayer
LLAMA_MODEL = transformers.models.llama.modeling_llama.LlamaForCausalLM
LLAMA_LAYER = transformers.models.llama.modeling_llama.LlamaDecoderLayer
QWEN_MODEL = transformers.models.qwen2.modeling_qwen2.Qwen2ForCausalLM
QWEN3MOE_MODEL = transformers.models.qwen3_moe.modeling_qwen3_moe.Qwen3MoeForCausalLM
QWEN_LAYER = transformers.models.qwen2.modeling_qwen2.Qwen2DecoderLayer
QWEN2_5_VL_MODEL = transformers.models.qwen2_5_vl.Qwen2_5_VLForConditionalGeneration
QWEN3_MODEL = transformers.Qwen3ForCausalLM


def model_type_extractor(model):
    if isinstance(model, LLAMA_MODEL):
        return LLAMA_MODEL
    elif isinstance(model, OPT_MODEL):
        return OPT_MODEL
    elif isinstance(model, QWEN_MODEL):
        return QWEN_MODEL
    elif isinstance(model, QWEN3MOE_MODEL):
        return QWEN3MOE_MODEL
    elif isinstance(model, QWEN3_MODEL):
        return QWEN3_MODEL
    elif isinstance(model, QWEN2_5_VL_MODEL):
        return QWEN2_5_VL_MODEL
    else:
        raise ValueError(f"Unknown model type {model}")


def skip(*args, **kwargs):
    # This is a helper function to save time during the initialization!
    pass


def get_rope_function_name(model):
    if isinstance(model, LLAMA_MODEL) or isinstance(model, QWEN_MODEL):
        return "apply_rotary_pos_emb"
    raise NotImplementedError


def get_layers(model):
    if isinstance(model, OPT_MODEL):
        return model.model.decoder.layers
    if isinstance(model, LLAMA_MODEL) or isinstance(model, QWEN_MODEL):
        return model.model.layers
    raise NotImplementedError


def get_llama(model_name, hf_token):
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    model = transformers.LlamaForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="cuda",
        use_auth_token=hf_token,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    model.seqlen = 2048
    logging.info("---> Loading {} Model with seq_len: {}".format(model_name, model.seqlen))
    return model


def get_qwen(model_name, hf_token, args):
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    config = transformers.Qwen2Config.from_pretrained(model_name)

    # for qwen2-1.5b & 3b
    if args.tie_embed_head:
        config.tie_word_embeddings = False

    model = transformers.Qwen2ForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="cuda",
        config=config,
        trust_remote_code=True,
        attn_implementation="eager",
    )

    if args.tie_embed_head:
        import copy

        model.lm_head.weight.data = copy.deepcopy(model.model.embed_tokens.weight.data)

    model.seqlen = 2048
    logging.info("---> Loading {} Model with seq_len: {}".format(model_name, model.seqlen))
    return model


def get_opt(model_name):
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    model = transformers.OPTForCausalLM.from_pretrained(model_name, torch_dtype="auto", low_cpu_mem_usage=True)
    model.seqlen = model.config.max_position_embeddings
    logging.info("---> Loading {} Model with seq_len: {}".format(model_name, model.seqlen))
    return model


def get_model(model_name, hf_token=None, args=None):
    if "llama" in model_name:
        return get_llama(model_name, hf_token)
    elif "opt" in model_name:
        return get_opt(model_name)
    elif "qwen" in model_name.lower():
        return get_qwen(model_name, hf_token, args)
    else:
        raise ValueError(f"Unknown model {model_name}")


def get_model_type(model):
    if isinstance(model, OPT_MODEL):
        model_type = OPT_MODEL
    elif isinstance(model, LLAMA_MODEL):
        model_type = LLAMA_MODEL
    elif isinstance(model, QWEN_MODEL):
        model_type = QWEN_MODEL
    elif isinstance(model, QWEN3_MODEL):
        model_type = QWEN3_MODEL
    elif isinstance(model, QWEN2_5_VL_MODEL):
        model_type = QWEN2_5_VL_MODEL
    elif isinstance(model, QWEN3MOE_MODEL):
        model_type = QWEN3MOE_MODEL
    else:
        raise ValueError(f"Unknown model type {model}")
    return model_type


def get_embeddings(model, model_type):
    if model_type in [LLAMA_MODEL, QWEN_MODEL, QWEN3_MODEL, QWEN2_5_VL_MODEL, QWEN3MOE_MODEL]:
        return [model.model.embed_tokens]
    elif model_type == OPT_MODEL:
        return [model.model.decoder.embed_tokens, model.model.decoder.embed_positions]
    else:
        raise ValueError(f"Unknown model type {model_type}")


def get_transformer_layers(model, model_type):
    if model_type in [LLAMA_MODEL, QWEN_MODEL, QWEN3_MODEL, QWEN2_5_VL_MODEL, QWEN3MOE_MODEL]:
        return [layer for layer in model.model.layers]
    elif model_type == OPT_MODEL:
        return [layer for layer in model.model.decoder.layers]
    else:
        raise ValueError(f"Unknown model type {model_type}")


def get_lm_head(model, model_type):
    if model_type in [LLAMA_MODEL, QWEN_MODEL, QWEN3_MODEL, QWEN2_5_VL_MODEL, QWEN3MOE_MODEL]:
        return model.lm_head
    elif model_type == OPT_MODEL:
        return model.lm_head
    else:
        raise ValueError(f"Unknown model type {model_type}")


def get_pre_head_layernorm(model, model_type):
    if model_type == LLAMA_MODEL:
        pre_head_layernorm = model.model.norm
        assert isinstance(pre_head_layernorm, transformers.models.llama.modeling_llama.LlamaRMSNorm)
    elif model_type in [QWEN_MODEL, QWEN3_MODEL, QWEN2_5_VL_MODEL, QWEN3MOE_MODEL]:
        pre_head_layernorm = model.model.norm
    elif model_type == OPT_MODEL:
        pre_head_layernorm = model.model.decoder.final_layer_norm
        assert pre_head_layernorm is not None
    else:
        raise ValueError(f"Unknown model type {model_type}")
    return pre_head_layernorm


def get_mlp_bottleneck_size(model):
    model_type = get_model_type(model)
    if (
        model_type == LLAMA_MODEL
        or model_type == QWEN_MODEL
        or model_type == QWEN3_MODEL
        or model_type == QWEN3MOE_MODEL
    ):
        return model.config.intermediate_size
    elif model_type == OPT_MODEL:
        return model.config.ffn_dim
    else:
        raise ValueError(f"Unknown model type {model_type}")


def replace_modules(
    root: torch.nn.Module,
    type_to_replace,
    new_module_factory,
    replace_layers: bool,
) -> None:
    """Replace modules of given type using the supplied module factory.

    Perform a depth-first search of a module hierarchy starting at root
    and replace all instances of type_to_replace with modules created by
    new_module_factory. Children of replaced modules are not processed.

    Args:
        root: the root of the module hierarchy where modules should be replaced
        type_to_replace: a type instances of which will be replaced
        new_module_factory: a function that given a module that should be replaced
            produces a module to replace it with.
    """
    for name, module in root.named_children():
        new_module = None
        if isinstance(module, type_to_replace):
            if replace_layers:  # layernorm_fusion.replace_layers case where transformer layers are replaced
                new_module = new_module_factory(module, int(name))
            else:  # layernorm_fusion.fuse_modules case where layernorms are fused
                new_module = new_module_factory(module)
        elif len(list(module.children())) > 0:
            replace_modules(module, type_to_replace, new_module_factory, replace_layers)

        if new_module is not None:
            setattr(root, name, new_module)


class RMSN(torch.nn.Module):
    """
    This class implements the Root Mean Square Normalization (RMSN) layer.
    We use the implementation from LLAMARMSNorm here:
    https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L75
    """

    def __init__(self, mean_dim: int, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.mean_dim = mean_dim
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        if x.dtype == torch.float16:
            x = x.to(torch.float32)
        variance = x.pow(2).sum(-1, keepdim=True) / self.mean_dim
        x = x * torch.rsqrt(variance + self.eps)
        return x.to(input_dtype)


def get_layer_io_save_path(args):
    return os.path.join(args.save_path, "layer_io", f"{args.layer_idx:03d}.pt")


def capture_layer_io(model_type, layer, layer_input):
    def hook_factory(module_name, captured_vals, is_input):
        def hook(module, input, output):
            if is_input:
                captured_vals[module_name].append(input[0].detach().cpu())
            else:
                captured_vals[module_name].append(output.detach().cpu())

        return hook

    handles = []

    if model_type == LLAMA_MODEL:
        captured_inputs = {
            "k_proj": [],  # q_proj, v_proj has the same input as k_proj
            "o_proj": [],
            "gate_proj": [],  # up_proj has the same input as gate_proj
            "down_proj": [],
        }

        captured_outputs = {
            "v_proj": [],
        }

        for name in captured_inputs.keys():
            module = getattr(layer.self_attn, name, None) or getattr(layer.mlp, name, None)
            handles.append(module.register_forward_hook(hook_factory(name, captured_inputs, True)))

        for name in captured_outputs.keys():
            module = getattr(layer.self_attn, name, None) or getattr(layer.mlp, name, None)
            handles.append(module.register_forward_hook(hook_factory(name, captured_outputs, False)))

    elif model_type == OPT_MODEL:
        captured_inputs = {
            "k_proj": [],  # q_proj, v_proj has the same input as k_proj
            "out_proj": [],
            "fc1": [],
            "fc2": [],
        }
        captured_outputs = {
            "v_proj": [],
        }
        for name in captured_inputs.keys():
            # In OPT, fc1 and fc2 are directly contained in OPTDecoderLayer
            module = getattr(layer.self_attn, name, None) or getattr(layer, name, None)
            handles.append(module.register_forward_hook(hook_factory(name, captured_inputs, True)))

        for name in captured_outputs.keys():
            # In OPT, fc1 and fc2 are directly contained in OPTDecoderLayer
            module = getattr(layer.self_attn, name, None) or getattr(layer, name, None)
            handles.append(module.register_forward_hook(hook_factory(name, captured_outputs, False)))
    else:
        raise ValueError(f"Unknown model type {model_type}")

    # Process each sequence in the batch one by one to avoid OOM.
    for seq_idx in range(layer_input.shape[0]):
        # Extract the current sequence across all dimensions.
        seq = layer_input[seq_idx : seq_idx + 1].to(utils.DEV)
        # Perform a forward pass for the current sequence.
        layer(seq)

    # After processing all sequences, concatenate the accumulated inputs for each sub-layer across the batch.
    for module_name in captured_inputs:
        captured_inputs[module_name] = torch.cat(captured_inputs[module_name], dim=0)
    for module_name in captured_outputs:
        captured_outputs[module_name] = torch.cat(captured_outputs[module_name], dim=0)

    # Cleanup.
    for h in handles:
        h.remove()

    return {"input": captured_inputs, "output": captured_outputs}


def get_system_memory_info():
    """
    获取系统内存信息

    Returns:
        dict: 包含以下内存信息：
            - total: 总物理内存（GB）
            - available: 可用内存（GB）
            - used: 已使用内存（GB）
            - percent: 内存使用百分比
    """
    memory = psutil.virtual_memory()

    # 转换为GB并保留2位小数
    GB = 1024 * 1024 * 1024
    memory_info = {
        "total": round(memory.total / GB, 2),
        "available": round(memory.available / GB, 2),
        "used": round(memory.used / GB, 2),
        "percent": memory.percent,
    }

    return memory_info


def get_process_memory_info():
    """
    获取当前进程的内存使用信息

    Returns:
        dict: 包含以下内存信息：
            - rss: 物理内存使用（GB）
            - vms: 虚拟内存使用（GB）
    """
    process = psutil.Process()
    memory_info = process.memory_info()

    GB = 1024 * 1024 * 1024
    return {"rss": round(memory_info.rss / GB, 2), "vms": round(memory_info.vms / GB, 2)}  # 物理内存  # 虚拟内存
