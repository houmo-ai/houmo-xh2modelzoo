import torch
import torch.nn as nn

from contextlib import nullcontext
import copy
import os
import gc
from .utils import *

def get_named_linears(module):
    from .int_linear import QuantLinear
    return {name: m for name, m in module.named_modules() if isinstance(m, QuantLinear)}


def add_new_module(name, original_module, added_module):
    levels = name.split('.')
    if len(levels) > 1:
        mod_ = original_module
        for l_idx in range(len(levels)-1):
            if levels[l_idx].isdigit():
                mod_ = mod_[int(levels[l_idx])]
            else:
                mod_ = getattr(mod_, levels[l_idx])
        setattr(mod_, levels[-1], added_module)
    else:
        setattr(original_module, name, added_module)     


def singlequant(
    lm,
    args,
    dataloader,
    logger=None,
):
    logger.info("Starting ...")
    from .int_linear import QuantLinear
    
    # move embedding layer and first layer to target device
    model = lm.model
    dev = lm.device
    use_cache = model.config.use_cache
    model.config.use_cache = False
    is_llama = False
    if "qwen" in args.net.lower():
        from .int_qwen_layer import QuantQwen3DecoderLayer
        is_llama = True
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
        DecoderLayer = QuantQwen3DecoderLayer
        pairs = {
            "q_proj":"qkv",
            "o_proj":"out",
            "up_proj":"fc1",
            "down_proj":"down",
        }
        layer_name_prefix = "model.layers"
    else:
        raise ValueError(f"Unsupported model: {args.net}")
    
    layers[0] = layers[0].to(dev)
    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)

    layers[0] = layers[0].to(dev)
    if args.deactive_amp:
        dtype = torch.float
        traincast = nullcontext
    else:
        dtype = torch.float16
        traincast = torch.cuda.amp.autocast
    inps = torch.zeros(
        (args.nsamples, lm.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            self.is_llama = False

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            if self.is_llama:
                cache["position_ids"] = kwargs["position_ids"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    layers[0].is_llama = is_llama
    input_ids = []

    with torch.no_grad():
        for batch in dataloader:
            if cache["i"] >= args.nsamples:
                break
            try:
                input_ids.append(batch[0])
                model(batch[0].to(dev))
            except ValueError:
                pass

    
    # move embedding layer and first layer to cpu
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()

    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    rotate_inps = copy.copy(inps).mean(dim=0)

    
    attention_mask = cache["attention_mask"]

    if is_llama:
        position_ids = cache["position_ids"]
    else:
        position_ids = None


    singlequant_parameters = {}

    for i in range(len(layers)):
        for name in ['q', 'k', 'v', 'gate', 'up', 'down', 'o']:
            exec(f"args.{name}_weight_quant_params = copy.copy(args.weight_quant_params)")
            exec(f"args.{name}_act_quant_params = copy.copy(args.act_quant_params)")

        args.q_quant_params = copy.copy(args.act_quant_params)
        args.k_quant_params = copy.copy(args.act_quant_params)

        logger.info(f"=== Start quantize layer {i} ===")
        layer = layers[i]
        qlayer = DecoderLayer(lm.model.config, layer, i, args)
                
        qlayer = qlayer.to(dev)        
        # if torch.cuda.device_count() > 1:
        #     qlayer.mlp.to("cuda:1")
        if args.quant_method == 'singlequant':
            set_init_singlequant_params_state(qlayer, True)

        set_quant_state(qlayer, weight_quant=False, act_quant=False)
        # init smooth parameters
        set_quant_state(qlayer, weight_quant=False, act_quant=True)  # weight will be manually quantized before forward

        qlayer.half()
        try:
            with torch.no_grad():
                qlayer.qkt_smooth_scale.clamp_(min=0.5)
        except:
            pass
        # smooth_and_let_inplace(qlayer, args)

        # real smooth and quantization      
        if args.quant_method == 'singlequant':
            set_init_singlequant_params_state(qlayer, False)
            set_quant_state(qlayer, weight_quant=True, act_quant=True)
            if singlequant_parameters.get(i):
                qlayer.load_singlequant_params(singlequant_parameters[i], dev)
            else:
                with torch.no_grad():
                    with torch.cuda.amp.autocast():
                        set_registered_x_none(qlayer)
                        if "qwen" in args.net.lower():
                            position_embeddings = lm.model.model.rotary_emb(rotate_inps, position_ids)

                            rotate_inps = qlayer(rotate_inps.unsqueeze(0), 
                                                 attention_mask=attention_mask,
                                                 position_ids=position_ids, 
                                                 position_embeddings=position_embeddings)[0][0]
                        else:
                            rotate_inps = qlayer(rotate_inps.unsqueeze(0), attention_mask=attention_mask,position_ids=position_ids)[0][0]
            
            qlayer.register_singlequant_params()
            set_init_singlequant_params_state(qlayer, True)


        qlayer.half()
        quant_soft_inplace(qlayer)
        set_quant_state(qlayer, weight_quant=False, act_quant=True)

        register_scales_and_zeros(qlayer)
        layers[i] = qlayer.to("cpu")
        singlequant_parameters[i] = singlequant_state_dict(qlayer)
        # if args.save_dir:
        #     torch.save(singlequant_parameters, os.path.join(args.save_dir, f"singlequant_parameters.pth"))

        del layer
        torch.cuda.empty_cache()

    del inps

    torch.cuda.empty_cache()
    gc.collect()                    
    model.config.use_cache = use_cache
    
    return model

def singlequant_quantize(model, tokenizer, args, device, logger=None):
    if not hasattr(args, 'cache_dir'):
        args.cache_dir = './cache'
    if not hasattr(args, 'quant_method'):
        args.quant_method = 'singlequant'
    if not hasattr(args, 'calib_dataset'):
        args.calib_dataset = 'wikitext2'
    if not hasattr(args, 'nsamples'):
        args.nsamples = 128
    if not hasattr(args, 'batch_size'):
        args.batch_size = 1
    if not hasattr(args, 'alpha'):
        args.alpha=0.6
    if not hasattr(args, 'group_size'):
        args.group_size=64
    if not hasattr(args, 'act_group_size'):
        args.act_group_size=64
    if not hasattr(args, 'w_bits'):
        args.wbits=4
    if not hasattr(args, 'abits'):
        args.abits=8
    if not hasattr(args, 'lac'):
        args.lac=0.9
    if not hasattr(args, 'swc'):
        args.swc=0.9
    if not hasattr(args, 'mse'):
        args.mse=2
    if not hasattr(args, 'weight_dominated'):
        args.weight_dominated = False
    if not hasattr(args, 'symmetric'):
        args.symmetric = True
    if not hasattr(args, 'a_dynamic_method'):
        args.a_dynamic_method = 'per_token'
    if not hasattr(args, 'w_dynamic_method'):
        args.w_dynamic_method = 'per_token'
    if not hasattr(args, 'deactive_amp'):
        args.deactive_amp = False # deactivate AMP when 8<=bits<16
    if not hasattr(args, 'seed'):
        args.seed=42
    if not hasattr(args, 'seqlen'):
        args.seqlen = 2048
    if not hasattr(args, 'fake_quant'):
        args.fake_quant = False
    if not hasattr(args, 'multigpu'):
        args.multigpu = False
    if not hasattr(args, 'save_dir'):
        args.save_dir = 'work_dirs'
    if not hasattr(args, 'net'):
        args.net = args.model.split('/')[-1]
    args.model_family = '-'.join(args.net.split('-')[:2])

    from .LMClass import LMClass
    from .parallel_utils import map_layers_to_multi_gpus, get_lowest_occupied_gpu
    from .datautils import get_loaders
    import random
    import numpy as np
    import datetime

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    
    lm = LMClass(tokenizer, model, args)
    lm.model.eval()
    for param in lm.model.parameters():
        param.requires_grad = False    

    args.weight_quant_params = {
        "n_bits": args.wbits,
        "per_channel_axes": [0],
        "symmetric": args.symmetric,
        "dynamic_method": args.w_dynamic_method,
        "group_size": args.group_size,
        "swc": args.swc,
        "quant_method": args.quant_method,
        "mse" : args.mse,
    }
    args.act_quant_params = {
        "n_bits": args.abits,
        "per_channel_axes": [],
        "symmetric": False,
        "lac": args.lac,
        "act_group_size": args.act_group_size,
        "dynamic_method": args.a_dynamic_method,
        "quant_method": args.quant_method,
    }
    args.q_quant_params = {
        "n_bits": args.abits,
        "per_channel_axes": [],
        "symmetric": False,
        "dynamic_method": args.a_dynamic_method,
        "quant_method": args.quant_method,
    }
    args.k_quant_params = {
        "n_bits": args.abits,
        "per_channel_axes": [],
        "symmetric": False,
        "dynamic_method": args.a_dynamic_method,
        "quant_method": args.quant_method,
    }
    args.v_quant_params = {
        "n_bits": args.abits,
        "per_channel_axes": [],
        "symmetric": False,
        "dynamic_method": args.a_dynamic_method,
    }
    args.p_quant_params = {
        "n_bits": 16,
        "metric": "fix0to1",
    }
    if args.multigpu:
        gpu_id = get_lowest_occupied_gpu(wait_memory=5000)
        lm._device = f"cuda:{gpu_id}"
        logger.info(f"set quantization in gpu {gpu_id}")
    if args.wbits < 16 or args.abits < 16:
        start = datetime.datetime.now()
        if not os.path.exists(args.cache_dir):
            os.makedirs(args.cache_dir)
        cache_dataloader = f'{args.cache_dir}/dataloader_{args.model_family}_{args.calib_dataset}_{args.nsamples}.cache'
        if os.path.exists(cache_dataloader):
            dataloader = torch.load(cache_dataloader)
            logger.info(f"load calibration from {cache_dataloader}")
        else:
            dataloader, _ = get_loaders(
                args.calib_dataset,
                nsamples=args.nsamples,
                seed=args.seed,
                model=args.model,
                seqlen=lm.seqlen,
            )
            torch.save(dataloader, cache_dataloader)
        
        singlequant(lm, args, dataloader, logger=logger)

        elapsed = datetime.datetime.now() - start
        logger.info(f'Runtime Spend: {elapsed}')

    return lm.model

class Rotate_Linear(nn.Module):
    def __init__(self, org_module=None, singlequant_parameters=None):
        super().__init__()
        if org_module is not None:
            R = org_module.act_quantizer.R
            R1_l = org_module.act_quantizer.R1_l
            R1_r = org_module.act_quantizer.R1_r
            R_ = org_module.act_quantizer.R_
            L_weight = R_@R1_l
            R_weight = (R1_r@R).T
            weight = org_module.weight
            bias = org_module.bias
        else:
            L_weight = singlequant_parameters["L_weight"]
            R_weight = singlequant_parameters["R_weight"]
            weight = singlequant_parameters["weight"]
            bias = singlequant_parameters["bias"]

        self.rotate_l= nn.Linear(L_weight.shape[0], L_weight.shape[1], bias=False)
        self.rotate_l.weight.data = L_weight
        self.rotate_l.keep16bit = True
        self.rotate_r= nn.Linear(R_weight.shape[0], R_weight.shape[1], bias=False)
        self.rotate_r.weight.data = R_weight
        self.rotate_r.keep16bit = True
        self.linear= nn.Linear(weight.shape[1], weight.shape[0], bias=bias is not None)
        self.linear.weight.data = weight
        if bias is not None:
            self.linear.bias.data = bias

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        bs, seq_len, hidden_size = x.shape
        x = x.reshape(-1, self.rotate_l.in_features, self.rotate_r.in_features)
        x = self.rotate_l.weight.data @ x
        # x = self.rotate_l(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.rotate_r(x)
        x = x.reshape(bs, seq_len, hidden_size)
        x = self.linear(x)
        return x
    
    def get_singlequant_parameters(self):
        return {
            "L_weight": self.rotate_l.weight.data,
            "R_weight": self.rotate_r.weight.data,
            "weight": self.linear.weight.data,
            "bias": None if self.linear.bias is None else self.linear.bias.data,
        }
        

def singlequant_for_model(model, tokenizer, args, device, gptq=False, logger=None):        
    logger.info("Start singlequant quantization")
    model = singlequant_quantize(model.half(), tokenizer, args, device, logger=logger)
    logger.info("End singlequant quantization")

    # 替换模型中的 QuantLinear 模块为 Rotate_Linear 模块
    for name, module in model.model.named_modules():
        if isinstance(module, QuantLinear):
            # 获取父模块和子模块名称
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            
            if parent_name:
                # 如果有父模块，获取父模块
                parent_module = model.model.get_submodule(parent_name)
                # 替换子模块
                setattr(parent_module, child_name, Rotate_Linear(org_module=module))
            else:
                # 如果没有父模块，直接在 model.model 上替换
                setattr(model.model, child_name, Rotate_Linear(org_module=module))

    if gptq:
        from xhquant_llm.singlequant.gptq_utils import gptq_fwrd
        from xhquant_llm.singlequant.datautils import get_loaders
        args.group_size = args.group_size
        args.act_order = False
        trainloader,_ = get_loaders(args.calib_dataset, 
                    seed=args.seed, 
                    model=args.model, 
                    seqlen=args.seqlen)
        model.seqlen = args.seqlen
        gptq_fwrd(model,tokenizer, trainloader, "cuda", args)
        model.cuda().half()
    
    from xhquant_llm.evaluation.wikitext_eval import evaluate_wikitext
    ppl = evaluate_wikitext(model, tokenizer, split="test", seqlen=2048)
    logger.info(f"PPL: {ppl}")

    #保存模型参数
    path = os.path.join(args.save_dir, 
                        f"{os.path.basename(args.model)}/{args.quant_method}_gptq{gptq}.pth")
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    singlequant_parameters = {}
    for name, module in model.model.named_modules():
        if isinstance(module, Rotate_Linear):
            singlequant_parameters[name] = module.get_singlequant_parameters()
    torch.save(singlequant_parameters, path)
        
    return model


