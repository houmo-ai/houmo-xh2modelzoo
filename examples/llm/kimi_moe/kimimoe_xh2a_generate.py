
from pathlib import Path
from typing import List, Tuple

import torch
import torch.fx
import torch.fx.config
import torch.nn as nn
import xhquant.utils.suppress_printing
from torch import Tensor
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm
from transformers import TextStreamer
from xhquant.api import FrontendType, Hook, PrecisionMode, ptq_quantize, set_random_seed
from xhquant.debug import GraphSnapshot

from xh_model_zoo.api import Config, EvalModelType, get_root_logger, xhquant_llm_init
from xh_model_zoo.datasets import WikiTextDataset
from xh_model_zoo.xh_llm.models.builder import MODELS
from xh_model_zoo.xh_llm.models.kimi_moe import KimiMoe_HFCompatible, kimi_moe_llm_model
from xh_model_zoo.utils import print_gpu_info
from xh_model_zoo.xh_llm.utils import auto_offload
from xh_model_zoo.utils.time_profiler import TimeProfiler
from modelscope import AutoModelForCausalLM, AutoTokenizer

def parse_arguments():
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--config",
        type=str,
        default="configs/kimi/3b_30b/kimi_a3b_30b_instruct_legacy_xh2a_2k_batch.py",
    )
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--offload", default=False, help="evaluate the model")
    parser.add_argument("--eval-type", type=str, default="pytorch", help="evaluate type")
    parser.add_argument("--prompt", type=str, default="你多大了？用中文回答。")
    return parser


def main(args):

    cfg = Config.fromfile(args.config)
    cfg_name = Path(args.config).stem
    cfg_name = f"{cfg_name}_eval"
    cfg.work_dir = str(Path("./work_dirs") / cfg_name)
    cfg.debug = False
    # version = f"-{args.eval_type}-{args.te_mode}-{args.act_bit}-{args.weight_bit}"
    log_file = Path(cfg.work_dir) / f"{cfg_name}.log"
    Path(cfg.work_dir).mkdir(exist_ok=True, parents=True)
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.dtype = "float16"
    debug_output_dir = Path(cfg.work_dir) / "debug"
    debug_output_dir.mkdir(exist_ok=True, parents=True)
    seed = cfg.get("seed", 1024)
    eval_input_seq_length = 2048
    set_random_seed(seed)

    torch.fx.config.disable_progress = True
    is_auto_offload = args.offload

    cfg.model.wrap_cfg.num_logits_to_keep = 1  # 输出所有logits，不仅仅是最后一个token的logits
    cfg.model.wrap_cfg.use_cache = True  # 禁用cache

    cfg.model.wrap_cfg.input_sequence_length = eval_input_seq_length
    xhquant_llm_init(log_file, cfg.debug)
    logger = get_root_logger()

    logger.info(f"Config:\n{cfg.pretty_text}")
    cfg.dump(Path(cfg.work_dir) / Path(args.config).name)

    xhquant.utils.suppress_printing.disable_printing = True  # 屏蔽不必要的打印信息

    device = cfg.device
    dtype = getattr(torch, cfg.dtype)

    frontend_type = cfg.get("frontend_type", "TorchFX")
    frontend_type = FrontendType(frontend_type)

    xh_model: kimi_moe_llm_model = MODELS.build(cfg.model)  # type: ignore
    native_model = xh_model.get_hf_model("cpu")  # CPU上加载原生模型

    if hasattr(cfg, "resume_from") and cfg.resume_from is not None:
        xh_model.load_wraped_model_state_dict(native_model, cfg.resume_from)

    # 构建模型，并做Module替换，将原生Module替换为可以fx.trace的Module
    xh_model.init_wrap_model(native_model)

    # prompt = "你多大了？用中文回答。"
    prompt = args.prompt
    messages = [{"role": "user", "content": prompt}]

    # messages = [
    #     {"role": "system", "content": "You are a helpful assistant."},
    #     {"role": "user", "content": prompt},
    # ]
    tokenizer = xh_model.get_tokenizer()
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    model_inputs = tokenizer([text], return_tensors="pt", truncation=True).to(device)
    input_ids = model_inputs.input_ids
    data_batch = {
        "input_ids": input_ids.to(device),
        "past_seq_length": [0],
    }
    is_auto_offload = args.offload
    if args.eval_type == "pytorch":
        eval_type = EvalModelType.WRAPED
    elif args.eval_type == "traced":
        eval_type = EvalModelType.FRONTEND
    elif args.eval_type == "quant-float":
        eval_type = EvalModelType.QUANTED_DISABLED
    elif args.eval_type in ["quant-fast", "fast"]:
        eval_type = EvalModelType.QUANTED_FAST
    elif args.eval_type in ["quant-aligned", "aligned", "align"]:
        eval_type = EvalModelType.QUANTED_ALIGNED
    else:
        raise ValueError(f"Unknown eval_type: {args.eval_type}")

    if eval_type in [
        EvalModelType.FRONTEND,
        EvalModelType.QUANTED_DISABLED,
        EvalModelType.QUANTED_ALIGNED,
        EvalModelType.QUANTED_FAST,
        EvalModelType.EXPORTED,
        EvalModelType.CALIBRATION,
    ]:
        xh_model.convert_to_fronted_graph(data_batch)

    if eval_type in [
        EvalModelType.QUANTED_DISABLED,
        EvalModelType.QUANTED_ALIGNED,
        EvalModelType.QUANTED_FAST,
        EvalModelType.EXPORTED,
        EvalModelType.CALIBRATION,
    ]:

        xh_model.convert_to_quant_graph(cfg.target_device)

    offload_flag = False
    if eval_type in [
        EvalModelType.QUANTED_FAST,
    ]:
        logger.info("*************** Start PTQ Quantize ***************")
        calib_data = xh_model.prepare_inputs(data_batch)
        ## 将输入的List展开
        new_args = []
        for arg in calib_data:
            if isinstance(arg, (List, Tuple)):
                new_args.extend(arg)
            else:
                new_args.append(arg)
        calib_data = new_args
        ptq_quantize(
            xh_model.quanted_model,
            [calib_data],
            PrecisionMode.FAST,
            [device],
            auto_release_unused_parameters=True,
        )
        logger.info("*************** Finished PTQ Quantize ***************")

    if eval_type in [
        EvalModelType.QUANTED_ALIGNED,
        # EvalModelType.QUANTED_FAST,
        EvalModelType.EXPORTED,
        EvalModelType.CALIBRATION,
    ]:
        ## 进行PTQ量化
        logger.info("*************** Start PTQ Quantize ***************")
        calib_data = xh_model.prepare_inputs(data_batch)
        ## 将输入的List展开
        new_args = []
        for arg in calib_data:
            if isinstance(arg, (List, Tuple)):
                new_args.extend(arg)
            else:
                new_args.append(arg)
        calib_data = new_args
        ptq_quantize(
            xh_model.quanted_model,
            [calib_data],
            PrecisionMode.ALIGNED,
            [device],
            auto_release_unused_parameters=True,
        )
        logger.info("*************** Finished PTQ Quantize ***************")

    xh_model.change_eval_type(eval_type)
    xh_model.to(dtype)

    # if eval_type == EvalModelType.WRAPED:
    #     GptOssQuantedDebugHook(xh_model.wrap_model, str(Path(cfg.work_dir) / "wraped"))

    # xh_model._quanted_model = torch.compile(xh_model._quanted_model)
    # hf_model = xh_model.get_empty_hf_model() 
    hf_model = AutoModelForCausalLM.from_pretrained(cfg.hf_model_dir, trust_remote_code=True)
    hf_model = KimiMoe_HFCompatible.to_hf_compatible(hf_model, xh_model)
    hf_model.dynamic_input = False
    if not is_auto_offload:
        hf_model.to(device)
    else:
        if not offload_flag:
            auto_offload(
                hf_model,
                "XHTrace_Qwen3MoeDecoderLayer",
            )


    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,  # Switches between thinking and non-thinking modes. Default is True.
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    # conduct text completion
    generated_ids = hf_model.generate(**model_inputs, max_new_tokens=2048, do_sample=True)
    # output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()

    # parsing thinking content
    # try:
    #     # rindex finding 151668 (</think>)
    #     index = len(output_ids) - output_ids[::-1].index(151668)
    # except ValueError:
    #     index = 0

    # thinking_content = tokenizer.decode(generated_ids, skip_special_tokens=True).strip("\n")
    content = tokenizer.batch_decode(generated_ids)[0]

    # logger.info(f"thinking content:{thinking_content}")
    logger.info(f"content:{content}")



if __name__ == "__main__":
    parser = parse_arguments()
    args = parser.parse_args()
    main(args)
