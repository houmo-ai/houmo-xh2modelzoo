import argparse
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from transformers import TextStreamer

from xhmodel_merak.xh_llm import (
    AutoLLMConfig,
    AutoLLMModel,
    LLMInferenceContextManager,
    LLMModelState,
)
from xhquant.utils import ContextManagers, MemoryTracker, TimeProfiler


if TYPE_CHECKING:
    from xhmodel_merak.xh_llm.models.glm_4_moe_lite import XHGlm4MoeLiteModel, XHGlm4MoeLiteModelConfig
from xhquant.api import Config, get_xhquant_logger, set_random_seed, xhquant_init


def main(args):
    cfg_name = Path(args.config).stem
    eval_type = args.eval_type
    debug = args.debug
    if debug:
        cfg_name += "_debug"

    args.work_dir = str(Path("./work_dirs") / cfg_name)
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    log_file = str(work_dir / f"generate_{eval_type}.log")

    xhquant_init(log_file, debug)
    seed = 1024
    set_random_seed(seed)
    logger = get_xhquant_logger()
    # 使用config文件,代替命令行参数,方便调试不同的配置
    cfg = Config.fromfile(args.config)
    cfg.seed = seed
    logger.info(f"Config:\n{cfg.pretty_text}")
    config_file = work_dir / Path(args.config).name
    cfg.dump(config_file)

    dtype = torch.float16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}, dtype: {dtype}")
    model_cfg: XHGlm4MoeLiteModelConfig = AutoLLMConfig.from_pretrained(cfg.model)
    enable_prefill_chunk = args.enable_prefill_chunk
    if enable_prefill_chunk:
        model_cfg.enable_prefill_chunk = True  # 开启prefill chunk后，设置chunk长度为512
        model_cfg.use_cache = True
    model_cfg.enable_auto_offload = args.auto_offload  # 是否启用自动显存卸载

    assert type(model_cfg).__name__ == "XHGlm4MoeLiteModelConfig", (
        f"Expected model config type XHGlm4MoeLiteModelConfig, but got {type(model_cfg).__name__}"
    )
    logger.info(f"Model Config:\n{model_cfg.to_json_string()}")
    xh_model: XHGlm4MoeLiteModel = AutoLLMModel.from_pretrained(config=model_cfg)
    assert type(xh_model).__name__ == "XHGlm4MoeLiteModel", (
        f"Expected model type XHGlm4MoeLiteModel, but got {type(xh_model).__name__}"
    )
    eval_type = LLMModelState.from_string(eval_type)
    xh_model.set_state(eval_type)

    # prompt = "你多大了？用中文回答。"
    prompt = args.prompt
    if Path(prompt).is_file():
        with open(prompt, "r") as f:
            prompt = f.read()
    messages = [
        {"role": "user", "content": prompt},
    ]

    # model_inputs = tokenizer([text], return_tensors="pt", truncation=True)
    processor = xh_model.get_tf_processor()
    tokenizer = processor.tokenizer
    model_inputs = processor.apply_chat_template(messages)
    model_inputs = model_inputs.to(device)
    streamer = TextStreamer(tokenizer=tokenizer)
    # logger.info(f"{'-' * 20} stream output {'-' * 20}")
    xh_model.to(device=device, dtype=dtype)
    xh_model.eval()
    contexts = [
        TimeProfiler("generate", logger),
        MemoryTracker(device=device, name="generate", logger=logger),
        LLMInferenceContextManager(xh_model),
        torch.no_grad(),
    ]
    with ContextManagers(contexts):
        generated_ids = xh_model.generate(
            **model_inputs,
            max_new_tokens=1024,
            streamer=streamer,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()
    content = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
    logger.info(f"{'-' * 20} eval type: {eval_type} {'-' * 20}")
    logger.info(f"{'-' * 20} content {'-' * 20}")
    logger.info(f"{content}")


if __name__ == "__main__":
    eval_types = LLMModelState.get_all_values()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs_merak/xh2a/llm_models/glm_4.7_flash/glm_4_7_flash_xh2a_2k.py",
    )
    parser.add_argument("--eval-type", type=str, default="wrap", choices=eval_types)
    parser.add_argument("--prompt", type=str, default="你多大了？用中文回答。")
    parser.add_argument("--debug", action="store_true", help="Whether to run in debug mode")
    parser.add_argument(
        "--auto-offload", action="store_true", help="Whether to enable auto offload, only for debug and development"
    )
    parser.add_argument(
        "--enable-prefill-chunk",
        action="store_true",
        help="Whether to enable prefill chunk, only for debug and development",
    )
    args = parser.parse_args()
    main(args)
