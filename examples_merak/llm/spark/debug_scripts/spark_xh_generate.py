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
from xhquant.api import Config, get_xhquant_logger, set_random_seed, xhquant_init
from xhquant.utils import ContextManagers, MemoryTracker, TimeProfiler


if TYPE_CHECKING:
    from xhmodel_merak.xh_llm.models.spark_xh import XHSparkModeModel, XHSparkModeModelConfig


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

    cfg = Config.fromfile(args.config)
    cfg.seed = seed
    logger.info(f"Config:\n{cfg.pretty_text}")
    config_file = work_dir / Path(args.config).name
    cfg.dump(config_file)

    dtype = torch.float16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}, dtype: {dtype}")
    model_cfg: XHSparkModeModelConfig = AutoLLMConfig.from_pretrained(cfg.model)
    enable_prefill_chunk = args.enable_prefill_chunk
    if enable_prefill_chunk:
        model_cfg.enable_prefill_chunk = True
        model_cfg.use_cache = True
    model_cfg.enable_auto_offload = args.auto_offload

    assert type(model_cfg).__name__ == "XHSparkModeModelConfig", (
        f"Expected model config type XHSparkModeModelConfig, but got {type(model_cfg).__name__}"
    )
    logger.info(f"Model Config:\n{model_cfg.to_json_string()}")
    xh_model: XHSparkModeModel = AutoLLMModel.from_pretrained(config=model_cfg)
    assert type(xh_model).__name__ == "XHSparkModeModel", (
        f"Expected model type XHSparkModeModel, but got {type(xh_model).__name__}"
    )
    eval_type = LLMModelState.from_string(eval_type)
    xh_model.set_state(eval_type)

    enable_think = args.think
    prompt = args.prompt
    if Path(prompt).is_file():
        with open(prompt, "r") as f:
            prompt = f.read()
    messages = [
        {"role": "user", "content": prompt},
    ]

    processor = xh_model.get_tf_processor()
    tokenizer = processor.tokenizer
    model_inputs = processor.apply_chat_template(messages, enable_think=enable_think)
    model_inputs = model_inputs.to(device)
    streamer = TextStreamer(tokenizer=tokenizer)
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
            **model_inputs, max_new_tokens=1024, streamer=streamer, do_sample=True, pad_token_id=tokenizer.eos_token_id
        )

    output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()

    # parsing thinking content
    try:
        think_end_token_id = tokenizer.encode("</think>")[-1]
        index = len(output_ids) - output_ids[::-1].index(think_end_token_id)
    except (ValueError, IndexError):
        index = 0

    thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
    logger.info(f"{'-' * 20} eval type: {eval_type} {'-' * 20}")
    logger.info(f"think: {enable_think}")
    if len(thinking_content) > 0:
        logger.info(f"{'-' * 20} thinking content {'-' * 20}")
        logger.info(f"{thinking_content}")
    logger.info(f"{'-' * 20} content {'-' * 20}")
    logger.info(f"{content}")


if __name__ == "__main__":
    eval_types = LLMModelState.get_all_values()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs_merak/xh2a/llm_models/spark/30b/spark_xh_30b_xh2a_2k.py",
    )
    parser.add_argument("--model", type=str)
    parser.add_argument("--eval-type", type=str, default="wrap", choices=eval_types)
    parser.add_argument("--prompt", type=str, default="你多大了？用中文回答。")
    parser.add_argument("--think", action="store_true", help="enable think mode")
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
