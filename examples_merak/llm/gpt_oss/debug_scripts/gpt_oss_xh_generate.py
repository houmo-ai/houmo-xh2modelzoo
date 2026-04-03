import argparse
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from transformers import TextStreamer

from xhmodel_merak.xh_llm import AutoLLMConfig, AutoLLMModel, LLMInferenceContextManager, LLMModelState
from xhquant.api import Config, get_xhquant_logger, set_random_seed, xhquant_init
from xhquant.utils import ContextManagers, MemoryTracker, TimeProfiler


if TYPE_CHECKING:
    from xhmodel_merak.xh_llm.models.gpt_oss import XHGptOssModel, XHGptOssModelConfig


def main(args):
    cfg_name = Path(args.config).stem
    eval_type = args.eval_type
    if args.debug:
        cfg_name += "_debug"

    args.work_dir = str(Path("./work_dirs") / cfg_name)
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    log_file = str(work_dir / f"generate_{eval_type}.log")

    xhquant_init(log_file, args.debug)
    seed = 1024
    set_random_seed(seed)
    logger = get_xhquant_logger()
    cfg = Config.fromfile(args.config)
    cfg.seed = seed
    logger.info(f"Config:\n{cfg.pretty_text}")
    dumped_config_file = work_dir / Path(args.config).name
    cfg.dump(dumped_config_file)

    dtype = torch.float16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}, dtype: {dtype}")
    model_cfg: XHGptOssModelConfig = AutoLLMConfig.from_pretrained(cfg.model)
    if args.enable_prefill_chunk:
        model_cfg.enable_prefill_chunk = True
        model_cfg.use_cache = True
    model_cfg.enable_auto_offload = args.auto_offload

    assert type(model_cfg).__name__ == "XHGptOssModelConfig", (
        f"Expected model config type XHGptOssModelConfig, but got {type(model_cfg).__name__}"
    )
    logger.info(f"Model Config:\n{model_cfg.to_json_string()}")
    xh_model: XHGptOssModel = AutoLLMModel.from_pretrained(config=model_cfg)
    assert type(xh_model).__name__ == "XHGptOssModel", (
        f"Expected model type XHGptOssModel, but got {type(xh_model).__name__}"
    )
    xh_model.set_state(LLMModelState.from_string(eval_type))

    prompt = args.prompt
    if Path(prompt).is_file():
        with open(prompt, "r") as f:
            prompt = f.read()
    messages = [{"role": "user", "content": prompt}]

    tokenizer = xh_model.get_tokenizer()
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt", truncation=True).to(device)
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
            **model_inputs,
            max_new_tokens=args.max_new_tokens,
            streamer=streamer,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()
    content = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
    logger.info(f"{'-' * 20} eval type: {eval_type} {'-' * 20}")
    logger.info(f"{'-' * 20} content {'-' * 20}")
    logger.info(content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs_merak/xh2a/llm_models/gpt_oss/20b/gpt_oss_20b_xh2a_2k.py",
    )
    parser.add_argument("--eval-type", type=str, default="quanted_fast", choices=LLMModelState.get_all_values())
    parser.add_argument("--prompt", type=str, default="Briefly introduce yourself.")
    parser.add_argument("--debug", action="store_true", help="Whether to run in debug mode")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--auto-offload",
        action="store_true",
        help="Whether to enable auto offload, only for debug and development",
    )
    parser.add_argument(
        "--enable-prefill-chunk",
        action="store_true",
        help="Whether to enable prefill chunk, only for debug and development",
    )
    main(parser.parse_args())
