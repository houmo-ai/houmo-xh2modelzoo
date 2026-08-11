import argparse
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from transformers import TextStreamer

from xhmodel_merak.xh_llm import AutoLLMHONNXModel, LLMInferenceContextManager
from xhquant.api import get_xhquant_logger, xhquant_init
from xhquant.utils import ContextManagers, MemoryTracker, TimeProfiler


if TYPE_CHECKING:
    from xhmodel_merak.xh_llm.models.glm_4_moe_lite import XHGlm4MoeLiteHMONNXModel


def main(args):
    debug = args.debug
    xhquant_init(None, debug)
    logger = get_xhquant_logger()

    model_meta_file = args.config
    hmonnx_model: XHGlm4MoeLiteHMONNXModel = AutoLLMHONNXModel.from_pretrained(model_meta_file)
    assert type(hmonnx_model).__name__ == "XHGlm4MoeLiteHMONNXModel", (
        f"Expected model type XHGlm4MoeLiteHMONNXModel, but got {type(hmonnx_model).__name__}"
    )

    if args.auto_offload:
        hmonnx_model.enable_auto_offload = True

    device = "cuda" if torch.cuda.is_available() else "cpu"
    prompt = args.prompt
    if Path(prompt).is_file():
        with open(prompt, "r") as f:
            prompt = f.read()

    messages = [{"role": "user", "content": prompt}]
    tokenizer = hmonnx_model.get_tokenizer()
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=not args.disable_thinking,
    )

    model_inputs = tokenizer([text], return_tensors="pt", truncation=True).to(device)
    streamer = TextStreamer(tokenizer)
    hmonnx_model.to(device)
    if args.fast:
        hmonnx_model.to_fast()

    max_new_tokens = args.max_new_tokens
    if args.golden:
        max_new_tokens = 2
        hmonnx_model.enable_golden = True

    contexts = [
        TimeProfiler("hmonnx_generate", logger),
        MemoryTracker(device=device, name="generate", logger=logger),
        LLMInferenceContextManager(hmonnx_model),
    ]
    with ContextManagers(contexts):
        generated_ids = hmonnx_model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            streamer=streamer,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()
    content = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
    logger.info(f"{'-' * 20} content {'-' * 20}")
    logger.info(f"{content}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="work_dirs/glm_4_7_flash_xh2a_2k/golden_meta_info.json",
    )
    parser.add_argument("--fast", action="store_true", help="run in fast mode")
    parser.add_argument("--debug", action="store_true", help="Whether to run in debug mode")
    parser.add_argument("--prompt", type=str, default="你多大了？用中文回答。")
    parser.add_argument("--golden", action="store_true", help="Whether to save golden outputs for testing.")
    parser.add_argument("--disable-thinking", action="store_true", help="Disable reasoning output in the chat template.")
    parser.add_argument("--max-new-tokens", type=int, default=64, help="The maximum number of tokens to generate.")
    parser.add_argument(
        "--auto-offload", action="store_true", help="Whether to enable auto offload, only for debug and development"
    )
    args = parser.parse_args()
    main(args)
