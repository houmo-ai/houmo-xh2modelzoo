"""Run a short generation with an exported MiniCPM5 HMONNX model."""

from __future__ import annotations

import argparse
from pathlib import Path

from transformers import TextStreamer

from xhmodel_merak.xh_llm import AutoLLMHONNXModel, LLMInferenceContextManager
from xhquant.api import get_xhquant_logger, xhquant_init
from xhquant.utils import ContextManagers, MemoryTracker, TimeProfiler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MiniCPM5 HMONNX inference.")
    parser.add_argument("--config", required=True, help="Path to golden_meta_info.json")
    parser.add_argument("--prompt", default="请用中文简单介绍 MiniCPM5。")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--golden", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    xhquant_init(None, args.debug)
    logger = get_xhquant_logger()
    hmonnx_model = AutoLLMHONNXModel.from_pretrained(args.config)
    assert type(hmonnx_model).__name__ == "XHMiniCPM5HMONNXModel", type(hmonnx_model).__name__

    prompt = args.prompt
    if Path(prompt).is_file():
        prompt = Path(prompt).read_text(encoding="utf-8")
    tokenizer = hmonnx_model.get_tokenizer()
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(args.device)
    hmonnx_model.to(args.device)
    if args.fast:
        hmonnx_model.to_fast()
    if args.golden:
        hmonnx_model.enable_golden = True
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    streamer = TextStreamer(tokenizer)
    with ContextManagers(
        [
            TimeProfiler("minicpm5_hmonnx_generate", logger),
            MemoryTracker(device=args.device, name="minicpm5_generate", logger=logger),
            LLMInferenceContextManager(hmonnx_model),
        ]
    ):
        generated_ids = hmonnx_model.generate(
            **model_inputs,
            max_new_tokens=2 if args.golden else args.max_new_tokens,
            streamer=streamer,
            do_sample=False,
            pad_token_id=pad_token_id,
        )
    output_ids = generated_ids[0, model_inputs.input_ids.shape[1] :]
    content = tokenizer.decode(
        output_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()
    logger.info(f"hmonnx MiniCPM5 output: {content}")


if __name__ == "__main__":
    main()
