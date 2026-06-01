import argparse
from pathlib import Path

import torch
from transformers import TextStreamer

from xhmodel_merak.xh_llm import AutoLLMHONNXModel, LLMInferenceContextManager
from xhquant.api import get_xhquant_logger, xhquant_init
from xhquant.utils import ContextManagers, MemoryTracker, TimeProfiler


def main(args):
    xhquant_init(None, args.debug)
    logger = get_xhquant_logger()

    hmonnx_model = AutoLLMHONNXModel.from_pretrained(args.config)

    if args.auto_offload:
        hmonnx_model.enable_auto_offload = True

    device = "cuda" if torch.cuda.is_available() else "cpu"
    prompt = args.prompt
    if Path(prompt).is_file():
        prompt = Path(prompt).read_text()

    tokenizer = hmonnx_model.get_tokenizer()
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=args.think,
    )

    model_inputs = tokenizer([text], return_tensors="pt", truncation=True).to(device)
    streamer = TextStreamer(tokenizer)
    hmonnx_model.to(device)

    max_new_tokens = args.max_new_tokens
    if args.golden:
        max_new_tokens = 2
        hmonnx_model.enable_golden = True

    if args.fast and not args.golden:
        hmonnx_model.to_fast()

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
    try:
        think_end_token_id = tokenizer.encode("</think>")[-1]
        index = len(output_ids) - output_ids[::-1].index(think_end_token_id)
    except (ValueError, IndexError):
        index = 0

    thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

    logger.info(f"think: {args.think}")
    if thinking_content:
        logger.info(f"{'-' * 20} thinking content {'-' * 20}")
        logger.info(thinking_content)
    logger.info(f"{'-' * 20} content {'-' * 20}")
    logger.info(content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="golden_meta_info.json path")
    parser.add_argument("--fast", action="store_true", help="run in fast mode")
    parser.add_argument("--debug", action="store_true", help="run in debug mode")
    parser.add_argument("--prompt", type=str, default="你多大了？用中文回答。")
    parser.add_argument("--think", action="store_true", help="enable think mode")
    parser.add_argument("--golden", action="store_true", help="save golden outputs for testing")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="maximum number of tokens to generate")
    parser.add_argument("--auto-offload", action="store_true", help="enable auto offload for debug/development")
    args = parser.parse_args()
    main(args)
