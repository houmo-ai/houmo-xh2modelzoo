import argparse
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from transformers import TextStreamer

from xhmodel_merak.xh_llm import AutoLLMHONNXModel, LLMInferenceContextManager
from xhquant.api import get_xhquant_logger, xhquant_init
from xhquant.utils import ContextManagers, MemoryTracker, TimeProfiler


if TYPE_CHECKING:
    from xhmodel_merak.xh_llm.models.qwen2_vl.qwen2_vl_hmonnx_inference import XHQwen2VLHMONNXModel


def main(args):
    xhquant_init(None, args.debug)
    logger = get_xhquant_logger()
    hmonnx_model: XHQwen2VLHMONNXModel = AutoLLMHONNXModel.from_pretrained(args.config)
    assert type(hmonnx_model).__name__ == "XHQwen2VLHMONNXModel", (
        f"Expected model type XHQwen2VLHMONNXModel, but got {type(hmonnx_model).__name__}"
    )
    if args.auto_offload:
        hmonnx_model.enable_auto_offload = True
    processor = hmonnx_model.get_tf_processor()
    tokenizer = processor.tokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": args.image_path,
                },
                {"type": "text", "text": args.prompt},
            ],
        }
    ]
    model_inputs = processor.apply_chat_template(messages).to(device)
    streamer = TextStreamer(tokenizer)
    hmonnx_model.to(device)

    max_new_tokens = args.max_new_tokens
    is_golden = args.golden
    if is_golden:
        max_new_tokens = 2
        hmonnx_model.enable_golden = True

    is_fast = args.fast
    if is_golden:
        is_fast = False
        logger.warning("Golden outputs should be generated in aligned precision for stability.")
    if is_fast:
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
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids, strict=False)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    precision_mode = "fast" if is_fast else "aligned"
    logger.info(f"{'-' * 20} Output for {precision_mode}, is_golden: {is_golden} {'-' * 20}")
    logger.info(f"{output_text}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--fast", action="store_true", help="run in fast mode")
    parser.add_argument("--debug", action="store_true", help="Whether to run in debug mode")
    parser.add_argument("--image-path", type=str, default="./data/images/qwen2_vl_demo.jpeg")
    parser.add_argument("--prompt", type=str, default="Describe this image.")
    parser.add_argument("--golden", action="store_true", help="Whether to save golden outputs for testing.")
    parser.add_argument("--max-new-tokens", type=int, default=1024, help="The maximum number of tokens to generate.")
    parser.add_argument(
        "--auto-offload", action="store_true", help="Whether to enable auto offload, only for debug and development"
    )
    args = parser.parse_args()
    main(args)
