import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from transformers import TextStreamer

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xhmodel_merak.xh_llm import AutoLLMHONNXModel, LLMInferenceContextManager
from xhquant.api import get_xhquant_logger, xhquant_init
from xhquant.utils import ContextManagers, MemoryTracker, TimeProfiler


if TYPE_CHECKING:
    from xhmodel_merak.xh_llm.models.unlimited_ocr.unlimited_ocr_hmonnx_inference import (
        XHUnlimitedOCRHMONNXModel,
    )


def main(args):
    xhquant_init(None, args.debug)
    logger = get_xhquant_logger()
    hmonnx_model: "XHUnlimitedOCRHMONNXModel" = AutoLLMHONNXModel.from_pretrained(args.config)
    assert type(hmonnx_model).__name__ == "XHUnlimitedOCRHMONNXModel", (
        f"Expected model type XHUnlimitedOCRHMONNXModel, but got {type(hmonnx_model).__name__}"
    )
    if args.auto_offload:
        hmonnx_model.enable_auto_offload = True

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = hmonnx_model.get_tf_processor()
    tokenizer = processor.tokenizer

    model_inputs = processor.process(args.prompt, args.image_path, device=device)
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
            input_ids=model_inputs["input_ids"],
            images_ori=model_inputs["images_ori"],
            images_crop=model_inputs.get("images_crop"),
            images_seq_mask=model_inputs["images_seq_mask"],
            images_spatial_crop=model_inputs["images_spatial_crop"],
            max_new_tokens=max_new_tokens,
            streamer=streamer,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(model_inputs["input_ids"], generated_ids, strict=False)
    ]
    output_text = tokenizer.batch_decode(
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
    parser.add_argument("--image-path", type=str, default="./data/images/unlimited_ocr_demo.jpg")
    parser.add_argument("--prompt", type=str, default="<image>\\nFree OCR. ")
    parser.add_argument("--golden", action="store_true", help="Whether to save golden outputs for testing.")
    parser.add_argument("--max-new-tokens", type=int, default=1024, help="The maximum number of tokens to generate.")
    parser.add_argument(
        "--auto-offload", action="store_true", help="Whether to enable auto offload, only for debug and development"
    )
    args = parser.parse_args()
    main(args)
