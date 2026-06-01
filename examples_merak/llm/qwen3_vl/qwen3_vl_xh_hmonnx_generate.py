import argparse
from typing import TYPE_CHECKING

import torch
from transformers import TextStreamer

from xhmodel_merak.xh_llm import (
    AutoLLMHONNXModel,
    LLMInferenceContextManager,
)
from xhquant.api import get_xhquant_logger, xhquant_init
from xhquant.utils import ContextManagers, MemoryTracker, TimeProfiler


if TYPE_CHECKING:
    from xhmodel_merak.xh_llm.models.qwen3_vl.qwen3_vl_hmonnx_inference import XHQwen3VLHMONNXModel


def main(args):
    debug = args.debug
    xhquant_init(None, debug)
    logger = get_xhquant_logger()
    model_meta_file = args.config
    hmonnx_model: XHQwen3VLHMONNXModel = AutoLLMHONNXModel.from_pretrained(model_meta_file)
    assert type(hmonnx_model).__name__ == "XHQwen3VLHMONNXModel", (
        f"Expected model type XHQwen3VLHMONNXModel, but got {type(hmonnx_model).__name__}"
    )
    if args.auto_offload:
        hmonnx_model.enable_auto_offload = True
    processor = hmonnx_model.get_tf_processor()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": args.image_path,  # 图片路径，模型会自动加载图片
                },
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]
    model_inputs = processor.apply_chat_template(messages)
    tokenizer = processor.tokenizer
    # text = tokenizer.apply_chat_template(
    #     messages,
    #     tokenize=False,
    #     add_generation_prompt=True,
    #     enable_thinking=enable_think,  # Switches between thinking and non-thinking modes. Default is True.
    # )

    # model_inputs = tokenizer([text], return_tensors="pt", truncation=True)
    model_inputs = model_inputs.to(device)
    streamer = TextStreamer(tokenizer)
    hmonnx_model.to(device)

    max_new_tokens = args.max_new_tokens
    is_golden = args.golden
    if is_golden:
        max_new_tokens = 2
        hmonnx_model.enable_golden = True

    is_fast = args.fast
    if is_golden:
        is_fast = False  # Golden outputs should be generated in aligned precision for stability
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
            do_sample=True,
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
    parser.add_argument(
        "--config",
        type=str,
        default="work_dirs/qwen3_vl_llm_2b_xh2a_2k/hmquant_xh2_qwen3_vl_2b_w8a8_256_2k_448x448_20260312/golden_meta_info.json",
    )

    parser.add_argument("--fast", action="store_true", help="run in fast mode")
    parser.add_argument("--debug", action="store_true", help="Whether to run in debug mode")
    parser.add_argument("--image-path", type=str, default="./data/images/demo_qwen3_vl.jpeg")
    parser.add_argument("--prompt", type=str, default="Describe this image.")
    parser.add_argument("--golden", action="store_true", help="Whether to save golden outputs for testing.")
    parser.add_argument("--max-new-tokens", type=int, default=1024, help="The maximum number of tokens to generate.")
    parser.add_argument(
        "--auto-offload", action="store_true", help="Whether to enable auto offload, only for debug and development"
    )
    args = parser.parse_args()
    main(args)
