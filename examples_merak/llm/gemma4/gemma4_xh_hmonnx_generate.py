import argparse
from typing import TYPE_CHECKING

import torch
from PIL import Image
from transformers import TextStreamer

from xhmodel_merak.xh_llm import AutoLLMHONNXModel, LLMInferenceContextManager
from xhquant.api import get_xhquant_logger, xhquant_init
from xhquant.utils import ContextManagers, MemoryTracker, TimeProfiler


if TYPE_CHECKING:
    from xhmodel_merak.xh_llm.models.gemma4 import XHGemma4HMONNXModel


def main(args):
    xhquant_init(None, args.debug)
    logger = get_xhquant_logger()
    hmonnx_model: XHGemma4HMONNXModel = AutoLLMHONNXModel.from_pretrained(args.config)
    assert type(hmonnx_model).__name__ == "XHGemma4HMONNXModel", (
        f"Expected XHGemma4HMONNXModel, got {type(hmonnx_model).__name__}"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.image_path:
        # Resize input image to match the export image size so that the number
        # of soft image tokens equals the visual model's num_image_tokens.
        vis_cfg = hmonnx_model.visual_meta
        export_w = getattr(vis_cfg, "image_size_w", 224)
        export_h = getattr(vis_cfg, "image_size_h", 224)
        image = Image.open(args.image_path).convert("RGB").resize((export_w, export_h))
        logger.info(f"Resized input image to {export_w}x{export_h}")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": args.prompt},
                ],
            },
        ]
    else:
        messages = [{"role": "user", "content": [{"type": "text", "text": args.prompt}]}]

    processor = hmonnx_model.get_tf_processor()
    tokenizer = processor.tokenizer
    model_inputs = processor.apply_chat_template(messages).to(device)

    streamer = TextStreamer(tokenizer)
    hmonnx_model.to(device)

    if args.fast:
        hmonnx_model.to_fast()
    if args.golden:
        hmonnx_model.enable_golden = True

    contexts = [
        TimeProfiler("hmonnx_generate", logger),
        MemoryTracker(device=device, name="generate", logger=logger),
        LLMInferenceContextManager(hmonnx_model),
    ]
    with ContextManagers(contexts):
        hmonnx_model.to(device)
        generated_ids = hmonnx_model.generate(
            **model_inputs,
            max_new_tokens=args.max_new_tokens,
            streamer=streamer,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()
    content = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
    logger.info(f"{'-' * 20} content {'-' * 20}")
    logger.info(content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gemma4 HMONNX Generate")
    parser.add_argument(
        "--config",
        type=str,
        default="work_dirs/gemma4_31b_it_xh2a_2k/hmquant_xh2_gemma4_31b_it_w8a8_256_2k_20260410/golden_meta_info.json",
    )
    parser.add_argument("--fast", action="store_true", help="run in fast mode")
    parser.add_argument("--debug", action="store_true", help="run in debug mode")
    parser.add_argument("--image-path", type=str, default=None, help="path to image (omit for text-only)")
    parser.add_argument("--prompt", type=str, default="描述下这张图片")
    parser.add_argument("--golden", action="store_true", help="save golden outputs")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    args = parser.parse_args()
    main(args)
