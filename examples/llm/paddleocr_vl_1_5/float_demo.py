# float test for paddleocr-vl
# Use flash-attn to boost performance and reduce memory usage
import argparse
import os
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor


def get_args():
    # ---- Settings ----
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", type=str, default="/data01/datasets/PaddleOCR-VL-1.5"
    )
    script_dir = Path(__file__).parent.parent.parent  # 返回到 xh2modelzoo 目录
    default_image_path = str(script_dir / "data" / "images" / "test.png")
    parser.add_argument("--image_path", type=str, default=default_image_path)
    parser.add_argument(
        "--task", type=str, default="ocr"
    )  # Options: 'ocr' | 'table' | 'chart' | 'formula' | 'spotting' | 'seal'
    return parser.parse_args()


def main():
    args = get_args()
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    PROMPTS = {
        "ocr": "OCR:",
        "table": "Table Recognition:",
        "formula": "Formula Recognition:",
        "chart": "Chart Recognition:",
        "spotting": "Spotting:",
        "seal": "Seal Recognition:",
    }

    image = Image.open(args.image_path).convert("RGB")
    orig_w, orig_h = image.size
    spotting_upscale_threshold = 1500
    if (
        args.task == "spotting"
        and orig_w < spotting_upscale_threshold
        and orig_h < spotting_upscale_threshold
    ):
        process_w, process_h = orig_w * 2, orig_h * 2
        try:
            resample_filter = Image.Resampling.LANCZOS
        except AttributeError:
            resample_filter = Image.LANCZOS
        image = image.resize((process_w, process_h), resample_filter)
    max_pixels = 2048 * 28 * 28 if args.task == "spotting" else 1280 * 28 * 28
    #     model_path, trust_remote_code=True, torch_dtype=torch.bfloat16
    # ).to(DEVICE).eval()

    # use flash-attn to boost performance and reduce memory usage
    model = (
        AutoModelForImageTextToText.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16
        )
        .to(DEVICE)
        .eval()
    )
    processor = AutoProcessor.from_pretrained(args.model_path)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": PROMPTS[args.task]},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        images_kwargs={
            "size": {
                "shortest_edge": processor.image_processor.min_pixels,
                "longest_edge": max_pixels,
            }
        },
    ).to(model.device)

    outputs = model.generate(**inputs, max_new_tokens=512)
    result = processor.decode(outputs[0][inputs["input_ids"].shape[-1] : -1])
    print(result)


if __name__ == "__main__":
    main()
