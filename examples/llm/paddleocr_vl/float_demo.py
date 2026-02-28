# float test for paddleocr-vl
# Use flash-attn to boost performance and reduce memory usage
import argparse
import os
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from transformers import AutoModelForCausalLM, AutoProcessor
from xhquant.api import prepare_quanted_model_to_compile, torch_compile_quanted_model

from xh_model_zoo.xh_llm.models.paddleocr_vl import PaddleOCRVLForConditionalGeneration, PaddleOCRVLProcessor


def get_args():
    # ---- Settings ----
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", type=str, default="/data01/datasets/PaddleOCR-VL"
    )
    script_dir = Path(__file__).parent.parent.parent  # 返回到 xhquant_llm 目录
    default_image_path = str(script_dir / "data" / "images" / "ocr_img.png")
    parser.add_argument("--image_path", type=str, default=default_image_path)
    parser.add_argument(
        "--task", type=str, default="ocr"
    )  # Options: 'ocr' | 'table' | 'chart' | 'formula'
    parser.add_argument(
        "--enable_quant_compile",
        action="store_true",
        help="prepare quanted model and compile with torch.compile",
    )
    parser.add_argument("--arch", type=str, default="XH2a")
    parser.add_argument("--precision_mode", type=str, default="fast")
    parser.add_argument("--use_flash_attention", action="store_true")
    parser.add_argument("--enable_op_check", action="store_true")
    parser.add_argument(
        "--skip_check_ops",
        type=str,
        default="",
        help="comma-separated op names to skip in check",
    )
    parser.add_argument("--enable_compiler_verbose", action="store_true")
    return parser.parse_args()


def main():
    args = get_args()
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    PROMPTS = {
        "ocr": "OCR:",
        "table": "Table Recognition:",
        "formula": "Formula Recognition:",
        "chart": "Chart Recognition:",
    }

    image = Image.open(args.image_path).convert("RGB")

    # model = AutoModelForCausalLM.from_pretrained(
    #     model_path, trust_remote_code=True, torch_dtype=torch.bfloat16
    # ).to(DEVICE).eval()

    # use flash-attn to boost performance and reduce memory usage
    model = (
        PaddleOCRVLForConditionalGeneration.from_pretrained(
            args.model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        .to(dtype=torch.bfloat16, device=DEVICE)
        .eval()
    )

    if args.enable_quant_compile:
        user_qconfig = dict(
            arch=args.arch,
            precision_mode=args.precision_mode,
            use_flash_attention=args.use_flash_attention,
            enable_op_check=args.enable_op_check,
            skip_check_ops=[op for op in args.skip_check_ops.split(",") if op],
            enable_compiler_verbose=args.enable_compiler_verbose,
        )
        model_name = os.path.basename(os.path.normpath(args.model_path))
        arch = user_qconfig.pop("arch")
        model, custom_backend = prepare_quanted_model_to_compile(
            model_name, model, arch, user_qconfig
        )
        model = torch_compile_quanted_model(model, custom_backend)
        model.eval()
    processor = PaddleOCRVLProcessor.from_pretrained(
        args.model_path, trust_remote_code=True
    )

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
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(DEVICE)

    with torch.inference_mode():
        out = model.generate(
            **inputs, max_new_tokens=1024, do_sample=False, use_cache=True
        )
    # outputs = model.generate(**inputs, max_new_tokens=1024)
    outputs = processor.batch_decode(out, skip_special_tokens=True)[0]
    print(outputs)


if __name__ == "__main__":
    main()
