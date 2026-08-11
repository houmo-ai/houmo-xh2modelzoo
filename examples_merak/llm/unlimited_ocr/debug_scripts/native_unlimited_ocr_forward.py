"""Native HF Unlimited-OCR base/no-crop single-image forward smoke.

Runs the localized ``UnlimitedOCRForCausalLM`` (no trust_remote_code) on a base
single image prompt and dumps prefill logits + argmax next token as the golden
reference for the wrap / HMONNX alignment scripts.

Usage:
    python examples_merak/llm/unlimited_ocr/debug_scripts/native_unlimited_ocr_forward.py \
        --model data/models/Unlimited-OCR \
        --image-path data/images/qwen2_vl_demo.jpeg \
        --prompt '<image>\nFree OCR. ' \
        --dump work_dirs/unlimited_ocr_debug/native_prefill.pt
"""

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xhmodel_merak.xh_llm.models.unlimited_ocr.modeling_unlimitedocr import UnlimitedOCRForCausalLM
from xhmodel_merak.xh_llm.models.unlimited_ocr.unlimited_ocr_processor import XHUnlimitedOCRProcessor


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    processor = XHUnlimitedOCRProcessor(
        tokenizer,
        image_token_id=args.image_token_id,
        image_size=args.image_size,
        base_size=args.image_size,
        patch_size=args.patch_size,
        downsample_ratio=args.downsample_ratio,
        crop_mode=False,
    )
    inputs = processor.process(args.prompt, args.image_path, device=device)
    input_ids = inputs["input_ids"]
    images_seq_mask = inputs["images_seq_mask"]
    images_ori = inputs["images_ori"].to(dtype)
    images_spatial_crop = inputs["images_spatial_crop"]

    # base/no-crop packs images as [(images_crop_zero, images_ori)]; the crop
    # tensor is all-zeros so the HF forward takes the global-view else branch.
    images_crop = torch.zeros((1, 3, args.image_size, args.image_size), device=device, dtype=dtype)
    images = [(images_crop, images_ori)]

    model = UnlimitedOCRForCausalLM.from_pretrained(
        args.model, dtype=dtype, trust_remote_code=False
    ).to(device).eval()

    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            images=images,
            images_seq_mask=images_seq_mask,
            images_spatial_crop=images_spatial_crop,
            use_cache=False,
            return_dict=True,
        )
    logits = out.logits
    next_token = int(logits[0, -1].argmax().item())
    print(f"input_ids shape: {tuple(input_ids.shape)}")
    print(f"image tokens: {int(images_seq_mask.sum().item())}")
    print(f"prefill logits shape: {tuple(logits.shape)}")
    print(f"prefill last-token argmax: {next_token} -> {tokenizer.decode([next_token])!r}")

    if args.dump:
        dump_path = Path(args.dump)
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "input_ids": input_ids.cpu(),
                "images_seq_mask": images_seq_mask.cpu(),
                "last_logits": logits[0, -1].float().cpu(),
                "next_token": next_token,
            },
            dump_path,
        )
        print(f"dumped golden reference to {dump_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="data/models/Unlimited-OCR")
    parser.add_argument("--image-path", type=str, default="data/images/qwen2_vl_demo.jpeg")
    parser.add_argument("--prompt", type=str, default="<image>\\nFree OCR. ")
    parser.add_argument("--image-token-id", type=int, default=128815)
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--downsample-ratio", type=int, default=4)
    parser.add_argument("--dump", type=str, default="")
    args = parser.parse_args()
    main(args)
