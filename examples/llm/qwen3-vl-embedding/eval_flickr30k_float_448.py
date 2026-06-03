#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ablation: run the FLOAT (HF BF16) Qwen3-VL-Embedding model on Flickr30K but
force the image side through the SAME fixed 448x448 square-pad preprocessing
that the HMONNX vision tower uses.

Goal: isolate the two loss sources in the HMONNX image path:
  - resolution/preprocess loss  (dynamic-res float  ->  float@448)
  - quantization loss           (float@448          ->  HMONNX@448 / W8A8)

Text side keeps the normal float dynamic processing (text cosine sim is
already ~0.987, not the bottleneck), so the image resolution is the only
变量 vs the float baseline.

Reuses dataset loading + retrieval_metrics + embed_in_batches from the
existing float eval script (qwen3_vl_embedding_eval_flickr30k.py).
"""

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import List

import torch
from loguru import logger
from PIL import Image, ImageOps

# Import utilities from the existing float eval script (same directory)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
_base_path = os.path.join(HERE, "qwen3_vl_embedding_eval_flickr30k.py")
_spec = importlib.util.spec_from_file_location("flickr_float_base", _base_path)
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)

DEFAULT_LOCAL_2B = base.DEFAULT_LOCAL_2B


def to_square(img, size: int = 448, fill=(114, 114, 114)) -> Image.Image:
    """Replicate HMONNX load_and_process_image: keep aspect ratio, resize to
    fit inside size x size, then pad (right/bottom) with gray (114,114,114)
    to an exact square size x size."""
    if isinstance(img, str):
        img = Image.open(img)
    img = img.convert("RGB")
    w, h = img.size
    if (w, h) != (size, size):
        scale = min(size / w, size / h)
        nw, nh = int(w * scale), int(h * scale)
        img = img.resize((nw, nh), Image.BICUBIC)
        pad_w = size - nw
        pad_h = size - nh
        img = ImageOps.expand(img, border=(0, 0, pad_w, pad_h), fill=fill)
    return img


def main(args):
    model_path = args.model_dir
    assert Path(model_path).exists(), f"HF model not found: {model_path}"

    logger.info(f"Loading Flickr30K test split (dataset_dir={args.dataset_dir!r}) ...")
    rows = base.load_flickr30k_test(args.dataset_dir)
    logger.info(f"Loaded {len(rows)} images with captions")

    if args.max_images:
        rows = rows[: args.max_images]
        logger.warning(f"--max-images={args.max_images} active; eval is NOT full 1K standard")

    # Build flat caption list + ground-truth index maps (same as base.main)
    images = []
    captions_flat: List[str] = []
    cap_owner: List[int] = []
    image_to_caps: List[List[int]] = []
    for img_idx, (img, caps) in enumerate(rows):
        images.append(img)  # keep raw (path or PIL); we preprocess below
        own_caps = []
        for c in caps:
            cap_id = len(captions_flat)
            captions_flat.append(c)
            cap_owner.append(img_idx)
            own_caps.append(cap_id)
        image_to_caps.append(own_caps)
    logger.info(
        f"#images={len(images)}  #captions={len(captions_flat)}  "
        f"(avg {len(captions_flat) / max(len(images), 1):.2f} cap/img)"
    )

    size = args.image_size
    # Force the float embedder to NOT re-resize: min==max==size*size pins the
    # vision processor at exactly size x size (size must be a multiple of 32).
    assert size % 32 == 0, f"image_size must be a multiple of 32, got {size}"
    Qwen3VLEmbedder = base.load_qwen3_vl_embedder(model_path)
    embedder = Qwen3VLEmbedder(
        model_name_or_path=model_path,
        torch_dtype=torch.bfloat16,
        min_pixels=size * size,
        max_pixels=size * size,
    )
    embedder.model.eval()
    logger.info(
        f"Loaded Qwen3VLEmbedder on {embedder.model.device}; "
        f"image forced to {size}x{size} square-pad (min=max={size*size} px)"
    )

    # Preprocess every image to fixed square `size` (HMONNX-style), pass as PIL
    logger.info(f"Preprocessing {len(images)} images to {size}x{size} square-pad ...")
    square_images = [to_square(img, size=size) for img in images]

    text_items = [{"text": c} for c in captions_flat]
    image_items = [{"image": im} for im in square_images]

    # Text: normal float dynamic; Image: float vision @ fixed square size
    text_emb = base.embed_in_batches(embedder, text_items, args.text_batch_size, "text")
    image_emb = base.embed_in_batches(embedder, image_items, args.image_batch_size, f"image@{size}")

    cap_owner_t = torch.tensor(cap_owner, dtype=torch.long)
    text_gt = [[int(o)] for o in cap_owner_t.tolist()]

    logger.info("Computing text->image metrics ...")
    t2i = base.retrieval_metrics(text_emb, image_emb, text_gt)

    logger.info("Computing image->text metrics ...")
    i2t = base.retrieval_metrics(image_emb, text_emb, image_to_caps)

    report = {
        "model_dir": model_path,
        "dataset": "Flickr30K (test split)",
        "variant": f"FLOAT BF16, image forced to {size}x{size} square-pad (HMONNX-style preprocess)",
        "image_size": size,
        "num_images": len(images),
        "num_captions": len(captions_flat),
        "text_to_image": t2i,
        "image_to_text": i2t,
    }

    out_path = Path(args.report) if args.report else Path("work_dirs") / f"flickr30k_float_{size}_eval.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info(f"===== Flickr30K Retrieval (FLOAT @ {size}x{size} square-pad) =====")
    fmt = lambda d: "  ".join(f"{k}={v:.4f}" for k, v in d.items())
    logger.info(f"text -> image  {fmt(t2i)}")
    logger.info(f"image -> text  {fmt(i2t)}")
    logger.info(f"Wrote report to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, default=DEFAULT_LOCAL_2B,
                        help="Path to Qwen3-VL-Embedding HF snapshot")
    parser.add_argument("--dataset-dir", type=str, default=None,
                        help="Local Flickr30K root; falls back to nlphuji/flickr30k HF cache")
    parser.add_argument("--image-size", type=int, default=448,
                        help="Force image side to this fixed square size (multiple of 32)")
    parser.add_argument("--text-batch-size", type=int, default=16)
    parser.add_argument("--image-batch-size", type=int, default=4)
    parser.add_argument("--max-images", type=int, default=0, help="0 = full 1K test (standard)")
    parser.add_argument("--report", type=str, default=None)
    args = parser.parse_args()
    main(args)
