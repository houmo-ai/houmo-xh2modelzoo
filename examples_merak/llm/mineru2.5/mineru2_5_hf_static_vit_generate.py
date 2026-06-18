#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run native HF MinerU2.5-Pro on one image with static ViT simulation."""

import argparse
import json
import sys
import time
from pathlib import Path

from PIL import Image

from mineru2_5_omnidocbench_hf_eval import (
    DEFAULT_VISUAL_BUCKETS_CONFIG,
    StaticBucketProcessorAdapter,
    build_sampling_params,
    build_static_bucket_processor,
    load_native_model,
    stringify_mineru_result,
)

Image.MAX_IMAGE_PIXELS = None


def parse_args():
    parser = argparse.ArgumentParser(
        description="MinerU2.5-Pro native HF single-image inference with static ViT simulation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", type=str, default="/data02/datasets/MinerU2.5-Pro-2604-1.2B")
    parser.add_argument("--image-path", type=str, default="data/images/0001.png")
    parser.add_argument("--output-json", type=str, default="", help="Optional path to save the raw MinerU result.")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--dtype", type=str, default="float16", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--device-map", type=str, default="auto")
    parser.add_argument("--use-fast-processor", action="store_true", default=True)
    parser.add_argument("--no-fast-processor", dest="use_fast_processor", action="store_false")
    parser.add_argument("--image-analysis", action="store_true")
    parser.add_argument("--mineru-batch-size", type=int, default=1)
    parser.add_argument("--layout-image-size", type=int, default=1036)
    parser.add_argument("--static-visual-buckets-config", type=str, default=DEFAULT_VISUAL_BUCKETS_CONFIG)
    parser.add_argument(
        "--static-vit-max-upscale",
        type=float,
        default=2.5,
        help="Maximum content upscale ratio before centering on a static visual bucket.",
    )
    parser.add_argument(
        "--static-vit-score-mode",
        type=str,
        default="fit_padding",
        choices=["fit_padding", "ratio"],
        help="Static visual bucket routing score. ratio is disabled because tests showed poor stability; use fit_padding.",
    )
    parser.add_argument("--static-vit-alpha-down", type=float, default=10.0)
    parser.add_argument("--static-vit-beta-up", type=float, default=1.0)
    parser.add_argument("--static-vit-gamma-pad", type=float, default=3.0)
    parser.add_argument("--static-vit-ref-area", type=float, default=448 * 448)
    parser.add_argument(
        "--allow-content-fallback-bucket",
        action="store_true",
        help="Allow non-layout content crops to route to the square fallback bucket.",
    )
    parser.add_argument("--no-tqdm", action="store_true", help="Disable MinerU progress bars.")
    parser.add_argument("--mineru-log-level", type=str, default="WARNING")
    return parser.parse_args()


def main():
    args = parse_args()

    import torch
    from loguru import logger
    from mineru_vl_utils import MinerUClient

    logger.remove()
    logger.add(sys.stderr, level=args.mineru_log_level)

    image_path = Path(args.image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    print(f"Loading MinerU2.5-Pro native HF model from {args.model}")
    model, processor = load_native_model(args)
    processor = build_static_bucket_processor(args, processor, model)
    args.mineru_batch_size = 1

    client = MinerUClient(
        backend="transformers",
        model=model,
        processor=processor,
        image_analysis=args.image_analysis,
        sampling_params=build_sampling_params(args.max_new_tokens),
        layout_image_size=(args.layout_image_size, args.layout_image_size),
        batch_size=args.mineru_batch_size,
        use_tqdm=not args.no_tqdm,
    )
    print("Model loaded.")

    image = Image.open(image_path).convert("RGB")
    start = time.perf_counter()
    with torch.no_grad():
        result = client.two_step_extract(image, image_analysis=args.image_analysis)
    elapsed = time.perf_counter() - start

    print(f"\nInference done in {elapsed:.2f}s")
    print("-" * 20 + " MinerU HF Static ViT Output " + "-" * 20)
    print(result)

    prediction = stringify_mineru_result(result)
    if prediction:
        print("-" * 20 + " Extracted Text " + "-" * 20)
        print(prediction)

    if isinstance(processor, StaticBucketProcessorAdapter):
        bucket_stats = {
            f"{bucket[0]}x{bucket[1]}": count
            for bucket, count in sorted(
                processor.bucket_counts.items(),
                key=lambda item: (item[0][0] * item[0][1], item[0]),
            )
        }
        print("-" * 20 + " Static ViT Bucket Stats " + "-" * 20)
        print(json.dumps({"bucket_counts": bucket_stats, "fallback_count": processor.fallback_count}, ensure_ascii=False))

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"Result saved to {output_path}")


if __name__ == "__main__":
    main()
