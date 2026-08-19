"""MiniCPM-V-4.5 HMONNX demo using only root export metadata."""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MiniCPM-V-4.5 HMONNX inference.")
    parser.add_argument("--export-meta", required=True, help="Root golden_meta_info.json of the HMONNX export.")
    parser.add_argument("--image", action="append", default=[])
    parser.add_argument("--video", action="append", default=[])
    parser.add_argument("--prompt", default="请描述这张图片。")
    parser.add_argument("--max-slice-nums", type=int, default=9)
    parser.add_argument("--video-fps", type=float, default=3)
    parser.add_argument("--video-packing", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.image and not args.video:
        raise ValueError("Pass at least one --image or --video")
    from xhmodel_merak.xh_llm.models.minicpm_v_4_5.inference import (
        MiniCPMV45HMONNXRuntime,
    )

    runtime = MiniCPMV45HMONNXRuntime(args.export_meta, device=args.device)
    output = runtime.generate(
        {
            "images": args.image,
            "videos": args.video,
            "text": args.prompt,
        },
        max_slice_nums=args.max_slice_nums,
        video_fps=args.video_fps,
        video_packing=args.video_packing,
        max_new_tokens=args.max_new_tokens,
    )
    print(output)


if __name__ == "__main__":
    main()
