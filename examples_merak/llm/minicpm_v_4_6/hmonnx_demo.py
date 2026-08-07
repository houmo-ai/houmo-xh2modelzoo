"""MiniCPM-V-4.6 HMONNX demo using only root export metadata."""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MiniCPM-V-4.6 HMONNX inference.")
    parser.add_argument("--export-meta", required=True, help="Root export_meta_info.json.")
    parser.add_argument("--image", action="append", default=[])
    parser.add_argument("--video", action="append", default=[])
    parser.add_argument("--prompt", default="请描述这张图片。")
    parser.add_argument("--downsample-mode", choices=("4x", "16x"), default="4x")
    parser.add_argument("--max-slice-nums", type=int, default=36)
    parser.add_argument("--video-max-slice-nums", type=int, default=1)
    parser.add_argument("--video-max-num-frames", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.image and not args.video:
        raise ValueError("Pass at least one --image or --video")
    from xhmodel_merak.xh_llm.models.minicpm_v_4_6.inference import (
        MiniCPMV46HMONNXRuntime,
    )

    runtime = MiniCPMV46HMONNXRuntime(args.export_meta, device=args.device)
    output = runtime.generate(
        {
            "images": args.image,
            "videos": args.video,
            "text": args.prompt,
        },
        downsample_mode=args.downsample_mode,
        max_slice_nums=args.max_slice_nums,
        video_max_slice_nums=args.video_max_slice_nums,
        video_max_num_frames=args.video_max_num_frames,
        max_new_tokens=args.max_new_tokens,
    )
    print(output)


if __name__ == "__main__":
    main()
