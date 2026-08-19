"""Native Transformers floating-point baseline for MiniCPM-V-4.5."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from packaging.version import Version
from transformers import AutoModel, AutoTokenizer
from transformers import __version__ as transformers_version


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run native MiniCPM-V-4.5 inference.")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--image", type=Path, action="append", default=[])
    parser.add_argument("--video", type=Path, action="append", default=[])
    parser.add_argument("--prompt", default="请描述这张图片。")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("auto", "bf16", "fp16", "fp32"), default="auto")
    parser.add_argument("--max-slice-nums", type=int, default=9)
    parser.add_argument("--video-fps", type=float, default=3)
    parser.add_argument("--video-packing", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--repetition-penalty", type=float, default=1.03)
    parser.add_argument("--seed", type=int, default=1024)
    return parser.parse_args()


def _dtype(name: str) -> str | torch.dtype:
    return {
        "auto": "auto",
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[name]


def _validate_args(args: argparse.Namespace) -> None:
    if Version(transformers_version) < Version("4.57.0"):
        raise RuntimeError(f"MiniCPM-V-4.5 requires transformers>=4.57.0; found transformers=={transformers_version}")
    if not args.model_dir.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {args.model_dir}")
    media = [*args.image, *args.video]
    if not media:
        raise ValueError("Pass at least one --image or --video")
    missing = [str(path) for path in media if not path.is_file()]
    if missing:
        raise FileNotFoundError("Media does not exist: " + ", ".join(missing))
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA was requested ({args.device}), but CUDA is unavailable")


def main() -> None:
    args = parse_args()
    _validate_args(args)
    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    model = (
        AutoModel.from_pretrained(
            args.model_dir,
            trust_remote_code=True,
            dtype=_dtype(args.dtype),
            attn_implementation="eager",
        )
        .eval()
        .to(args.device)
    )

    from PIL import Image

    images = [Image.open(path).convert("RGB") for path in args.image]
    chat_kwargs: dict = {}
    if args.video:
        from xhmodel_merak.xh_llm.models.minicpm_v_4_5.inference import encode_video

        frames = []
        temporal_ids = [[-1] for _ in images]
        for path in args.video:
            video_frames, groups = encode_video(
                str(path),
                choose_fps=args.video_fps,
                force_packing=args.video_packing,
            )
            frames.extend(video_frames)
            temporal_ids.extend(group.tolist() for group in groups)
        content = [*images, *frames, args.prompt]
        chat_kwargs.update(
            use_image_id=False,
            max_slice_nums=1,
            temporal_ids=temporal_ids,
        )
    else:
        content = [*images, args.prompt]
        chat_kwargs["max_slice_nums"] = args.max_slice_nums
    msgs = [{"role": "user", "content": content}]

    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        output = model.chat(
            image=None,
            msgs=msgs,
            tokenizer=tokenizer,
            sampling=args.do_sample,
            max_new_tokens=args.max_new_tokens,
            **chat_kwargs,
            **(
                {
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "top_k": args.top_k,
                    "repetition_penalty": args.repetition_penalty,
                }
                if args.do_sample
                else {}
            ),
        )
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    print(output)
    print(f"elapsed={elapsed:.3f}s")


if __name__ == "__main__":
    main()
