"""Native Transformers floating-point baseline for MiniCPM-V-4.6."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from packaging.version import Version
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers import __version__ as transformers_version


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run native MiniCPM-V-4.6 inference.")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--image", type=Path, action="append", default=[])
    parser.add_argument("--video", type=Path, action="append", default=[])
    parser.add_argument("--prompt", default="请描述这张图片。")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("auto", "bf16", "fp16", "fp32"), default="auto")
    parser.add_argument("--downsample-mode", choices=("4x", "16x"), default="4x")
    parser.add_argument("--max-slice-nums", type=int, default=36)
    parser.add_argument("--video-max-slice-nums", type=int, default=1)
    parser.add_argument("--video-max-num-frames", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
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
    if Version(transformers_version) < Version("5.7.0"):
        raise RuntimeError(f"MiniCPM-V-4.6 requires transformers>=5.7.0; found transformers=={transformers_version}")
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


def _build_inputs(processor, args: argparse.Namespace):
    content = [{"type": "image", "path": str(path.resolve())} for path in args.image]
    content.extend({"type": "video", "path": str(path.resolve())} for path in args.video)
    content.append({"type": "text", "text": args.prompt})
    messages = [{"role": "user", "content": content}]
    return processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        processor_kwargs={
            "images_kwargs": {
                "downsample_mode": args.downsample_mode,
                "max_slice_nums": args.max_slice_nums,
            },
            "videos_kwargs": {
                "downsample_mode": args.downsample_mode,
                "max_slice_nums": args.video_max_slice_nums,
                "max_num_frames": args.video_max_num_frames,
            },
        },
    )


def main() -> None:
    args = parse_args()
    _validate_args(args)
    torch.manual_seed(args.seed)

    processor = AutoProcessor.from_pretrained(args.model_dir)
    model = (
        AutoModelForImageTextToText.from_pretrained(
            args.model_dir,
            dtype=_dtype(args.dtype),
            attn_implementation="eager",
        )
        .eval()
        .to(args.device)
    )
    inputs = _build_inputs(processor, args)
    prompt_length = int(inputs["input_ids"].shape[-1])
    inputs = inputs.to(args.device)

    generation_kwargs = {
        "downsample_mode": args.downsample_mode,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
        "repetition_penalty": args.repetition_penalty,
        "disable_compile": True,
    }
    if args.do_sample:
        generation_kwargs.update(
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
        )
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        generated_ids = model.generate(**inputs, **generation_kwargs)
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    generated_ids = generated_ids[:, prompt_length:]
    output = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    print(output)
    print(
        f"tokens={generated_ids.shape[-1]}, elapsed={elapsed:.3f}s, "
        f"throughput={generated_ids.shape[-1] / elapsed:.2f} token/s"
    )


if __name__ == "__main__":
    main()
