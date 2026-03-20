# Copyright 2025 HOUMO AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration as HFQwen3VLForConditionalGeneration

from xh_model_zoo.xh_llm.models.qwen3_vl import Qwen3VLForConditionalGeneration, Qwen3VLProcessor


@dataclass
class CompareResult:
    name: str
    max_abs_diff: float
    passed: bool


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument(
        "--backend",
        type=str,
        default="both",
        choices=["transformers", "xh2", "both"],
        help="选择只运行 transformers、本地 xh2modelzoo，或者同时做一致性对比。",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32", "bfloat16"])
    parser.add_argument("--mode", type=str, default="video", choices=["image", "video"])
    parser.add_argument("--image-path", type=str, default=None)
    parser.add_argument("--video-path", type=str, default=None)
    parser.add_argument("--sample-dir", type=str, default="data/test")
    parser.add_argument("--video-frames", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--atol", type=float, default=1e-4)
    return parser.parse_args()


def build_dtype(dtype_name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }[dtype_name]


def create_synthetic_image(image_size: int) -> Image.Image:
    gradient = np.linspace(0, 255, image_size * image_size * 3, dtype=np.uint8).reshape(image_size, image_size, 3)
    return Image.fromarray(gradient)


def create_synthetic_video(num_frames: int, image_size: int) -> list[Image.Image]:
    frames = []
    for frame_index in range(num_frames):
        value = (frame_index * 37) % 255
        frame = np.full((image_size, image_size, 3), value, dtype=np.uint8)
        frame[..., 1] = np.arange(image_size, dtype=np.uint8)[:, None]
        frame[..., 2] = np.arange(image_size, dtype=np.uint8)[None, :]
        frames.append(Image.fromarray(frame))
    return frames


def find_sample_file(sample_dir: str, suffixes: tuple[str, ...]) -> str | None:
    directory = Path(sample_dir)
    if not directory.exists():
        return None
    for suffix in suffixes:
        matches = sorted(directory.glob(f"*{suffix}"))
        if matches:
            return str(matches[0])
    return None


def load_image(image_path: str) -> Image.Image:
    return Image.open(image_path).convert("RGB")


def move_inputs(inputs, device: str):
    moved = {}
    for key, value in inputs.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        elif isinstance(value, list):
            moved[key] = [item.to(device) if torch.is_tensor(item) else item for item in value]
        else:
            moved[key] = value
    return moved


def build_standard_chain_inputs(inputs):
    standard_inputs = dict(inputs)
    standard_inputs.pop("hm_pixel_values", None)
    return standard_inputs


def run_model_once(model, inputs, args):
    model = model.to(args.device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits.detach().float().cpu()
        generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False).cpu()
    model = model.cpu()
    if torch.cuda.is_available() and args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    return logits, generated


def compare_tensor(name: str, lhs: torch.Tensor, rhs: torch.Tensor, atol: float) -> CompareResult:
    lhs = lhs.detach().float().cpu()
    rhs = rhs.detach().float().cpu()
    max_abs_diff = float((lhs - rhs).abs().max().item())
    return CompareResult(name=name, max_abs_diff=max_abs_diff, passed=max_abs_diff <= atol)


def build_demo_inputs(args, processor):
    if args.mode == "image":
        image_path = args.image_path or find_sample_file(args.sample_dir, (".png", ".jpg", ".jpeg", ".bmp", ".webp"))
        if image_path is not None:
            image = load_image(image_path)
            text = f"<|vision_start|><|image_pad|><|vision_end|>\nDescribe the image."
            print(f"using_image={image_path}")
        else:
            image = create_synthetic_image(args.image_size)
            text = f"<|vision_start|><|image_pad|><|vision_end|>\nDescribe the synthetic image."
            print("using_image=synthetic")
        inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
    else:
        video_path = args.video_path or find_sample_file(args.sample_dir, (".mp4", ".mov", ".avi", ".mkv", ".webm"))
        if video_path is not None:
            text = f"<|vision_start|><|video_pad|><|vision_end|>\nDescribe the video."
            print(f"using_video={video_path}, input_mode=path_with_metadata")
            inputs = processor(text=[text], videos=[video_path], padding=True, return_tensors="pt")
        else:
            frames = create_synthetic_video(args.video_frames, args.image_size)
            text = f"<|vision_start|><|video_pad|><|vision_end|>\nDescribe the synthetic video."
            print(f"using_video=synthetic, sampled_frames={len(frames)}")
            inputs = processor(text=[text], videos=[frames], padding=True, return_tensors="pt")
    return inputs


def run_single_backend(name, model, processor, args):
    raw_inputs = build_demo_inputs(args, processor)
    inputs = move_inputs(build_standard_chain_inputs(raw_inputs), args.device)

    logits, generated = run_model_once(model, inputs, args)
    decoded = processor.batch_decode(generated, skip_special_tokens=False, clean_up_tokenization_spaces=False)

    print(f"backend={name}")
    print(f"logits_shape={tuple(logits.shape)}")
    print(f"generated_ids_shape={tuple(generated.shape)}")
    print(f"generated_text={decoded[0]}")

    return raw_inputs, logits, generated


def main():
    args = parse_args()
    dtype = build_dtype(args.dtype)

    if args.backend == "transformers":
        hf_processor = AutoProcessor.from_pretrained(args.model_path)
        hf_model = HFQwen3VLForConditionalGeneration.from_pretrained(
            args.model_path,
            torch_dtype=dtype,
            device_map=None,
        ).eval().to(args.device)
        run_single_backend("transformers", hf_model, hf_processor, args)
        return

    if args.backend == "xh2":
        local_processor = Qwen3VLProcessor.from_pretrained(args.model_path)
        local_model = Qwen3VLForConditionalGeneration.from_pretrained(
            args.model_path,
            torch_dtype=dtype,
            device_map=None,
        ).eval().to(args.device)
        run_single_backend("xh2", local_model, local_processor, args)
        return

    hf_processor = AutoProcessor.from_pretrained(args.model_path)
    local_processor = Qwen3VLProcessor.from_pretrained(args.model_path)

    hf_model = HFQwen3VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        device_map=None,
    ).eval()
    local_model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        device_map=None,
    ).eval()

    local_model.load_state_dict(hf_model.state_dict(), strict=True)

    hf_inputs = build_demo_inputs(args, hf_processor)
    local_inputs = build_demo_inputs(args, local_processor)

    common_keys = sorted(set(hf_inputs.keys()) & set(local_inputs.keys()))
    for key in common_keys:
        hf_value = hf_inputs[key]
        local_value = local_inputs[key]
        if torch.is_tensor(hf_value):
            result = compare_tensor(f"processor:{key}", hf_value, local_value, atol=args.atol)
            print(f"{result.name}: max_abs_diff={result.max_abs_diff:.6g}, passed={result.passed}")
        else:
            passed = hf_value == local_value
            print(f"processor:{key}: passed={passed}")

    hf_inputs = move_inputs(build_standard_chain_inputs(hf_inputs), args.device)
    local_inputs = move_inputs(build_standard_chain_inputs(local_inputs), args.device)

    hf_logits, hf_generated = run_model_once(hf_model, hf_inputs, args)
    local_logits, local_generated = run_model_once(local_model, local_inputs, args)

    logits_result = compare_tensor("logits", hf_logits, local_logits, atol=args.atol)
    print(f"{logits_result.name}: max_abs_diff={logits_result.max_abs_diff:.6g}, passed={logits_result.passed}")

    generation_equal = torch.equal(hf_generated, local_generated)
    print(f"generation_equal={generation_equal}")


if __name__ == "__main__":
    main()