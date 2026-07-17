from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image


MODES = ("t2i", "t2v", "ti2v")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a LingBot Video HMONNX pipeline.")
    parser.add_argument("--mode", required=True, choices=MODES)
    parser.add_argument("--export-dir", required=True)
    parser.add_argument("--prompt-json", required=True)
    parser.add_argument("--image", help="First frame for TI2V mode.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument(
        "--allow-duration-mismatch",
        action="store_true",
        help="Allow a video profile whose frame count differs from the prompt JSON duration.",
    )
    return parser.parse_args()


def _sample_from_json(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        if not data:
            raise ValueError(f"Empty prompt JSON: {path}")
        data = data[0]
    if not isinstance(data, dict):
        raise TypeError("prompt JSON must contain an object or a non-empty object list")
    return data


def _validate_static_profile(mode: str, artifact_mode: str, num_frames: int) -> None:
    if artifact_mode != mode:
        raise ValueError(f"Requested mode={mode!r}, but the HMONNX artifact uses mode={artifact_mode!r}")
    if mode == "t2i" and num_frames != 1:
        raise ValueError("T2I requires an HMONNX artifact exported with num_frames=1")
    if mode != "t2i" and num_frames <= 1:
        raise ValueError(f"{mode.upper()} requires an HMONNX artifact exported with num_frames > 1")


def _validate_prompt_duration(sample: dict, *, artifact_num_frames: int, fps: int) -> None:
    if "duration" not in sample:
        return

    from lingbot_video.utils import num_frames_from_duration

    prompt_num_frames = num_frames_from_duration(float(sample["duration"]), fps)
    if prompt_num_frames != artifact_num_frames:
        raise ValueError(
            f"Prompt duration={sample['duration']} at fps={fps} requires "
            f"num_frames={prompt_num_frames}, but this static HMONNX artifact uses "
            f"num_frames={artifact_num_frames}. Re-export that frame profile, or pass "
            "--allow-duration-mismatch for a short profile smoke test."
        )


def main() -> None:
    args = parse_args()
    export_dir = Path(args.export_dir).resolve()
    prompt_json = Path(args.prompt_json).resolve()
    output_file = Path(args.output).resolve()
    device = torch.device(args.device)
    root_meta = json.loads((export_dir / "export_meta_info.json").read_text(encoding="utf-8"))
    transformer_meta = root_meta["transformer"]
    num_frames = int(transformer_meta["num_frames"])
    artifact_mode = str(root_meta["geometry"].get("mode", ""))
    _validate_static_profile(args.mode, artifact_mode, num_frames)

    prompt_sample = _sample_from_json(prompt_json)
    if args.mode != "t2i" and not args.allow_duration_mismatch:
        _validate_prompt_duration(prompt_sample, artifact_num_frames=num_frames, fps=args.fps)

    first_frame = None
    if args.mode == "ti2v":
        if args.image is None:
            raise ValueError("--image is required for TI2V mode")
        image_file = Path(args.image).resolve()
        if not image_file.is_file():
            raise FileNotFoundError(f"Missing TI2V first frame: {image_file}")
        with Image.open(image_file) as source_image:
            first_frame = source_image.convert("RGB")

    from lingbot_video.pipeline_lingbot_video import (
        DEFAULT_NEGATIVE_PROMPT,
        DEFAULT_NEGATIVE_PROMPT_IMAGE,
        LingBotVideoPipeline,
    )
    from lingbot_video.scheduling_flow_unipc import FlowUniPCMultistepScheduler
    from lingbot_video.utils import caption_from_sample

    from xhmodel_merak.xh_other_model.models.lingbot_video.components_hmonnx import (
        build_hmonnx_pipeline_components,
        configure_pipeline_token_length,
        resolve_runtime_artifact,
    )

    if args.mode == "ti2v":
        from lingbot_video.pipeline_lingbot_video_i2v import LingBotVideoImageToVideoPipeline

        pipeline_class = LingBotVideoImageToVideoPipeline
    else:
        pipeline_class = LingBotVideoPipeline

    transformer, vae, text_encoder, processor = build_hmonnx_pipeline_components(
        export_dir=export_dir,
        device=device,
    )
    scheduler = FlowUniPCMultistepScheduler.from_pretrained(str(resolve_runtime_artifact(export_dir, "scheduler")))
    pipeline = pipeline_class(
        transformer=transformer,
        vae=vae,
        text_encoder=text_encoder,
        processor=processor,
        scheduler=scheduler,
    )
    configure_pipeline_token_length(pipeline, text_encoder)

    call_kwargs = {
        "prompt": caption_from_sample(prompt_sample),
        "negative_prompt": DEFAULT_NEGATIVE_PROMPT_IMAGE if args.mode == "t2i" else DEFAULT_NEGATIVE_PROMPT,
        "height": int(transformer_meta["height"]),
        "width": int(transformer_meta["width"]),
        "num_frames": num_frames,
        "num_inference_steps": int(transformer_meta["num_inference_steps"]),
        "guidance_scale": float(root_meta["geometry"].get("guidance_scale", 3.0)),
        "shift": float(transformer_meta["shift"]),
        "generator": torch.Generator(device=device).manual_seed(int(root_meta["geometry"].get("seed", 42))),
        "output_type": "latent",
        "batch_cfg": False,
    }
    if first_frame is not None:
        call_kwargs["image"] = first_frame

    latents = pipeline(**call_kwargs).frames
    frames = pipeline._decode_latents(latents)[0]
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if args.mode == "t2i":
        image = np.clip(np.asarray(frames[0]) * 255.0, 0, 255).astype(np.uint8)
        Image.fromarray(image).save(output_file)
        print(f"saved {output_file}")
        return

    from diffusers.utils import export_to_video

    export_to_video(frames, str(output_file), fps=args.fps)
    print(f"saved {output_file} frames={len(frames)} fps={args.fps}")


if __name__ == "__main__":
    main()
