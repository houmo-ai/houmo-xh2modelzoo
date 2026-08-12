from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image


MODES = ("t2i", "t2v", "ti2v")
_TORCHVISION_SCHEMA_LIBRARIES = []


def _ensure_torchvision_nms_schema() -> None:
    try:
        import torchvision  # noqa: F401

        return
    except RuntimeError as error:
        if "operator torchvision::nms does not exist" not in str(error):
            raise

    try:
        torch._C._dispatch_has_kernel_for_dispatch_key("torchvision::nms", "Meta")
    except RuntimeError:
        library = torch.library.Library("torchvision", "DEF")
        library.define("nms(Tensor dets, Tensor scores, float iou_threshold) -> Tensor")
        _TORCHVISION_SCHEMA_LIBRARIES.append(library)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a LingBot Video HMONNX pipeline.")
    parser.add_argument("--mode", required=True, choices=MODES)
    parser.add_argument("--export-dir", help="Self-contained export directory containing all HMONNX components.")
    parser.add_argument("--text-visual-export-dir", help="Export directory containing text_encoder and visual_encoder.")
    parser.add_argument("--transformer-export-dir", help="Export directory containing transformer.")
    parser.add_argument("--vae-export-dir", help="Export directory containing VAE encoder and decoder.")
    parser.add_argument("--prompt-json", required=True)
    parser.add_argument("--image", help="First frame for TI2V mode.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        help="Override the exported scheduler step count for short smoke tests.",
    )
    parser.add_argument(
        "--allow-duration-mismatch",
        action="store_true",
        help="Allow a video profile whose frame count differs from the prompt JSON duration.",
    )
    args = parser.parse_args()
    component_dirs = [args.text_visual_export_dir, args.transformer_export_dir, args.vae_export_dir]
    if args.export_dir is None and not all(component_dirs):
        raise ValueError(
            "Provide --export-dir, or provide all of --text-visual-export-dir, "
            "--transformer-export-dir, and --vae-export-dir."
        )
    if args.export_dir is not None and any(component_dirs):
        raise ValueError("Use either --export-dir or the three component export dir arguments, not both.")
    return args


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
    _ensure_torchvision_nms_schema()
    args = parse_args()
    export_dir = Path(args.export_dir).resolve() if args.export_dir is not None else None
    text_visual_export_dir = (
        Path(args.text_visual_export_dir).resolve() if args.text_visual_export_dir is not None else export_dir
    )
    transformer_export_dir = (
        Path(args.transformer_export_dir).resolve() if args.transformer_export_dir is not None else export_dir
    )
    vae_export_dir = Path(args.vae_export_dir).resolve() if args.vae_export_dir is not None else export_dir
    prompt_json = Path(args.prompt_json).resolve()
    output_file = Path(args.output).resolve()
    device = torch.device(args.device)
    root_meta = json.loads((transformer_export_dir / "export_meta_info.json").read_text(encoding="utf-8"))
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
        build_hmonnx_pipeline_components_from_dirs,
        configure_pipeline_token_length,
        resolve_runtime_artifact,
    )

    if args.mode == "ti2v":
        from lingbot_video.pipeline_lingbot_video_i2v import LingBotVideoImageToVideoPipeline

        pipeline_class = LingBotVideoImageToVideoPipeline
    else:
        pipeline_class = LingBotVideoPipeline

    if export_dir is not None:
        transformer, vae, text_encoder, processor = build_hmonnx_pipeline_components(
            export_dir=export_dir,
            device=device,
        )
    else:
        transformer, vae, text_encoder, processor = build_hmonnx_pipeline_components_from_dirs(
            text_visual_export_dir=text_visual_export_dir,
            transformer_export_dir=transformer_export_dir,
            vae_export_dir=vae_export_dir,
            device=device,
        )
    scheduler = FlowUniPCMultistepScheduler.from_pretrained(
        str(resolve_runtime_artifact(transformer_export_dir, "scheduler"))
    )
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
        "num_inference_steps": args.num_inference_steps or int(transformer_meta["num_inference_steps"]),
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
