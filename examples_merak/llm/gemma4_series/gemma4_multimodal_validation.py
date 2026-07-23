#!/usr/bin/env python3
# ruff: noqa: E402,I001
"""Lightweight Gemma4 multimodal preset/processor validation.

The script is intentionally safe by default: ``--dry-run`` only resolves the
preset, HF config, modality support, and synthetic/local media plan.  It never
loads model weights and never starts quant/export.  Add ``--validate-processor``
to instantiate the HF processor and validate the prompt/media packaging only.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples_merak.llm.gemma4_series.gemma4_workflow_demo import DEFAULT_PROMPT, PRESETS, Gemma4Preset

Modality = Literal["image", "video", "audio"]
UNSUPPORTED_BY_MODEL = "unsupported_by_model"
SUPPORTED = "supported"
MIN_SYNTHETIC_VIDEO_FRAMES = 16
PRESET_ORDER = ("12b-unified", "e2b", "e4b", "31b", "26b-a4b")


@dataclass(frozen=True)
class ValidationCase:
    preset: Gemma4Preset
    modality: Modality
    support: str
    reason: str
    media: Any
    prompt: str


def _load_hf_config(hf_model_dir: str | Path) -> dict[str, Any]:
    config_path = Path(hf_model_dir) / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"HF config not found: {config_path}")
    with config_path.open(encoding="utf-8") as f:
        return json.load(f)


def _preset_with_override(preset_name: str, hf_model_dir: str | None) -> Gemma4Preset:
    preset = PRESETS[preset_name]
    if not hf_model_dir:
        return preset
    return Gemma4Preset(
        name=preset.name,
        hf_model_dir=hf_model_dir,
        output_slug=preset.output_slug,
        topology=preset.topology,
        config_path=preset.config_path,
        mtp_config_path=preset.mtp_config_path,
        assistant_model_dir=preset.assistant_model_dir,
        public_model_entry=preset.public_model_entry,
    )


def _audio_support(preset: Gemma4Preset, hf_config: dict[str, Any]) -> tuple[str, str]:
    # E-series checkpoints expose the audio tower; 12B Unified exposes the
    # encoder-free audio projection in the same config field.  31B and
    # 26B-A4B may still contain audio token ids, but they are documented as
    # unsupported by the model preset here.
    if preset.name not in {"12b-unified", "e2b", "e4b"}:
        return UNSUPPORTED_BY_MODEL, "audio is only enabled for 12B Unified and E-series presets"
    if isinstance(hf_config.get("audio_config"), dict):
        return SUPPORTED, "HF config has audio_config"
    return UNSUPPORTED_BY_MODEL, "HF config has no audio_config"


def _vision_support(hf_config: dict[str, Any], modality: Modality) -> tuple[str, str]:
    if isinstance(hf_config.get("vision_config"), dict):
        return SUPPORTED, f"HF config has vision_config for {modality}"
    return UNSUPPORTED_BY_MODEL, "HF config has no vision_config"


def _make_synthetic_image():
    from PIL import Image

    return Image.new("RGB", (32, 32), color=(32, 96, 160))


def _make_synthetic_video(num_frames: int) -> list[Any]:
    from PIL import Image

    frames = max(num_frames, MIN_SYNTHETIC_VIDEO_FRAMES)
    return [Image.new("RGB", (32, 32), color=(i % 255, 96, 160)) for i in range(frames)]


def _make_synthetic_wav(sample_rate: int = 16_000, seconds: float = 1.0) -> Path:
    path = Path(tempfile.gettempdir()) / "gemma4_validation_sine.wav"
    total = int(sample_rate * seconds)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        for i in range(total):
            value = int(0.2 * 32767 * math.sin(2 * math.pi * 440 * i / sample_rate))
            wav.writeframesraw(value.to_bytes(2, byteorder="little", signed=True))
    return path


def _resolve_media(args: argparse.Namespace, modality: Modality) -> Any:
    if modality == "image":
        return Path(args.image_path).resolve() if args.image_path else _make_synthetic_image()
    if modality == "video":
        if args.video_path:
            return Path(args.video_path).resolve()
        return _make_synthetic_video(args.video_num_frames)
    if args.audio_path:
        return Path(args.audio_path).resolve()
    return _make_synthetic_wav()


def _media_summary(media: Any) -> str:
    if isinstance(media, Path):
        return str(media)
    if isinstance(media, list):
        return f"synthetic_frames[{len(media)}]"
    return f"synthetic_{type(media).__name__}"


def build_validation_case(args: argparse.Namespace, preset_name: str, modality: Modality) -> ValidationCase:
    preset = _preset_with_override(preset_name, args.hf_model_dir)
    hf_config = _load_hf_config(preset.hf_model_dir)
    if modality == "audio":
        support, reason = _audio_support(preset, hf_config)
    else:
        support, reason = _vision_support(hf_config, modality)
    media = _resolve_media(args, modality) if support == SUPPORTED else None
    return ValidationCase(
        preset=preset,
        modality=modality,
        support=support,
        reason=reason,
        media=media,
        prompt=args.prompt,
    )


def _content_for_case(case: ValidationCase) -> list[dict[str, Any]]:
    media = str(case.media) if isinstance(case.media, Path) else case.media
    content: list[dict[str, Any]] = []
    if case.modality == "image":
        content.append({"type": "image", "image": media})
    elif case.modality == "video":
        content.append({"type": "video", "video": media})
    elif case.modality == "audio":
        content.append({"type": "audio", "audio": media})
    content.append({"type": "text", "text": case.prompt})
    return content


def validate_processor(case: ValidationCase) -> str:
    if case.support != SUPPORTED:
        return "skipped: unsupported_by_model"

    try:
        from xhmodel_merak.xh_llm.models.gemma4_series.gemma4_series_processor import (
            XHGemma4SeriesProcessor,
        )
    except Exception as exc:  # pragma: no cover - depends on local env
        return f"skipped: Gemma4 Series processor unavailable ({exc.__class__.__name__}: {exc})"

    processor = XHGemma4SeriesProcessor.from_pretrained(case.preset.hf_model_dir)
    messages = [{"role": "user", "content": _content_for_case(case)}]

    # Prefer chat-template validation because it exercises processor modality
    # schema without loading model weights.  Some processor versions return text,
    # others return tokenized tensors when tokenize=True; both are acceptable.
    try:
        rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except TypeError:
        rendered = processor.apply_chat_template(messages, add_generation_prompt=True)
    if not rendered:
        raise RuntimeError("processor.apply_chat_template returned empty output")

    return f"ok: {type(processor).__name__} chat_template"


def iter_modalities(value: str) -> list[Modality]:
    if value == "all":
        return ["image", "video", "audio"]
    return [value]  # type: ignore[list-item]


def iter_presets(value: str) -> list[str]:
    if value == "all":
        return [name for name in PRESET_ORDER if name in PRESETS]
    return [value]


def run(args: argparse.Namespace) -> None:
    for preset_name in iter_presets(args.preset):
        for modality in iter_modalities(args.modality):
            case = build_validation_case(args, preset_name, modality)
            print(f"preset={case.preset.name} topology={case.preset.topology} modality={case.modality}")
            print(f"  hf_model_dir={case.preset.hf_model_dir}")
            print(f"  support={case.support} reason={case.reason}")
            if case.support == SUPPORTED:
                print(f"  media={_media_summary(case.media)}")
            if args.validate_processor:
                print(f"  processor={validate_processor(case)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=["all", *sorted(PRESETS)], default="all")
    parser.add_argument("--modality", choices=("all", "image", "video", "audio"), default="all")
    parser.add_argument("--hf-model-dir", help="Override the HF model dir for a single --preset run.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--image-path", help="Use a local image path instead of a synthetic PIL image.")
    parser.add_argument("--video-path", help="Use a local video path instead of synthetic frames.")
    parser.add_argument(
        "--video-num-frames",
        type=int,
        default=MIN_SYNTHETIC_VIDEO_FRAMES,
        help=f"Synthetic video frame count; clamped to at least {MIN_SYNTHETIC_VIDEO_FRAMES}.",
    )
    parser.add_argument("--audio-path", help="Use a local audio path instead of a generated sine-wave wav.")
    parser.add_argument(
        "--validate-processor",
        action="store_true",
        help="Instantiate HF AutoProcessor and validate chat-template packaging.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Compatibility flag: this script is dry-run by default.")
    args = parser.parse_args()
    if args.hf_model_dir and args.preset == "all":
        parser.error("--hf-model-dir can only be used with a single --preset")
    if args.video_num_frames < 1:
        parser.error("--video-num-frames must be positive")
    return args


if __name__ == "__main__":
    run(parse_args())
