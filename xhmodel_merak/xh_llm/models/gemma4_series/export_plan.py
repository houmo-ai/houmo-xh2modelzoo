"""Export planning for the unified Gemma4 Series workflow."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .variants import Gemma4SeriesVariantSpec, resolve_gemma4_series_variant


REQUIRED_CONTEXT_MAX_LENGTH = 2048
REQUIRED_INPUT_SEQUENCE_LENGTH = 320
MIN_INPUT_SEQUENCE_LENGTH = 280
DEFAULT_IMAGE_VISUAL_SEQ_LENGTH = 280
DEFAULT_IMAGE_VISUAL_MAX_PATCHES = 2520
DEFAULT_VIDEO_VISUAL_SEQ_LENGTH = 70
DEFAULT_VIDEO_VISUAL_MAX_PATCHES = 630
DEFAULT_QUANT_TYPE = "w8a8h1_sefp"
ALLOWED_QUANT_TYPES = (DEFAULT_QUANT_TYPE, "w4a8h0_ssfp")


@dataclass(frozen=True)
class Gemma4SeriesExportPlan:
    """Capability-level export plan derived from HF config and YAML.

    The plan is intentionally small: it records *what* the series workflow
    should export.  Vision/audio/LLM modules still own *how* to export their
    subgraphs.
    """

    variant: Gemma4SeriesVariantSpec
    context_max_length: int
    input_sequence_length: int
    quant_type: str | None
    export_image_visual: bool
    export_video_visual: bool
    export_audio: bool
    export_per_layer_input: bool
    image_visual_seq_length: int | None
    image_visual_max_patches: int | None
    video_visual_seq_length: int | None
    video_visual_max_patches: int | None

    @property
    def name(self) -> str:
        return self.variant.name

    @property
    def capabilities(self) -> dict[str, bool]:
        return self.variant.capabilities

    def validate_fixed_contract(self) -> None:
        """Validate the Gemma4 Series public export contract."""

        if self.input_sequence_length < MIN_INPUT_SEQUENCE_LENGTH:
            raise ValueError(
                "Gemma4 Series exports must use prefill/input length >= "
                f"{MIN_INPUT_SEQUENCE_LENGTH}, got {self.input_sequence_length}"
            )
        if self.context_max_length < self.input_sequence_length:
            raise ValueError(
                "Gemma4 Series context_max_length must be >= prefill/input length; "
                f"got context_max_length={self.context_max_length}, "
                f"input_sequence_length={self.input_sequence_length}"
            )
        if self.quant_type not in ALLOWED_QUANT_TYPES:
            allowed_quant_types = ", ".join(repr(quant_type) for quant_type in ALLOWED_QUANT_TYPES)
            raise ValueError(
                "Gemma4 Series export.model.quant_scheme.quant_type must be "
                f"one of [{allowed_quant_types}], got {self.quant_type!r}"
            )
        if not self.export_image_visual:
            raise ValueError("Gemma4 Series exports require an image visual subgraph")
        if not self.export_video_visual:
            raise ValueError("Gemma4 Series exports require a separate video visual subgraph")
        if self.image_visual_seq_length != DEFAULT_IMAGE_VISUAL_SEQ_LENGTH:
            raise ValueError(
                "Gemma4 image visual must use image_seq_length="
                f"{DEFAULT_IMAGE_VISUAL_SEQ_LENGTH}, got {self.image_visual_seq_length}"
            )
        if self.image_visual_max_patches != DEFAULT_IMAGE_VISUAL_MAX_PATCHES:
            raise ValueError(
                "Gemma4 image visual must use max_patches="
                f"{DEFAULT_IMAGE_VISUAL_MAX_PATCHES}, got {self.image_visual_max_patches}"
            )
        if self.video_visual_seq_length != DEFAULT_VIDEO_VISUAL_SEQ_LENGTH:
            raise ValueError(
                "Gemma4 video visual must use image_seq_length="
                f"{DEFAULT_VIDEO_VISUAL_SEQ_LENGTH}, got {self.video_visual_seq_length}"
            )
        if self.video_visual_max_patches != DEFAULT_VIDEO_VISUAL_MAX_PATCHES:
            raise ValueError(
                "Gemma4 video visual must use max_patches="
                f"{DEFAULT_VIDEO_VISUAL_MAX_PATCHES}, got {self.video_visual_max_patches}"
            )

    def to_log_dict(self) -> dict[str, Any]:
        return {
            "variant": self.name,
            "topology": self.variant.topology,
            "capabilities": self.capabilities,
            "context_max_length": self.context_max_length,
            "input_sequence_length": self.input_sequence_length,
            "quant_type": self.quant_type,
            "exports": {
                "image_visual": self.export_image_visual,
                "video_visual": self.export_video_visual,
                "audio": self.export_audio,
                "per_layer_input": self.export_per_layer_input,
            },
        }


def build_gemma4_series_export_plan(
    *,
    hf_model_dir: str,
    export_model_cfg: Mapping[str, Any],
) -> Gemma4SeriesExportPlan:
    """Build an explicit export plan from HF config and workflow model config."""

    hf_config = _load_hf_config(hf_model_dir)
    variant = resolve_gemma4_series_variant(hf_config)

    if export_model_cfg.get("mm_prefill_chunk_length") is not None:
        raise ValueError(
            "Gemma4 Series no longer supports mm_prefill_chunk_length/prefill_mm; "
            "use prefill_chunk_length instead."
        )

    visual_cfg = _as_mapping(export_model_cfg.get("visual_config"))
    video_visual_cfg = _as_mapping(export_model_cfg.get("video_visual_config"))
    quant_scheme = _as_mapping(export_model_cfg.get("quant_scheme"))

    return Gemma4SeriesExportPlan(
        variant=variant,
        context_max_length=_as_int(export_model_cfg.get("context_max_length"), 0),
        input_sequence_length=_as_int(
            export_model_cfg.get("prefill_chunk_length", export_model_cfg.get("input_sequence_length")),
            REQUIRED_INPUT_SEQUENCE_LENGTH,
        ),
        quant_type=quant_scheme.get("quant_type"),
        export_image_visual=bool(variant.has_image and visual_cfg),
        export_video_visual=bool(variant.has_video and video_visual_cfg),
        export_audio=bool(variant.has_audio),
        export_per_layer_input=bool(variant.has_per_layer_input),
        image_visual_seq_length=_optional_int(visual_cfg.get("image_seq_length")),
        image_visual_max_patches=_optional_int(visual_cfg.get("max_patches")),
        video_visual_seq_length=_optional_int(video_visual_cfg.get("image_seq_length")),
        video_visual_max_patches=_optional_int(video_visual_cfg.get("max_patches")),
    )


def _load_hf_config(hf_model_dir: str) -> dict[str, Any]:
    config_path = Path(hf_model_dir) / "config.json"
    if not config_path.exists():
        return {}
    return json.loads(config_path.read_text(encoding="utf-8"))


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _as_int(value: Any, default: int) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _optional_int(value: Any) -> int | None:
    return None if value is None else _as_int(value, 0)


__all__ = [
    "DEFAULT_IMAGE_VISUAL_MAX_PATCHES",
    "DEFAULT_IMAGE_VISUAL_SEQ_LENGTH",
    "DEFAULT_QUANT_TYPE",
    "DEFAULT_VIDEO_VISUAL_MAX_PATCHES",
    "DEFAULT_VIDEO_VISUAL_SEQ_LENGTH",
    "ALLOWED_QUANT_TYPES",
    "Gemma4SeriesExportPlan",
    "REQUIRED_CONTEXT_MAX_LENGTH",
    "MIN_INPUT_SEQUENCE_LENGTH",
    "REQUIRED_INPUT_SEQUENCE_LENGTH",
    "build_gemma4_series_export_plan",
]
