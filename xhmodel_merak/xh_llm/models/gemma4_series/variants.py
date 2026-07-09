"""Variant detection for the unified Gemma4 Series implementation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal


Gemma4SeriesVariantName = Literal["e2b", "e4b", "31b", "26b_a4b", "unknown"]
Gemma4SeriesTopology = Literal["dense", "moe"]


@dataclass(frozen=True)
class Gemma4SeriesVariantSpec:
    """Config-derived topology/capability summary for a Gemma4 checkpoint."""

    name: Gemma4SeriesVariantName
    topology: Gemma4SeriesTopology
    has_audio: bool
    has_image: bool
    has_video: bool
    has_per_layer_input: bool
    has_shared_kv_layers: bool
    attention_k_eq_v: bool
    bidirectional_vision_attention: bool
    visual_hidden_size: int | None
    audio_feature_size: int | None
    sliding_window: int | None
    local_attention_window_size: int | None
    global_attention_window_size: int | None

    @property
    def capabilities(self) -> dict[str, bool]:
        return {
            "text": True,
            "image": self.has_image,
            "video": self.has_video,
            "audio": self.has_audio,
        }


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _as_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def resolve_gemma4_series_variant(hf_config: Mapping[str, Any] | None) -> Gemma4SeriesVariantSpec:
    """Resolve Gemma4 Series variant from HF config contents, not paths.

    The three supported checkpoints share the same HF architecture name, so the
    public model class must branch on structural config facts: MoE enablement,
    audio tower, PLE width, shared KV layers, and vision/audio dimensions.
    """

    cfg = _as_mapping(hf_config)
    text_config = _as_mapping(cfg.get("text_config"))
    vision_config = _as_mapping(cfg.get("vision_config"))
    audio_config = _as_mapping(cfg.get("audio_config"))

    enable_moe = bool(text_config.get("enable_moe_block", False))
    has_audio = bool(audio_config)
    hidden_size_per_layer_input = _as_int(text_config.get("hidden_size_per_layer_input"), 0)
    num_kv_shared_layers = _as_int(text_config.get("num_kv_shared_layers"), 0)
    attention_k_eq_v = bool(text_config.get("attention_k_eq_v", False))
    bidirectional_vision_attention = text_config.get("use_bidirectional_attention") == "vision"

    if enable_moe:
        name: Gemma4SeriesVariantName = "26b_a4b"
        topology: Gemma4SeriesTopology = "moe"
    elif has_audio or hidden_size_per_layer_input > 0 or num_kv_shared_layers > 0:
        hidden_size = _as_int(text_config.get("hidden_size"), 0)
        num_hidden_layers = _as_int(text_config.get("num_hidden_layers"), 0)
        name = "e2b" if hidden_size == 1536 and num_hidden_layers == 35 else "e4b"
        topology = "dense"
    elif text_config:
        name = "31b"
        topology = "dense"
    else:
        name = "unknown"
        topology = "dense"

    sliding_window = text_config.get("sliding_window")
    context_window = text_config.get("max_position_embeddings")
    return Gemma4SeriesVariantSpec(
        name=name,
        topology=topology,
        has_audio=has_audio,
        has_image=bool(vision_config),
        has_video=bool(vision_config),
        has_per_layer_input=hidden_size_per_layer_input > 0,
        has_shared_kv_layers=num_kv_shared_layers > 0,
        attention_k_eq_v=attention_k_eq_v,
        bidirectional_vision_attention=bidirectional_vision_attention,
        visual_hidden_size=vision_config.get("hidden_size") if vision_config else None,
        audio_feature_size=audio_config.get("feature_size") if audio_config else None,
        sliding_window=sliding_window,
        local_attention_window_size=sliding_window,
        global_attention_window_size=context_window,
    )


__all__ = [
    "Gemma4SeriesTopology",
    "Gemma4SeriesVariantName",
    "Gemma4SeriesVariantSpec",
    "resolve_gemma4_series_variant",
]
