"""Gemma4 attention visibility facts independent of graph lowering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


_SUPPORTED_LAYER_TYPES = frozenset({"full_attention", "sliding_attention"})


@dataclass(frozen=True)
class Gemma4AttentionVisibilitySpec:
    """Checkpoint-derived attention semantics shared by legacy and Flash graphs.

    ``attention_contract_version`` is intentionally absent.  It selects how
    these facts are lowered (legacy mask graph or FlashAttention), not which
    keys a query is allowed to see.
    """

    layer_types: tuple[str, ...]
    is_causal: bool
    bidirectional_vision_attention: bool
    sliding_window: int | None
    max_mm_ranges_per_chunk: int

    def __post_init__(self) -> None:
        unknown = sorted(set(self.layer_types) - _SUPPORTED_LAYER_TYPES)
        if unknown:
            raise ValueError(f"Gemma4 attention visibility has unsupported layer types: {unknown}")
        if not self.is_causal:
            raise ValueError("Gemma4 text attention visibility must be causal")
        if self.has_sliding_attention:
            if type(self.sliding_window) is not int or self.sliding_window <= 0:
                raise ValueError(
                    "Gemma4 sliding attention visibility requires a positive sliding_window, "
                    f"got {self.sliding_window!r}"
                )
        elif self.sliding_window is not None:
            raise ValueError(
                "Gemma4 full-only attention visibility requires sliding_window=None, "
                f"got {self.sliding_window!r}"
            )
        if self.max_mm_ranges_per_chunk <= 0:
            raise ValueError("Gemma4 max_mm_ranges_per_chunk must be positive")

    @property
    def has_full_attention(self) -> bool:
        return "full_attention" in self.layer_types

    @property
    def has_sliding_attention(self) -> bool:
        return "sliding_attention" in self.layer_types

    @property
    def requires_kv_window_metadata(self) -> bool:
        return self.has_sliding_attention

    @property
    def requires_mm_prefix_ranges(self) -> bool:
        return self.bidirectional_vision_attention

    @property
    def requires_compact_metadata(self) -> bool:
        return self.requires_kv_window_metadata or self.requires_mm_prefix_ranges

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer_types": list(self.layer_types),
            "is_causal": self.is_causal,
            "bidirectional_vision_attention": self.bidirectional_vision_attention,
            "sliding_window": self.sliding_window,
            "max_mm_ranges_per_chunk": self.max_mm_ranges_per_chunk,
            "has_full_attention": self.has_full_attention,
            "has_sliding_attention": self.has_sliding_attention,
            "requires_kv_window_metadata": self.requires_kv_window_metadata,
            "requires_mm_prefix_ranges": self.requires_mm_prefix_ranges,
        }

    @classmethod
    def from_checkpoint_semantics(
        cls,
        *,
        layer_types: Sequence[str],
        sliding_window: int | None,
        bidirectional_vision_attention: bool,
        max_mm_ranges_per_chunk: int,
    ) -> "Gemma4AttentionVisibilitySpec":
        normalized_layer_types = tuple(str(value) for value in layer_types)
        normalized_window = (
            int(sliding_window)
            if "sliding_attention" in normalized_layer_types and sliding_window is not None
            else None
        )
        return cls(
            layer_types=normalized_layer_types,
            is_causal=True,
            bidirectional_vision_attention=bool(bidirectional_vision_attention),
            sliding_window=normalized_window,
            max_mm_ranges_per_chunk=int(max_mm_ranges_per_chunk),
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Gemma4AttentionVisibilitySpec":
        return cls.from_checkpoint_semantics(
            layer_types=value.get("layer_types", ()),
            sliding_window=value.get("sliding_window"),
            bidirectional_vision_attention=bool(value.get("bidirectional_vision_attention", False)),
            max_mm_ranges_per_chunk=int(value.get("max_mm_ranges_per_chunk", 1)),
        )


def resolve_gemma4_attention_visibility_spec(
    value: Gemma4AttentionVisibilitySpec | Mapping[str, Any] | None,
    *,
    layer_types: Sequence[str],
    sliding_window: int | None,
    bidirectional_vision_attention: bool,
    max_mm_ranges_per_chunk: int,
) -> Gemma4AttentionVisibilitySpec:
    """Normalize serialized/new and legacy visibility configuration."""

    if isinstance(value, Gemma4AttentionVisibilitySpec):
        return value
    if isinstance(value, Mapping) and value:
        return Gemma4AttentionVisibilitySpec.from_dict(value)
    return Gemma4AttentionVisibilitySpec.from_checkpoint_semantics(
        layer_types=layer_types,
        sliding_window=sliding_window,
        bidirectional_vision_attention=bidirectional_vision_attention,
        max_mm_ranges_per_chunk=max_mm_ranges_per_chunk,
    )


__all__ = [
    "Gemma4AttentionVisibilitySpec",
    "resolve_gemma4_attention_visibility_spec",
]
