"""Host/runtime cache ABI for fixed-shape DeepSeek-V4 Flash execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

from torch import Tensor

from .static_cache import DeepSeekV4StaticCacheSpec
from .swa_layout import aligned_swa_attention_length


SLIDING = "sliding_attention"
CSA = "compressed_sparse_attention"
HCA = "heavily_compressed_attention"

_GRAPH_CACHE_FIELDS = {
    SLIDING: ("swa_kv",),
    CSA: (
        "swa_kv",
        "main",
        "index_k",
        "main_kv_state",
        "main_score_state",
        "index_kv_state",
        "index_score_state",
    ),
    HCA: (
        "swa_kv",
        "main",
        "main_kv_state",
        "main_score_state",
    ),
}


class SWACacheInputs(NamedTuple):
    swa_kv: Tensor


class CSACacheInputs(NamedTuple):
    swa_kv: Tensor
    main: Tensor
    index_k: Tensor
    main_kv_state: Tensor
    main_score_state: Tensor
    index_kv_state: Tensor
    index_score_state: Tensor


class HCACacheInputs(NamedTuple):
    swa_kv: Tensor
    main: Tensor
    main_kv_state: Tensor
    main_score_state: Tensor


class CSAStateOutput(NamedTuple):
    main_kv_state: Tensor
    main_score_state: Tensor
    index_kv_state: Tensor
    index_score_state: Tensor


class HCAStateOutput(NamedTuple):
    main_kv_state: Tensor
    main_score_state: Tensor


LayerCacheInputs = SWACacheInputs | CSACacheInputs | HCACacheInputs
FlatCacheInputs = tuple[Tensor, ...]


def flatten_cache_inputs(caches: tuple[LayerCacheInputs, ...]) -> FlatCacheInputs:
    return tuple(value for cache in caches for value in cache)


def default_layer_types(num_layers: int = 43) -> tuple[str, ...]:
    """Return the released Flash checkpoint's static attention schedule."""

    num_layers = int(num_layers)
    if not 1 <= num_layers <= 43:
        raise ValueError("num_layers must be in [1, 43]")
    schedule = (SLIDING, SLIDING) + tuple(CSA if index % 2 == 0 else HCA for index in range(2, 43))
    return schedule[:num_layers]


@dataclass(frozen=True)
class DeepSeekV4CacheABI:
    """Shapes and ordering shared by export and runtime cache handling."""

    spec: DeepSeekV4StaticCacheSpec = field(default_factory=DeepSeekV4StaticCacheSpec)
    layer_types: tuple[str, ...] = field(default_factory=default_layer_types)
    batch_size: int = 1

    def __post_init__(self) -> None:
        if int(self.batch_size) <= 0:
            raise ValueError("batch_size must be positive")
        if not self.layer_types or len(self.layer_types) > 43:
            raise ValueError("layer_types must contain between 1 and 43 entries")
        unsupported = set(self.layer_types) - set(_GRAPH_CACHE_FIELDS)
        if unsupported:
            raise ValueError(f"unsupported layer types: {sorted(unsupported)}")

    @property
    def swa_layers(self) -> tuple[int, ...]:
        return tuple(index for index, kind in enumerate(self.layer_types) if kind == SLIDING)

    @property
    def csa_layers(self) -> tuple[int, ...]:
        return tuple(index for index, kind in enumerate(self.layer_types) if kind == CSA)

    @property
    def hca_layers(self) -> tuple[int, ...]:
        return tuple(index for index, kind in enumerate(self.layer_types) if kind == HCA)

    @property
    def persistent_swa_length(self) -> int:
        return self.spec.swa_physical_length

    def graph_swa_length(self, stage: str) -> int:
        if str(stage).lower() not in {"prefill", "decode"}:
            raise ValueError("stage must be prefill or decode")
        return self.persistent_swa_length

    def attention_swa_length(self, stage: str) -> int:
        stage = str(stage).lower()
        if stage == "prefill":
            input_sequence_length = self.spec.prefill_chunk_length
        elif stage == "decode":
            input_sequence_length = 1
        else:
            raise ValueError("stage must be prefill or decode")
        return aligned_swa_attention_length(
            input_sequence_length,
            self.spec.sliding_window,
        )

    def attention_mask_shapes(self, stage: str) -> dict[str, tuple[int, ...]]:
        stage = str(stage).lower()
        if stage == "prefill":
            query_length = self.spec.prefill_chunk_length
        elif stage == "decode":
            query_length = 1
        else:
            raise ValueError("stage must be prefill or decode")
        swa_length = self.attention_swa_length(stage)
        return {
            "swa_attention_mask": (
                self.batch_size,
                1,
                query_length,
                swa_length,
            ),
            "csa_index_validity": (
                self.batch_size,
                query_length,
                self.spec.csa_capacity,
            ),
            "csa_attention_mask": (
                self.batch_size,
                1,
                query_length,
                self.spec.index_topk + swa_length,
            ),
            "hca_attention_mask": (
                self.batch_size,
                1,
                query_length,
                self.spec.hca_capacity + swa_length,
            ),
        }

    def graph_cache_input_names_by_layer(self) -> tuple[tuple[str, ...], ...]:
        return tuple(
            tuple(f"layer_{layer}_{field}_input" for field in _GRAPH_CACHE_FIELDS[layer_type])
            for layer, layer_type in enumerate(self.layer_types)
        )


__all__ = [
    "CSA",
    "HCA",
    "SLIDING",
    "CSACacheInputs",
    "CSAStateOutput",
    "DeepSeekV4CacheABI",
    "FlatCacheInputs",
    "HCACacheInputs",
    "HCAStateOutput",
    "LayerCacheInputs",
    "SWACacheInputs",
    "default_layer_types",
    "flatten_cache_inputs",
]
