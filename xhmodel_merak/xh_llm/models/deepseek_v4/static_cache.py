"""Static cache contracts for DeepSeek-V4 Flash CSA/HCA attention.

The official C4 compressor keeps two four-token groups.  For every completed
group it pools eight candidates: the first branch of the preceding group and
the second branch of the current group.  This module expresses the same state
transition with fixed-shape tensors and ``DynamicSlice`` starts, so prefill and
decode can be exported as static graphs.

DeepSeek-V4 uses one latent vector as both K and V. The main CSA/HCA cache stays
FP16 because query-dependent TopK prevents pre-packing it as a V-SEFP cache, so
one physical latent cache is shared by QK and PV. The C4 Indexer remains key-only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch
from torch import Tensor, nn

from xhquant import nn as xhnn
from xhquant.core import CacheTensor

from ._trace import is_fx_proxy, one_element_int
from .swa_layout import aligned_swa_backing_length


def _ceil_div(value: int, divisor: int) -> int:
    return (int(value) + int(divisor) - 1) // int(divisor)


@dataclass(frozen=True)
class DeepSeekV4StaticCacheSpec:
    """Fixed-capacity cache ABI for one request and one attention layer."""

    max_context_length: int = 256 * 1024
    prefill_chunk_length: int = 256
    sliding_window: int = 128
    latent_head_dim: int = 512
    index_head_dim: int = 128
    csa_ratio: int = 4
    hca_ratio: int = 128
    index_topk: int = 512

    def __post_init__(self) -> None:
        for name in (
            "max_context_length",
            "prefill_chunk_length",
            "sliding_window",
            "latent_head_dim",
            "index_head_dim",
            "csa_ratio",
            "hca_ratio",
            "index_topk",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.max_context_length < self.sliding_window:
            raise ValueError("max_context_length must cover the sliding window")

    @property
    def swa_logical_backing_length(self) -> int:
        """Rows reserved for one rolling SWA K/V backing allocation.

        The backing must hold one complete prefill input in addition to the
        slice window.  This is deliberately distinct from the stage attention
        view emitted by ``LLMCache``, whose logical width is
        ``window + stage_input - 1``.
        """

        return self.sliding_window + self.prefill_chunk_length

    @property
    def swa_physical_length(self) -> int:
        """Aligned ``window + prefill_input`` backing shared by both graphs."""

        return aligned_swa_backing_length(
            self.prefill_chunk_length,
            self.sliding_window,
        )

    @property
    def csa_capacity(self) -> int:
        return self.max_context_length // self.csa_ratio

    @property
    def hca_capacity(self) -> int:
        return self.max_context_length // self.hca_ratio

    def tensor_shapes(self, batch_size: int = 1) -> dict[str, tuple[int, ...]]:
        """Return physical tensor shapes for the unified latent-cache ABI."""

        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        d = self.latent_head_dim
        index_d = self.index_head_dim
        return {
            "swa_kv": (batch_size, 1, self.swa_physical_length, d),
            "csa_main": (batch_size, 1, self.csa_capacity, d),
            "csa_index_k": (batch_size, 1, self.csa_capacity, index_d),
            "hca_main": (batch_size, 1, self.hca_capacity, d),
            "csa_main_kv_state": (
                batch_size,
                2 * self.csa_ratio,
                2 * d,
            ),
            "csa_main_score_state": (
                batch_size,
                2 * self.csa_ratio,
                2 * d,
            ),
            "csa_index_kv_state": (
                batch_size,
                2 * self.csa_ratio,
                2 * index_d,
            ),
            "csa_index_score_state": (
                batch_size,
                2 * self.csa_ratio,
                2 * index_d,
            ),
            "hca_main_kv_state": (batch_size, self.hca_ratio, d),
            "hca_main_score_state": (batch_size, self.hca_ratio, d),
        }


class CompressorStepOutput(NamedTuple):
    pooled: Tensor
    pooled_valid: Tensor
    next_kv_state: Tensor
    next_score_state: Tensor
    new_count: Tensor


class _CompressorStepBase(nn.Module):
    def __init__(
        self,
        *,
        input_sequence_length: int,
        ratio: int,
        projection_dim: int,
        state_rows: int,
        candidate_rows: int,
        max_outputs: int,
        invalid_score: float,
    ) -> None:
        super().__init__()
        self.input_sequence_length = int(input_sequence_length)
        self.ratio = int(ratio)
        self.projection_dim = int(projection_dim)
        self.state_rows = int(state_rows)
        self.max_outputs = int(max_outputs)
        self.invalid_score = float(invalid_score)
        if (
            min(
                self.input_sequence_length,
                self.ratio,
                self.projection_dim,
                self.state_rows,
                self.max_outputs,
            )
            <= 0
        ):
            raise ValueError("compressor dimensions must be positive")
        self.candidate_slice = xhnn.DynamicSlice([candidate_rows], [1], [1])
        self.tail_slice = xhnn.DynamicSlice([state_rows], [1], [1])
        self.gather_ape = xhnn.Gather(axis=0)
        right_pad = self.ratio - 1
        self.register_buffer(
            "kv_right_pad",
            torch.zeros(1, right_pad, self.projection_dim, dtype=torch.float16),
            persistent=False,
        )
        self.register_buffer(
            "score_right_pad",
            torch.full(
                (1, right_pad, self.projection_dim),
                self.invalid_score,
                dtype=torch.float16,
            ),
            persistent=False,
        )

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float16,
    ) -> tuple[Tensor, Tensor]:
        shape = (int(batch_size), self.state_rows, self.projection_dim)
        kv_state = torch.zeros(shape, dtype=dtype, device=device)
        score_state = torch.full(
            shape,
            self.invalid_score,
            dtype=dtype,
            device=device,
        )
        return kv_state, score_state

    def _validate_inputs(
        self,
        current_kv: Tensor,
        current_score: Tensor,
        kv_state: Tensor,
        score_state: Tensor,
        ape: Tensor,
    ) -> None:
        if is_fx_proxy(current_kv):
            return
        expected_current = (
            current_kv.shape[0],
            self.input_sequence_length,
            self.projection_dim,
        )
        expected_state = (
            current_kv.shape[0],
            self.state_rows,
            self.projection_dim,
        )
        if tuple(current_kv.shape) != expected_current:
            raise ValueError(f"current_kv must have shape {expected_current}, got {tuple(current_kv.shape)}")
        if tuple(current_score.shape) != expected_current:
            raise ValueError("current_score must match current_kv")
        if tuple(kv_state.shape) != expected_state:
            raise ValueError(f"kv_state must have shape {expected_state}, got {tuple(kv_state.shape)}")
        if tuple(score_state.shape) != expected_state:
            raise ValueError("score_state must match kv_state")
        if tuple(ape.shape) != (self.ratio, self.projection_dim):
            raise ValueError(f"ape must have shape ({self.ratio}, {self.projection_dim}), got {tuple(ape.shape)}")

    def _prepare_source(
        self,
        current_kv: Tensor,
        current_score: Tensor,
        kv_state: Tensor,
        score_state: Tensor,
        offset: Tensor,
        phase_indices: Tensor,
        current_length: Tensor,
        compressor_validity: Tensor,
        compressor_new_count: Tensor,
        ape: Tensor,
    ) -> tuple[
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
    ]:
        self._validate_inputs(current_kv, current_score, kv_state, score_state, ape)
        offset = one_element_int(offset, name="offset")
        current_length = one_element_int(current_length, name="current_length")
        compressor_new_count = one_element_int(
            compressor_new_count,
            name="compressor_new_count",
        )
        if not is_fx_proxy(compressor_validity):
            if tuple(compressor_validity.shape) != (self.max_outputs,):
                raise ValueError(
                    f"compressor_validity must have shape ({self.max_outputs},), got {tuple(compressor_validity.shape)}"
                )
            if not compressor_validity.dtype.is_floating_point:
                raise TypeError("compressor_validity must be a floating-point 0/1 Host tensor")
        if not is_fx_proxy(phase_indices):
            if tuple(phase_indices.shape) != (self.input_sequence_length,):
                raise ValueError(
                    f"phase_indices must have shape ({self.input_sequence_length},), got {tuple(phase_indices.shape)}"
                )
            if phase_indices.dtype not in {torch.int32, torch.int64}:
                raise TypeError("phase_indices must be an integer Host tensor")

        current_score = current_score + self.gather_ape(
            ape,
            phase_indices,
        ).unsqueeze(0)

        raw_kv = torch.cat((kv_state, current_kv), dim=1)
        raw_score = torch.cat((score_state, current_score), dim=1)
        next_kv_state = self.tail_slice(raw_kv, current_length)
        next_score_state = self.tail_slice(raw_score, current_length)

        # The DeepSeek-V4 export contract fixes batch size to one, so these
        # registered buffers already have the exact runtime batch shape.
        candidate_kv_source = torch.cat((raw_kv, self.kv_right_pad), dim=1)
        candidate_score_source = torch.cat((raw_score, self.score_right_pad), dim=1)

        return (
            candidate_kv_source,
            candidate_score_source,
            next_kv_state,
            next_score_state,
            compressor_new_count,
            compressor_validity,
            offset,
        )


class OverlappingCompressorStep(_CompressorStepBase):
    """One fixed-shape C4 compressor transition.

    Projection tensors have width ``2 * head_dim``.  After dynamic alignment,
    two adjacent four-token groups are sliced and concatenated into
    ``[B, max_outputs, 8, head_dim]``.  Softmax is evaluated along that
    eight-candidate dimension independently for every output feature.
    """

    def __init__(
        self,
        *,
        input_sequence_length: int,
        head_dim: int,
        ratio: int = 4,
        invalid_score: float = -65504.0,
    ) -> None:
        if int(ratio) != 4:
            raise ValueError("DeepSeek-V4 overlap compression is defined for ratio=4")
        self.head_dim = int(head_dim)
        max_outputs = (int(input_sequence_length) + int(ratio) - 1) // int(ratio)
        candidate_groups = max_outputs + 1
        super().__init__(
            input_sequence_length=input_sequence_length,
            ratio=ratio,
            projection_dim=2 * self.head_dim,
            state_rows=2 * int(ratio),
            candidate_rows=candidate_groups * int(ratio),
            max_outputs=max_outputs,
            invalid_score=invalid_score,
        )
        self.candidate_groups = candidate_groups

    def forward(
        self,
        current_kv: Tensor,
        current_score: Tensor,
        kv_state: Tensor,
        score_state: Tensor,
        offset: Tensor,
        phase_indices: Tensor,
        current_length: Tensor,
        compressor_validity: Tensor,
        compressor_new_count: Tensor,
        ape: Tensor,
    ) -> CompressorStepOutput:
        (
            candidate_kv_source,
            candidate_score_source,
            next_kv_state,
            next_score_state,
            new_count,
            valid,
            offset,
        ) = self._prepare_source(
            current_kv,
            current_score,
            kv_state,
            score_state,
            offset,
            phase_indices,
            current_length,
            compressor_validity,
            compressor_new_count,
            ape,
        )
        aligned_kv = self.candidate_slice(candidate_kv_source, offset).reshape(
            current_kv.shape[0],
            self.candidate_groups,
            self.ratio,
            2 * self.head_dim,
        )
        aligned_score = self.candidate_slice(
            candidate_score_source,
            offset,
        ).reshape(
            current_kv.shape[0],
            self.candidate_groups,
            self.ratio,
            2 * self.head_dim,
        )

        candidate_kv = torch.cat(
            (
                aligned_kv[:, :-1, :, : self.head_dim],
                aligned_kv[:, 1:, :, self.head_dim :],
            ),
            dim=2,
        )
        candidate_score = torch.cat(
            (
                aligned_score[:, :-1, :, : self.head_dim],
                aligned_score[:, 1:, :, self.head_dim :],
            ),
            dim=2,
        )
        weights = torch.softmax(candidate_score, dim=2)
        pooled = torch.sum(candidate_kv * weights, dim=2)
        pooled = pooled * valid.reshape(1, -1, 1)
        return CompressorStepOutput(
            pooled,
            valid,
            next_kv_state,
            next_score_state,
            new_count,
        )


class NonOverlappingCompressorStep(_CompressorStepBase):
    """One fixed-shape C128 compressor transition for HCA."""

    def __init__(
        self,
        *,
        input_sequence_length: int,
        head_dim: int,
        ratio: int = 128,
        invalid_score: float = -65504.0,
    ) -> None:
        self.head_dim = int(head_dim)
        max_outputs = _ceil_div(input_sequence_length, ratio)
        super().__init__(
            input_sequence_length=input_sequence_length,
            ratio=ratio,
            projection_dim=self.head_dim,
            state_rows=int(ratio),
            candidate_rows=max_outputs * int(ratio),
            max_outputs=max_outputs,
            invalid_score=invalid_score,
        )

    def forward(
        self,
        current_kv: Tensor,
        current_score: Tensor,
        kv_state: Tensor,
        score_state: Tensor,
        offset: Tensor,
        phase_indices: Tensor,
        current_length: Tensor,
        compressor_validity: Tensor,
        compressor_new_count: Tensor,
        ape: Tensor,
    ) -> CompressorStepOutput:
        (
            candidate_kv_source,
            candidate_score_source,
            next_kv_state,
            next_score_state,
            new_count,
            valid,
            offset,
        ) = self._prepare_source(
            current_kv,
            current_score,
            kv_state,
            score_state,
            offset,
            phase_indices,
            current_length,
            compressor_validity,
            compressor_new_count,
            ape,
        )
        candidate_kv = self.candidate_slice(candidate_kv_source, offset).reshape(
            current_kv.shape[0],
            self.max_outputs,
            self.ratio,
            self.head_dim,
        )
        candidate_score = self.candidate_slice(
            candidate_score_source,
            offset,
        ).reshape(
            current_kv.shape[0],
            self.max_outputs,
            self.ratio,
            self.head_dim,
        )
        weights = torch.softmax(candidate_score, dim=2)
        pooled = torch.sum(candidate_kv * weights, dim=2)
        pooled = pooled * valid.reshape(1, -1, 1)
        return CompressorStepOutput(
            pooled,
            valid,
            next_kv_state,
            next_score_state,
            new_count,
        )


def update_fixed_cache(
    past_cache: Tensor,
    values: Tensor,
    write_start: Tensor,
    valid_count: Tensor,
) -> Tensor:
    """Traceable fixed-capacity prefix update without writing padded values."""

    if past_cache.ndim != 3 or values.ndim != 3:
        raise ValueError("past_cache and values must be [B, sequence, D]")
    if past_cache.shape[0] != values.shape[0] or past_cache.shape[2] != values.shape[2]:
        raise ValueError("cache/value batch and feature dimensions must match")
    write_start = one_element_int(write_start, name="write_start")
    valid_count = one_element_int(valid_count, name="valid_count")
    capacity = past_cache.shape[1]
    slots = write_start + torch.arange(values.shape[1], device=values.device)
    in_bounds = (slots >= 0) & (slots < capacity)
    # Index bounds are integer control flow.  XH Clip is an FP16-only kernel,
    # so redirect padded/out-of-capacity rows to the harmless zero slot with
    # Where instead of emitting an integer Clip.
    safe_slots = torch.where(in_bounds, slots, slots * 0)
    index = safe_slots.reshape(1, -1, 1).expand(
        values.shape[0],
        values.shape[1],
        values.shape[2],
    )
    old = torch.gather(past_cache, 1, index)
    valid = (torch.arange(values.shape[1], device=values.device) < valid_count) & in_bounds
    source = torch.where(valid.reshape(1, -1, 1), values, old)
    return torch.scatter(past_cache, 1, index, source)


class FixedCapacityCacheWriter(nn.Module):
    """Contiguous fixed-cache write lowered to the existing XH LLMCache op.

    The input values keep their fixed graph width, while valid_count controls
    how many leading rows are committed. Using LLMCache avoids materializing a
    full-capacity scatter/copy in the deployed graph.
    """

    def __init__(self) -> None:
        super().__init__()
        self.cache = xhnn.LLMCache(axis=2, attention_max_length=-1, inplace=True)

    def forward(
        self,
        past_cache: Tensor,
        values: Tensor,
        write_start: Tensor,
        valid_count: Tensor,
    ) -> Tensor:
        if not is_fx_proxy(past_cache):
            if past_cache.ndim != 4 or past_cache.shape[1] != 1 or values.ndim != 3:
                raise ValueError("past_cache and values must be [B,1,C,D] and [B,P,D]")
            if past_cache.shape[0] != values.shape[0] or past_cache.shape[3] != values.shape[2]:
                raise ValueError("cache/value batch and feature dimensions must match")
        states = values.unsqueeze(1)
        start = one_element_int(write_start, name="write_start")
        count = one_element_int(valid_count, name="valid_count")
        if is_fx_proxy(past_cache) or isinstance(past_cache, CacheTensor) or torch.compiler.is_compiling():
            updated = self.cache(states, start, count, past_cache)
        else:
            updated = torch.ops.xh.LLMCache.default(
                states,
                start,
                count,
                past_cache,
                2,
                -1,
                False,
            )
        return updated.squeeze(1)


class CSAIndexerTopK(nn.Module):
    """Direct full-cache CSA score followed by one global TopK."""

    def __init__(
        self,
        *,
        topk: int = 512,
        cache_capacity: int | None = None,
        score_scale: float | None = None,
        invalid_score: float = -65504.0,
    ) -> None:
        super().__init__()
        self.topk = int(topk)
        self.cache_capacity = None if cache_capacity is None else int(cache_capacity)
        self.score_scale = None if score_scale is None else float(score_scale)
        self.invalid_score = float(invalid_score)
        if self.topk <= 0:
            raise ValueError("topk must be positive")

    def forward(
        self,
        query: Tensor,
        index_weights: Tensor,
        key_cache: Tensor,
        validity: Tensor,
    ) -> tuple[Tensor, Tensor]:
        capacity = key_cache.shape[1] if self.cache_capacity is None else self.cache_capacity
        if not is_fx_proxy(query):
            if query.ndim != 4:
                raise ValueError("query must be [B, P, H, D]")
            if index_weights.shape != query.shape[:-1]:
                raise ValueError("index_weights must be [B, P, H]")
            if key_cache.ndim != 3 or key_cache.shape[0] != query.shape[0]:
                raise ValueError("key_cache must be [B, C, D]")
            if key_cache.shape[-1] != query.shape[-1]:
                raise ValueError("query and key_cache feature dimensions must match")
            expected_mask = (query.shape[0], query.shape[1], capacity)
            if tuple(validity.shape) != expected_mask:
                raise ValueError(f"validity must have shape {expected_mask}")
            if not validity.dtype.is_floating_point:
                raise TypeError("validity must be a floating-point 0/1 Host tensor")
            if capacity < self.topk:
                raise ValueError("key cache capacity must be at least topk")

        index_scale = self.score_scale
        if index_scale is None:
            index_scale = (query.shape[-1] * query.shape[-2]) ** -0.5

        score_per_head = torch.matmul(
            query,
            key_cache.transpose(1, 2).unsqueeze(1),
        )
        score = torch.matmul(
            index_weights.unsqueeze(-2),
            torch.relu(score_per_head),
        ).squeeze(-2)
        score = score * index_scale
        # Host validity is normalized to floating 0/1. Pure arithmetic keeps
        # the graph free of Where/comparison control flow; -65504 is the finite
        # FP16 deployment sentinel and cannot introduce Inf/NaN constants.
        block_validity = validity
        invalid_floor = torch.finfo(torch.float16).min
        score = score * block_validity + invalid_floor * (1.0 - block_validity)
        return torch.topk(score, self.topk, dim=-1, sorted=True)


class LatentCacheGather(nn.Module):
    """Gather selected rows from the shared FP16 latent cache."""

    def __init__(self, *, feature_dim: int) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        # The static ABI has batch=1. Gathering directly from the
        # flattened [capacity, feature] cache avoids materializing the old
        # [1, query, capacity, feature] expansion (16 GiB at
        # prefill=256, capacity=65536, feature=512).
        self.gather = xhnn.Gather(axis=0)

    def forward(
        self,
        latent_cache: Tensor,
        indices: Tensor,
    ) -> Tensor:
        if not is_fx_proxy(latent_cache):
            if latent_cache.ndim != 3 or indices.ndim != 3:
                raise ValueError("cache must be [B,C,D] and indices [B,P,K]")
            if latent_cache.shape[0] != indices.shape[0]:
                raise ValueError("cache and indices batch dimensions must match")
            if latent_cache.shape[0] != 1:
                raise ValueError("DeepSeek-V4 static LatentCacheGather requires batch=1")
            if latent_cache.shape[-1] != self.feature_dim:
                raise ValueError(f"cache feature dimension {latent_cache.shape[-1]} != {self.feature_dim}")
        return self.gather(latent_cache.reshape(-1, self.feature_dim), indices)


__all__ = [
    "CSAIndexerTopK",
    "CompressorStepOutput",
    "DeepSeekV4StaticCacheSpec",
    "FixedCapacityCacheWriter",
    "NonOverlappingCompressorStep",
    "OverlappingCompressorStep",
    "LatentCacheGather",
    "update_fixed_cache",
]
