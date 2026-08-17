"""Host-built attention masks for the static DeepSeek-V4 graph ABI.

Token positions and valid lengths are scheduler state, not model arithmetic.
The Host builds attention masks and compressor transitions once per invocation
and shares them across all layers of the same attention kind.  The exported
graph therefore never needs integer comparisons or selection operators to
reconstruct position-dependent state.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor

from .static_cache import DeepSeekV4StaticCacheSpec
from .swa_layout import aligned_swa_attention_length


FP16_MASK_MIN = -65504.0


class DeepSeekV4AttentionMasks(NamedTuple):
    """Shared Host inputs consumed by the fixed prefill/decode graphs."""

    swa_attention_mask: Tensor
    csa_index_validity: Tensor
    csa_attention_mask: Tensor
    hca_attention_mask: Tensor
    csa_compressor_validity: Tensor
    csa_compressor_new_count: Tensor
    csa_compressor_offset: Tensor
    csa_compressor_phase_indices: Tensor
    hca_compressor_validity: Tensor
    hca_compressor_new_count: Tensor
    hca_compressor_offset: Tensor
    hca_compressor_phase_indices: Tensor


def _additive_mask(valid: Tensor, *, dtype: torch.dtype) -> Tensor:
    zeros = torch.zeros((), device=valid.device, dtype=dtype)
    minimum = torch.full((), FP16_MASK_MIN, device=valid.device, dtype=dtype)
    return torch.where(valid, zeros, minimum)


def build_compressor_host_transition(
    *,
    input_sequence_length: int,
    past_length: int,
    current_length: int,
    ratio: int,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float16,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return ``(validity, new_count, offset, phase_indices)`` for one ratio.

    ``offset`` is the alignment of the *current* invocation.  It is derived
    from the absolute Host-owned ``past_length`` and therefore is neither
    persistent learned state nor a graph output. ``phase_indices`` assigns the
    learned absolute-position bias to every fixed-shape input row. Keeping this
    coordinate calculation on Host avoids device-side Sub/Mod arithmetic and,
    crucially, does not depend on a backend's binary-Sub operand convention.
    """

    input_sequence_length = int(input_sequence_length)
    past_length = int(past_length)
    current_length = int(current_length)
    ratio = int(ratio)
    if min(input_sequence_length, current_length, ratio) <= 0:
        raise ValueError("compressor transition dimensions must be positive")
    if current_length > input_sequence_length or past_length < 0:
        raise ValueError("compressor transition lengths are invalid")
    if not dtype.is_floating_point:
        raise TypeError("compressor validity must use a floating-point dtype")

    max_outputs = (input_sequence_length + ratio - 1) // ratio
    pending = past_length % ratio
    new_count = (pending + current_length) // ratio
    if new_count > max_outputs:
        raise RuntimeError(f"compressor transition produced {new_count} rows, capacity is {max_outputs}")
    validity = (torch.arange(max_outputs, device=device, dtype=torch.int64) < new_count).to(dtype=dtype)
    count = torch.tensor([new_count], device=device, dtype=torch.int32)
    offset = torch.tensor(
        [ratio - pending],
        device=device,
        dtype=torch.int32,
    )
    phase_indices = (
        pending
        + torch.arange(
            input_sequence_length,
            device=device,
            dtype=torch.int32,
        )
    ) % ratio
    return validity, count, offset, phase_indices


def build_deepseek_v4_attention_masks(
    spec: DeepSeekV4StaticCacheSpec,
    *,
    input_sequence_length: int,
    past_length: int,
    current_length: int,
    batch_size: int = 1,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float16,
) -> DeepSeekV4AttentionMasks:
    """Build all position-dependent inputs outside the exported model graph.

    Additive attention masks use ``0`` for a visible token and ``-65504`` for
    a hidden token.  ``csa_index_validity`` uses FP16 ``1/0`` because the
    tiled TopK path consumes it with pure multiply/add arithmetic.
    """

    input_sequence_length = int(input_sequence_length)
    past_length = int(past_length)
    current_length = int(current_length)
    batch_size = int(batch_size)
    if input_sequence_length not in {1, spec.prefill_chunk_length}:
        raise ValueError("input_sequence_length must be the wrapped prefill chunk length or decode length 1")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if current_length <= 0 or current_length > input_sequence_length:
        raise ValueError("current_length is outside the static graph input range")
    if past_length < 0 or past_length + current_length > spec.max_context_length:
        raise ValueError("request exceeds the wrapped maximum context length")
    if not dtype.is_floating_point:
        raise TypeError("attention masks must use a floating-point dtype")

    query = torch.arange(input_sequence_length, device=device, dtype=torch.int64)
    query_valid = query < current_length

    # LLMCache retains at most window-1 rows before the current fixed-shape
    # input.  Its returned coordinate system is therefore independent of the
    # absolute request position once the window is full.
    swa_length = aligned_swa_attention_length(
        input_sequence_length,
        spec.sliding_window,
    )
    retained_past = min(past_length, spec.sliding_window - 1)
    swa_position = torch.arange(swa_length, device=device, dtype=torch.int64)
    swa_left = torch.clamp(
        retained_past + query - spec.sliding_window + 1,
        min=0,
    )
    swa_right = torch.clamp(
        retained_past + query + 1,
        max=swa_length,
    )
    swa_valid = (
        query_valid.unsqueeze(-1)
        & (swa_position.unsqueeze(0) >= swa_left.unsqueeze(-1))
        & (swa_position.unsqueeze(0) < swa_right.unsqueeze(-1))
    )

    absolute_query_end = past_length + query + 1
    csa_count = torch.div(absolute_query_end, spec.csa_ratio, rounding_mode="floor")
    csa_count = torch.clamp(csa_count, max=spec.csa_capacity)
    csa_position = torch.arange(spec.csa_capacity, device=device, dtype=torch.int64)
    csa_valid = query_valid.unsqueeze(-1) & (csa_position.unsqueeze(0) < csa_count.unsqueeze(-1))

    topk_position = torch.arange(spec.index_topk, device=device, dtype=torch.int64)
    selected_count = torch.clamp(csa_count, max=spec.index_topk)
    selected_valid = query_valid.unsqueeze(-1) & (topk_position.unsqueeze(0) < selected_count.unsqueeze(-1))
    csa_attention_valid = torch.cat((selected_valid, swa_valid), dim=-1)

    hca_count = torch.div(absolute_query_end, spec.hca_ratio, rounding_mode="floor")
    hca_count = torch.clamp(hca_count, max=spec.hca_capacity)
    hca_position = torch.arange(spec.hca_capacity, device=device, dtype=torch.int64)
    hca_valid = query_valid.unsqueeze(-1) & (hca_position.unsqueeze(0) < hca_count.unsqueeze(-1))
    hca_attention_valid = torch.cat((hca_valid, swa_valid), dim=-1)

    def batched(value: Tensor) -> Tensor:
        return value.unsqueeze(0).expand(batch_size, *value.shape).contiguous()

    csa_transition = build_compressor_host_transition(
        input_sequence_length=input_sequence_length,
        past_length=past_length,
        current_length=current_length,
        ratio=spec.csa_ratio,
        device=device,
        dtype=dtype,
    )
    hca_transition = build_compressor_host_transition(
        input_sequence_length=input_sequence_length,
        past_length=past_length,
        current_length=current_length,
        ratio=spec.hca_ratio,
        device=device,
        dtype=dtype,
    )

    return DeepSeekV4AttentionMasks(
        _additive_mask(batched(swa_valid).unsqueeze(1), dtype=dtype),
        batched(csa_valid).to(dtype=dtype),
        _additive_mask(batched(csa_attention_valid).unsqueeze(1), dtype=dtype),
        _additive_mask(batched(hca_attention_valid).unsqueeze(1), dtype=dtype),
        *csa_transition,
        *hca_transition,
    )


__all__ = [
    "DeepSeekV4AttentionMasks",
    "FP16_MASK_MIN",
    "aligned_swa_attention_length",
    "build_compressor_host_transition",
    "build_deepseek_v4_attention_masks",
]
