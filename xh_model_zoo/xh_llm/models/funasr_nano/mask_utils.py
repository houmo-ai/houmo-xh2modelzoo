"""Mask helpers for FunASR-Nano warp/export."""

from __future__ import annotations

import torch


_MASK_HALF = -32752.0


def sequence_mask(lengths: torch.Tensor, maxlen: int) -> torch.Tensor:
    row = torch.arange(0, maxlen, 1, device=lengths.device)
    return (row < lengths.unsqueeze(-1)).to(torch.float32)[:, None, :]


def attention_additive_mask(valid_mask: torch.Tensor) -> torch.Tensor:
    """Build additive mask outside attention forward.

    ``valid_mask`` is [B, 1, T] with 1 for valid positions. The returned mask is
    [B, 1, 1, T] and broadcasts to attention scores [B, H, Q, T]. xhquant
    MaskedAdd applies the mask-add pattern twice, so half of fp16 min is used.
    """
    invalid = valid_mask[:, None, :, :].eq(0)
    return torch.zeros_like(invalid, dtype=torch.float32).masked_fill(invalid, _MASK_HALF)


def downsample_lengths(lengths: torch.Tensor, rate: int) -> torch.Tensor:
    return ((lengths - 1) // int(rate) + 1).to(torch.int32)


def downsample_time(time_length: int, rate: int) -> int:
    return (int(time_length) - 1) // int(rate) + 1
