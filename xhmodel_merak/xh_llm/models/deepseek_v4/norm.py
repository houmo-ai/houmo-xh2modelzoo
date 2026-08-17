"""Canonical DeepSeek-V4 RMSNorm construction for XH2 lowering."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from xhquant import nn as xhnn


def build_xh_rms_norm(
    hidden_size: int,
    eps: float,
    *,
    weight: Tensor | None = None,
) -> xhnn.RMSNorm:
    """Build one XH RMSNorm, including the checkpoint's unweighted variant.

    XH2 RMSNorm requires an explicit weight. DeepSeek-V4 also uses an
    unweighted RMSNorm in mHC and Q normalization, which is mathematically an
    all-ones weight. Store those synthetic ones as a non-persistent buffer so
    streamed export never expects a nonexistent checkpoint parameter.
    """
    hidden_size = int(hidden_size)
    if hidden_size <= 0:
        raise ValueError("hidden_size must be positive")
    module = xhnn.RMSNorm(hidden_size, eps=float(eps))
    if weight is not None:
        if tuple(weight.shape) != (hidden_size,):
            raise ValueError(f"RMSNorm weight must have shape ({hidden_size},), got {tuple(weight.shape)}")
        module.weight = weight if isinstance(weight, nn.Parameter) else nn.Parameter(weight)
        return module

    del module._parameters["weight"]
    module.register_buffer(
        "weight",
        torch.ones(hidden_size, dtype=torch.float16),
        persistent=False,
    )
    return module


def rms_norm_from_hf(
    source: nn.Module,
    *,
    hidden_size: int | None = None,
) -> xhnn.RMSNorm:
    """Convert weighted or unweighted HF V4 RMSNorm to one XH module."""
    weight = getattr(source, "weight", None)
    if hidden_size is None:
        if weight is None:
            raise ValueError("hidden_size is required for an unweighted RMSNorm")
        hidden_size = int(weight.shape[-1])
    eps = getattr(source, "variance_epsilon", getattr(source, "eps", None))
    if eps is None:
        raise ValueError(f"Cannot determine RMSNorm epsilon from {type(source).__name__}")
    return build_xh_rms_norm(hidden_size, float(eps), weight=weight)


__all__ = ["build_xh_rms_norm", "rms_norm_from_hf"]
