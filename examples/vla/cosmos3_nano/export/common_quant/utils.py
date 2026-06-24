# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0

"""Shared utilities used by common_quant export modules."""

from __future__ import annotations

import torch

__all__ = ["tensor_summary", "torch_dtype"]


def torch_dtype(name: str) -> torch.dtype:
    """Convert a dtype name to ``torch.dtype``."""
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float64":
        return torch.float64
    raise KeyError("Unsupported dtype: " + name)


def tensor_summary(tensor: torch.Tensor) -> dict[str, object]:
    """Return lightweight tensor metadata for debug and logs."""
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "min": float(tensor.min().item()) if tensor.numel() else 0.0,
        "max": float(tensor.max().item()) if tensor.numel() else 0.0,
        "mean": float(tensor.float().mean().item()) if tensor.numel() else 0.0,
    }
