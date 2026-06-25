# Copyright 2025 HOUMO AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# SPDX-License-Identifier: Apache-2.0

"""Shared HMONNX utilities for the streaming pipeline.

Functions here are lightweight wrappers around ``_hmonnx_pipeline`` helpers
and HMONNX session management, kept separate to avoid circular imports and
to provide a clean interface for the streaming stages.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import torch

# ---------------------------------------------------------------------------
# HMONNX session helpers
# ---------------------------------------------------------------------------

try:
    from xhquant.xhonnxruntime.hmonnx_inference import HMONNXInference
except ImportError:  # pragma: no cover – allow import in test mocks
    HMONNXInference = None  # type: ignore[assignment,misc]


def create_hmonnx_session(onnx_path: Path, logger: Optional[logging.Logger] = None) -> Any:
    """Create an ``HMONNXInference`` session from an ONNX file path.

    Applies the Split→Slice auto-fix from ``_hmonnx_pipeline`` when available.
    """
    try:
        from _hmonnx_pipeline import _create_hmonnx_session

        return _create_hmonnx_session(onnx_path)
    except ImportError:
        pass

    if HMONNXInference is None:
        raise ImportError("HMONNXInference is not available – install xhquant first")
    session = HMONNXInference(str(onnx_path))
    if logger is not None:
        logger.info(f"Created HMONNX session from {onnx_path}")
    return session


def resolve_meta_path(meta: dict, field_name: str) -> Path:
    """Resolve a relative ONNX path stored in a meta dict."""
    root_dir = Path(meta["_root_dir"])
    return root_dir / meta[field_name]


# ---------------------------------------------------------------------------
# KV cache management
# ---------------------------------------------------------------------------


def make_kv_cache_state(kv_cache_info: dict) -> dict:
    """Create empty KV cache state from meta info."""
    kv_shape = kv_cache_info["shape"]
    num_layers = kv_cache_info["num_decoder_layers"]
    return {
        "past_seq_length": 0,
        "past_key_caches": [torch.zeros(kv_shape, dtype=torch.float16) for _ in range(num_layers)],
        "past_value_caches": [torch.zeros(kv_shape, dtype=torch.float16) for _ in range(num_layers)],
    }


def reset_kv_cache(kv_cache_info: dict) -> dict:
    """Return a fresh KV cache state (same as ``make_kv_cache_state``)."""
    return make_kv_cache_state(kv_cache_info)


# ---------------------------------------------------------------------------
# Output extraction helpers
# ---------------------------------------------------------------------------


def ensure_tensor(value: Any, device: torch.device = torch.device("cpu"), dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Convert any value to a tensor on the given device/dtype."""
    if isinstance(value, torch.Tensor):
        return value.detach().to(device=device, dtype=dtype)
    if hasattr(value, "last_hidden_state"):
        return ensure_tensor(value.last_hidden_state, device, dtype)
    if hasattr(value, "hidden_states") and isinstance(value.hidden_states, torch.Tensor):
        return ensure_tensor(value.hidden_states, device, dtype)
    return torch.as_tensor(value, device=device, dtype=dtype)


def extract_primary_output(output: Any) -> torch.Tensor:
    """Get the first element from an HMONNX output tuple/list."""
    if isinstance(output, (list, tuple)):
        output = output[0]
    if hasattr(output, "last_hidden_state"):
        output = output.last_hidden_state
    return output


def extract_logits_and_hidden_states(
    output: Any,
    device: torch.device,
    actual_seq_len: int,
    hidden_dtype: torch.dtype = torch.float16,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Extract (logits, hidden_states) from an HMONNX 2-output session."""
    outputs = output if isinstance(output, (list, tuple)) else [output]
    logits = ensure_tensor(outputs[0], torch.device("cpu"), torch.float32)
    if logits.ndim == 2:
        logits = logits.unsqueeze(1)
    logits = logits[:, :actual_seq_len, :]

    hidden_states = None
    if len(outputs) >= 2:
        hidden_states = ensure_tensor(outputs[1], torch.device("cpu"), hidden_dtype)
        if hidden_states.ndim == 2:
            hidden_states = hidden_states.unsqueeze(1)
        hidden_states = hidden_states[:, :actual_seq_len, :]
    return logits, hidden_states


# ---------------------------------------------------------------------------
# Padding helpers
# ---------------------------------------------------------------------------


def pad_to_length(
    tensor: torch.Tensor, target_len: int, dim: int = 1, fill_value: float = 0.0
) -> torch.Tensor:
    """Pad ``tensor`` along ``dim`` to ``target_len``."""
    current_len = int(tensor.shape[dim])
    if current_len >= target_len:
        slices = [slice(None)] * tensor.ndim
        slices[dim] = slice(0, target_len)
        return tensor[tuple(slices)]
    pad_shape = list(tensor.shape)
    pad_shape[dim] = target_len - current_len
    pad = torch.full(pad_shape, fill_value, dtype=tensor.dtype, device=tensor.device)
    return torch.cat([tensor, pad], dim=dim)


# ---------------------------------------------------------------------------
# Length-only past_key_values (minimal cache object)
# ---------------------------------------------------------------------------


class LengthOnlyPastKeyValues:
    """Minimal past_key_values that only exposes ``get_seq_length()``."""

    def __init__(self, seq_length: int = 0):
        self._seq_length = seq_length

    def get_seq_length(self, *args, **kwargs) -> int:
        return self._seq_length
