"""Small helpers shared by the fixed DeepSeek-V4 TorchFX graph."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor
from torch.fx import Proxy


def is_fx_proxy(value: Any) -> bool:
    """Return whether ``value`` is a symbolic tensor in Merak's FX tracer."""

    return isinstance(value, Proxy)


def one_element_int(value: Tensor, *, name: str) -> Tensor:
    """Validate the Host scalar-control ABI without mutating the graph.

    DeepSeek-V4 exports every scalar position/count as an integer tensor with
    shape ``[1]``. XH DynamicSlice, Gather and KVcache all accept INT32
    directly, so reshaping and casting an already-correct Host input only adds
    redundant graph nodes.
    """

    if not is_fx_proxy(value):
        if tuple(value.shape) != (1,):
            raise ValueError(f"{name} must have shape [1]")
        if value.dtype not in (torch.int32, torch.int64):
            raise TypeError(f"{name} must be INT32 or INT64")
    return value


__all__ = ["is_fx_proxy", "one_element_int"]
