"""Separate physical K/V sliding-cache update for DeepSeek-V4."""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor, nn

from xhquant import nn as xhnn
from xhquant.core import CacheTensor

from ._trace import is_fx_proxy, one_element_int
from .swa_layout import aligned_swa_attention_length, aligned_swa_backing_length


class StaticSWAOutput(NamedTuple):
    physical_k: Tensor
    physical_v: Tensor


class StaticSWAUpdate(nn.Module):
    """Write one logical latent into distinct persistent K/V cache tensors.

    The graph ABI keeps the singleton KV-head dimension, ``[B,1,C,D]``.
    Passing that object directly to ``xh.LLMCache`` is intentional: inserting
    an ``unsqueeze`` between a ``HybridCacheTensor`` graph input and the cache
    operator erases the tensor subclass and therefore its rolling
    ``cache_valid_len`` state.
    """

    def __init__(
        self,
        *,
        input_sequence_length: int,
        window_size: int = 128,
        alignment: int = 16,
        backing_length: int | None = None,
    ) -> None:
        super().__init__()
        self.input_sequence_length = int(input_sequence_length)
        self.window_size = int(window_size)
        self.alignment = int(alignment)
        if min(self.input_sequence_length, self.window_size, self.alignment) <= 0:
            raise ValueError("SWA dimensions must be positive")
        self.attention_length = aligned_swa_attention_length(
            self.input_sequence_length,
            self.window_size,
            alignment=self.alignment,
        )
        required_backing_length = aligned_swa_backing_length(
            self.input_sequence_length,
            self.window_size,
            alignment=self.alignment,
        )
        self.backing_length = required_backing_length if backing_length is None else int(backing_length)
        if self.backing_length < required_backing_length:
            raise ValueError(
                "SWA backing_length must cover window + static input: "
                f"backing={self.backing_length}, required={required_backing_length}"
            )
        self.k_cache = xhnn.LLMCache(
            axis=2,
            attention_max_length=self.window_size,
            inplace=True,
        )
        self.v_cache = xhnn.LLMCache(
            axis=2,
            attention_max_length=self.window_size,
            inplace=True,
        )

    def _update(
        self,
        cache_op: nn.Module,
        latent: Tensor,
        past_length: Tensor,
        current_length: Tensor,
        past_cache: Tensor,
    ) -> Tensor:
        if not is_fx_proxy(past_cache):
            if past_cache.ndim != 4 or past_cache.shape[1] != 1:
                raise ValueError("past SWA cache must be [B,1,C,D]")
            if past_cache.shape[2] != self.backing_length:
                raise ValueError(f"past SWA cache must use the {self.backing_length}-row persistent backing")
        states = latent.unsqueeze(1)
        past = one_element_int(past_length, name="past_length")
        current = one_element_int(current_length, name="current_length")
        if is_fx_proxy(past_cache) or isinstance(past_cache, CacheTensor) or torch.compiler.is_compiling():
            updated = cache_op(states, past, current, past_cache)
        else:
            # Plain tensors remain useful for eager numerical unit tests; the
            # deployed graph always takes the module path above.
            updated = torch.ops.xh.LLMCache.default(
                states,
                past,
                current,
                past_cache,
                2,
                self.window_size,
                False,
            )
        return updated.squeeze(1)

    def forward(
        self,
        latent_kv: Tensor,
        past_length: Tensor,
        current_length: Tensor,
        past_k_cache: Tensor,
        past_v_cache: Tensor,
    ) -> StaticSWAOutput:
        if not is_fx_proxy(latent_kv):
            expected = (
                latent_kv.shape[0],
                self.input_sequence_length,
                latent_kv.shape[-1],
            )
            if tuple(latent_kv.shape) != expected:
                raise ValueError(f"latent_kv must have shape {expected}")
        physical_k = self._update(
            self.k_cache,
            latent_kv,
            past_length,
            current_length,
            past_k_cache,
        )
        physical_v = self._update(
            self.v_cache,
            latent_kv,
            past_length,
            current_length,
            past_v_cache,
        )
        return StaticSWAOutput(
            physical_k,
            physical_v,
        )


__all__ = ["StaticSWAOutput", "StaticSWAUpdate"]
