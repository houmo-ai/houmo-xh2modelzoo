from __future__ import annotations

import torch

from xhquant.core import HybridCacheTensor

from ...kv_cache_mixin import KVCacheMixin
from ...types import KVCacheConfig


def build_laguna_layer_kv_shapes(
    *,
    layer_types: list[str],
    batch_size: int,
    num_key_value_heads: int,
    context_max_length: int,
    sliding_window: int,
    input_sequence_length: int,
    head_dim: int,
) -> list[list[int]]:
    sliding_cache_length = ((sliding_window + input_sequence_length + 15) // 16) * 16
    shapes = []
    for layer_index, layer_type in enumerate(layer_types):
        if layer_type == "sliding_attention":
            cache_length = sliding_cache_length
        elif layer_type == "full_attention":
            cache_length = context_max_length
        else:
            raise ValueError(f"Unsupported Laguna layer type {layer_type!r} at layer {layer_index}.")
        shapes.append([batch_size, num_key_value_heads, cache_length, head_dim])
    return shapes


class LagunaKVCacheMixin(KVCacheMixin):
    def __init__(self, kv_cache_config: KVCacheConfig):
        super().__init__(kv_cache_config)
        self.layer_kv_shapes: list[list[int]] = []

    def set_layer_kv_shapes(self, layer_kv_shapes: list[list[int]]) -> None:
        self.layer_kv_shapes = layer_kv_shapes
        self.kvcache_config.num_layers = len(layer_kv_shapes)
        if layer_kv_shapes:
            self.kvcache_config.kv_cache_shape = layer_kv_shapes[0]

    def prepare_kv_cache(self, dtype=torch.float16) -> None:
        if self._cache_initialized or self.enable_page_attention or not self.use_cache:
            return
        self.past_key_caches.clear()
        self.past_value_caches.clear()
        for shape in self.layer_kv_shapes:
            self.past_key_caches.append(self.CACHCE_TENSOR_TYPE(torch.zeros(shape, dtype=dtype)))
            self.past_value_caches.append(self.CACHCE_TENSOR_TYPE(torch.zeros(shape, dtype=dtype)))
        self._cache_initialized = True


class LagunaKVCacheMixinHMONNX(LagunaKVCacheMixin):
    def prepare_kv_cache(self, dtype=torch.float16) -> None:
        if self._cache_initialized or self.enable_page_attention or not self.use_cache:
            return
        self.past_key_caches.clear()
        self.past_value_caches.clear()
        full_cache_length = max(int(shape[2]) for shape in self.layer_kv_shapes)
        for shape in self.layer_kv_shapes:
            cache_type = HybridCacheTensor if int(shape[2]) < full_cache_length else self.CACHCE_TENSOR_TYPE
            self.past_key_caches.append(cache_type(torch.zeros(shape, dtype=dtype)))
            self.past_value_caches.append(cache_type(torch.zeros(shape, dtype=dtype)))
        self._cache_initialized = True
