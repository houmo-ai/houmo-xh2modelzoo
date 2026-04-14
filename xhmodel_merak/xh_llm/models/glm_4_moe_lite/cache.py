import torch

from xhquant.core import CacheTensor

from ...kv_cache_mixin import KVCacheMixin
from ...types import KVCacheConfig


class Glm4MoeLiteKVCacheConfig(KVCacheConfig):
    def __init__(
        self,
        *,
        num_layers: int = -1,
        kv_cache_shape: list[int] = None,
        key_cache_shape: list[int] = None,
        value_cache_shape: list[int] = None,
        cache_axis: int = 2,
        batch_size: int = 1,
        cache_dtype: str = "float16",
    ):
        super().__init__(
            num_layers=num_layers,
            kv_cache_shape=kv_cache_shape,
            cache_axis=cache_axis,
            batch_size=batch_size,
            cache_dtype=cache_dtype,
        )
        self.key_cache_shape = key_cache_shape if key_cache_shape is not None else kv_cache_shape
        self.value_cache_shape = value_cache_shape if value_cache_shape is not None else kv_cache_shape


class Glm4MoeLiteKVCacheMixin(KVCacheMixin):
    kvcache_config: Glm4MoeLiteKVCacheConfig

    def __init__(self, kv_cache_config: Glm4MoeLiteKVCacheConfig) -> None:
        super().__init__(kv_cache_config)

    def prepare_kv_cache(self):
        if not self.use_cache:
            return
        if self.kvcache_config.num_layers <= 0:
            return

        cache_dtype = self.kvcache_config.cache_torch_dtype
        key_cache_shape = self.kvcache_config.key_cache_shape
        value_cache_shape = self.kvcache_config.value_cache_shape
        assert key_cache_shape is not None, "GLM key cache shape is not configured."
        assert value_cache_shape is not None, "GLM value cache shape is not configured."

        for _ in range(self.kvcache_config.num_layers):
            self.past_key_caches.append(CacheTensor(torch.zeros(key_cache_shape, dtype=cache_dtype)))
            self.past_value_caches.append(CacheTensor(torch.zeros(value_cache_shape, dtype=cache_dtype)))
