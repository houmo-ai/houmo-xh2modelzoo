from contextlib import contextmanager
from typing import Generator

import torch
from torch import Tensor

from xhquant.core import CacheTensor

from .types import CacheList, KVCacheConfig, KVCacheWithLinearConfig


class KVCacheMixin:
    """KV Cache Mixin 类，提供 KV Cache 相关的方法接口。"""

    CACHCE_TENSOR_TYPE = CacheTensor  # KV Cache使用的Tensor类型，默认为CacheTensor
    kvcache_config: KVCacheConfig | KVCacheWithLinearConfig

    def __init__(self, kv_cache_config: KVCacheConfig | KVCacheWithLinearConfig) -> None:
        super().__init__()
        self.kvcache_config = kv_cache_config
        self.past_key_caches: CacheList[Tensor | CacheTensor] = CacheList()
        self.past_value_caches: CacheList[Tensor | CacheTensor] = CacheList()
        self._device = None
        self._dtype = None
        self._cache_initialized = False

    @property
    def use_cache(self) -> bool:
        return self.kvcache_config.use_cache

    def prepare_kv_cache(self):
        if self._cache_initialized:
            return
        if not self.use_cache:
            return
        if self.kvcache_config.num_layers <= 0:
            return
        num_decoder_layers = self.kvcache_config.num_layers
        kv_cache_shape = self.kvcache_config.kv_cache_shape

        if isinstance(kv_cache_shape[0], list):
            k_cache_shape = kv_cache_shape[0]
            v_cache_shape = kv_cache_shape[1]
        else:
            k_cache_shape = kv_cache_shape
            v_cache_shape = kv_cache_shape
        for _i in range(num_decoder_layers):
            self.past_key_caches.append(self.CACHCE_TENSOR_TYPE(torch.zeros(k_cache_shape, dtype=torch.float16)))
            self.past_value_caches.append(self.CACHCE_TENSOR_TYPE(torch.zeros(v_cache_shape, dtype=torch.float16)))
        self._cache_initialized = True

    def clear_kv_cache(self):
        self.past_key_caches.clear()
        self.past_value_caches.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._cache_initialized = False

    def prepare_other_cache(self):
        pass

    def clear_other_cache(self):
        pass

    @contextmanager
    def kv_cache_scope(self, device: torch.device | str | None = None) -> Generator[None, None, None]:
        """上下文管理器，确保 prepare_kv_cache 和 clear_kv_cache 成对调用。

        Args:
            device: 可选的设备上下文，例如 torch.device("meta") 用于创建 meta tensor。

        Examples:
            >>> with model.kv_cache_scope():
            ...     # kv cache 已准备好
            ...     do_something()
            ... # kv cache 已自动清理

            >>> with model.kv_cache_scope(device="meta"):
            ...     # 在 meta device 上创建 kv cache
            ...     do_something()
        """
        try:
            if device is not None:
                with torch.device(device):
                    self.prepare_kv_cache()
                if torch.device(device).type == "meta":
                    device = "cpu"
                with torch.device(device):
                    self.prepare_other_cache()
            else:
                self.prepare_kv_cache()
                self.prepare_other_cache()
            yield
        finally:
            self.clear_kv_cache()
            self.clear_other_cache()

    def _set_device(self, device):
        self._device = device

    def _set_dtype(self, dtype):
        self._dtype = dtype

    def to(self, *args, **kwargs):
        device, dtype = torch._C._nn._parse_to(*args, **kwargs)[:2]
        if device is not None:
            self._set_device(device)
        if dtype is not None:
            self._set_dtype(dtype)
        return self


class EmptyKVCacheMixin:
    def __init__(self) -> None:
        self.use_cache = False

    def prepare_kv_cache(self):
        pass

    def clear_kv_cache(self):
        pass

    def clear_other_cache(self):
        pass

    def prepare_other_cache(self):
        pass

    @contextmanager
    def kv_cache_scope(self, device: torch.device | str | None = None) -> Generator[None, None, None]:
        try:
            self.prepare_kv_cache()
            self.prepare_other_cache()
            yield
        finally:
            self.clear_kv_cache()
            self.clear_other_cache()


class KVCacheWithLinearMixin(KVCacheMixin):
    kvcache_config: KVCacheWithLinearConfig

    def __init__(self, kv_cache_config: KVCacheWithLinearConfig) -> None:
        super().__init__(kv_cache_config)
        self.past_conv_caches: CacheList[Tensor | CacheTensor] = CacheList()
        self.past_recurrent_states: CacheList[Tensor | CacheTensor] = CacheList()

    def prepare_other_cache(self):
        linear_kv_cache_config = self.kvcache_config.linear_kv_cache_config
        for _i in range(linear_kv_cache_config.num_layers):
            conv_cache_shape = [
                linear_kv_cache_config.batch_size,
                linear_kv_cache_config.conv_dim,
                linear_kv_cache_config.conv_kernel_size,
            ]
            recurrent_cache_shape = [
                linear_kv_cache_config.batch_size,
                linear_kv_cache_config.num_v_heads,
                linear_kv_cache_config.head_k_dim,
                linear_kv_cache_config.head_v_dim,
            ]
            cache_dtype = linear_kv_cache_config.cache_torch_dtype
            self.past_conv_caches.append(
                CacheTensor(torch.zeros(conv_cache_shape, dtype=cache_dtype, device=self._device))
            )
            self.past_recurrent_states.append(
                CacheTensor(torch.zeros(recurrent_cache_shape, dtype=cache_dtype, device=self._device))
            )

    def clear_other_cache(self):
        self.past_conv_caches.clear()
        self.past_recurrent_states.clear()

    def _set_device(self, device):
        super()._set_device(device)
        for cache_tensor in self.past_conv_caches:
            cache_tensor.to(device)
        for cache_tensor in self.past_recurrent_states:
            cache_tensor.to(device)


class KVCacheContextManager:
    """KV Cache 上下文管理器类，提供更灵活的使用方式。"""

    def __init__(self, llm_model, devices: list[torch.device | str] | None = None):
        if isinstance(devices, (list, tuple)):
            device = devices[0]
        else:
            device = devices
        self.device = device
        self._model = llm_model

    def __enter__(self):
        if self.device is not None:
            with torch.device(self.device):
                self._model.get_kvcache_mixin().prepare_kv_cache()
            device = self.device
            if torch.device(device).type == "meta":
                device = "cpu"
            with torch.device(device):
                self._model.get_kvcache_mixin().prepare_other_cache()
        else:
            self._model.get_kvcache_mixin().prepare_kv_cache()
            self._model.get_kvcache_mixin().prepare_other_cache()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._model.get_kvcache_mixin().clear_kv_cache()
        self._model.get_kvcache_mixin().clear_other_cache()
