from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

import torch
from torch import Tensor
from transformers.cache_utils import Cache, DynamicCache

from ...kv_cache_mixin import KVCacheMixin
from ...types import KVCacheConfig


def require_capacity(component: str, current: int, requested: int, capacity: int) -> None:
    if current + requested > capacity:
        raise RuntimeError(
            f"{component} cache capacity exceeded: current={current}, requested={requested}, capacity={capacity}"
        )


class _CacheLayerList(list[Tensor]):
    def __init__(self, owner: FixedCapacityDynamicCache, values: Sequence[Tensor], is_key: bool) -> None:
        super().__init__(values)
        self._owner = owner
        self._is_key = is_key

    def __setitem__(self, index: int, value: Tensor) -> None:
        target = self._owner._stage_layer(index, value, self._is_key)
        super().__setitem__(index, target[:, :, : self._owner._valid_length, :])


class FixedCapacityDynamicCache(DynamicCache):
    """HF-compatible view over fixed-capacity key/value cache buffers.

    This adapter exists because MiniCPM's official streaming/duplex host code
    treats ``past_key_values`` as live, mutable HF state: it reads layer
    ``key_cache``/``value_cache`` tensors, replaces compacted or reindexed
    layers, calls ``update``/``crop``, and retains the same cache object across
    calls.  The repository's ordinary text wrappers do not need this bridge:
    they keep the real KV state entirely in ``KVCacheMixin`` and use the HF
    cache only as an opaque generation-protocol value while tracking sequence
    length separately.

    At the same time, MiniCPM's HMONNX graph ABI requires full-capacity
    ``CacheTensor`` inputs with stable identity.  A normal growing
    ``DynamicCache`` reallocates/replaces tensors and would therefore diverge
    from the buffers read and written by HMONNX.  This class exposes the one
    shared fixed backing store through the mutable HF Cache API; it is the
    compatibility boundary between those two contracts, not another cache
    owner or allocator.

    Backing tensors are full-capacity and shared with the on-device runtime; only the
    committed valid region is exposed through :attr:`key_cache` / :attr:`value_cache`.
    Layer indices must be updated strictly in order ``0..num_layers-1`` within a step; the
    shared valid length advances only once every layer has been written.

    ``KVCacheMixin`` owns the backing ``CacheTensor`` allocation.  MiniCPM-o-4.5's
    published decoder graphs take those fixed tensors, while the official HF
    generation path passes a ``DynamicCache``.  This adapter preserves both object
    identities and is therefore a graph-ABI bridge, not a second cache allocator.
    """

    def __init__(
        self,
        key_caches: Sequence[Tensor],
        value_caches: Sequence[Tensor],
        capacity: int,
        component: str,
    ) -> None:
        Cache.__init__(self, layers=[])
        self._key_caches = list(key_caches)
        self._value_caches = list(value_caches)
        self._capacity = capacity
        self._component = component
        self._validate_buffers()
        self._valid_length = 0
        self._step_next_layer = 0
        self._staged_keys: dict[int, Tensor] = {}
        self._staged_values: dict[int, Tensor] = {}

    def _validate_buffers(self) -> None:
        num_key = len(self._key_caches)
        num_value = len(self._value_caches)
        if num_key != num_value:
            raise ValueError(
                f"{self._component} cache layer mismatch: {num_key} key layers vs {num_value} value layers"
            )
        if num_key == 0:
            raise ValueError(f"{self._component} cache has no layers")
        for index, (key, value) in enumerate(zip(self._key_caches, self._value_caches, strict=True)):
            if key.shape != value.shape:
                raise ValueError(
                    f"{self._component} cache layer {index} key/value shape mismatch: "
                    f"{tuple(key.shape)} vs {tuple(value.shape)}"
                )
            if key.shape[2] != self._capacity:
                raise ValueError(
                    f"{self._component} cache layer {index} capacity mismatch: "
                    f"shape[2]={key.shape[2]}, capacity={self._capacity}"
                )

    @property
    def _num_layers(self) -> int:
        return len(self._key_caches)

    @property
    def max_cache_len(self) -> int:
        """Return the maximum cache length of the cache."""
        return self._capacity

    @property
    def key_cache(self) -> list[Tensor]:
        return _CacheLayerList(self, [cache[:, :, : self._valid_length, :] for cache in self._key_caches], True)

    @key_cache.setter
    def key_cache(self, values: Sequence[Tensor]) -> None:
        self._replace_cache_values(self._key_caches, values)

    @property
    def value_cache(self) -> list[Tensor]:
        return _CacheLayerList(self, [cache[:, :, : self._valid_length, :] for cache in self._value_caches], False)

    @value_cache.setter
    def value_cache(self, values: Sequence[Tensor]) -> None:
        self._replace_cache_values(self._value_caches, values)

    def _replace_cache_values(self, targets: Sequence[Tensor], values: Sequence[Tensor]) -> None:
        if len(targets) != len(values):
            raise RuntimeError(f"{self._component} cache layer count cannot change")
        is_key = targets is self._key_caches
        for index, value in enumerate(values):
            self._stage_layer(index, value, is_key)

    def _stage_layer(self, index: int, value: Tensor, is_key: bool) -> Tensor:
        target = self._key_caches[index] if is_key else self._value_caches[index]
        try:
            self._validate_replacement(target, value)
        except RuntimeError:
            self._clear_staged_replacements()
            raise
        staged = self._staged_keys if is_key else self._staged_values
        staged[index] = value.clone()
        if len(self._staged_keys) == self._num_layers and len(self._staged_values) == self._num_layers:
            self._apply_staged_replacements()
        return target

    def _validate_replacement(self, target: Tensor, value: Tensor) -> None:
        if value.shape[:2] + value.shape[-1:] != target.shape[:2] + target.shape[-1:]:
            raise RuntimeError(f"{self._component} cache compaction shape mismatch")
        if value.shape[-2] > self._capacity:
            raise RuntimeError(f"{self._component} cache compaction exceeds capacity")
        if value.dtype != target.dtype or value.device != target.device:
            raise RuntimeError(f"{self._component} cache compaction dtype or device mismatch")

    def _apply_staged_replacements(self) -> None:
        staged = [*self._staged_keys.values(), *self._staged_values.values()]
        lengths = {int(value.shape[-2]) for value in staged}
        if len(lengths) != 1:
            self._clear_staged_replacements()
            raise RuntimeError(f"{self._component} cache compaction requires equal layer lengths")
        length = next(iter(lengths))
        targets = [*self._key_caches, *self._value_caches]
        values = [self._staged_keys[index] for index in range(self._num_layers)] + [
            self._staged_values[index] for index in range(self._num_layers)
        ]
        backups = [target.clone() for target in targets]
        try:
            for target, value in zip(targets, values, strict=True):
                target.zero_()
                target[:, :, :length, :].copy_(value)
        except RuntimeError:
            for target, backup in zip(targets, backups, strict=True):
                target.copy_(backup)
            self._clear_staged_replacements()
            raise
        self._valid_length = length
        self._step_next_layer = 0
        self._clear_staged_replacements()

    def _clear_staged_replacements(self) -> None:
        self._staged_keys.clear()
        self._staged_values.clear()

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self._valid_length

    def get_max_cache_shape(self, layer_idx: int = 0) -> int:
        return self._capacity

    def __len__(self) -> int:
        return self._num_layers

    def __getitem__(self, layer_idx: int) -> tuple[Tensor, Tensor]:
        if not 0 <= layer_idx < self._num_layers:
            raise KeyError(
                f"Cache only has {self._num_layers} layers, attempted to access layer with index {layer_idx}"
            )
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def __iter__(self) -> Iterator[tuple[Tensor, Tensor]]:
        for layer_idx in range(self._num_layers):
            yield self.key_cache[layer_idx], self.value_cache[layer_idx]

    def update(
        self,
        key_states: Tensor,
        value_states: Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[Tensor, Tensor]:
        if not 0 <= layer_idx < self._num_layers:
            raise RuntimeError(
                f"{self._component} cache update out of range: layer_idx={layer_idx}, num_layers={self._num_layers}"
            )
        if layer_idx != self._step_next_layer:
            raise RuntimeError(
                f"{self._component} cache update out of order: expected layer {self._step_next_layer}, got {layer_idx}"
            )
        requested = int(key_states.shape[-2])
        current = self._valid_length
        require_capacity(self._component, current, requested, self._capacity)
        step_length = current + requested
        key_target = self._key_caches[layer_idx]
        value_target = self._value_caches[layer_idx]
        expected = key_target.shape[:2] + (requested, key_target.shape[-1])
        if key_states.shape != expected:
            raise RuntimeError(
                f"{self._component} cache update shape mismatch: expected "
                f"[{key_target.shape[0]}, {key_target.shape[1]}, {requested}, {key_target.shape[-1]}], "
                f"got {tuple(key_states.shape)}"
            )
        if value_states.shape != expected:
            raise RuntimeError(
                f"{self._component} cache value update shape mismatch: expected {tuple(expected)}, "
                f"got {tuple(value_states.shape)}"
            )
        if key_states.dtype != key_target.dtype or value_states.dtype != value_target.dtype:
            raise RuntimeError(f"{self._component} cache update dtype mismatch")
        if key_states.device != key_target.device or value_states.device != value_target.device:
            raise RuntimeError(f"{self._component} cache update device mismatch")
        key_target[:, :, current:step_length, :] = key_states
        value_target[:, :, current:step_length, :] = value_states
        if layer_idx == self._num_layers - 1:
            self._valid_length = step_length
            self._step_next_layer = 0
        else:
            self._step_next_layer = layer_idx + 1
        return key_target[:, :, :step_length, :], value_target[:, :, :step_length, :]

    def rebind(self, key_caches: Sequence[Tensor], value_caches: Sequence[Tensor]) -> None:
        """Repoint this cache object at moved backing buffers without replacing it.

        Callers that hold a live reference to this cache (e.g. official generation code
        keeping ``past_key_values``) must keep observing the moved buffers, so the binding
        is updated in place and the current valid length is preserved.
        """
        if len(key_caches) != self._num_layers or len(value_caches) != self._num_layers:
            raise RuntimeError(f"{self._component} cache rebind layer count cannot change")
        self._key_caches = list(key_caches)
        self._value_caches = list(value_caches)
        self._validate_buffers()

    def crop(self, max_length: int) -> None:
        if max_length >= self._valid_length:
            return
        for key, value in zip(self._key_caches, self._value_caches, strict=True):
            key[:, :, max_length:, :].zero_()
            value[:, :, max_length:, :].zero_()
        self._valid_length = max_length
        self._step_next_layer = 0

    def reset(self) -> None:
        self._valid_length = 0
        self._step_next_layer = 0
        self._clear_staged_replacements()

    def commit_length(self, valid_length: int) -> None:
        if not 0 <= valid_length <= self._capacity:
            raise RuntimeError(
                f"{self._component} cache valid length out of range: "
                f"valid_length={valid_length}, capacity={self._capacity}"
            )
        self._valid_length = valid_length
        self._step_next_layer = 0
        self._clear_staged_replacements()

    def to_legacy_cache(self) -> tuple[tuple[Tensor, Tensor], ...]:
        return tuple(
            (key[:, :, : self._valid_length, :], value[:, :, : self._valid_length, :])
            for key, value in zip(self._key_caches, self._value_caches, strict=True)
        )


def bind_cache_tensors(
    key_caches: Sequence[Tensor],
    value_caches: Sequence[Tensor],
    capacity: int,
    component: str,
) -> FixedCapacityDynamicCache:
    return FixedCapacityDynamicCache(key_caches, value_caches, capacity, component)


class FixedCapacityKVCacheMixin(KVCacheMixin):
    """Shared KV storage with MiniCPM's fixed-capacity HF-cache view.

    ``KVCacheMixin`` remains the sole allocator and owner of the graph-facing
    ``CacheTensor`` lists.  MiniCPM-o-4.5 additionally needs an official
    remote-code ``DynamicCache`` object, which is why this small adapter keeps
    only validity/commit state in :class:`FixedCapacityDynamicCache`.

    This must not be replaced with a normal growing ``DynamicCache``: the
    exported graphs receive fixed-capacity CacheTensor inputs and retain their
    object identities between calls.
    """

    def __init__(self, config: KVCacheConfig, component: str) -> None:
        super().__init__(config)
        self._component = component
        self._cache_adapter: FixedCapacityDynamicCache | None = None

    @property
    def cache_capacity(self) -> int:
        shape = self.kvcache_config.kv_cache_shape
        if isinstance(shape[0], list):
            shape = shape[0]
        return int(shape[self.kvcache_config.cache_axis])

    def prepare_fixed_cache(self, device: torch.device | str) -> None:
        """Prepare/move shared backing buffers and preserve the HF-cache handle."""
        target = torch.device(device)
        if not self._cache_initialized:
            with torch.device(target):
                self.prepare_kv_cache()
        elif self.past_key_caches and self.past_key_caches[0].device != target:
            # KVCacheMixin deliberately leaves movement policy to its caller.
            # Here that caller is the MiniCPM fixed-graph bridge, which must
            # rebind the already-published HF cache object after the move.
            self.past_key_caches.to(target)
            self.past_value_caches.to(target)
        self._device = target
        if self._cache_adapter is not None:
            self._cache_adapter.rebind(self.past_key_caches, self.past_value_caches)

    @property
    def hf_cache(self) -> FixedCapacityDynamicCache:
        if self._cache_adapter is None:
            if not self._cache_initialized:
                raise RuntimeError(f"{self._component} KV cache has not been prepared")
            self._cache_adapter = bind_cache_tensors(
                self.past_key_caches,
                self.past_value_caches,
                self.cache_capacity,
                self._component,
            )
        return self._cache_adapter

    def reset_fixed_cache(self) -> None:
        for cache in [*self.past_key_caches, *self.past_value_caches]:
            reset = getattr(cache, "reset", None)
            if callable(reset):
                reset()
            else:
                cache.zero_()
        if self._cache_adapter is not None:
            self._cache_adapter.reset()

    def release_fixed_cache(self) -> None:
        self.reset_fixed_cache()
        self.clear_kv_cache()
        self._cache_adapter = None


__all__ = [
    "FixedCapacityKVCacheMixin",
    "FixedCapacityDynamicCache",
    "bind_cache_tensors",
    "require_capacity",
]
