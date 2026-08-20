"""Runtime cache ownership and fixed-shape input preparation.

This is the host half of the static graph contract.  Persistent K/V tensors
are ``CacheTensor`` instances updated in place by ``xh.LLMCache``.  Learned
compressor overlap states are ordinary explicit graph outputs and are copied
back into the next invocation's input tuple by ``apply_state_outputs``.
Alignment/count/mask control is rebuilt from absolute Host positions for each
invocation and is deliberately not persisted per layer.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import torch
from torch import Tensor, nn

from xhquant.core import CacheTensor, HybridCacheTensor

from ...kv_cache_mixin import KVCacheMixin
from ...llm_data_processor import BaseInputProcessorConfig, BaseLLMInputProcessor
from ...types import KVCacheConfig
from .cache_abi import (
    CSA,
    SLIDING,
    CSACacheInputs,
    CSAStateOutput,
    DeepSeekV4CacheABI,
    HCACacheInputs,
    HCAStateOutput,
    LayerCacheInputs,
    SWACacheInputs,
    flatten_cache_inputs,
)
from .host_masks import build_deepseek_v4_attention_masks


class DeepSeekV4CacheMixin(KVCacheMixin):
    """Own all heterogeneous caches for one static DeepSeek-V4 request."""

    def __init__(
        self,
        abi: DeepSeekV4CacheABI,
        kv_cache_config: KVCacheConfig | None = None,
        input_devices: Sequence[Sequence[torch.device | str]] | None = None,
    ) -> None:
        config = kv_cache_config or KVCacheConfig(
            num_layers=len(abi.layer_types),
            batch_size=abi.batch_size,
            cache_dtype="float16",
            use_cache=True,
        )
        super().__init__(config)
        self.abi = abi
        self.layer_caches: list[LayerCacheInputs] = []
        self.input_devices: tuple[tuple[torch.device, ...], ...] | None = None
        if input_devices is not None:
            expected_names = self.abi.graph_cache_input_names_by_layer()
            normalized = tuple(
                tuple(torch.device(device) for device in layer_devices) for layer_devices in input_devices
            )
            if len(normalized) != len(expected_names):
                raise ValueError(f"cache input device layers {len(normalized)} != {len(expected_names)}")
            for layer, (devices, names) in enumerate(zip(normalized, expected_names, strict=True)):
                if len(devices) != len(names):
                    raise ValueError(f"layer {layer} cache input devices {len(devices)} != {len(names)}")
            self.input_devices = normalized

    @staticmethod
    def _cache(shape: tuple[int, ...], *, device=None, dtype=torch.float16) -> CacheTensor:
        return CacheTensor(torch.zeros(shape, device=device, dtype=dtype))

    @staticmethod
    def _swa_cache(
        shape: tuple[int, ...],
        *,
        device=None,
        dtype=torch.float16,
    ) -> HybridCacheTensor:
        return HybridCacheTensor(torch.zeros(shape, device=device, dtype=dtype))

    @staticmethod
    def _state(
        shape: tuple[int, ...],
        *,
        score: bool,
        device=None,
        dtype=torch.float16,
    ) -> Tensor:
        if score:
            return torch.full(shape, -65504.0, device=device, dtype=dtype)
        return torch.zeros(shape, device=device, dtype=dtype)

    def prepare_kv_cache(self, dtype: torch.dtype = torch.float16) -> None:
        if self._cache_initialized or not self.use_cache:
            return
        device = self._device
        effective_device = torch.get_default_device() if device is None else torch.device(device)
        # HMONNX export intentionally allocates the large persistent K/V
        # tensors on meta.  Compressor overlap states are explicit ordinary
        # graph inputs, however, and are concatenated with CPU projection
        # results before LLMCache.  They are small, so keep them on CPU during
        # meta-cache export to avoid mixed-device native arithmetic.
        state_device = torch.device("cpu") if effective_device.type == "meta" else effective_device
        spec = self.abi.spec
        shapes = spec.tensor_shapes(self.abi.batch_size)
        cache_input_names = self.abi.graph_cache_input_names_by_layer()
        for layer_index, layer_type in enumerate(self.abi.layer_types):
            layer_devices = (
                self.input_devices[layer_index]
                if self.input_devices is not None
                else tuple(
                    torch.device(state_device) if "_state_input" in name else effective_device
                    for name in cache_input_names[layer_index]
                )
            )
            device_iter = iter(layer_devices)
            swa_shape = (
                self.abi.batch_size,
                1,
                self.abi.persistent_swa_length,
                spec.latent_head_dim,
            )
            swa_k = self._swa_cache(swa_shape, device=next(device_iter), dtype=dtype)
            swa_v = self._swa_cache(swa_shape, device=next(device_iter), dtype=dtype)
            if layer_type == SLIDING:
                self.layer_caches.append(SWACacheInputs(swa_k, swa_v))
                continue

            if layer_type == CSA:
                main_shape = (self.abi.batch_size, 1, spec.csa_capacity, spec.latent_head_dim)
                index_shape = (self.abi.batch_size, 1, spec.csa_capacity, spec.index_head_dim)
                self.layer_caches.append(
                    CSACacheInputs(
                        swa_k,
                        swa_v,
                        self._cache(main_shape, device=next(device_iter), dtype=dtype),
                        self._cache(index_shape, device=next(device_iter), dtype=dtype),
                        self._state(shapes["csa_main_kv_state"], score=False, device=next(device_iter), dtype=dtype),
                        self._state(shapes["csa_main_score_state"], score=True, device=next(device_iter), dtype=dtype),
                        self._state(shapes["csa_index_kv_state"], score=False, device=next(device_iter), dtype=dtype),
                        self._state(shapes["csa_index_score_state"], score=True, device=next(device_iter), dtype=dtype),
                    )
                )
                continue

            main_shape = (self.abi.batch_size, 1, spec.hca_capacity, spec.latent_head_dim)
            self.layer_caches.append(
                HCACacheInputs(
                    swa_k,
                    swa_v,
                    self._cache(main_shape, device=next(device_iter), dtype=dtype),
                    self._state(shapes["hca_main_kv_state"], score=False, device=next(device_iter), dtype=dtype),
                    self._state(shapes["hca_main_score_state"], score=True, device=next(device_iter), dtype=dtype),
                )
            )
        self._cache_initialized = True

    def clear_kv_cache(self) -> None:
        if self.layer_caches and not any(
            tensor.device.type == "meta" for cache in self.layer_caches for tensor in cache
        ):
            self.reset_kv_cache()
            return
        self.layer_caches.clear()
        self.past_key_caches.clear()
        self.past_value_caches.clear()
        self._cache_initialized = False

    def reset_kv_cache(self) -> None:
        """Zero request state in place while preserving CUDA Graph addresses."""

        for layer_type, cache in zip(self.abi.layer_types, self.layer_caches, strict=True):
            swa_k = cache.k if isinstance(cache, SWACacheInputs) else cache.swa_k
            swa_v = cache.v if isinstance(cache, SWACacheInputs) else cache.swa_v
            swa_k.zero_()
            swa_v.zero_()
            swa_k.cache_valid_len_tensor.zero_()
            swa_v.cache_valid_len_tensor.zero_()
            if layer_type == CSA:
                cache.main.zero_()
                cache.index_k.zero_()
                cache.main_kv_state.zero_()
                cache.main_score_state.fill_(-65504.0)
                cache.index_kv_state.zero_()
                cache.index_score_state.fill_(-65504.0)
            elif layer_type != SLIDING:
                cache.main.zero_()
                cache.main_kv_state.zero_()
                cache.main_score_state.fill_(-65504.0)

    def cache_inputs(self) -> tuple[LayerCacheInputs, ...]:
        if not self._cache_initialized:
            self.prepare_kv_cache()
        return tuple(self.layer_caches)

    def residency_summary(self) -> dict[str, object]:
        """Prove physical cache/state storage matches graph consumer devices."""

        caches = self.cache_inputs()
        names_by_layer = self.abi.graph_cache_input_names_by_layer()
        actual_by_device: dict[str, int] = {}
        mismatches: list[dict[str, str]] = []
        tensor_count = 0
        default_device = torch.get_default_device() if self._device is None else torch.device(self._device)
        for layer, (cache, names) in enumerate(zip(caches, names_by_layer, strict=True)):
            expected_devices = (
                self.input_devices[layer] if self.input_devices is not None else (default_device,) * len(names)
            )
            for name, tensor, expected in zip(names, cache, expected_devices, strict=True):
                actual = tensor.device
                actual_by_device[str(actual)] = actual_by_device.get(str(actual), 0) + 1
                tensor_count += 1
                if actual != expected:
                    mismatches.append(
                        {
                            "input": name,
                            "expected": str(expected),
                            "actual": str(actual),
                        }
                    )
        return {
            "tensor_count": tensor_count,
            "devices": actual_by_device,
            "mismatches": mismatches,
        }

    @staticmethod
    def _move(values: Iterable[Tensor], *args, **kwargs) -> tuple[Tensor, ...]:
        return tuple(value.to(*args, **kwargs) for value in values)

    def _set_device(self, device):
        super()._set_device(device)
        if self.input_devices is not None:
            # Persistent model-parallel cache ownership is fixed by the graph
            # consumers.  A model-level ``to(first_device)`` moves embeddings
            # and Host inputs only; collapsing caches would lose in-place
            # updates made by LLMCache on the other stages.
            return
        if not self._cache_initialized:
            return
        moved: list[LayerCacheInputs] = []
        for cache in self.layer_caches:
            values = self._move(cache, device=device)
            moved.append(type(cache)(*values))
        self.layer_caches = moved

    def apply_state_outputs(
        self,
        csa_states: tuple[CSAStateOutput, ...],
        hca_states: tuple[HCAStateOutput, ...],
    ) -> None:
        """Feed only learned compressor carry tensors back into the request."""

        if len(csa_states) != len(self.abi.csa_layers):
            raise ValueError("CSA state output count does not match the cache ABI")
        if len(hca_states) != len(self.abi.hca_layers):
            raise ValueError("HCA state output count does not match the cache ABI")
        csa_iter = iter(csa_states)
        hca_iter = iter(hca_states)
        updated: list[LayerCacheInputs] = []
        for layer_type, cache in zip(self.abi.layer_types, self.layer_caches, strict=True):
            if layer_type == SLIDING:
                updated.append(cache)
            elif layer_type == CSA:
                if not isinstance(cache, CSACacheInputs):
                    raise TypeError("CSA cache ABI was corrupted")
                state = next(csa_iter)
                cache.main_kv_state.copy_(state.main_kv_state)
                cache.main_score_state.copy_(state.main_score_state)
                cache.index_kv_state.copy_(state.index_kv_state)
                cache.index_score_state.copy_(state.index_score_state)
                updated.append(
                    CSACacheInputs(
                        cache.swa_k,
                        cache.swa_v,
                        cache.main,
                        cache.index_k,
                        cache.main_kv_state,
                        cache.main_score_state,
                        cache.index_kv_state,
                        cache.index_score_state,
                    )
                )
            else:
                if not isinstance(cache, HCACacheInputs):
                    raise TypeError("HCA cache ABI was corrupted")
                state = next(hca_iter)
                cache.main_kv_state.copy_(state.main_kv_state)
                cache.main_score_state.copy_(state.main_score_state)
                updated.append(
                    HCACacheInputs(
                        cache.swa_k,
                        cache.swa_v,
                        cache.main,
                        cache.main_kv_state,
                        cache.main_score_state,
                    )
                )
        self.layer_caches = updated

    def apply_flat_state_outputs(self, outputs: Tensor | Sequence[Tensor]) -> Tensor:
        """Commit the flat HMONNX output ABI and return its logits tensor."""

        values = (outputs,) if isinstance(outputs, Tensor) else tuple(outputs)
        expected = 1 + 4 * len(self.abi.csa_layers) + 2 * len(self.abi.hca_layers)
        if len(values) != expected:
            raise ValueError(f"expected {expected} DeepSeek-V4 graph outputs, got {len(values)}")

        cursor = 1
        csa_states = []
        for _ in self.abi.csa_layers:
            csa_states.append(CSAStateOutput(*values[cursor : cursor + 4]))
            cursor += 4
        hca_states = []
        for _ in self.abi.hca_layers:
            hca_states.append(HCAStateOutput(*values[cursor : cursor + 2]))
            cursor += 2
        self.apply_state_outputs(tuple(csa_states), tuple(hca_states))
        return values[0]


class DeepSeekV4DataPreprocess(BaseLLMInputProcessor):
    """Pad one fixed prefill/decode call and flatten its static state inputs."""

    def __init__(
        self,
        *,
        token_embedding: nn.Module,
        input_sequence_length: int,
        cache_mixin: DeepSeekV4CacheMixin,
        pad_token_id: int = 0,
    ) -> None:
        super().__init__(
            BaseInputProcessorConfig(
                embed_tokens=token_embedding,
                input_sequence_length=input_sequence_length,
                past_key_caches=cache_mixin.past_key_caches,
                past_value_caches=cache_mixin.past_value_caches,
                pad_token_id=pad_token_id,
            )
        )
        self.cache_mixin = cache_mixin

    def forward(self, data: dict | tuple | list):
        if not isinstance(data, dict):
            raise TypeError("DeepSeek-V4 inputs must be a dictionary")
        input_ids = data.get("input_ids")
        inputs_embeds = data.get("inputs_embeds")
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("provide exactly one of input_ids or inputs_embeds")
        if input_ids is None:
            raise ValueError("DeepSeek-V4 hash-MoE routing requires input_ids")
        # Tokenizers conventionally return int64, while the exported
        # DeepSeek-V4 hash-routing ABI is deliberately int32.  Normalize at
        # the host boundary so the static graph remains free of Cast nodes.
        input_ids = input_ids.to(device=self._device, dtype=torch.int32)
        if input_ids.ndim != 2 or input_ids.shape[0] != self.cache_mixin.abi.batch_size:
            raise ValueError("input_ids must be [B,S] with the wrapped batch size")
        current_length = int(input_ids.shape[1])
        if current_length <= 0 or current_length > self.input_sequence_length:
            raise ValueError(f"current input length must be in [1,{self.input_sequence_length}], got {current_length}")
        if current_length < self.input_sequence_length:
            padding = torch.full(
                (input_ids.shape[0], self.input_sequence_length - current_length),
                int(self.pad_token_id),
                device=input_ids.device,
                dtype=input_ids.dtype,
            )
            input_ids = torch.cat((input_ids, padding), dim=1)
        inputs_embeds = self.embed_tokens.to(self._device)(input_ids)

        past_length = int(data.get("past_seq_length", 0))
        spec = self.cache_mixin.abi.spec
        if past_length < 0 or past_length + current_length > spec.max_context_length:
            raise ValueError("request exceeds the wrapped maximum context length")
        masks = build_deepseek_v4_attention_masks(
            spec,
            input_sequence_length=self.input_sequence_length,
            past_length=past_length,
            current_length=current_length,
            batch_size=input_ids.shape[0],
            device=inputs_embeds.device,
            dtype=torch.float16,
        )
        return (
            inputs_embeds,
            input_ids,
            torch.tensor([past_length], device=self._device, dtype=torch.int32),
            torch.tensor([current_length], device=self._device, dtype=torch.int32),
            torch.tensor([current_length - 1], device=self._device, dtype=torch.int32),
            torch.tensor([past_length // spec.csa_ratio], device=self._device, dtype=torch.int32),
            torch.tensor([past_length // spec.hca_ratio], device=self._device, dtype=torch.int32),
            masks.swa_attention_mask,
            masks.csa_index_validity,
            masks.csa_attention_mask,
            masks.hca_attention_mask,
            masks.csa_compressor_validity,
            masks.csa_compressor_new_count,
            masks.csa_compressor_offset,
            masks.csa_compressor_phase_indices,
            masks.hca_compressor_validity,
            masks.hca_compressor_new_count,
            masks.hca_compressor_offset,
            masks.hca_compressor_phase_indices,
            flatten_cache_inputs(self.cache_mixin.cache_inputs()),
        )


__all__ = ["DeepSeekV4CacheMixin", "DeepSeekV4DataPreprocess"]
