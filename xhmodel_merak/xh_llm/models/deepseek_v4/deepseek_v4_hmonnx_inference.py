"""HMONNX runtime for the fixed-shape DeepSeek-V4 Flash graphs."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from ...hmonnx import TextLLMHMONNXModel
from ...types import LLMModelMeta
from .cache_abi import DeepSeekV4CacheABI, default_layer_types
from .runtime import DeepSeekV4CacheMixin, DeepSeekV4DataPreprocess
from .static_cache import DeepSeekV4StaticCacheSpec


def _cache_input_devices_from_layer_infos(
    abi: DeepSeekV4CacheABI,
    layer_infos: Sequence[object],
) -> tuple[tuple[torch.device, ...], ...]:
    """Resolve persistent cache ownership from stable exported layer tags."""

    devices_by_layer: dict[int, torch.device] = {}
    for layer_info in layer_infos:
        device = torch.device(layer_info.device)
        for tag_name in getattr(layer_info, "llm_tags", ()):
            prefix = "layer_"
            if not str(tag_name).startswith(prefix):
                continue
            suffix = str(tag_name)[len(prefix) :]
            if not suffix.isdigit():
                continue
            layer = int(suffix)
            previous = devices_by_layer.setdefault(layer, device)
            if previous != device:
                raise RuntimeError(
                    f"DeepSeek-V4 layer {layer} was assigned to both {previous} and {device}"
                )

    missing = sorted(set(range(len(abi.layer_types))) - devices_by_layer.keys())
    if missing:
        raise RuntimeError(f"missing auto-offload placement for DeepSeek-V4 layers: {missing}")

    return tuple(
        (devices_by_layer[layer],) * len(input_names)
        for layer, input_names in enumerate(abi.graph_cache_input_names_by_layer())
    )


class XHDeepSeekV4HMONNXModel(TextLLMHMONNXModel):
    """Own the heterogeneous SWA/CSA/HCA caches and commit graph states."""

    def __init__(
        self,
        meta_info: LLMModelMeta,
        *,
        enable_auto_offload: bool = False,
        **kwargs,
    ) -> None:
        if kwargs.get("enable_page_attention", False):
            raise ValueError("DeepSeek-V4 uses its dedicated SWA/CSA/HCA cache ABI, not PageAttention")
        super().__init__(
            meta_info,
            enable_auto_offload=enable_auto_offload,
            **kwargs,
        )

        if self.pad_token_id is None:
            raise ValueError("DeepSeek-V4 HMONNX metadata must contain pad_token_id")

        model_config = meta_info.model_config
        configured_layers = getattr(model_config, "max_layers", None)
        configured_layers = 43 if configured_layers is None else int(configured_layers)
        layer_types = tuple(getattr(meta_info, "layer_types", default_layer_types(configured_layers)))
        spec = DeepSeekV4StaticCacheSpec(
            max_context_length=int(model_config.context_max_length),
            prefill_chunk_length=int(model_config.prefill_chunk_length),
        )
        self.cache_abi = DeepSeekV4CacheABI(
            spec=spec,
            layer_types=layer_types,
            batch_size=int(model_config.batch_size),
        )
        input_devices = None
        if enable_auto_offload:
            input_devices = _cache_input_devices_from_layer_infos(
                self.cache_abi,
                self.prefill_model.hmonnx_session.get_layer_infos(),
            )
        self._kvcache_mixin = DeepSeekV4CacheMixin(
            self.cache_abi,
            self.kvcache_config,
            input_devices=input_devices,
        )
        self._kvcache_mixin.to(device=self.device, dtype=self.dtype)
        self._data_processor = None

    def _get_data_preprocessor(self) -> DeepSeekV4DataPreprocess:
        return DeepSeekV4DataPreprocess(
            token_embedding=self.embed_tokens,
            input_sequence_length=self.get_input_sequence_length(),
            cache_mixin=self._kvcache_mixin,
            pad_token_id=self.pad_token_id,
        )

    def forward(self, *args):
        outputs = super().forward(*args)
        return self._kvcache_mixin.apply_flat_state_outputs(outputs)


__all__ = ["XHDeepSeekV4HMONNXModel"]
