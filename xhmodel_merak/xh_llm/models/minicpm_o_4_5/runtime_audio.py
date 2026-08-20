from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import onnx
import torch
from torch import Tensor
from transformers.modeling_outputs import BaseModelOutputWithPast

from ...hmonnx.hmonnx_model import HMONNXBaseModel, HMONNXModel
from ...types import KVCacheConfig
from .runtime_cache import FixedCapacityDynamicCache, FixedCapacityKVCacheMixin, require_capacity


TensorSession = Callable[..., Tensor | Sequence[Tensor]]


def _input_shapes(path: Path) -> tuple[tuple[int, ...], ...]:
    graph = onnx.load(str(path), load_external_data=False).graph
    return tuple(tuple(int(dim.dim_value or 1) for dim in value.type.tensor_type.shape.dim) for value in graph.input)


def _streaming_current_length(valid_mel_length: int, prefix_overlap: int, suffix_overlap: int) -> int:
    return (valid_mel_length + 1) // 2 - (prefix_overlap + 1) // 2 - (suffix_overlap + 1) // 2


def _validate_and_copy_present_caches(
    present_keys: Sequence[Tensor],
    present_values: Sequence[Tensor],
    target_keys: Sequence[Tensor],
    target_values: Sequence[Tensor],
    next_length: int,
    capacity: int,
) -> None:
    for kind, values, targets in (
        ("present_k_cache", present_keys, target_keys),
        ("present_v_cache", present_values, target_values),
    ):
        for index, (present, target) in enumerate(zip(values, targets, strict=True)):
            if present.shape[0] != target.shape[0]:
                raise RuntimeError(f"Streaming Audio {kind}_{index} batch mismatch")
            if present.shape[1] != target.shape[1]:
                raise RuntimeError(f"Streaming Audio {kind}_{index} heads mismatch")
            if present.shape[2] < next_length or present.shape[2] > capacity:
                raise RuntimeError(
                    f"Streaming Audio {kind}_{index} sequence mismatch: "
                    f"length={present.shape[2]}, required={next_length}, capacity={capacity}"
                )
            if present.shape[3] != target.shape[3]:
                raise RuntimeError(f"Streaming Audio {kind}_{index} head_dim mismatch")
            if present.dtype != target.dtype:
                raise RuntimeError(f"Streaming Audio {kind}_{index} dtype mismatch")
            if present.device != target.device:
                raise RuntimeError(f"Streaming Audio {kind}_{index} device mismatch")
    for target, present in zip(target_keys, present_keys, strict=True):
        target[:, :, :next_length, :].copy_(present[:, :, :next_length, :])
    for target, present in zip(target_values, present_values, strict=True):
        target[:, :, :next_length, :].copy_(present[:, :, :next_length, :])


def run_static_audio_batch(
    session: TensorSession,
    input_features: Tensor,
    attention_mask: Tensor,
    exported_batch: int,
    exported_frames: int | None = None,
    exported_mask_size: int | None = None,
) -> Tensor | tuple[Tensor, ...]:
    real_batch = input_features.shape[0]
    if exported_batch <= 0:
        raise ValueError(f"exported audio batch must be positive, got {exported_batch}")
    if real_batch <= 0:
        raise ValueError("audio input batch must be non-empty")
    if exported_frames is not None:
        input_features = input_features[..., :exported_frames]
        if input_features.shape[-1] < exported_frames:
            input_features = torch.nn.functional.pad(input_features, (0, exported_frames - input_features.shape[-1]))
    if exported_mask_size is not None:
        attention_mask = attention_mask[..., :exported_mask_size, :exported_mask_size]
        row_padding = exported_mask_size - attention_mask.shape[-2]
        column_padding = exported_mask_size - attention_mask.shape[-1]
        if row_padding > 0 or column_padding > 0:
            attention_mask = torch.nn.functional.pad(
                attention_mask,
                (0, max(0, column_padding), 0, max(0, row_padding)),
                value=torch.finfo(attention_mask.dtype).min,
            )
    chunk_outputs: list[tuple[Tensor, ...]] = []
    output_is_tensor: bool | None = None
    for start in range(0, real_batch, exported_batch):
        features = input_features[start : start + exported_batch]
        mask = attention_mask[start : start + exported_batch]
        chunk_batch = features.shape[0]
        if chunk_batch < exported_batch:
            features = torch.cat([features, features.new_zeros((exported_batch - chunk_batch, *features.shape[1:]))])
            mask = torch.cat([mask, mask.new_zeros((exported_batch - chunk_batch, *mask.shape[1:]))])
        output = session(features.to(torch.float16), mask.to(torch.float16))
        current_is_tensor = isinstance(output, Tensor)
        if output_is_tensor is None:
            output_is_tensor = current_is_tensor
        elif output_is_tensor != current_is_tensor:
            raise RuntimeError("static Audio graph returned inconsistent output types across batches")
        values = (output,) if current_is_tensor else tuple(output)
        if chunk_outputs and len(values) != len(chunk_outputs[0]):
            raise RuntimeError("static Audio graph returned inconsistent output arity across batches")
        chunk_outputs.append(tuple(value[:chunk_batch] for value in values))

    merged = tuple(
        torch.cat([chunk[index] for chunk in chunk_outputs], dim=0) for index in range(len(chunk_outputs[0]))
    )
    return merged[0] if output_is_tensor else merged


class MiniCPMO45AudioHMONNXRuntime(HMONNXBaseModel):
    """Audio graph state machine plus the shared fixed-cache allocation.

    Audio is the sole consumer of the streaming overlap/pooling protocol, so
    keeping its logical cache lifecycle here removes the otherwise one-client
    cache-lifecycle abstraction.  ``FixedCapacityKVCacheMixin`` remains the
    shared allocator; this class only exposes the MiniCPM audio cache view.
    """

    cache_component = "audio"

    def __init__(
        self,
        root: Path,
        meta: dict[str, Any],
        *,
        enable_cuda_graph: bool = False,
        enable_auto_offload: bool = False,
        enable_golden: bool = False,
        device_map: str | torch.device | list[str | torch.device] | None = None,
    ) -> None:
        super().__init__(device_map=device_map)
        graph_options = {
            "enable_golden": enable_golden,
            "enable_cuda_graph": enable_cuda_graph,
            "enable_auto_offload": enable_auto_offload,
            "device_map": self._valid_devices,
        }
        self.num_hidden_layers = 0
        self._kvcache_mixin: FixedCapacityKVCacheMixin | None = None
        self.graph = root / meta["graphs"]["main"]
        self.input_shapes = _input_shapes(self.graph)
        self.session = HMONNXModel(str(self.graph), **graph_options)
        streaming_keys = {"stream_prefill", "stream_decode"}
        if not streaming_keys.issubset(meta["graphs"]):
            self.streaming_enabled = False
            return
        self.streaming_enabled = True
        self.stream_prefill_session = HMONNXModel(str(root / meta["graphs"]["stream_prefill"]), **graph_options)
        self.stream_decode_session = HMONNXModel(str(root / meta["graphs"]["stream_decode"]), **graph_options)
        for role in ("session_prefill", "session_decode"):
            name = f"{role}_session"
            if role in meta["graphs"]:
                setattr(self, name, HMONNXModel(str(root / meta["graphs"][role]), **graph_options))
            else:
                setattr(self, name, None)
        self.stream_prefill_frames = int(meta["stream_prefill_frames"])
        self.stream_decode_frames = int(meta["stream_decode_frames"])
        self.session_prefill_frames = int(meta.get("session_prefill_frames", self.stream_prefill_frames))
        self.session_decode_frames = int(meta.get("session_decode_frames", self.stream_decode_frames))
        self.kv_cache_shape = tuple(int(value) for value in meta["kv_cache_shape"])
        cache_capacity = int(meta["cache_capacity"])
        self.prefix_overlap_first = int(meta["prefix_overlap_first"])
        self.prefix_overlap_later = int(meta["prefix_overlap_later"])
        self.suffix_overlap = int(meta["suffix_overlap"])
        self.pool_step = int(meta.get("pool_step", 5))
        self.audio_encoder_layer = int(meta["audio_encoder_layer"])
        if self.audio_encoder_layer != -1:
            raise RuntimeError(
                f"Streaming Audio requires audio_encoder_layer=-1 for the selected-state graph, "
                f"got {self.audio_encoder_layer}"
            )
        self.configure_kv_cache(
            num_hidden_layers=int(meta["num_hidden_layers"]),
            cache_shape=self.kv_cache_shape,
            capacity=cache_capacity,
        )

    def configure_kv_cache(
        self,
        *,
        num_hidden_layers: int,
        cache_shape: Sequence[int],
        capacity: int,
    ) -> None:
        self.num_hidden_layers = int(num_hidden_layers)
        shape = tuple(int(value) for value in cache_shape)
        if int(shape[2]) != int(capacity):
            raise ValueError(
                f"{self.cache_component} cache capacity {capacity} does not match graph cache shape {shape[2]}"
            )
        self._kvcache_mixin = FixedCapacityKVCacheMixin(
            KVCacheConfig(
                num_layers=self.num_hidden_layers,
                kv_cache_shape=list(shape),
                cache_axis=2,
                batch_size=int(shape[0]),
                cache_dtype="float16",
                use_cache=True,
            ),
            self.cache_component,
        )
        self._kvcache_mixin.prepare_fixed_cache(self.device)

    @property
    def past_key_caches(self) -> list[Tensor]:
        return [] if self._kvcache_mixin is None else self._kvcache_mixin.past_key_caches

    @property
    def past_value_caches(self) -> list[Tensor]:
        return [] if self._kvcache_mixin is None else self._kvcache_mixin.past_value_caches

    @property
    def cache_capacity(self) -> int:
        return 0 if self._kvcache_mixin is None else self._kvcache_mixin.cache_capacity

    @property
    def cache_adapter(self) -> FixedCapacityDynamicCache:
        if self._kvcache_mixin is None:
            raise RuntimeError("audio KV cache is not configured")
        return self._kvcache_mixin.hf_cache

    @property
    def hf_cache(self) -> FixedCapacityDynamicCache:
        return self.cache_adapter

    @property
    def cache_length(self) -> int:
        return 0 if self._kvcache_mixin is None else self._kvcache_mixin.hf_cache.get_seq_length()

    def commit_cache_length(self, valid_length: int) -> None:
        self.cache_adapter.commit_length(valid_length)

    def reset_kvcache(self) -> None:
        if self._kvcache_mixin is not None:
            self._kvcache_mixin.reset_fixed_cache()

    def _set_device(self, device: torch.device):
        super()._set_device(device)
        if self._kvcache_mixin is not None:
            self._kvcache_mixin.prepare_fixed_cache(device)
        return self

    def reset_state(self) -> None:
        self.reset_kvcache()

    def release_state(self) -> None:
        """Release only the audio cache; graph release is centralized."""
        if self._kvcache_mixin is not None:
            self._kvcache_mixin.release_fixed_cache()
            self._kvcache_mixin = None

    @property
    def streaming_cache_length(self) -> int:
        return self.cache_length

    def forward_streaming(
        self,
        input_features: Tensor,
        valid_mel_length: Tensor | int | None = None,
        past_key_values=None,
        use_extra_context: bool = True,
        **_kwargs,
    ) -> BaseModelOutputWithPast:
        if not self.streaming_enabled:
            raise RuntimeError("Streaming Audio metadata/graphs are not available in this artifact")
        if self.audio_encoder_layer != -1:
            raise RuntimeError(f"Streaming Audio requires audio_encoder_layer=-1, got {self.audio_encoder_layer}")
        if input_features.shape[0] != 1:
            raise RuntimeError(f"Streaming Audio requires batch_size=1, got batch_size={input_features.shape[0]}")
        valid_length = (
            input_features.shape[-1] if valid_mel_length is None else int(torch.as_tensor(valid_mel_length).item())
        )
        if past_key_values is not None and past_key_values is not self.hf_cache:
            raise RuntimeError("Streaming Audio requires the runtime-owned past_key_values cache")
        mid_stream_reset = past_key_values is None and self.cache_length > 0
        if mid_stream_reset:
            self.reset_state()
        # A mid-stream reset chunk is still a later (decode-geometry) chunk: it carries the
        # later CNN prefix overlap, so route it through the decode graph on an empty cache.
        use_prefill = self.cache_length == 0 and not mid_stream_reset
        if use_extra_context:
            expected_frames = self.stream_prefill_frames if use_prefill else self.stream_decode_frames
            prefix_overlap = self.prefix_overlap_first if use_prefill else self.prefix_overlap_later
            suffix_overlap = self.suffix_overlap
            session = self.stream_prefill_session if use_prefill else self.stream_decode_session
        else:
            expected_frames = self.session_prefill_frames if use_prefill else self.session_decode_frames
            prefix_overlap = 0
            suffix_overlap = 0
            session = self.session_prefill_session if use_prefill else self.session_decode_session
            if session is None:
                raise RuntimeError("Session Audio graphs are not available in this artifact")
        if input_features.shape[-1] > expected_frames:
            raise RuntimeError(
                f"Streaming Audio accepts at most {expected_frames} Mel frames, got {input_features.shape[-1]}"
            )
        if not 0 < valid_length <= input_features.shape[-1]:
            raise RuntimeError(
                f"Streaming Audio valid_mel_length must be within input width: "
                f"valid={valid_length}, width={input_features.shape[-1]}"
            )
        current_length = _streaming_current_length(valid_length, prefix_overlap, suffix_overlap)
        if current_length <= 0:
            raise RuntimeError(f"Streaming Audio produced invalid current length: {current_length}")
        if self.cache_length + current_length >= self.cache_capacity:
            self.reset_state()
            use_prefill = True
            expected_frames = self.stream_prefill_frames if use_extra_context else self.session_prefill_frames
            prefix_overlap = self.prefix_overlap_first if use_extra_context else 0
            suffix_overlap = self.suffix_overlap if use_extra_context else 0
            session = self.stream_prefill_session if use_extra_context else self.session_prefill_session
            current_length = _streaming_current_length(valid_length, prefix_overlap, suffix_overlap)
        require_capacity("audio", self.cache_length, current_length, self.cache_capacity)
        padded_features = torch.nn.functional.pad(
            input_features,
            (0, expected_frames - input_features.shape[-1]),
        )
        if session is None:
            raise RuntimeError("Streaming Audio graph sessions are not loaded")
        query_capacity = _streaming_current_length(expected_frames, prefix_overlap, suffix_overlap)
        attention_mask = torch.full(
            (1, 1, query_capacity, self.cache_capacity),
            float("-inf"),
            dtype=torch.float16,
            device=input_features.device,
        )
        attention_mask[..., : self.cache_length + current_length] = 0
        outputs = session(
            padded_features.to(torch.float16),
            torch.tensor([valid_length], dtype=torch.int32, device=input_features.device),
            torch.tensor([self.cache_length], dtype=torch.int32, device=input_features.device),
            torch.tensor([current_length], dtype=torch.int32, device=input_features.device),
            attention_mask,
            *self.past_key_caches,
            *self.past_value_caches,
        )
        if isinstance(outputs, Tensor) or len(outputs) != 1 + 2 * self.num_hidden_layers:
            raise RuntimeError("Streaming Audio graph returned unexpected output arity")
        # The streaming graphs export projection AND pooling in-graph (pool_step=5):
        # the graph output is already pooled to (query_capacity - 5) // 5 + 1 frames.
        # The official host path pools the raw hidden states with the same AvgPool1d,
        # so the graph output frame count equals the official pooled frame count.
        pooled_capacity = (query_capacity - self.pool_step) // self.pool_step + 1
        audio_hidden_states = outputs[0][:, :pooled_capacity, :]
        present_keys = outputs[1 : 1 + self.num_hidden_layers]
        present_values = outputs[1 + self.num_hidden_layers :]
        next_length = self.cache_length + current_length
        _validate_and_copy_present_caches(
            present_keys,
            present_values,
            self.past_key_caches,
            self.past_value_caches,
            next_length,
            self.cache_capacity,
        )
        self.commit_cache_length(next_length)
        return BaseModelOutputWithPast(
            last_hidden_state=audio_hidden_states,
            hidden_states=(audio_hidden_states,),
            past_key_values=self.hf_cache,
        )

    def forward(self, input_features: Tensor, attention_mask: Tensor) -> Tensor | tuple[Tensor, ...]:
        return run_static_audio_batch(
            self.session,
            input_features,
            attention_mask,
            self.input_shapes[0][0],
            self.input_shapes[0][-1],
            self.input_shapes[1][-1],
        )


__all__ = ["MiniCPMO45AudioHMONNXRuntime", "run_static_audio_batch"]
