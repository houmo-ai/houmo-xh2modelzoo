"""Static DeepSeek-V4 Flash block stack and causal-LM head.

The checkpoint contains a heterogeneous attention schedule: two sliding
layers followed by alternating CSA and HCA layers.  This module keeps that
schedule as Python constants while building the graph, so tracing/export sees
an ordinary, fully-unrolled static graph for either a diagnostic prefix or all
43 decoder blocks.

Long-lived K/V tensors are updated by ``xh.LLMCache`` inside each attention
module.  Compressor overlap tensors cannot be hidden in that operator and are
therefore explicit graph inputs/outputs, one state tuple per compressed layer.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor, nn

from xhquant import nn as xhnn

from ._trace import is_fx_proxy, one_element_int
from .cache_abi import (
    CSACacheInputs,
    CSAStateOutput,
    FlatCacheInputs,
    HCACacheInputs,
    HCAStateOutput,
    LayerCacheInputs,
    SWACacheInputs,
    flatten_cache_inputs,
)
from .decoder import (
    StaticCSADecoderLayer,
    StaticHCADecoderLayer,
    StaticSWADecoderLayer,
)
from .mhc import BoundHCHead
from .norm import rms_norm_from_hf


_CACHE_INPUT_TYPE = {
    "sliding_attention": SWACacheInputs,
    "compressed_sparse_attention": CSACacheInputs,
    "heavily_compressed_attention": HCACacheInputs,
}


def _build_rope_cache(
    inv_freq: Tensor,
    attention_scaling: Tensor | float,
    positions: Tensor,
) -> tuple[Tensor, Tensor]:
    """Precompute one interleaved-pair RoPE table for static slicing."""

    frequencies = positions.float().unsqueeze(-1) * inv_freq.detach().float().reshape(1, -1)
    scaling = torch.as_tensor(
        attention_scaling,
        device=frequencies.device,
        dtype=torch.float32,
    )
    return (
        (torch.cos(frequencies) * scaling).unsqueeze(0),
        (torch.sin(frequencies) * scaling).unsqueeze(0),
    )


class StaticBlocksOutput(NamedTuple):
    hidden_states: Tensor
    csa_states: tuple[CSAStateOutput, ...]
    hca_states: tuple[HCAStateOutput, ...]


class StaticCausalLMOutput(NamedTuple):
    logits: Tensor
    csa_states: tuple[CSAStateOutput, ...]
    hca_states: tuple[HCAStateOutput, ...]


class StaticDeepSeekV4Blocks(nn.Module):
    """An arbitrary fixed prefix of the checkpoint's decoder blocks."""

    _SUPPORTED_LAYER_TYPES = {
        "sliding_attention",
        "compressed_sparse_attention",
        "heavily_compressed_attention",
    }

    def __init__(
        self,
        layers: list[nn.Module],
        layer_types: tuple[str, ...],
        *,
        hc_mult: int,
    ) -> None:
        super().__init__()
        if not layers:
            raise ValueError("at least one decoder layer is required")
        if len(layers) != len(layer_types):
            raise ValueError("layers and layer_types must have the same length")
        unsupported = set(layer_types) - self._SUPPORTED_LAYER_TYPES
        if unsupported:
            raise ValueError(f"unsupported attention layer types: {sorted(unsupported)}")
        self.layers = nn.ModuleList(layers)
        self.layer_types = tuple(layer_types)
        self.hc_mult = int(hc_mult)
        # These are execution boundaries, not optional graph decoration.
        # HMONNX model parallelism uses one LLM Tag per decoder block to keep
        # every block and its cache inputs on one GPU, and to project the same
        # placement from the prefill graph onto decode.
        self.layer_tags = nn.ModuleList(
            [xhnn.XHTag(f"layer_{index}", "LLM", f"layer_{index}") for index in range(len(self.layers))]
        )

    @classmethod
    def from_hf(
        cls,
        model: nn.Module,
        *,
        input_sequence_length: int,
        max_context_length: int = 256 * 1024,
        max_layers: int | None = None,
        moe_fast_mode: bool = True,
        swa_backing_length: int | None = None,
    ) -> "StaticDeepSeekV4Blocks":
        base = model.model if hasattr(model, "model") and hasattr(model.model, "layers") else model
        if not hasattr(base, "layers"):
            raise ValueError("model must expose decoder layers")
        available = len(base.layers)
        count = available if max_layers is None else int(max_layers)
        if count <= 0 or count > available:
            raise ValueError(f"max_layers must be in [1, {available}], got {count}")

        source_layers = list(base.layers[:count])
        layers: list[nn.Module] = []
        layer_types: list[str] = []
        for source in source_layers:
            layer_type = str(source.self_attn.layer_type)
            layer_types.append(layer_type)
            if layer_type == "sliding_attention":
                layer = StaticSWADecoderLayer.from_hf(
                    source,
                    input_sequence_length=input_sequence_length,
                    swa_backing_length=swa_backing_length,
                    moe_fast_mode=moe_fast_mode,
                )
            elif layer_type == "compressed_sparse_attention":
                compressor = source.self_attn.compressor
                cache_capacity = int(max_context_length) // int(compressor.compress_rate)
                if cache_capacity < int(compressor.indexer.index_topk):
                    raise ValueError(
                        "max_context_length is too small for the CSA index_topk: "
                        f"capacity={cache_capacity}, topk={compressor.indexer.index_topk}"
                    )
                layer = StaticCSADecoderLayer.from_hf(
                    source,
                    input_sequence_length=input_sequence_length,
                    cache_capacity=cache_capacity,
                    swa_backing_length=swa_backing_length,
                    moe_fast_mode=moe_fast_mode,
                )
            elif layer_type == "heavily_compressed_attention":
                compressor = source.self_attn.compressor
                cache_capacity = int(max_context_length) // int(compressor.compress_rate)
                layer = StaticHCADecoderLayer.from_hf(
                    source,
                    input_sequence_length=input_sequence_length,
                    cache_capacity=cache_capacity,
                    swa_backing_length=swa_backing_length,
                    moe_fast_mode=moe_fast_mode,
                )
            else:
                raise ValueError(f"unsupported attention layer type: {layer_type}")
            layers.append(layer)

        return cls(
            layers,
            tuple(layer_types),
            hc_mult=int(source_layers[0].attn_hc.hc_mult),
        )

    def initial_cache_inputs(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        cache_dtype: torch.dtype = torch.float16,
        state_dtype: torch.dtype = torch.float16,
    ) -> tuple[LayerCacheInputs, ...]:
        """Allocate the exact heterogeneous cache inputs for this graph.

        Export defaults to fp16 because the target graph represents invalid
        compressor gates with the finite fp16 sentinel ``-65504``.  Tests may
        request fp32 explicitly when comparing against the BF16 HF reference.
        """

        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        caches: list[LayerCacheInputs] = []
        for layer_type, layer in zip(self.layer_types, self.layers, strict=True):
            attention = layer.attention
            swa_length = attention.swa_update.backing_length
            swa_shape = (batch_size, 1, swa_length, attention.head_dim)
            swa_k = torch.zeros(swa_shape, device=device, dtype=cache_dtype)
            swa_v = torch.zeros(swa_shape, device=device, dtype=cache_dtype)
            if layer_type == "sliding_attention":
                caches.append(SWACacheInputs(swa_k, swa_v))
                continue

            main_shape = (batch_size, 1, attention.cache_capacity, attention.head_dim)
            main = torch.zeros(main_shape, device=device, dtype=cache_dtype)
            states = attention.initial_state(batch_size, device=device, dtype=state_dtype)
            if layer_type == "compressed_sparse_attention":
                index_shape = (
                    batch_size,
                    1,
                    attention.cache_capacity,
                    attention.indexer.head_dim,
                )
                caches.append(
                    CSACacheInputs(
                        swa_k,
                        swa_v,
                        main,
                        torch.zeros(index_shape, device=device, dtype=cache_dtype),
                        *states,
                    )
                )
            else:
                caches.append(
                    HCACacheInputs(
                        swa_k,
                        swa_v,
                        main,
                        *states,
                    )
                )
        return tuple(caches)

    def initial_flat_cache_inputs(self, *args, **kwargs) -> FlatCacheInputs:
        """Allocate caches directly in the flattened exported-ABI order."""

        return flatten_cache_inputs(self.initial_cache_inputs(*args, **kwargs))

    @property
    def flat_cache_input_count(self) -> int:
        return sum(len(_CACHE_INPUT_TYPE[layer_type]._fields) for layer_type in self.layer_types)

    def normalize_cache_inputs(
        self,
        caches: FlatCacheInputs | tuple[LayerCacheInputs, ...],
    ) -> tuple[LayerCacheInputs, ...]:
        """Accept eager layer tuples or rebuild them from the flat graph ABI.

        TorchFX sees ``caches`` as one Proxy.  Indexing that Proxy with the
        statically known field offsets lets xhquant's input-list expansion
        materialize one graph input per physical cache/state tensor.  Eager
        callers may retain the more convenient tuple-per-layer form.
        """

        if not is_fx_proxy(caches):
            expected_types = tuple(_CACHE_INPUT_TYPE[layer_type] for layer_type in self.layer_types)
            if len(caches) == len(expected_types) and all(
                isinstance(cache, expected_type) for cache, expected_type in zip(caches, expected_types, strict=True)
            ):
                return tuple(caches)
            if len(caches) != self.flat_cache_input_count:
                raise ValueError(f"expected {self.flat_cache_input_count} flat cache tensors, got {len(caches)}")

        rebuilt: list[LayerCacheInputs] = []
        cursor = 0
        for layer_type in self.layer_types:
            cache_type = _CACHE_INPUT_TYPE[layer_type]
            width = len(cache_type._fields)
            rebuilt.append(cache_type(*(caches[cursor + index] for index in range(width))))
            cursor += width
        return tuple(rebuilt)

    @staticmethod
    def _csa_state(output) -> CSAStateOutput | tuple[Tensor, ...]:
        attention = output.attention
        values = (
            attention.main_next_kv_state,
            attention.main_next_score_state,
            attention.index_next_kv_state,
            attention.index_next_score_state,
        )
        return values if is_fx_proxy(values[0]) else CSAStateOutput(*values)

    @staticmethod
    def _hca_state(output) -> HCAStateOutput | tuple[Tensor, ...]:
        attention = output.attention
        values = (
            attention.main_next_kv_state,
            attention.main_next_score_state,
        )
        return values if is_fx_proxy(values[0]) else HCAStateOutput(*values)

    def forward(
        self,
        inputs_embeds: Tensor,
        input_ids: Tensor,
        past_length: Tensor,
        current_length: Tensor,
        csa_write_start: Tensor,
        hca_write_start: Tensor,
        swa_attention_mask: Tensor,
        csa_index_validity: Tensor,
        csa_attention_mask: Tensor,
        hca_attention_mask: Tensor,
        csa_compressor_validity: Tensor,
        csa_compressor_new_count: Tensor,
        csa_compressor_offset: Tensor,
        csa_compressor_phase_indices: Tensor,
        hca_compressor_validity: Tensor,
        hca_compressor_new_count: Tensor,
        hca_compressor_offset: Tensor,
        hca_compressor_phase_indices: Tensor,
        caches: FlatCacheInputs | tuple[LayerCacheInputs, ...],
        main_cos: Tensor,
        main_sin: Tensor,
        compress_cos: Tensor,
        compress_sin: Tensor,
        csa_compressed_cos: Tensor,
        csa_compressed_sin: Tensor,
        hca_compressed_cos: Tensor,
        hca_compressed_sin: Tensor,
    ) -> StaticBlocksOutput:
        caches = self.normalize_cache_inputs(caches)
        if not is_fx_proxy(inputs_embeds) and inputs_embeds.ndim != 3:
            raise ValueError("inputs_embeds must be [B,P,D]")
        if not is_fx_proxy(caches) and len(caches) != len(self.layers):
            raise ValueError(f"expected {len(self.layers)} layer caches, got {len(caches)}")

        hidden_states = inputs_embeds.unsqueeze(2).expand(-1, -1, self.hc_mult, -1)
        csa_states: list[CSAStateOutput] = []
        hca_states: list[HCAStateOutput] = []
        for layer_index, (layer_type, layer) in enumerate(zip(self.layer_types, self.layers, strict=True)):
            cache = caches[layer_index]
            if layer_type == "sliding_attention":
                if not is_fx_proxy(cache) and not isinstance(cache, SWACacheInputs):
                    raise TypeError("sliding layer requires SWACacheInputs")
                output = layer(
                    hidden_states,
                    input_ids,
                    past_length,
                    current_length,
                    swa_attention_mask,
                    cache[0],
                    cache[1],
                    main_cos,
                    main_sin,
                )
            elif layer_type == "compressed_sparse_attention":
                if not is_fx_proxy(cache) and not isinstance(cache, CSACacheInputs):
                    raise TypeError("CSA layer requires CSACacheInputs")
                output = layer(
                    hidden_states,
                    input_ids,
                    past_length,
                    current_length,
                    csa_index_validity,
                    csa_attention_mask,
                    cache[0],
                    cache[1],
                    cache[2],
                    cache[3],
                    csa_write_start,
                    csa_compressor_validity,
                    csa_compressor_new_count,
                    csa_compressor_offset,
                    csa_compressor_phase_indices,
                    cache[4],
                    cache[5],
                    cache[6],
                    cache[7],
                    compress_cos,
                    compress_sin,
                    csa_compressed_cos,
                    csa_compressed_sin,
                )
                csa_states.append(self._csa_state(output))
            else:
                if not is_fx_proxy(cache) and not isinstance(cache, HCACacheInputs):
                    raise TypeError("HCA layer requires HCACacheInputs")
                output = layer(
                    hidden_states,
                    input_ids,
                    past_length,
                    current_length,
                    hca_attention_mask,
                    cache[0],
                    cache[1],
                    cache[2],
                    hca_write_start,
                    hca_compressor_validity,
                    hca_compressor_new_count,
                    hca_compressor_offset,
                    hca_compressor_phase_indices,
                    cache[3],
                    cache[4],
                    compress_cos,
                    compress_sin,
                    hca_compressed_cos,
                    hca_compressed_sin,
                )
                hca_states.append(self._hca_state(output))
            hidden_states = self.layer_tags[layer_index](output.hidden_states)

        values = (hidden_states, tuple(csa_states), tuple(hca_states))
        return values if is_fx_proxy(hidden_states) else StaticBlocksOutput(*values)


class StaticDeepSeekV4ForCausalLM(nn.Module):
    """Static decoder prefix plus the checkpoint's HC head, norm and LM head."""

    def __init__(
        self,
        blocks: StaticDeepSeekV4Blocks,
        rotary_emb: nn.Module,
        hc_head: nn.Module,
        norm: nn.Module,
        lm_head: nn.Module,
        *,
        csa_ratio: int,
        hca_ratio: int,
        max_context_length: int,
    ) -> None:
        super().__init__()
        self.blocks = blocks
        self.hc_head = hc_head
        self.norm = norm
        self.lm_head = lm_head
        self.last_token_gather = xhnn.Gather(axis=1)
        self.csa_ratio = int(csa_ratio)
        self.hca_ratio = int(hca_ratio)
        self.max_context_length = int(max_context_length)
        self.input_sequence_length = int(blocks.layers[0].attention.input_sequence_length)
        self.csa_outputs = (self.input_sequence_length + self.csa_ratio - 1) // self.csa_ratio
        self.hca_outputs = (self.input_sequence_length + self.hca_ratio - 1) // self.hca_ratio

        # A short final chunk is padded to P but still traverses the fixed
        # graph. Cache through max_context+P-1 so those masked rows can always
        # be sliced without graph-side bounds logic. CSA/HCA use pre-sampled
        # logical tables, turning their strided absolute positions into the
        # same contiguous DynamicSlice contract.
        device = rotary_emb.main_inv_freq.device
        raw_length = self.max_context_length + self.input_sequence_length - 1
        raw_positions = torch.arange(raw_length, device=device)
        main_cos, main_sin = _build_rope_cache(
            rotary_emb.main_inv_freq,
            rotary_emb.main_attention_scaling,
            raw_positions,
        )
        compress_cos, compress_sin = _build_rope_cache(
            rotary_emb.compress_inv_freq,
            rotary_emb.compress_attention_scaling,
            raw_positions,
        )
        csa_length = self.max_context_length // self.csa_ratio + self.csa_outputs
        hca_length = self.max_context_length // self.hca_ratio + self.hca_outputs
        csa_cos, csa_sin = _build_rope_cache(
            rotary_emb.compress_inv_freq,
            rotary_emb.compress_attention_scaling,
            torch.arange(csa_length, device=device) * self.csa_ratio,
        )
        hca_cos, hca_sin = _build_rope_cache(
            rotary_emb.compress_inv_freq,
            rotary_emb.compress_attention_scaling,
            torch.arange(hca_length, device=device) * self.hca_ratio,
        )
        for name, value in (
            ("main_cos_cached", main_cos),
            ("main_sin_cached", main_sin),
            ("compress_cos_cached", compress_cos),
            ("compress_sin_cached", compress_sin),
            ("csa_cos_cached", csa_cos),
            ("csa_sin_cached", csa_sin),
            ("hca_cos_cached", hca_cos),
            ("hca_sin_cached", hca_sin),
        ):
            self.register_buffer(name, value, persistent=False)
        self.raw_rope_slice = xhnn.DynamicSlice(self.input_sequence_length, 1, 1)
        self.csa_rope_slice = xhnn.DynamicSlice(self.csa_outputs, 1, 1)
        self.hca_rope_slice = xhnn.DynamicSlice(self.hca_outputs, 1, 1)

    @classmethod
    def from_hf(
        cls,
        model: nn.Module,
        *,
        input_sequence_length: int,
        max_context_length: int = 256 * 1024,
        max_layers: int | None = None,
        moe_fast_mode: bool = True,
        swa_backing_length: int | None = None,
    ) -> "StaticDeepSeekV4ForCausalLM":
        if not hasattr(model, "model") or not hasattr(model, "lm_head"):
            raise ValueError("model must be a DeepseekV4ForCausalLM-like module")
        blocks = StaticDeepSeekV4Blocks.from_hf(
            model,
            input_sequence_length=input_sequence_length,
            max_context_length=max_context_length,
            max_layers=max_layers,
            moe_fast_mode=moe_fast_mode,
            swa_backing_length=swa_backing_length,
        )
        return cls(
            blocks,
            model.model.rotary_emb,
            BoundHCHead.from_hf(model.model.hc_head),
            rms_norm_from_hf(model.model.norm),
            model.lm_head,
            csa_ratio=int(model.config.compress_rates["compressed_sparse_attention"]),
            hca_ratio=int(model.config.compress_rates["heavily_compressed_attention"]),
            max_context_length=int(max_context_length),
        )

    def initial_cache_inputs(self, *args, **kwargs) -> tuple[LayerCacheInputs, ...]:
        return self.blocks.initial_cache_inputs(*args, **kwargs)

    def initial_flat_cache_inputs(self, *args, **kwargs) -> FlatCacheInputs:
        return flatten_cache_inputs(self.initial_cache_inputs(*args, **kwargs))

    def _rope(
        self,
        past_length: Tensor,
        csa_write_start: Tensor,
        hca_write_start: Tensor,
    ) -> tuple[Tensor, ...]:
        main_cos = self.raw_rope_slice(self.main_cos_cached, past_length)
        main_sin = self.raw_rope_slice(self.main_sin_cached, past_length)
        compress_cos = self.raw_rope_slice(self.compress_cos_cached, past_length)
        compress_sin = self.raw_rope_slice(self.compress_sin_cached, past_length)
        csa_cos = self.csa_rope_slice(self.csa_cos_cached, csa_write_start)
        csa_sin = self.csa_rope_slice(self.csa_sin_cached, csa_write_start)
        hca_cos = self.hca_rope_slice(self.hca_cos_cached, hca_write_start)
        hca_sin = self.hca_rope_slice(self.hca_sin_cached, hca_write_start)
        return (
            main_cos,
            main_sin,
            compress_cos,
            compress_sin,
            csa_cos,
            csa_sin,
            hca_cos,
            hca_sin,
        )

    def forward(
        self,
        inputs_embeds: Tensor,
        input_ids: Tensor,
        past_length: Tensor,
        current_length: Tensor,
        last_token_index: Tensor,
        csa_write_start: Tensor,
        hca_write_start: Tensor,
        swa_attention_mask: Tensor,
        csa_index_validity: Tensor,
        csa_attention_mask: Tensor,
        hca_attention_mask: Tensor,
        csa_compressor_validity: Tensor,
        csa_compressor_new_count: Tensor,
        csa_compressor_offset: Tensor,
        csa_compressor_phase_indices: Tensor,
        hca_compressor_validity: Tensor,
        hca_compressor_new_count: Tensor,
        hca_compressor_offset: Tensor,
        hca_compressor_phase_indices: Tensor,
        caches: FlatCacheInputs | tuple[LayerCacheInputs, ...],
    ) -> StaticCausalLMOutput:
        rope = self._rope(
            past_length,
            csa_write_start,
            hca_write_start,
        )
        output = self.blocks(
            inputs_embeds,
            input_ids,
            past_length,
            current_length,
            csa_write_start,
            hca_write_start,
            swa_attention_mask,
            csa_index_validity,
            csa_attention_mask,
            hca_attention_mask,
            csa_compressor_validity,
            csa_compressor_new_count,
            csa_compressor_offset,
            csa_compressor_phase_indices,
            hca_compressor_validity,
            hca_compressor_new_count,
            hca_compressor_offset,
            hca_compressor_phase_indices,
            caches,
            *rope,
        )
        hidden_states = self.norm(self.hc_head(output[0]))
        # Host already knows the valid length.  Supplying the final-token index
        # directly avoids rebuilding a coordinate with Sub in the static graph.
        token_index = one_element_int(last_token_index, name="last_token_index")
        last_hidden = self.last_token_gather(hidden_states, token_index)
        logits = self.lm_head(last_hidden)
        if is_fx_proxy(logits):
            return (
                logits,
                *(value for state in output[1] for value in state),
                *(value for state in output[2] for value in state),
            )
        return StaticCausalLMOutput(logits, output[1], output[2])


__all__ = [
    "FlatCacheInputs",
    "LayerCacheInputs",
    "StaticBlocksOutput",
    "StaticCausalLMOutput",
    "StaticDeepSeekV4Blocks",
    "StaticDeepSeekV4ForCausalLM",
    "flatten_cache_inputs",
]
