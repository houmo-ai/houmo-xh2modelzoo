# Copyright (C) 2026 THL A29 Limited, a Tencent company and the HuggingFace Inc. team.
# Copyright 2025 HOUMO AI (modifications)
# Licensed under the Apache License, Version 2.0.
# See licenses/transformers/LICENSE for the full license text.
# Source: https://github.com/huggingface/transformers/blob/v5.13.0/src/transformers/models/hunyuan_vl/modular_hunyuan_vl.py

"""Traceable text modules for the HunyuanOCR-1.5 target model."""

from __future__ import annotations

import math
import types
from copy import deepcopy
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor
from transformers.models.hunyuan_vl.modeling_hunyuan_vl import (
    HunYuanVLDenseV1Attention,
    HunYuanVLDenseV1DecoderLayer,
    HunYuanVLForConditionalGeneration,
    HunYuanVLMLP,
    HunYuanVLModel,
    HunYuanVLRMSNorm,
    HunYuanVLRotaryEmbedding,
    HunYuanVLTextModel,
)

from xhquant import nn as xhnn
from xhquant.nn import LLMCacheV2, MaskedSoftmax, RMSNorm, Rope
from xhquant.utils.registry import DynamicModule

from ...register import XHLLM_TRACEABLE_MODULES


HUNYUAN_OCR_15_MROPE_SECTION = (16, 16, 16, 16)


class HunyuanOCRTextOneLayer(nn.Module):
    """Small validation seam joining one rotary module and one decoder layer."""

    def __init__(self, rotary_emb: HunYuanVLRotaryEmbedding, decoder_layer: HunYuanVLDenseV1DecoderLayer) -> None:
        super().__init__()
        self.rotary_emb = rotary_emb
        self.decoder_layer = decoder_layer

    def forward(self, hidden_states: Tensor, position_ids: Tensor, past_seq_length: Tensor) -> Tensor:
        return self.graph_forward(hidden_states, position_ids, past_seq_length)

    def graph_forward(self, hidden_states: Tensor, position_ids: Tensor, past_seq_length: Tensor) -> Tensor:
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        return self.decoder_layer(
            hidden_states,
            past_seq_length=past_seq_length,
            position_embeddings=position_embeddings,
        )


@XHLLM_TRACEABLE_MODULES.register_module({HunYuanVLRMSNorm: "HunYuanVLRMSNorm"})
class _HunYuanOCRRMSNorm(DynamicModule):
    def _setup(self, cfg: Any = None):
        hidden_size = self.weight.shape[0]
        self.norm = RMSNorm(hidden_size, self.variance_epsilon).to(
            device=self.weight.device,
            dtype=self.weight.dtype,
        )
        self.norm.weight = nn.Parameter(deepcopy(self.weight.data))
        return self

    def graph_forward(self, hidden_states: Tensor) -> Tensor:
        return self.norm(hidden_states)


@XHLLM_TRACEABLE_MODULES.register_module({HunYuanVLRotaryEmbedding: "HunYuanVLRotaryEmbedding"})
class _HunYuanOCRRotaryEmbedding(DynamicModule):
    def _setup(self, cfg: Any):
        section = tuple(int(value) for value in (self.mrope_section or ()))
        if section != HUNYUAN_OCR_15_MROPE_SECTION:
            raise ValueError("HunyuanOCR-1.5 requires four-axis mrope_section=(16, 16, 16, 16), " f"got {section!r}")

        max_pe_length = int(cfg.get("max_pe_length", cfg.get("context_max_length", self.max_seq_len_cached)))
        if max_pe_length <= 0:
            raise ValueError(f"max_pe_length must be positive, got {max_pe_length}")

        positions = torch.arange(max_pe_length, device=self.inv_freq.device, dtype=torch.float32)
        frequencies = torch.outer(positions, self.inv_freq.float())
        embeddings = torch.cat((frequencies, frequencies), dim=-1)
        cache_dtype = self.inv_freq.dtype
        self.register_buffer(
            "cos_cached",
            (embeddings.cos() * self.attention_scaling).to(cache_dtype),
            persistent=False,
        )
        self.register_buffer(
            "sin_cached",
            (embeddings.sin() * self.attention_scaling).to(cache_dtype),
            persistent=False,
        )
        self.max_seq_len_cached = max_pe_length
        return self

    def graph_forward(self, hidden_states: Tensor, position_ids: Tensor) -> tuple[Tensor, Tensor]:
        cos = self.cos_cached[position_ids].to(dtype=hidden_states.dtype)
        sin = self.sin_cached[position_ids].to(dtype=hidden_states.dtype)
        return cos, sin


@XHLLM_TRACEABLE_MODULES.register_module({HunYuanVLMLP: "HunYuanVLMLP"})
class _HunYuanOCRMLP(DynamicModule):
    def _setup(self, cfg: Any = None):
        if self.hidden_size != 1024 or self.intermediate_size != 3584:
            raise ValueError(
                "HunyuanOCR-1.5 MLP expects hidden_size=1024 and intermediate_size=3584, "
                f"got hidden_size={self.hidden_size}, intermediate_size={self.intermediate_size}"
            )
        return self

    def graph_forward(self, hidden_states: Tensor) -> Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


@XHLLM_TRACEABLE_MODULES.register_module({HunYuanVLDenseV1Attention: "HunYuanVLDenseV1Attention"})
class _HunYuanOCRAttention(DynamicModule):
    def _setup(self, cfg: Any):
        self.num_heads = int(self.config.num_attention_heads)
        self.num_key_value_heads = int(self.config.num_key_value_heads)
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = int(self.head_dim)

        expected = (self.config.hidden_size, self.num_heads, self.num_key_value_heads, self.head_dim)
        if expected != (1024, 16, 8, 128):
            raise ValueError(
                "HunyuanOCR-1.5 attention expects (hidden, query heads, KV heads, head dim)="
                f"(1024, 16, 8, 128), got {expected}"
            )
        section = tuple(int(value) for value in (self.mrope_section or ()))
        if section != HUNYUAN_OCR_15_MROPE_SECTION:
            raise ValueError("HunyuanOCR-1.5 requires four-axis mrope_section=(16, 16, 16, 16), " f"got {section!r}")

        self.rope = Rope()
        self.masked_softmax = MaskedSoftmax(dim=-1)
        self.use_cache = bool(cfg.get("use_cache", False))
        if self.use_cache:
            cache_axis = int(cfg.kv_cache.cache_axis)
            self.k_cache = LLMCacheV2(axis=cache_axis)
            self.v_cache = LLMCacheV2(axis=cache_axis)
        else:
            self.k_cache = None
            self.v_cache = None
        self.register_buffer(
            "attention_scale",
            torch.tensor(1.0 / math.sqrt(self.head_dim), device=self.q_proj.weight.device),
            persistent=False,
        )
        return self

    @staticmethod
    def _compose_xdrope_axis(tensor: Tensor) -> Tensor:
        return torch.cat(
            (
                tensor[0, ..., 0:32],
                tensor[1, ..., 32:64],
                tensor[2, ..., 64:96],
                tensor[3, ..., 96:128],
            ),
            dim=-1,
        ).unsqueeze(1)

    def graph_forward(
        self,
        hidden_states: Tensor,
        past_seq_length: Tensor,
        position_embeddings: tuple[Tensor, Tensor],
        current_input_length: Tensor | None = None,
        past_k_cache: Tensor | None = None,
        past_v_cache: Tensor | None = None,
    ) -> Tensor:
        batch_size = hidden_states.shape[0]
        sequence_length = hidden_states.shape[1]

        query_states = self.q_proj(hidden_states).view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim,
        )
        key_states = self.k_proj(hidden_states).view(
            batch_size,
            sequence_length,
            self.num_key_value_heads,
            self.head_dim,
        )
        value_states = self.v_proj(hidden_states).view(
            batch_size,
            sequence_length,
            self.num_key_value_heads,
            self.head_dim,
        )

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        cos, sin = position_embeddings
        cos = self._compose_xdrope_axis(cos)
        sin = self._compose_xdrope_axis(sin)
        query_states = self.rope(query_states, cos, sin)
        key_states = self.rope(key_states, cos, sin)

        # Transformers 5.13 creates these norms unconditionally and applies them after XD-RoPE.
        query_states = self.query_layernorm(query_states)
        key_states = self.key_layernorm(key_states)

        if self.use_cache:
            key_states = self.k_cache(key_states, past_seq_length, current_input_length, past_k_cache)
            value_states = self.v_cache(value_states, past_seq_length, current_input_length, past_v_cache)

        query_states = query_states * self.attention_scale
        key_states = torch.repeat_interleave(key_states, self.num_key_value_groups, dim=1)
        value_states = torch.repeat_interleave(value_states, self.num_key_value_groups, dim=1)

        attention_scores = torch.matmul(query_states, key_states.transpose(2, 3))
        attention_probs = self.masked_softmax(attention_scores, past_seq_length)
        attention_output = torch.matmul(attention_probs, value_states)
        attention_output = attention_output.transpose(1, 2).reshape(
            batch_size,
            sequence_length,
            self.num_heads * self.head_dim,
        )
        return self.o_proj(attention_output)


@XHLLM_TRACEABLE_MODULES.register_module({HunYuanVLDenseV1DecoderLayer: "HunYuanVLDenseV1DecoderLayer"})
class _HunYuanOCRDecoderLayer(DynamicModule):
    def _setup(self, cfg: Any = None):
        return self

    def graph_forward(
        self,
        hidden_states: Tensor,
        past_seq_length: Tensor,
        position_embeddings: tuple[Tensor, Tensor],
        current_input_length: Tensor | None = None,
        past_k_cache: Tensor | None = None,
        past_v_cache: Tensor | None = None,
    ) -> Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states,
            past_seq_length=past_seq_length,
            current_input_length=current_input_length,
            past_k_cache=past_k_cache,
            past_v_cache=past_v_cache,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


@XHLLM_TRACEABLE_MODULES.register_module({HunYuanVLTextModel: "HunYuanVLTextModel"})
class _HunYuanOCRTextModel(DynamicModule):
    def _setup(self, cfg: Any):
        self.use_cache = bool(cfg.get("use_cache", False))
        self.only_first_block = bool(cfg.get("only_first_block", False))
        self.num_logits_to_keep = int(cfg.get("num_logits_to_keep", 1))
        if self.num_logits_to_keep not in (0, 1):
            raise ValueError(f"num_logits_to_keep must be 0 or 1, got {self.num_logits_to_keep}")
        self.output_hidden_state_indices = cfg.get("output_hidden_state_indices")
        self._output_hidden_set = set(self.output_hidden_state_indices or ())

        input_sequence_length = int(cfg.input_sequence_length)
        batch_size = int(cfg.get("batch_size", 1))
        self.output_slice = xhnn.Slice([0], [input_sequence_length], [1], [1])
        self.last_token_gather = xhnn.BatchGather(1)
        self.last_token_gather.update_offset_indices(batch_size, input_sequence_length)

        def _slice_update_cfg(slice_module, new_cfg=None):
            slice_module.ends = [int(new_cfg.input_sequence_length)]

        def _gather_update_cfg(gather_module, new_cfg=None):
            gather_module.update_offset_indices(
                int(new_cfg.get("batch_size", 1)),
                int(new_cfg.input_sequence_length),
            )

        self.output_slice._update_cfg = types.MethodType(_slice_update_cfg, self.output_slice)
        self.last_token_gather._update_cfg = types.MethodType(_gather_update_cfg, self.last_token_gather)
        return self

    def graph_forward(
        self,
        inputs_embeds: Tensor,
        sequence_position_ids: Tensor,
        width_position_ids: Tensor,
        height_position_ids: Tensor,
        image_position_ids: Tensor,
        past_seq_length: Tensor,
        current_input_length: Tensor,
        past_key_cache: list[Tensor] | None = None,
        past_value_cache: list[Tensor] | None = None,
    ) -> Tensor:
        hidden_states = inputs_embeds
        collected_hidden_states: dict[int, Tensor] = {}
        position_ids = torch.stack(
            (
                sequence_position_ids,
                width_position_ids,
                height_position_ids,
                image_position_ids,
            ),
            dim=0,
        )
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for layer_index, decoder_layer in enumerate(self.layers):
            if self.use_cache:
                past_k_cache = past_key_cache[layer_index]
                past_v_cache = past_value_cache[layer_index]
            else:
                past_k_cache = None
                past_v_cache = None
            hidden_states = decoder_layer(
                hidden_states,
                past_seq_length=past_seq_length,
                current_input_length=current_input_length,
                past_k_cache=past_k_cache,
                past_v_cache=past_v_cache,
                position_embeddings=position_embeddings,
            )
            if layer_index in self._output_hidden_set:
                collected_hidden_states[layer_index] = hidden_states
            if self.only_first_block:
                break

        target_hidden = None
        if self.output_hidden_state_indices is not None:
            missing_layer_ids = [
                layer_id for layer_id in self.output_hidden_state_indices if layer_id not in collected_hidden_states
            ]
            if missing_layer_ids:
                raise ValueError(
                    "HunyuanOCR target hidden layers were not executed: "
                    f"missing={missing_layer_ids}, configured={self.output_hidden_state_indices}"
                )
            target_hidden = torch.cat(
                [collected_hidden_states[layer_id] for layer_id in self.output_hidden_state_indices],
                dim=-1,
            )

        if self.num_logits_to_keep == 0:
            hidden_states = self.output_slice(hidden_states)
        else:
            hidden_states = self.last_token_gather(hidden_states, current_input_length - 1)
        hidden_states = self.norm(hidden_states)
        if target_hidden is not None:
            return hidden_states, target_hidden
        return hidden_states


@XHLLM_TRACEABLE_MODULES.register_module({HunYuanVLModel: "HunYuanVLModel"})
class _HunYuanOCRModel(DynamicModule):
    def _setup(self, cfg: Any):
        if "vision_tower" in self._modules:
            del self._modules["vision_tower"]
        return self

    def graph_forward(
        self,
        inputs_embeds: Tensor,
        sequence_position_ids: Tensor,
        width_position_ids: Tensor,
        height_position_ids: Tensor,
        image_position_ids: Tensor,
        past_seq_length: Tensor,
        current_input_length: Tensor,
        past_key_cache: list[Tensor] | None = None,
        past_value_cache: list[Tensor] | None = None,
    ) -> Tensor:
        return self.language_model(
            inputs_embeds=inputs_embeds,
            sequence_position_ids=sequence_position_ids,
            width_position_ids=width_position_ids,
            height_position_ids=height_position_ids,
            image_position_ids=image_position_ids,
            past_seq_length=past_seq_length,
            current_input_length=current_input_length,
            past_key_cache=past_key_cache,
            past_value_cache=past_value_cache,
        )


@XHLLM_TRACEABLE_MODULES.register_module(
    {HunYuanVLForConditionalGeneration: "HunYuanVLForConditionalGeneration"}
)
class _HunYuanOCRForConditionalGeneration(DynamicModule):
    def _setup(self, cfg: Any):
        if hasattr(self, "model") and "vision_tower" in self.model._modules:
            del self.model._modules["vision_tower"]
        return self

    def graph_forward(
        self,
        inputs_embeds: Tensor,
        sequence_position_ids: Tensor,
        width_position_ids: Tensor,
        height_position_ids: Tensor,
        image_position_ids: Tensor,
        past_seq_length: Tensor,
        current_input_length: Tensor,
        past_key_cache: list[Tensor] | None = None,
        past_value_cache: list[Tensor] | None = None,
    ) -> Tensor:
        hidden_states = self.model(
            inputs_embeds=inputs_embeds,
            sequence_position_ids=sequence_position_ids,
            width_position_ids=width_position_ids,
            height_position_ids=height_position_ids,
            image_position_ids=image_position_ids,
            past_seq_length=past_seq_length,
            current_input_length=current_input_length,
            past_key_cache=past_key_cache,
            past_value_cache=past_value_cache,
        )
        if isinstance(hidden_states, tuple):
            normalized_hidden, target_hidden = hidden_states
            return self.lm_head(normalized_hidden), target_hidden
        return self.lm_head(hidden_states)


__all__ = ["HunyuanOCRTextOneLayer"]
