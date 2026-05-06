# Copyright 2025 HOUMO AI
#
# File: _model_impl.py
# Description:
#   Model implementation for HOUMO AI xh2modelzoo.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""
FM9G model wrap implementations for HMONNX export.

FM9G is an LLM model whose code is not in the transformers library.
It is loaded via trust_remote_code=True, similar to cogvlm2.
The architecture is structurally similar to Qwen2 (RMSNorm, GQA, SwiGLU MLP, RoPE),
but with scale_depth and scale_emb differences.

Since the model is loaded dynamically, we use DynamicRegister.register() to register
the actual module classes at runtime, rather than string-based static registration.
"""

import math
import sys
import types
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from transformers.modeling_outputs import BaseModelOutputWithPast
from xhquant import nn as xhnn
from xhquant.nn import LLMCache, MaskedSoftmax, RMSNorm, Rope

from ..builder import DynamicRegister


# ---------------------------------------------------------------------------
# Helper: RotaryEmbedding for precomputing cos/sin cache
# ---------------------------------------------------------------------------
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, device=device).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len_cached = 0

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len, device=x.device, dtype=x.dtype)
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


class LongRoPEEmbedding(RotaryEmbedding):
    """FM9G LongRoPE with short/long factor support."""

    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        short_factor=None,
        long_factor=None,
        original_max_position_embeddings=None,
    ):
        self.short_factor = short_factor
        self.long_factor = long_factor
        self.original_max_position_embeddings = original_max_position_embeddings
        scale = max_position_embeddings / self.original_max_position_embeddings
        self.scaling_factor = math.sqrt(1 + math.log(scale) / math.log(self.original_max_position_embeddings))
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        if seq_len > self.original_max_position_embeddings:
            ext_factors = torch.tensor(self.long_factor, dtype=torch.float32, device=device)
        else:
            ext_factors = torch.tensor(self.short_factor, dtype=torch.float32, device=device)
        freqs = torch.mul(
            torch.outer(t, 1.0 / ext_factors).to(device=device),
            self.inv_freq.to(device=device).to(dtype),
        )
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype) * self.scaling_factor, persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype) * self.scaling_factor, persistent=False)


# ---------------------------------------------------------------------------
# Wrap classes for FM9G modules.
# Registration happens dynamically in register_wrap_cls() since module classes
# are only available at runtime after trust_remote_code model loading.
# ---------------------------------------------------------------------------


class _FM9GRMSNorm(DynamicRegister):
    def forward(self, hidden_states):
        return self.norm(hidden_states)

    def _setup(self, cfg: Optional[Dict] = None):
        hidden_size = self.weight.shape[0]
        self.norm = RMSNorm(hidden_size, self.variance_epsilon)
        self.norm.weight = self.weight
        return self


class _FM9GMLP(DynamicRegister):
    """FM9G MLP wrap - no changes needed, just pass through."""

    def _setup(self, cfg: Optional[Dict] = None):
        return self


class _FM9GAttention(DynamicRegister):
    def rotate_half(self, x: Tensor):
        """Rotates half the hidden dims of the input."""
        x1 = self.slice_1(x)
        x2 = self.slice_2(x)
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(
        self,
        q: Tensor,
        k: Tensor,
        cos: Tensor,
        sin: Tensor,
    ):
        if self.enable_rope:
            q_embed = self.rope(q, cos, sin)
            k_embed = self.rope(k, cos, sin)
        else:
            q_embed = (q * cos) + (self.rotate_half(q) * sin)
            k_embed = (k * cos) + (self.rotate_half(k) * sin)
        return q_embed, k_embed

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_seq_length: Optional[Tensor] = None,
        current_input_length: Optional[Tensor] = None,
        past_k_cache: Optional[Tensor] = None,
        past_v_cache: Optional[Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = self.apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if self.use_cache:
            key_states = self.k_cache(key_states, past_seq_length, current_input_length, past_k_cache)
            value_states = self.v_cache(value_states, past_seq_length, current_input_length, past_v_cache)

        # GQA: repeat_interleave for key/value
        query_states = query_states * self.kv_scale
        key_states = key_states.transpose(2, 3)
        key_states = torch.repeat_interleave(key_states, self.num_key_value_groups, dim=1)

        attn_weights = torch.matmul(query_states, key_states)
        attn_weights: Optional[Tensor] = self.masked_softmax(attn_weights, past_seq_length)

        value_states = torch.repeat_interleave(value_states, self.num_key_value_groups, dim=1)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        return attn_output, None, None

    def _setup(self, cfg: Union[Dict, Any]):
        if not hasattr(self, "num_key_value_heads"):
            self.num_key_value_heads = self.config.num_key_value_heads
        if not hasattr(self, "num_heads"):
            self.num_heads = self.config.num_attention_heads
        if not hasattr(self, "hidden_size"):
            self.hidden_size = self.config.hidden_size

        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        self.enable_rope = cfg.get("enable_rope", True) if hasattr(cfg, "get") else True
        if self.enable_rope:
            self.rope = Rope()

        self.slice_1 = xhnn.Slice([0], [self.head_dim // 2], [3], [1])
        self.slice_2 = xhnn.Slice([self.head_dim // 2], [sys.maxsize], [3], [1])
        self.masked_softmax = MaskedSoftmax(dim=-1)

        use_cache = cfg.use_cache if hasattr(cfg, "use_cache") else cfg.get("use_cache", True)
        self.use_cache = use_cache

        if use_cache:
            cache_axis = cfg.kv_cache.cache_axis
            self.k_cache = LLMCache(axis=cache_axis)
            self.v_cache = LLMCache(axis=cache_axis)
        else:
            self.k_cache = None
            self.v_cache = None

        _kv_scale = 1 / math.sqrt(self.head_dim)
        self.kv_scale = _kv_scale

        return self


class _FM9GDecoderLayer(DynamicRegister):
    def forward(
        self,
        hidden_states: torch.Tensor,
        past_seq_length: Optional[Tensor] = None,
        current_input_length: Optional[Tensor] = None,
        past_k_cache: Optional[Tensor] = None,
        past_v_cache: Optional[Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_seq_length=past_seq_length,
            current_input_length=current_input_length,
            past_k_cache=past_k_cache,
            past_v_cache=past_v_cache,
            position_embeddings=position_embeddings,
        )

        # FM9G specific: scale_depth / sqrt(num_hidden_layers)
        hidden_states = residual + hidden_states * self._residual_scale

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states * self._residual_scale

        outputs = (hidden_states,)
        return outputs

    def _setup(self, cfg: Optional[Dict] = None):
        # FM9G specific residual scaling
        self._residual_scale = self.scale_depth / math.sqrt(self.num_hidden_layers)
        return self


class _FM9GModel(DynamicRegister):
    def forward(
        self,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_seq_length: Optional[Tensor] = None,
        current_input_length: Optional[Tensor] = None,
        past_key_cache: Optional[List[Tensor]] = None,
        past_value_cache: Optional[List[Tensor]] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        hidden_states = inputs_embeds
        hidden_states = hidden_states * self._scale_emb
        cos = self.cos_slice(self.rotary_emb.cos_cached, past_seq_length)
        sin = self.sin_slice(self.rotary_emb.sin_cached, past_seq_length)
        position_embeddings = (cos, sin)

        for idx, decoder_layer in enumerate(self.layers):
            if self.use_cache:
                _past_k_cache = past_key_cache[idx]
                _past_v_cache = past_value_cache[idx]
            else:
                _past_k_cache = None
                _past_v_cache = None

            layer_outputs = decoder_layer(
                hidden_states,
                past_seq_length=past_seq_length,
                current_input_length=current_input_length,
                past_k_cache=_past_k_cache,
                past_v_cache=_past_v_cache,
                position_embeddings=position_embeddings,
            )
            hidden_states = layer_outputs[0]
            if self.only_first_block:
                break

        if self.num_logits_to_keep == 0:
            pass
        else:
            # Take last token output
            hidden_states = self.llm_gather(hidden_states, current_input_length - 1)
        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
        )

    def _setup(self, cfg: Any):
        # FM9G specific: scale_emb factor applied to input embeddings
        # self._scale_emb = getattr(self.config, "scale_emb", 1)
        self._scale_emb = float(self.config.scale_emb)
        self.only_first_block = cfg.get("only_first_block", False)
        self.num_logits_to_keep = cfg.num_logits_to_keep
        assert self.num_logits_to_keep in [0, 1]

        input_seq_len = cfg.input_sequence_length
        self.slice = xhnn.Slice([0], [input_seq_len], [1], [1])

        self.llm_gather = xhnn.BatchGather(1)
        self.llm_gather.update_offset_indices(1, input_seq_len)

        def _llm_gather_update_cfg(self_gather, cfg=None):
            input_seq_len = cfg.input_sequence_length
            batch_size = cfg.get("batch_size", 1)
            self_gather.update_offset_indices(batch_size, input_seq_len)

        self.llm_gather._update_cfg = types.MethodType(_llm_gather_update_cfg, self.llm_gather)

        def _update_cfg(self_slice, cfg=None):
            input_seq_len = cfg.input_sequence_length
            self_slice.ends = [input_seq_len]

        self.slice._update_cfg = types.MethodType(_update_cfg, self.slice)
        self.use_cache = cfg.use_cache

        # Build RoPE cache
        max_sequence_length = cfg.max_sequence_length
        hidden_size = self.config.hidden_size
        num_attention_heads = self.config.num_attention_heads
        head_dim = hidden_size // num_attention_heads
        rope_theta = getattr(self.config, "rope_theta", 10000.0)
        rope_scaling = getattr(self.config, "rope_scaling", None)

        if rope_scaling is not None:
            scaling_type = rope_scaling.get("rope_type", rope_scaling.get("type", None))
            if scaling_type == "longrope":
                self.rotary_emb = LongRoPEEmbedding(
                    dim=head_dim,
                    max_position_embeddings=max_sequence_length,
                    base=rope_theta,
                    short_factor=rope_scaling["short_factor"],
                    long_factor=rope_scaling["long_factor"],
                    original_max_position_embeddings=rope_scaling["original_max_position_embeddings"],
                )
            else:
                self.rotary_emb = RotaryEmbedding(
                    dim=head_dim,
                    max_position_embeddings=max_sequence_length,
                    base=rope_theta,
                )
        else:
            self.rotary_emb = RotaryEmbedding(
                dim=head_dim,
                max_position_embeddings=max_sequence_length,
                base=rope_theta,
            )

        self.rotary_emb._set_cos_sin_cache(seq_len=max_sequence_length, device=torch.device("cpu"), dtype=torch.float32)

        # cos_cached shape: [max_seq_len, head_dim]
        # We need shape [1, 1, max_seq_len, head_dim] for DynamicSlice
        cos = self.rotary_emb.cos_cached
        sin = self.rotary_emb.sin_cached
        # Reshape to [1, 1, seq_len, head_dim] for broadcasting with [bsz, num_heads, seq_len, head_dim]
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
        self.rotary_emb.cos_cached = cos
        self.rotary_emb.sin_cached = sin

        self.sin_slice = xhnn.DynamicSlice([input_seq_len], [2], [1])
        self.cos_slice = xhnn.DynamicSlice([input_seq_len], [2], [1])

        def _sin_cos_slice_update_cfg(self_ds, cfg=None):
            input_seq_len = cfg.input_sequence_length
            self_ds.valid_length = [input_seq_len]

        self.sin_slice._update_cfg = types.MethodType(_sin_cos_slice_update_cfg, self.sin_slice)
        self.cos_slice._update_cfg = types.MethodType(_sin_cos_slice_update_cfg, self.cos_slice)

        return self


class _FM9GForCausalLM(DynamicRegister):
    def forward(
        self,
        inputs_embeds: Optional[Tensor] = None,
        past_seq_length: Optional[Tensor] = None,
        current_input_length: Optional[Tensor] = None,
        past_key_cache: Optional[List[Tensor]] = None,
        past_value_cache: Optional[List[Tensor]] = None,
    ):
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            past_seq_length=past_seq_length,
            current_input_length=current_input_length,
            past_key_cache=past_key_cache,
            past_value_cache=past_value_cache,
        )

        hidden_states = outputs.last_hidden_state
        # FM9G specific: lm_head scaling by hidden_size / dim_model_base
        logits = self.lm_head(hidden_states * self._lm_head_scale)
        return logits

    def _setup(self, cfg: Optional[Dict] = None):
        # FM9G specific scaling: logits = lm_head(hidden / (hidden_size / dim_model_base))
        # Original: self.lm_head(hidden_states / (hidden_size / dim_model_base))
        # Rewritten as: self.lm_head(hidden_states * (dim_model_base / hidden_size))
        hidden_size = self.config.hidden_size
        dim_model_base = getattr(self.config, "dim_model_base", 1)
        self._lm_head_scale = dim_model_base / hidden_size
        return self


def register_wrap_cls(hf_model):
    """Register FM9G model classes for wrapping.

    Since FM9G model code is loaded via trust_remote_code, the module classes
    live under `transformers_modules.<model_dir>.modeling_fm9g` and their
    fully-qualified names depend on the model directory. We use dynamic
    registration with actual class objects to avoid hardcoding paths.
    """
    if hf_model is None:
        return
    # FM9GForCausalLM
    _FM9GForCausalLM.register(type(hf_model))
    # FM9GModel
    _FM9GModel.register(type(hf_model.model))
    # FM9GDecoderLayer
    _FM9GDecoderLayer.register(type(hf_model.model.layers[0]))
    # FM9GAttention (could be FM9GAttention, FM9GFlashAttention2, or FM9GSdpaAttention)
    _FM9GAttention.register(type(hf_model.model.layers[0].self_attn))
    # FM9GRMSNorm
    _FM9GRMSNorm.register(type(hf_model.model.layers[0].input_layernorm))
    # FM9GMLP
    _FM9GMLP.register(type(hf_model.model.layers[0].mlp))
    # Final norm
    _FM9GRMSNorm.register(type(hf_model.model.norm))
