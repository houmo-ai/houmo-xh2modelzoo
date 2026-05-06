# Copyright 2025 HOUMO AI
#
# File: _llm_model_impl.py
# Description:
#   LLM model implementation for HOUMO AI xh2modelzoo.
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

import math
import sys
import types
from copy import deepcopy
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from transformers.modeling_outputs import BaseModelOutputWithPast
from xhquant import nn as xhnn
from xhquant.api import ConfigDict
from xhquant.nn import LLMCacheV2, MaskedSoftmax, RMSNorm
from xhquant.utils.registry import DynamicModule

from ..builder import XHLLM_TRACEABLE_MODULES
from .modeling_glm_ocr import (
    GlmOcrForConditionalGeneration,
    GlmOcrModel,
    GlmOcrRMSNorm,
    GlmOcrTextAttention,
    GlmOcrTextDecoderLayer,
    GlmOcrTextMLP,
    GlmOcrTextModel,
    GlmOcrTextRotaryEmbedding,
)


@XHLLM_TRACEABLE_MODULES.register_module({GlmOcrRMSNorm: "GlmOcrRMSNorm"})
class _GlmOcrRMSNorm(DynamicModule):
    def forward(self, hidden_states):
        return self.norm(hidden_states)

    def _setup(self, cfg: Optional[Dict] = None):
        hidden_size = self.weight.shape[0]
        self.norm = RMSNorm(hidden_size, self.variance_epsilon)
        self.norm.weight = nn.Parameter(deepcopy(self.weight.data))
        return self


@XHLLM_TRACEABLE_MODULES.register_module({GlmOcrTextRotaryEmbedding: "GlmOcrTextRotaryEmbedding"})
class _GlmOcrTextRotaryEmbedding(DynamicModule):
    def _setup(self, cfg: Optional[Dict] = None):
        # Export path only supports static rope.
        assert "dynamic" not in self.rope_type, f"{self.rope_type} is not supported in dynamic mode"

    @torch.no_grad()
    def forward(self, x: Tensor, position_ids: Tensor):
        # Avoid expand with symbolic shape to keep torch.fx tracing stable.
        # Keep RoPE core in fp32, aligned with modeling_glm_ocr.py and glm4v trace path.
        inv_freq_expanded = self.inv_freq[None, None, :, None].float()
        position_ids_expanded = position_ids[:, :, None, :].float()
        freqs = torch.matmul(inv_freq_expanded, position_ids_expanded).transpose(2, 3)
        freqs = self._apply_mrope_export(freqs)

        emb = torch.cat((freqs, freqs), dim=-1)
        # Use function-style trig ops so normalizer can map them to supported modules.
        cos = torch.cos(emb) * self.attention_scaling
        sin = torch.sin(emb) * self.attention_scaling
        return cos.half(), sin.half()

    def _apply_mrope_export(self, freqs: Tensor) -> Tensor:
        # Keep Python control flow only on static metadata (mrope_section), not on Proxy tensors.
        start = 0
        parts = []
        for idx, section in enumerate(self.mrope_section):
            end = start + int(section)
            part = freqs[..., start:end]
            parts.append(part[idx % 3])
            start = end
        return torch.cat(parts, dim=-1)


@XHLLM_TRACEABLE_MODULES.register_module({GlmOcrTextAttention: "GlmOcrTextAttention"})
class _GlmOcrTextAttention(DynamicModule):
    def rotate_half(self, x: Tensor):
        x1 = self.slice_1(x)
        x2 = self.slice_2(x)
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(self, q: Tensor, k: Tensor, cos: Tensor, sin: Tensor):
        if self.enable_rope:
            q_embed = self.rope(q, cos, sin)
            k_embed = self.rope(k, cos, sin)
        else:
            q_embed = (q * cos) + (self.rotate_half(q) * sin)
            k_embed = (k * cos) + (self.rotate_half(k) * sin)
        return q_embed, k_embed

    def apply_multimodal_rotary_pos_emb(self, q: Tensor, k: Tensor, cos: Tensor, sin: Tensor):
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

        q_rot = self.rotary_slice_1(q)
        q_pass = self.rotary_slice_2(q)
        k_rot = self.rotary_slice_1(k)
        k_pass = self.rotary_slice_2(k)

        q_rot_embed, k_rot_embed = self.apply_rotary_pos_emb(q_rot, k_rot, cos, sin)
        q_embed = torch.cat([q_rot_embed, q_pass], dim=-1)
        k_embed = torch.cat([k_rot_embed, k_pass], dim=-1)
        return q_embed, k_embed

    def _setup(self, cfg: ConfigDict):
        self.enable_rope = cfg.get("enable_rope", True)
        if self.enable_rope:
            self.rope = xhnn.Rope()

        partial_rotary_factor = self.rope_parameters.get("partial_rotary_factor", 1.0)
        self.rotary_dim = int(self.head_dim * partial_rotary_factor)
        assert self.rotary_dim <= self.head_dim and self.rotary_dim % 2 == 0
        self.rotary_half_dim = self.rotary_dim // 2
        self.slice_1 = xhnn.Slice([0], [self.rotary_half_dim], [3], [1])
        self.slice_2 = xhnn.Slice([self.rotary_half_dim], [sys.maxsize], [3], [1])
        self.rotary_slice_1 = xhnn.Slice([0], [self.rotary_dim], [3], [1])
        self.rotary_slice_2 = xhnn.Slice([self.rotary_dim], [sys.maxsize], [3], [1])
        self.masked_softmax = MaskedSoftmax(dim=-1)
        self.use_cache = cfg.use_cache
        if self.use_cache:
            cache_axis = cfg.kv_cache.cache_axis
            self.k_cache = LLMCacheV2(axis=cache_axis)
            self.v_cache = LLMCacheV2(axis=cache_axis)
        else:
            self.k_cache = None
            self.v_cache = None

        _kv_scale = 1 / math.sqrt(self.head_dim)
        self.register_buffer("kv_scale", torch.tensor(_kv_scale, dtype=torch.float16), persistent=False)

        if not getattr(self, "_qk_weight_permuted", False):
            perm = torch.cat(
                [
                    torch.arange(0, self.rotary_dim, 2, device=self.q_proj.weight.device),
                    torch.arange(1, self.rotary_dim, 2, device=self.q_proj.weight.device),
                    torch.arange(self.rotary_dim, self.head_dim, device=self.q_proj.weight.device),
                ]
            )
            self._permute_qk_weights(perm)
            self._qk_weight_permuted = True

    def _permute_qk_weights(self, perm: Tensor):
        q_w = self.q_proj.weight.data
        q_w = q_w.view(self.num_heads, self.head_dim, -1)[:, perm, :]
        self.q_proj.weight.data = q_w.reshape(-1, q_w.shape[-1])
        if self.q_proj.bias is not None:
            q_b = self.q_proj.bias.data.view(self.num_heads, self.head_dim)[:, perm]
            self.q_proj.bias.data = q_b.reshape(-1)

        k_w = self.k_proj.weight.data
        k_w = k_w.view(self.num_key_value_heads, self.head_dim, -1)[:, perm, :]
        self.k_proj.weight.data = k_w.reshape(-1, k_w.shape[-1])
        if self.k_proj.bias is not None:
            k_b = self.k_proj.bias.data.view(self.num_key_value_heads, self.head_dim)[:, perm]
            self.k_proj.bias.data = k_b.reshape(-1)

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
        query_states, key_states = self.apply_multimodal_rotary_pos_emb(query_states, key_states, cos, sin)

        if self.use_cache:
            key_states = self.k_cache(key_states, past_seq_length, current_input_length, past_k_cache)
            value_states = self.v_cache(value_states, past_seq_length, current_input_length, past_v_cache)

        query_states = query_states * self.kv_scale
        key_states = key_states.transpose(2, 3)
        key_states = torch.repeat_interleave(key_states, self.num_key_value_groups, dim=1)

        attn_weights = torch.matmul(query_states, key_states)
        attn_weights = self.masked_softmax(attn_weights, past_seq_length)

        value_states = torch.repeat_interleave(value_states, self.num_key_value_groups, dim=1)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        return attn_output, None, None


@XHLLM_TRACEABLE_MODULES.register_module({GlmOcrTextMLP: "GlmOcrTextMLP"})
class _GlmOcrTextMLP(DynamicModule):
    def _setup(self, cfg: ConfigDict):
        in_features = self.gate_up_proj.in_features
        out_features = self.gate_up_proj.out_features // 2
        has_bias = self.gate_up_proj.bias is not None

        weight = self.gate_up_proj.weight.data.clone()  # [2*intermediate, hidden]
        gate_weight, up_weight = weight.chunk(2, dim=0)

        self.gate_proj = nn.Linear(in_features, out_features, bias=has_bias)
        self.up_proj = nn.Linear(in_features, out_features, bias=has_bias)
        self.gate_proj.weight.data.copy_(gate_weight)
        self.up_proj.weight.data.copy_(up_weight)

        if has_bias:
            bias = self.gate_up_proj.bias.data.clone()
            gate_bias, up_bias = bias.chunk(2, dim=0)
            self.gate_proj.bias.data.copy_(gate_bias)
            self.up_proj.bias.data.copy_(up_bias)

        del self.gate_up_proj

    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        gate = self.gate_proj(hidden_states)
        up_states = self.up_proj(hidden_states)
        up_states = up_states * self.activation_fn(gate)
        return self.down_proj(up_states)


@XHLLM_TRACEABLE_MODULES.register_module({GlmOcrTextDecoderLayer: "GlmOcrTextDecoderLayer"})
class _GlmOcrTextDecoderLayer(DynamicModule):
    def _setup(self, cfg: ConfigDict):
        pass

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_seq_length: Optional[Tensor] = None,
        current_input_length: Optional[Tensor] = None,
        past_k_cache: Optional[Tensor] = None,
        past_v_cache: Optional[Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.FloatTensor:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _, _ = self.self_attn(
            hidden_states=hidden_states,
            past_seq_length=past_seq_length,
            current_input_length=current_input_length,
            past_k_cache=past_k_cache,
            past_v_cache=past_v_cache,
            position_embeddings=position_embeddings,
        )
        hidden_states = self.post_self_attn_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_mlp_layernorm(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


@XHLLM_TRACEABLE_MODULES.register_module({GlmOcrTextModel: "GlmOcrTextModel"})
class _GlmOcrTextModel(DynamicModule):
    def _setup(self, cfg: ConfigDict):
        self.batch_size = cfg.get("batch_size", 1)
        self.only_first_block = cfg.get("only_first_block", False)
        self.num_logits_to_keep = cfg.num_logits_to_keep
        assert self.num_logits_to_keep in [0, 1]

        input_seq_len = cfg.input_sequence_length
        self.slice = xhnn.Slice([0], [input_seq_len], [1], [1])

        self.llm_gather = xhnn.BatchGather(1)
        self.llm_gather.update_offset_indices(self.batch_size, input_seq_len)

        def _llm_gather_update_cfg(self: xhnn.BatchGather, cfg: Optional[Dict] = None):
            input_seq_len = cfg.input_sequence_length
            batch_size = cfg.get("batch_size", 1)
            self.update_offset_indices(batch_size, input_seq_len)

        self.llm_gather._update_cfg = types.MethodType(_llm_gather_update_cfg, self.llm_gather)

        def _slice_update_cfg(self, cfg: Optional[Dict] = None):
            input_seq_len = cfg.input_sequence_length
            self.ends = [input_seq_len]

        self.slice._update_cfg = types.MethodType(_slice_update_cfg, self.slice)
        self.use_cache = cfg.use_cache

    def forward(
        self,
        inputs_embeds: Optional[Tensor] = None,
        past_seq_length: Optional[Tensor] = None,
        current_input_length: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_key_cache: Optional[List[Tensor]] = None,
        past_value_cache: Optional[List[Tensor]] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)

        for layer_idx, decoder_layer in enumerate(self.layers):
            if self.use_cache:
                _past_k_cache = past_key_cache[layer_idx]
                _past_v_cache = past_value_cache[layer_idx]
            else:
                _past_k_cache = None
                _past_v_cache = None

            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=None,
                past_seq_length=past_seq_length,
                current_input_length=current_input_length,
                past_k_cache=_past_k_cache,
                past_v_cache=_past_v_cache,
                position_embeddings=position_embeddings,
            )
            if self.only_first_block:
                break

        if self.num_logits_to_keep == 0:
            hidden_states = self.slice(hidden_states)
        else:
            hidden_states = self.llm_gather(hidden_states, current_input_length - 1)
        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(last_hidden_state=hidden_states)


@XHLLM_TRACEABLE_MODULES.register_module({GlmOcrModel: "GlmOcrModel"})
class _GlmOcrModel(DynamicModule):
    def _setup(self, cfg: ConfigDict):
        self.cfg = cfg
        del self.visual

    def forward(
        self,
        inputs_embeds: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_seq_length: Optional[Tensor] = None,
        current_input_length: Optional[Tensor] = None,
        past_key_cache: Optional[List[Tensor]] = None,
        past_value_cache: Optional[List[Tensor]] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        return self.language_model(
            inputs_embeds=inputs_embeds,
            past_seq_length=past_seq_length,
            current_input_length=current_input_length,
            position_ids=position_ids,
            past_key_cache=past_key_cache,
            past_value_cache=past_value_cache,
        )


@XHLLM_TRACEABLE_MODULES.register_module({GlmOcrForConditionalGeneration: "GlmOcrForConditionalGeneration"})
class _GlmOcrForConditionalGeneration(DynamicModule):
    def _setup(self, cfg: ConfigDict):
        self.cfg = cfg

    def forward(
        self,
        inputs_embeds: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_seq_length: Optional[Tensor] = None,
        current_input_length: Optional[Tensor] = None,
        past_key_cache: Optional[List[Tensor]] = None,
        past_value_cache: Optional[List[Tensor]] = None,
    ):
        outputs = self.model.language_model(
            inputs_embeds=inputs_embeds,
            past_seq_length=past_seq_length,
            current_input_length=current_input_length,
            position_ids=position_ids,
            past_key_cache=past_key_cache,
            past_value_cache=past_value_cache,
        )
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        return logits


def register_wrap_cls(hf_model):
    pass
