# Copyright 2025 HOUMO AI
#
# File: _llm_model_impl.py
# Description:
#   Internal LLM model implementation for Qwen3 VL.
#   This module provides traceable implementations of
#   Qwen3 VL LLM components.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
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
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from transformers.modeling_outputs import BaseModelOutputWithPast

from xhquant import nn as xhnn
from xhquant.api import ConfigDict
from xhquant.nn import BfpFlashAttention, LLMCacheV2, MaskedSoftmax, RMSNorm
from xhquant.utils.registry import DynamicModule

from ...register import XHLLM_TRACEABLE_MODULES
from .modeling_qwen3_vl import (
    Qwen3VLForConditionalGeneration,
    Qwen3VLModel,
    Qwen3VLTextAttention,
    Qwen3VLTextDecoderLayer,
    Qwen3VLTextModel,
    Qwen3VLTextRMSNorm,
    Qwen3VLTextRotaryEmbedding,
)


if TYPE_CHECKING:

    class _Qwen3VLTextRMSNormBase(DynamicModule, Qwen3VLTextRMSNorm):  # type: ignore[misc]
        ...

else:
    _Qwen3VLTextRMSNormBase = DynamicModule


@XHLLM_TRACEABLE_MODULES.register_module({Qwen3VLTextRMSNorm: "Qwen3VLTextRMSNorm"})
class _Qwen3VLTextRMSNorm(_Qwen3VLTextRMSNormBase):
    def forward(self, hidden_states):
        return self.norm(hidden_states)

    def _setup(self, cfg: Optional[Dict] = None):
        hidden_size = self.weight.shape[0]
        self.norm = RMSNorm(hidden_size, self.variance_epsilon)
        self.norm.weight = nn.Parameter(deepcopy(self.weight.data))
        return self


if TYPE_CHECKING:

    class _Qwen3VLTextRotaryEmbeddingBase(DynamicModule, Qwen3VLTextRotaryEmbedding):  # type: ignore[misc]
        ...

else:
    _Qwen3VLTextRotaryEmbeddingBase = DynamicModule


@XHLLM_TRACEABLE_MODULES.register_module(
    {
        Qwen3VLTextRotaryEmbedding: "Qwen3VLTextRotaryEmbedding",
    }
)
class _Qwen3VLTextRotaryEmbedding(_Qwen3VLTextRotaryEmbeddingBase):
    def _setup(self, cfg: ConfigDict):
        max_pe_length = cfg.max_pe_length
        # self.max_position_embeddings = max_position_embeddings
        # Build here to make `torch.jit.trace` work.
        self._setup_cos_sin_cache(seq_len=max_pe_length)
        if hasattr(self, "setup_after_callback"):
            self.setup_after_callback()

    def _setup_cos_sin_cache(self, seq_len):
        cos, sin = self.forward(max_seq_len_cached=seq_len)
        self.register_buffer("cos_cached", cos[0], persistent=False)
        self.register_buffer("sin_cached", sin[0], persistent=False)

    def forward(self, max_seq_len_cached=2048):
        inv_freq = self.inv_freq
        if torch.cuda.is_available():
            inv_freq = inv_freq.cuda()
        inv_freq_expanded = inv_freq[None, None, :, None].float().expand(1, max_seq_len_cached, -1, 1)
        # device_type = self.inv_freq.device
        # with torch.autocast(device_type=device_type, enabled=False):  # Force float32
        freqs = (
            inv_freq_expanded.float()
            @ torch.arange(max_seq_len_cached, device=inv_freq.device).float()[None, :, None, None]
        ).transpose(2, 3)
        # freqs = self.apply_interleaved_mrope(freqs, self.mrope_section)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling
        return cos.to(dtype=self.inv_freq.dtype).to(self.inv_freq.device), sin.to(dtype=self.inv_freq.dtype).to(
            self.inv_freq.device
        )


if TYPE_CHECKING:

    class _Qwen3VLTextAttentionBase(DynamicModule, Qwen3VLTextAttention):  # type: ignore[misc]
        ...

else:
    _Qwen3VLTextAttentionBase = DynamicModule


@XHLLM_TRACEABLE_MODULES.register_module({Qwen3VLTextAttention: "Qwen3VLTextAttention"})
class _Qwen3VLTextAttention(_Qwen3VLTextAttentionBase):
    def _setup(self, cfg: ConfigDict):
        if not hasattr(self, "num_key_value_heads"):
            self.num_key_value_heads = self.config.num_key_value_heads
        if not hasattr(self, "num_heads"):
            self.num_heads = self.config.num_attention_heads
        self.enable_rope = cfg.get("enable_rope", True)
        if self.enable_rope:
            self.rope = xhnn.Rope()

        self.slice_1 = xhnn.Slice([0], [self.head_dim // 2], [3], [1])
        self.slice_2 = xhnn.Slice([self.head_dim // 2], [sys.maxsize], [3], [1])
        self.masked_softmax = MaskedSoftmax(dim=-1)
        self.attn_hidden_dim = self.head_dim * self.num_heads
        bfp_flash_attention_cfg = cfg.get("bfp_flash_attention", None)
        self.use_bfp_flash_attention = False
        if bfp_flash_attention_cfg is not None:
            self.use_bfp_flash_attention = bfp_flash_attention_cfg.enable
            self.sefp_manbit = bfp_flash_attention_cfg.sefp_manbit
            self.out_fp_manbit = bfp_flash_attention_cfg.out_fp_manbit
            self.out_fp_expbit = bfp_flash_attention_cfg.out_fp_expbit
        if self.use_bfp_flash_attention:
            self.bfp_attn = BfpFlashAttention(
                self.attn_hidden_dim,
                self.num_heads,
                True,
                self.sefp_manbit,
                self.out_fp_expbit,
                self.out_fp_manbit,
            )
        use_cache = cfg.use_cache
        self.use_cache = use_cache
        if use_cache:
            cache_axis = cfg.kv_cache.cache_axis
            self.k_cache = LLMCacheV2(
                axis=cache_axis,
            )
            self.v_cache = LLMCacheV2(
                axis=cache_axis,
            )
        else:
            self.k_cache = None
            self.v_cache = None
        _kv_scale = 1 / math.sqrt(self.head_dim)
        # self.kv_scale = _kv_scale
        self.register_buffer("kv_scale", torch.tensor(_kv_scale, dtype=torch.float16), persistent=False)

    def apply_rotary_pos_emb(self, q: Tensor, k: Tensor, cos: Tensor, sin: Tensor, unsqueeze_dim: int = 1):
        if self.enable_rope:
            q_embed = self.rope(q, cos, sin)
            k_embed = self.rope(k, cos, sin)
        else:
            q_embed = (q * cos) + (self.rotate_half(q) * sin)
            k_embed = (k * cos) + (self.rotate_half(k) * sin)
        return q_embed, k_embed

    def graph_forward(
        self,
        hidden_states: torch.Tensor,
        past_seq_length: Optional[Tensor] = None,
        current_input_length: Optional[Tensor] = None,
        past_k_cache: Optional[Tensor] = None,
        past_v_cache: Optional[Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        bsz, q_len, _ = hidden_states.size()
        causal_mask = None
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = self.q_norm(
            query_states.view(bsz, q_len, self.config.num_attention_heads, self.head_dim)
        ).transpose(1, 2)
        key_states = self.k_norm(key_states.view(bsz, q_len, self.config.num_key_value_heads, self.head_dim)).transpose(
            1, 2
        )
        value_states = value_states.view(bsz, q_len, self.config.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = self.apply_rotary_pos_emb(query_states, key_states, cos, sin, unsqueeze_dim=1)

        if self.use_cache:
            key_states = self.k_cache(key_states, past_seq_length, current_input_length, past_k_cache)
            value_states = self.v_cache(value_states, past_seq_length, current_input_length, past_v_cache)
        if self.use_bfp_flash_attention:
            attn_output = self.bfp_attn(query_states, key_states, value_states)
        else:
            query_states = query_states * self.kv_scale
            key_states = key_states.transpose(2, 3)

            key_states = torch.repeat_interleave(
                key_states,
                self.num_key_value_groups,
                dim=1,
            )

            attn_weights = torch.matmul(query_states, key_states)

            attn_weights: Optional[Tensor] = self.masked_softmax(attn_weights, past_seq_length)

            value_states = torch.repeat_interleave(
                value_states,
                self.num_key_value_groups,
                dim=1,
            )
            attn_output = torch.matmul(attn_weights, value_states)

            attn_output = attn_output.transpose(1, 2)
            attn_output = attn_output.reshape(bsz, q_len, self.config.num_attention_heads * self.head_dim)
        attn_output = self.o_proj(attn_output)
        return attn_output, None, None


if TYPE_CHECKING:

    class _Qwen3VLTextDecoderLayerBase(DynamicModule, Qwen3VLTextDecoderLayer):  # type: ignore[misc]
        ...

else:
    _Qwen3VLTextDecoderLayerBase = DynamicModule


@XHLLM_TRACEABLE_MODULES.register_module({Qwen3VLTextDecoderLayer: "Qwen3VLTextDecoderLayer"})
class _Qwen3VLTextDecoderLayer(_Qwen3VLTextDecoderLayerBase):
    def _setup(self, cfg: ConfigDict):
        pass

    def graph_forward(
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
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


if TYPE_CHECKING:

    class _Qwen3VLTextModelBase(DynamicModule, Qwen3VLTextModel):  # type: ignore[misc]
        ...

else:
    _Qwen3VLTextModelBase = DynamicModule


@XHLLM_TRACEABLE_MODULES.register_module(
    {
        Qwen3VLTextModel: "Qwen3VLTextModel",
    }
)
class _Qwen3VLTextModel(_Qwen3VLTextModelBase):
    def _setup_cos_sin_embeding(self):
        cos = self.rotary_emb.cos_cached.squeeze(1)
        sin = self.rotary_emb.sin_cached.squeeze(1)
        num_embeddings, embedding_dim = cos.shape

        self.cos_embeding = nn.Embedding(num_embeddings, embedding_dim)
        self.cos_embeding.weight.data = cos

        self.sin_embeding = nn.Embedding(num_embeddings, embedding_dim)
        self.sin_embeding.weight.data = sin

    def _setup(self, cfg: ConfigDict):
        self.batch_size = cfg.get("batch_size", 1)
        self.only_first_block = cfg.get("only_first_block", False)

        self.num_logits_to_keep = cfg.num_logits_to_keep  # 1,取最后一个token的输出，0,取所有token的输出
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

        time_mask = torch.ones(64)
        h_ids = torch.arange(1, 60, 3)
        w_ids = torch.arange(2, 60, 3)
        time_mask[h_ids] = 0
        time_mask[w_ids] = 0
        time_mask = torch.cat([time_mask, time_mask], 0)
        time_mask.unsqueeze_(0).unsqueeze_(0)
        self.register_buffer("time_mask", time_mask.half(), persistent=False)

        hight_mask = torch.zeros(64)
        hight_mask[h_ids] = 1
        hight_mask = torch.cat([hight_mask, hight_mask], 0)
        hight_mask.unsqueeze_(0).unsqueeze_(0)
        self.register_buffer("hight_mask", hight_mask.half(), persistent=False)

        width_mask = torch.zeros(64)
        width_mask[w_ids] = 1
        width_mask = torch.cat([width_mask, width_mask], 0)
        width_mask.unsqueeze_(0).unsqueeze_(0)
        self.register_buffer("width_mask", width_mask.half(), persistent=False)

        self.only_first_block = cfg.get("only_first_block", False)
        self.max_layers = -1
        if self.only_first_block:
            self.max_layers = 1
        if "max_layers" in cfg and cfg.max_layers is not None:
            self.max_layers = cfg.max_layers

        if not hasattr(self.rotary_emb, "cos_cached"):
            self.rotary_emb.setup_after_callback = self._setup_cos_sin_embeding
        else:
            self._setup_cos_sin_embeding()

    def graph_forward(
        self,
        input_embeds: Optional[Tensor] = None,
        time_position_ids: Optional[Tensor] = None,
        hight_position_ids: Optional[Tensor] = None,
        width_position_ids: Optional[Tensor] = None,
        past_seq_length: Optional[Tensor] = None,
        current_input_length: Optional[Tensor] = None,
        deepstack_visual_embed_0: Optional[Tensor] = None,
        deepstack_visual_embed_1: Optional[Tensor] = None,
        deepstack_visual_embed_2: Optional[Tensor] = None,
        past_key_cache: Optional[List[Tensor]] = None,
        past_value_cache: Optional[List[Tensor]] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        causal_mask = None
        cos = self.rotary_emb.cos_cached
        sin = self.rotary_emb.sin_cached

        # TODO:可以改成nn.Embeding
        time_cos = cos[time_position_ids]  # time_position_ids需要是torch.int64类型，才能正确索引到cos中的位置
        time_sin = sin[time_position_ids]

        hight_cos = cos[hight_position_ids]  # hight_position_ids.int64类型，才能正确索引到cos中的位置
        hight_sin = sin[hight_position_ids]

        width_cos = cos[width_position_ids]  # width_position_ids.int64类型，才能正确索引到cos中的位置
        width_sin = sin[width_position_ids]

        time_cos = time_cos * self.time_mask
        time_sin = time_sin * self.time_mask
        hight_cos = hight_cos * self.hight_mask
        hight_sin = hight_sin * self.hight_mask
        width_cos = width_cos * self.width_mask
        width_sin = width_sin * self.width_mask

        cos = time_cos + hight_cos + width_cos
        sin = time_sin + hight_sin + width_sin

        cos = cos.squeeze(1).unsqueeze(0).unsqueeze(0)
        sin = sin.squeeze(1).unsqueeze(0).unsqueeze(0)
        position_embeddings = (cos, sin)

        hidden_states = input_embeds

        for layer_idx, decoder_layer in enumerate(self.layers):
            if self.use_cache:
                _past_k_cache = past_key_cache[layer_idx]
                _past_v_cache = past_value_cache[layer_idx]
            else:
                _past_k_cache = None
                _past_v_cache = None

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                past_seq_length=past_seq_length,
                current_input_length=current_input_length,
                past_k_cache=_past_k_cache,
                past_v_cache=_past_v_cache,
                position_embeddings=position_embeddings,
            )
            hidden_states = layer_outputs

            # add visual features to the hidden states of first several layers
            if deepstack_visual_embed_0 is not None and layer_idx == 0:
                hidden_states = hidden_states + deepstack_visual_embed_0
            if deepstack_visual_embed_1 is not None and layer_idx == 1:
                hidden_states = hidden_states + deepstack_visual_embed_1
            if deepstack_visual_embed_2 is not None and layer_idx == 2:
                hidden_states = hidden_states + deepstack_visual_embed_2

            if self.max_layers > 0 and layer_idx + 1 >= self.max_layers:
                break

        if self.num_logits_to_keep == 0:
            # hidden_states = torch_ops_xh2a_slice(hidden_states, [0], [current_input_length], [1], [1])
            hidden_states = self.slice(
                hidden_states
            )  # 此时返回的结果，含有padding,调用者需要根据current_input_length切片
        else:
            # 取最后一个token的输出
            hidden_states = self.llm_gather(hidden_states, current_input_length - 1)

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
        )


if TYPE_CHECKING:

    class _Qwen3VLModelBase(DynamicModule, Qwen3VLModel):  # type: ignore[misc]
        ...

else:
    _Qwen3VLModelBase = DynamicModule


@XHLLM_TRACEABLE_MODULES.register_module(
    {
        Qwen3VLModel: "Qwen3VLModel",
    }
)
class _Qwen3VLModel(_Qwen3VLModelBase):
    def _setup(self, cfg: ConfigDict):
        self.cfg = cfg
        del self.visual

    def graph_forward(
        self,
        inputs_embeds: Optional[Tensor] = None,
        time_position_ids: Optional[Tensor] = None,
        hight_position_ids: Optional[Tensor] = None,
        width_position_ids: Optional[Tensor] = None,
        past_seq_length: Optional[Tensor] = None,
        current_input_length: Optional[Tensor] = None,
        deepstack_visual_embed_0: Optional[Tensor] = None,
        deepstack_visual_embed_1: Optional[Tensor] = None,
        deepstack_visual_embed_2: Optional[Tensor] = None,
        past_key_cache: Optional[List[Tensor]] = None,
        past_value_cache: Optional[List[Tensor]] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        outputs = self.language_model(
            input_embeds=inputs_embeds,
            time_position_ids=time_position_ids,
            hight_position_ids=hight_position_ids,
            width_position_ids=width_position_ids,
            deepstack_visual_embed_0=deepstack_visual_embed_0,
            deepstack_visual_embed_1=deepstack_visual_embed_1,
            deepstack_visual_embed_2=deepstack_visual_embed_2,
            past_seq_length=past_seq_length,
            current_input_length=current_input_length,
            past_key_cache=past_key_cache,
            past_value_cache=past_value_cache,
        )
        return outputs


if TYPE_CHECKING:

    class _Qwen3VLForConditionalGenerationBase(DynamicModule, Qwen3VLForConditionalGeneration):  # type: ignore[misc]
        ...

else:
    _Qwen3VLForConditionalGenerationBase = DynamicModule


@XHLLM_TRACEABLE_MODULES.register_module(
    {
        Qwen3VLForConditionalGeneration: "Qwen3VLForConditionalGeneration",
    }
)
class _Qwen3VLForConditionalGeneration(_Qwen3VLForConditionalGenerationBase):
    def _setup(self, cfg: ConfigDict):
        self.cfg = cfg

    def graph_forward(
        self,
        inputs_embeds: Optional[Tensor] = None,
        time_position_ids: Optional[Tensor] = None,
        hight_position_ids: Optional[Tensor] = None,
        width_position_ids: Optional[Tensor] = None,
        past_seq_length: Optional[Tensor] = None,
        current_input_length: Optional[Tensor] = None,
        deepstack_visual_embed_0: Optional[Tensor] = None,
        deepstack_visual_embed_1: Optional[Tensor] = None,
        deepstack_visual_embed_2: Optional[Tensor] = None,
        past_key_cache: Optional[List[Tensor]] = None,
        past_value_cache: Optional[List[Tensor]] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        outputs = self.model.language_model.forward(
            input_embeds=inputs_embeds,
            time_position_ids=time_position_ids,
            hight_position_ids=hight_position_ids,
            width_position_ids=width_position_ids,
            deepstack_visual_embed_0=deepstack_visual_embed_0,
            deepstack_visual_embed_1=deepstack_visual_embed_1,
            deepstack_visual_embed_2=deepstack_visual_embed_2,
            past_seq_length=past_seq_length,
            current_input_length=current_input_length,
            past_key_cache=past_key_cache,
            past_value_cache=past_value_cache,
        )
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        return logits


def register_wrap_cls(hf_model):
    pass
