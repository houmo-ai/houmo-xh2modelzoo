# -*- coding: utf-8 -*-
# Copyright 2025 OpenAI. All rights reserved.
# Copyright 2025 HOUMO AI. All rights reserved.
#
# Modifications:
# - Portions of this file have been modified by HOUMO AI.
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
# File: _model.py
# Description:
#   gpt-oss model implementation adapted for the xh2 model zoo (xh2modelzoo).

"""
_Qwen3MoeTopKRouter(DynamicModule) 直接被删除

_setup 参数类型注释改变

_Qwen3MoeRotaryEmbedding：大改动

_Qwen3MoeSparseMoeBlock 取消hidden_states.view(-1, hidden_dim)，_setup函数参数初始化大改动

Qwen3 Attn：

Qwen3 DecoderLayer：

_Qwen3MoeRMSNorm：改为deepcopy初始化权重。self.norm.weight = nn.Parameter(deepcopy(self.weight.data))

_Qwen3MoeModel(DynamicModule): 改为return hidden_states，

_Qwen3MoeForCausalLM：return 所有token的logits

"""


import math
import sys
import types
from copy import deepcopy
from typing import Any, Optional, List, Tuple, Dict

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from transformers.models.gpt_oss.modeling_gpt_oss import (
    GptOssRMSNorm,
    GptOssRotaryEmbedding,
    GptOssAttention,
    GptOssExperts,
    GptOssTopKRouter,
    GptOssMLP,
    GptOssDecoderLayer,
    GptOssModel,
    GptOssForCausalLM,
)

from xhquant import nn as xhnn
from xhquant.api import ConfigDict
# from xhquant.core import CacheTensor, HybridCacheTensor
from xhquant.nn import LLMCache, MoeBlock, RMSNorm, SinksMaskedSoftmax
from xhquant.utils.registry import DynamicModule

from ...register import XHLLM_TRACEABLE_MODULES
# from ...kv_cache_mixin import KVCacheMixin
# from ...types import CacheList, KVCacheConfig


# def _align_to(size: int, align: int = 16) -> int:
#     return ((size + align - 1) // align) * align


# def build_gpt_oss_cache_specs(
#     *,
#     layer_types: list[str],
#     num_key_value_heads: int,
#     head_dim: int,
#     context_max_length: int,
#     prefill_chunk_length: int,
#     sliding_window: int,
#     cache_dtype: torch.dtype = torch.float16,
# ) -> list[dict[str, Any]]:
#     sliding_cache_length = _align_to(sliding_window + prefill_chunk_length - 1, 16)
#     cache_specs: list[dict[str, Any]] = []
#     for layer_type in layer_types:
#         cache_length = context_max_length
#         if layer_type == "sliding_attention":
#             cache_length = sliding_cache_length
#         cache_specs.append(
#             dict(
#                 attention_type=layer_type,
#                 shape=[1, num_key_value_heads, cache_length, head_dim],
#                 dtype=cache_dtype,
#             )
#         )
#     return cache_specs


# class GptOssKVCacheMixin(KVCacheMixin):
#     def __init__(self, kv_cache_config: KVCacheConfig, cache_specs: list[dict[str, Any]] | None = None) -> None:
#         super().__init__(kv_cache_config)
#         self.cache_specs = cache_specs or []

#     def set_cache_specs(self, cache_specs: list[dict[str, Any]]) -> None:
#         self.cache_specs = cache_specs

#     def prepare_kv_cache(self):
#         if not self.use_cache:
#             return
#         if not self.cache_specs:
#             return super().prepare_kv_cache()

#         self.clear_kv_cache()
#         for cache_spec in self.cache_specs:
#             cache_cls = HybridCacheTensor if cache_spec["attention_type"] == "sliding_attention" else CacheTensor
#             cache_shape = cache_spec["shape"]
#             cache_dtype = cache_spec.get("dtype", torch.float16)
#             self.past_key_caches.append(cache_cls(torch.zeros(cache_shape, dtype=cache_dtype)))
#             self.past_value_caches.append(cache_cls(torch.zeros(cache_shape, dtype=cache_dtype)))

#         if self._device is not None:
#             self._set_device(self._device)

#     def _set_device(self, device):
#         super()._set_device(device)
#         self.past_key_caches = CacheList([cache.to(device) for cache in self.past_key_caches])
#         self.past_value_caches = CacheList([cache.to(device) for cache in self.past_value_caches])


@XHLLM_TRACEABLE_MODULES.register_module(
    {
        GptOssRotaryEmbedding: "GptOssRotaryEmbedding",
    }
)
class _GptOssRotaryEmbedding(DynamicModule):
    def _setup(self, cfg: dict[str, Any] | None = None):
        assert "dynamic" not in self.rope_type, f"{self.rope_type} is not supported in dynamic mode"
        max_seq_len = cfg.context_max_length
        max_seq_len = max(max_seq_len, self.config.max_position_embeddings)

        self.config.max_position_embeddings = max_seq_len
        self.max_seq_len_cached = max_seq_len
        self.original_max_seq_len = max_seq_len
        
        self._setup_cos_sin_cache(seq_len=self.max_seq_len_cached, dtype=self.inv_freq.dtype)
        if hasattr(self, "setup_after_callback"):
            self.setup_after_callback()

    def _setup_cos_sin_cache(self, seq_len, dtype):
        
        position_ids = torch.arange(0, seq_len, dtype=torch.long, device=self.inv_freq.device).unsqueeze(0)
        self.inv_freq = self.inv_freq.to(torch.float16)
        cos, sin = self.forward(self.inv_freq, position_ids)
        sin = sin.squeeze(0)
        cos = cos.squeeze(0)

        # TODO:临时处理
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

        sin = sin.contiguous().to(dtype=dtype)
        cos = cos.contiguous().to(dtype=dtype)

        self.register_buffer("sin_cached", sin, persistent=False)
        self.register_buffer("cos_cached", cos, persistent=False)

    def _set_dtype(self, dtype: torch.dtype) -> None:
        self.inv_freq = self.inv_freq.to(dtype)
        self._setup_cos_sin_cache(seq_len=self.max_seq_len_cached, dtype=self.inv_freq.dtype)

    @torch.no_grad()
    def forward(self, x, position_ids):
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)
        
         # Core RoPE block
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        device_ = x.device
        inv_freq_expanded = inv_freq_expanded.to(device_)
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


@XHLLM_TRACEABLE_MODULES.register_module(
    {
        GptOssRMSNorm: "GptOssRMSNorm",
    }
)
class _GptOssRMSNorm(DynamicModule):
    def forward(self, hidden_states):
        return self.norm(hidden_states)

    def _setup(self, cfg: dict[str, Any] | None = None):
        hidden_size = self.weight.shape[0]
        self.norm = RMSNorm(hidden_size, self.variance_epsilon)
        self.norm.weight = nn.Parameter(deepcopy(self.weight.data))
        return self


@XHLLM_TRACEABLE_MODULES.register_module(
    {
        GptOssMLP: "GptOssMLP",
    }
)
class _GptOssMLP(DynamicModule):
    def forward(self, hidden_states):
        router_scores, _ = self.router(hidden_states)
        routed_out = self.experts(hidden_states, routing_weights=router_scores)
        return routed_out, router_scores

    def _setup(self, cfg: dict[str, Any] | None = None):
        return self


@XHLLM_TRACEABLE_MODULES.register_module(
    {
        GptOssExperts: "GptOssExperts",
    }
)
class _GptOssExperts(DynamicModule):
    def graph_forward(self, hidden_states: torch.Tensor, routing_weights=None) -> torch.Tensor:
        return self.moeblock(hidden_states, routing_weights)

    def _setup(self, cfg: dict[str, Any] | None = None):
        self.moeblock = MoeBlock(
            "glu_oss",
            cfg.num_experts_per_tok,
            normalize_routing_weights=True,
        )
        if hasattr(self, "gate_up_proj"):
            self.moeblock.expert_gate_proj_weight = torch.nn.Parameter(self.gate_up_proj[..., ::2].transpose(1, 2))
            self.moeblock.expert_gate_proj_bias = torch.nn.Parameter(self.gate_up_proj_bias[..., ::2])
            self.moeblock.expert_up_proj_weight = torch.nn.Parameter(self.gate_up_proj[..., 1::2].transpose(1, 2))
            self.moeblock.expert_up_proj_bias = torch.nn.Parameter(self.gate_up_proj_bias[..., 1::2])
            self.moeblock.expert_down_proj_weight = torch.nn.Parameter(self.down_proj.transpose(1, 2))
            self.moeblock.expert_down_proj_bias = torch.nn.Parameter(self.down_proj_bias)
            del self.gate_up_proj, self.gate_up_proj_bias, self.down_proj, self.down_proj_bias
        else:
            gate_up_shape = self.gate_up[0].weight.shape
            self.moeblock.expert_gate_proj_weight = torch.nn.Parameter(torch.cat([i.weight.reshape(1,*gate_up_shape) for i in self.gate_up],dim=0)[:, ::2, :])
            self.moeblock.expert_gate_proj_bias = torch.nn.Parameter(torch.cat([i.bias.reshape(1,gate_up_shape[0]) for i in self.gate_up],dim=0)[:, ::2])
            self.moeblock.expert_up_proj_weight = torch.nn.Parameter(torch.cat([i.weight.reshape(1,*gate_up_shape) for i in self.gate_up],dim=0)[:, 1::2, :])
            self.moeblock.expert_up_proj_bias = torch.nn.Parameter(torch.cat([i.bias.reshape(1,gate_up_shape[0]) for i in self.gate_up],dim=0)[:, 1::2])
            down_shape = self.down[0].weight.shape
            self.moeblock.expert_down_proj_weight = torch.nn.Parameter(torch.cat([i.weight.reshape(1,*down_shape) for i in self.down],dim=0))
            self.moeblock.expert_down_proj_bias = torch.nn.Parameter(torch.cat([i.bias.reshape(1,down_shape[0]) for i in self.down],dim=0))
            del self.gate_up, self.down

        self.moeblock.limit = self.limit

        
        torch.cuda.empty_cache()
        return self


@XHLLM_TRACEABLE_MODULES.register_module(
    {
        GptOssAttention: "GptOssAttention",
    }
)
class _GptOssAttention(DynamicModule):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def rotate_half(self, x: Tensor):
        """Rotates half the hidden dims of the input."""
        x1 = self.slice_1(x)
        x2 = self.slice_2(x)
        return torch.cat((-x2, x1), dim=-1)

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
        # position_ids: torch.Tensor,
        past_seq_length: Optional[Tensor] = None,
        current_input_length: Optional[Tensor] = None,
        past_k_cache: Optional[Tensor] = None,
        past_v_cache: Optional[Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()
        # bsz = hidden_states.shape[0]
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2).contiguous()
        key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2).contiguous()
        value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2).contiguous()

        cos, sin = position_embeddings
        query_states, key_states = self.apply_rotary_pos_emb(query_states, key_states, cos, sin, unsqueeze_dim=0)

        if self.use_cache:
            key_states = self.k_cache(key_states, past_seq_length, current_input_length, past_k_cache)
            value_states = self.v_cache(value_states, past_seq_length, current_input_length, past_v_cache)

        query_states = query_states * self.kv_scale  # [bsz, self.num_key_value_heads, seq_len, self.head_dim]
        key_states = key_states.transpose(2, 3)
        key_states = torch.repeat_interleave(
            key_states,
            self.num_key_value_groups,
            dim=1,
        )

        attn_weights = torch.matmul(query_states, key_states)  # [1, 64, 216, 64], [1, 64, 64, 2048]
        # sinks = self.sinks.reshape(1, -1, 1, 1)

        # sinks = sinks.expand((query_states.shape[0], -1, query_states.shape[-2], -1))
        # attn_weights = torch.cat([attn_weights, sinks], dim=-1)
        attn_weights: Optional[Tensor] = self.masked_softmax(attn_weights, past_seq_length, sinks=self.sinks)
        # attn_weights = attn_weights[..., :-1] # we drop the sink here

        value_states = torch.repeat_interleave(
            value_states,
            self.num_key_value_groups,
            dim=1,
        )
        attn_output = torch.matmul(attn_weights, value_states)  # [4, 28, 256, 32768], [4, 28, 32768, 128]
        attn_output = attn_output.transpose(1, 2)

        attn_output = attn_output.reshape(bsz, q_len, self.config.num_attention_heads * self.head_dim)
        attn_output = self.o_proj(attn_output)

        # return attn_output, attn_weights, past_key_value
        return attn_output, None, None

    def _setup(self, cfg: ConfigDict | dict[str, Any]):
        if isinstance(cfg, dict):
            cfg = ConfigDict(cfg)
        if not hasattr(self, "num_key_value_heads"):
            self.num_key_value_heads = self.config.num_key_value_heads
        if not hasattr(self, "num_heads"):
            self.num_heads = self.config.num_attention_heads  # 28
        if not hasattr(self, "hidden_size"):
            self.hidden_size = self.config.hidden_size  # 3584

        self.enable_rope = cfg.get("enable_rope", True)
        if self.enable_rope:
            self.rope = xhnn.Rope()

        self.slice_1 = xhnn.Slice([0], [self.head_dim // 2], [3], [1])
        self.slice_2 = xhnn.Slice([self.head_dim // 2], [sys.maxsize], [3], [1])

        self.sinks.data = self.sinks.data.reshape(1, -1, 1, 1)

        attention_max_length = self.sliding_window if self.sliding_window is not None else -1

        self.masked_softmax = SinksMaskedSoftmax(dim=-1, attention_max_length=attention_max_length)
        use_cache = cfg.use_cache
        self.use_cache = use_cache
        self.key_extra_scale = 1.0 if "key_extra_scale" not in cfg else cfg.key_extra_scale
        self.query_extra_scale = 1.0 if "query_extra_scale" not in cfg else cfg.query_extra_scale

        self.cos_gather = xhnn.Gather(0)
        self.sin_gather = xhnn.Gather(0)

        if use_cache:
            cache_axis = cfg.kv_cache.cache_axis
            self.k_cache = LLMCache(
                axis=cache_axis,
                attention_max_length=attention_max_length,
            )
            self.v_cache = LLMCache(
                axis=cache_axis,
                attention_max_length=attention_max_length,
            )
        else:
            self.k_cache = None
            self.v_cache = None
        _kv_scale = 1 / math.sqrt(self.head_dim)
        self.kv_scale = _kv_scale

        return self


@XHLLM_TRACEABLE_MODULES.register_module(
    {
        GptOssDecoderLayer: "GptOssDecoderLayer",
    }
)
class _GptOssDecoderLayer(DynamicModule):
    def graph_forward(
        self,
        hidden_states: torch.Tensor,
        # position_ids: Optional[torch.LongTensor] = None,
        # rotary_matrix: Optional[torch.Tensor] = None,
        past_seq_length: Optional[Tensor] = None,
        current_input_length: Optional[Tensor] = None,
        past_k_cache: Optional[Tensor] = None,
        past_v_cache: Optional[Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence.
            position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            # rotary_matrix=rotary_matrix,
            # position_ids=position_ids,
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
        hidden_states, _ = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        return outputs

    def _setup(self, cfg: Optional[Dict] = None):
        return self


@XHLLM_TRACEABLE_MODULES.register_module(
    {
        GptOssModel: "GptOssModel",
    }
)
class _GptOssModel(DynamicModule):
    def graph_forward(
        self,
        # position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_seq_length: Optional[Tensor] = None,
        current_input_length: Optional[Tensor] = None,
        past_key_cache: Optional[List[Tensor]] = None,
        past_value_cache: Optional[List[Tensor]] = None,
    ):

        causal_mask = None  # 在Qwen2Attention中处理
        hidden_states = inputs_embeds

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
            # if idx == 0:
            #    import time
            #    time_start = time.time()
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                # # position_ids=position_ids,
                past_seq_length=past_seq_length,
                current_input_length=current_input_length,
                past_k_cache=_past_k_cache,
                past_v_cache=_past_v_cache,
                position_embeddings=position_embeddings,
            )
            # if idx == 0:
            #    time_interval = time.time() - time_start
            #    print(f"Time {time_interval}")
            hidden_states = layer_outputs[0]
            if self.only_first_block:
                break

        # hidden_states = hidden_states[
        #     :,
        #     -num_logits_to_keep:,
        # ]
        # hidden_states = hidden_states[:, :current_input_length, :]
        # hidden_states = self.llm_gather(hidden_states, current_input_length, num_logits_to_keep)
        if self.num_logits_to_keep == 0:
            # hidden_states = torch_ops_xh2a_slice(hidden_states, [0], [current_input_length], [1], [1])
            # hidden_states = self.slice(
            #     hidden_states
            # )  # 此时返回的结果，含有padding,调用者需要根据current_input_length切片
            pass
        else:
            # 取最后一个token的输出
            hidden_states = self.llm_gather(hidden_states, current_input_length - 1)
        hidden_states = self.norm(hidden_states)

        return hidden_states

    def _setup_cos_sin_embeding(self):
        self.rotary_emb.cos_cached
        self.rotary_emb.sin_cached

    def _setup(self, cfg: Optional[Dict] = None):
        self.only_first_block = cfg.get("only_first_block", False)
        # max_seq_len = cfg.max_sequence_length
        # self.rotary_matrix_cache = RotaryMatrixCache(self.rotary_emb, max_seq_len)

        self.num_logits_to_keep = cfg.num_logits_to_keep  # 1,取最后一个token的输出，0,取所有token的输出
        assert self.num_logits_to_keep in [0, 1]

        input_seq_len = cfg.input_sequence_length
        self.slice = xhnn.Slice([0], [input_seq_len], [1], [1])

        self.llm_gather = xhnn.BatchGather(1)
        self.llm_gather.update_offset_indices(1, input_seq_len)

        def _llm_gather_update_cfg(self: xhnn.BatchGather, cfg: Optional[Dict] = None):
            self.num_logits_to_keep = cfg.num_logits_to_keep
            input_seq_len = cfg.input_sequence_length
            batch_size = cfg.get("batch_size", 1)
            self.update_offset_indices(batch_size, input_seq_len)

        self.llm_gather._update_cfg = types.MethodType(_llm_gather_update_cfg, self.llm_gather)

        def _update_cfg(self, cfg: Optional[Dict] = None):
            input_seq_len = cfg.input_sequence_length
            self.ends = [input_seq_len]

        self.slice._update_cfg = types.MethodType(_update_cfg, self.slice)
        self.use_cache = cfg.use_cache

        self.sin_slice = xhnn.DynamicSlice([input_seq_len], [2], [1])
        self.cos_slice = xhnn.DynamicSlice([input_seq_len], [2], [1])

        def _sin_cos_slice_update_cfg(self, cfg: Optional[Dict] = None):
            input_seq_len = cfg.input_sequence_length
            self.valid_length = [input_seq_len]

        self.sin_slice._update_cfg = types.MethodType(_sin_cos_slice_update_cfg, self.sin_slice)
        self.cos_slice._update_cfg = types.MethodType(_sin_cos_slice_update_cfg, self.cos_slice)

        self.cos_unsqueeze = xhnn.Unsqueeze(0)
        self.sin_unsqueeze = xhnn.Unsqueeze(0)

        if not hasattr(self.rotary_emb, "cos_cached"):
            self.rotary_emb.setup_after_callback = self._setup_cos_sin_embeding
        else:
            self._setup_cos_sin_embeding()

        return self

@XHLLM_TRACEABLE_MODULES.register_module(
    {
        GptOssForCausalLM: "GptOssForCausalLM",
    }
)
class _GptOssForCausalLM(DynamicModule):
    def graph_forward(
        self,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_seq_length: Optional[Tensor] = None,
        current_input_length: Optional[Tensor] = None,
        past_key_cache: Optional[List[Tensor]] = None,
        past_value_cache: Optional[List[Tensor]] = None,
    ):
        hidden_states = self.model(
            inputs_embeds=inputs_embeds,
            past_seq_length=past_seq_length,
            current_input_length=current_input_length,
            past_key_cache=past_key_cache,
            past_value_cache=past_value_cache,
        )
        logits = self.lm_head(hidden_states)
        return logits

    def _setup(self, cfg: Optional[Dict] = None):
        return self


@XHLLM_TRACEABLE_MODULES.register_module(
    {
        GptOssTopKRouter: "GptOssTopKRouter",
    }
)
class _GptOssTopKRouter(DynamicModule):
    def forward(self, hidden_states):
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        # router_logits = F.linear(hidden_states, self.weight, self.bias)  # (seq_len, num_experts)
        # router_top_value, router_indices = torch.topk(router_logits, self.top_k, dim=-1)  # (seq_len, top_k)
        # router_top_value = torch.nn.functional.softmax(router_top_value, dim=1, dtype=router_top_value.dtype)
        # router_scores = torch.zeros_like(router_logits).scatter_(1, router_indices, router_top_value)
        router_logits = self.proj(hidden_states)
        router_top_value, router_indices = self.topk(router_logits)
        router_top_value = self.softmax(router_top_value)
        router_scores = self.mul(router_logits, 0)
        router_scores = self.scatter(router_scores, router_indices, router_top_value)
        return router_scores, router_indices
    
    # def forward(self, hidden_states):
    #     router_value = self.proj(hidden_states)
    #     # return router_value
    #     router_value = self.softmax(router_value)
    #     return router_value
    

    def _setup(self, cfg: Optional[Dict] = None):
        if hasattr(self, "weight"):
            self.proj = nn.Linear(self.weight.shape[1], self.weight.shape[0], bias=self.bias is not None)
            self.proj.weight.data = self.weight.data
            self.proj.bias.data = self.bias.data
            self.topk = xhnn.TopK(self.top_k, axis=-1)
            self.softmax = xhnn.Softmax(dim=-1)
            self.mul = xhnn.Mul()
            self.scatter = xhnn.ScatterElements(axis=1)
        else:
            self.proj = self.gate
            self.topk = xhnn.TopK(self.top_k, axis=-1)
            self.softmax = xhnn.Softmax(dim=-1)
            self.mul = xhnn.Mul()
            self.scatter = xhnn.ScatterElements(axis=1)
        return self


def register_wrap_modules(hf_model: GptOssForCausalLM | None = None):
    pass
