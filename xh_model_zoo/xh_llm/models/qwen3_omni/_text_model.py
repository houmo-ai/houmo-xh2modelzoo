# -*- coding: utf-8 -*-
# Copyright 2025 The Qwen Team, Alibaba Group and The HuggingFace Inc. team. All rights reserved.
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
# File: _text_model.py
# Description:
#   Qwen3-Omni text-tower model adapted for the xh2 model zoo (xh2modelzoo).

import math
import sys
import types
from copy import deepcopy
from typing import Dict, List, Optional, Tuple, Union
from xhquant.api import ConfigDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from torch import Tensor
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeAttention,
    Qwen3MoeDecoderLayer,
    Qwen3MoeForCausalLM,
    Qwen3MoeModel,
    Qwen3MoeRMSNorm,
    Qwen3MoeRotaryEmbedding,
    Qwen3MoeSparseMoeBlock,
)
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeThinkerForConditionalGeneration,
    Qwen3OmniMoeThinkerTextRotaryEmbedding,
    Qwen3OmniMoeThinkerTextAttention,
    Qwen3OmniMoeThinkerTextDecoderLayer,
    Qwen3OmniMoeThinkerTextSparseMoeBlock,
    Qwen3OmniMoeThinkerTextRMSNorm,
    Qwen3OmniMoeThinkerTextModel,
)
from xhquant import nn as xhnn
from xhquant.nn import LLMCacheV2, MaskedSoftmax, RMSNorm, LLMCache
from xhquant.nn.modules.moeblock import MoeBlock
from xhquant.utils import digit_version
from xhquant.utils.registry import DynamicModule

from ..builder import XHLLM_TRACEABLE_MODULES


@XHLLM_TRACEABLE_MODULES.register_module(
    {
        Qwen3MoeRotaryEmbedding: "Qwen3MoeRotaryEmbedding",
        Qwen3OmniMoeThinkerTextRotaryEmbedding: "Qwen3OmniMoeThinkerTextRotaryEmbedding",
    })
class _Qwen3MoeRotaryEmbedding(DynamicModule):
    def _setup(self, cfg: Optional[Dict] = None):
        assert "dynamic" not in self.rope_type, f"{self.rope_type} is not supported in dynamic mode"

        # self.max_position_embeddings = max_position_embeddings
        # Build here to make `torch.jit.trace` work.
        self._setup_cos_sin_cache(seq_len=self.max_seq_len_cached, dtype=self.inv_freq.dtype)
        if hasattr(self, "setup_after_callback"):
            self.setup_after_callback()

    def _setup_cos_sin_cache(self, seq_len, dtype):
        """
        4.45 版本实现
        """
        position_ids = torch.arange(0, seq_len, dtype=torch.long, device=self.inv_freq.device).unsqueeze(0)
        # self.inv_freq = self.inv_freq.to(torch.float16)
        inv_freq = self.inv_freq.to(torch.float32)
        device = self.inv_freq.device
        if torch.cuda.is_available() and inv_freq.device.type != "cuda":
            inv_freq = self.inv_freq.cuda()

        cos, sin = self.forward(inv_freq, position_ids)
        cos = cos.to(device)
        sin = sin.to(device)
        sin = sin.squeeze(0)
        cos = cos.squeeze(0)

        # TODO:临时处理
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

        self.register_buffer("sin_cached", sin.to(dtype=dtype), persistent=False)
        self.register_buffer("cos_cached", cos.to(dtype=dtype), persistent=False)
        # self.sin_cached = nn.Parameter(sin.to(device=device, dtype=dtype), requires_grad=False)
        # self.cos_cached = nn.Parameter(cos.to(device=device, dtype=dtype), requires_grad=False)

    def _set_dtype(self, dtype: torch.dtype) -> None:
        self.inv_freq = self.inv_freq.to(dtype)
        self._setup_cos_sin_cache(seq_len=self.max_seq_len_cached, dtype=self.inv_freq.dtype)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        """
        4.37 版本实现
        """
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)

        # Core RoPE block
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        # with torch.autocast(device_type=device_type, enabled=False):
        #     freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        #     emb = torch.cat((freqs, freqs), dim=-1)
        #     cos = emb.cos()
        #     sin = emb.sin()
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
        Qwen3MoeAttention: "Qwen3MoeAttention",
        Qwen3OmniMoeThinkerTextAttention: "Qwen3OmniMoeThinkerTextAttention",
    })
class _Qwen3MoeAttention(DynamicModule):
    def rotate_half(self, x: Tensor):
        """Rotates half the hidden dims of the input."""
        # x1 = x[..., : x.shape[-1] // 2]
        # x2 = x[..., x.shape[-1] // 2 :]
        # x1 = torch_ops_xh2a_slice(x, [0], [self.head_dim // 2], [3], [1])
        # x2 = torch_ops_xh2a_slice(x, [self.head_dim // 2], [sys.maxsize], [3], [1])
        x1 = self.slice_1(x)
        x2 = self.slice_2(x)
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(self, q: Tensor, k: Tensor, cos: Tensor, sin: Tensor, unsqueeze_dim: int = 1):
        # cos = cos.unsqueeze(unsqueeze_dim)
        # sin = sin.unsqueeze(unsqueeze_dim)
        # cos = self.cos_unsqueeze(cos)
        # sin = self.sin_unsqueeze(sin)
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
        # position_ids: torch.Tensor = None,
        past_k_cache: Optional[Tensor] = None,
        past_v_cache: Optional[Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()
        causal_mask = None
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = self.q_norm(query_states.view(bsz, q_len, self.num_heads, self.head_dim)).transpose(1, 2)
        key_states = self.k_norm(key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # cos = self.rotary_emb.cos_cached[position_ids]
        # sin = self.rotary_emb.sin_cached[position_ids]

        # cos = self.cos_slice(self.rotary_emb.cos_cached, past_seq_length)
        # sin = self.sin_slice(self.rotary_emb.sin_cached, past_seq_length)

        # cos = self.cos_embeding(position_ids)
        # sin = self.sin_embeding(position_ids)
        cos, sin = position_embeddings

        # cos = self.cos_unsqueeze(cos)
        # sin = self.sin_unsqueeze(sin)
        query_states, key_states = self.apply_rotary_pos_emb(query_states, key_states, cos, sin, unsqueeze_dim=1)

        if self.use_cache:
            key_states = self.k_cache(key_states, past_seq_length, current_input_length, past_k_cache)
            value_states = self.v_cache(value_states, past_seq_length, current_input_length, past_v_cache)

        # if self.use_cache:
        #     bz, num_heads, seq_len, head_dim = past_k_cache.shape
        # else:
        #     bz, num_heads, seq_len, head_dim = key_states.shape

        query_states = query_states * self.kv_scale

        # query_states [bsz, self.num_key_value_heads, seq_len, self.head_dim]

        # query_states = query_states / math.sqrt(self.head_dim)

        key_states = key_states.transpose(2, 3)
        # TODO: HMMatMul broadcast
        # key_states = self.key_unsqueeze(key_states)
        # key_states = self.key_expand(key_states)
        # key_states = key_states.reshape(bz, self.num_heads, self.head_dim, -1)

        key_states = torch.repeat_interleave(
            key_states,
            self.num_key_value_groups,
            dim=1,
        )

        attn_weights = torch.matmul(query_states, key_states)  # [4, 28, 256, 128], [4, 28, 128, 32768]
        # attn_weights = self.key_group_broadcast_matmul(query_states, key_states)
        # attn_weights = torch.matmul(query_states, key_states) / math.sqrt(self.head_dim) #fp16下会出现nan
        attn_weights: Optional[Tensor] = self.masked_softmax(attn_weights, past_seq_length)

        # TODO: HMMatMul broadcast
        # value_states = self.value_unsqueeze(value_states)
        # value_states = self.value_expand(value_states)
        # value_states = value_states.reshape(bz, self.num_heads, -1, self.head_dim)
        value_states = torch.repeat_interleave(
            value_states,
            self.num_key_value_groups,
            dim=1,
        )
        attn_output = torch.matmul(attn_weights, value_states)  # [4, 28, 256, 32768], [4, 28, 32768, 128]
        # attn_output = self.value_group_broadcast_matmul(attn_weights, value_states)

        # attn_weights = nn.functional.dropout(
        #     attn_weights, p=self.attention_dropout, training=self.training
        # )

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.config.num_attention_heads * self.head_dim)

        attn_output = self.o_proj(attn_output)

        # return attn_output, attn_weights, past_key_value
        return attn_output, None, None

    def _setup(self, cfg: Optional[Dict] = None):
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

        self.masked_softmax = MaskedSoftmax(dim=-1)
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
                attention_max_length=-1,
            )
            self.v_cache = LLMCache(
                axis=cache_axis,
                attention_max_length=-1,
            )
        else:
            self.k_cache = None
            self.v_cache = None
        _kv_scale = 1 / math.sqrt(self.head_dim)
        self.kv_scale = _kv_scale
        return self


@XHLLM_TRACEABLE_MODULES.register_module(
    {
        Qwen3MoeDecoderLayer: "Qwen3MoeDecoderLayer",
        Qwen3OmniMoeThinkerTextDecoderLayer: "Qwen3OmniMoeThinkerTextDecoderLayer",
    })
class _Qwen3MoeDecoderLayer(DynamicModule):
    def forward(
        self,
        hidden_states: torch.Tensor,
        past_seq_length: Optional[Tensor] = None,
        current_input_length: Optional[Tensor] = None,
        # position_ids: Optional[torch.LongTensor] = None,
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
        hidden_states = self.mlp(hidden_states)
        if isinstance(hidden_states, tuple):
            hidden_states, router_logits = hidden_states
        else:
            router_logits = None
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        return outputs

    def _setup(self, cfg: Optional[Dict] = None):
        return self


@XHLLM_TRACEABLE_MODULES.register_module(
    {
        Qwen3MoeSparseMoeBlock: "Qwen3MoeSparseMoeBlock",
        Qwen3OmniMoeThinkerTextSparseMoeBlock: "Qwen3OmniMoeThinkerTextSparseMoeBlock",
    })
class _Qwen3MoeSparseMoeBlock(DynamicModule):
    def forward(self, hidden_states):
        ori_hidden_states = hidden_states
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        # Use nn.Linear gate (created in _setup) for FX compatibility
        router_logits = self.gate(hidden_states)
        routing_weights = F.softmax(router_logits, dim=-1)
        return self.moeblock(
            ori_hidden_states,
            routing_weights,
        )

    def _setup(self, cfg: Optional[Dict] = None):
        self.input_seq_len = cfg.input_sequence_length
        self.batch_size = cfg.batch_size
        self.device = self.gate.weight.device

        # In Qwen3Omni, top_k/norm_topk_prob live on the SparseMoeBlock itself
        # (self), and self.gate is already a plain nn.Linear; experts is a
        # fused Qwen3OmniMoeThinkerTextExperts object. Older transformers
        # versions exposed these on self.gate (TopKRouter); fall back if so.
        top_k = getattr(self, "top_k", None)
        if top_k is None:
            top_k = self.gate.top_k
        norm_topk_prob = getattr(self, "norm_topk_prob", None)
        if norm_topk_prob is None:
            norm_topk_prob = self.gate.norm_topk_prob
        num_experts = getattr(self, "num_experts", None)
        if num_experts is None:
            num_experts = getattr(self.experts, "num_experts", None)
        if num_experts is None:
            num_experts = len(self.experts)

        # Ensure gate is a plain nn.Linear for FX tracing (it already is on
        # newer transformers, but older TopKRouter-based gates need a rewrap).
        if not isinstance(self.gate, nn.Linear):
            gate_weight = self.gate.weight.data  # (num_experts, hidden_dim)
            gate_linear = nn.Linear(gate_weight.shape[1], gate_weight.shape[0], bias=False)
            gate_linear.weight = nn.Parameter(gate_weight)
            self.gate = gate_linear

        experts = self.experts
        if isinstance(experts, nn.ModuleList):
            # transformers 4.57+: experts is a ModuleList of per-expert MLPs.
            first = experts[0]
            act_fn_name = first.act_fn._get_name().lower()
            intermediate_dim = first.intermediate_size
            gate_proj_weight = torch.stack([e.gate_proj.weight.data for e in experts], dim=0)  # (E, I, H)
            up_proj_weight = torch.stack([e.up_proj.weight.data for e in experts], dim=0)      # (E, I, H)
            down_proj_weight = torch.stack([e.down_proj.weight.data for e in experts], dim=0)  # (E, H, I)
        else:
            # Fused Qwen3OmniMoeThinkerTextExperts object (older transformers).
            act_fn_name = experts.act_fn._get_name().lower()
            intermediate_dim = experts.intermediate_dim
            gate_up_proj = experts.gate_up_proj.data  # (E, 2*I, H)
            down_proj = experts.down_proj.data  # (E, H, I)

            gate_proj_weight = gate_up_proj[:, :intermediate_dim, :]   # (E, I, H)
            up_proj_weight = gate_up_proj[:, intermediate_dim:, :]     # (E, I, H)
            down_proj_weight = down_proj                                # (E, H, I)

        self.moeblock = MoeBlock(act_fn_name, top_k, norm_topk_prob)

        self.moeblock.expert_gate_proj_weight = torch.nn.Parameter(gate_proj_weight.to(self.device))
        self.moeblock.expert_gate_proj_bias = None
        self.moeblock.expert_up_proj_weight = torch.nn.Parameter(up_proj_weight.to(self.device))
        self.moeblock.expert_up_proj_bias = None
        self.moeblock.expert_down_proj_weight = torch.nn.Parameter(down_proj_weight.to(self.device))
        self.moeblock.expert_down_proj_bias = None

        del self.experts
        torch.cuda.empty_cache()

        return self


@XHLLM_TRACEABLE_MODULES.register_module(
    {
        Qwen3MoeRMSNorm: "Qwen3MoeRMSNorm",
        Qwen3OmniMoeThinkerTextRMSNorm: "Qwen3OmniMoeThinkerTextRMSNorm",
    })
class _Qwen3MoeRMSNorm(DynamicModule):
    def forward(self, hidden_states):
        return self.norm(hidden_states)

    def _setup(self, cfg: Optional[Dict] = None):
        hidden_size = self.weight.shape[0]
        self.norm = RMSNorm(hidden_size, self.variance_epsilon)
        self.norm.weight = nn.Parameter(deepcopy(self.weight.data))
        return self


@XHLLM_TRACEABLE_MODULES.register_module(
    {
        Qwen3MoeModel: "Qwen3MoeModel",
        Qwen3OmniMoeThinkerTextModel: "Qwen3OmniMoeThinkerTextModel",
    })
class _Qwen3MoeModel(DynamicModule):
    def forward(
        self,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        time_position_ids: Optional[Tensor] = None,
        height_position_ids: Optional[Tensor] = None,
        width_position_ids: Optional[Tensor] = None,
        past_seq_length: Optional[Tensor] = None,
        current_input_length: Optional[Tensor] = None,
        deepstack_visual_embed_0: Optional[Tensor] = None,
        deepstack_visual_embed_1: Optional[Tensor] = None,
        deepstack_visual_embed_2: Optional[Tensor] = None,
        past_key_cache: Optional[List[Tensor]] = None,
        past_value_cache: Optional[List[Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,

    ) -> Union[Tuple, BaseModelOutputWithPast]:

        causal_mask = None  # 在Qwen2Attention中处理
        hidden_states = inputs_embeds
        accept_hidden_states = inputs_embeds if self.accept_hidden_layer == 0 else None

        if self.use_multimodal_position_ids:
            cos_cache = self.rotary_emb.cos_cached[0, 0]
            sin_cache = self.rotary_emb.sin_cached[0, 0]

            time_cos = cos_cache[time_position_ids] * self.time_mask
            time_sin = sin_cache[time_position_ids] * self.time_mask
            height_cos = cos_cache[height_position_ids] * self.height_mask
            height_sin = sin_cache[height_position_ids] * self.height_mask
            width_cos = cos_cache[width_position_ids] * self.width_mask
            width_sin = sin_cache[width_position_ids] * self.width_mask

            cos = (time_cos + height_cos + width_cos).unsqueeze(0).unsqueeze(0)
            sin = (time_sin + height_sin + width_sin).unsqueeze(0).unsqueeze(0)
        else:
            cos = self.cos_slice(self.rotary_emb.cos_cached, past_seq_length)
            sin = self.sin_slice(self.rotary_emb.sin_cached, past_seq_length)

        # cos = self.cos_embeding(position_ids)
        # sin = self.sin_embeding(position_ids)
        # cos = cos.unsqueeze(1)
        # sin = sin.unsqueeze(1)

        position_embeddings = (cos, sin)

        for idx, decoder_layer in enumerate(self.layers):
            # print("processing: ", idx)
            if self.use_cache:
                _past_k_cache = past_key_cache[idx]
                _past_v_cache = past_value_cache[idx]
            else:
                _past_k_cache = None
                _past_v_cache = None

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                past_seq_length=past_seq_length,
                current_input_length=current_input_length,
                # position_ids=position_ids,
                past_k_cache=_past_k_cache,
                past_v_cache=_past_v_cache,
                position_embeddings=position_embeddings,
            )

            hidden_states = layer_outputs[0]
            if deepstack_visual_embed_0 is not None and idx == self.deepstack_inject_layers[0]:
                hidden_states = hidden_states + deepstack_visual_embed_0
            if deepstack_visual_embed_1 is not None and idx == self.deepstack_inject_layers[1]:
                hidden_states = hidden_states + deepstack_visual_embed_1
            if deepstack_visual_embed_2 is not None and idx == self.deepstack_inject_layers[2]:
                hidden_states = hidden_states + deepstack_visual_embed_2
            if self.accept_hidden_capture_layer_idx is not None and idx == self.accept_hidden_capture_layer_idx:
                accept_hidden_states = hidden_states
            # break
            if self.only_first_block:
                break

        if accept_hidden_states is None:
            raise RuntimeError(f"accept_hidden_layer={self.accept_hidden_layer} was not captured")

        # hidden_states = hidden_states[
        #     :,
        #     -num_logits_to_keep:,
        # ]
        # hidden_states = hidden_states[:, :current_input_length, :]
        # hidden_states = self.llm_gather(hidden_states, current_input_length, num_logits_to_keep)
        if self.num_logits_to_keep == 0:  # for PPL task
            # hidden_states = torch_ops_xh2a_slice(hidden_states, [0], [current_input_length], [1], [1])
            # hidden_states = self.slice(
            #     hidden_states
            # )  # 此时返回的结果，含有padding,调用者需要根据current_input_length切片
            pass
        else:
            # 取最后一个token的输出
            if accept_hidden_states is not None and not self.prefill_full_accept_hidden:
                accept_hidden_states = self.llm_gather(accept_hidden_states, current_input_length - 1)
            hidden_states = self.llm_gather(hidden_states, current_input_length - 1)
        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            hidden_states=(accept_hidden_states,),
        )

    def _setup_cos_sin_embeding(self):
        self.rotary_emb.cos_cached
        self.rotary_emb.sin_cached

    def _setup(self, cfg: Optional[Dict] = None):
        self.batch_size = cfg.get("batch_size", 1)
        self.only_first_block = cfg.get("only_first_block", False)
        # Keep this aligned with Qwen3OmniMoeThinkerTextModel.forward, which adds
        # the three deepstack feature tensors after the first three decoder layers.
        self.deepstack_inject_layers = [0, 1, 2]
        self.use_multimodal_position_ids = bool(cfg.get("use_multimodal_position_ids", False))
        self.prefill_full_accept_hidden = bool(cfg.get("prefill_full_accept_hidden", False))
        accept_hidden_layer = cfg.get("accept_hidden_layer", 0)
        self.accept_hidden_layer = max(int(accept_hidden_layer), 0)
        if self.accept_hidden_layer == 0:
            self.accept_hidden_capture_layer_idx = None
        else:
            self.accept_hidden_capture_layer_idx = self.accept_hidden_layer - 1
        # max_seq_len = cfg.max_sequence_length
        # self.rotary_matrix_cache = RotaryMatrixCache(self.rotary_emb, max_seq_len)

        self.num_logits_to_keep = cfg.num_logits_to_keep  # 1,取最后一个token的输出，0,取所有token的输出
        assert self.num_logits_to_keep in [0, 1]

        input_seq_len = cfg.input_sequence_length
        self.slice = xhnn.Slice([0], [input_seq_len], [1], [1])

        self.llm_gather = xhnn.BatchGather(1)
        self.llm_gather.update_offset_indices(self.batch_size, input_seq_len)

        def _llm_gather_update_cfg(self: xhnn.BatchGather, cfg: Optional[Dict] = None):
            self.num_logits_to_keep = cfg.num_logits_to_keep
            input_seq_len = cfg.input_sequence_length
            batch_size = cfg.get("batch_size", 1)
            self.update_offset_indices(batch_size, input_seq_len)

        self.llm_gather._update_cfg = types.MethodType(_llm_gather_update_cfg, self.llm_gather)

        def _slice_update_cfg(self, cfg: Optional[Dict] = None):
            input_seq_len = cfg.input_sequence_length
            self.ends = [input_seq_len]

        self.slice._update_cfg = types.MethodType(_slice_update_cfg, self.slice)
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

        rope_scaling = None
        mrope_section = getattr(self.rotary_emb, "mrope_section", None)
        if mrope_section is None:
            rope_scaling = getattr(getattr(self, "config", None), "rope_scaling", None)
            if rope_scaling is None:
                rope_scaling = getattr(getattr(self, "config", None), "rope_parameters", None)
            if isinstance(rope_scaling, dict):
                mrope_section = rope_scaling.get("mrope_section")
        if mrope_section is None:
            mrope_section = [24, 20, 20]
        if rope_scaling is None:
            rope_scaling = getattr(getattr(self, "config", None), "rope_scaling", None)
            if rope_scaling is None:
                rope_scaling = getattr(getattr(self, "config", None), "rope_parameters", None)
        mrope_interleaved = True
        if isinstance(rope_scaling, dict):
            mrope_interleaved = bool(rope_scaling.get("mrope_interleaved", rope_scaling.get("interleaved", True)))

        section_sizes = [int(value) for value in mrope_section]
        rotary_half_dim = sum(section_sizes)
        time_mask = torch.ones(rotary_half_dim, dtype=torch.float16)
        height_mask = torch.zeros(rotary_half_dim, dtype=torch.float16)
        width_mask = torch.zeros(rotary_half_dim, dtype=torch.float16)
        if mrope_interleaved:
            def _interleaved_ids(offset: int, section_size: int) -> torch.Tensor:
                end = min(section_size * 3, rotary_half_dim)
                if end <= offset:
                    return torch.empty(0, dtype=torch.long)
                return torch.arange(offset, end, 3)

            height_ids = _interleaved_ids(1, section_sizes[1])
            width_ids = _interleaved_ids(2, section_sizes[2])
            height_mask[height_ids] = 1
            width_mask[width_ids] = 1
            time_mask[height_ids] = 0
            time_mask[width_ids] = 0
        else:
            time_mask.zero_()
            time_mask[: section_sizes[0]] = 1
            height_start = section_sizes[0]
            height_end = height_start + section_sizes[1]
            height_mask[height_start:height_end] = 1
            width_mask[height_end:] = 1
        self.register_buffer("time_mask", torch.cat([time_mask, time_mask], dim=0), persistent=False)
        self.register_buffer("height_mask", torch.cat([height_mask, height_mask], dim=0), persistent=False)
        self.register_buffer("width_mask", torch.cat([width_mask, width_mask], dim=0), persistent=False)

        return self


@XHLLM_TRACEABLE_MODULES.register_module(
    {
        Qwen3MoeForCausalLM: "Qwen3MoeForCausalLM",
        Qwen3OmniMoeThinkerForConditionalGeneration: "Qwen3OmniMoeThinkerForConditionalGeneration",
    }
)
class _Qwen3OmniMoeThinkerForConditionalGeneration(DynamicModule):
    def forward(
        self,
        inputs_embeds: Optional[Tensor] = None,
        time_position_ids: Optional[Tensor] = None,
        height_position_ids: Optional[Tensor] = None,
        width_position_ids: Optional[Tensor] = None,
        past_seq_length: Optional[Tensor] = None,
        current_input_length: Optional[Tensor] = None,
        deepstack_visual_embed_0: Optional[Tensor] = None,
        deepstack_visual_embed_1: Optional[Tensor] = None,
        deepstack_visual_embed_2: Optional[Tensor] = None,
        past_key_cache: Optional[List[Tensor]] = None,
        past_value_cache: Optional[List[Tensor]] = None,
    ):
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            time_position_ids=time_position_ids,
            height_position_ids=height_position_ids,
            width_position_ids=width_position_ids,
            past_seq_length=past_seq_length,
            current_input_length=current_input_length,
            deepstack_visual_embed_0=deepstack_visual_embed_0,
            deepstack_visual_embed_1=deepstack_visual_embed_1,
            deepstack_visual_embed_2=deepstack_visual_embed_2,
            past_key_cache=past_key_cache,
            past_value_cache=past_value_cache,
        )
        hidden_states = outputs.last_hidden_state
        accept_hidden_states = outputs.hidden_states[0]
        logits = self.lm_head(hidden_states)
        return logits, accept_hidden_states

    def _setup(self, cfg: Optional[Dict] = None):
        return self


def register_wrap_modules():
    pass
