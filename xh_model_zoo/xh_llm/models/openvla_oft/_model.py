# Copyright 2025 HOUMO AI
#
# File: _model.py
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

import math
import sys
import types
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import transformers
from torch import Tensor
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
)
from xhquant import nn as xhnn
from xhquant.nn import LLMCacheV2, MaskedSoftmax, RMSNorm
from xhquant.utils.registry import DynamicModule
from xhquant.utils.version_utils import digit_version

from ..builder import XHLLM_TRACEABLE_MODULES

if digit_version(transformers.__version__) < digit_version("4.48.0"):
    from transformers.models.llama.modeling_llama import LlamaSdpaAttention
else:
    class LlamaSdpaAttention(LlamaAttention):
        pass


@XHLLM_TRACEABLE_MODULES.register_module(
    {
        LlamaRotaryEmbedding: "LlamaRotaryEmbedding",
    }
)
class _LlamaRotaryEmbedding(DynamicModule):
    def _setup(self, cfg: Optional[Dict] = None):
        assert "dynamic" not in self.rope_type, f"{self.rope_type} is not supported in dynamic mode"

        # self.max_position_embeddings = max_position_embeddings
        # Build here to make `torch.jit.trace` work.
        self._setup_cos_sin_cache(
            seq_len=self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype
        )
        if hasattr(self, "setup_after_callback"):
            self.setup_after_callback()

    def _setup_cos_sin_cache(self, seq_len, device, dtype):
        """
        4.45 版本实现
        """
        position_ids = torch.arange(0, seq_len, dtype=torch.long, device=device).unsqueeze(0)
        self.inv_freq = self.inv_freq.to(torch.float16)
        cos, sin = super().forward(self.inv_freq, position_ids)
        sin = sin.squeeze(0)
        cos = cos.squeeze(0)
        self.register_buffer("sin_cached", sin.to(device=device, dtype=dtype), persistent=False)
        self.register_buffer("cos_cached", cos.to(device=device, dtype=dtype), persistent=False)

    def _set_dtype(self, dtype: torch.dtype) -> None:
        self.inv_freq = self.inv_freq.to(dtype)
        self._setup_cos_sin_cache(
            seq_len=self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype
        )

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
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling
        cos = cos.to(dtype=x.dtype)
        sin = sin.to(dtype=x.dtype)
        raise NotImplementedError("Advanced RoPE types are not supported yet")
        return cos, sin


@XHLLM_TRACEABLE_MODULES.register_module(
    {
        LlamaAttention: "LlamaAttention",
        LlamaSdpaAttention: "LlamaSdpaAttention",
    }
)
class _LlamaAttention(DynamicModule):
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
        q_embed = (q * cos) + (self.rotate_half(q) * sin)
        k_embed = (k * cos) + (self.rotate_half(k) * sin)
        return q_embed, k_embed

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_seq_length: Optional[Tensor] = None,
        current_input_length: Optional[Tensor] = None,
        position_ids: torch.Tensor = None,
        attention_mask: Optional[Tensor] = None,
        past_k_cache: Optional[Tensor] = None,
        past_v_cache: Optional[Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()
        causal_mask = None
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # cos = self.cos_embeding(position_ids)
        # sin = self.sin_embeding(position_ids)

        cos, sin = position_embeddings

        query_states, key_states = self.apply_rotary_pos_emb(query_states, key_states, cos, sin, unsqueeze_dim=1)

        if self.use_cache:
            key_states = self.k_cache(key_states, past_seq_length, current_input_length, past_k_cache)
            value_states = self.v_cache(value_states, past_seq_length, current_input_length, past_v_cache)

        query_states = query_states * self.kv_scale
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
        # attn_weights = torch.matmul(query_states, key_states) / math.sqrt(self.head_dim) #fp16下会出现nan
        attn_weights = attn_weights + attention_mask
        # attn_weights: Optional[Tensor] = self.masked_softmax(attn_weights, past_seq_length)
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        # TODO:  HMMatMul broadcast
        # value_states = self.value_unsqueeze(value_states)
        # value_states = self.value_expand(value_states)
        # value_states = value_states.reshape(bz, self.num_heads, -1, self.head_dim)
        value_states = torch.repeat_interleave(
            value_states,
            self.num_key_value_groups,
            dim=1,
        )
        attn_output = torch.matmul(attn_weights, value_states)  # [4, 28, 256, 32768], [4, 28, 32768, 128]

        # attn_weights = nn.functional.dropout(
        #     attn_weights, p=self.attention_dropout, training=self.training
        # )

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, -1, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        # # value_states = torch.repeat_interleave(
        # #     value_states,
        # #     self.num_key_value_groups,
        # #     dim=1,
        # # )
        # # query_states [bsz, self.num_key_value_heads, seq_len, self.head_dim]
        # # 官方实现
        # if False:
        #     is_causal = True if causal_mask is None and q_len > 1 else False
        #     attn_output = torch.nn.functional.scaled_dot_product_attention(
        #         query_states,
        #         key_states,
        #         value_states,
        #         attn_mask=None,
        #         dropout_p=self.attention_dropout if self.training else 0.0,
        #         is_causal=is_causal,
        #     )
        # else:
        #     key_states = key_states.transpose(2, 3)
        #     # query_states = query_states / math.sqrt(self.head_dim)
        #     query_states = query_states * self.kv_scale
        #     attn_weights = torch.matmul(query_states, key_states)
        #     # attn_weights = torch.matmul(query_states, key_states) / math.sqrt(self.head_dim) #fp16下会出现nan
        #     attn_weights: Optional[Tensor] = self.masked_softmax(attn_weights, past_seq_length)
        #     # attn_weights = nn.functional.dropout(
        #     #     attn_weights, p=self.attention_dropout, training=self.training
        #     # )
        #     attn_output = torch.matmul(attn_weights, value_states)

        # attn_output = attn_output.transpose(1, 2).contiguous()
        # attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        # attn_output = self.o_proj(attn_output)

        # return attn_output, attn_weights, past_key_value
        return attn_output, None, None

    def _setup_cos_sin_embeding(self):
        cos = self.rotary_emb.cos_cached.squeeze(1)
        sin = self.rotary_emb.sin_cached.squeeze(1)
        num_embeddings, embedding_dim = cos.shape

        self.cos_embeding = nn.Embedding(num_embeddings, embedding_dim)
        self.cos_embeding.weight.data = cos

        self.sin_embeding = nn.Embedding(num_embeddings, embedding_dim)
        self.sin_embeding.weight.data = sin

    def _setup(self, cfg: Optional[Dict] = None):
        if not hasattr(self, "num_key_value_heads"):
            self.num_key_value_heads = self.config.num_key_value_heads
        if not hasattr(self, "num_heads"):
            self.num_heads = self.config.num_attention_heads  # 28
        if not hasattr(self, "hidden_size"):
            self.hidden_size = self.config.hidden_size  # 3584

        self.slice_1 = xhnn.Slice([0], [self.head_dim // 2], [3], [1])
        self.slice_2 = xhnn.Slice([self.head_dim // 2], [sys.maxsize], [3], [1])
        self.masked_softmax = MaskedSoftmax(dim=-1)
        use_cache = cfg.use_cache
        self.use_cache = use_cache
        max_sequence_length = cfg.max_sequence_length
        self.max_sequence_length = max_sequence_length
        input_seq_len = cfg.input_sequence_length
        self.cos_gather = xhnn.Gather(0)
        self.sin_gather = xhnn.Gather(0)

        # self.sin_slice = xhnn.DynamicSlice([input_seq_len], [0], [1])
        # self.cos_slice = xhnn.DynamicSlice([input_seq_len], [0], [1])

        # def _sin_cos_slice_update_cfg(self, cfg: Optional[Dict] = None):
        #     input_seq_len = cfg.input_sequence_length
        #     self.valid_length = [input_seq_len]

        # self.sin_slice._update_cfg = types.MethodType(_sin_cos_slice_update_cfg, self.sin_slice)
        # self.cos_slice._update_cfg = types.MethodType(_sin_cos_slice_update_cfg, self.cos_slice)

        # self.cos_unsqueeze = xhnn.Unsqueeze(0)
        # self.sin_unsqueeze = xhnn.Unsqueeze(0)

        # self.key_unsqueeze = xhnn.Unsqueeze(2)
        # self.value_unsqueeze = xhnn.Unsqueeze(2)

        # if not hasattr(self.rotary_emb, "cos_cached"):
        #     self.rotary_emb.setup_after_callback = self._setup_cos_sin_embeding
        # else:
        #     self._setup_cos_sin_embeding()

        batch_size = cfg.get("batch_size", 1)

        # self.hidden_size = config.hidden_size #3584
        # self.num_heads = config.num_attention_heads #28
        # self.head_dim = self.hidden_size // self.num_heads #128
        # self.num_key_value_heads = config.num_key_value_heads  #4
        # self.num_key_value_groups = self.num_heads // self.num_key_value_heads #7
        # self.max_position_embeddings = config.max_position_embeddings #32768

        # if use_cache:
        #     self.key_expand = xhnn.Expand(
        #         [1, self.num_key_value_heads, self.num_key_value_groups, max_sequence_length, self.head_dim]
        #     )
        #     self.value_expand = xhnn.Expand(
        #         [1, self.num_key_value_heads, self.num_key_value_groups, max_sequence_length, self.head_dim]
        #     )
        # else:
        #     self.key_expand = xhnn.Expand(
        #         [1, self.num_key_value_heads, self.num_key_value_groups, input_seq_len, self.head_dim]
        #     )
        #     self.value_expand = xhnn.Expand(
        #         [1, self.num_key_value_heads, self.num_key_value_groups, input_seq_len, self.head_dim]
        #     )

        # def _expand_update_cfg(self, cfg: Optional[Dict] = None):
        #     input_seq_len = cfg.input_sequence_length
        #     self.size[2] = input_seq_len

        # self.key_expand._update_cfg = types.MethodType(_expand_update_cfg, self.key_expand)
        # self.value_expand._update_cfg = types.MethodType(_expand_update_cfg, self.key_expand)

        # self.key_expand = xhnn.Expand(
        #     [batch_size, self.num_key_value_heads, self.num_key_value_groups, self.head_dim, -1]
        # )
        # self.value_expand = xhnn.Expand(
        #     [batch_size, self.num_key_value_heads, self.num_key_value_groups, -1, self.head_dim]
        # )

        # def _expand_update_cfg(self: xhnn.Expand, cfg: Optional[Dict] = None):
        #     input_seq_len = cfg.input_sequence_length
        #     batch_size = cfg.get("batch_size", 1)
        #     size = list(self.size)
        #     size[0] = batch_size
        #     self.size = type(self.size)(size)

        # self.key_expand._update_cfg = types.MethodType(_expand_update_cfg, self.key_expand)
        # self.value_expand._update_cfg = types.MethodType(_expand_update_cfg, self.value_expand)

        if use_cache:
            cache_axis = cfg.kv_cache.cache_axis
            self.k_cache = LLMCacheV2(axis=cache_axis)
            self.v_cache = LLMCacheV2(axis=cache_axis)
        else:
            self.k_cache = None
            self.v_cache = None

        # self.register_buffer("kv_scale", 1 / torch.tensor(math.sqrt(self.head_dim)))
        _kv_scale = 1 / math.sqrt(self.head_dim)
        self.kv_scale = _kv_scale
        # self.register_buffer("kv_scale", torch.tensor(_kv_scale, dtype=torch.float16), persistent=False)
        return self


@XHLLM_TRACEABLE_MODULES.register_module({LlamaDecoderLayer: "LlamaDecoderLayer"})
class _LlamaDecoderLayer(DynamicModule):
    def forward(
        self,
        hidden_states: torch.Tensor,
        past_seq_length: Optional[Tensor] = None,
        current_input_length: Optional[Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[Tensor] = None,
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
            position_ids=position_ids,
            past_seq_length=past_seq_length,
            current_input_length=current_input_length,
            attention_mask=attention_mask,
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

        outputs = (hidden_states,)

        return outputs

    def _setup(self, cfg: Optional[Dict] = None):
        return self


@XHLLM_TRACEABLE_MODULES.register_module({LlamaRMSNorm: "LlamaRMSNorm"})
class _LlamaRMSNorm(DynamicModule):
    def forward(self, hidden_states):
        return self.norm(hidden_states)

    def _setup(self, cfg: Optional[Dict] = None):
        hidden_size = self.weight.shape[0]
        self.norm = RMSNorm(hidden_size, self.variance_epsilon)
        self.norm.weight.data = self.weight.data.clone()
        return self


@XHLLM_TRACEABLE_MODULES.register_module({transformers.LlamaModel: "LlamaModel"})
class _LlamaModel(DynamicModule):
    def forward(
        self,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_seq_length: Optional[Tensor] = None,
        current_input_length: Optional[Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[Tensor] = None,
        past_key_cache: Optional[List[Tensor]] = None,
        past_value_cache: Optional[List[Tensor]] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        causal_mask = None  # 在Qwen2Attention中处理
        hidden_states = inputs_embeds

        cos = self.cos_embeding(position_ids)
        sin = self.sin_embeding(position_ids)
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        position_embeddings = (cos, sin)
        """
        位置编码和Attention中的旋转编码合并成一个矩阵
        """
        # rotary_matrix = self.rotary_matrix_cache(position_ids)
        for idx, decoder_layer in enumerate(self.layers):
            if self.use_cache:
                _past_k_cache = past_key_cache[idx]
                _past_v_cache = past_value_cache[idx]
            else:
                _past_k_cache = None
                _past_v_cache = None

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                past_seq_length=past_seq_length,
                current_input_length=current_input_length,
                position_ids=position_ids,
                past_k_cache=_past_k_cache,
                past_v_cache=_past_v_cache,
                position_embeddings=position_embeddings,
            )

            hidden_states = layer_outputs[0]
            if self.only_first_block:
                break

        # hidden_states = hidden_states[
        #     :,
        #     -num_logits_to_keep:,
        # ]
        # hidden_states = hidden_states[:, :current_input_length, :]
        # hidden_states = self.llm_gather(hidden_states, current_input_length, num_logits_to_keep)
        # if self.num_logits_to_keep == 0:
        #     # hidden_states = torch_ops_xh2a_slice(hidden_states, [0], [current_input_length], [1], [1])
        #     hidden_states = self.slice(
        #         hidden_states
        #     )  # 此时返回的结果，含有padding,调用者需要根据current_input_length切片
        # else:
        #     # 取最后一个token的输出
        #     hidden_states = self.llm_gather(hidden_states, current_input_length - 1)
        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
        )

    def _setup_cos_sin_embeding(self):
        cos = self.rotary_emb.cos_cached.squeeze(1)
        sin = self.rotary_emb.sin_cached.squeeze(1)
        num_embeddings, embedding_dim = cos.shape

        self.cos_embeding = nn.Embedding(num_embeddings, embedding_dim)
        self.cos_embeding.weight.data = cos

        self.sin_embeding = nn.Embedding(num_embeddings, embedding_dim)
        self.sin_embeding.weight.data = sin

    def _setup(self, cfg: Optional[Dict] = None):
        self.batch_size = cfg.get("batch_size", 1)
        self.only_first_block = cfg.get("only_first_block", False)

        self.num_logits_to_keep = cfg.num_logits_to_keep  # 1,取最后一个token的输出，0,取所有token的输出
        assert self.num_logits_to_keep in [0, 1]

        input_seq_len = cfg.input_sequence_length
        self.slice = xhnn.Slice([0], [input_seq_len], [1], [1])

        def _update_cfg(self, cfg: Optional[Dict] = None):
            input_seq_len = cfg.input_sequence_length
            self.ends = [input_seq_len]

        self.slice._update_cfg = types.MethodType(_update_cfg, self.slice)

        self.llm_gather = xhnn.BatchGather(1)
        self.llm_gather.update_offset_indices(self.batch_size, input_seq_len)

        def _llm_gather_update_cfg(self: xhnn.BatchGather, cfg: Optional[Dict] = None):
            input_seq_len = cfg.input_sequence_length
            batch_size = cfg.get("batch_size", 1)
            self.update_offset_indices(batch_size, input_seq_len)

        self.llm_gather._update_cfg = types.MethodType(_llm_gather_update_cfg, self.llm_gather)
        self.use_cache = cfg.use_cache

        if not hasattr(self.rotary_emb, "cos_cached"):
            self.rotary_emb.setup_after_callback = self._setup_cos_sin_embeding
        else:
            self._setup_cos_sin_embeding()
        return self


@XHLLM_TRACEABLE_MODULES.register_module({transformers.LlamaForCausalLM: "transformers.LlamaForCausalLM"})
class _LlamaForCausalLM(DynamicModule):
    def forward(
        self,
        inputs_embeds: Optional[Tensor] = None,
        past_seq_length: Optional[Tensor] = None,
        current_input_length: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        past_key_cache: Optional[List[Tensor]] = None,
        past_value_cache: Optional[List[Tensor]] = None,
    ):
        # past_seq_length = past_seq_length[0]
        # current_input_length = current_input_length[0]
        # num_logits_to_keep = num_logits_to_keep[0]

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            past_seq_length=past_seq_length,
            current_input_length=current_input_length,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_cache=past_key_cache,
            past_value_cache=past_value_cache,
        )

        # hidden_states = outputs[0]
        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)
        return hidden_states

    def _setup(self, cfg: Optional[Dict] = None):
        return self


def register_wrap_modules():
    pass
