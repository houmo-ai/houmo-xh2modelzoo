# Copyright 2025 HOUMO AI
#
# File: _model.py
# Description:
#   Model model implementation.
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
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from torch import Tensor
import sys
import importlib
import os
user_home = os.path.expanduser("~")
target_path = os.path.join(
    user_home,
    ".cache/huggingface/modules/transformers_modules/Eagle-Block2A-2B-v2"
)
sys.path.append(target_path) # modeling.py

eagle3_full_module_name = "transformers_modules.Eagle-Block2A-2B-v2.modeling_eagle3_vl"
modeling_eagle3_vl = importlib.import_module(eagle3_full_module_name)

siglip2_full_module_name = "transformers_modules.Eagle-Block2A-2B-v2.modeling_siglip2"
modeling_siglip2 = importlib.import_module(siglip2_full_module_name)

NewModel = modeling_gte.NewModel


NTKScalingRotaryEmbedding = modeling_gte.NTKScalingRotaryEmbedding
NewAttention = modeling_gte.NewAttention
NewLayer = modeling_gte.NewLayer
NewEncoder = modeling_gte.NewEncoder

# DeepseekV3Model = modeling_gte.DeepseekV3Model
# DeepseekV3ForCausalLM = modeling_gte.DeepseekV3ForCausalLM

from transformers.modeling_outputs import BaseModelOutputWithPast
# from transformers.models.qwen3_moe.modeling_qwen3_moe import (
#     Qwen3MoeAttention,
#     Qwen3MoeDecoderLayer,
#     Qwen3MoeForCausalLM,
#     Qwen3MoeModel,
#     Qwen3MoeRMSNorm,
#     Qwen3MoeRotaryEmbedding,
#     Qwen3MoeSparseMoeBlock,
# )
from xhquant import nn as xhnn
from xhquant.nn import LLMCacheV2, MaskedSoftmax, RMSNorm
from xhquant.nn.modules.moeblock import MoeBlock
from xhquant.utils import digit_version
from xhquant.utils.registry import DynamicModule

from ..builder import XHLLM_TRACEABLE_MODULES
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    ModelOutput,
)

@XHLLM_TRACEABLE_MODULES.register_module(
    {
        NTKScalingRotaryEmbedding: "NTKScalingRotaryEmbedding",
    }
)
class _NTKScalingRotaryEmbedding(DynamicModule):
    def _setup(self, cfg: Optional[Dict] = None):
        # assert "dynamic" not in self.rope_type, f"{self.rope_type} is not supported in dynamic mode"

        # self.max_position_embeddings = max_position_embeddings
        # Build here to make `torch.jit.trace` work.
        self.max_seq_len_cached = 8192
        self._setup_cos_sin_cache(seq_len=self.max_seq_len_cached, dtype=self.inv_freq.dtype)
        if hasattr(self, "setup_after_callback"):
            self.setup_after_callback()

    def _setup_cos_sin_cache(self, seq_len, dtype):
        """
        4.45 版本实现
        """
        position_ids = torch.arange(0, seq_len, dtype=torch.long, device=self.inv_freq.device).unsqueeze(0)
        # self.inv_freq = self.inv_freq.to(torch.float16)
        # cos, sin = self.forward(self.inv_freq, position_ids)

        inv_freq = self.inv_freq.to(torch.float32)
        device = self.inv_freq.device
        if torch.cuda.is_available() and inv_freq.device.type != "cuda":
            inv_freq = self.inv_freq.cuda()
        cos, sin = self.forward(self.inv_freq, position_ids)
        cos = cos.to(device)
        sin = sin.to(device)

        sin = sin.squeeze(0)
        cos = cos.squeeze(0)
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
        # if "dynamic" in self.rope_type:
        #     self._dynamic_frequency_update(position_ids, device=x.device)

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
        cos = cos #* self.attention_scaling
        sin = sin #* self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


@XHLLM_TRACEABLE_MODULES.register_module(
    {
        NewAttention: "NewAttention",
    }
)
class _NewAttention(DynamicModule):
    def rotate_half(self, x: Tensor):
        """Rotates half the hidden dims of the input."""
        # x1 = x[..., : x.shape[-1] // 2]
        # x2 = x[..., x.shape[-1] // 2 :]
        # x1 = torch_ops_xh2a_slice(x, [0], [self.head_dim // 2], [3], [1])
        # x2 = torch_ops_xh2a_slice(x, [self.head_dim // 2], [sys.maxsize], [3], [1])
        x1 = self.rotate_half_slice_1(x)
        x2 = self.rotate_half_slice_2(x)
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(self, q: Tensor, k: Tensor, cos: Tensor, sin: Tensor, unsqueeze_dim: int = 1):
        # cos = cos.unsqueeze(unsqueeze_dim)
        # sin = sin.unsqueeze(unsqueeze_dim)
        # cos = self.cos_unsqueeze(cos)
        # sin = self.sin_unsqueeze(sin)

        # b, h, s, d = q.shape
        # q = q.view(b, h, s, 32, 2).transpose(4, 3).reshape(b, h, s, d)

        # b, h, s, d = k.shape
        # k = k.view(b, h, s, 32, 2).transpose(4, 3).reshape(b, h, s, d)

        if self.enable_rope:
            q_embed = self.rope(q, cos, sin)
            k_embed = self.rope(k, cos, sin)
        else:
            q_embed = (q * cos) + (self.rotate_half(q) * sin)
            k_embed = (k * cos) + (self.rotate_half(k) * sin)

        # q_embed = self.rope(q, cos, sin)
        # k_embed = self.rope(k, cos, sin)
        return q_embed, k_embed

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_seq_length: Optional[Tensor] = None,
        current_input_length: Optional[Tensor] = None,
        position_ids: torch.Tensor = None,
        past_k_cache: Optional[Tensor] = None,
        past_v_cache: Optional[Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()
        causal_mask = None
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)

        cos, sin = position_embeddings

        # cos = self.cos_unsqueeze(cos)
        # sin = self.sin_unsqueeze(sin)
        query_states, key_states = self.apply_rotary_pos_emb(query_states, key_states, cos, sin, unsqueeze_dim=1)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        if self.use_cache:
            key_states = self.k_cache(key_states, past_seq_length, current_input_length, past_k_cache)
            value_states = self.v_cache(value_states, past_seq_length, current_input_length, past_v_cache)

        query_states = query_states * self.kv_scale

        # query_states [bsz, self.num_key_value_heads, seq_len, self.head_dim]

        # query_states = query_states / math.sqrt(self.head_dim)

        key_states = key_states.transpose(2, 3)
        # TODO: HMMatMul broadcast
        # key_states = self.key_unsqueeze(key_states)
        # key_states = self.key_expand(key_states)
        # key_states = key_states.reshape(bz, self.num_heads, self.head_dim, -1)

        # key_states = torch.repeat_interleave(
        #     key_states,
        #     self.num_key_value_groups,
        #     dim=1,
        # )

        attn_weights = torch.matmul(query_states, key_states)  # [4, 28, 256, 128], [4, 28, 128, 32768]
        # attn_weights = self.key_group_broadcast_matmul(query_states, key_states)
        # attn_weights = torch.matmul(query_states, key_states) / math.sqrt(self.head_dim) #fp16下会出现nan
        
        
        # attn_weights: Optional[Tensor] = self.masked_softmax(attn_weights, past_seq_length)

        attn_weights = self.maskedadd(attn_weights, attention_mask)
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)


        # TODO: HMMatMul broadcast
        # value_states = self.value_unsqueeze(value_states)
        # value_states = self.value_expand(value_states)
        # value_states = value_states.reshape(bz, self.num_heads, -1, self.head_dim)
        # value_states = torch.repeat_interleave(
        #     value_states,
        #     self.num_key_value_groups,
        #     dim=1,
        # )
        attn_output = torch.matmul(attn_weights, value_states)  # [4, 28, 256, 32768], [4, 28, 32768, 128]
        # attn_output = self.value_group_broadcast_matmul(attn_weights, value_states)

        # attn_weights = nn.functional.dropout(
        #     attn_weights, p=self.attention_dropout, training=self.training
        # )

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, -1, self.all_head_size)

        attn_output = self.o_proj(attn_output)

        # return attn_output, attn_weights, past_key_value
        return attn_output, None, None   


    def _setup(self, cfg: Optional[Dict] = None):
        self.enable_rope = cfg.get("enable_rope", True)
        if self.enable_rope:
            self.rope = xhnn.Rope()

        if not hasattr(self, "num_key_value_heads"):
            self.num_key_value_heads = self.config.num_hidden_layers

        if not hasattr(self, "num_heads"):
            self.num_heads = self.config.num_attention_heads  # 28

        # self.attn_hidden_dim = self.head_dim * self.num_heads
        self.head_dim = 64

        self.rotate_half_slice_1 = xhnn.Slice([0], [self.head_dim // 2], [3], [1])
        self.rotate_half_slice_2 = xhnn.Slice([self.head_dim // 2], [sys.maxsize], [3], [1])
        self.masked_softmax = MaskedSoftmax(dim=-1)
        use_cache = cfg.use_cache
        self.use_cache = use_cache
        max_sequence_length = cfg.max_sequence_length
        self.max_sequence_length = max_sequence_length
        input_seq_len = cfg.input_sequence_length
        

        
        weight = self.qkv_proj.weight.data.clone()
        bias = self.qkv_proj.bias.data.clone()

        dim = weight.shape[0] // 3

        weight = weight.permute(1, 0).reshape(dim, 3, dim).permute(1, 2, 0)
        bias = bias.reshape(3, dim)
        self.q_proj = nn.Linear(dim, dim, bias=True)
        self.k_proj = nn.Linear(dim, dim, bias=True)
        self.v_proj = nn.Linear(dim, dim, bias=True)

        self.q_proj.weight.data = weight[0]
        self.q_proj.bias.data = bias[0]
        self.k_proj.weight.data = weight[1]
        self.k_proj.bias.data = bias[1]
        self.v_proj.weight.data = weight[2]
        self.v_proj.bias.data = bias[2]
        self.maskedadd = xhnn.MaskedAdd()

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
        self.kv_scale = _kv_scale
        # self.register_buffer("kv_scale", torch.tensor(_kv_scale, dtype=torch.float16), persistent=False)
        # self.register_parameter(
        #     "kv_scale", nn.Parameter(torch.tensor([_kv_scale], dtype=torch.float16), requires_grad=False)
        # )
        return self


@XHLLM_TRACEABLE_MODULES.register_module(
    {
        NewLayer: "NewLayer",
    }
)
class _NewLayer(DynamicModule):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask = None,
        past_seq_length: Optional[Tensor] = None,
        current_input_length: Optional[Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
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
        residual = hidden_states # 20.7344

        # hidden_states = self.input_layernorm(hidden_states)
        # Self Attention

        hidden_states, self_attn_weights, present_key_value = self.attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            # position_ids=position_ids,
            # past_seq_length=past_seq_length,
            # current_input_length=current_input_length,
            # past_k_cache=past_k_cache,
            # past_v_cache=past_v_cache,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + hidden_states

        hidden_states = self.attn_ln(hidden_states)

        # Fully Connected
        residual = hidden_states
        hidden_states = self.mlp(hidden_states)
        if isinstance(hidden_states, tuple):
            hidden_states, router_logits = hidden_states
        else:
            router_logits = None
        
        hidden_states = residual + hidden_states
        hidden_states = self.mlp_ln(hidden_states)
        
        outputs = (hidden_states,)
        return outputs

    def _setup(self, cfg: Optional[Dict] = None):
        return self



@XHLLM_TRACEABLE_MODULES.register_module(
    {
        NewEncoder: "NewEncoder",
    }
)
class _NewEncoder(DynamicModule):
    def forward(
        self,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        attention_mask= None,
        past_seq_length: Optional[Tensor] = None,
        current_input_length: Optional[Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_cache: Optional[List[Tensor]] = None,
        past_value_cache: Optional[List[Tensor]] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        causal_mask = None  # 在Qwen2Attention中处理
        hidden_states = inputs_embeds

        cos = self.cos_embeding(position_ids) # # [bs, seq_len, 1, dim]
        sin = self.sin_embeding(position_ids)
        cos = cos.unsqueeze(2)
        sin = sin.unsqueeze(2)

        position_embeddings = (cos, sin)

        for idx, decoder_layer in enumerate(self.layer):
            print("processing: ", idx)
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
            # break
            if self.only_first_block:
                break


        return BaseModelOutput(
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

        # self.rotary_emb = self.layers[0].self_attn.rotary_emb
        if not hasattr(self.rotary_emb, "cos_cached"):
            self.rotary_emb.setup_after_callback = self._setup_cos_sin_embeding
        else:
            self._setup_cos_sin_embeding()

        return self


@XHLLM_TRACEABLE_MODULES.register_module(
    {
        NewModel: "NewModel",
    }
)
class _NewModel(DynamicModule):
    def forward(
        self,
        inputs_embeds: Optional[Tensor] = None,
        token_type_embeds: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        # past_seq_length: Optional[Tensor] = None,
        # current_input_length: Optional[Tensor] = None,
        # past_key_cache: Optional[List[Tensor]] = None,
        # past_value_cache: Optional[List[Tensor]] = None,
    ):
        embeddings = inputs_embeds + token_type_embeds

        inputs_embeds = self.embeddings.LayerNorm(embeddings)
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            # past_seq_length=past_seq_length,
            # current_input_length=current_input_length,
            position_ids=position_ids,
            # past_key_cache=past_key_cache,
            # past_value_cache=past_value_cache,
        )

        hidden_states = outputs.last_hidden_state
        # logits = self.lm_head(hidden_states)
        return hidden_states
    
    def _setup(self, cfg: Optional[Dict] = None):
        # assert not self.config.use_sliding_window, f"{self.__class__.__name__} does not support sliding window."
        self.encoder.rotary_emb = self.embeddings.rotary_emb
        return self


def register_wrap_modules():
    pass
