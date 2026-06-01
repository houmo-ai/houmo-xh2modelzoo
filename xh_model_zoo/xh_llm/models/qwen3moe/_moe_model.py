# Copyright 2025 HOUMO AI
#
# File: _moe_model.py
# Description:
#   Moe Model model implementation.
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
from xhquant import nn as xhnn
from xhquant.nn import LLMCacheV2, MaskedSoftmax, RMSNorm
from xhquant.nn.modules.moeblock import MoeBlock
from xhquant.utils import digit_version
from xhquant.utils.registry import DynamicModule

from ..builder import XHLLM_TRACEABLE_MODULES

# 省的多次建表
silu_lut_cut_points = torch.tensor(
    [
        -41760.0,
        -11.75,
        -8.7578125,
        -5.5859375,
        -1.970703125,
        -0.281982421875,
        0.073486328125,
        1.2568359375,
        7.68359375,
        41696.0,
        65504.0,
    ]
)
silu_lut_values = torch.tensor(
    [
        -0.0,
        -9.268522262573242e-05,
        -0.00010097026824951172,
        -0.0001099705696105957,
        -0.00011980533599853516,
        -0.0001304149627685547,
        -0.00014209747314453125,
        -0.00015473365783691406,
        -0.0001684427261352539,
        -0.00018334388732910156,
        -0.0001996755599975586,
        -0.00021731853485107422,
        -0.00023663043975830078,
        -0.0002574920654296875,
        -0.00028014183044433594,
        -0.00030493736267089844,
        -0.000331878662109375,
        -0.0003612041473388672,
        -0.000392913818359375,
        -0.00042748451232910156,
        -0.0004649162292480469,
        -0.0005059242248535156,
        -0.0005502700805664062,
        -0.0005984306335449219,
        -0.0006504058837890625,
        -0.0007071495056152344,
        -0.0007691383361816406,
        -0.0008358955383300781,
        -0.0009088516235351562,
        -0.000988006591796875,
        -0.0010728836059570312,
        -0.0011663436889648438,
        -0.0012674331665039062,
        -0.001377105712890625,
        -0.00150299072265625,
        -0.00164031982421875,
        -0.0017900466918945312,
        -0.001953125,
        -0.002132415771484375,
        -0.0023250579833984375,
        -0.002536773681640625,
        -0.002765655517578125,
        -0.003017425537109375,
        -0.00328826904296875,
        -0.0035858154296875,
        -0.00390625,
        -0.0042572021484375,
        -0.004638671875,
        -0.005054473876953125,
        -0.005504608154296875,
        -0.005992889404296875,
        -0.00652313232421875,
        -0.007099151611328125,
        -0.007724761962890625,
        -0.0084075927734375,
        -0.0091400146484375,
        -0.00994110107421875,
        -0.01080322265625,
        -0.01174163818359375,
        -0.01276397705078125,
        -0.01386260986328125,
        -0.01505279541015625,
        -0.0163421630859375,
        -0.017730712890625,
        -0.0192413330078125,
        -0.0208740234375,
        -0.02288818359375,
        -0.02508544921875,
        -0.0274658203125,
        -0.0300750732421875,
        -0.03289794921875,
        -0.035980224609375,
        -0.039337158203125,
        -0.04296875,
        -0.046875,
        -0.051116943359375,
        -0.05572509765625,
        -0.0606689453125,
        -0.06597900390625,
        -0.07171630859375,
        -0.077880859375,
        -0.08447265625,
        -0.09149169921875,
        -0.09893798828125,
        -0.10693359375,
        -0.1153564453125,
        -0.124267578125,
        -0.133544921875,
        -0.1434326171875,
        -0.153564453125,
        -0.1641845703125,
        -0.175048828125,
        -0.1861572265625,
        -0.1973876953125,
        -0.2086181640625,
        -0.2197265625,
        -0.2305908203125,
        -0.2410888671875,
        -0.2457275390625,
        -0.250244140625,
        -0.25439453125,
        -0.25830078125,
        -0.26220703125,
        -0.265625,
        -0.268798828125,
        -0.271484375,
        -0.27392578125,
        -0.275634765625,
        -0.277099609375,
        -0.278076171875,
        -0.278564453125,
        -0.2783203125,
        -0.27734375,
        -0.27587890625,
        -0.2734375,
        -0.270263671875,
        -0.266357421875,
        -0.26171875,
        -0.256103515625,
        -0.2493896484375,
        -0.24169921875,
        -0.2330322265625,
        -0.2232666015625,
        -0.2122802734375,
        -0.2001953125,
        -0.18701171875,
        -0.1724853515625,
        -0.15673828125,
        -0.1396484375,
        -0.1212158203125,
        -0.1171875,
        -0.11309814453125,
        -0.10894775390625,
        -0.104736328125,
        -0.1004638671875,
        -0.09613037109375,
        -0.09173583984375,
        -0.0872802734375,
        -0.082763671875,
        -0.07818603515625,
        -0.07354736328125,
        -0.06884765625,
        -0.0640869140625,
        -0.059234619140625,
        -0.054351806640625,
        -0.049407958984375,
        -0.044403076171875,
        -0.039337158203125,
        -0.034210205078125,
        -0.0290069580078125,
        -0.0237579345703125,
        -0.0184478759765625,
        -0.01306915283203125,
        -0.00762939453125,
        -0.002132415771484375,
        0.003429412841796875,
        0.00905609130859375,
        0.014739990234375,
        0.020477294921875,
        0.0262908935546875,
        0.03216552734375,
        0.0380859375,
        0.05828857421875,
        0.07916259765625,
        0.1007080078125,
        0.1229248046875,
        0.145751953125,
        0.1693115234375,
        0.1934814453125,
        0.2183837890625,
        0.243896484375,
        0.27001953125,
        0.296630859375,
        0.323974609375,
        0.35205078125,
        0.380615234375,
        0.40966796875,
        0.439208984375,
        0.469482421875,
        0.5,
        0.53173828125,
        0.5634765625,
        0.595703125,
        0.62841796875,
        0.66162109375,
        0.6953125,
        0.72900390625,
        0.763671875,
        0.79833984375,
        0.833984375,
        0.86962890625,
        0.9052734375,
        0.94189453125,
        0.978515625,
        1.1826171875,
        1.3935546875,
        1.6083984375,
        1.8271484375,
        2.046875,
        2.267578125,
        2.48828125,
        2.708984375,
        2.927734375,
        3.14453125,
        3.361328125,
        3.576171875,
        3.7890625,
        4.0,
        4.2109375,
        4.41796875,
        4.62890625,
        4.8359375,
        5.04296875,
        5.24609375,
        5.453125,
        5.65625,
        5.859375,
        6.0625,
        6.265625,
        6.46875,
        6.671875,
        6.875,
        7.07421875,
        7.27734375,
        7.48046875,
        7.6796875,
        1310.0,
        2614.0,
        3916.0,
        5220.0,
        6520.0,
        7824.0,
        9128.0,
        10432.0,
        11736.0,
        13032.0,
        14336.0,
        15640.0,
        16944.0,
        18240.0,
        19552.0,
        20848.0,
        22160.0,
        23456.0,
        24768.0,
        26064.0,
        27360.0,
        28672.0,
        29968.0,
        31280.0,
        32576.0,
        33888.0,
        35168.0,
        36480.0,
        37792.0,
        39104.0,
        40384.0,
        41696.0,
        65504.0,
    ]
)
silu_lut_scale = torch.tensor(
    [
        2.396106719970703e-05,
        10.6953125,
        10.0859375,
        8.8515625,
        18.953125,
        90.0,
        27.046875,
        4.98046875,
        0.0007677078247070312,
        4.202127456665039e-05,
    ]
)


def create_silu_lut_tensors():
    """创建独立的SiLU查表张量，避免多个MoeBlock共享同一个参数"""
    return (silu_lut_cut_points.clone(), silu_lut_values.clone(), silu_lut_scale.clone())


@XHLLM_TRACEABLE_MODULES.register_module(
    {
        Qwen3MoeRotaryEmbedding: "Qwen3MoeRotaryEmbedding",
    }
)
class _Qwen3MoeRotaryEmbedding(DynamicModule):
    def _setup(self, cfg: Optional[Dict] = None):
        assert "dynamic" not in self.rope_type, f"{self.rope_type} is not supported in dynamic mode"

        # self.max_position_embeddings = max_position_embeddings
        # Build here to make `torch.jit.trace` work.
        self._setup_cos_sin_cache(seq_len=self.max_seq_len_cached, dtype=self.inv_freq.dtype)

    def _setup_cos_sin_cache(self, seq_len, dtype):
        """
        4.45 版本实现
        """
        position_ids = torch.arange(0, seq_len, dtype=torch.long, device=self.inv_freq.device).unsqueeze(0)

        # use fp32 for cos and sin calculation
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
    }
)
class _Qwen3MoeAttention(DynamicModule):
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
        q_embed = (q * cos) + (self.rotate_half(q) * sin)
        k_embed = (k * cos) + (self.rotate_half(k) * sin)
        # q_embed = self.rope(q, cos, sin)
        # k_embed = self.rope(k, cos, sin)
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
        attn_output = attn_output.reshape(bsz, q_len, self.attn_hidden_dim)

        attn_output = self.o_proj(attn_output)

        # return attn_output, attn_weights, past_key_value
        return attn_output, None, None

    def _setup(self, cfg: Optional[Dict] = None):
        if not hasattr(self, "num_key_value_heads"):
            self.num_key_value_heads = self.config.num_key_value_heads
        if not hasattr(self, "num_heads"):
            self.num_heads = self.config.num_attention_heads  # 28

        self.attn_hidden_dim = self.head_dim * self.num_heads

        self.rotate_half_slice_1 = xhnn.Slice([0], [self.head_dim // 2], [3], [1])
        self.rotate_half_slice_2 = xhnn.Slice([self.head_dim // 2], [sys.maxsize], [3], [1])
        self.masked_softmax = MaskedSoftmax(dim=-1)
        use_cache = cfg.use_cache
        self.use_cache = use_cache
        max_sequence_length = cfg.max_sequence_length
        self.max_sequence_length = max_sequence_length
        input_seq_len = cfg.input_sequence_length
        self.cos_gather = xhnn.Gather(0)
        self.sin_gather = xhnn.Gather(0)

        # self.rope = Rope()

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
        Qwen3MoeDecoderLayer: "Qwen3MoeDecoderLayer",
    }
)
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
    }
)
class _Qwen3MoeSparseMoeBlock(DynamicModule):
    def forward(self, hidden_states):
        ori_hidden_states = hidden_states
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1)
        return self.moeblock(
            ori_hidden_states,
            routing_weights,
            #  self.expert_gate_proj_weight.to(ori_hidden_states),
            #  self.expert_gate_proj_bias,
            #  self.expert_up_proj_weight.to(ori_hidden_states),
            #  self.expert_up_proj_bias,
            #  self.expert_down_proj_weight.to(ori_hidden_states),
            #  self.expert_down_proj_bias
        )

        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:  # only diff with mixtral sparse moe block!
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(torch.float16)
        # final_hidden_states = torch.zeros(
        #     (self.batch_size * self.input_seq_len, 2048), dtype=torch.float16, device=self.device
        # )
        final_hidden_states = torch.zeros_like(hidden_states)
        # final_hidden_states = torch.zeros(
        #     (self.batch_size * self.input_seq_len, 2048), dtype=hidden_states.dtype, device=hidden_states.device
        # )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])
            # original index add
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]
            final_hidden_states = final_hidden_states.index_add_(0, top_x, current_hidden_states.to(torch.float16))
        final_hidden_states = final_hidden_states.reshape(self.batch_size, sequence_length, 2048)
        # final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits

    def _setup(self, cfg: Optional[Dict] = None):
        self.input_seq_len = cfg.input_sequence_length
        self.batch_size = cfg.batch_size
        self.device = self.gate.weight.device

        # 为每个MoeBlock创建独立的查表参数，避免trace时参数共享问题
        act_lut_cut_points, act_lut_values, act_lut_scale = create_silu_lut_tensors()
        self.moeblock = MoeBlock(
            self.experts[0].act_fn._get_name().lower(),
            self.top_k,
            self.norm_topk_prob,
            # act_lut_cut_points=act_lut_cut_points,
            # act_lut_values=act_lut_values,
            # act_lut_scale=act_lut_scale,
        )

        self.moeblock.expert_gate_proj_weight = torch.nn.Parameter(
            torch.zeros(
                self.num_experts,
                self.experts[0].gate_proj.weight.shape[0],
                self.experts[0].gate_proj.weight.shape[1],
                device=self.device,
                dtype=self.experts[0].gate_proj.weight.dtype,
            )
        )
        if hasattr(self.experts[0].gate_proj, "quant_weight") and self.experts[0].gate_proj.quant_weight is not None:
            self.moeblock.expert_gate_proj_quant_weight = torch.tensor(
                torch.zeros(
                    self.num_experts,
                    self.experts[0].gate_proj.quant_weight.shape[0],
                    self.experts[0].gate_proj.quant_weight.shape[1],
                    device=self.device,
                    dtype=self.experts[0].gate_proj.quant_weight.dtype,
                )
            )

        if self.experts[0].gate_proj.bias is not None:
            self.moeblock.expert_gate_proj_bias = torch.nn.Parameter(
                torch.zeros(self.num_experts, self.experts[0].gate_proj.bias.shape[0], device=self.device)
            )
        else:
            self.moeblock.expert_gate_proj_bias = None

        self.moeblock.expert_up_proj_weight = torch.nn.Parameter(
            torch.zeros(
                self.num_experts,
                self.experts[0].up_proj.weight.shape[0],
                self.experts[0].up_proj.weight.shape[1],
                device=self.device,
                dtype=self.experts[0].up_proj.weight.dtype,
            )
        )

        if hasattr(self.experts[0].up_proj, "quant_weight") and self.experts[0].up_proj.quant_weight is not None:
            self.moeblock.expert_up_proj_quant_weight = torch.tensor(
                torch.zeros(
                    self.num_experts,
                    self.experts[0].up_proj.quant_weight.shape[0],
                    self.experts[0].up_proj.quant_weight.shape[1],
                    device=self.device,
                    dtype=self.experts[0].up_proj.quant_weight.dtype,
                )
            )

        if self.experts[0].up_proj.bias is not None:
            self.moeblock.expert_up_proj_bias = torch.nn.Parameter(
                torch.zeros(self.num_experts, self.experts[0].up_proj.bias.shape[0], device=self.device)
            )
        else:
            self.moeblock.expert_up_proj_bias = None

        self.moeblock.expert_down_proj_weight = torch.nn.Parameter(
            torch.zeros(
                self.num_experts,
                self.experts[0].down_proj.weight.shape[0],
                self.experts[0].down_proj.weight.shape[1],
                device=self.device,
                dtype=self.experts[0].down_proj.weight.dtype,
            )
        )

        if hasattr(self.experts[0].down_proj, "quant_weight") and self.experts[0].down_proj.quant_weight is not None:
            self.moeblock.expert_down_proj_quant_weight = torch.tensor(
                torch.zeros(
                    self.num_experts,
                    self.experts[0].down_proj.quant_weight.shape[0],
                    self.experts[0].down_proj.quant_weight.shape[1],
                    device=self.device,
                    dtype=self.experts[0].down_proj.quant_weight.dtype,
                )
            )

        if self.experts[0].down_proj.bias is not None:
            self.moeblock.expert_down_proj_bias = torch.nn.Parameter(
                torch.zeros(self.num_experts, self.experts[0].down_proj.bias.shape[0], device=self.device)
            )
        else:
            self.moeblock.expert_down_proj_bias = None

        with torch.no_grad():
            self.moeblock.expert_gate_proj_weight.copy_(
                torch.cat([expert.gate_proj.weight.data.to(self.device).unsqueeze(0) for expert in self.experts], dim=0)
            )
            if (
                hasattr(self.experts[0].gate_proj, "quant_weight")
                and self.experts[0].gate_proj.quant_weight is not None
            ):
                self.moeblock.expert_gate_proj_quant_weight.copy_(
                    torch.cat(
                        [expert.gate_proj.quant_weight.data.to(self.device).unsqueeze(0) for expert in self.experts],
                        dim=0,
                    )
                )
            if self.moeblock.expert_gate_proj_bias is not None:
                self.moeblock.expert_gate_proj_bias.copy_(
                    torch.cat(
                        [expert.gate_proj.bias.data.to(self.device).unsqueeze(0) for expert in self.experts], dim=0
                    )
                )

            self.moeblock.expert_up_proj_weight.copy_(
                torch.cat([expert.up_proj.weight.data.to(self.device).unsqueeze(0) for expert in self.experts], dim=0)
            )
            if hasattr(self.experts[0].up_proj, "quant_weight") and self.experts[0].up_proj.quant_weight is not None:
                self.moeblock.expert_up_proj_quant_weight.copy_(
                    torch.cat(
                        [expert.up_proj.quant_weight.data.to(self.device).unsqueeze(0) for expert in self.experts],
                        dim=0,
                    )
                )
            if self.moeblock.expert_up_proj_bias is not None:
                self.moeblock.expert_up_proj_bias.copy_(
                    torch.cat([expert.up_proj.bias.data.to(self.device).unsqueeze(0) for expert in self.experts], dim=0)
                )

            self.moeblock.expert_down_proj_weight.copy_(
                torch.cat([expert.down_proj.weight.data.to(self.device).unsqueeze(0) for expert in self.experts], dim=0)
            )
            if (
                hasattr(self.experts[0].down_proj, "quant_weight")
                and self.experts[0].down_proj.quant_weight is not None
            ):
                self.moeblock.expert_down_proj_quant_weight.copy_(
                    torch.cat(
                        [expert.down_proj.quant_weight.data.to(self.device).unsqueeze(0) for expert in self.experts],
                        dim=0,
                    )
                )
            if self.moeblock.expert_down_proj_bias is not None:
                self.moeblock.expert_down_proj_bias.copy_(
                    torch.cat(
                        [expert.down_proj.bias.data.to(self.device).unsqueeze(0) for expert in self.experts], dim=0
                    )
                )

        del self.experts
        torch.cuda.empty_cache()

        return self

    # def create_zeros(self, shape, dtype, device):
    #     return torch.zeros(shape, dtype=dtype, device=device)


@XHLLM_TRACEABLE_MODULES.register_module(
    {
        Qwen3MoeRMSNorm: "Qwen3MoeRMSNorm",
    }
)
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
    }
)
class _Qwen3MoeModel(DynamicModule):
    def forward(
        self,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_seq_length: Optional[Tensor] = None,
        current_input_length: Optional[Tensor] = None,
        # position_ids: Optional[torch.LongTensor] = None,
        past_key_cache: Optional[List[Tensor]] = None,
        past_value_cache: Optional[List[Tensor]] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        causal_mask = None  # 在Qwen2Attention中处理
        hidden_states = inputs_embeds

        cos = self.cos_slice(self.rotary_emb.cos_cached, past_seq_length)
        sin = self.sin_slice(self.rotary_emb.sin_cached, past_seq_length)
        # cos = self.cos_embeding(position_ids)
        # sin = self.sin_embeding(position_ids)
        # cos = cos.unsqueeze(1)
        # sin = sin.unsqueeze(1)

        position_embeddings = (cos, sin)

        for idx, decoder_layer in enumerate(self.layers):
            print("processing: ", idx)
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
            # break
            if self.only_first_block:
                break

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
            hidden_states = self.llm_gather(hidden_states, current_input_length - 1)
        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
        )

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

        # def _gather_update_cfg(self, cfg: Optional[Dict] = None):
        #     num_logits_to_keep = cfg.num_logits_to_keep
        #     self.num_logits_to_keep = num_logits_to_keep

        # self.llm_gather._update_cfg = types.MethodType(_gather_update_cfg, self.llm_gather)

        return self


@XHLLM_TRACEABLE_MODULES.register_module(
    {
        Qwen3MoeForCausalLM: "Qwen3MoeForCausalLM",
    }
)
class _Qwen3MoeForCausalLM(DynamicModule):
    def forward(
        self,
        inputs_embeds: Optional[Tensor] = None,
        past_seq_length: Optional[Tensor] = None,
        current_input_length: Optional[Tensor] = None,
        # position_ids: Optional[Tensor] = None,
        past_key_cache: Optional[List[Tensor]] = None,
        past_value_cache: Optional[List[Tensor]] = None,
    ):
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            past_seq_length=past_seq_length,
            current_input_length=current_input_length,
            # position_ids=position_ids,
            past_key_cache=past_key_cache,
            past_value_cache=past_value_cache,
        )

        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)
        return logits

    def _setup(self, cfg: Optional[Dict] = None):
        assert not self.config.use_sliding_window, f"{self.__class__.__name__} does not support sliding window."
        return self


def register_wrap_modules(hf_model: Optional[Qwen3MoeForCausalLM] = None):
    pass
