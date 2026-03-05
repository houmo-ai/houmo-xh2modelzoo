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
from pickle import NONE
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
from transformers.utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    can_return_tuple,
    logging,
    replace_return_docstrings,
)

user_home = os.path.expanduser("~")
target_path = os.path.join(
    user_home,
    ".cache/huggingface/modules/transformers_modules/Eagle-Block2A-2B-v2"
)
sys.path.append(target_path) # modeling.py

# eagle3_full_module_name = "transformers_modules.Eagle-Block2A-2B-v2.modeling_eagle3_vl"
# modeling_eagle3_vl = importlib.import_module(eagle3_full_module_name)

siglip2_full_module_name = "transformers_modules.Eagle-Block2A-2B-v2.modeling_siglip2"
modeling_siglip2 = importlib.import_module(siglip2_full_module_name)

NewModel = modeling_siglip2.Siglip2VisionModel
NewTransformer = modeling_siglip2.Siglip2VisionTransformer
NewEncoder = modeling_siglip2.Siglip2Encoder
NewEncoderLayer = modeling_siglip2.Siglip2EncoderLayer
NewAttention = modeling_siglip2.Siglip2Attention


from xhquant import nn as xhnn
from xhquant.nn import LLMCacheV2, MaskedSoftmax, RMSNorm
from xhquant.nn.modules.moeblock import MoeBlock
from xhquant.utils import digit_version
from xhquant.utils.registry import DynamicModule
from ..builder import XHLLM_TRACEABLE_MODULES


class Siglip2VisionOutput(ModelOutput):
    image_embeds: Optional[torch.FloatTensor] = None
    last_hidden_state: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    spatial_shapes: Optional[torch.LongTensor] = None


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
        bsz, q_len, embed_dim = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
        key_states = key_states.view(bsz, q_len, self.num_heads, self.head_dim)
        value_states = value_states.view(bsz, q_len, self.num_heads, self.head_dim)

        # cos, sin = position_embeddings
        # cos = self.cos_unsqueeze(cos)
        # sin = self.sin_unsqueeze(sin)
        if self.use_rope:
            print("find use rope")
            query_states, key_states = self.apply_rotary_pos_emb(query_states, key_states, cos, sin, unsqueeze_dim=1)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        if self.use_cache:
            key_states = self.k_cache(key_states, past_seq_length, current_input_length, past_k_cache)
            value_states = self.v_cache(value_states, past_seq_length, current_input_length, past_v_cache)

        query_states = query_states 

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

        attn_weights = torch.matmul(query_states, key_states) * self.kv_scale  # [4, 28, 256, 128], [4, 28, 128, 32768]
        # attn_weights = self.key_group_broadcast_matmul(query_states, key_states)
        # attn_weights = torch.matmul(query_states, key_states) / math.sqrt(self.head_dim) #fp16下会出现nan
        
        
        # attn_weights: Optional[Tensor] = self.masked_softmax(attn_weights, past_seq_length)

        # attn_weights = self.maskedadd(attn_weights, attention_mask)
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, value_states)  # [4, 28, 256, 32768], [4, 28, 32768, 128]
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, -1, embed_dim)

        attn_output = self.out_proj(attn_output)

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
        self.head_dim = 72

        self.rotate_half_slice_1 = xhnn.Slice([0], [self.head_dim // 2], [3], [1])
        self.rotate_half_slice_2 = xhnn.Slice([self.head_dim // 2], [sys.maxsize], [3], [1])
        self.masked_softmax = MaskedSoftmax(dim=-1)
        use_cache = cfg.use_cache
        self.use_cache = use_cache
        max_sequence_length = cfg.max_sequence_length
        self.max_sequence_length = max_sequence_length
        input_seq_len = cfg.input_sequence_length
        

        # weight = self.qkv_proj.weight.data.clone()
        # bias = self.qkv_proj.bias.data.clone()
        # dim = weight.shape[0] // 3
        # weight = weight.permute(1, 0).reshape(dim, 3, dim).permute(1, 2, 0)
        # bias = bias.reshape(3, dim)
        # self.q_proj = nn.Linear(dim, dim, bias=True)
        # self.k_proj = nn.Linear(dim, dim, bias=True)
        # self.v_proj = nn.Linear(dim, dim, bias=True)
        # self.q_proj.weight.data = weight[0]
        # self.q_proj.bias.data = bias[0]
        # self.k_proj.weight.data = weight[1]
        # self.k_proj.bias.data = bias[1]
        # self.v_proj.weight.data = weight[2]
        # self.v_proj.bias.data = bias[2]

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

        _kv_scale = 1 / math.sqrt(self.head_dim) # 72
        self.kv_scale = _kv_scale
        # self.register_buffer("kv_scale", torch.tensor(_kv_scale, dtype=torch.float16), persistent=False)
        # self.register_parameter(
        #     "kv_scale", nn.Parameter(torch.tensor([_kv_scale], dtype=torch.float16), requires_grad=False)
        # )
        return self


@XHLLM_TRACEABLE_MODULES.register_module(
    {
        NewEncoderLayer: "Siglip2EncoderLayer",
    }
)
class _NewLayer(DynamicModule):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask = None,
        # past_seq_length: Optional[Tensor] = None,
        # current_input_length: Optional[Tensor] = None,
        # position_ids: Optional[torch.LongTensor] = None,
        # past_k_cache: Optional[Tensor] = None,
        # past_v_cache: Optional[Tensor] = None,
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


        # Self Attention
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
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

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states

    def _setup(self, cfg: Optional[Dict] = None):
        return self



@XHLLM_TRACEABLE_MODULES.register_module(
    {
        NewEncoder: "Siglip2Encoder",
    }
)
class _NewEncoder(DynamicModule):
    def forward(
        self,
        inputs_embeds,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        win_meta_list: Optional[List[Dict]] = None,
        spatial_shapes: Optional[torch.Tensor] = None,
        attention_mask= None,
    ):

        hidden_states = inputs_embeds

        # cos = self.cos_embeding(position_ids) # # [bs, seq_len, 1, dim]
        # sin = self.sin_embeding(position_ids)
        # cos = cos.unsqueeze(2)
        # sin = sin.unsqueeze(2)
        # position_embeddings = (cos, sin)

        for idx, encoder_layer in enumerate(self.layers):
            print("processing: ", idx)
            if self.use_cache:
                _past_k_cache = past_key_cache[idx]
                _past_v_cache = past_value_cache[idx]
            else:
                _past_k_cache = None
                _past_v_cache = None

            layer_outputs = encoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                # past_seq_length=past_seq_length,
                # current_input_length=current_input_length,
                # position_ids=position_ids,
                # past_k_cache=_past_k_cache,
                # past_v_cache=_past_v_cache,
                # position_embeddings=position_embeddings,
            )

            hidden_states = layer_outputs
            # break
            if self.only_first_block:
                break


        return hidden_states

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
        # if not hasattr(self.rotary_emb, "cos_cached"):
        #     self.rotary_emb.setup_after_callback = self._setup_cos_sin_embeding
        # else:
        #     self._setup_cos_sin_embeding()

        return self


@XHLLM_TRACEABLE_MODULES.register_module(
    {
        NewTransformer: "Siglip2VisionTransformer",
    }
)
class _NewTransformer(DynamicModule):
    def forward(
        self,
        windows_tensor: torch.FloatTensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ):
        r"""
        Returns:

        """
        # output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # output_hidden_states = (
        #     output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        # )

        # windows_tensor, win_meta_list, spatial_shapes, reverse_mapping = self.embeddings(pixel_values)
        
        encoder_outputs = self.encoder(
            inputs_embeds=windows_tensor,
            output_attentions=False,
            output_hidden_states=False,
            win_meta_list=None,
            spatial_shapes=None,
        )

        last_hidden_state = encoder_outputs
        last_hidden_state = self.post_layernorm(last_hidden_state)
        return last_hidden_state

    def _setup(self, cfg: Optional[Dict] = None):
        return self


@XHLLM_TRACEABLE_MODULES.register_module(
    {
        NewModel: "Siglip2VisionModel",
    }
)
class _NewModel(DynamicModule):
    def forward(
        self,
        pixel_values: torch.FloatTensor,
    ):

        return self.vision_model(
            windows_tensor=pixel_values,
            output_attentions=None,
            output_hidden_states=None,
        )

    def _setup(self, cfg: Optional[Dict] = None):
        return self


def register_wrap_modules():
    pass
