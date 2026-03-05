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
from cmath import tau
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
from xhquant import nn as xhnn
from xhquant.nn import LLMCacheV2, MaskedSoftmax, RMSNorm
from xhquant.nn.modules.moeblock import MoeBlock
from xhquant.utils import digit_version

from xhquant.utils.registry import DynamicModule
from .gr00t.model.modules.dit import AlternateVLDiT, BasicTransformerBlock, AdaLayerNorm
from .gr00t.model.modules.embodiment_conditioned_mlp import MultiEmbodimentActionEncoder, SinusoidalPositionalEncoding
from .gr00t.model.modules.embodiment_conditioned_mlp import CategorySpecificMLP

import xh_model_zoo.xh_llm.models.groot.gr00t as gr00t
from gr00t.model.modules.dit import AlternateVLDiT, BasicTransformerBlock, AdaLayerNorm
from gr00t.model.modules.embodiment_conditioned_mlp import MultiEmbodimentActionEncoder, SinusoidalPositionalEncoding
from gr00t.model.modules.embodiment_conditioned_mlp import CategorySpecificMLP

from ..builder import XHLLM_TRACEABLE_MODULES

from diffusers.models.activations import GELU

@XHLLM_TRACEABLE_MODULES.register_module(
    {
        GELU: "GEGLU",
    }
)
class _GELU(DynamicModule):
    def gelu(self, gate: torch.Tensor) -> torch.Tensor:
        return F.gelu(gate, approximate=self.approximate)

    def forward(self, hidden_states):
        hidden_states = self.proj(hidden_states)
        hidden_states = self.gelu(hidden_states)
        return hidden_states
    
    def _setup(self, cfg: Optional[Dict] = None):
        return self

class Siglip2VisionOutput(ModelOutput):
    image_embeds: Optional[torch.FloatTensor] = None
    last_hidden_state: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    spatial_shapes: Optional[torch.LongTensor] = None


@XHLLM_TRACEABLE_MODULES.register_module(
    {
        AdaLayerNorm: "AdaLayerNorm",
    }
)
class _AdaLayerNorm(DynamicModule):
    def forward(
        self,
        x: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        temb = self.linear(self.silu(temb))
        # scale, shift = temb.chunk(2, dim=1)

        scale = self.slice_1(temb)
        shift = self.slice_2(temb)

        x = self.norm(x) * (1 + scale[:, None]) + shift[:, None]
        return x

    def _setup(self, cfg: Optional[Dict] = None):
        self.slice_1 = xhnn.Slice([0], [1536], [1], [1])
        self.slice_2 = xhnn.Slice([1536], [sys.maxsize], [1], [1])
        return self


@XHLLM_TRACEABLE_MODULES.register_module(
    {
        BasicTransformerBlock: "BasicTransformerBlock",
    }
)
class _BasicTransformerBlock(DynamicModule):
    def attn_forwad(
        attn,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        temb: Optional[torch.LongTensor] = None,
        *args,
        **kwargs,
    ):
        residual = hidden_states
        if attn.attn1.spatial_norm is not None:
            hidden_states = attn.attn1.spatial_norm(hidden_states, temb)

        # input_ndim = hidden_states.ndim

        # if input_ndim == 4:
        #     print("****************")
        #     # batch_size, channel, height, width = hidden_states.shape
        #     hidden_states = hidden_states.view(1, attn.sequence_length, -1).transpose(1, 2)

        # batch_size, sequence_length, _ = (
        #     hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        # )
        # sequence_length = attn.sequence_length
        batch_size = 1

        # if attention_mask is not None:
        #     attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        #     attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.attn1.group_norm is not None:
            hidden_states = attn.attn1.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.attn1.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.attn1.norm_cross:
            encoder_hidden_states = attn.attn1.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.attn1.to_k(encoder_hidden_states)
        value = attn.attn1.to_v(encoder_hidden_states)

        # inner_dim = key.shape[-1]
        # head_dim = attn.head_dim

        query = query.view(batch_size, -1, attn.attn1.heads, attn.head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.attn1.heads, attn.head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.attn1.heads, attn.head_dim).transpose(1, 2)

        if attn.attn1.norm_q is not None:
            query = attn.attn1.norm_q(query)
        if attn.attn1.norm_k is not None:
            key = attn.attn1.norm_k(key)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        if False:
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )
        else:
            # _kv_scale = 1 / math.sqrt(512)
            query = query * attn.kv_scale
            key = key.transpose(2, 3)
            attn_weights = torch.matmul(query, key) 
            if attention_mask is not None:
                attn_weights = attn.maskedadd(attn_weights, attention_mask)
            attn_weights = F.softmax(attn_weights, dim=-1) 
            hidden_states = torch.matmul(attn_weights, value) # [1, 1, 16384, 512]

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.attn1.heads * attn.head_dim)
        # hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.attn1.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.attn1.to_out[1](hidden_states)

        # if input_ndim == 4:
        #     print("****************")
        #     raise NotImplementedError
        #     # hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, attn.sequence_length, height, width)

        if attn.attn1.residual_connection:
            hidden_states = hidden_states + residual

        # hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states        

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        # 0. Self-Attention
        if self.norm_type == "ada_norm":
            norm_hidden_states = self.norm1(hidden_states, temb)
        else:
            norm_hidden_states = self.norm1(hidden_states)

        if self.pos_embed is not None:
            raise NotImplementedError
            norm_hidden_states = self.pos_embed(norm_hidden_states)

        attn_output = self.attn_forwad(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=(
                encoder_attention_mask if encoder_hidden_states is not None else attention_mask
            ),
        )

        if self.final_dropout:
            attn_output = self.final_dropout(attn_output)

        hidden_states = attn_output + hidden_states
        # if hidden_states.ndim == 4:
        #     hidden_states = hidden_states.squeeze(1)

        # 4. Feed-forward
        norm_hidden_states = self.norm3(hidden_states)
        ff_output = self.ff(norm_hidden_states)

        hidden_states = ff_output + hidden_states
        # if hidden_states.ndim == 4:
        #     hidden_states = hidden_states.squeeze(1)
        return hidden_states

    def _setup(self, cfg: Optional[Dict] = None):
        _kv_scale = 1 / math.sqrt(48)
        self.kv_scale = _kv_scale

        self.maskedadd = xhnn.MaskedAdd()

        self.head_dim = 48
        self.sequence_length = 51
        return self


@XHLLM_TRACEABLE_MODULES.register_module(
    {
        AlternateVLDiT: "AlternateVLDiT",
    }
)
class _AlternateVLDiT(DynamicModule):
    def forward(
        self,
        hidden_states: torch.Tensor,  # Shape: (B, T, D)
        encoder_hidden_states: torch.Tensor,  # Shape: (B, S, D)
        timestep: Optional[torch.LongTensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_all_hidden_states: bool = False,
        # image_mask: Optional[torch.Tensor] = None,
        # backbone_attention_mask: Optional[torch.Tensor] = None,
        image_attention_mask = None,
        non_image_attention_mask = None,
    ):
        # assert image_mask is not None, "Image mask is required"

        # Encode timesteps
        # temb = self.timestep_encoder(timestep)

        temb = self.embed[timestep]

        # Process through transformer blocks - single pass through the blocks

        # hidden_states = hidden_states.contiguous()
        # encoder_hidden_states = encoder_hidden_states.contiguous()

        # Create attention masks for image and non-image tokens
        # image_mask shape: (B, S) where True indicates image tokens
        # For attention, we need to invert: False means "don't attend to this token"

        # image_attention_mask = image_mask & backbone_attention_mask
        # non_image_attention_mask = (~image_mask) & backbone_attention_mask

        # all_hidden_states = [hidden_states]
        # assert self.config.interleave_self_attention, "Interleave self attention must be enabled"

        # Process through transformer blocks
        for idx, block in enumerate(self.transformer_blocks):
            if idx % 2 == 1:
                # Self-attention blocks
                hidden_states = block(
                    hidden_states,
                    attention_mask=None,
                    encoder_hidden_states=None,
                    encoder_attention_mask=None,
                    temb=temb,
                )
            else:
                # Cross-attention blocks - alternate between non-image and image tokens
                if idx % (2 * self.attend_text_every_n_blocks) == 0:
                    # Attend to non-image tokens
                    curr_encoder_attention_mask = non_image_attention_mask
                else:
                    # Attend to image tokens
                    curr_encoder_attention_mask = image_attention_mask

                hidden_states = block(
                    hidden_states,
                    attention_mask=None,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=curr_encoder_attention_mask,
                    temb=temb,
                )
            # all_hidden_states.append(hidden_states)

        # Output processing
        conditioning = temb
        # shift, scale = self.proj_out_1(F.silu(conditioning)).chunk(2, dim=1)

        temb_data = self.proj_out_1(F.silu(conditioning))
        shift = self.slice_1(temb_data)
        scale = self.slice_2(temb_data)

        hidden_states = self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
        if return_all_hidden_states:
            return self.proj_out_2(hidden_states), all_hidden_states
        else:
            return self.proj_out_2(hidden_states)

    def _setup(self, cfg: Optional[Dict] = None):
        self.embed = []
        for temb in [0, 250, 500, 750]:
            tembed = torch.tensor([temb]).cuda()
            self.embed.append( self.timestep_encoder( tembed ) )
        
        self.embed = torch.cat(self.embed, dim=0) # (4, 50, 512)

        self.slice_1 = xhnn.Slice([0], [1536], [1], [1])
        self.slice_2 = xhnn.Slice([1536], [sys.maxsize], [1], [1])

        return self

def swish(x):
    """Swish activation function."""
    return x * torch.sigmoid(x)

@XHLLM_TRACEABLE_MODULES.register_module(
    {
        MultiEmbodimentActionEncoder: "MultiEmbodimentActionEncoder",
    }
)
class _MultiEmbodimentActionEncoder(DynamicModule):
    def forward(self, actions, timesteps, cat_ids):
        """
        Args:
            actions: [B, T, action_dim] action tensor
            timesteps: [B,] timesteps - a single scalar per batch item
            cat_ids: [B,] category/embodiment IDs
        Returns:
            [B, T, hidden_size] encoded action features
        """
        # B, T, _ = actions.shape
        B = 1
        T = 50
        # 1) Expand each batch's single scalar time 'tau' across all T steps
        #    so that shape => (B, T)
        #    e.g. if timesteps is (B,), replicate across T

        # if timesteps.dim() == 1 and timesteps.shape[0] == B:
            # shape (B,) => (B,T)
        
        # timesteps = timesteps.unsqueeze(1).expand(-1, T)

        # else:
        #     raise ValueError(
        #         "Expected `timesteps` to have shape (B,) so we can replicate across T."
        #     )

        # 2) Standard action MLP step for shape => (B, T, w)
        # a_emb = self.W1(actions, cat_ids)
        a_emb = self.w1_linear(actions)

        # 3) Get the sinusoidal encoding (B, T, w)

        # tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)
        tau_emb = self.embed[timesteps] # 0, 250, 500, 750

        # 4) Concat along last dim => (B, T, 2w), then W2 => (B, T, w), swish
        x = torch.cat([a_emb, tau_emb], dim=-1)
        # x = swish(self.W2(x, cat_ids))
        x = swish(self.w2_linear(x))

        # 5) Finally W3 => (B, T, w)
        # x = self.W3(x, cat_ids)
        x = self.w3_linear(x)
        return x    
    
    def _setup(self, cfg: Optional[Dict] = None):
        self.w1_linear = nn.Linear(self.W1.W[20].shape[0],  self.W1.W[20].shape[1], bias=True)
        self.w1_linear.weight.data = self.W1.W[20].transpose(0, 1).half()
        self.w1_linear.bias.data = self.W1.b[20].half()

        self.w2_linear = nn.Linear(self.W2.W[20].shape[0],  self.W2.W[20].shape[1], bias=True)
        self.w2_linear.weight.data = self.W2.W[20].transpose(0, 1).half()
        self.w2_linear.bias.data = self.W2.b[20].half()

        self.w3_linear = nn.Linear(self.W3.W[20].shape[0],  self.W3.W[20].shape[1], bias=True)
        self.w3_linear.weight.data = self.W3.W[20].transpose(0, 1).half()
        self.w3_linear.bias.data = self.W3.b[20].half()

        self.embed = []
        for temb in [0, 250, 500, 750]:
            tembed = torch.tensor([temb]).unsqueeze(0).repeat(1, 50).cuda()
            self.embed.append( self.pos_encoding( tembed ).to(self.w3_linear.weight.dtype) )
        
        self.embed = torch.cat(self.embed, dim=0) # (4, 50, 512)

        # self.q_proj = nn.Linear(dim, dim, bias=True)
        # self.k_proj = nn.Linear(dim, dim, bias=True)
        # self.v_proj = nn.Linear(dim, dim, bias=True)
        # self.q_proj.weight.data = weight[0]
        # self.q_proj.bias.data = bias[0]
        return self


@XHLLM_TRACEABLE_MODULES.register_module(
    {
        CategorySpecificMLP: "CategorySpecificMLP",
    }
)
class _CategorySpecificMLP(DynamicModule):
    """Two-layer MLP with category-specific weights for multi-embodiment support."""
    def forward(self, x, cat_ids):
        """
        Args:
            x: [B, T, input_dim] input tensor
            cat_ids: [B] category/embodiment IDs
        Returns:
            [B, T, output_dim] output tensor
        """
        # hidden = F.relu(self.layer1(x, cat_ids))
        # return self.layer2(hidden, cat_ids)

        hidden = F.relu(self.layer1_linear(x))
        return self.layer2_linear(hidden)


    def _setup(self, cfg: Optional[Dict] = None):
        self.layer1_linear = nn.Linear(self.layer1.W[20].shape[0],  self.layer1.W[20].shape[1], bias=True)
        self.layer1_linear.weight.data = self.layer1.W[20].transpose(0, 1)
        self.layer1_linear.bias.data = self.layer1.b[20]

        self.layer2_linear = nn.Linear(self.layer2.W[20].shape[0],  self.layer2.W[20].shape[1], bias=True)
        self.layer2_linear.weight.data = self.layer2.W[20].transpose(0, 1)
        self.layer2_linear.bias.data = self.layer2.b[20]

        # self.q_proj = nn.Linear(dim, dim, bias=True)
        # self.k_proj = nn.Linear(dim, dim, bias=True)
        # self.v_proj = nn.Linear(dim, dim, bias=True)
        # self.q_proj.weight.data = weight[0]
        # self.q_proj.bias.data = bias[0]
        return self


# @XHLLM_TRACEABLE_MODULES.register_module(
#     {
#         SinusoidalPositionalEncoding: "SinusoidalPositionalEncoding",
#     }
# )
# class _SinusoidalPositionalEncoding(DynamicModule):
#     """
#     Produces a sinusoidal encoding of shape (B, T, w)
#     given timesteps of shape (B, T).
#     """
#     def forward(self, timesteps):
#         # timesteps: shape (B, T)
#         # We'll compute sin/cos frequencies across dim T
#         timesteps = timesteps.float()  # ensure float

#         B, T = timesteps.shape
#         device = timesteps.device

#         half_dim = self.embedding_dim // 2
#         # typical log space frequencies for sinusoidal encoding
#         exponent = -torch.arange(half_dim, dtype=torch.float, device=device) * (
#             torch.log(torch.tensor(10000.0)) / half_dim
#         )
#         # Expand timesteps to (B, T, 1) then multiply
#         freqs = timesteps.unsqueeze(-1) * exponent.exp()  # (B, T, half_dim)

#         sin = torch.sin(freqs)
#         cos = torch.cos(freqs)
#         enc = torch.cat([sin, cos], dim=-1)  # (B, T, w)

#         return enc

def register_wrap_modules():
    pass
