# Copyright 2025 HOUMO AI
#
# File: _vision_model_impl.py
# Description:
#   Example script: xh_model_zoo/xh_llm/models/glm_ocr/_vision_model_impl.py
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# SPDX-License-Identifier: MIT

"""
GLM-OCR Vision Model wrap implementation for xh2a export.

Key transformations:
1. PatchEmbed: Conv3d → Conv2d stack (decomposes temporal dimension)
2. VisionAttention: qkv → q_proj, k_proj, v_proj (removes split in forward)
3. VisionModel: Pre-computes rotary pos emb, cu_seqlens, attention_mask as static buffers
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modeling_glm_ocr import (
    GlmOcrRMSNorm,
    GlmOcrVisionAttention,
    GlmOcrVisionBlock,
    GlmOcrVisionModel,
    GlmOcrVisionPatchEmbed,
    GlmOcrVisionPatchMerger,
    rotate_half,
)
from xhquant import nn as xhnn
from xhquant.api import ConfigDict
from xhquant.utils.registry import DynamicModule

from ..builder import XHLLM_TRACEABLE_MODULES


@XHLLM_TRACEABLE_MODULES.register_module(
    {
        GlmOcrVisionAttention: "GlmOcrVisionAttention",
    }
)
class _GlmOcrVisionAttention(DynamicModule):
    """Wrap GlmOcrVisionAttention: split qkv into q_proj, k_proj, v_proj."""

    def apply_rotary_pos_emb(
        self, q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed

    @torch.no_grad()
    def _setup(self, cfg: ConfigDict):
        head_dim = self.dim // self.num_heads
        self.kv_scale = 1 / math.sqrt(head_dim)

        # Decompose qkv [hidden_size * 3, hidden_size] → q_proj, k_proj, v_proj
        weight = self.qkv.weight.data.clone()  # [3*dim, dim]
        has_bias = self.qkv.bias is not None
        dim = self.dim

        # weight: [3*dim, dim] → reshape to [3, dim, dim]
        weight = weight.reshape(3, dim, dim)

        if has_bias:
            bias = self.qkv.bias.data.clone()  # [3*dim]
            bias = bias.reshape(3, dim)

        self.q_proj = nn.Linear(dim, dim, bias=has_bias)
        self.k_proj = nn.Linear(dim, dim, bias=has_bias)
        self.v_proj = nn.Linear(dim, dim, bias=has_bias)

        self.q_proj.weight.data.copy_(weight[0])
        self.k_proj.weight.data.copy_(weight[1])
        self.v_proj.weight.data.copy_(weight[2])

        if has_bias:
            self.q_proj.bias.data.copy_(bias[0])
            self.k_proj.bias.data.copy_(bias[1])
            self.v_proj.bias.data.copy_(bias[2])

        # Delete original qkv to save memory
        del self.qkv

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch, seq_length = hidden_states.shape[:2]

        # Separate q/k/v projections (no split needed)
        q = self.q_proj(hidden_states).reshape(batch, seq_length, self.num_heads, -1)
        k = self.k_proj(hidden_states).reshape(batch, seq_length, self.num_heads, -1)
        v = self.v_proj(hidden_states).reshape(batch, seq_length, self.num_heads, -1)

        # Apply q_norm and k_norm
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Apply rotary position embeddings
        cos, sin = position_embeddings
        q, k = self.apply_rotary_pos_emb(q, k, cos, sin)

        # Transpose to [batch, num_heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product attention
        k = k.transpose(-2, -1)
        q = q * self.kv_scale
        dtype = q.dtype
        attn_weights = torch.matmul(q, k).to(dtype)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(dtype)
        attn_output = torch.matmul(attn_weights, v)

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(batch, seq_length, -1)
        attn_output = self.proj(attn_output)
        return attn_output


@XHLLM_TRACEABLE_MODULES.register_module(
    {
        GlmOcrVisionBlock: "GlmOcrVisionBlock",
    }
)
class _GlmOcrVisionBlock(DynamicModule):
    @torch.no_grad()
    def _setup(self, cfg: ConfigDict):
        pass

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


@XHLLM_TRACEABLE_MODULES.register_module(
    {
        GlmOcrVisionPatchEmbed: "GlmOcrVisionPatchEmbed",
    }
)
class _GlmOcrVisionPatchEmbed(DynamicModule):
    """Wrap PatchEmbed: Conv3d → Conv2d stack."""

    @torch.no_grad()
    def _setup(self, cfg: ConfigDict):
        proj3d = self.proj
        if not isinstance(proj3d, nn.Conv3d):
            raise TypeError(f"Expected Conv3d, got {type(proj3d)}")

        kernel_size = (self.patch_size, self.patch_size)
        dev = proj3d.weight.device
        dtype = proj3d.weight.dtype
        has_bias = proj3d.bias is not None

        # Create one Conv2d per temporal slice
        self.proj2d = nn.ModuleList()
        for i in range(self.temporal_patch_size):
            conv = nn.Conv2d(
                self.in_channels,
                self.embed_dim,
                kernel_size=kernel_size,
                stride=kernel_size,
                bias=False,
                device=dev,
                dtype=dtype,
            )
            conv.weight.data.copy_(proj3d.weight[:, :, i, :, :].contiguous())
            self.proj2d.append(conv)

        if has_bias:
            self._conv_bias = nn.Parameter(proj3d.bias.data.clone())
        else:
            self._conv_bias = None

        # Remove original Conv3d
        del self.proj

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj2d[0].weight.dtype
        hidden_states = hidden_states.view(
            -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
        )
        hidden_states = hidden_states.to(dtype=target_dtype)

        out = None
        for i in range(self.temporal_patch_size):
            term = self.proj2d[i](hidden_states[:, :, i, :, :])
            out = term if out is None else (out + term)

        if self._conv_bias is not None:
            out = out + self._conv_bias.view(1, -1, 1, 1)

        return out.view(-1, self.embed_dim)


@XHLLM_TRACEABLE_MODULES.register_module(
    {
        GlmOcrVisionPatchMerger: "GlmOcrVisionPatchMerger",
    }
)
class _GlmOcrVisionPatchMerger(DynamicModule):
    @torch.no_grad()
    def _setup(self, cfg: ConfigDict):
        pass

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        hidden_state = self.proj(hidden_state)
        hidden_state = self.act1(self.post_projection_norm(hidden_state))
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))


@XHLLM_TRACEABLE_MODULES.register_module(
    {
        GlmOcrVisionModel: "GlmOcrVisionModel",
    }
)
class _GlmOcrVisionModel(DynamicModule):
    """Wrap GlmOcrVisionModel: pre-compute rotary pos emb and attention mask as static buffers."""

    @torch.no_grad()
    def _setup(self, cfg: ConfigDict):
        self.only_first_block = False

        device = next(self.parameters()).device

        self.max_size_w = cfg.max_size_w
        self.max_size_h = cfg.max_size_h
        max_size_t = cfg.get("max_size_t", 1)
        temporal_patch_size = cfg.get("temporal_patch_size", 2)

        grid_size_w = self.max_size_w // self.patch_size
        grid_size_h = self.max_size_h // self.patch_size
        grid_size_t = max_size_t // temporal_patch_size

        seq_length = grid_size_t * grid_size_h * grid_size_w

        # Compute grid_thw
        grid_thw = torch.tensor([[grid_size_t, grid_size_h, grid_size_w]], device=device)

        # Compute rotary position embeddings
        pos_ids = []
        for t, h, w in grid_thw:
            hpos_ids = torch.arange(h, device=device).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3).flatten()

            wpos_ids = torch.arange(w, device=device).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3).flatten()
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))

        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        cos = emb.cos().unsqueeze(-2).unsqueeze(0)  # [1, seq_len, 1, head_dim]
        sin = emb.sin().unsqueeze(-2).unsqueeze(0)

        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        # Compute cu_seqlens for attention mask
        cu_seqlens = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
        ).cumsum(dim=0, dtype=torch.int32)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        # Build attention mask from cu_seqlens (block-diagonal)
        attention_mask = torch.zeros([1, seq_length, seq_length], dtype=torch.bool, device=device)
        for i in range(1, len(cu_seqlens)):
            attention_mask[..., cu_seqlens[i - 1]:cu_seqlens[i], cu_seqlens[i - 1]:cu_seqlens[i]] = True

        attention_bias = torch.zeros(1, 1, seq_length, seq_length, dtype=torch.float16, device=device)
        attention_bias.masked_fill_(attention_mask.unsqueeze(1).logical_not(), -torch.finfo(torch.float16).max)
        self.register_buffer("attention_bias", attention_bias, persistent=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.patch_embed(hidden_states)

        seq_len = hidden_states.shape[0]
        cos = self.cos
        sin = self.sin
        attention_bias = self.attention_bias

        if cos.shape[1] != seq_len:
            if cos.shape[1] > seq_len:
                cos = cos[:, :seq_len]
                sin = sin[:, :seq_len]
                attention_bias = attention_bias[:, :, :seq_len, :seq_len]
            else:
                repeat_factor = (seq_len + cos.shape[1] - 1) // cos.shape[1]
                cos = cos.repeat(1, repeat_factor, 1, 1)[:, :seq_len]
                sin = sin.repeat(1, repeat_factor, 1, 1)[:, :seq_len]
                pad_len = seq_len - attention_bias.shape[-1]
                if pad_len > 0:
                    pad_val = -torch.finfo(attention_bias.dtype).max
                    attention_bias = F.pad(attention_bias, (0, pad_len, 0, pad_len), value=pad_val)
                attention_bias = attention_bias[:, :, :seq_len, :seq_len]

        position_embeddings = (cos, sin)

        batch = 1
        hidden_states = hidden_states.unsqueeze(0)  # [1, seq_len, dim]

        for blk in self.blocks:
            hidden_states = blk(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_bias,
            )

        hidden_states = hidden_states.squeeze(0)  # [seq_len, dim]
        hidden_states = self.post_layernorm(hidden_states)

        hidden_states = hidden_states.view(
            -1, self.spatial_merge_size, self.spatial_merge_size, hidden_states.shape[-1]
        )
        hidden_states = hidden_states.permute(0, 3, 1, 2)
        hidden_states = self.downsample(hidden_states).view(-1, self.config.out_hidden_size)

        merged_hidden_states = self.merger(hidden_states)
        return merged_hidden_states


def register_wrap_cls(hf_model):
    """Register wrap classes for tracing. Called before init_wrap_model."""
    pass
