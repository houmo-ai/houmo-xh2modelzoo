# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0

"""Export Cosmos3-Nano vision encoder core from patch tokens.

This graph intentionally starts after Qwen3-VL patch_embed. The original
patch_embed is Conv3d over [T, H, W] patches and currently trips xhquant ONNX
frontend Conv padding handling for 5D inputs. Keeping it outside this graph lets
the 27 transformer blocks, merger, and deepstack mergers be exported and
converted independently.
"""

from __future__ import annotations

import argparse
import sys
import json
from pathlib import Path
from typing import Any


_COSMOS3_ROOT = Path(__file__).resolve().parents[3]
if str(_COSMOS3_ROOT) not in sys.path:
    sys.path.insert(0, str(_COSMOS3_ROOT))

from common.paths import default_model_root  # noqa: E402

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel


DEFAULT_MODEL = default_model_root()
_COSMOS3_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_ROOT = _COSMOS3_ROOT / "data" / "vision_encoder_core_smoke"
OUTPUT_NAMES = ["image_embeds", "deepstack_feature_0", "deepstack_feature_1", "deepstack_feature_2"]


def rotate_half_4d(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_vision_4d(
    query: torch.Tensor, key: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    orig_q_dtype = query.dtype
    orig_k_dtype = key.dtype
    query = query.float()
    key = key.float()
    cos = cos.unsqueeze(0).unsqueeze(0).float()
    sin = sin.unsqueeze(0).unsqueeze(0).float()
    query_embed = (query * cos) + (rotate_half_4d(query) * sin)
    key_embed = (key * cos) + (rotate_half_4d(key) * sin)
    return query_embed.to(orig_q_dtype), key_embed.to(orig_k_dtype)


class Qwen3VLVisionAttention4D(nn.Module):
    """Equivalent eager attention with RoPE applied after 4D layout conversion."""

    def __init__(self, source: nn.Module):
        super().__init__()
        self.qkv = source.qkv
        self.proj = source.proj
        self.num_heads = int(source.num_heads)
        self.scaling = float(source.scaling)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        del cu_seqlens, rotary_pos_emb, kwargs
        seq_length = hidden_states.shape[0]
        query_states, key_states, value_states = (
            self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        )
        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)

        if position_embeddings is None:
            raise RuntimeError("position_embeddings is required")
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb_vision_4d(query_states, key_states, cos, sin)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
        attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(seq_length, -1)
        return self.proj(attn_output)


def use_4d_attention(model: Qwen3VLVisionModel) -> None:
    for block in model.blocks:
        block.attn = Qwen3VLVisionAttention4D(block.attn)


class Cosmos3VisionEncoderCoreWrapper(nn.Module):
    """Qwen3-VL vision tower after patch_embed."""

    def __init__(self, model: Qwen3VLVisionModel, height: int, width: int):
        super().__init__()
        self.model = model
        patch_size = int(model.config.patch_size)
        grid_thw = torch.tensor([[1, height // patch_size, width // patch_size]], dtype=torch.int64)
        self.register_buffer("grid_thw", grid_thw, persistent=False)

    def forward(self, patch_embeds: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        grid_thw = self.grid_thw.to(patch_embeds.device)
        hidden_states = patch_embeds + self.model.fast_pos_embed_interpolate(grid_thw)

        rotary_pos_emb = self.model.rot_pos_emb(grid_thw)
        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        deepstack_feature_lists = []
        for layer_num, block in enumerate(self.model.blocks):
            hidden_states = block(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
            )
            if layer_num in self.model.deepstack_visual_indexes:
                merger_index = self.model.deepstack_visual_indexes.index(layer_num)
                deepstack_feature = self.model.deepstack_merger_list[merger_index](hidden_states)
                deepstack_feature_lists.append(deepstack_feature)

        image_embeds = self.model.merger(hidden_states)
        return image_embeds, deepstack_feature_lists[0], deepstack_feature_lists[1], deepstack_feature_lists[2]


