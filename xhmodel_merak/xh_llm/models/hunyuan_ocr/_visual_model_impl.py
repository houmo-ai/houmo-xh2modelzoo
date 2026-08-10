# Copyright (C) 2026 THL A29 Limited, a Tencent company and the HuggingFace Inc. team.
# Copyright 2025 HOUMO AI (modifications)
# Licensed under the Apache License, Version 2.0.
# See licenses/transformers/LICENSE for the full license text.
# Source: https://github.com/huggingface/transformers/blob/v5.13.0/src/transformers/models/hunyuan_vl/modular_hunyuan_vl.py

"""Traceable single-bucket vision modules for the HunyuanOCR-1.5 target model."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers.models.hunyuan_vl.modeling_hunyuan_vl import (
    HunYuanVLVisionAttention,
    HunYuanVLVisionBlock,
    HunYuanVLVisionPatchEmbed,
    HunYuanVLVisionPatchMerger,
    HunYuanVLVisionTransformer,
)

from xhquant.utils.registry import DynamicModule

from ...register import XHLLM_TRACEABLE_MODULES


def _optional_bucket_dimension(cfg: Any, name: str) -> int | None:
    value = cfg.get(name, None)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value)


@XHLLM_TRACEABLE_MODULES.register_module({HunYuanVLVisionAttention: "HunYuanVLVisionAttention"})
class _HunYuanOCRVisionAttention(DynamicModule):
    def _setup(self, cfg: Any = None):
        self.attention_scale = float(self.head_dim**-0.5)
        return self

    def graph_forward(self, hidden_states: Tensor) -> Tensor:
        batch_size, sequence_length, _ = hidden_states.shape
        query = self.q_proj(hidden_states).reshape(batch_size, sequence_length, self.num_heads, self.head_dim)
        key = self.k_proj(hidden_states).reshape(batch_size, sequence_length, self.num_heads, self.head_dim)
        value = self.v_proj(hidden_states).reshape(batch_size, sequence_length, self.num_heads, self.head_dim)

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        attention_weights = torch.matmul(query, key.transpose(-2, -1)) * self.attention_scale
        attention_weights = F.softmax(attention_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        attention_output = torch.matmul(attention_weights, value)
        attention_output = attention_output.transpose(1, 2).reshape(batch_size, sequence_length, -1)
        return self.o_proj(attention_output)


@XHLLM_TRACEABLE_MODULES.register_module({HunYuanVLVisionBlock: "HunYuanVLVisionBlock"})
class _HunYuanOCRVisionBlock(DynamicModule):
    def _setup(self, cfg: Any = None):
        return self

    def graph_forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = hidden_states + self.self_attn(self.layer_norm1(hidden_states))
        return hidden_states + self.mlp(self.layer_norm2(hidden_states))


@XHLLM_TRACEABLE_MODULES.register_module({HunYuanVLVisionPatchMerger: "HunYuanVLVisionPatchMerger"})
class _HunYuanOCRVisionPatchMerger(DynamicModule):
    def _setup(self, cfg: Any):
        grid_h = _optional_bucket_dimension(cfg, "grid_h")
        grid_w = _optional_bucket_dimension(cfg, "grid_w")
        num_patches = _optional_bucket_dimension(cfg, "num_patches")
        if grid_h is None or grid_w is None or num_patches is None:
            return self
        self.grid_h = grid_h
        self.grid_w = grid_w
        if self.grid_h * self.grid_w != num_patches:
            raise ValueError("HunyuanOCR patch merger grid does not match num_patches")
        if self.grid_h % self.spatial_merge_size or self.grid_w % self.spatial_merge_size:
            raise ValueError("HunyuanOCR patch merger grid must be divisible by spatial_merge_size")
        self.merged_h = self.grid_h // self.spatial_merge_size
        self.merged_w = self.grid_w // self.spatial_merge_size
        return self

    def graph_forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.before_rms(hidden_states)
        batch_size = hidden_states.shape[0]
        hidden_states = hidden_states.permute(0, 2, 1).reshape(
            batch_size,
            -1,
            self.grid_h,
            self.grid_w,
        )
        hidden_states = self.proj_out(self.proj_act(self.proj_conv(hidden_states)))
        channels = hidden_states.shape[1]
        newline = self.image_newline.reshape(1, channels, 1, 1).expand(
            batch_size,
            channels,
            self.merged_h,
            1,
        )
        hidden_states = torch.cat((hidden_states, newline), dim=-1)
        hidden_states = hidden_states.reshape(batch_size, channels, -1).permute(0, 2, 1)
        hidden_states = self.mlp(hidden_states)

        begin = self.image_begin.reshape(1, 1, -1).expand(batch_size, 1, -1)
        end = self.image_end.reshape(1, 1, -1).expand(batch_size, 1, -1)
        return self.after_rms(torch.cat((begin, hidden_states, end), dim=1))


@XHLLM_TRACEABLE_MODULES.register_module({HunYuanVLVisionPatchEmbed: "HunYuanVLVisionPatchEmbed"})
class _HunYuanOCRVisionPatchEmbed(DynamicModule):
    @torch.no_grad()
    def _setup(self, cfg: Any):
        grid_h = _optional_bucket_dimension(cfg, "grid_h")
        grid_w = _optional_bucket_dimension(cfg, "grid_w")
        num_patches = _optional_bucket_dimension(cfg, "num_patches")
        if grid_h is None or grid_w is None or num_patches is None:
            return self
        if grid_h * grid_w != num_patches:
            raise ValueError("HunyuanOCR visual grid does not match num_patches")

        conv = self.patch_embedding
        patch_dim = int(conv.in_channels * self.patch_size * self.patch_size)
        self.patch_projection = nn.Linear(
            patch_dim,
            self.embed_dim,
            bias=conv.bias is not None,
            device=conv.weight.device,
            dtype=conv.weight.dtype,
        )
        self.patch_projection.weight = nn.Parameter(deepcopy(conv.weight.data.reshape(self.embed_dim, patch_dim)))
        if conv.bias is not None:
            self.patch_projection.bias = nn.Parameter(deepcopy(conv.bias.data))

        position_grid = self.position_embedding.weight[1:].reshape(
            1,
            self.position_edge,
            self.position_edge,
            self.embed_dim,
        )
        position_grid = position_grid.permute(0, 3, 1, 2).float()
        position_grid = F.interpolate(
            position_grid,
            size=(grid_h, grid_w),
            mode=self.config.interpolate_mode,
            align_corners=False,
        )
        position_grid = position_grid.permute(0, 2, 3, 1).reshape(1, grid_h * grid_w, self.embed_dim)
        self.register_buffer(
            "static_position_embedding",
            position_grid.to(dtype=conv.weight.dtype),
            persistent=False,
        )
        del self.patch_embedding
        del self.position_embedding
        return self

    def graph_forward(self, pixel_values: Tensor) -> Tensor:
        patch_embeddings = self.patch_projection(pixel_values.to(self.patch_projection.weight.dtype)).unsqueeze(0)
        return patch_embeddings + self.static_position_embedding


@XHLLM_TRACEABLE_MODULES.register_module({HunYuanVLVisionTransformer: "HunYuanVLVisionTransformer"})
class _HunYuanOCRVisionTransformer(DynamicModule):
    def _setup(self, cfg: Any):
        self.only_first_block = bool(cfg.get("only_first_block", False))
        return self

    def graph_forward(self, pixel_values: Tensor) -> Tensor:
        hidden_states = self.embeddings(pixel_values)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
            if self.only_first_block:
                break
        return self.patch_merger(hidden_states)
