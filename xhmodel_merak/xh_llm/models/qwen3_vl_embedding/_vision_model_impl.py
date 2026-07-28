# Copyright 2025 HOUMO AI
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
from collections.abc import Generator
from contextlib import contextmanager
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from xhquant import nn as xhnn
from xhquant.api import ConfigDict
from xhquant.utils.registry import DynamicModule

from ...register import XHLLM_TRACEABLE_MODULES
from ..qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLVisionAttention,
    Qwen3VLVisionBlock,
    Qwen3VLVisionModel,
    Qwen3VLVisionPatchEmbed,
    Qwen3VLVisionPatchMerger,
    rotate_half,
)


class _Qwen3VLEmbeddingVisionAttention(DynamicModule):
    def apply_rotary_pos_emb(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.enable_rope:
            q_embed = self.rope(q, cos, sin)
            k_embed = self.rope(k, cos, sin)
        else:
            q_embed = (q * cos) + (rotate_half(q) * sin)
            k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed

    def _setup(self, cfg: ConfigDict):
        self.only_first_block = False
        self.enable_rope = cfg.get("enable_rope", False)
        if self.enable_rope:
            self.rope = xhnn.Rope()

        self.max_size_w = cfg.max_size_w
        self.max_size_h = cfg.max_size_h
        self.patch_size = cfg.patch_size

        head_dim = self.qkv.out_features // 3 // self.num_heads
        self.kv_scale = 1 / math.sqrt(head_dim)

        weight = self.qkv.weight.data.clone()
        bias = self.qkv.bias.data.clone()
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

        if head_dim % 64 == 0:
            return

        new_head_dim = math.ceil(head_dim / 64) * 64
        padding_size = new_head_dim - head_dim
        if padding_size % 2 != 0:
            raise ValueError("padding_size must keep rotary halves aligned")
        qk_padding_size = padding_size // 2
        in_features = self.q_proj.in_features
        device = self.q_proj.weight.device
        weight_dtype = self.q_proj.weight.dtype
        bias_dtype = self.q_proj.bias.dtype

        def expand_qk(linear: nn.Linear) -> nn.Linear:
            linear_weight = linear.weight.data.transpose(0, 1).reshape(
                in_features,
                self.num_heads,
                2,
                head_dim // 2,
            )
            padded_weight = torch.zeros(
                (
                    in_features,
                    self.num_heads,
                    2,
                    head_dim // 2 + qk_padding_size,
                ),
                device=device,
                dtype=weight_dtype,
            )
            padded_weight[:, :, :, : head_dim // 2] = linear_weight
            padded_weight = padded_weight.reshape(
                in_features,
                self.num_heads * new_head_dim,
            ).transpose(0, 1)

            linear_bias = linear.bias.data.reshape(
                self.num_heads,
                2,
                head_dim // 2,
            )
            padded_bias = torch.zeros(
                (
                    self.num_heads,
                    2,
                    head_dim // 2 + qk_padding_size,
                ),
                device=device,
                dtype=bias_dtype,
            )
            padded_bias[:, :, : head_dim // 2] = linear_bias
            padded_bias = padded_bias.reshape(self.num_heads * new_head_dim)

            expanded = nn.Linear(
                in_features,
                self.num_heads * new_head_dim,
                bias=True,
            )
            expanded = expanded.to(
                device=device,
                dtype=weight_dtype,
            )
            expanded.weight.data.copy_(padded_weight)
            expanded.bias.data.copy_(padded_bias.to(weight_dtype))
            return expanded

        self.q_proj = expand_qk(self.q_proj)
        self.k_proj = expand_qk(self.k_proj)

        v_weight = self.v_proj.weight.data.transpose(0, 1).reshape(in_features, self.num_heads, head_dim)
        padded_v_weight = torch.zeros(
            (
                in_features,
                self.num_heads,
                head_dim + padding_size,
            ),
            device=device,
            dtype=weight_dtype,
        )
        padded_v_weight[:, :, :head_dim] = v_weight
        padded_v_weight = padded_v_weight.reshape(
            in_features,
            self.num_heads * new_head_dim,
        ).transpose(0, 1)

        v_bias = self.v_proj.bias.data.reshape(
            self.num_heads,
            head_dim,
        )
        padded_v_bias = torch.zeros(
            (self.num_heads, head_dim + padding_size),
            device=device,
            dtype=bias_dtype,
        )
        padded_v_bias[:, :head_dim] = v_bias
        padded_v_bias = padded_v_bias.reshape(self.num_heads * new_head_dim)

        expanded_v = nn.Linear(
            in_features,
            self.num_heads * new_head_dim,
            bias=True,
        )
        expanded_v = expanded_v.to(
            device=device,
            dtype=weight_dtype,
        )
        expanded_v.weight.data.copy_(padded_v_weight)
        expanded_v.bias.data.copy_(padded_v_bias.to(weight_dtype))
        self.v_proj = expanded_v

        padded_proj_weight = torch.zeros(
            (
                self.proj.out_features,
                self.num_heads,
                new_head_dim,
            ),
            device=device,
            dtype=weight_dtype,
        )
        padded_proj_weight[:, :, :head_dim] = self.proj.weight.data.reshape(
            self.proj.out_features,
            self.num_heads,
            head_dim,
        )
        padded_proj_weight = padded_proj_weight.reshape(
            self.proj.out_features,
            self.num_heads * new_head_dim,
        )
        proj_bias = self.proj.bias.data.clone()
        proj_out_features = self.proj.weight.shape[0]
        expanded_proj = nn.Linear(
            self.num_heads * new_head_dim,
            proj_out_features,
            bias=True,
        )
        expanded_proj = expanded_proj.to(
            device=device,
            dtype=weight_dtype,
        )
        expanded_proj.weight.data.copy_(padded_proj_weight)
        expanded_proj.bias.data.copy_(proj_bias)
        self.proj = expanded_proj

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        batch, seq_length, _ = hidden_states.shape
        cos, sin = position_embeddings
        lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        sample_outputs = []
        for batch_index in range(batch):
            sample_hidden_states = hidden_states[batch_index : batch_index + 1]
            sample_q = self.q_proj(sample_hidden_states).reshape(
                1,
                seq_length,
                self.num_heads,
                -1,
            )
            sample_k = self.k_proj(sample_hidden_states).reshape(
                1,
                seq_length,
                self.num_heads,
                -1,
            )
            sample_v = self.v_proj(sample_hidden_states).reshape(
                1,
                seq_length,
                self.num_heads,
                -1,
            )
            sample_q, sample_k = self.apply_rotary_pos_emb(
                sample_q,
                sample_k,
                cos.unsqueeze(0),
                sin.unsqueeze(0),
            )
            sample_q = sample_q.transpose(1, 2)
            sample_k = sample_k.transpose(1, 2)
            sample_v = sample_v.transpose(1, 2)

            chunk_outputs = []
            start = 0
            for length in lengths:
                end = start + length
                q_chunk = sample_q[:, :, start:end, :]
                k_chunk = sample_k[
                    :,
                    :,
                    start:end,
                    :,
                ].transpose(-2, -1)
                v_chunk = sample_v[:, :, start:end, :]
                q_chunk = q_chunk * self.kv_scale
                dtype = q_chunk.dtype
                attn_weights = torch.matmul(
                    q_chunk,
                    k_chunk,
                ).to(dtype)
                attn_weights = F.softmax(
                    attn_weights,
                    dim=-1,
                    dtype=torch.float32,
                ).to(dtype)
                chunk_outputs.append(torch.matmul(attn_weights, v_chunk))
                start = end

            sample_output = torch.cat(
                chunk_outputs,
                dim=2,
            )
            sample_output = sample_output.transpose(1, 2).reshape(
                1,
                seq_length,
                -1,
            )
            sample_outputs.append(sample_output)

        attn_output = torch.cat(sample_outputs, dim=0)
        return self.proj(attn_output)


class _Qwen3VLEmbeddingVisionBlock(DynamicModule):
    @torch.no_grad()
    def _setup(self, cfg: ConfigDict):
        pass

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


def _weight_denorm(conv2d: nn.Conv2d):
    norm = torch.tensor([0.5, 0.5, 0.5]).view(1, -1, 1, 1).to(conv2d.weight.device) * 255
    std = torch.tensor([0.5, 0.5, 0.5]).view(1, -1, 1, 1).to(conv2d.weight.device) * 255
    conv2d.weight.data = conv2d.weight.data / std
    bias_data = -(norm * conv2d.weight.data).sum(dim=[1, 2, 3])
    conv2d.bias = nn.Parameter(bias_data)


class _Qwen3VLEmbeddingVisionPatchEmbed(DynamicModule):
    @torch.no_grad()
    def _setup(self, cfg: ConfigDict):
        self.only_first_block = False
        self.max_size_w = cfg.max_size_w
        self.max_size_h = cfg.max_size_h
        self.patch_size = cfg.patch_size

        kernel_size = [self.patch_size, self.patch_size]
        device = self.proj.weight.device
        self.proj1 = nn.Conv2d(
            self.in_channels,
            self.embed_dim,
            kernel_size=kernel_size,
            stride=kernel_size,
            bias=False,
            device=device,
        )
        self.proj2 = nn.Conv2d(
            self.in_channels,
            self.embed_dim,
            kernel_size=kernel_size,
            stride=kernel_size,
            bias=False,
            device=device,
        )
        self.proj1.weight.data.copy_(self.proj.weight[:, :, 0, :, :].contiguous())
        self.proj2.weight.data.copy_(self.proj.weight[:, :, 1, :, :].contiguous())
        original_bias = self.proj.bias.data.clone() if self.proj.bias is not None else None

        _weight_denorm(self.proj1)
        _weight_denorm(self.proj2)
        if original_bias is not None:
            half_bias = (original_bias / 2).to(self.proj1.bias.dtype)
            self.proj1.bias.data.add_(half_bias)
            self.proj2.bias.data.add_(half_bias)

        self.proj1.to(
            self.proj.weight.device,
            dtype=self.proj.weight.dtype,
        )
        self.proj2.to(
            self.proj.weight.device,
            dtype=self.proj.weight.dtype,
        )
        del self.proj
        if self.temporal_patch_size != 2:
            raise ValueError("temporal_patch_size must be 2")

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        batch, channels, frames, height, width = hidden_states.shape
        temporal_size = frames // self.temporal_patch_size
        hidden_states = hidden_states.reshape(
            batch,
            channels,
            temporal_size,
            self.temporal_patch_size,
            height,
            width,
        )
        hidden_states = hidden_states.permute(0, 2, 1, 3, 4, 5).reshape(
            batch * temporal_size,
            channels,
            self.temporal_patch_size,
            height,
            width,
        )
        hidden_states0, hidden_states1 = hidden_states.unbind(2)
        hidden_states = self.proj1(hidden_states0) + self.proj2(hidden_states1)

        _, channels, height, width = hidden_states.shape
        hidden_states = hidden_states.reshape(
            batch,
            temporal_size,
            channels,
            height // 2,
            2,
            width // 2,
            2,
        )
        hidden_states = hidden_states.permute(
            0,
            1,
            3,
            5,
            4,
            6,
            2,
        )
        return hidden_states.reshape(batch, -1, channels)


class _Qwen3VLEmbeddingVisionPatchMerger(DynamicModule):
    def _setup(self, cfg: ConfigDict):
        pass

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch = hidden_states.shape[0]
        if self.use_postshuffle_norm:
            hidden_states = hidden_states.view(
                batch,
                -1,
                self.hidden_size,
            )
        hidden_states = self.norm(hidden_states).view(
            batch,
            -1,
            self.hidden_size,
        )
        return self.linear_fc2(self.act_fn(self.linear_fc1(hidden_states)))


class _Qwen3VLEmbeddingVisionModel(DynamicModule):
    @torch.no_grad()
    def _setup(self, cfg: ConfigDict):
        self.only_first_block = False
        self.patch_size = cfg.patch_size
        device = next(self.parameters()).device
        self.max_size_w = cfg.max_size_w
        self.max_size_h = cfg.max_size_h
        self.max_size_t = cfg.max_size_t
        self.temporal_patch_size = cfg.temporal_patch_size
        grid_size_w = self.max_size_w // self.patch_size
        grid_size_h = self.max_size_h // self.patch_size
        grid_size_t = self.max_size_t // self.temporal_patch_size
        seq_length = grid_size_t * grid_size_h * grid_size_w

        if self.max_size_w % self.patch_size != 0:
            raise ValueError("max_size_w must be divisible by patch_size")
        if self.max_size_h % self.patch_size != 0:
            raise ValueError("max_size_h must be divisible by patch_size")
        cu_seqlens = torch.repeat_interleave(
            torch.tensor(
                [grid_size_h * grid_size_w],
                device=device,
                dtype=torch.int32,
            ),
            repeats=grid_size_t,
        ).cumsum(dim=0)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
        self.register_buffer(
            "cu_seqlens",
            cu_seqlens,
            persistent=False,
        )

        grid_thw = torch.tensor(
            [[grid_size_t, grid_size_h, grid_size_w]],
            device=device,
        )
        self.register_buffer(
            "patch_pos_embeds",
            self.fast_pos_embed_interpolate(grid_thw).unsqueeze(0),
            persistent=False,
        )
        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        rotary_pos_emb = rotary_pos_emb.reshape(
            seq_length,
            -1,
        )
        emb = torch.cat(
            (rotary_pos_emb, rotary_pos_emb),
            dim=-1,
        )
        cos = emb.cos().unsqueeze(-2)
        sin = emb.sin().unsqueeze(-2)

        last_rope_dim = emb.shape[-1]
        if last_rope_dim % 64 != 0:
            padding_size = (64 - last_rope_dim % 64) // 2
            cos_first, cos_second = cos.chunk(2, dim=-1)
            sin_first, sin_second = sin.chunk(2, dim=-1)
            cos_first = F.pad(
                cos_first,
                (0, padding_size),
            )
            cos_second = F.pad(
                cos_second,
                (0, padding_size),
            )
            sin_first = F.pad(
                sin_first,
                (0, padding_size),
            )
            sin_second = F.pad(
                sin_second,
                (0, padding_size),
            )
            cos = torch.cat(
                (cos_first, cos_second),
                dim=-1,
            )
            sin = torch.cat(
                (sin_first, sin_second),
                dim=-1,
            )

        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, list[torch.Tensor]]:
        hidden_states = self.patch_embed(hidden_states)
        hidden_states = hidden_states + self.patch_pos_embeds
        position_embeddings = (self.cos, self.sin)
        deepstack_feature_lists = []
        for layer_num, block in enumerate(self.blocks):
            hidden_states = block(
                hidden_states,
                cu_seqlens=self.cu_seqlens,
                position_embeddings=position_embeddings,
            )
            if layer_num in self.deepstack_visual_indexes:
                merger_index = self.deepstack_visual_indexes.index(layer_num)
                deepstack_feature = self.deepstack_merger_list[merger_index](hidden_states)
                deepstack_feature_lists.append(deepstack_feature)
        hidden_states = self.merger(hidden_states)
        return hidden_states, deepstack_feature_lists


_VISION_WRAPPERS = {
    Qwen3VLVisionAttention: _Qwen3VLEmbeddingVisionAttention,
    Qwen3VLVisionBlock: _Qwen3VLEmbeddingVisionBlock,
    Qwen3VLVisionPatchEmbed: _Qwen3VLEmbeddingVisionPatchEmbed,
    Qwen3VLVisionPatchMerger: (_Qwen3VLEmbeddingVisionPatchMerger),
    Qwen3VLVisionModel: _Qwen3VLEmbeddingVisionModel,
}


@contextmanager
def embedding_vision_wrap_cls_scope() -> Generator[None, None, None]:
    """Install the legacy embedding Vision wrappers temporarily."""
    # Load the default registrations before saving and replacing them.
    from ..qwen3_vl import _vision_model_impl as _default_impl  # noqa: F401

    registry = XHLLM_TRACEABLE_MODULES
    previous = {}
    for model_cls, wrapper_cls in _VISION_WRAPPERS.items():
        previous[model_cls] = (
            registry._registry.get(model_cls),
            registry._key_registry.get(model_cls),
            model_cls in registry._dynamic_classes,
            registry._dynamic_classes.get(model_cls),
        )
        registry._registry[model_cls] = wrapper_cls
        registry._key_registry[model_cls] = model_cls.__name__
        registry._dynamic_classes.pop(model_cls, None)

    try:
        yield
    finally:
        for model_cls, state in previous.items():
            (
                previous_wrapper,
                previous_key,
                had_dynamic_cls,
                previous_dynamic_cls,
            ) = state
            if previous_wrapper is None:
                registry._registry.pop(model_cls, None)
            else:
                registry._registry[model_cls] = previous_wrapper
            if previous_key is None:
                registry._key_registry.pop(model_cls, None)
            else:
                registry._key_registry[model_cls] = previous_key
            if had_dynamic_cls:
                registry._dynamic_classes[model_cls] = previous_dynamic_cls
            else:
                registry._dynamic_classes.pop(model_cls, None)


__all__ = ["embedding_vision_wrap_cls_scope"]
