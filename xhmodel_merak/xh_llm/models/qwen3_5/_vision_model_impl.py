"""
Qwen3.5 Vision Model Implementation for XH2a export.

Adapted from xhquant_llm/models/qwen3_vl/_vision_model_impl.py.

Key differences from Qwen3VL:
  - Qwen3.5 does NOT have deepstack_visual_indexes / deepstack_merger_list.
    The VisionModel directly returns merged_hidden_states via BaseModelOutputWithPooling.
  - All other components (PatchEmbed, VisionAttention, VisionBlock, PatchMerger,
    RotaryEmbedding) are structurally identical to Qwen3VL.
"""

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from xhquant import nn as xhnn
from xhquant.api import ConfigDict
from xhquant.utils.registry import DynamicModule

from ...register import XHLLM_TRACEABLE_MODULES
from .modeling_qwen3_5 import (
    Qwen3_5VisionAttention,
    Qwen3_5VisionBlock,
    Qwen3_5VisionModel,
    Qwen3_5VisionPatchEmbed,
    Qwen3_5VisionPatchMerger,
    rotate_half,
)


@XHLLM_TRACEABLE_MODULES.register_module(
    {
        Qwen3_5VisionAttention: "Qwen3_5VisionAttention",
    }
)
class _Qwen3_5VisionAttention(DynamicModule):  # noqa: N801
    def apply_rotary_pos_emb(
        self, q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
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

        if head_dim % 64 != 0:
            new_head_dim = math.ceil(head_dim / 64) * 64
            padding_size = new_head_dim - head_dim
            assert padding_size % 2 == 0, "padding_size must keep rotary halves aligned"
            qk_padding_size = padding_size // 2
            in_features = self.q_proj.in_features
            device = self.q_proj.weight.device
            weight_dtype = self.q_proj.weight.dtype
            bias_dtype = self.q_proj.bias.dtype

            def _expand_qk(linear: nn.Linear) -> nn.Linear:
                weight = linear.weight.data.transpose(0, 1).reshape(in_features, self.num_heads, 2, head_dim // 2)
                padded_weight = torch.zeros(
                    (in_features, self.num_heads, 2, head_dim // 2 + qk_padding_size),
                    device=device,
                    dtype=weight_dtype,
                )
                padded_weight[:, :, :, : head_dim // 2] = weight
                padded_weight = padded_weight.reshape(in_features, self.num_heads * new_head_dim).transpose(0, 1)

                bias = linear.bias.data.reshape(self.num_heads, 2, head_dim // 2)
                padded_bias = torch.zeros(
                    (self.num_heads, 2, head_dim // 2 + qk_padding_size),
                    device=device,
                    dtype=bias_dtype,
                )
                padded_bias[:, :, : head_dim // 2] = bias
                padded_bias = padded_bias.reshape(self.num_heads * new_head_dim)

                expanded = nn.Linear(in_features, self.num_heads * new_head_dim, bias=True)
                expanded = expanded.to(device=device, dtype=weight_dtype)
                expanded.weight.data.copy_(padded_weight)
                expanded.bias.data.copy_(padded_bias.to(weight_dtype))
                return expanded

            self.q_proj = _expand_qk(self.q_proj)
            self.k_proj = _expand_qk(self.k_proj)

            v_weight = self.v_proj.weight.data.transpose(0, 1).reshape(in_features, self.num_heads, head_dim)
            padded_v_weight = torch.zeros(
                (in_features, self.num_heads, head_dim + padding_size),
                device=device,
                dtype=weight_dtype,
            )
            padded_v_weight[:, :, :head_dim] = v_weight
            padded_v_weight = padded_v_weight.reshape(in_features, self.num_heads * new_head_dim).transpose(0, 1)

            v_bias = self.v_proj.bias.data.reshape(self.num_heads, head_dim)
            padded_v_bias = torch.zeros(
                (self.num_heads, head_dim + padding_size),
                device=device,
                dtype=bias_dtype,
            )
            padded_v_bias[:, :head_dim] = v_bias
            padded_v_bias = padded_v_bias.reshape(self.num_heads * new_head_dim)

            expanded_v = nn.Linear(in_features, self.num_heads * new_head_dim, bias=True)
            expanded_v = expanded_v.to(device=device, dtype=weight_dtype)
            expanded_v.weight.data.copy_(padded_v_weight)
            expanded_v.bias.data.copy_(padded_v_bias.to(weight_dtype))
            self.v_proj = expanded_v

            padded_proj_weight = torch.zeros(
                (self.proj.out_features, self.num_heads, new_head_dim),
                device=device,
                dtype=weight_dtype,
            )
            padded_proj_weight[:, :, :head_dim] = self.proj.weight.data.reshape(
                self.proj.out_features, self.num_heads, head_dim
            )
            padded_proj_weight = padded_proj_weight.reshape(self.proj.out_features, self.num_heads * new_head_dim)
            proj_bias = self.proj.bias.data.clone()
            proj_out_features, proj_in_features = self.proj.weight.shape
            expanded_proj = nn.Linear(self.num_heads * new_head_dim, proj_out_features, bias=True)
            expanded_proj = expanded_proj.to(device=device, dtype=weight_dtype)
            expanded_proj.weight.data.copy_(padded_proj_weight)
            expanded_proj.bias.data.copy_(proj_bias)
            self.proj = expanded_proj

    def forward(self, hidden_states, position_embeddings):
        batch, seq_length, _ = hidden_states.shape

        q = self.q_proj(hidden_states).reshape(batch, seq_length, self.num_heads, -1)
        k = self.k_proj(hidden_states).reshape(batch, seq_length, self.num_heads, -1)
        v = self.v_proj(hidden_states).reshape(batch, seq_length, self.num_heads, -1)
        cos, sin = position_embeddings
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
        q, k = self.apply_rotary_pos_emb(q, k, cos, sin)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        k = k.transpose(-2, -1)
        q = q * self.kv_scale
        dtype = q.dtype
        attn_weights = torch.matmul(q, k).to(dtype)
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(dtype)
        attn_output = torch.matmul(attn_weights, v)

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(batch, seq_length, -1)
        attn_output = self.proj(attn_output)
        return attn_output


@XHLLM_TRACEABLE_MODULES.register_module(
    {
        Qwen3_5VisionBlock: "Qwen3_5VisionBlock",
    }
)
class _Qwen3_5VisionBlock(DynamicModule):  # noqa: N801
    @torch.no_grad()
    def _setup(self, cfg: ConfigDict):
        pass

    def forward(self, hidden_states, position_embeddings):
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            position_embeddings=position_embeddings,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


def weight_denorm(
    conv2d: nn.Conv2d,
):
    norm = torch.tensor([0.5, 0.5, 0.5]).view(1, -1, 1, 1).to(conv2d.weight.device) * 255
    std = torch.tensor([0.5, 0.5, 0.5]).view(1, -1, 1, 1).to(conv2d.weight.device) * 255
    conv2d.weight.data = conv2d.weight.data / std
    bias_data = -(norm * conv2d.weight.data).sum(dim=[1, 2, 3])
    conv2d.bias = nn.Parameter(bias_data)


@XHLLM_TRACEABLE_MODULES.register_module(
    {
        Qwen3_5VisionPatchEmbed: "Qwen3_5VisionPatchEmbed",
    }
)
class _Qwen3_5VisionPatchEmbed(DynamicModule):  # noqa: N801
    @torch.no_grad()
    def _setup(self, cfg: ConfigDict):
        self.only_first_block = False
        self.max_size_w = cfg.max_size_w
        self.max_size_h = cfg.max_size_h
        self.patch_size = cfg.patch_size

        kernel_size = [self.patch_size, self.patch_size]
        dev = self.proj.weight.device
        self.proj1 = nn.Conv2d(
            self.in_channels,
            self.embed_dim,
            kernel_size=kernel_size,
            stride=kernel_size,
            bias=False,
            device=dev,
        )
        self.proj2 = nn.Conv2d(
            self.in_channels,
            self.embed_dim,
            kernel_size=kernel_size,
            stride=kernel_size,
            bias=False,
            device=dev,
        )
        self.proj1.weight.data.copy_(self.proj.weight[:, :, 0, :, :].contiguous())
        self.proj2.weight.data.copy_(self.proj.weight[:, :, 1, :, :].contiguous())

        # Save original Conv3d bias before decomposition
        original_bias = self.proj.bias.data.clone() if self.proj.bias is not None else None

        weight_denorm(self.proj1)
        weight_denorm(self.proj2)

        # Add original Conv3d bias back (it was lost during decomposition)
        if original_bias is not None:
            self.proj2.bias.data += original_bias.to(self.proj2.bias.device)

        self.proj1.to(self.proj.weight.device, dtype=self.proj.weight.dtype)
        self.proj2.to(self.proj.weight.device, dtype=self.proj.weight.dtype)

        del self.proj
        assert self.temporal_patch_size == 2, "temporal_patch_size must be 2"

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states: [b, c, t, h, w]
        b, c, t, h, w = hidden_states.shape
        t_new = t // self.temporal_patch_size
        hidden_states = hidden_states.reshape(b, c, t_new, self.temporal_patch_size, h, w)
        hidden_states = hidden_states.permute(0, 2, 1, 3, 4, 5).reshape(b * t_new, c, self.temporal_patch_size, h, w)
        hidden_states0, hidden_states1 = hidden_states.unbind(2)
        y1 = self.proj1(hidden_states0)
        y2 = self.proj2(hidden_states1)
        hidden_states = y1 + y2

        _, c, h, w = hidden_states.shape
        hidden_states = hidden_states.reshape(b, t_new, c, h // 2, 2, w // 2, 2)
        hidden_states = hidden_states.permute(0, 1, 3, 5, 4, 6, 2)
        hidden_states = hidden_states.reshape(b, -1, c)
        return hidden_states


@XHLLM_TRACEABLE_MODULES.register_module(
    {
        Qwen3_5VisionPatchMerger: "Qwen3_5VisionPatchMerger",
    }
)
class _Qwen3_5VisionPatchMerger(DynamicModule):  # noqa: N801
    def _setup(self, cfg: ConfigDict):
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        x = self.norm(x.view(batch, -1, self.hidden_size) if self.use_postshuffle_norm else x).view(
            batch, -1, self.hidden_size
        )
        x = self.linear_fc2(self.act_fn(self.linear_fc1(x)))
        return x


@XHLLM_TRACEABLE_MODULES.register_module(
    {
        Qwen3_5VisionModel: "Qwen3_5VisionModel",
    }
)
class _Qwen3_5VisionModel(DynamicModule):  # noqa: N801
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

        assert self.max_size_w % self.patch_size == 0, "max_size_w must be divisible by patch_size"
        assert self.max_size_h % self.patch_size == 0, "max_size_h must be divisible by patch_size"
        cu_seqlens = torch.tensor([0, seq_length]).to(device)
        self.register_buffer("cu_seqlens", cu_seqlens, persistent=False)

        grid_thw = torch.tensor(
            [[grid_size_t, self.max_size_h // self.patch_size, self.max_size_w // self.patch_size]]
        ).to(device)

        self.register_buffer(
            "patch_pos_embeds", self.fast_pos_embed_interpolate(grid_thw).unsqueeze(0), persistent=False
        )
        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_length, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        cos = emb.cos().unsqueeze(-2)
        sin = emb.sin().unsqueeze(-2)

        last_rope_dim = emb.shape[-1]
        if last_rope_dim % 64 != 0:
            padding_size = (64 - last_rope_dim % 64) // 2
            cos_first, cos_second = cos.chunk(2, dim=-1)
            sin_first, sin_second = sin.chunk(2, dim=-1)
            cos_first = F.pad(cos_first, (0, padding_size))
            cos_second = F.pad(cos_second, (0, padding_size))
            sin_first = F.pad(sin_first, (0, padding_size))
            sin_second = F.pad(sin_second, (0, padding_size))
            cos = torch.cat((cos_first, cos_second), dim=-1)
            sin = torch.cat((sin_first, sin_second), dim=-1)

        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.patch_embed(hidden_states)
        hidden_states = hidden_states + self.patch_pos_embeds
        cos = self.cos
        sin = self.sin
        position_embeddings = (cos, sin)
        for blk in self.blocks:
            hidden_states = blk(hidden_states, position_embeddings=position_embeddings)
        hidden_states = self.merger(hidden_states)
        return hidden_states


def register_wrap_cls(hf_model):
    pass
