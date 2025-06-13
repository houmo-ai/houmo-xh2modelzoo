import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.qwen2_vl.modeling_qwen2_vl import (  # apply_rotary_pos_emb_vision,
    PatchEmbed,
    Qwen2VisionTransformerPretrainedModel,
    VisionSdpaAttention,
    rotate_half,
)
from xhquant.api import ConfigDict
from xhquant.utils.registry import DynamicModule

from ..builder import XHLLM_TRACEABLE_MODULES


def apply_rotary_pos_emb_vision(tensor: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    # orig_dtype = tensor.dtype
    cos = freqs.cos()
    sin = freqs.sin()

    # cos = cos.unsqueeze(1).repeat(1, 1, 2).unsqueeze(0)
    # sin = sin.unsqueeze(1).repeat(1, 1, 2).unsqueeze(0)
    cos = cos.unsqueeze(1).repeat(1, 1, 2)
    sin = sin.unsqueeze(1).repeat(1, 1, 2)
    output = (tensor * cos) + (rotate_half(tensor) * sin)
    # output = output.to(orig_dtype)
    return output


@XHLLM_TRACEABLE_MODULES.register_module(
    {
        VisionSdpaAttention: "VisionSdpaAttention",
    }
)
class _VisionSdpaAttention(DynamicModule):
    def _setup(self, cfg: ConfigDict):
        self.only_first_block = False
        self.max_size = cfg.max_size
        self.patch_size = cfg.patch_size

        grid_size = self.max_size // self.patch_size
        seq_length = grid_size**2
        attention_mask = torch.zeros([1, seq_length, seq_length], dtype=torch.bool)
        cu_seqlens = torch.tensor([0, seq_length])
        for i in range(1, len(cu_seqlens)):
            attention_mask[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = True

        attention_bias = torch.zeros(1, seq_length, seq_length, dtype=torch.float32)
        attention_bias.masked_fill_(attention_mask.logical_not(), float("inf"))
        self.register_buffer("attention_bias", attention_bias, persistent=False)
        self.register_buffer("attention_mask", attention_mask, persistent=False)
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

        # self.q_proj = nn.Linear(self.qkv.in_features, self.qkv.out_features // 3, bias=self.qkv.bias is not None)
        # self.k_proj = nn.Linear(self.qkv.in_features, self.qkv.out_features // 3, bias=self.qkv.bias is not None)
        # self.v_proj = nn.Linear(self.qkv.in_features, self.qkv.out_features // 3, bias=self.qkv.bias is not None)
        # self.q_proj.weight.data = self.qkv.weight.data[: self.qkv.out_features // 3, :]
        # self.k_proj.weight.data = self.qkv.weight.data[self.qkv.out_features // 3 : 2 * self.qkv.out_features // 3, :]
        # self.v_proj.weight.data = self.qkv.weight.data[2 * self.qkv.out_features // 3 :, :]
        # if self.qkv.bias is not None:
        #     self.q_proj.bias.data = self.qkv.bias.data[: self.qkv.out_features // 3]
        #     self.k_proj.bias.data = self.qkv.bias.data[self.qkv.out_features // 3 : 2 * self.qkv.out_features // 3]
        #     self.v_proj.bias.data = self.qkv.bias.data[2 * self.qkv.out_features // 3 :]

    def forward(
        self, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor, rotary_pos_emb: torch.Tensor = None
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        # q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        q = self.q_proj(hidden_states).reshape(seq_length, self.num_heads, -1)
        k = self.k_proj(hidden_states).reshape(seq_length, self.num_heads, -1)
        v = self.v_proj(hidden_states).reshape(seq_length, self.num_heads, -1)
        # q = apply_rotary_pos_emb_vision(q.unsqueeze(0), rotary_pos_emb).squeeze(0)
        # k = apply_rotary_pos_emb_vision(k.unsqueeze(0), rotary_pos_emb).squeeze(0)
        q = apply_rotary_pos_emb_vision(q, rotary_pos_emb)
        k = apply_rotary_pos_emb_vision(k, rotary_pos_emb)

        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)
        if False:
            attn_output = F.scaled_dot_product_attention(q, k, v, self.attention_mask, dropout_p=0.0)
        else:
            k = k.transpose(-2, -1)
            q = q * self.kv_scale
            attn_weights = torch.matmul(q, k)
            attn_weights += self.attention_bias
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_weights = F.dropout(attn_weights, p=0.0, training=self.training)
            attn_output = torch.matmul(attn_weights, v)

        attn_output = attn_output.transpose(0, 1)
        attn_output = attn_output.reshape(seq_length, -1)
        attn_output = self.proj(attn_output)
        return attn_output


@XHLLM_TRACEABLE_MODULES.register_module(
    {
        PatchEmbed: "PatchEmbed",
    }
)
class _PatchEmbed(DynamicModule):
    def _setup(self, cfg: ConfigDict):
        self.only_first_block = False
        self.max_size = cfg.max_size
        self.patch_size = cfg.patch_size

        kernel_size = [self.patch_size, self.patch_size]
        self.proj1 = nn.Conv2d(
            self.in_channels,
            self.embed_dim,
            kernel_size=kernel_size,
            stride=kernel_size,
            bias=False,
        )
        self.proj2 = nn.Conv2d(
            self.in_channels,
            self.embed_dim,
            kernel_size=kernel_size,
            stride=kernel_size,
            bias=False,
        )

        self.proj1.weight.data = self.proj.weight[:, :, 0, :, :].contiguous()
        self.proj2.weight.data = self.proj.weight[:, :, 1, :, :].contiguous()
        del self.proj
        assert self.temporal_patch_size == 2, "temporal_patch_size must be 2"

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # target_dtype = self.proj.weight.dtype
        # hidden_states = hidden_states.view(
        #     -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
        # )
        # hidden_states = self.proj(hidden_states).view(-1, self.embed_dim)
        # x1 = hidden_states[:, :, 0, :, :]
        # x2 = hidden_states[:, :, 1, :, :]

        # 在单图输入中x1 == x2

        hidden_states = hidden_states.view(-1, self.in_channels, self.patch_size, self.patch_size)

        y1 = self.proj1(hidden_states)
        y2 = self.proj2(hidden_states)
        hidden_states = (y1 + y2).view(-1, self.embed_dim)

        return hidden_states


@XHLLM_TRACEABLE_MODULES.register_module(
    {
        Qwen2VisionTransformerPretrainedModel: "Qwen2VisionTransformerPretrainedModel",
    }
)
class _Qwen2VisionTransformerPretrainedModel(DynamicModule):
    def _setup(self, cfg: ConfigDict):
        self.only_first_block = False
        self.max_size = cfg.max_size
        self.patch_size = cfg.patch_size

        assert self.max_size % self.patch_size == 0, "max_size must be divisible by patch_size"
        cu_seqlens = torch.tensor([0, (self.max_size // self.patch_size) ** 2])
        self.register_buffer("cu_seqlens", cu_seqlens, persistent=False)

        self._rot_pos_emb(
            grid_thw=torch.tensor([[1, self.max_size // self.patch_size, self.max_size // self.patch_size]])
        )
        # self.register_buffer("rotary_pos_emb_cache", rotary_pos_emb, persistent=False)

    def _rot_pos_emb(self, grid_thw):
        pos_ids = []
        for t, h, w in grid_thw:
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3)
            hpos_ids = hpos_ids.flatten()

            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3)
            wpos_ids = wpos_ids.flatten()
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        self.pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        self.rotary_pos_emb_full = nn.Embedding(*list(rotary_pos_emb_full.shape))
        self.rotary_pos_emb_full.weight.data = rotary_pos_emb_full
        # rotary_pos_emb = self.rotary_pos_emb_full(self.pos_ids).flatten(1)
        # rotary_pos_emb = rotary_pos_emb_full[self.pos_ids].flatten(1)
        # return rotary_pos_emb

    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        hidden_states = self.patch_embed(hidden_states)
        rotary_pos_emb = self.rotary_pos_emb_full(self.pos_ids.to(hidden_states.device)).flatten(1)
        # rotary_pos_emb = self.rot_pos_emb(grid_thw)
        # cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
        #     dim=0,
        #     # Select dtype based on the following factors:
        #     #  - FA2 requires that cu_seqlens_q must have dtype int32
        #     #  - torch.onnx.export requires that cu_seqlens_q must have same dtype as grid_thw
        #     # See https://github.com/huggingface/transformers/pull/34852 for more information
        #     dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        # )
        # cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
        cu_seqlens = self.cu_seqlens

        for blk in self.blocks:
            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(
                    blk.__call__, hidden_states, cu_seqlens, rotary_pos_emb
                )
            else:
                hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens, rotary_pos_emb=rotary_pos_emb)

        return self.merger(hidden_states)


def register_wrap_cls(hf_model):
    pass
