import math
from typing import Optional
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from xhquant import nn as xhnn
from xhquant.api import ConfigDict
from xhquant.utils.registry import DynamicModule

from ..builder import XHLLM_TRACEABLE_MODULES
from .modeling_qwen2_5_vl import (
    Qwen2_5_VisionPatchEmbed,  # apply_rotary_pos_emb_vision,; PatchEmbed,; VisionSdpaAttention,
)
from .modeling_qwen2_5_vl import Qwen2_5_VisionTransformerPretrainedModel
from .modeling_qwen2_5_vl import Qwen2_5_VLVisionAttention
from .modeling_qwen2_5_vl import Qwen2_5_VLVisionBlock
from .modeling_qwen2_5_vl import Qwen2_5_VLVisionSdpaAttention
from .modeling_qwen2_5_vl import rotate_half


@XHLLM_TRACEABLE_MODULES.register_module(
    {
        Qwen2_5_VLVisionAttention: "Qwen2_5_VLVisionAttention",
        Qwen2_5_VLVisionSdpaAttention: "Qwen2_5_VLVisionSdpaAttention",
    }
)
class _Qwen2_5_VLVisionAttention(DynamicModule):
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
        # assert head_dim == 80  # 80

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
        self.masked_softmax = xhnn.MaskedSoftmax(-1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        # q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        q = self.q_proj(hidden_states).reshape(seq_length, self.num_heads, -1)
        k = self.k_proj(hidden_states).reshape(seq_length, self.num_heads, -1)
        v = self.v_proj(hidden_states).reshape(seq_length, self.num_heads, -1)
        cos, sin = position_embeddings
        q, k = self.apply_rotary_pos_emb(q, k, cos, sin)
        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)

        # attention_mask = torch.zeros([1, seq_length, seq_length], device=q.device, dtype=torch.bool)
        # for i in range(1, len(cu_seqlens)):
        #     attention_mask[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = True
        k = k.transpose(-2, -1)
        q = q * self.kv_scale
        dtype = q.dtype
        attn_weights = torch.matmul(q, k).to(dtype)
        if attention_mask is not None:
            attn_weights += attention_mask
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(dtype)
        attn_output = torch.matmul(attn_weights, v)

        attn_output = attn_output.transpose(0, 1)
        attn_output = attn_output.reshape(seq_length, -1)
        attn_output = self.proj(attn_output)
        return attn_output


@XHLLM_TRACEABLE_MODULES.register_module(
    {
        Qwen2_5_VLVisionBlock: "Qwen2_5_VLVisionBlock",
    }
)
class _Qwen2_5_VLVisionBlock(DynamicModule):
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


def weight_denorm(
    conv2d: nn.Conv2d,
):
    norm = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, -1, 1, 1).to(conv2d.weight.device) * 255
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, -1, 1, 1).to(conv2d.weight.device) * 255
    conv2d.weight.data = conv2d.weight.data / std
    bias_data = -(norm * conv2d.weight.data).sum(dim=[1, 2, 3])
    conv2d.bias = nn.Parameter(bias_data)


@XHLLM_TRACEABLE_MODULES.register_module(
    {
        Qwen2_5_VisionPatchEmbed: "Qwen2_5_VisionPatchEmbed",
    }
)
class _Qwen2_5_VisionPatchEmbed(DynamicModule):
    @torch.no_grad
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

        weight_denorm(self.proj1)
        weight_denorm(self.proj2)

        self.proj1.to(self.proj.weight.device, dtype=self.proj.weight.dtype)
        self.proj2.to(self.proj.weight.device, dtype=self.proj.weight.dtype)

        # self.proj1.weight.data = self.proj.weight[:, :, 0, :, :].contiguous()
        # self.proj2.weight.data = self.proj.weight[:, :, 1, :, :].contiguous()
        del self.proj
        assert self.temporal_patch_size == 2, "temporal_patch_size must be 2"

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        y1 = self.proj1(hidden_states)
        y2 = self.proj2(hidden_states)
        hidden_states = y1 + y2

        b, c, h, w = hidden_states.shape
        hidden_states = hidden_states.reshape(b, c, h // 2, 2, w // 2, 2)
        hidden_states = hidden_states.permute(0, 2, 4, 3, 5, 1)
        hidden_states = hidden_states.reshape(-1, c)
        return hidden_states


@XHLLM_TRACEABLE_MODULES.register_module(
    {
        Qwen2_5_VisionTransformerPretrainedModel: "Qwen2_5_VisionTransformerPretrainedModel",
    }
)
class Qwen2_5_VisionTransformerPretrainedModel(DynamicModule):
    @torch.no_grad()
    def _setup(self, cfg: ConfigDict):
        self.only_first_block = False
        self.patch_size = cfg.patch_size
        device = next(self.parameters()).device
        self.max_size_w = cfg.max_size_w
        self.max_size_h = cfg.max_size_h
        grid_size_w = self.max_size_w // self.patch_size
        grid_size_h = self.max_size_h // self.patch_size
        seq_length = grid_size_w * grid_size_h

        assert self.max_size_w % self.patch_size == 0, "max_size_w must be divisible by patch_size"
        assert self.max_size_h % self.patch_size == 0, "max_size_h must be divisible by patch_size"
        cu_seqlens = torch.tensor([0, seq_length]).to(device)
        self.register_buffer("cu_seqlens", cu_seqlens, persistent=False)

        grid_thw = torch.tensor([[1, self.max_size_h // self.patch_size, self.max_size_w // self.patch_size]]).to(
            device
        )
        self._rot_pos_emb(grid_thw)

        window_index, cu_window_seqlens = self.get_window_index(grid_thw)
        cu_window_seqlens = torch.tensor(cu_window_seqlens, dtype=torch.int32, device=device)
        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)
        reverse_indices = torch.argsort(window_index)

        self.register_buffer("cu_window_seqlens", cu_window_seqlens)
        self.register_buffer("window_index", window_index)
        self.register_buffer("reverse_indices", reverse_indices)

        rotary_pos_emb = self.rotary_pos_emb_full(self.pos_ids).flatten(1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_length // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        rotary_pos_emb = rotary_pos_emb[self.window_index, :, :]
        rotary_pos_emb = rotary_pos_emb.reshape(seq_length, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        cos = emb.cos().unsqueeze(-2)
        sin = emb.sin().unsqueeze(-2)

        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        attention_mask = torch.zeros([1, seq_length, seq_length], dtype=torch.bool)
        cu_seqlens = self.cu_window_seqlens
        for i in range(1, len(cu_seqlens)):
            attention_mask[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = True
        attention_mask = attention_mask.to(device)
        attention_bias = torch.zeros(1, seq_length, seq_length, dtype=torch.float16, device=device)
        attention_bias.masked_fill_(attention_mask.logical_not(), -torch.finfo(torch.float16).max)
        self.register_buffer("attention_bias", attention_bias, persistent=False)

    @torch.no_grad
    def get_dynamic_buffer(self, grid_thw):
        res = {}
        self.only_first_block = False

        device = next(self.parameters()).device
        seq_length = torch.sum(grid_thw.prod(dim=1)).item()
        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0, dtype=torch.int32
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
        # cu_seqlens = torch.tensor([0, seq_length]).to(device)
        # # res["cu_seqlens"] = cu_seqlens
        # self.cu_seqlens = cu_seqlens

        self._rot_pos_emb(grid_thw)
        window_index, cu_window_seqlens = self.get_window_index(grid_thw)
        cu_window_seqlens = torch.tensor(cu_window_seqlens, dtype=torch.int32, device=device)
        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)
        reverse_indices = torch.argsort(window_index)

        self.cu_window_seqlens = cu_window_seqlens
        self.window_index = window_index
        self.reverse_indices = reverse_indices
        res["window_index"] = window_index
        res["reverse_indices"] = reverse_indices
        # res["cu_window_seqlens"] = cu_window_seqlens

        rotary_pos_emb = self.rotary_pos_emb_full(self.pos_ids).flatten(1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_length // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        rotary_pos_emb = rotary_pos_emb[self.window_index, :, :]
        rotary_pos_emb = rotary_pos_emb.reshape(seq_length, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        cos = emb.cos().unsqueeze(-2)
        sin = emb.sin().unsqueeze(-2)
        self.cos = cos
        self.sin = sin
        res["cos"] = cos
        res["sin"] = sin
        # self.register_buffer("cos", cos, persistent=False)
        # self.register_buffer("sin", sin, persistent=False)

        for layer_num, blk in enumerate(self.blocks):
            if layer_num in self.fullatt_block_indexes:
                cu_seqlens = self.cu_seqlens
            else:
                cu_seqlens = self.cu_window_seqlens

            # attention_mask = torch.zeros(
            #     [1, seq_length, seq_length], dtype=torch.bool)

            # for i in range(1, len(cu_seqlens)):
            #     attention_mask[..., cu_seqlens[i - 1]: cu_seqlens[i],
            #                    cu_seqlens[i - 1]: cu_seqlens[i]] = True
            # attention_mask = attention_mask.to(device)
            # attention_bias = torch.zeros(
            #     1, seq_length, seq_length, dtype=torch.float16, device=device)
            # attention_bias.masked_fill_(
            #     attention_mask.logical_not(), -torch.finfo(torch.float16).max)
            # blk.attn.attention_bias = attention_bias
            # del attention_mask
            torch.cuda.empty_cache()
            # key = "blocks.%d.attn.attention_bias" % layer_num
            # res[key] = attention_bias
        return res

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
        self.pos_ids = torch.cat(pos_ids, dim=0).to(grid_thw.device)
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        self.rotary_pos_emb_full = nn.Embedding(*list(rotary_pos_emb_full.shape))
        self.rotary_pos_emb_full.weight.data = rotary_pos_emb_full

    def forward(
        self,
        hidden_states: torch.Tensor,
        window_index: torch.Tensor,
        window_mask: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.patch_embed(hidden_states)
        cos = self.cos
        sin = self.sin
        position_embeddings = (cos, sin)
        seq_len, dim = hidden_states.size()
        hidden_states = hidden_states.reshape(-1, self.spatial_merge_unit, dim)
        hidden_states = hidden_states[window_index, :, :]
        hidden_states = hidden_states.reshape(seq_len, -1)

        for layer_num, blk in enumerate(self.blocks):
            if layer_num in self.fullatt_block_indexes:
                hidden_states = blk(hidden_states, position_embeddings=position_embeddings)
            else:
                hidden_states = blk(hidden_states, position_embeddings=position_embeddings, attention_mask=window_mask)

        hidden_states = self.merger(hidden_states)
        reverse_indices = torch.argsort(window_index)
        hidden_states = hidden_states[reverse_indices, :]
        return hidden_states


def register_wrap_cls(hf_model):
    pass
