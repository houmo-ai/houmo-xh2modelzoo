import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeVisionAttention,
    Qwen3OmniMoeVisionBlock,
    Qwen3OmniMoeVisionEncoder,
    Qwen3OmniMoeVisionMLP,
    Qwen3OmniMoeVisionPatchEmbed,
    Qwen3OmniMoeVisionPatchMerger,
    Qwen3OmniMoeVisionRotaryEmbedding,
)

from xhquant import nn as xhnn
from xhquant.api import ConfigDict
from xhquant.utils.registry import DynamicModule

from ...register import XHLLM_TRACEABLE_MODULES


class _Qwen3OmniVisionDynamicModule(DynamicModule):
    """Base wrapper for Qwen3 Omni vision modules."""

    def _setup(self, cfg: Optional[ConfigDict] = None):
        self.wrap_cfg: Optional[ConfigDict] = cfg
        return None


def weight_denorm(conv2d: nn.Conv2d) -> None:
    """Undo CLIP-style input normalization baked into Conv weights."""

    norm = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=conv2d.weight.device).view(1, -1, 1, 1) * 255
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=conv2d.weight.device).view(1, -1, 1, 1) * 255
    conv2d.weight.data = conv2d.weight.data / std
    bias_data = -(norm * conv2d.weight.data).sum(dim=[1, 2, 3])
    conv2d.bias = nn.Parameter(bias_data, requires_grad=False)


def _cfg_get(cfg: Optional[ConfigDict], key: str, default=None):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


@XHLLM_TRACEABLE_MODULES.register_module(
    {
        Qwen3OmniMoeVisionRotaryEmbedding: "Qwen3OmniMoeVisionRotaryEmbedding",
    }
)
class _Qwen3OmniVisionRotaryEmbedding(_Qwen3OmniVisionDynamicModule):
    """Register rotary embedding wrapper for vision encoder."""

    @torch.no_grad()
    def _setup(self, cfg: Optional[ConfigDict] = None):
        self.max_seq_len_cached = int(_cfg_get(cfg, "max_position_embeddings", 4096))
        self._build_cache(self.max_seq_len_cached, self.inv_freq.dtype)

    def _build_cache(self, seq_len: int, dtype: torch.dtype) -> None:
        device = self.inv_freq.device
        seq = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq.to(device=device))
        if hasattr(self, "freqs_cached"):
            self.freqs_cached = freqs.to(dtype=dtype)
        else:
            self.register_buffer("freqs_cached", freqs.to(dtype=dtype), persistent=False)

    def _set_dtype(self, dtype: torch.dtype) -> None:
        self.inv_freq = self.inv_freq.to(dtype=dtype)
        self._build_cache(self.max_seq_len_cached, dtype)

    def forward(self, seqlen: int) -> torch.Tensor:
        if not hasattr(self, "freqs_cached"):
            self._build_cache(max(seqlen, getattr(self, "max_seq_len_cached", seqlen)), self.inv_freq.dtype)

        if seqlen > self.freqs_cached.shape[0]:
            self.max_seq_len_cached = seqlen
            self._build_cache(seqlen, self.freqs_cached.dtype)

        return self.freqs_cached[:seqlen]


@XHLLM_TRACEABLE_MODULES.register_module(
    {
        Qwen3OmniMoeVisionPatchEmbed: "Qwen3OmniMoeVisionPatchEmbed",
    }
)
class _Qwen3OmniMoeVisionPatchEmbed(DynamicModule):
    @torch.no_grad()
    def _setup(self, cfg: ConfigDict):
        # self.only_first_block = cfg.only_first_block
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

        # 将processor的归一化，合并到卷积权重中，避免在推理时增加额外的计算
        weight_denorm(self.proj1)
        weight_denorm(self.proj2)

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
        Qwen3OmniMoeVisionPatchMerger: "Qwen3OmniMoeVisionPatchMerger",
    }
)
class _Qwen3OmniMoeVisionPatchMerger(_Qwen3OmniVisionDynamicModule):
    """Register patch merger wrapper for DeepStack vision blocks."""

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.use_postshuffle_norm:
            hidden = hidden.view(-1, self.hidden_size)

        hidden = self.ln_q(hidden).view(-1, self.hidden_size)

        for layer in self.mlp:
            hidden = layer(hidden)
        return hidden

    def _setup(self, cfg: ConfigDict):
        pass


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


@XHLLM_TRACEABLE_MODULES.register_module(
    {
        Qwen3OmniMoeVisionAttention: "Qwen3OmniMoeVisionAttention",
    }
)
class _Qwen3OmniMoeVisionAttention(_Qwen3OmniVisionDynamicModule):
    """Register attention wrapper for vision encoder."""

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
        Qwen3OmniMoeVisionMLP: "Qwen3OmniMoeVisionMLP",
    }
)
class _Qwen3OmniMoeVisionMLP(_Qwen3OmniVisionDynamicModule):
    """Register MLP wrapper for vision encoder."""

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        hidden_state = self.linear_fc1(hidden_state)
        hidden_state = self.act_fn(hidden_state)
        hidden_state = self.linear_fc2(hidden_state)
        return hidden_state

    def _setup(self, cfg: ConfigDict):
        pass


@XHLLM_TRACEABLE_MODULES.register_module(
    {
        Qwen3OmniMoeVisionBlock: "Qwen3OmniMoeVisionBlock",
    }
)
class _Qwen3OmniMoeVisionBlock(_Qwen3OmniVisionDynamicModule):
    """Register block wrapper for vision encoder."""

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


@XHLLM_TRACEABLE_MODULES.register_module(
    {
        Qwen3OmniMoeVisionEncoder: "Qwen3OmniMoeVisionEncoder",
    }
)
class _Qwen3OmniMoeVisionEncoder(_Qwen3OmniVisionDynamicModule):
    """Register top-level vision encoder wrapper."""

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

        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.patch_embed(hidden_states)
        hidden_states = hidden_states + self.patch_pos_embeds
        cos = self.cos
        sin = self.sin
        position_embeddings = (cos, sin)
        # batch, seq_len, dim = hidden_states.size()
        deepstack_feature_lists = []
        for layer_num, blk in enumerate(self.blocks):
            hidden_states = blk(hidden_states, position_embeddings=position_embeddings)
            if layer_num in self.deepstack_visual_indexes:
                deepstack_feature = self.deepstack_merger_list[self.deepstack_visual_indexes.index(layer_num)](
                    hidden_states
                )
                deepstack_feature_lists.append(deepstack_feature)
        hidden_states = self.merger(hidden_states)
        return hidden_states, deepstack_feature_lists


def register_wrap_modules():
    """Compatibility helper for API parity with thinker registration."""
    return None
