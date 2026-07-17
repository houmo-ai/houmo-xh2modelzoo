from __future__ import annotations

import math
import sys
import types
from copy import deepcopy
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLTextAttention,
    Qwen3VLTextDecoderLayer,
    Qwen3VLTextModel,
    Qwen3VLTextRMSNorm,
    Qwen3VLTextRotaryEmbedding,
    Qwen3VLVisionAttention,
    Qwen3VLVisionBlock,
    Qwen3VLVisionModel,
    Qwen3VLVisionPatchEmbed,
    Qwen3VLVisionPatchMerger,
    rotate_half,
)

from xhquant import nn as xhnn
from xhquant.api import ConfigDict
from xhquant.nn import BfpFlashAttention, LLMCacheV2, MaskedSoftmax, RMSNorm
from xhquant.utils.registry import DynamicModule

from ...builder import XHLLM_TRACEABLE_MODULES


class _LingBotQwen3VLTextRMSNorm(DynamicModule):
    def _setup(self, cfg: ConfigDict | None = None):
        del cfg
        hidden_size = self.weight.shape[0]
        self.norm = RMSNorm(hidden_size, self.variance_epsilon)
        self.norm.weight = nn.Parameter(deepcopy(self.weight.data))

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.norm(hidden_states)


class _LingBotQwen3VLTextRotaryEmbedding(DynamicModule):
    def _setup(self, cfg: ConfigDict):
        self._setup_cos_sin_cache(int(cfg.max_pe_length))
        if hasattr(self, "setup_after_callback"):
            self.setup_after_callback()

    def _setup_cos_sin_cache(self, seq_len: int) -> None:
        cos, sin = self.forward(max_seq_len_cached=seq_len)
        self.register_buffer("cos_cached", cos[0], persistent=False)
        self.register_buffer("sin_cached", sin[0], persistent=False)

    def forward(self, max_seq_len_cached: int = 2048) -> tuple[Tensor, Tensor]:
        inv_freq = self.inv_freq
        if torch.cuda.is_available():
            inv_freq = inv_freq.cuda()
        inv_freq_expanded = inv_freq[None, None, :, None].float().expand(1, max_seq_len_cached, -1, 1)
        positions = torch.arange(max_seq_len_cached, device=inv_freq.device).float()[None, :, None, None]
        freqs = (inv_freq_expanded @ positions).transpose(2, 3)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling
        return cos.to(dtype=self.inv_freq.dtype).to(self.inv_freq.device), sin.to(dtype=self.inv_freq.dtype).to(
            self.inv_freq.device
        )


class _LingBotQwen3VLTextAttention(DynamicModule):
    def _setup(self, cfg: ConfigDict):
        if not hasattr(self, "num_key_value_heads"):
            self.num_key_value_heads = self.config.num_key_value_heads
        if not hasattr(self, "num_heads"):
            self.num_heads = self.config.num_attention_heads

        self.enable_rope = bool(cfg.get("enable_rope", True))
        if self.enable_rope:
            self.rope = xhnn.Rope()
        self.slice_1 = xhnn.Slice([0], [self.head_dim // 2], [3], [1])
        self.slice_2 = xhnn.Slice([self.head_dim // 2], [sys.maxsize], [3], [1])
        self.masked_softmax = MaskedSoftmax(dim=-1)
        self.attn_hidden_dim = self.head_dim * self.num_heads

        flash_cfg = cfg.get("bfp_flash_attention")
        self.use_bfp_flash_attention = bool(flash_cfg and flash_cfg.enable)
        if self.use_bfp_flash_attention:
            self.bfp_attn = BfpFlashAttention(
                self.attn_hidden_dim,
                self.num_heads,
                True,
                flash_cfg.sefp_manbit,
                flash_cfg.out_fp_expbit,
                flash_cfg.out_fp_manbit,
            )

        self.use_cache = bool(cfg.use_cache)
        cache_axis = cfg.kv_cache.cache_axis
        self.k_cache = LLMCacheV2(axis=cache_axis) if self.use_cache else None
        self.v_cache = LLMCacheV2(axis=cache_axis) if self.use_cache else None
        self.register_buffer(
            "kv_scale",
            torch.tensor(1 / math.sqrt(self.head_dim), dtype=torch.float16),
            persistent=False,
        )

    def apply_rotary_pos_emb(self, q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor]:
        if self.enable_rope:
            return self.rope(q, cos, sin), self.rope(k, cos, sin)
        return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)

    def graph_forward(
        self,
        hidden_states: Tensor,
        past_seq_length: Tensor | None = None,
        current_input_length: Tensor | None = None,
        past_k_cache: Tensor | None = None,
        past_v_cache: Tensor | None = None,
        position_embeddings: tuple[Tensor, Tensor] | None = None,
        **kwargs: Any,
    ) -> tuple[Tensor, None, None]:
        del kwargs
        batch_size, query_length, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = self.q_norm(
            query_states.view(batch_size, query_length, self.config.num_attention_heads, self.head_dim)
        ).transpose(1, 2)
        key_states = self.k_norm(
            key_states.view(batch_size, query_length, self.config.num_key_value_heads, self.head_dim)
        ).transpose(1, 2)
        value_states = value_states.view(
            batch_size,
            query_length,
            self.config.num_key_value_heads,
            self.head_dim,
        ).transpose(1, 2)

        if position_embeddings is None:
            raise ValueError("Qwen3-VL text attention requires rotary position embeddings")
        query_states, key_states = self.apply_rotary_pos_emb(query_states, key_states, *position_embeddings)

        if self.use_cache:
            key_states = self.k_cache(key_states, past_seq_length, current_input_length, past_k_cache)
            value_states = self.v_cache(value_states, past_seq_length, current_input_length, past_v_cache)
        if self.use_bfp_flash_attention:
            attention_output = self.bfp_attn(query_states, key_states, value_states)
        else:
            query_states = query_states * self.kv_scale
            key_states = torch.repeat_interleave(key_states.transpose(2, 3), self.num_key_value_groups, dim=1)
            attention_weights = self.masked_softmax(torch.matmul(query_states, key_states), past_seq_length)
            value_states = torch.repeat_interleave(value_states, self.num_key_value_groups, dim=1)
            attention_output = torch.matmul(attention_weights, value_states)
            attention_output = attention_output.transpose(1, 2).reshape(
                batch_size,
                query_length,
                self.config.num_attention_heads * self.head_dim,
            )
        return self.o_proj(attention_output), None, None


class _LingBotQwen3VLTextDecoderLayer(DynamicModule):
    def _setup(self, cfg: ConfigDict):
        del cfg

    def graph_forward(
        self,
        hidden_states: Tensor,
        past_seq_length: Tensor | None = None,
        current_input_length: Tensor | None = None,
        past_k_cache: Tensor | None = None,
        past_v_cache: Tensor | None = None,
        position_embeddings: tuple[Tensor, Tensor] | None = None,
        **kwargs: Any,
    ) -> Tensor:
        del kwargs
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            past_seq_length=past_seq_length,
            current_input_length=current_input_length,
            past_k_cache=past_k_cache,
            past_v_cache=past_v_cache,
            position_embeddings=position_embeddings,
        )[0]
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        return residual + self.mlp(hidden_states)


class _LingBotQwen3VLTextModel(DynamicModule):
    def _setup(self, cfg: ConfigDict):
        self.num_logits_to_keep = int(cfg.num_logits_to_keep)
        if self.num_logits_to_keep not in (0, 1):
            raise ValueError("num_logits_to_keep must be 0 or 1")
        input_sequence_length = int(cfg.input_sequence_length)
        batch_size = int(cfg.get("batch_size", 1))
        self.slice = xhnn.Slice([0], [input_sequence_length], [1], [1])
        self.llm_gather = xhnn.BatchGather(1)
        self.llm_gather.update_offset_indices(batch_size, input_sequence_length)

        def update_gather(module: xhnn.BatchGather, update_cfg: ConfigDict):
            module.update_offset_indices(
                int(update_cfg.get("batch_size", 1)),
                int(update_cfg.input_sequence_length),
            )

        def update_slice(module: xhnn.Slice, update_cfg: ConfigDict):
            module.ends = [int(update_cfg.input_sequence_length)]

        self.llm_gather._update_cfg = types.MethodType(update_gather, self.llm_gather)
        self.slice._update_cfg = types.MethodType(update_slice, self.slice)
        self.use_cache = bool(cfg.use_cache)

        time_mask = torch.ones(64)
        height_ids = torch.arange(1, 60, 3)
        width_ids = torch.arange(2, 60, 3)
        time_mask[height_ids] = 0
        time_mask[width_ids] = 0
        self.register_buffer("time_mask", torch.cat([time_mask, time_mask]).view(1, 1, -1).half(), persistent=False)

        height_mask = torch.zeros(64)
        height_mask[height_ids] = 1
        self.register_buffer(
            "height_mask",
            torch.cat([height_mask, height_mask]).view(1, 1, -1).half(),
            persistent=False,
        )

        width_mask = torch.zeros(64)
        width_mask[width_ids] = 1
        self.register_buffer(
            "width_mask",
            torch.cat([width_mask, width_mask]).view(1, 1, -1).half(),
            persistent=False,
        )

        self.max_layers = 1 if bool(cfg.get("only_first_block", False)) else -1
        if cfg.get("max_layers") is not None:
            self.max_layers = int(cfg.max_layers)

    def graph_forward(
        self,
        input_embeds: Tensor,
        time_position_ids: Tensor,
        height_position_ids: Tensor,
        width_position_ids: Tensor,
        past_seq_length: Tensor,
        current_input_length: Tensor,
        deepstack_visual_embed_0: Tensor,
        deepstack_visual_embed_1: Tensor,
        deepstack_visual_embed_2: Tensor,
        past_key_cache: list[Tensor] | None = None,
        past_value_cache: list[Tensor] | None = None,
    ) -> BaseModelOutputWithPast:
        cos = self.rotary_emb.cos_cached
        sin = self.rotary_emb.sin_cached
        rotary_cos = (
            (
                cos[time_position_ids] * self.time_mask
                + cos[height_position_ids] * self.height_mask
                + cos[width_position_ids] * self.width_mask
            )
            .squeeze(1)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        rotary_sin = (
            (
                sin[time_position_ids] * self.time_mask
                + sin[height_position_ids] * self.height_mask
                + sin[width_position_ids] * self.width_mask
            )
            .squeeze(1)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        position_embeddings = (rotary_cos, rotary_sin)

        hidden_states = input_embeds
        visual_embeds = (
            deepstack_visual_embed_0,
            deepstack_visual_embed_1,
            deepstack_visual_embed_2,
        )
        for layer_index, decoder_layer in enumerate(self.layers):
            key_cache = past_key_cache[layer_index] if self.use_cache else None
            value_cache = past_value_cache[layer_index] if self.use_cache else None
            hidden_states = decoder_layer(
                hidden_states,
                past_seq_length=past_seq_length,
                current_input_length=current_input_length,
                past_k_cache=key_cache,
                past_v_cache=value_cache,
                position_embeddings=position_embeddings,
            )
            if layer_index < len(visual_embeds):
                hidden_states = hidden_states + visual_embeds[layer_index]
            if self.max_layers > 0 and layer_index + 1 >= self.max_layers:
                break

        hidden_states = (
            self.slice(hidden_states)
            if self.num_logits_to_keep == 0
            else self.llm_gather(hidden_states, current_input_length - 1)
        )
        return BaseModelOutputWithPast(last_hidden_state=self.norm(hidden_states))


class _LingBotQwen3VLVisionAttention(DynamicModule):
    def _setup(self, cfg: ConfigDict):
        self.enable_rope = bool(cfg.get("enable_rope", False))
        if self.enable_rope:
            self.rope = xhnn.Rope()
        head_dim = self.qkv.out_features // 3 // self.num_heads
        self.kv_scale = 1 / math.sqrt(head_dim)

        weight = self.qkv.weight.data.clone()
        bias = self.qkv.bias.data.clone()
        hidden_size = weight.shape[0] // 3
        weight = weight.permute(1, 0).reshape(hidden_size, 3, hidden_size).permute(1, 2, 0)
        bias = bias.reshape(3, hidden_size)
        projection_kwargs = {"device": weight.device, "dtype": weight.dtype}
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=True, **projection_kwargs)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=True, **projection_kwargs)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=True, **projection_kwargs)
        for projection, projection_weight, projection_bias in zip(
            (self.q_proj, self.k_proj, self.v_proj),
            weight,
            bias,
            strict=True,
        ):
            projection.weight.data.copy_(projection_weight)
            projection.bias.data.copy_(projection_bias)

        if head_dim % 64:
            self._pad_attention_head(head_dim)
        del self.qkv

    def _pad_attention_head(self, head_dim: int) -> None:
        padded_head_dim = math.ceil(head_dim / 64) * 64
        padding_size = padded_head_dim - head_dim
        half_padding = padding_size // 2
        input_features = self.q_proj.in_features
        device = self.q_proj.weight.device
        dtype = self.q_proj.weight.dtype

        def expand_qk(linear: nn.Linear) -> nn.Linear:
            weight = linear.weight.data.transpose(0, 1).reshape(input_features, self.num_heads, 2, head_dim // 2)
            padded_weight = torch.zeros(
                input_features,
                self.num_heads,
                2,
                head_dim // 2 + half_padding,
                device=device,
                dtype=dtype,
            )
            padded_weight[..., : head_dim // 2] = weight
            bias = linear.bias.data.reshape(self.num_heads, 2, head_dim // 2)
            padded_bias = torch.zeros(
                self.num_heads,
                2,
                head_dim // 2 + half_padding,
                device=device,
                dtype=dtype,
            )
            padded_bias[..., : head_dim // 2] = bias
            expanded = nn.Linear(
                input_features,
                self.num_heads * padded_head_dim,
                bias=True,
                device=device,
                dtype=dtype,
            )
            expanded.weight.data.copy_(padded_weight.reshape(input_features, -1).transpose(0, 1))
            expanded.bias.data.copy_(padded_bias.reshape(-1))
            return expanded

        self.q_proj = expand_qk(self.q_proj)
        self.k_proj = expand_qk(self.k_proj)

        value_weight = self.v_proj.weight.data.transpose(0, 1).reshape(input_features, self.num_heads, head_dim)
        padded_value_weight = torch.zeros(
            input_features,
            self.num_heads,
            padded_head_dim,
            device=device,
            dtype=dtype,
        )
        padded_value_weight[..., :head_dim] = value_weight
        value_bias = self.v_proj.bias.data.reshape(self.num_heads, head_dim)
        padded_value_bias = torch.zeros(self.num_heads, padded_head_dim, device=device, dtype=dtype)
        padded_value_bias[..., :head_dim] = value_bias
        expanded_value = nn.Linear(
            input_features,
            self.num_heads * padded_head_dim,
            bias=True,
            device=device,
            dtype=dtype,
        )
        expanded_value.weight.data.copy_(padded_value_weight.reshape(input_features, -1).transpose(0, 1))
        expanded_value.bias.data.copy_(padded_value_bias.reshape(-1))
        self.v_proj = expanded_value

        projected_weight = torch.zeros(
            self.proj.out_features,
            self.num_heads,
            padded_head_dim,
            device=device,
            dtype=dtype,
        )
        projected_weight[..., :head_dim] = self.proj.weight.data.reshape(
            self.proj.out_features,
            self.num_heads,
            head_dim,
        )
        expanded_projection = nn.Linear(
            self.num_heads * padded_head_dim,
            self.proj.out_features,
            bias=True,
            device=device,
            dtype=dtype,
        )
        expanded_projection.weight.data.copy_(projected_weight.reshape(self.proj.out_features, -1))
        expanded_projection.bias.data.copy_(self.proj.bias.data)
        self.proj = expanded_projection

    def forward(self, hidden_states: Tensor, position_embeddings: tuple[Tensor, Tensor]) -> Tensor:
        batch_size, sequence_length, _ = hidden_states.shape
        query = self.q_proj(hidden_states).reshape(batch_size, sequence_length, self.num_heads, -1)
        key = self.k_proj(hidden_states).reshape(batch_size, sequence_length, self.num_heads, -1)
        value = self.v_proj(hidden_states).reshape(batch_size, sequence_length, self.num_heads, -1)
        cos, sin = position_embeddings
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
        if self.enable_rope:
            query, key = self.rope(query, cos, sin), self.rope(key, cos, sin)
        else:
            query = (query * cos) + (rotate_half(query) * sin)
            key = (key * cos) + (rotate_half(key) * sin)

        query = query.transpose(1, 2) * self.kv_scale
        key = key.transpose(1, 2).transpose(-2, -1)
        value = value.transpose(1, 2)
        attention = F.softmax(torch.matmul(query, key), dim=-1, dtype=torch.float32).to(query.dtype)
        output = torch.matmul(attention, value).transpose(1, 2).reshape(batch_size, sequence_length, -1)
        return self.proj(output)


class _LingBotQwen3VLVisionBlock(DynamicModule):
    def _setup(self, cfg: ConfigDict):
        del cfg

    def forward(self, hidden_states: Tensor, position_embeddings: tuple[Tensor, Tensor]) -> Tensor:
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states), position_embeddings=position_embeddings)
        return hidden_states + self.mlp(self.norm2(hidden_states))


def _fold_image_normalization(conv2d: nn.Conv2d) -> None:
    mean = torch.tensor([0.5, 0.5, 0.5], device=conv2d.weight.device).view(1, -1, 1, 1) * 255
    std = torch.tensor([0.5, 0.5, 0.5], device=conv2d.weight.device).view(1, -1, 1, 1) * 255
    conv2d.weight.data = conv2d.weight.data / std
    conv2d.bias = nn.Parameter(-(mean * conv2d.weight.data).sum(dim=(1, 2, 3)))


class _LingBotQwen3VLVisionPatchEmbed(DynamicModule):
    @torch.no_grad()
    def _setup(self, cfg: ConfigDict):
        patch_size = int(cfg.patch_size)
        self.spatial_merge_size = int(cfg.spatial_merge_size)
        device = self.proj.weight.device
        dtype = self.proj.weight.dtype
        projection_bias = self.proj.bias.data.clone() if self.proj.bias is not None else None
        self.proj1 = nn.Conv2d(
            self.in_channels,
            self.embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.proj2 = nn.Conv2d(
            self.in_channels,
            self.embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.proj1.weight.data.copy_(self.proj.weight[:, :, 0].contiguous())
        self.proj2.weight.data.copy_(self.proj.weight[:, :, 1].contiguous())
        _fold_image_normalization(self.proj1)
        _fold_image_normalization(self.proj2)
        if projection_bias is not None:
            self.proj2.bias.data.add_(projection_bias)
        del self.proj
        if self.temporal_patch_size != 2:
            raise ValueError("Qwen3-VL visual export requires temporal_patch_size=2")

    def forward(self, hidden_states: Tensor) -> Tensor:
        batch_size, channels, frames, height, width = hidden_states.shape
        temporal_patches = frames // self.temporal_patch_size
        hidden_states = hidden_states.reshape(
            batch_size,
            channels,
            temporal_patches,
            self.temporal_patch_size,
            height,
            width,
        )
        hidden_states = hidden_states.permute(0, 2, 1, 3, 4, 5).reshape(
            batch_size * temporal_patches,
            channels,
            self.temporal_patch_size,
            height,
            width,
        )
        first_frame, second_frame = hidden_states.unbind(2)
        hidden_states = self.proj1(first_frame) + self.proj2(second_frame)
        _, output_channels, output_height, output_width = hidden_states.shape
        merge_size = self.spatial_merge_size
        if output_height % merge_size or output_width % merge_size:
            raise ValueError("Visual patch grid must be divisible by spatial_merge_size")
        hidden_states = hidden_states.reshape(
            batch_size,
            temporal_patches,
            output_channels,
            output_height // merge_size,
            merge_size,
            output_width // merge_size,
            merge_size,
        )
        return hidden_states.permute(0, 1, 3, 5, 4, 6, 2).reshape(batch_size, -1, output_channels)


class _LingBotQwen3VLVisionPatchMerger(DynamicModule):
    def _setup(self, cfg: ConfigDict):
        del cfg

    def forward(self, hidden_states: Tensor) -> Tensor:
        batch_size = hidden_states.shape[0]
        if self.use_postshuffle_norm:
            hidden_states = hidden_states.view(batch_size, -1, self.hidden_size)
        hidden_states = self.norm(hidden_states).view(batch_size, -1, self.hidden_size)
        return self.linear_fc2(self.act_fn(self.linear_fc1(hidden_states)))


class _LingBotQwen3VLVisionModel(DynamicModule):
    @torch.no_grad()
    def _setup(self, cfg: ConfigDict):
        self.patch_size = int(cfg.patch_size)
        self.max_size_w = int(cfg.max_size_w)
        self.max_size_h = int(cfg.max_size_h)
        self.max_size_t = int(cfg.max_size_t)
        self.temporal_patch_size = int(cfg.temporal_patch_size)
        if self.max_size_w % self.patch_size or self.max_size_h % self.patch_size:
            raise ValueError("Visual width and height must be divisible by patch_size")

        grid_width = self.max_size_w // self.patch_size
        grid_height = self.max_size_h // self.patch_size
        grid_time = self.max_size_t // self.temporal_patch_size
        sequence_length = grid_time * grid_height * grid_width
        device = next(self.parameters()).device
        grid_thw = torch.tensor([[grid_time, grid_height, grid_width]], device=device)
        self.register_buffer(
            "patch_pos_embeds",
            self.fast_pos_embed_interpolate(grid_thw).unsqueeze(0),
            persistent=False,
        )
        rotary = self.rot_pos_emb(grid_thw).reshape(sequence_length, -1)
        embedding = torch.cat((rotary, rotary), dim=-1)
        cos = embedding.cos().unsqueeze(-2)
        sin = embedding.sin().unsqueeze(-2)
        if embedding.shape[-1] % 64:
            half_padding = (64 - embedding.shape[-1] % 64) // 2
            cos_first, cos_second = cos.chunk(2, dim=-1)
            sin_first, sin_second = sin.chunk(2, dim=-1)
            cos = torch.cat((F.pad(cos_first, (0, half_padding)), F.pad(cos_second, (0, half_padding))), dim=-1)
            sin = torch.cat((F.pad(sin_first, (0, half_padding)), F.pad(sin_second, (0, half_padding))), dim=-1)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, pixel_values: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        hidden_states = self.patch_embed(pixel_values) + self.patch_pos_embeds
        deepstack_features = []
        for layer_index, block in enumerate(self.blocks):
            hidden_states = block(hidden_states, position_embeddings=(self.cos, self.sin))
            if layer_index in self.deepstack_visual_indexes:
                merger_index = self.deepstack_visual_indexes.index(layer_index)
                deepstack_features.append(self.deepstack_merger_list[merger_index](hidden_states))
        image_embeds = self.merger(hidden_states)
        if len(deepstack_features) != 3:
            raise RuntimeError(f"Qwen3-VL visual graph requires 3 deepstack outputs, got {len(deepstack_features)}")
        return image_embeds, deepstack_features[0], deepstack_features[1], deepstack_features[2]


_WRAPPERS = {
    Qwen3VLTextRMSNorm: _LingBotQwen3VLTextRMSNorm,
    Qwen3VLTextRotaryEmbedding: _LingBotQwen3VLTextRotaryEmbedding,
    Qwen3VLTextAttention: _LingBotQwen3VLTextAttention,
    Qwen3VLTextDecoderLayer: _LingBotQwen3VLTextDecoderLayer,
    Qwen3VLTextModel: _LingBotQwen3VLTextModel,
    Qwen3VLVisionAttention: _LingBotQwen3VLVisionAttention,
    Qwen3VLVisionBlock: _LingBotQwen3VLVisionBlock,
    Qwen3VLVisionPatchEmbed: _LingBotQwen3VLVisionPatchEmbed,
    Qwen3VLVisionPatchMerger: _LingBotQwen3VLVisionPatchMerger,
    Qwen3VLVisionModel: _LingBotQwen3VLVisionModel,
}


def register_qwen3_vl_wrappers() -> None:
    for source_class, wrapper_class in _WRAPPERS.items():
        if source_class not in XHLLM_TRACEABLE_MODULES:
            XHLLM_TRACEABLE_MODULES.register_module(
                {source_class: source_class.__name__},
                wrapper_class,
            )


def activate_qwen3_vl_language_graph(model: nn.Module) -> nn.Module:
    for module in model.modules():
        graph_forward = getattr(module, "graph_forward", None)
        if graph_forward is not None:
            module.forward = graph_forward
    return model
