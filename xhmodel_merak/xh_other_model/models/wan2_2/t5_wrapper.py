# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: I001

# pyright: reportMissingImports=false

from copy import deepcopy
import logging
from typing import Sequence

import accelerate
import torch
import torch.nn as nn

from xhquant import nn as xhnn
from xhquant.utils.registry.dynamic_module import DynamicModule

from .common import ensure_wan2_2_repo

ensure_wan2_2_repo()
from wan.modules.t5 import (  # noqa: E402
    T5Attention,
    T5CrossAttention,
    T5LayerNorm,
    T5RelativeEmbedding,
    T5SelfAttention,
)

def fp16_clamp(x):
    # if x.dtype == torch.float16 and torch.isinf(x).any():
    clamp = torch.finfo(x.dtype).max - 1000
    x = torch.clamp(x, min=-clamp, max=clamp)
    return x


class _WanT5LayerNorm(DynamicModule):
    def forward(self, x):
        return self.norm(x)

    def _setup(self, *args, **kwargs):
        del args, kwargs
        hidden_size = self.weight.shape[0]
        eps = getattr(self, "eps", 1e-6)
        self.norm = xhnn.RMSNorm(hidden_size, eps)
        self.norm.weight = nn.Parameter(deepcopy(self.weight.data))
        return self


class _WanT5RelativeEmbedding(DynamicModule):
    def _setup(self, *args, **kwargs):
        del args, kwargs
        return self

    def _relative_position_bucket(self, rel_pos):
        if self.bidirectional:
            num_buckets = self.num_buckets // 2
            rel_buckets = (rel_pos > 0).long() * num_buckets
            rel_pos = torch.abs(rel_pos)
        else:
            num_buckets = self.num_buckets
            rel_buckets = 0
            rel_pos = -torch.min(rel_pos, torch.zeros_like(rel_pos))

        max_exact = num_buckets // 2
        safe_rel_pos = torch.clamp(rel_pos, min=1)
        rel_pos_large = (
            max_exact
            + (
                torch.log(safe_rel_pos / max_exact)
                / torch.log(torch.tensor(self.max_dist / max_exact, device=rel_pos.device, dtype=rel_pos.dtype))
                * (num_buckets - max_exact)
            ).long()
        )
        rel_pos_large = torch.min(rel_pos_large, torch.full_like(rel_pos_large, num_buckets - 1))
        rel_buckets += torch.where(rel_pos < max_exact, rel_pos, rel_pos_large)
        return rel_buckets

    def forward(self, lq, lk):
        device = self.embedding.weight.device
        q_ids = torch.arange(0, lq, device=device, dtype=torch.long)
        k_ids = torch.arange(0, lk, device=device, dtype=torch.long)
        rel_pos = k_ids.unsqueeze(0) - q_ids.unsqueeze(1)
        rel_pos = self._relative_position_bucket(rel_pos)
        rel_pos_embeds = self.embedding(rel_pos)
        rel_pos_embeds = rel_pos_embeds.permute(2, 0, 1).unsqueeze(0)
        return rel_pos_embeds.contiguous()


class _WanT5Attention(DynamicModule):
    def _setup(self, *args, **kwargs):
        del args, kwargs
        self.maskedadd = xhnn.MaskedAdd()
        return self

    def forward(self, x, context=None, attn_bias=None, pos_bias=None):
        context = x if context is None else context
        b, n, c = x.size(0), self.num_heads, self.head_dim

        q = self.q(x).reshape(b, -1, n, c).permute(0, 2, 1, 3)
        k = self.k(context).reshape(b, -1, n, c).permute(0, 2, 1, 3)
        v = self.v(context).reshape(b, -1, n, c).permute(0, 2, 1, 3)

        attn = torch.matmul(q, k.transpose(-1, -2))
        attn = attn + pos_bias
        if attn_bias is not None:
            # attn = attn + attn_bias
            attn = self.maskedadd(attn, attn_bias)

        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).reshape(b, -1, n * c)
        out = self.o(out)
        return out


class _WanT5SelfAttention(DynamicModule):
    def _setup(self, *args, **kwargs):
        del args, kwargs
        return self

    def forward(self, x, attn_bias=None, pos_bias=None):
        # bias = pos_bias if attn_bias is None else attn_bias + pos_bias
        # if torch.any(torch.isnan(bias)):
        #     raise ValueError("bias contains NaN values")
        x = fp16_clamp(x + self.attn(self.norm1(x), attn_bias=attn_bias, pos_bias=pos_bias))
        x = fp16_clamp(x + self.ffn(self.norm2(x)))
        return x


class _WanT5CrossAttention(DynamicModule):
    def _setup(self, *args, **kwargs):
        del args, kwargs
        return self

    def forward(
        self,
        x,
        self_attn_bias=None,
        encoder_states=None,
        encoder_attn_bias=None,
        pos_bias=None,
    ):
        self_bias = pos_bias if self_attn_bias is None else self_attn_bias + pos_bias
        if torch.any(torch.isnan(self_bias)):
            raise ValueError("self_bias contains NaN values")
        x = x + self.self_attn(self.norm1(x), attn_bias=self_bias)
        x = x + self.cross_attn(
            self.norm2(x),
            context=encoder_states,
            attn_bias=encoder_attn_bias,
        )
        x = x + self.ffn(self.norm3(x))
        return x


class _WanT5Blocks(nn.Module):
    def __init__(self, blocks: nn.ModuleList):
        super().__init__()
        self.blocks = blocks

    def forward(self, x: torch.Tensor, attn_bias: torch.Tensor, pos_biases: Sequence[torch.Tensor | None]):
        for block, pos_bias in zip(self.blocks, pos_biases, strict=True):
            x = block(x, attn_bias=attn_bias, pos_bias=pos_bias)
        return x


def wrap_t5_encoder_model(text_encoder_model: nn.Module) -> nn.Module:
    text_encoder_model = accelerate.hooks.remove_hook_from_module(text_encoder_model, recurse=True)
    wrapped_classes = {
        T5Attention: _WanT5Attention,
        T5CrossAttention: _WanT5CrossAttention,
        T5LayerNorm: _WanT5LayerNorm,
        T5RelativeEmbedding: _WanT5RelativeEmbedding,
        T5SelfAttention: _WanT5SelfAttention,
    }
    for _, module in list(text_encoder_model.named_modules()):
        dynamic_cls = wrapped_classes.get(type(module))
        if dynamic_cls is not None and not isinstance(module, DynamicModule):
            dynamic_cls.convert(module)
    return text_encoder_model


class Wan22T5EncoderExportWrapper(nn.Module):
    """Wrap Wan2.2 T5 encoder for export.

    The wrapped runtime keeps the original tokenization path outside and only
    exposes a stable `forward(inputs_embeds, mask_bias)` for export.
    """

    def __init__(self, text_encoder_model: nn.Module, max_sequence_length: int):
        super().__init__()
        self.text_encoder_model = wrap_t5_encoder_model(text_encoder_model)
        self.max_sequence_length = max_sequence_length

        blocks = list(self.text_encoder_model.blocks)
        self.blocks = _WanT5Blocks(self.text_encoder_model.blocks)
        self._cached_pos_bias_names: list[str] = []

        if self.text_encoder_model.shared_pos:
            pos_bias = self.text_encoder_model.pos_embedding(
                max_sequence_length, max_sequence_length
            ).detach()
            self.register_buffer("cached_pos_bias_shared", pos_bias, persistent=False)
            self._cached_pos_bias_names = ["cached_pos_bias_shared"] * len(blocks)
        else:
            for idx, block in enumerate(blocks):
                pos_bias = block.pos_embedding(max_sequence_length, max_sequence_length).detach()
                buffer_name = f"cached_pos_bias_{idx}"
                self.register_buffer(buffer_name, pos_bias, persistent=False)
                self._cached_pos_bias_names.append(buffer_name)

    def forward(self, inputs_embeds: torch.Tensor, mask_bias: torch.Tensor) -> torch.Tensor:
        x = inputs_embeds

        pos_biases = [getattr(self, buffer_name) for buffer_name in self._cached_pos_bias_names]
        x = self.blocks(x, mask_bias, pos_biases)

        x = self.text_encoder_model.norm(x)
        return x

    def encode_texts(self, tokenizer, texts: Sequence[str], device: torch.device):
        ids, mask = tokenizer(texts, return_mask=True, add_special_tokens=True)
        inputs_embeds = self.text_encoder_model.token_embedding(ids.long().to(device))
        mask = mask.to(device)
        mask_4d = mask.view(mask.shape[0], 1, 1, -1)
        mask_bias = torch.zeros(
            mask_4d.shape,
            device=device,
            dtype=inputs_embeds.dtype,
        ).masked_fill(mask_4d == 0, -65504.0)
        original_pos_biases = {
            buffer_name: getattr(self, buffer_name) for buffer_name in set(self._cached_pos_bias_names)
        }

        for buffer_name, pos_bias in original_pos_biases.items():
            setattr(self, buffer_name, pos_bias.masked_fill(mask_4d == 0, 0))
        return self.forward(inputs_embeds, mask_bias)




Wan2_2T5EncoderExportWrapper = Wan22T5EncoderExportWrapper
