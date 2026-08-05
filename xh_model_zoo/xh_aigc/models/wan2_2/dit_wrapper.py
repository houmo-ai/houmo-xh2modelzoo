# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: I001

# pyright: reportMissingImports=false

import math
from copy import deepcopy
from typing import Optional

import torch
import torch.nn as nn

from xhquant import nn as xhnn
from xhquant.utils.registry.dynamic_module import DynamicModule

from .common import ensure_wan2_2_repo

ensure_wan2_2_repo()
from wan.modules.model import (  # noqa: E402
    Head,
    WanModel,
    WanAttentionBlock,
    WanCrossAttention,
    WanLayerNorm,
    WanRMSNorm,
    WanSelfAttention,
    sinusoidal_embedding_1d,
)


FLASH_ATTN_V_SCALE = 16.0


def build_wan_time_embeddings(model: nn.Module, t: torch.Tensor, seq_len: int, output_dtype: torch.dtype):
    if t.dim() == 1:
        t = t.expand(t.size(0), seq_len)

    bt = t.size(0)
    t = t.flatten()
    model.time_embedding = model.time_embedding.to(torch.float32)
    model.time_projection = model.time_projection.to(torch.float32)
    e = model.time_embedding(sinusoidal_embedding_1d(model.freq_dim, t).unflatten(0, (bt, seq_len)).float())
    e0 = model.time_projection(e).unflatten(2, (6, model.dim))
    if output_dtype == torch.float16:
        e = e.to(torch.float16)
        e0 = e0.to(torch.float16)
    return e, e0


class _Wan3DRope(nn.Module):
    def __init__(self):
        super().__init__()
        self.rope = xhnn.Rope()

    def forward(
        self,
        x: torch.Tensor,
        grid_sizes,
        freqs_real: torch.Tensor,
        freqs_imag: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = x.size(0)
        if len(grid_sizes) != batch_size:
            raise ValueError(f"Batch mismatch: x has {batch_size}, grid_sizes has {len(grid_sizes)}")
        num_heads = x.size(2)
        head_dim = x.size(3)
        half_dim = head_dim // 2

        freq_f_dim = half_dim - 2 * (half_dim // 3)
        freq_h_dim = half_dim // 3
        freq_w_dim = half_dim // 3
        freqs_real_f, freqs_real_h, freqs_real_w = freqs_real.split([freq_f_dim, freq_h_dim, freq_w_dim], dim=1)
        freqs_imag_f, freqs_imag_h, freqs_imag_w = freqs_imag.split([freq_f_dim, freq_h_dim, freq_w_dim], dim=1)

        outputs = []
        for batch_idx, grid_size in enumerate(grid_sizes):
            f, h, w = [int(v) for v in grid_size]
            seq_len = f * h * w
            freq_real_grid = torch.cat(
                [
                    freqs_real_f[:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                    freqs_real_h[:h].view(1, h, 1, -1).expand(f, h, w, -1),
                    freqs_real_w[:w].view(1, 1, w, -1).expand(f, h, w, -1),
                ],
                dim=-1,
            ).reshape(1, seq_len, 1, half_dim)
            freq_imag_grid = torch.cat(
                [
                    freqs_imag_f[:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                    freqs_imag_h[:h].view(1, h, 1, -1).expand(f, h, w, -1),
                    freqs_imag_w[:w].view(1, 1, w, -1).expand(f, h, w, -1),
                ],
                dim=-1,
            ).reshape(1, seq_len, 1, half_dim)

            x_item = x[batch_idx : batch_idx + 1]
            x_valid = x_item[:, :seq_len].reshape(1, seq_len, num_heads, half_dim, 2)
            x_real = x_valid[..., 0]
            x_imag = x_valid[..., 1]

            rope_input = torch.cat([x_real, x_imag], dim=-1)
            cos = torch.cat([freq_real_grid, freq_real_grid], dim=-1)
            sin = torch.cat([freq_imag_grid, freq_imag_grid], dim=-1)
            rope_output = self.rope(rope_input, cos, sin)
            out_valid = (
                rope_output.reshape(1, seq_len, num_heads, 2, half_dim)
                .transpose(-1, -2)
                .reshape(1, seq_len, num_heads, head_dim)
            )
            outputs.append(torch.cat([out_valid, x_item[:, seq_len:]], dim=1))
        return torch.cat(outputs, dim=0)


class _WanModel(DynamicModule):
    def _setup(self, *args, **kwargs):
        del args, kwargs
        freqs_real = self.freqs.real.to(dtype=torch.float16)
        freqs_imag = self.freqs.imag.to(dtype=torch.float16)
        self.register_buffer("freqs_real", freqs_real, persistent=False)
        self.register_buffer("freqs_imag", freqs_imag, persistent=False)
        return self

    def unpatchify(self, x, grid_sizes):
        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes, strict=True):
            f, h, w = [int(item) for item in v]
            pf, ph, pw = self.patch_size
            seq_len = f * h * w
            u = u[:seq_len].view(f, h, w, pf, ph, pw, c)
            u = u.permute(6, 0, 3, 1, 4, 2, 5)
            u = u.reshape(c, f * pf, h * ph, w * pw)
            out.append(u)
        return out

    def forward(
        self,
        x,
        context,
        e,
        e0,
        context_lens,
    ):
        device = self.patch_embedding.weight.device
        if self.freqs_real.device != device:
            self.freqs_real = self.freqs_real.to(device)
            self.freqs_imag = self.freqs_imag.to(device)

        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = [u.shape[2:] for u in x]
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long, device=device)
        target_seq_len = e.size(1)
        x = torch.cat(
            [
                torch.cat(
                    [u, torch.zeros(1, target_seq_len - u.size(1), u.size(2), device=device, dtype=u.dtype)], dim=1
                )
                for u in x
            ]
        )

        if isinstance(context, list):
            context = torch.concat([u.unsqueeze(0) for u in context], dim=0)
        context = self.text_embedding(context)

        if context.dtype != torch.float16:
            context = context.to(torch.float16)

        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs_real=self.freqs_real,
            freqs_imag=self.freqs_imag,
            context=context,
            context_lens=context_lens,
        )

        for block in self.blocks:
            x = block(x, **kwargs)

        x = self.head(x, e)
        x = self.unpatchify(x, grid_sizes)
        return [u for u in x]


class _WanRMSNorm(DynamicModule):
    def _setup(self, *args, **kwargs):
        del args, kwargs
        self.norm = xhnn.RMSNorm(self.dim, self.eps)
        self.norm.weight = nn.Parameter(deepcopy(self.weight.data))
        return self

    def forward(self, x):
        return self.norm(x)


class _WanLayerNorm(DynamicModule):
    def _setup(self, *args, **kwargs):
        del args, kwargs
        self.norm = nn.LayerNorm(
            self.normalized_shape,
            eps=self.eps,
            elementwise_affine=self.elementwise_affine,
        ).half().cuda()
        if self.weight is not None:
            self.norm.weight = nn.Parameter(deepcopy(self.weight.data)).half()
        if self.bias is not None:
            self.norm.bias = nn.Parameter(deepcopy(self.bias.data)).half()
        return self

    def forward(self, x):
        return self.norm(x)


class _WanSelfAttention(DynamicModule):
    def _setup(self, *args, **kwargs):
        del args, kwargs
        self.rope = _Wan3DRope()
        self.flash_attn = xhnn.FlashAttention(
            self.num_heads,
            scale=1 / math.sqrt(self.head_dim),
            is_causal=False,
            sliding_window=None,
            # q_bits=16,
            # k_bits=16,
            # v_bits=16,
            # s_bits=16,
            # p_bits=16
        )
        return self

    def forward(self, x, seq_lens, grid_sizes, freqs_real, freqs_imag):
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)
        q = self.rope(q, grid_sizes, freqs_real, freqs_imag).transpose(1, 2)
        k = self.rope(k, grid_sizes, freqs_real, freqs_imag).transpose(1, 2)
        v = v.transpose(1, 2)
        x = self.flash_attn(q, k, v / FLASH_ATTN_V_SCALE, kv_valid_length=seq_lens) * FLASH_ATTN_V_SCALE
        x = x.transpose(1, 2).flatten(2)
        x = self.o(x)
        # print(x.abs().mean())
        return x


class _WanCrossAttention(DynamicModule):
    def _setup(self, *args, **kwargs):
        del args, kwargs
        self.flash_attn = xhnn.FlashAttention(
            self.num_heads,
            scale=1 / math.sqrt(self.head_dim),
            is_causal=False,
            sliding_window=None,
            # q_bits=16,
            # k_bits=16,
            # v_bits=16,
            # s_bits=16,
            # p_bits=16
        )
        return self

    def forward(self, x, context, context_lens):
        b, s, n, d = x.size(0), x.size(1), self.num_heads, self.head_dim
        q = self.norm_q(self.q(x)).view(b, s, n, d).transpose(1, 2)
        k = self.norm_k(self.k(context)).view(b, -1, n, d).transpose(1, 2)
        v = self.v(context).view(b, -1, n, d).transpose(1, 2)

        if context_lens is None:
            x = self.flash_attn(q, k, v / FLASH_ATTN_V_SCALE) * FLASH_ATTN_V_SCALE
        else:
            x = self.flash_attn(q, k, v / FLASH_ATTN_V_SCALE, kv_valid_length=context_lens) * FLASH_ATTN_V_SCALE
        x = x.transpose(1, 2).flatten(2)
        x = self.o(x)
        return x


class _WanAttentionBlock(DynamicModule):
    def _setup(self, *args, **kwargs):
        del args, kwargs
        return self

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs_real,
        freqs_imag,
        context,
        context_lens,
    ):
        mod = self.modulation.unsqueeze(0) + e
        e0, e1, e2, e3, e4, e5 = mod.split(1, dim=2)

        y = self.self_attn(
            self.norm1(x) * (1 + e1.squeeze(2)) + e0.squeeze(2),
            seq_lens,
            grid_sizes,
            freqs_real,
            freqs_imag,
        )
        x = x + y * e2.squeeze(2)

        x = x + self.cross_attn(self.norm3(x), context, context_lens)
        y = self.ffn(self.norm2(x) * (1 + e4.squeeze(2)) + e3.squeeze(2))
        x = x + y * e5.squeeze(2)
        return x


class _Head(DynamicModule):
    def _setup(self, *args, **kwargs):
        del args, kwargs
        return self

    def forward(self, x, e):
        mod = self.modulation.unsqueeze(0) + e.unsqueeze(2)
        e0, e1 = mod.split(1, dim=2)
        x = self.head(self.norm(x) * (1 + e1.squeeze(2)) + e0.squeeze(2))
        return x


def wrap_dit_model(model: nn.Module) -> nn.Module:
    wrapped_classes = {
        Head: _Head,
        WanModel: _WanModel,
        WanAttentionBlock: _WanAttentionBlock,
        WanCrossAttention: _WanCrossAttention,
        WanLayerNorm: _WanLayerNorm,
        WanRMSNorm: _WanRMSNorm,
        WanSelfAttention: _WanSelfAttention,
    }
    for _, module in list(model.named_modules()):
        dynamic_cls = wrapped_classes.get(type(module))
        if dynamic_cls is not None and not isinstance(module, DynamicModule):
            dynamic_cls.convert(module)
    return model


class Wan22DiTExportWrapper(nn.Module):
    """Wrap Wan2.2 low/high noise WanModel for export.

    This wrapper keeps the original forward signature but normalizes list/tensor
    inputs for tracing/export.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = wrap_dit_model(model)

    def _prepare_latent(self, latent_model_input, y: Optional[torch.Tensor] = None):
        latent_arg = latent_model_input if isinstance(latent_model_input, list) else [latent_model_input]
        if self.model.model_type == "i2v":
            assert y is not None
        if y is None:
            return latent_arg

        y_arg = y if isinstance(y, list) else [y]
        return [torch.cat([u, v], dim=0) for u, v in zip(latent_arg, y_arg, strict=True)]

    def _prepare_context(self, context):
        context_arg = context if isinstance(context, list) else [context]
        if not context_arg:
            return context_arg

        device = self.model.patch_embedding.weight.device
        text_len = getattr(self.model, "text_len", None)
        if text_len is None:
            return context_arg

        padded_context = [
            torch.cat(
                [
                    u,
                    torch.zeros(text_len - u.size(0), u.size(1), device=device, dtype=u.dtype),
                ]
            )
            for u in context_arg
        ]
        return torch.concat([u.unsqueeze(0) for u in padded_context], dim=0)

    def forward(
        self,
        latent_model_input,
        context,
        e: torch.Tensor,
        e0: torch.Tensor,
        context_lens: torch.Tensor,
        y: Optional[torch.Tensor] = None,
    ):
        latent_arg = self._prepare_latent(latent_model_input, y)
        context_arg = self._prepare_context(context)
        kwargs = {
            "context": context_arg,
            "e": e,
            "e0": e0,
            "context_lens": context_lens,
        }
        return self.model(latent_arg, **kwargs)[0]


Wan2_2DiTExportWrapper = Wan22DiTExportWrapper
