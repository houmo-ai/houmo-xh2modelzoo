# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0

"""Export Cosmos3-Nano transformer RoPE/mRoPE probes."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from export.common_quant.linear.export_transformer_linear import load_json  # noqa: E402
from export.common_quant.attention.utils import torch_dtype  # noqa: E402


__all__ = [
    "load_rope_config",
    "build_inv_freq",
    "torch_dtype",
    "rotate_half",
    "TransformerRoPEApplyWrapper",
    "TransformerMropeCacheWrapper",
]


def load_rope_config(model_or_config):
    if isinstance(model_or_config, (str, bytes)):
        model_or_config = Path(model_or_config)
    if isinstance(model_or_config, Path):
        transformer_dir = model_or_config
        if transformer_dir.name != "transformer":
            transformer_dir = transformer_dir / "transformer"
        return load_json(transformer_dir / "config.json")
    if hasattr(model_or_config, "get"):
        return dict(model_or_config)
    raise TypeError("model_or_config must be model dir path or config dict")



def build_inv_freq(config_or_head_dim, base: float = 10000.0) -> torch.Tensor:
    if isinstance(config_or_head_dim, dict):
        head_dim = int(config_or_head_dim.get("head_dim", 0))
    elif isinstance(config_or_head_dim, int):
        head_dim = int(config_or_head_dim)
    else:
        head_dim = int(getattr(config_or_head_dim, "head_dim", 0))
    if head_dim <= 0:
        raise ValueError("invalid head_dim for build_inv_freq")
    half_dim = max(head_dim // 2, 1)
    freqs = torch.arange(0, half_dim, dtype=torch.float32)
    return 1.0 / (base ** (2.0 * freqs / float(half_dim)))



def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class TransformerRoPEApplyWrapper(nn.Module):
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed


class TransformerMropeCacheWrapper(nn.Module):
    def __init__(self, inv_freq: torch.Tensor, mrope_section: list[int], attention_scaling: float = 1.0):
        super().__init__()
        self.attention_scaling = float(attention_scaling)
        self.register_buffer("inv_freq", inv_freq.float().clone(), persistent=True)

        head_half_dim = int(inv_freq.numel())
        source = torch.zeros((head_half_dim,), dtype=torch.long)
        for dim, offset in enumerate((1, 2), start=1):
            length = int(mrope_section[dim]) * 3
            source[offset:length:3] = dim
        mask = torch.nn.functional.one_hot(source, num_classes=3).to(torch.float32).transpose(0, 1)
        self.register_buffer("source_mask", mask.reshape(3, 1, 1, head_half_dim), persistent=True)

    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if position_ids.ndim == 2:
            position_ids = position_ids.unsqueeze(0).expand(3, position_ids.shape[0], position_ids.shape[1])
        inv_freq = self.inv_freq.reshape(1, 1, -1, 1).expand(3, position_ids.shape[1], -1, 1)
        position_ids = position_ids[:, :, None, :].float()
        freqs_3d = torch.matmul(inv_freq.float(), position_ids).transpose(2, 3)
        freqs = (freqs_3d * self.source_mask.to(freqs_3d.dtype)).sum(dim=0)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling
        return cos, sin
