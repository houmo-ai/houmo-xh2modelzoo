# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0

"""Reusable Cosmos3-Nano transformer attention projection loader and wrapper."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


_COSMOS3_ROOT = Path(__file__).resolve().parents[3]
if str(_COSMOS3_ROOT) not in sys.path:
    sys.path.insert(0, str(_COSMOS3_ROOT))

from export.common_quant.linear.export_transformer_linear import load_json, load_tensor  # noqa: E402
from export.common_quant.attention.utils import tensor_summary, torch_dtype  # noqa: E402


__all__ = [
    "TransformerAttentionProjWrapper",
    "load_attention_proj",
    "torch_dtype",
    "tensor_summary",
]


class TransformerAttentionProjWrapper(nn.Module):
    def __init__(
        self,
        input_norm_weight: torch.Tensor,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        v_weight: torch.Tensor,
        o_weight: torch.Tensor,
        norm_q_weight: torch.Tensor | None,
        norm_k_weight: torch.Tensor | None,
        *,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        eps: float,
        include_qk_norm: bool,
    ):
        super().__init__()
        self.eps = float(eps)
        self.num_attention_heads = int(num_attention_heads)
        self.num_key_value_heads = int(num_key_value_heads)
        self.head_dim = int(head_dim)
        self.include_qk_norm = bool(include_qk_norm)
        self.input_norm_weight = nn.Parameter(input_norm_weight.float().clone())
        self.q_weight = nn.Parameter(q_weight.float().clone())
        self.k_weight = nn.Parameter(k_weight.float().clone())
        self.v_weight = nn.Parameter(v_weight.float().clone())
        self.o_weight = nn.Parameter(o_weight.float().clone())
        if norm_q_weight is None:
            norm_q_weight = torch.ones((head_dim,), dtype=torch.float32)
        if norm_k_weight is None:
            norm_k_weight = torch.ones((head_dim,), dtype=torch.float32)
        self.norm_q_weight = nn.Parameter(norm_q_weight.float().clone(), requires_grad=False)
        self.norm_k_weight = nn.Parameter(norm_k_weight.float().clone(), requires_grad=False)

    def _rms_norm(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        variance = x.float().pow(2).mean(dim=-1, keepdim=True)
        normed = x.float() * torch.rsqrt(variance + self.eps)
        return normed.to(x.dtype) * weight.to(x.dtype)

    def forward(self, hidden_states: torch.Tensor, attn_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        normed = self._rms_norm(hidden_states, self.input_norm_weight)
        q = F.linear(normed, self.q_weight)
        k = F.linear(normed, self.k_weight)
        v = F.linear(normed, self.v_weight)
        batch, seq, _ = hidden_states.shape
        q = q.reshape(batch, seq, self.num_attention_heads, self.head_dim)
        k = k.reshape(batch, seq, self.num_key_value_heads, self.head_dim)
        v = v.reshape(batch, seq, self.num_key_value_heads, self.head_dim)
        if self.include_qk_norm:
            q = self._rms_norm(q, self.norm_q_weight)
            k = self._rms_norm(k, self.norm_k_weight)
        out = F.linear(attn_output, self.o_weight)
        return q, k, v, out


def load_attention_proj(model_dir: Path, layer: int, include_qk_norm: bool) -> tuple[TransformerAttentionProjWrapper, dict[str, Any]]:
    transformer_dir = model_dir / "transformer"
    config = load_json(transformer_dir / "config.json")
    index = load_json(transformer_dir / "diffusion_pytorch_model.safetensors.index.json")
    weight_map = index["weight_map"]
    prefix = f"layers.{layer}"
    keys = {
        "input_norm": f"{prefix}.input_layernorm.weight",
        "q": f"{prefix}.self_attn.to_q.weight",
        "k": f"{prefix}.self_attn.to_k.weight",
        "v": f"{prefix}.self_attn.to_v.weight",
        "o": f"{prefix}.self_attn.to_out.weight",
        "norm_q": f"{prefix}.self_attn.norm_q.weight",
        "norm_k": f"{prefix}.self_attn.norm_k.weight",
    }
    tensors = {name: load_tensor(transformer_dir, weight_map, key) for name, key in keys.items()}
    num_attention_heads = int(config["num_attention_heads"])
    num_key_value_heads = int(config["num_key_value_heads"])
    head_dim = int(config["head_dim"])
    wrapper = TransformerAttentionProjWrapper(
        tensors["input_norm"],
        tensors["q"],
        tensors["k"],
        tensors["v"],
        tensors["o"],
        tensors["norm_q"],
        tensors["norm_k"],
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        eps=float(config.get("rms_norm_eps", 1e-6)),
        include_qk_norm=include_qk_norm,
    )
    meta = {
        "keys": keys,
        "shapes": {name: list(tensor.shape) for name, tensor in tensors.items()},
        "dtypes": {name: str(tensor.dtype) for name, tensor in tensors.items()},
        "rms_norm_eps": float(config.get("rms_norm_eps", 1e-6)),
        "hidden_size": int(config["hidden_size"]),
        "num_attention_heads": num_attention_heads,
        "num_key_value_heads": num_key_value_heads,
        "head_dim": head_dim,
        "include_qk_norm": include_qk_norm,
    }
    return wrapper, meta
