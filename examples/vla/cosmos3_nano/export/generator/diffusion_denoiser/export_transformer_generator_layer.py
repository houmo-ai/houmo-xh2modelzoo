# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0

"""Export one Cosmos3-Nano generator denoiser layer smoke graph."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


_COSMOS3_ROOT = Path(__file__).resolve().parents[3]
if str(_COSMOS3_ROOT) not in sys.path:
    sys.path.insert(0, str(_COSMOS3_ROOT))

from export.common_quant.attention.export_transformer_attention_core import (  # noqa: E402
    TransformerAttentionCoreWrapper,
    make_causal_mask,
)
from export.common_quant.attention.export_transformer_rope import torch_dtype  # noqa: E402
from export.common_quant.linear.export_transformer_linear import DEFAULT_MODEL  # noqa: E402
from export.generator.diffusion_denoiser.export_transformer_generator_attention_proj import (  # noqa: E402
    load_generator_attention_proj,
)
from export.generator.diffusion_denoiser.export_transformer_generator_mlp import load_generator_mlp  # noqa: E402


DEFAULT_OUTPUT_ROOT = _COSMOS3_ROOT / "data" / "transformer_generator_layer0_smoke"


class TransformerGeneratorLayerWrapper(nn.Module):
    def __init__(self, attention_proj: nn.Module, mlp: nn.Module, scaling: float):
        super().__init__()
        self.attention_proj = attention_proj
        self.attention_core = TransformerAttentionCoreWrapper(scaling=scaling)
        self.mlp = mlp

    def _project_qkv(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        attn = self.attention_proj
        normed = attn._rms_norm(hidden_states, attn.input_norm_weight)
        q = F.linear(normed, attn.q_weight)
        k = F.linear(normed, attn.k_weight)
        v = F.linear(normed, attn.v_weight)
        batch, seq, _ = hidden_states.shape
        q = q.reshape(batch, seq, attn.num_attention_heads, attn.head_dim)
        k = k.reshape(batch, seq, attn.num_key_value_heads, attn.head_dim)
        v = v.reshape(batch, seq, attn.num_key_value_heads, attn.head_dim)
        q = attn._rms_norm(q, attn.norm_q_weight)
        k = attn._rms_norm(k, attn.norm_k_weight)
        return q, k, v

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        q, k, v = self._project_qkv(hidden_states)
        attn_output = self.attention_core(
            q.transpose(1, 2).contiguous(),
            k.transpose(1, 2).contiguous(),
            v.transpose(1, 2).contiguous(),
            attention_mask,
        )
        projected_attn = F.linear(attn_output, self.attention_proj.o_weight)
        hidden_states = hidden_states + projected_attn
        hidden_states = hidden_states + self.mlp(hidden_states)
        return hidden_states


def load_generator_layer(model_dir: Path, layer: int) -> tuple[TransformerGeneratorLayerWrapper, dict[str, Any]]:
    attention_proj, attention_meta = load_generator_attention_proj(model_dir, layer, include_qk_norm=True)
    mlp, mlp_meta = load_generator_mlp(model_dir, layer)
    head_dim = int(attention_meta["head_dim"])
    wrapper = TransformerGeneratorLayerWrapper(attention_proj, mlp, scaling=head_dim**-0.5)
    return wrapper, {"layer": layer, "attention": attention_meta, "mlp": mlp_meta, "scaling": head_dim**-0.5}


def make_zero_mask(batch: int, seq: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.zeros((batch, 1, seq, seq), device=device, dtype=dtype)


