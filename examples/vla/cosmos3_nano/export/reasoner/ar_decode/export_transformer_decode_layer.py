# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0

"""Export one Cosmos3-Nano AR decode transformer layer probe with KV cache."""

from __future__ import annotations

import argparse
import json
import math
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
)
from export.common_quant.attention.export_transformer_attention_proj import load_attention_proj  # noqa: E402
from export.common_quant.attention.export_transformer_rope import (  # noqa: E402
    TransformerMropeCacheWrapper,
    TransformerRoPEApplyWrapper,
    build_inv_freq,
)
from export.common_quant.linear.export_transformer_linear import DEFAULT_MODEL, load_json  # noqa: E402
from export.common_quant.mlp.export_transformer_text_mlp import load_text_mlp  # noqa: E402


DEFAULT_OUTPUT_ROOT = _COSMOS3_ROOT / "data" / "transformer_decode_layer0_smoke"


class TransformerDecodeSelfAttentionWrapper(nn.Module):
    def __init__(
        self,
        projection: nn.Module,
        rope_cache: nn.Module,
        rope_apply: nn.Module,
        attention_core: nn.Module,
    ):
        super().__init__()
        self.projection = projection
        self.rope_cache = rope_cache
        self.rope_apply = rope_apply
        self.attention_core = attention_core

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key: torch.Tensor,
        past_value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        normed = self.projection._rms_norm(hidden_states, self.projection.input_norm_weight)
        q = F.linear(normed, self.projection.q_weight)
        k = F.linear(normed, self.projection.k_weight)
        v = F.linear(normed, self.projection.v_weight)

        batch, seq, _ = hidden_states.shape
        q = q.reshape(batch, seq, self.projection.num_attention_heads, self.projection.head_dim)
        k = k.reshape(batch, seq, self.projection.num_key_value_heads, self.projection.head_dim)
        v = v.reshape(batch, seq, self.projection.num_key_value_heads, self.projection.head_dim)
        q = self.projection._rms_norm(q, self.projection.norm_q_weight)
        k = self.projection._rms_norm(k, self.projection.norm_k_weight)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        cos, sin = self.rope_cache(position_ids)
        q, k = self.rope_apply(q, k, cos, sin)

        present_key = torch.cat((past_key, k), dim=2)
        present_value = torch.cat((past_value, v), dim=2)
        attn_output = self.attention_core(q, present_key, present_value, attention_mask)
        attn_output = F.linear(attn_output, self.projection.o_weight)
        return attn_output, present_key, present_value


class TransformerDecodeLayerWrapper(nn.Module):
    def __init__(self, self_attention: nn.Module, mlp: nn.Module):
        super().__init__()
        self.self_attention = self_attention
        self.mlp = mlp

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key: torch.Tensor,
        past_value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        attn_output, present_key, present_value = self.self_attention(
            hidden_states,
            position_ids,
            attention_mask,
            past_key,
            past_value,
        )
        hidden_states = hidden_states + attn_output
        hidden_states = hidden_states + self.mlp(hidden_states)
        return hidden_states, present_key, present_value


def load_decode_layer(model_dir: Path, layer: int) -> tuple[TransformerDecodeLayerWrapper, dict[str, Any]]:
    transformer_dir = model_dir / "transformer"
    config = load_json(transformer_dir / "config.json")

    projection, projection_meta = load_attention_proj(model_dir, layer, include_qk_norm=True)
    mrope_section = config.get("mrope_section", [1, 1, 1])
    if not isinstance(mrope_section, list):
        mrope_section = [1, 1, 1]
    attention_scaling = float(config.get("attention_scaling", 1.0))

    rope_cache = TransformerMropeCacheWrapper(build_inv_freq(config), list(mrope_section), attention_scaling=attention_scaling)
    rope_apply = TransformerRoPEApplyWrapper()
    head_dim = int(config.get("head_dim", projection.head_dim))
    if head_dim <= 0:
        head_dim = int(projection.head_dim)
    attention_core = TransformerAttentionCoreWrapper(1.0 / math.sqrt(float(head_dim)))

    self_attention = TransformerDecodeSelfAttentionWrapper(
        projection,
        rope_cache,
        rope_apply,
        attention_core,
    )
    mlp, mlp_meta = load_text_mlp(model_dir, layer)
    wrapper = TransformerDecodeLayerWrapper(self_attention, mlp)
    return wrapper, {
        "projection": projection_meta,
        "mlp": mlp_meta,
        "layer": int(layer),
        "mrope_section": list(mrope_section),
        "attention_scaling": attention_scaling,
    }


def make_inputs(
    hidden_states: torch.Tensor,
    position_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    past_key: torch.Tensor,
    past_value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return hidden_states, position_ids, attention_mask, past_key, past_value
