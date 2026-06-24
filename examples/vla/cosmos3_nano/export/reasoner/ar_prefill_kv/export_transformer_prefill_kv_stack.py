# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0

"""Export Cosmos3-Nano prefill stack with per-layer present KV outputs."""

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
from export.common_quant.attention.export_transformer_attention_proj import load_attention_proj  # noqa: E402
from export.common_quant.attention.export_transformer_rope import (  # noqa: E402
    TransformerMropeCacheWrapper,
    TransformerRoPEApplyWrapper,
    build_inv_freq,
)
from export.common_quant.linear.export_transformer_linear import DEFAULT_MODEL, load_json  # noqa: E402
from export.common_quant.mlp.export_transformer_text_mlp import load_text_mlp  # noqa: E402


DEFAULT_OUTPUT_ROOT = _COSMOS3_ROOT / "data" / "transformer_prefill_kv_stack_36layers_smoke"


class TransformerPrefillKVSelfAttentionWrapper(nn.Module):
    def __init__(self, projection: nn.Module, rope_cache: nn.Module, rope_apply: nn.Module, attention_core: nn.Module):
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

        attn_output = self.attention_core(q, k, v, attention_mask)
        attn_output = F.linear(attn_output, self.projection.o_weight)
        return attn_output, k, v


class TransformerPrefillKVLayerWrapper(nn.Module):
    def __init__(self, self_attention: nn.Module, mlp: nn.Module):
        super().__init__()
        self.self_attention = self_attention
        self.mlp = mlp

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        attn_output, present_key, present_value = self.self_attention(hidden_states, position_ids, attention_mask)
        hidden_states = hidden_states + attn_output
        hidden_states = hidden_states + self.mlp(hidden_states)
        return hidden_states, present_key, present_value


class TransformerPrefillKVStackWrapper(nn.Module):
    def __init__(self, layers: list[nn.Module]):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        present_key_values: list[torch.Tensor] = []
        for layer in self.layers:
            hidden_states, present_key, present_value = layer(hidden_states, position_ids, attention_mask)
            present_key_values.extend([present_key, present_value])
        return (hidden_states, *present_key_values)


