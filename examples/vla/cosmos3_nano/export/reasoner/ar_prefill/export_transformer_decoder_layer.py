# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0

"""Export one Cosmos3-Nano transformer decoder layer probe."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

_COSMOS3_ROOT = Path(__file__).resolve().parents[3]
if str(_COSMOS3_ROOT) not in sys.path:
    sys.path.insert(0, str(_COSMOS3_ROOT))

from export.common_quant.attention.export_transformer_self_attention import (  # noqa: E402
    load_self_attention,
    make_inputs as make_attention_inputs,
)
from export.common_quant.linear.export_transformer_linear import DEFAULT_MODEL  # noqa: E402
from export.common_quant.mlp.export_transformer_text_mlp import load_text_mlp  # noqa: E402


DEFAULT_OUTPUT_ROOT = _COSMOS3_ROOT / "data" / "transformer_decoder_layer0_smoke"


class TransformerDecoderLayerWrapper(nn.Module):
    def __init__(self, self_attention: nn.Module, mlp: nn.Module):
        super().__init__()
        self.self_attention = self_attention
        self.mlp = mlp

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.self_attention(hidden_states, position_ids, attention_mask)
        hidden_states = hidden_states + self.mlp(hidden_states)
        return hidden_states


def load_decoder_layer(model_dir: Path, layer: int) -> tuple[TransformerDecoderLayerWrapper, dict[str, Any]]:
    self_attention, self_attention_meta = load_self_attention(model_dir, layer)
    mlp, mlp_meta = load_text_mlp(model_dir, layer)
    wrapper = TransformerDecoderLayerWrapper(self_attention, mlp)
    return wrapper, {
        "self_attention": self_attention_meta,
        "mlp": mlp_meta,
        "layer": int(layer),
    }


def make_inputs(
    hidden_states: torch.Tensor,
    position_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _ = make_attention_inputs
    return hidden_states, position_ids, attention_mask
