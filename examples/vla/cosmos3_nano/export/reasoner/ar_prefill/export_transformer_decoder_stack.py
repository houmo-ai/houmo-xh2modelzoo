# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0

"""Export a small Cosmos3-Nano transformer decoder layer stack probe."""

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

from export.common_quant.linear.export_transformer_linear import DEFAULT_MODEL  # noqa: E402
from export.reasoner.ar_prefill.export_transformer_decoder_layer import (  # noqa: E402
    load_decoder_layer,
    make_inputs as make_layer_inputs,
)


DEFAULT_OUTPUT_ROOT = _COSMOS3_ROOT / "data" / "transformer_decoder_stack_2layers_smoke"


class TransformerDecoderStackWrapper(nn.Module):
    def __init__(self, layers: list[nn.Module]):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states, position_ids, attention_mask)
        return hidden_states


def make_inputs(
    hidden_states: torch.Tensor,
    position_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _ = make_layer_inputs
    return hidden_states, position_ids, attention_mask


def load_decoder_stack(
    model_dir: Path,
    start_layer: int,
    num_layers: int,
) -> tuple[TransformerDecoderStackWrapper, dict[str, Any]]:
    layers = []
    for layer_id in range(start_layer, start_layer + num_layers):
        layer, _layer_meta = load_decoder_layer(model_dir, layer_id)
        layers.append(layer)
    return TransformerDecoderStackWrapper(layers), {
        "start_layer": int(start_layer),
        "num_layers": int(num_layers),
    }
