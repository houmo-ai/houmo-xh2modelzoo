# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0

"""Export a small Cosmos3-Nano AR decode transformer stack probe with KV cache."""

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
from export.reasoner.ar_decode.export_transformer_decode_layer import load_decode_layer  # noqa: E402


DEFAULT_OUTPUT_ROOT = _COSMOS3_ROOT / "data" / "transformer_decode_stack_2layers_smoke"


class TransformerDecodeStackWrapper(nn.Module):
    def __init__(self, layers: list[nn.Module]):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *past_key_values: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        present_key_values: list[torch.Tensor] = []
        for layer_idx, layer in enumerate(self.layers):
            past_key = past_key_values[2 * layer_idx]
            past_value = past_key_values[2 * layer_idx + 1]
            hidden_states, present_key, present_value = layer(
                hidden_states,
                position_ids,
                attention_mask,
                past_key,
                past_value,
            )
            present_key_values.extend([present_key, present_value])
        return (hidden_states, *present_key_values)


