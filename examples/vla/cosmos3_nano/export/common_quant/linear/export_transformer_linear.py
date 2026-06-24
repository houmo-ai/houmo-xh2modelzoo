# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0

"""Export one Cosmos3-Nano transformer Linear weight from safetensors."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from safetensors import safe_open


from common.paths import default_model_root  # noqa: E402


DEFAULT_MODEL = default_model_root()
_COSMOS3_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_ROOT = _COSMOS3_ROOT / "data" / "transformer_linear_smoke"


class TransformerLinearWrapper(nn.Module):
    def __init__(self, weight: torch.Tensor, bias: torch.Tensor | None):
        super().__init__()
        self.linear = nn.Linear(weight.shape[1], weight.shape[0], bias=bias is not None)
        self.linear.weight.data.copy_(weight.float())
        if bias is not None:
            self.linear.bias.data.copy_(bias.float())

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.linear(hidden_states)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_tensor(transformer_dir: Path, weight_map: dict[str, str], key: str) -> torch.Tensor:
    shard = weight_map.get(key)
    if shard is None:
        raise KeyError(f"Missing tensor key in transformer weight map: {key}")
    with safe_open(transformer_dir / shard, framework="pt", device="cpu") as f:
        return f.get_tensor(key)
