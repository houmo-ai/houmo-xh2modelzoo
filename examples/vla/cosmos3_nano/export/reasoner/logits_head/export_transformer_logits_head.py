# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0

"""Export Cosmos3-Nano reasoner final RMSNorm + lm_head logits probe."""

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

from export.common_quant.linear.export_transformer_linear import (  # noqa: E402
    DEFAULT_MODEL,
    load_json,
    load_tensor,
)


DEFAULT_OUTPUT_ROOT = _COSMOS3_ROOT / "data" / "transformer_logits_head_smoke"


class TransformerLogitsHeadWrapper(nn.Module):
    def __init__(self, norm_weight: torch.Tensor, lm_head_weight: torch.Tensor, eps: float = 1e-6):
        super().__init__()
        self.norm_weight = nn.Parameter(norm_weight.float().clone())
        self.lm_head = nn.Linear(lm_head_weight.shape[1], lm_head_weight.shape[0], bias=False)
        self.lm_head.weight.data.copy_(lm_head_weight.float())
        self.eps = float(eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_float = hidden_states.float()
        variance = (hidden_float * hidden_float).mean(dim=-1, keepdim=True)
        normed = hidden_states * torch.rsqrt(variance.to(hidden_states.dtype) + self.eps)
        normed = normed * self.norm_weight.to(hidden_states.dtype)
        return self.lm_head(normed)


