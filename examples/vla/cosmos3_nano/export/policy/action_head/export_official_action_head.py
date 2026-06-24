# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0

"""Export Cosmos3-Nano policy action_proj_out/action head as an FP ONNX graph."""

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

from common.official_cosmos3 import DEFAULT_OFFICIAL_TRANSFORMER, load_official_transformer  # noqa: E402

from export.common_quant.attention.export_transformer_rope import torch_dtype  # noqa: E402


DEFAULT_OUTPUT_ROOT = _COSMOS3_ROOT / "data" / "policy_official_action_head_fp16_smoke"


class OfficialCosmos3ActionHeadWrapper(nn.Module):
    def __init__(self, transformer: nn.Module):
        super().__init__()
        if not bool(transformer.config.action_gen):
            raise ValueError("Cosmos3 transformer was not configured with action_gen=True")
        self.action_proj_out = transformer.action_proj_out

    def forward(self, action_hidden_states: torch.Tensor, domain_id: torch.Tensor) -> torch.Tensor:
        if domain_id.ndim == 0 or domain_id.numel() == 1:
            per_token_domain_id = domain_id.reshape(1).expand(action_hidden_states.shape[0])
        else:
            per_token_domain_id = domain_id.reshape(-1)
        return self.action_proj_out(action_hidden_states, per_token_domain_id)


