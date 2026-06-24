# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0

"""Export Cosmos3-Nano reasoner token embedding probe."""

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


DEFAULT_OUTPUT_ROOT = _COSMOS3_ROOT / "data" / "text_embedding_smoke"


class TextEmbeddingWrapper(nn.Module):
    def __init__(self, embedding_weight: torch.Tensor):
        super().__init__()
        self.embed_tokens = nn.Embedding(embedding_weight.shape[0], embedding_weight.shape[1])
        self.embed_tokens.weight.data.copy_(embedding_weight.float())

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)


