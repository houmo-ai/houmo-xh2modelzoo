# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0

"""Export Cosmos3-Nano Qwen3-VL vision patch embedding as a Linear graph."""

from __future__ import annotations

import argparse
import sys
import json
from pathlib import Path
from typing import Any


_COSMOS3_ROOT = Path(__file__).resolve().parents[3]
if str(_COSMOS3_ROOT) not in sys.path:
    sys.path.insert(0, str(_COSMOS3_ROOT))

from common.paths import default_model_root  # noqa: E402

import torch
import torch.nn as nn
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel


DEFAULT_MODEL = default_model_root()
_COSMOS3_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_ROOT = _COSMOS3_ROOT / "data" / "vision_patch_embed_smoke"


class Cosmos3VisionPatchEmbedLinearWrapper(nn.Module):
    """Equivalent to Qwen3VLVisionPatchEmbed for fixed flattened pixel input.

    HF Qwen3-VL patch_embed reshapes input to
    [-1, C, temporal_patch, patch, patch] and applies Conv3d with kernel and
    stride equal to the whole patch. For that exact input shape, Conv3d is a
    per-patch Linear projection over C*T*P*P values.
    """

    def __init__(self, patch_embed: nn.Module):
        super().__init__()
        proj = patch_embed.proj
        self.in_channels = int(patch_embed.in_channels)
        self.temporal_patch_size = int(patch_embed.temporal_patch_size)
        self.patch_size = int(patch_embed.patch_size)
        self.embed_dim = int(patch_embed.embed_dim)
        self.weight = nn.Parameter(proj.weight.detach().reshape(self.embed_dim, -1).clone())
        self.bias = nn.Parameter(proj.bias.detach().clone())

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        patches = pixel_values.reshape(
            -1,
            self.in_channels * self.temporal_patch_size * self.patch_size * self.patch_size,
        )
        return torch.nn.functional.linear(patches, self.weight, self.bias)


