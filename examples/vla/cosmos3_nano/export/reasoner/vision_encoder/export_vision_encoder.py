# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0

"""Export Cosmos3-Nano Qwen3-VL vision encoder.

This is the first Stage 0 graph: load only the `vision_encoder` component,
run a fixed-shape PyTorch forward, export ONNX, and optionally convert it to
HMONNX. The script uses the explicit transformers Qwen3VLVisionModel class
because AutoModel treats the standalone vision directory as a full Qwen3VLModel
and expects `visual.*` weight prefixes.
"""

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
DEFAULT_OUTPUT_ROOT = Path("outputs/cosmos3_nano/vision_encoder")


class Cosmos3VisionEncoderExportWrapper(nn.Module):
    """Flatten list outputs so ONNX/HMONNX get stable named tensors."""

    def __init__(self, model: Qwen3VLVisionModel, height: int, width: int):
        super().__init__()
        self.model = model
        patch_size = int(model.config.patch_size)
        grid_thw = torch.tensor([[1, height // patch_size, width // patch_size]], dtype=torch.int64)
        self.register_buffer("grid_thw", grid_thw, persistent=False)

    def forward(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        outputs = self.model(pixel_values, grid_thw=self.grid_thw.to(pixel_values.device))
        image_embeds = outputs.pooler_output
        deepstack = outputs.deepstack_features
        return image_embeds, deepstack[0], deepstack[1], deepstack[2]


