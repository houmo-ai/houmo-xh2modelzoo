# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0

"""Export Cosmos3-Nano sound tokenizer decoder as split HMONNX stages."""

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

from common.audio_codec import DEFAULT_SOUND_TOKENIZER_DIR, load_sound_tokenizer_model, make_sound_latents  # noqa: E402

DEFAULT_OUTPUT_ROOT = _COSMOS3_ROOT / "data" / "sound_tokenizer_decoder_split"
STAGES = ("conv_in", "block0", "block1", "block2", "block3", "block4", "head")


class SoundDecoderStageWrapper(nn.Module):
    def __init__(self, decoder: nn.Module, stage: str):
        super().__init__()
        if stage not in STAGES:
            raise ValueError(f"Unsupported stage {stage!r}, expected one of {STAGES}")
        self.decoder = decoder
        self.stage = stage

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        layers = self.decoder.layers
        if self.stage == "conv_in":
            return layers[0](hidden_states)
        if self.stage.startswith("block"):
            block_index = int(self.stage.removeprefix("block"))
            return layers[block_index + 1](hidden_states)
        x = layers[6](hidden_states)
        x = layers[7](x)
        x = layers[8](x)
        return torch.clamp(x, min=-1.0, max=1.0)


def tensor_summary(tensor: torch.Tensor) -> dict[str, Any]:
    return {"shape": list(tensor.shape), "dtype": str(tensor.dtype), "device": str(tensor.device)}


