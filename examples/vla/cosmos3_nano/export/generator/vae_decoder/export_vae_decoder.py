# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0

"""Export Cosmos3-Nano AutoencoderKLWan decoder."""

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

def replace_wan_causal_conv3d_with_2d(vae: object) -> list[str]:
    names: list[str] = []
    for name, module in vae.named_modules():
        if module.__class__.__name__ == "WanCausalConv3d":
            names.append(name)
    return names


import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKLWan


DEFAULT_MODEL = default_model_root()
_COSMOS3_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_ROOT = _COSMOS3_ROOT / "data" / "vae_decoder_smoke"


class Cosmos3VAEDecoderWrapper(nn.Module):
    def __init__(self, vae: AutoencoderKLWan):
        super().__init__()
        self.vae = vae

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        return self.vae.decode(latents, return_dict=False)[0]


