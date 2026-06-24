# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0

"""Export Cosmos3-Nano AutoencoderKLWan encoder."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from diffusers import AutoencoderKLWan


_COSMOS3_ROOT = Path(__file__).resolve().parents[3]
if str(_COSMOS3_ROOT) not in sys.path:
    sys.path.insert(0, str(_COSMOS3_ROOT))

from common.paths import default_model_root  # noqa: E402

DEFAULT_OUTPUT_ROOT = _COSMOS3_ROOT / "data" / "vae_encoder_smoke"

from export.generator.vae_decoder.export_vae_decoder import replace_wan_causal_conv3d_with_2d  # noqa: E402


def patch_vae_encoder_for_onnx_export(vae: AutoencoderKLWan, lower_conv3d_to_2d: bool = False) -> dict[str, Any]:
    conv3d_modules: list[str] = []
    if lower_conv3d_to_2d:
        conv3d_modules.extend(f"encoder.{name}" for name in replace_wan_causal_conv3d_with_2d(vae.encoder))
        if vae.quant_conv.__class__.__name__ == "WanCausalConv3d":
            conv3d_modules.extend(f"quant_conv.{name}" for name in replace_wan_causal_conv3d_with_2d(vae.quant_conv))
        else:
            conv3d_modules.extend(f"quant_conv.{name}" for name in replace_wan_causal_conv3d_with_2d(vae.quant_conv))
    return {
        "conv3d_to_temporal_conv2d": len(conv3d_modules),
        "conv3d_modules": conv3d_modules,
        "reason": "5D Conv is not supported by the current ONNX/HMONNX path",
    }


class Cosmos3VAEEncoderWrapper(nn.Module):
    def __init__(self, vae: AutoencoderKLWan, output_mode: str = "mode"):
        super().__init__()
        if output_mode not in {"mode", "parameters"}:
            raise ValueError(f"Unsupported output_mode: {output_mode}")
        self.vae = vae
        self.output_mode = output_mode

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        latent_dist = self.vae.encode(sample, return_dict=False)[0]
        if self.output_mode == "parameters":
            return latent_dist.parameters
        return latent_dist.mode()


def torch_dtype(name: str) -> torch.dtype:
    return {"float16": torch.float16, "float32": torch.float32}[name]


def tensor_summary(tensor: torch.Tensor) -> dict[str, Any]:
    return {"shape": list(tensor.shape), "dtype": str(tensor.dtype), "device": str(tensor.device)}


