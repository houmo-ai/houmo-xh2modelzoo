# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0

"""Reusable host/runtime wrappers for Cosmos3-Nano VAE encoder graphs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from xhquant.api import HMONNXInference, xhquant_init


@dataclass(frozen=True)
class VaeEncoderHmonnxPath:
    path: Path

    def require(self) -> None:
        if not self.path.exists():
            raise FileNotFoundError(f"Missing VAE encoder HMONNX file: {self.path}")


def make_vae_encoder_sample(
    *,
    frames: int,
    height: int,
    width: int,
    seed: int,
    input_mode: str = "random",
) -> torch.Tensor:
    shape = (1, 3, int(frames), int(height), int(width))
    if input_mode == "zeros":
        return torch.zeros(shape, dtype=torch.float32)
    if input_mode != "random":
        raise ValueError(f"Unsupported input_mode {input_mode!r}; expected random or zeros")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    return torch.randn(shape, generator=generator, dtype=torch.float32)


def as_hmonnx_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (list, tuple)) and len(output) == 1 and isinstance(output[0], torch.Tensor):
        return output[0]
    raise TypeError(f"Unsupported HMONNX output type: {type(output)!r}")


class VaeEncoderHmonnxRuntime:
    """Host runtime for a fixed-shape VAE encoder HMONNX graph."""

    def __init__(
        self,
        hmonnx_path: VaeEncoderHmonnxPath,
        *,
        device: str = "cuda",
        input_dtype: str = "float16",
        debug: bool = False,
        init_xhquant: bool = True,
    ) -> None:
        hmonnx_path.require()
        if init_xhquant:
            xhquant_init(None, debug=debug)
        self.hmonnx_path = hmonnx_path
        self.device = torch.device(device if torch.cuda.is_available() or not device.startswith("cuda") else "cpu")
        self.hmonnx_dtype = torch.float16 if input_dtype == "float16" else torch.float32
        self.session = HMONNXInference(str(hmonnx_path.path)).to(self.device)

    @torch.no_grad()
    def encode(self, sample: torch.Tensor) -> torch.Tensor:
        output = self.session.forward(sample.to(self.device, dtype=self.hmonnx_dtype))
        return as_hmonnx_tensor(output).detach().cpu()
