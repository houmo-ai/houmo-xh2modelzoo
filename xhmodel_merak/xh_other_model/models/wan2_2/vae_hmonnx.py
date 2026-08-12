# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0

# pyright: reportMissingImports=false

import json
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn

from xhquant.api import HMONNXGoldenInference


class Wan22VAEEncoderInference(nn.Module):
    def __init__(
        self,
        hmonnx_path: Union[str, Path],
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        self.runtime = HMONNXGoldenInference(str(hmonnx_path))
        self._device = (
            torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._dtype = dtype
        self.runtime.exec_device = self._device

    @classmethod
    def from_meta(
        cls,
        meta_path: Union[str, Path],
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float16,
    ):
        meta_file = Path(meta_path)
        meta = json.load(open(meta_file, "r"))
        return cls(meta_file.parent / meta["hmonnx_file"], device=device, dtype=dtype)

    @property
    def device(self):
        return self._device

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        outputs = self.runtime(video.to(self.device, dtype=self._dtype))
        latent = outputs[0] if isinstance(outputs, tuple) else outputs
        return latent.to(self.device, dtype=self._dtype)


class Wan22VAEDecoderInference(nn.Module):
    def __init__(
        self,
        hmonnx_path: Union[str, Path],
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        self.runtime = HMONNXGoldenInference(str(hmonnx_path))
        self._device = (
            torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._dtype = dtype
        self.runtime.exec_device = self._device

    @classmethod
    def from_meta(
        cls,
        meta_path: Union[str, Path],
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float16,
    ):
        meta_file = Path(meta_path)
        meta = json.load(open(meta_file, "r"))
        return cls(meta_file.parent / meta["hmonnx_file"], device=device, dtype=dtype)

    @property
    def device(self):
        return self._device

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        outputs = self.runtime(latent.to(self.device, dtype=self._dtype))
        video = outputs[0] if isinstance(outputs, tuple) else outputs
        return video.to(self.device, dtype=torch.float32).clamp_(-1, 1)


Wan2_2VAEEncoderInference = Wan22VAEEncoderInference
Wan2_2VAEDecoderInference = Wan22VAEDecoderInference
