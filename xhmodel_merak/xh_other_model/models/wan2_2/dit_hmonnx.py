# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0

# pyright: reportMissingImports=false

import json
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn

from xhquant.api import HMONNXGoldenInference


def _parse_torch_dtype(dtype_name: str) -> torch.dtype:
    dtype = getattr(torch, dtype_name, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"Unsupported torch dtype: {dtype_name}")
    return dtype


class Wan22DiTInference(nn.Module):
    def __init__(
        self,
        hmonnx_path: Union[str, Path],
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.bfloat16,
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
        dtype: Optional[torch.dtype] = None,
    ):
        meta_file = Path(meta_path)
        meta = json.load(open(meta_file, "r"))
        runtime_dtype = dtype
        if runtime_dtype is None:
            runtime_dtype = _parse_torch_dtype(meta.get("dtype", "float16"))
        return cls(meta_file.parent / meta["hmonnx_file"], device=device, dtype=runtime_dtype)

    @property
    def device(self):
        return self._device

    def forward(
        self,
        latent_model_input,
        context,
        e,
        e0,
        context_lens,
        y=None,
    ):
        latent = latent_model_input[0] if isinstance(latent_model_input, (list, tuple)) else latent_model_input
        cond = context[0] if isinstance(context, (list, tuple)) else context
        inputs = [
            latent.to(self.device, dtype=self._dtype),
            cond.to(self.device, dtype=self._dtype),
            e.to(self.device, dtype=self._dtype),
            e0.to(self.device, dtype=self._dtype),
            context_lens.to(self.device, dtype=torch.long),
        ]
        if y is not None:
            y_tensor = y[0] if isinstance(y, (list, tuple)) else y
            inputs.append(y_tensor.to(self.device, dtype=self._dtype))
        outputs = self.runtime(*inputs)
        sample = outputs[0] if isinstance(outputs, tuple) else outputs
        return (sample,)


Wan2_2DiTInference = Wan22DiTInference
