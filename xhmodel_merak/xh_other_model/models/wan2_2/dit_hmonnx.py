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
        self._dtype = torch.float16
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

    @staticmethod
    def _as_list(value, name: str):
        if isinstance(value, (list, tuple)):
            if not value:
                raise ValueError(f"{name} must not be empty")
            return list(value)
        return [value]

    def _prepare_tensor(self, value: torch.Tensor, dtype: torch.dtype):
        return value.to(self.device, dtype=dtype)

    def _prepare_single_tensor(self, value, name: str, dtype: torch.dtype):
        values = self._as_list(value, name)
        if len(values) != 1:
            raise ValueError(f"{name} expects one tensor for current Wan2.2 HMONNX export, got {len(values)}")
        return self._prepare_tensor(values[0], dtype=dtype)

    def forward(
        self,
        latent_model_input,
        context,
        e,
        e0,
        y=None,
    ):
        latent = self._prepare_single_tensor(latent_model_input, "latent_model_input", self._dtype)
        cond = self._prepare_single_tensor(context, "context", self._dtype)
        e = self._prepare_tensor(e, self._dtype)
        e0 = self._prepare_tensor(e0, self._dtype)

        inputs = [
            latent,
            cond,
            e,
            e0,
        ]
        if y is not None:
            inputs.append(self._prepare_single_tensor(y, "y", self._dtype))
        outputs = self.runtime(*inputs)
        sample = outputs[0] if isinstance(outputs, tuple) else outputs
        return (sample,)


Wan2_2DiTInference = Wan22DiTInference
