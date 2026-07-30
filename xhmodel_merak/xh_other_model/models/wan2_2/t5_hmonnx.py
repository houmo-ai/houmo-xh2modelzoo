# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0

# pyright: reportMissingImports=false

import json
from pathlib import Path
from typing import Optional, Sequence, Union

import torch
import torch.nn as nn

from xhquant.api import HMONNXGoldenInference


def _parse_torch_dtype(dtype_name: str) -> torch.dtype:
    dtype = getattr(torch, dtype_name, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"Unsupported torch dtype: {dtype_name}")
    return dtype


class Wan22T5EncoderInference(nn.Module):
    def __init__(
        self,
        hmonnx_path: Union[str, Path],
        tokenizer,
        token_embedding: nn.Embedding,
        max_sequence_length: int,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.runtime = HMONNXGoldenInference(str(hmonnx_path))
        self.tokenizer = tokenizer
        self.token_embedding = token_embedding
        self.max_sequence_length = max_sequence_length
        self._device = (
            torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._dtype = dtype
        self.runtime.exec_device = self._device
        self.token_embedding = self.token_embedding.to(device=self._device, dtype=self._dtype)

    @classmethod
    def from_meta(
        cls,
        meta_path: Union[str, Path],
        tokenizer,
        token_embedding: nn.Embedding,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "Wan22T5EncoderInference":
        meta_file = Path(meta_path)
        meta = json.load(open(meta_file, "r"))
        runtime_dtype = dtype
        if runtime_dtype is None:
            runtime_dtype = _parse_torch_dtype(meta.get("dtype", "bfloat16"))
        return cls(
            hmonnx_path=meta_file.parent / meta["hmonnx_file"],
            tokenizer=tokenizer,
            token_embedding=token_embedding,
            max_sequence_length=int(meta.get("max_sequence_length", 512)),
            device=device,
            dtype=runtime_dtype,
        )

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    def to(self, *args, **kwargs):
        device, dtype = torch._C._nn._parse_to(*args, **kwargs)[:2]
        if device is not None:
            self._device = torch.device(device)
            self.runtime.exec_device = self._device
            self.token_embedding = self.token_embedding.to(self._device)
        if dtype is not None:
            self._dtype = dtype
            self.token_embedding = self.token_embedding.to(dtype=dtype)
        return self

    def prepare_inputs(self, texts: Sequence[str]):
        ids, mask = self.tokenizer(texts, return_mask=True, add_special_tokens=True)
        return ids.to(self.device), mask.to(self.device)

    def forward(self, ids: torch.Tensor, mask: torch.Tensor):
        seq_lens = mask.gt(0).sum(dim=1).long()
        inputs_embeds = self.token_embedding(ids.long().to(self.device)).to(self.device, dtype=self.dtype)
        mask = mask.to(self.device, dtype=torch.int32)
        mask_4d = mask.view(mask.shape[0], 1, 1, -1)
        mask_bias = torch.zeros(mask_4d.shape, device=self.device, dtype=self.dtype).masked_fill(mask_4d == 0, -65504.0)
        outputs = self.runtime(inputs_embeds, mask_bias)
        context = outputs[0] if isinstance(outputs, tuple) else outputs
        return [u[:v] for u, v in zip(context, seq_lens, strict=True)]

    def __call__(self, texts, device):
        del device
        ids, mask = self.prepare_inputs(texts)
        return self.forward(ids, mask)


Wan2_2T5EncoderInference = Wan22T5EncoderInference
