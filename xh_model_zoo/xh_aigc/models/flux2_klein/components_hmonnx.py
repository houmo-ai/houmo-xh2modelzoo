import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from xhquant.api import HMONNXGoldenInference

from .transformer_wrapper import Flux2TransformerExportWrapper


class Flux2KleinTransformerInference(nn.Module):
    def __init__(
        self,
        hmonnx_path: Union[str, Path],
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        self.runtime = HMONNXGoldenInference(str(hmonnx_path))
        self._device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._dtype = dtype
        self.runtime.exec_device = self._device
        self.dtype = dtype
        self.config = SimpleNamespace(in_channels=128)
        self.pos_embed = None

    @classmethod
    def from_root_meta(
        cls,
        meta_path: Union[str, Path],
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float16,
    ) -> "Flux2KleinTransformerInference":
        meta_file = Path(meta_path)
        meta_info = json.load(open(meta_file, "r"))
        if "transformer_hmonnx" not in meta_info:
            raise ValueError(f"meta 中没有 transformer_hmonnx: {meta_file}")
        return cls(meta_file.parent / meta_info["transformer_hmonnx"], device=device, dtype=dtype)

    @property
    def device(self) -> torch.device:
        return self._device

    def to(self, *args, **kwargs):
        device, dtype = torch._C._nn._parse_to(*args, **kwargs)[:2]
        if device is not None:
            self._device = torch.device(device)
            self.runtime.exec_device = self._device
        if dtype is not None:
            self._dtype = dtype
            self.dtype = dtype
        return self

    def cache_context(self, name: str):
        del name
        return torch.no_grad()

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        img_ids: torch.Tensor,
        txt_ids: torch.Tensor,
        guidance: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor]:
        del kwargs
        if guidance is None:
            guidance = torch.ones((hidden_states.shape[0],), device=hidden_states.device, dtype=torch.float32)
        if self.pos_embed is None:
            raise RuntimeError("Flux2KleinTransformerInference 缺少 pos_embed，请通过 attach_hmonnx_transformer(pipe, hm_transformer) 使用。")
        concat_rotary_cos, concat_rotary_sin = Flux2TransformerExportWrapper.build_rotary_inputs(
            self, img_ids.to(device=self.device), txt_ids.to(device=self.device)
        )
        out = self.runtime(
            hidden_states.to(device=self.device, dtype=self._dtype),
            encoder_hidden_states.to(device=self.device, dtype=self._dtype),
            timestep.to(device=self.device, dtype=torch.int32),
            concat_rotary_cos.to(device=self.device, dtype=self._dtype),
            concat_rotary_sin.to(device=self.device, dtype=self._dtype),
            guidance.to(device=self.device, dtype=torch.float16),
        )
        sample = out[0] if isinstance(out, tuple) else out
        return (sample.to(device=self.device, dtype=self._dtype),)


class Flux2KleinVAEInference(nn.Module):
    def __init__(
        self,
        hmonnx_path: Union[str, Path],
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        self.runtime = HMONNXGoldenInference(str(hmonnx_path))
        self._device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._dtype = dtype
        self.runtime.exec_device = self._device
        self.dtype = dtype

    @classmethod
    def from_root_meta(
        cls,
        meta_path: Union[str, Path],
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float16,
    ) -> "Flux2KleinVAEInference":
        meta_file = Path(meta_path)
        meta_info = json.load(open(meta_file, "r"))
        if "vae_hmonnx" not in meta_info:
            raise ValueError(f"meta 中没有 vae_hmonnx: {meta_file}")
        return cls(meta_file.parent / meta_info["vae_hmonnx"], device=device, dtype=dtype)

    @property
    def device(self) -> torch.device:
        return self._device

    def to(self, *args, **kwargs):
        device, dtype = torch._C._nn._parse_to(*args, **kwargs)[:2]
        if device is not None:
            self._device = torch.device(device)
            self.runtime.exec_device = self._device
        if dtype is not None:
            self._dtype = dtype
            self.dtype = dtype
        return self

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        out = self.runtime(
            latents.to(device=self.device, dtype=self._dtype),
        )
        image = out[0] if isinstance(out, tuple) else out
        return image.to(device=self.device, dtype=self._dtype)


class Flux2KleinVAEEncoderInference(nn.Module):
    def __init__(
        self,
        hmonnx_path: Union[str, Path],
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        self.runtime = HMONNXGoldenInference(str(hmonnx_path))
        self._device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._dtype = dtype
        self.runtime.exec_device = self._device
        self.dtype = dtype

    @classmethod
    def from_root_meta(
        cls,
        meta_path: Union[str, Path],
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float16,
    ) -> "Flux2KleinVAEEncoderInference":
        meta_file = Path(meta_path)
        meta_info = json.load(open(meta_file, "r"))
        if "vae_encoder_hmonnx" not in meta_info:
            raise ValueError(f"meta 中没有 vae_encoder_hmonnx: {meta_file}")
        return cls(meta_file.parent / meta_info["vae_encoder_hmonnx"], device=device, dtype=dtype)

    @property
    def device(self) -> torch.device:
        return self._device

    def to(self, *args, **kwargs):
        device, dtype = torch._C._nn._parse_to(*args, **kwargs)[:2]
        if device is not None:
            self._device = torch.device(device)
            self.runtime.exec_device = self._device
        if dtype is not None:
            self._dtype = dtype
            self.dtype = dtype
        return self

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        out = self.runtime(image.to(device=self.device, dtype=self._dtype))
        image_latents = out[0] if isinstance(out, tuple) else out
        return image_latents.to(device=self.device, dtype=self._dtype)


def attach_hmonnx_transformer(pipe, transformer: Flux2KleinTransformerInference):
    if hasattr(pipe, "transformer") and hasattr(pipe.transformer, "config"):
        transformer.config = pipe.transformer.config
    if hasattr(pipe, "transformer") and hasattr(pipe.transformer, "pos_embed"):
        transformer.pos_embed = pipe.transformer.pos_embed.to(transformer.device)
    pipe.transformer = transformer
    return pipe


def load_root_meta(meta_path: Union[str, Path]) -> Dict[str, Any]:
    return json.load(open(meta_path, "r"))
