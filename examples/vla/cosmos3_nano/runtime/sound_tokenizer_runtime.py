# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0

"""Runtime helpers for Cosmos3-Nano sound tokenizer HMONNX graphs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from xhquant.api import HMONNXInference, xhquant_init

SOUND_DECODER_SPLIT_STAGES = ("conv_in", "block0", "block1", "block2", "block3", "block4", "head")
SOUND_DECODER_STAGE_INPUT_SHAPES_T2 = {
    "conv_in": (1, 64, 2),
    "block0": (1, 5120, 2),
    "block1": (1, 2560, 16),
    "block2": (1, 1280, 96),
    "block3": (1, 640, 480),
    "block4": (1, 320, 1920),
    "head": (1, 320, 3840),
}


@dataclass(frozen=True)
class SoundTokenizerDecoderHmonnxPath:
    path: Path

    def require(self) -> None:
        if not self.path.exists():
            raise FileNotFoundError(f"Missing sound tokenizer decoder HMONNX file: {self.path}")


def as_hmonnx_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (list, tuple)) and len(output) == 1 and isinstance(output[0], torch.Tensor):
        return output[0]
    raise TypeError(f"Unsupported HMONNX output type: {type(output)!r}")


class SoundTokenizerDecoderHmonnxRuntime:
    """Host runtime for fixed-shape sound tokenizer decoder HMONNX graph."""

    def __init__(
        self,
        hmonnx_path: SoundTokenizerDecoderHmonnxPath,
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
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        output = self.session.forward(latents.to(self.device, dtype=self.hmonnx_dtype))
        return as_hmonnx_tensor(output).detach().cpu()


@dataclass(frozen=True)
class SoundTokenizerDecoderSplitHmonnxPaths:
    conv_in: Path
    block0: Path
    block1: Path
    block2: Path
    block3: Path
    block4: Path
    head: Path

    def all(self) -> list[Path]:
        return [getattr(self, stage) for stage in SOUND_DECODER_SPLIT_STAGES]

    def as_dict(self) -> dict[str, str]:
        return {stage: str(getattr(self, stage)) for stage in SOUND_DECODER_SPLIT_STAGES}


def require_sound_decoder_split_paths(paths: SoundTokenizerDecoderSplitHmonnxPaths) -> None:
    missing = [str(path) for path in paths.all() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing sound tokenizer decoder split HMONNX files: " + ", ".join(missing))


def _align_stage_input(tensor: torch.Tensor, expected_shape: tuple[int, int, int] | None) -> torch.Tensor:
    if expected_shape is None or tuple(tensor.shape) == expected_shape:
        return tensor
    if tuple(tensor.shape[:2]) != expected_shape[:2]:
        raise ValueError(f"Cannot align sound decoder tensor shape {tuple(tensor.shape)} to {expected_shape}")
    expected_t = expected_shape[2]
    current_t = tensor.shape[2]
    if current_t > expected_t:
        return tensor[:, :, :expected_t]
    pad_t = expected_t - current_t
    return torch.nn.functional.pad(tensor, (0, pad_t))


class SoundTokenizerDecoderSplitHmonnxRuntime:
    """Host runtime for fixed-shape split sound tokenizer decoder HMONNX graphs."""

    def __init__(
        self,
        paths: SoundTokenizerDecoderSplitHmonnxPaths,
        *,
        device: str = "cuda",
        input_dtype: str = "float16",
        debug: bool = False,
        init_xhquant: bool = True,
    ) -> None:
        require_sound_decoder_split_paths(paths)
        if init_xhquant:
            xhquant_init(None, debug=debug)
        self.paths = paths
        self.device = torch.device(device if torch.cuda.is_available() or not device.startswith("cuda") else "cpu")
        self.hmonnx_dtype = torch.float16 if input_dtype == "float16" else torch.float32
        self.sessions = {
            stage: HMONNXInference(str(getattr(paths, stage))).to(self.device)
            for stage in SOUND_DECODER_SPLIT_STAGES
        }

    @torch.no_grad()
    def run(self, latents: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs: dict[str, torch.Tensor] = {}
        hidden_states = latents
        for stage in SOUND_DECODER_SPLIT_STAGES:
            expected_shape = SOUND_DECODER_STAGE_INPUT_SHAPES_T2.get(stage) if latents.shape[2] == 2 else None
            hidden_states = _align_stage_input(hidden_states, expected_shape)
            output = self.sessions[stage].forward(hidden_states.to(self.device, dtype=self.hmonnx_dtype))
            hidden_states = as_hmonnx_tensor(output).detach().cpu()
            outputs[stage] = hidden_states
        return outputs

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.run(latents)["head"]
