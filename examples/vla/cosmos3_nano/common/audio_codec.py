# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0

"""Cosmos3-Nano sound tokenizer loading helpers."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from safetensors.torch import load_file


try:
    from common.paths import default_cosmos_framework_root, default_sound_tokenizer_dir
except ModuleNotFoundError:
    from paths import default_cosmos_framework_root, default_sound_tokenizer_dir


COSMOS_FRAMEWORK_ROOT = default_cosmos_framework_root()
if str(COSMOS_FRAMEWORK_ROOT) not in sys.path:
    sys.path.insert(0, str(COSMOS_FRAMEWORK_ROOT))

from cosmos_framework.model.vfm.tokenizers.audio.avae_utils.env import AttrDict  # noqa: E402
from cosmos_framework.model.vfm.tokenizers.audio.avae_utils.models import load_generator  # noqa: E402


DEFAULT_SOUND_TOKENIZER_DIR = default_sound_tokenizer_dir()

_RES_UNIT_INNER_INDEX = {"snake1": 0, "conv1": 1, "snake2": 2, "conv2": 3}


def load_sound_config(sound_tokenizer_dir: Path = DEFAULT_SOUND_TOKENIZER_DIR) -> AttrDict:
    return AttrDict(json.loads((sound_tokenizer_dir / "config.json").read_text(encoding="utf-8")))


def _decoder_block_key_to_framework(key: str, num_blocks: int) -> str:
    snake1_idx = num_blocks + 1
    conv2_idx = num_blocks + 2

    match = re.fullmatch(r"decoder\.block\.(\d+)\.res_unit(\d+)\.(snake1|conv1|snake2|conv2)\.(.+)", key)
    if match:
        block_idx = int(match.group(1))
        res_idx = int(match.group(2))
        inner = match.group(3)
        rest = match.group(4)
        return f"decoder.layers.{block_idx + 1}.layers.{res_idx + 1}.layers.{_RES_UNIT_INNER_INDEX[inner]}.{rest}"
    match = re.fullmatch(r"decoder\.block\.(\d+)\.snake1\.(.+)", key)
    if match:
        return f"decoder.layers.{int(match.group(1)) + 1}.layers.0.{match.group(2)}"
    match = re.fullmatch(r"decoder\.block\.(\d+)\.conv_t1\.(.+)", key)
    if match:
        return f"decoder.layers.{int(match.group(1)) + 1}.layers.1.{match.group(2)}"
    match = re.fullmatch(r"decoder\.conv1\.(.+)", key)
    if match:
        return f"decoder.layers.0.{match.group(1)}"
    match = re.fullmatch(r"decoder\.snake1\.(.+)", key)
    if match:
        return f"decoder.layers.{snake1_idx}.{match.group(1)}"
    match = re.fullmatch(r"decoder\.conv2\.(.+)", key)
    if match:
        return f"decoder.layers.{conv2_idx}.{match.group(1)}"
    return key


def load_sound_tokenizer_state_dict(sound_tokenizer_dir: Path = DEFAULT_SOUND_TOKENIZER_DIR) -> dict[str, torch.Tensor]:
    safetensors_path = sound_tokenizer_dir / "diffusion_pytorch_model.safetensors"
    state = load_file(str(safetensors_path), device="cpu")
    block_ids = {
        int(match.group(1))
        for key in state
        if (match := re.fullmatch(r"decoder\.block\.(\d+)\..+", key)) is not None
    }
    if not block_ids:
        raise RuntimeError(f"No decoder.block.* keys found in {safetensors_path}")
    num_blocks = max(block_ids) + 1

    remapped: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        framework_key = _decoder_block_key_to_framework(key, num_blocks)
        if (framework_key.endswith(".alpha") or framework_key.endswith(".beta")) and value.ndim == 3:
            value = value.reshape(-1).contiguous()
        remapped[framework_key] = value
    return remapped


def load_sound_tokenizer_model(
    sound_tokenizer_dir: Path = DEFAULT_SOUND_TOKENIZER_DIR,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
    decoder_only: bool = False,
    allow_random_encoder: bool = False,
) -> nn.Module:
    config = load_sound_config(sound_tokenizer_dir)
    model = load_generator(config.model_type, config, device).eval()
    state_dict = load_sound_tokenizer_state_dict(sound_tokenizer_dir)
    load_result = model.load_state_dict(state_dict, strict=False)
    missing_encoder_keys = [key for key in load_result.missing_keys if key.startswith("encoder.")]
    if missing_encoder_keys and not decoder_only and not allow_random_encoder:
        raise RuntimeError(
            "Sound tokenizer checkpoint does not contain encoder weights. "
            "Pass decoder_only=True for decoder export, or allow_random_encoder=True for structural encoder smoke tests."
        )
    model.remove_weight_norm()
    model = model.to(device=device, dtype=dtype).eval()
    if decoder_only:
        model.encoder = nn.Identity()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


class SoundTokenizerDecoderWrapper(nn.Module):
    """Decode AVAE latents into clamped waveform samples."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        output: Any = self.model.decode(latents)
        waveform = output["decoder_out"] if isinstance(output, dict) else output
        return torch.clamp(waveform, min=-1.0, max=1.0)


class SoundTokenizerEncoderSpectrogramWrapper(nn.Module):
    """Encode host-computed spectrograms into VAE mean/scale parameters."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.layers = model.encoder.layers

    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        return self.layers(spectrogram)


def sound_encoder_params_to_mean(params: torch.Tensor) -> torch.Tensor:
    return params[:, :64, :]


def make_sound_audio(*, channels: int = 2, samples: int = 3840, seed: int = 19) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    return torch.randn((1, int(channels), int(samples)), generator=generator, dtype=torch.float32)


def make_sound_spectrogram(model: nn.Module, audio: torch.Tensor) -> torch.Tensor:
    encoder = model.encoder
    batch, channels, length = audio.shape
    x = audio
    if channels > 1:
        x = x.reshape(batch * channels, 1, length)
    with torch.autocast(device_type=x.device.type, enabled=False):
        spec = encoder.spectrogram_fn(x.float().squeeze(1))
        mag, ph = torch.view_as_real(spec).chunk(2, dim=-1)
        spectrogram = torch.cat([mag, ph], dim=1).squeeze(-1)
    spectrogram = spectrogram.to(audio.dtype)
    if channels > 1:
        freq = spectrogram.shape[1]
        spectrogram = spectrogram.reshape(batch, channels * freq, *spectrogram.shape[2:])
    return spectrogram


def make_sound_latents(*, channels: int = 64, latent_frames: int = 2, seed: int = 17) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    return torch.randn((1, int(channels), int(latent_frames)), generator=generator, dtype=torch.float32)
