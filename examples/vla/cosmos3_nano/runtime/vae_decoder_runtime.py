# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0

"""Reusable host/runtime wrappers for Cosmos3-Nano VAE decoder graphs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from diffusers import AutoencoderKLWan
from diffusers.models.autoencoders.autoencoder_kl_wan import CACHE_T
from xhquant.api import HMONNXInference, xhquant_init


VAE_DECODER_STAGES = (
    "post_quant",
    "conv_in",
    "mid_res0_norm1",
    "mid_res0_act1",
    "mid_res0_conv1",
    "mid_res0_norm2",
    "mid_res0_act2",
    "mid_res0_conv2",
    "mid_res0",
    "mid_attn",
    "mid_res1",
    "mid",
    "up0",
    "up0_res0",
    "up0_res1",
    "up0_res1_norm1",
    "up0_res1_conv1",
    "up0_res1_norm2",
    "up0_res1_conv2",
    "up0_res2",
    "up0_upsampler",
    "up0_shortcut",
    "up1",
    "up2",
    "up3",
    "head",
)

VAE_DECODER_CACHE_LOOP_STAGES = (
    "mid",
    "up0",
    "up0_res0",
    "up0_res1",
    "up0_res1_norm1",
    "up0_res1_conv1",
    "up0_res1_norm2",
    "up0_res1_conv2",
    "up0_res2",
    "up0_upsampler",
    "up0_shortcut",
    "up1",
    "up2",
    "up3",
    "head",
)

VAE_DECODER_SPLIT_STAGES = (
    "up0",
    "up1",
    "up2",
    "up3",
    "head",
)


def tensor_summary(tensor: torch.Tensor) -> dict[str, Any]:
    return {"shape": list(tensor.shape), "dtype": str(tensor.dtype), "device": str(tensor.device)}


def diff_summary(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    ref = reference.detach().cpu().float().numpy().reshape(-1)
    cand = candidate.detach().cpu().float().numpy().reshape(-1)
    diff = np.abs(ref - cand)
    denom = np.linalg.norm(ref) * np.linalg.norm(cand)
    cosine = float(np.dot(ref, cand) / denom) if denom != 0 else float("nan")
    return {
        "reference": tensor_summary(reference),
        "candidate": tensor_summary(candidate),
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "p95_abs_diff": float(np.percentile(diff, 95)),
        "cosine": cosine,
    }


def as_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (list, tuple)) and len(output) == 1 and isinstance(output[0], torch.Tensor):
        return output[0]
    raise TypeError(f"Unsupported HMONNX output type: {type(output)!r}")


def make_vae_decoder_latents(
    *,
    latent_frames: int,
    latent_height: int,
    latent_width: int,
    seed: int,
    input_mode: str = "random",
) -> torch.Tensor:
    shape = (1, 48, int(latent_frames), int(latent_height), int(latent_width))
    if input_mode == "zeros":
        return torch.zeros(shape, dtype=torch.float32)
    if input_mode != "random":
        raise ValueError(f"Unsupported input_mode {input_mode!r}; expected random or zeros")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    return torch.randn(shape, generator=generator, dtype=torch.float32)


class VaeDecoderStageWrapper(nn.Module):
    """Single-frame-friendly VAE decoder prefix stage wrapper."""

    def __init__(self, vae: AutoencoderKLWan, stage: str):
        super().__init__()
        if stage not in VAE_DECODER_STAGES:
            raise ValueError(f"Unsupported stage {stage!r}, expected one of {VAE_DECODER_STAGES}")
        self.vae = vae
        self.stage = stage

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        x = self.vae.post_quant_conv(latents)
        if self.stage == "post_quant":
            return x
        x = self.vae.decoder.conv_in(x)
        if self.stage == "conv_in":
            return x
        feat_idx = [0]
        resnet0 = self.vae.decoder.mid_block.resnets[0]
        shortcut = resnet0.conv_shortcut(x)
        x = resnet0.norm1(x)
        if self.stage == "mid_res0_norm1":
            return x
        x = resnet0.nonlinearity(x)
        if self.stage == "mid_res0_act1":
            return x
        x = resnet0.conv1(x)
        if self.stage == "mid_res0_conv1":
            return x
        x = resnet0.norm2(x)
        if self.stage == "mid_res0_norm2":
            return x
        x = resnet0.nonlinearity(x)
        if self.stage == "mid_res0_act2":
            return x
        x = resnet0.dropout(x)
        x = resnet0.conv2(x)
        if self.stage == "mid_res0_conv2":
            return x
        x = x + shortcut
        if self.stage == "mid_res0":
            return x
        for attn, resnet in zip(self.vae.decoder.mid_block.attentions, self.vae.decoder.mid_block.resnets[1:]):
            if attn is not None:
                x = attn(x)
            if self.stage == "mid_attn":
                return x
            x = resnet(x, feat_cache=None, feat_idx=feat_idx)
            if self.stage == "mid_res1":
                return x
        if self.stage == "mid":
            return x
        for index, up_block in enumerate(self.vae.decoder.up_blocks):
            x = up_block(x, first_chunk=True)
            if self.stage == f"up{index}":
                return x
        x = self.vae.decoder.norm_out(x)
        x = self.vae.decoder.nonlinearity(x)
        x = self.vae.decoder.conv_out(x)
        return torch.clamp(x, min=-1.0, max=1.0)


class VaeDecoderCacheLoopStageWrapper(nn.Module):
    """Decoder stage wrapper that follows AutoencoderKLWan.decode() per-frame cache semantics."""

    def __init__(self, vae: AutoencoderKLWan, stage: str):
        super().__init__()
        if stage not in VAE_DECODER_CACHE_LOOP_STAGES:
            raise ValueError(f"official cache loop supports only {VAE_DECODER_CACHE_LOOP_STAGES}, got {stage!r}")
        self.vae = vae
        self.stage = stage

    def _cached_conv(self, conv: nn.Module, x: torch.Tensor) -> torch.Tensor:
        idx = self.vae._conv_idx[0]
        cache_x = x[:, :, -CACHE_T:, :, :].clone()
        if cache_x.shape[2] < 2 and self.vae._feat_map[idx] is not None:
            cache_x = torch.cat([self.vae._feat_map[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
        x = conv(x, self.vae._feat_map[idx])
        self.vae._feat_map[idx] = cache_x
        self.vae._conv_idx[0] += 1
        return x

    def _forward_resnet_with_internal_stage(self, resnet: nn.Module, x: torch.Tensor, prefix: str) -> torch.Tensor:
        shortcut = resnet.conv_shortcut(x)
        x = resnet.norm1(x)
        if self.stage == f"{prefix}_norm1":
            return x
        x = resnet.nonlinearity(x)
        x = self._cached_conv(resnet.conv1, x)
        if self.stage == f"{prefix}_conv1":
            return x
        x = resnet.norm2(x)
        if self.stage == f"{prefix}_norm2":
            return x
        x = resnet.nonlinearity(x)
        x = resnet.dropout(x)
        x = self._cached_conv(resnet.conv2, x)
        if self.stage == f"{prefix}_conv2":
            return x
        return x + shortcut

    def _forward_frame(self, x: torch.Tensor, first_chunk: bool) -> torch.Tensor:
        x = self._cached_conv(self.vae.decoder.conv_in, x)
        x = self.vae.decoder.mid_block(x, feat_cache=self.vae._feat_map, feat_idx=self.vae._conv_idx)
        if self.stage == "mid":
            return x
        for index, up_block in enumerate(self.vae.decoder.up_blocks):
            x_copy = x.clone()
            for resnet_index, resnet in enumerate(up_block.resnets):
                internal_prefix = f"up{index}_res{resnet_index}"
                if self.stage.startswith(f"{internal_prefix}_"):
                    return self._forward_resnet_with_internal_stage(resnet, x, internal_prefix)
                x = resnet(x, feat_cache=self.vae._feat_map, feat_idx=self.vae._conv_idx)
                if self.stage == internal_prefix:
                    return x

            upsampler = getattr(up_block, "upsampler", None)
            if upsampler is not None:
                x = upsampler(x, feat_cache=self.vae._feat_map, feat_idx=self.vae._conv_idx)
                if self.stage == f"up{index}_upsampler":
                    return x

            upsamplers = getattr(up_block, "upsamplers", None)
            if upsamplers is not None:
                x = upsamplers[0](x, feat_cache=self.vae._feat_map, feat_idx=self.vae._conv_idx)
                if self.stage == f"up{index}_upsampler":
                    return x

            avg_shortcut = getattr(up_block, "avg_shortcut", None)
            if avg_shortcut is not None:
                x = x + avg_shortcut(x_copy, first_chunk=first_chunk)
                if self.stage == f"up{index}_shortcut":
                    return x

            if self.stage == f"up{index}":
                return x
        x = self.vae.decoder.norm_out(x)
        x = self.vae.decoder.nonlinearity(x)
        x = self._cached_conv(self.vae.decoder.conv_out, x)
        return torch.clamp(x, min=-1.0, max=1.0)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        self.vae.clear_cache()
        x = self.vae.post_quant_conv(latents)
        outputs = []
        for frame_idx in range(x.shape[2]):
            self.vae._conv_idx = [0]
            outputs.append(self._forward_frame(x[:, :, frame_idx : frame_idx + 1, :, :], first_chunk=frame_idx == 0))
        output = torch.cat(outputs, dim=2)
        self.vae.clear_cache()
        return output

class VaeDecoderSplitStageWrapper(VaeDecoderCacheLoopStageWrapper):
    """Run one VAE decoder boundary stage from an intermediate tensor.

    This is the runtime form used for realistic-size VAE decoder quantization:
    each HMONNX graph owns one up block or the final head, while the host passes
    the intermediate 5D tensor to the next graph.
    """

    def __init__(self, vae: AutoencoderKLWan, stage: str):
        if stage not in VAE_DECODER_SPLIT_STAGES:
            raise ValueError(f"split stage supports only {VAE_DECODER_SPLIT_STAGES}, got {stage!r}")
        super().__init__(vae, stage)

    def _forward_up_block(self, up_block: nn.Module, x: torch.Tensor, first_chunk: bool) -> torch.Tensor:
        x_copy = x.clone()
        for resnet in up_block.resnets:
            x = resnet(x, feat_cache=self.vae._feat_map, feat_idx=self.vae._conv_idx)

        upsampler = getattr(up_block, "upsampler", None)
        if upsampler is not None:
            x = upsampler(x, feat_cache=self.vae._feat_map, feat_idx=self.vae._conv_idx)

        upsamplers = getattr(up_block, "upsamplers", None)
        if upsamplers is not None:
            x = upsamplers[0](x, feat_cache=self.vae._feat_map, feat_idx=self.vae._conv_idx)

        avg_shortcut = getattr(up_block, "avg_shortcut", None)
        if avg_shortcut is not None:
            x = x + avg_shortcut(x_copy, first_chunk=first_chunk)
        return x

    def _forward_split_frame(self, x: torch.Tensor, first_chunk: bool) -> torch.Tensor:
        if self.stage.startswith("up"):
            block_index = int(self.stage.removeprefix("up"))
            return self._forward_up_block(self.vae.decoder.up_blocks[block_index], x, first_chunk)
        x = self.vae.decoder.norm_out(x)
        x = self.vae.decoder.nonlinearity(x)
        x = self._cached_conv(self.vae.decoder.conv_out, x)
        return torch.clamp(x, min=-1.0, max=1.0)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        self.vae.clear_cache()
        outputs = []
        for frame_idx in range(hidden_states.shape[2]):
            self.vae._conv_idx = [0]
            outputs.append(
                self._forward_split_frame(
                    hidden_states[:, :, frame_idx : frame_idx + 1, :, :],
                    first_chunk=frame_idx == 0,
                )
            )
        output = torch.cat(outputs, dim=2)
        self.vae.clear_cache()
        return output

@dataclass(frozen=True)
class VaeDecoderSplitHmonnxPaths:
    up0: Path
    up1: Path
    up2: Path
    up3: Path
    head: Path

    def all(self) -> list[Path]:
        return [self.up0, self.up1, self.up2, self.up3, self.head]

    def as_dict(self) -> dict[str, str]:
        return {stage: str(getattr(self, stage)) for stage in VAE_DECODER_SPLIT_STAGES}


def require_vae_decoder_split_paths(paths: VaeDecoderSplitHmonnxPaths) -> None:
    missing = [str(path) for path in paths.all() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing VAE decoder split HMONNX files: " + ", ".join(missing))


@dataclass(frozen=True)
class VaeDecoderFullHmonnxPaths:
    pre_mid: Path
    split: VaeDecoderSplitHmonnxPaths

    def all(self) -> list[Path]:
        return [self.pre_mid, *self.split.all()]

    def as_dict(self) -> dict[str, str]:
        return {"pre_mid": str(self.pre_mid), **self.split.as_dict()}


def require_vae_decoder_full_paths(paths: VaeDecoderFullHmonnxPaths) -> None:
    missing = [str(path) for path in paths.all() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing VAE decoder full-chain HMONNX files: " + ", ".join(missing))


def _as_hmonnx_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (list, tuple)) and len(output) == 1 and isinstance(output[0], torch.Tensor):
        return output[0]
    raise TypeError(f"Unsupported HMONNX output type: {type(output)!r}")


class VaeDecoderHmonnxSplitRuntime:
    """Host runtime for a fixed-shape VAE decoder split-chain.

    The host owns graph ordering and intermediate tensors. Each stage is a
    separate HMONNX graph, which avoids tracing/exporting the large full decoder
    prefix-to-head graph for realistic latent sizes.
    """

    def __init__(
        self,
        paths: VaeDecoderSplitHmonnxPaths,
        *,
        device: str = "cuda",
        input_dtype: str = "float16",
        debug: bool = False,
        init_xhquant: bool = True,
    ) -> None:
        require_vae_decoder_split_paths(paths)
        if init_xhquant:
            xhquant_init(None, debug=debug)
        self.paths = paths
        self.device = torch.device(device if torch.cuda.is_available() or not device.startswith("cuda") else "cpu")
        self.hmonnx_dtype = torch.float16 if input_dtype == "float16" else torch.float32
        self.sessions = {
            stage: HMONNXInference(str(getattr(paths, stage))).to(self.device)
            for stage in VAE_DECODER_SPLIT_STAGES
        }

    @torch.no_grad()
    def run_from_mid(self, mid_hidden_states: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs: dict[str, torch.Tensor] = {}
        hidden_states = mid_hidden_states
        for stage in VAE_DECODER_SPLIT_STAGES:
            hidden_states = _as_hmonnx_tensor(
                self.sessions[stage].forward(hidden_states.to(self.device, dtype=self.hmonnx_dtype))
            ).detach().cpu()
            outputs[stage] = hidden_states
        return outputs

    @torch.no_grad()
    def decode_from_mid(self, mid_hidden_states: torch.Tensor) -> torch.Tensor:
        return self.run_from_mid(mid_hidden_states)["head"]

class VaeDecoderFullHmonnxRuntime:
    """Host runtime for the full fixed-shape VAE decoder HMONNX chain."""

    def __init__(
        self,
        paths: VaeDecoderFullHmonnxPaths,
        *,
        device: str = "cuda",
        input_dtype: str = "float16",
        debug: bool = False,
    ) -> None:
        require_vae_decoder_full_paths(paths)
        xhquant_init(None, debug=debug)
        self.paths = paths
        self.device = torch.device(device if torch.cuda.is_available() or not device.startswith("cuda") else "cpu")
        self.hmonnx_dtype = torch.float16 if input_dtype == "float16" else torch.float32
        self.pre_mid_session = HMONNXInference(str(paths.pre_mid)).to(self.device)
        self.split_runtime = VaeDecoderHmonnxSplitRuntime(
            paths.split,
            device=device,
            input_dtype=input_dtype,
            debug=debug,
            init_xhquant=False,
        )

    @torch.no_grad()
    def run(self, latents: torch.Tensor) -> dict[str, torch.Tensor]:
        mid_hidden_states = _as_hmonnx_tensor(
            self.pre_mid_session.forward(latents.to(self.device, dtype=self.hmonnx_dtype))
        ).detach().cpu()
        outputs = {"pre_mid": mid_hidden_states, "mid": mid_hidden_states}
        outputs.update(self.split_runtime.run_from_mid(mid_hidden_states))
        return outputs

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.run(latents)["head"]
