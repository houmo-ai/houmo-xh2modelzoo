# Copyright 2026 LingBot-Video Contributors
# Copyright 2026 HOUMO AI (modifications)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Source: https://github.com/Robbyant/lingbot-video
# See licenses/lingbot-video/LICENSE for the full license text.

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import onnx
import torch
import torch.nn as nn
from diffusers.models.autoencoders.autoencoder_kl_wan import CACHE_T

from xhquant.api import convert_onnx_to_hmonnx

from ._quant_utils import build_quant_config, require_xh2a


class WanVAEEncoderExportWrapper(nn.Module):
    def __init__(self, vae: nn.Module):
        super().__init__()
        self.vae = vae

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        # Return posterior parameters. Sampling remains deterministic host logic.
        return self.vae._encode(sample)


class WanVAEDecoderFirstFrameExportWrapper(nn.Module):
    """Initialize Wan's temporal decoder caches from the first latent frame."""

    def __init__(
        self,
        vae: nn.Module,
        cache_count: int,
        rep_cache_defaults: dict[int, torch.Tensor],
    ):
        super().__init__()
        self.vae = vae
        self.cache_count = int(cache_count)
        self.rep_cache_indices = tuple(sorted(rep_cache_defaults))
        for index, value in rep_cache_defaults.items():
            self.register_buffer(f"rep_cache_{index}", value, persistent=False)

    def forward(self, latents: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if latents.shape[2] != 1:
            raise ValueError("Wan stateful first-frame decoder expects one latent frame.")
        self.vae.clear_cache()
        x = self.vae.post_quant_conv(latents)
        self.vae._conv_idx = [0]
        sample = self.vae.decoder(
            x,
            feat_cache=self.vae._feat_map,
            feat_idx=self.vae._conv_idx,
            first_chunk=True,
        )
        cache_outputs = []
        if self.vae._conv_idx[0] != self.cache_count:
            raise RuntimeError(
                "Wan first-frame decoder consumed an unexpected number of caches: "
                f"expected={self.cache_count}, actual={self.vae._conv_idx[0]}."
            )
        for index, cache in enumerate(self.vae._feat_map[: self.cache_count]):
            if isinstance(cache, str):
                if index not in self.rep_cache_indices:
                    raise RuntimeError(f"Missing tensor template for Wan cache {index}={cache!r}.")
                # Keep the zero-valued Rep cache connected to a graph input.
                # xhquant requires every ONNX graph output to have a producer node.
                input_zero = latents.reshape(-1)[0] * 0.0
                cache = getattr(self, f"rep_cache_{index}").to(sample) + input_zero
            if not isinstance(cache, torch.Tensor):
                raise TypeError(f"Wan decoder cache {index} is not a tensor: {type(cache)!r}.")
            if cache.shape[2] > CACHE_T:
                cache = cache[:, :, -CACHE_T:]
            if cache.shape[2] < CACHE_T:
                padding = torch.zeros(
                    cache.shape[0],
                    cache.shape[1],
                    CACHE_T - cache.shape[2],
                    cache.shape[3],
                    cache.shape[4],
                    device=cache.device,
                    dtype=cache.dtype,
                )
                cache = torch.cat((padding, cache), dim=2)
            cache_outputs.append(cache)
        self.vae.clear_cache()
        return (torch.clamp(sample, min=-1.0, max=1.0), *cache_outputs)


class WanVAEDecoderNextFrameExportWrapper(nn.Module):
    """Decode one later latent frame while carrying fixed-shape temporal caches."""

    def __init__(self, vae: nn.Module, cache_count: int, internal_cache_count: int):
        super().__init__()
        self.vae = vae
        self.cache_count = int(cache_count)
        self.internal_cache_count = int(internal_cache_count)

    def forward(self, latents: torch.Tensor, *cache_inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if latents.shape[2] != 1:
            raise ValueError("Wan stateful next-frame decoder expects one latent frame.")
        if len(cache_inputs) != self.cache_count:
            raise ValueError(f"Expected {self.cache_count} Wan decoder caches, got {len(cache_inputs)}.")
        self.vae.clear_cache()
        self.vae._feat_map = list(cache_inputs) + [None] * (self.internal_cache_count - self.cache_count)
        self.vae._conv_idx = [0]
        x = self.vae.post_quant_conv(latents)
        sample = self.vae.decoder(
            x,
            feat_cache=self.vae._feat_map,
            feat_idx=self.vae._conv_idx,
            first_chunk=False,
        )
        if self.vae._conv_idx[0] != self.cache_count:
            raise RuntimeError(
                "Wan next-frame decoder consumed an unexpected number of caches: "
                f"expected={self.cache_count}, actual={self.vae._conv_idx[0]}."
            )
        cache_outputs = tuple(self.vae._feat_map[: self.cache_count])
        if not all(isinstance(cache, torch.Tensor) for cache in cache_outputs):
            raise TypeError("Wan next-frame decoder produced a non-tensor cache.")
        self.vae.clear_cache()
        return (torch.clamp(sample, min=-1.0, max=1.0), *cache_outputs)


class WanCausalConv3dAs2d(nn.Module):
    """Exact static-time decomposition of Wan causal Conv3d into Conv2d calls."""

    def __init__(self, source: nn.Conv3d):
        super().__init__()
        kernel_t, kernel_h, kernel_w = source.kernel_size
        stride_t, stride_h, stride_w = source.stride
        dilation_t, dilation_h, dilation_w = source.dilation
        padding = tuple(int(value) for value in source._padding)
        if padding[0] != padding[1] or padding[2] != padding[3]:
            raise ValueError(f"Wan VAE Conv3d decomposition requires symmetric spatial padding, got {padding[:4]}.")

        self.temporal_stride = int(stride_t)
        self.temporal_dilation = int(dilation_t)
        self.temporal_padding_front = padding[4]
        self.temporal_padding_back = padding[5]
        self.frame_convs = nn.ModuleList()
        for kernel_index in range(kernel_t):
            # Causal convolution always includes the final temporal tap. Keeping
            # bias there applies it exactly once to every output frame.
            has_bias = source.bias is not None and kernel_index == kernel_t - 1
            frame_conv = nn.Conv2d(
                source.in_channels,
                source.out_channels,
                kernel_size=(kernel_h, kernel_w),
                stride=(stride_h, stride_w),
                padding=(padding[2], padding[0]),
                dilation=(dilation_h, dilation_w),
                groups=source.groups,
                bias=has_bias,
                device=source.weight.device,
                dtype=source.weight.dtype,
            )
            with torch.no_grad():
                frame_conv.weight.copy_(source.weight[:, :, kernel_index])
                if has_bias:
                    frame_conv.bias.copy_(source.bias)
            self.frame_convs.append(frame_conv)

    def forward(self, x: torch.Tensor, cache_x: torch.Tensor | None = None) -> torch.Tensor:
        temporal_padding_front = self.temporal_padding_front
        temporal_padding_back = self.temporal_padding_back
        if cache_x is not None and temporal_padding_front > 0:
            cache_x = cache_x.to(x.device)
            x = torch.cat((cache_x, x), dim=2)
            temporal_padding_front -= int(cache_x.shape[2])

        # Match torch.nn.functional.pad's negative-padding crop semantics.
        if temporal_padding_front < 0:
            x = x[:, :, -temporal_padding_front:]
            temporal_padding_front = 0
        if temporal_padding_back < 0:
            x = x[:, :, :temporal_padding_back]
            temporal_padding_back = 0

        input_frames = int(x.shape[2])
        kernel_extent = self.temporal_dilation * (len(self.frame_convs) - 1) + 1
        output_frames = (
            input_frames + temporal_padding_front + temporal_padding_back - kernel_extent
        ) // self.temporal_stride + 1
        if output_frames <= 0:
            raise ValueError(
                "Wan VAE Conv3d decomposition produced no temporal output: "
                f"input_frames={input_frames}, kernel_extent={kernel_extent}."
            )

        outputs = []
        for output_index in range(output_frames):
            terms = []
            for kernel_index, frame_conv in enumerate(self.frame_convs):
                input_index = (
                    output_index * self.temporal_stride - temporal_padding_front + kernel_index * self.temporal_dilation
                )
                if 0 <= input_index < input_frames:
                    terms.append(frame_conv(x[:, :, input_index]))
            if not terms:
                raise RuntimeError(f"No input contributes to Wan VAE output frame {output_index}.")
            output = terms[0]
            for term in terms[1:]:
                output = output + term
            outputs.append(output.unsqueeze(2))
        return torch.cat(outputs, dim=2)


def decompose_wan_causal_conv3d(model: nn.Module) -> int:
    """Replace every Wan causal Conv3d with an HMONNX-compatible 2D form."""
    from diffusers.models.autoencoders.autoencoder_kl_wan import WanCausalConv3d

    replacements = [
        (name, module) for name, module in model.named_modules() if name and isinstance(module, WanCausalConv3d)
    ]
    for name, module in replacements:
        model.set_submodule(name, WanCausalConv3dAs2d(module))
    return len(replacements)


def replace_wan_nearest_exact_upsampling(model: nn.Module) -> int:
    """Use the ONNX-supported equivalent for Wan's fixed 2x upsampling."""
    replacements = 0
    for module in model.modules():
        if isinstance(module, nn.Upsample) and module.mode == "nearest-exact":
            scale_factor = module.scale_factor
            if scale_factor not in (2.0, (2.0, 2.0)):
                raise ValueError(
                    f"nearest-exact can only be replaced for fixed 2x Wan upsampling, got scale_factor={scale_factor}."
                )
            module.mode = "nearest"
            replacements += 1
    return replacements


@torch.no_grad()
def _discover_decoder_cache_layout(
    vae: nn.Module, latent_pair: torch.Tensor
) -> tuple[int, int, dict[int, torch.Tensor]]:
    """Discover active decoder caches and materialize ``Rep`` as zero tensors."""
    if latent_pair.shape[2] < 2:
        raise ValueError("Cache discovery requires at least two latent frames.")
    post_quant = vae.post_quant_conv
    if not isinstance(post_quant, WanCausalConv3dAs2d) or len(post_quant.frame_convs) != 1:
        raise ValueError("Stateful Wan decoder requires a frame-local post_quant_conv.")

    vae.clear_cache()
    projected = post_quant(latent_pair[:, :, :2])
    vae._conv_idx = [0]
    vae.decoder(
        projected[:, :, :1],
        feat_cache=vae._feat_map,
        feat_idx=vae._conv_idx,
        first_chunk=True,
    )
    cache_count = int(vae._conv_idx[0])
    internal_cache_count = len(vae._feat_map)
    if cache_count <= 0 or cache_count > internal_cache_count:
        raise RuntimeError(f"Invalid Wan decoder cache layout: active={cache_count}, allocated={internal_cache_count}.")
    rep_indices = [index for index, cache in enumerate(vae._feat_map[:cache_count]) if isinstance(cache, str)]
    vae._conv_idx = [0]
    vae.decoder(
        projected[:, :, 1:2],
        feat_cache=vae._feat_map,
        feat_idx=vae._conv_idx,
        first_chunk=False,
    )
    if int(vae._conv_idx[0]) != cache_count:
        raise RuntimeError(
            "Wan decoder changed its cache count between first and later frames: "
            f"first={cache_count}, next={vae._conv_idx[0]}."
        )
    defaults = {}
    for index in rep_indices:
        cache = vae._feat_map[index]
        if not isinstance(cache, torch.Tensor) or cache.shape[2] != CACHE_T:
            raise RuntimeError(f"Unable to materialize Wan Rep cache {index}; got {type(cache)!r}.")
        defaults[index] = torch.zeros_like(cache)
    vae.clear_cache()
    return cache_count, internal_cache_count, defaults


def export_vae_components(
    *,
    model_dir: Path,
    output_dir: Path,
    target_device: str,
    encoder_cfg: dict[str, Any],
    decoder_cfg: dict[str, Any],
    geometry_cfg: dict[str, Any],
    exec_device: str,
) -> dict[str, Any]:
    from diffusers import AutoencoderKLWan

    device = torch.device(exec_device)
    vae = AutoencoderKLWan.from_pretrained(str(model_dir), subfolder="vae", torch_dtype=torch.float16).eval()
    vae.disable_tiling()
    decomposed_conv_count = decompose_wan_causal_conv3d(vae)
    if decomposed_conv_count == 0:
        raise RuntimeError("LingBot Wan VAE contains no causal Conv3d modules.")
    upsample_replacement_count = replace_wan_nearest_exact_upsampling(vae)
    vae.to(device)

    height = int(geometry_cfg.get("height", 480))
    width = int(geometry_cfg.get("width", 832))
    num_frames = int(geometry_cfg.get("num_frames", 1))
    encoder_frames = int(encoder_cfg.get("num_frames", 1))
    if encoder_frames <= 0 or (encoder_frames != 1 and (encoder_frames - 1) % 4):
        raise ValueError(f"VAE encoder num_frames must be 1 or 4n+1; got {encoder_frames}.")
    latent_frames = (num_frames - 1) // 4 + 1
    latent_height = height // 8
    latent_width = width // 8
    generator = torch.Generator(device=device).manual_seed(int(geometry_cfg.get("seed", 42)))

    result: dict[str, Any] = {
        "latents_mean": list(vae.config.latents_mean),
        "latents_std": list(vae.config.latents_std),
        "height": height,
        "width": width,
        "num_frames": num_frames,
        "causal_conv3d_decomposition": {
            "type": "static_temporal_conv2d_sum",
            "module_count": decomposed_conv_count,
        },
        "nearest_exact_upsample_replacements": upsample_replacement_count,
    }
    if bool(encoder_cfg.get("enabled", True)):
        encoder_input = torch.randn(
            1,
            3,
            encoder_frames,
            height,
            width,
            generator=generator,
            device=device,
            dtype=torch.float16,
        )
        result["encoder_frames"] = encoder_frames
        result["encoder"] = _export_onnx_quant_component(
            model=WanVAEEncoderExportWrapper(vae).eval(),
            inputs=(encoder_input,),
            input_names=("sample",),
            output_names=("posterior_parameters",),
            output_file=output_dir
            / "encoder"
            / "hmonnx"
            / (f"lingbot_video_vae_encoder_{target_device}_{encoder_cfg.get('quant_type', 'w8a8h1_sefp')}.onnx"),
            output_root=output_dir,
            target_device=target_device,
            component_cfg=encoder_cfg,
        )

    if bool(decoder_cfg.get("enabled", True)):
        decoder_input = torch.randn(
            1,
            int(vae.config.z_dim),
            latent_frames,
            latent_height,
            latent_width,
            generator=generator,
            device=device,
            dtype=torch.float16,
        )
        discovery_input = decoder_input
        if latent_frames == 1:
            discovery_input = torch.cat((decoder_input, decoder_input), dim=2)
        cache_count, internal_cache_count, rep_cache_defaults = _discover_decoder_cache_layout(vae, discovery_input)
        first_wrapper = WanVAEDecoderFirstFrameExportWrapper(vae, cache_count, rep_cache_defaults).eval()
        first_input = decoder_input[:, :, :1].contiguous()
        with torch.no_grad():
            first_outputs = first_wrapper(first_input)
        first_outputs = tuple(first_outputs)
        if len(first_outputs) - 1 != cache_count:
            raise RuntimeError(
                f"Wan decoder cache count mismatch: wrapper={len(first_outputs) - 1}, discovered={cache_count}."
            )
        cache_names = tuple(f"cache_{index}" for index in range(cache_count))
        first_meta = _export_onnx_quant_component(
            model=first_wrapper,
            inputs=(first_input,),
            input_names=("latents",),
            output_names=("sample", *tuple(f"{name}_out" for name in cache_names)),
            output_file=output_dir
            / "decoder"
            / "first"
            / "hmonnx"
            / (f"lingbot_video_vae_decoder_first_{target_device}_{decoder_cfg.get('quant_type', 'w8a8h1_sefp')}.onnx"),
            output_root=output_dir,
            target_device=target_device,
            component_cfg=decoder_cfg,
        )
        next_wrapper = WanVAEDecoderNextFrameExportWrapper(vae, cache_count, internal_cache_count).eval()
        next_inputs = (
            discovery_input[:, :, 1:2].contiguous(),
            *tuple(cache.detach() for cache in first_outputs[1:]),
        )
        with torch.no_grad():
            next_outputs = tuple(next_wrapper(*next_inputs))
        if int(next_outputs[0].shape[2]) != 4:
            raise RuntimeError(f"Wan next-frame decoder must emit four video frames, got {next_outputs[0].shape[2]}.")
        next_meta = _export_onnx_quant_component(
            model=next_wrapper,
            inputs=next_inputs,
            input_names=("latents", *tuple(f"{name}_in" for name in cache_names)),
            output_names=("sample", *tuple(f"{name}_out" for name in cache_names)),
            output_file=output_dir
            / "decoder"
            / "next"
            / "hmonnx"
            / (f"lingbot_video_vae_decoder_next_{target_device}_{decoder_cfg.get('quant_type', 'w8a8h1_sefp')}.onnx"),
            output_root=output_dir,
            target_device=target_device,
            component_cfg=decoder_cfg,
        )
        result["decoder"] = {
            "type": "stateful_temporal_cache",
            "variable_length": True,
            "profile_latent_frames": latent_frames,
            "profile_sample_frames": num_frames,
            "cache_count": cache_count,
            "internal_cache_count": internal_cache_count,
            "cache_shapes": [list(cache.shape) for cache in first_outputs[1:]],
            "first_sample_frames": int(first_outputs[0].shape[2]),
            "next_sample_frames": int(next_outputs[0].shape[2]),
            "first": first_meta,
            "next": next_meta,
        }
    return result


def _export_onnx_quant_component(
    *,
    model: nn.Module,
    inputs: tuple[torch.Tensor, ...],
    input_names: tuple[str, ...],
    output_names: tuple[str, ...],
    output_file: Path,
    output_root: Path,
    target_device: str,
    component_cfg: dict[str, Any],
) -> dict[str, Any]:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    input_shape_tag = "x".join(str(dim) for dim in inputs[0].shape)
    float_onnx_file = output_file.parent.parent / f"float_model_{input_shape_tag}.onnx"
    if not float_onnx_file.exists():
        with tempfile.TemporaryDirectory() as temp_dir:
            temporary_file = Path(temp_dir) / "model.onnx"
            torch.onnx.export(
                model,
                inputs,
                str(temporary_file),
                export_params=True,
                opset_version=18,
                do_constant_folding=True,
                input_names=list(input_names),
                output_names=list(output_names),
            )
            onnx_model = onnx.load(str(temporary_file), load_external_data=True)
            onnx.save(
                onnx_model,
                str(float_onnx_file),
                save_as_external_data=True,
                all_tensors_to_one_file=True,
                location=f"{float_onnx_file.stem}_external_data",
            )

    convert_onnx_to_hmonnx(
        str(float_onnx_file),
        list(inputs),
        require_xh2a(target_device),
        str(output_file),
        quant_config=build_quant_config(target_device, component_cfg),
        input_names=list(input_names),
        output_names=list(output_names),
    )
    calibration_file = output_file.parent / "calibration_inputs.pt"
    torch.save([tensor.detach().cpu() for tensor in inputs], calibration_file)
    return {
        "float_onnx_file": str(float_onnx_file.relative_to(output_root)),
        "hmonnx_file": str(output_file.relative_to(output_root)),
        "quant_type": str(component_cfg.get("quant_type", "w8a8h1_sefp")),
        "input_names": list(input_names),
        "output_names": list(output_names),
        "input_shapes": [list(tensor.shape) for tensor in inputs],
        "input_dtypes": [str(tensor.dtype).removeprefix("torch.") for tensor in inputs],
        "calibration_inputs": str(calibration_file.relative_to(output_root)),
        "calibration_method": "shape_sample_dynamic_sefp_no_dataset_statistics",
    }
