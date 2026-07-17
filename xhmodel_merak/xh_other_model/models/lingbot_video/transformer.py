from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ._quant_utils import export_torch_component
from .transformer_wrapper import (
    LingBotVideoTransformerExportWrapper,
    install_export_geometry,
    install_fixed_conditioning,
    register_transformer_wrap_modules,
)


def export_transformer(
    *,
    model_dir: Path,
    output_dir: Path,
    target_device: str,
    component_cfg: dict[str, Any],
    geometry_cfg: dict[str, Any],
    exec_device: str,
) -> dict[str, Any]:
    from lingbot_video.scheduling_flow_unipc import FlowUniPCMultistepScheduler
    from lingbot_video.transformer_lingbot_video import LingBotVideoTransformer3DModel

    device = torch.device(exec_device)
    dtype = torch.float16
    transformer = LingBotVideoTransformer3DModel.from_pretrained(
        str(model_dir),
        subfolder="transformer",
        torch_dtype=dtype,
    ).eval()
    transformer.to(device)

    num_inference_steps = int(geometry_cfg.get("num_inference_steps", 40))
    shift = float(geometry_cfg.get("shift", 3.0))
    scheduler = FlowUniPCMultistepScheduler.from_pretrained(str(model_dir), subfolder="scheduler")
    scheduler.set_timesteps(num_inference_steps, device=device, shift=shift)
    timestep_values = _transformer_timestep_values(scheduler.timesteps, dtype)
    install_fixed_conditioning(transformer, timestep_values)

    height = int(geometry_cfg.get("height", 480))
    width = int(geometry_cfg.get("width", 832))
    num_frames = int(geometry_cfg.get("num_frames", 1))
    latent_frames = (num_frames - 1) // 4 + 1
    latent_height = height // 8
    latent_width = width // 8
    text_sequence_length = int(component_cfg.get("text_sequence_length", 2048))
    text_dim = int(transformer.config.text_dim)
    generator = torch.Generator(device=device).manual_seed(int(geometry_cfg.get("seed", 42)))
    hidden_states = torch.randn(
        1,
        int(transformer.config.in_channels),
        latent_frames,
        latent_height,
        latent_width,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    encoder_hidden_states = torch.randn(
        1,
        text_sequence_length,
        text_dim,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    rotary_cos, rotary_sin, current_input_length = LingBotVideoTransformerExportWrapper.build_rotary_inputs(
        transformer,
        hidden_states,
        valid_text_length=text_sequence_length,
        padded_text_length=text_sequence_length,
    )
    install_export_geometry(
        transformer,
        hidden_states.shape,
        padded_text_length=text_sequence_length,
    )
    timestep_index = torch.zeros(1, device=device, dtype=torch.int32)

    flash_attention_bits = _flash_attention_bits(component_cfg)
    register_transformer_wrap_modules(
        transformer,
        flash_attention_bits=flash_attention_bits,
    )
    wrapper = LingBotVideoTransformerExportWrapper(transformer).to(device).eval()
    output_file = (
        output_dir
        / "hmonnx"
        / (f"lingbot_video_transformer_{target_device}_{component_cfg.get('quant_type', 'w8a8h1_sefp')}.onnx")
    )
    meta = export_torch_component(
        model=wrapper,
        inputs=(
            hidden_states,
            encoder_hidden_states,
            timestep_index,
            rotary_cos,
            rotary_sin,
            current_input_length,
        ),
        input_names=(
            "hidden_states",
            "encoder_hidden_states",
            "timestep_index",
            "rotary_cos",
            "rotary_sin",
            "current_input_length",
        ),
        output_names=("sample",),
        output_file=output_file,
        target_device=target_device,
        component_cfg=component_cfg,
        exec_device=exec_device,
        frontend_type=str(component_cfg.get("frontend_type", "TorchFX")),
    )
    meta.update(
        {
            "hmonnx_file": str(Path(meta["hmonnx_file"]).relative_to(output_dir)),
            "calibration_inputs": str(Path(meta["calibration_inputs"]).relative_to(output_dir)),
            "timestep_values": timestep_values,
            "num_inference_steps": num_inference_steps,
            "shift": shift,
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "latent_shape": list(hidden_states.shape),
            "text_sequence_length": text_sequence_length,
            "text_dim": text_dim,
            "batch_size": 1,
            "attention_backend": "xh2a_flash_attention",
            "flash_attention_bits": flash_attention_bits,
            "flash_attention_value_scale": transformer.lingbot_flash_attention_value_scale,
            "calibration_note": (
                "The saved random tensors define the static graph shape. SEFP "
                "activation scaling is input-adaptive and does not collect dataset "
                "statistics from this sample."
            ),
        }
    )
    return meta


def _flash_attention_bits(component_cfg: dict[str, Any]) -> dict[str, int]:
    flash_cfg = dict(component_cfg.get("flash_attention", {}))
    if flash_cfg.get("enable", True) is not True:
        raise ValueError(
            "LingBot video profiles require FlashAttention so full 121-frame "
            "attention does not materialize a quadratic score tensor."
        )
    quant_type = str(component_cfg.get("quant_type", "w8a8h1_sefp"))
    default_bits = 16 if "a16" in quant_type.lower() else 8
    return {name: int(flash_cfg.get(name, default_bits)) for name in ("q_bits", "k_bits", "v_bits", "s_bits", "p_bits")}


def _transformer_timestep_values(
    scheduler_timesteps: torch.Tensor,
    transformer_dtype: torch.dtype,
) -> list[float]:
    values = []
    for timestep in scheduler_timesteps:
        sigma = timestep.float() / 1000.0
        sigma = sigma.to(transformer_dtype)
        values.append(float((sigma * 1000.0).float().item()))
    return values
