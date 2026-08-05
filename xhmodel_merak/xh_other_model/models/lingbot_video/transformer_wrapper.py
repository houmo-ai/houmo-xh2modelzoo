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

from typing import Dict, Optional, Sequence, Tuple

import accelerate
import torch
import torch.nn as nn

from xhquant import nn as xhnn
from xhquant.utils.registry.dynamic_module import DynamicModule


_FLASH_ATTENTION_ACCUMULATION_TOKEN_LIMIT = 4096


def flash_attention_value_scale(sequence_length: int) -> float:
    if sequence_length <= 0:
        raise ValueError(f"sequence_length must be positive, got {sequence_length}.")
    required_scale = max(
        1,
        (sequence_length + _FLASH_ATTENTION_ACCUMULATION_TOKEN_LIMIT - 1) // _FLASH_ATTENTION_ACCUMULATION_TOKEN_LIMIT,
    )
    return float(1 << (required_scale - 1).bit_length())


def _lingbot_symbols():
    try:
        from lingbot_video.transformer_lingbot_video import (
            LingBotVideoAttention,
            LingBotVideoBlock,
            LingBotVideoRMSNorm,
            LingBotVideoSparseMoeBlock,
            LingBotVideoTransformer3DModel,
            make_joint_position_ids,
        )
    except ImportError as error:
        raise ImportError(
            "Install the pinned LingBot Video source package described in the Merak example README."
        ) from error

    return {
        "attention": LingBotVideoAttention,
        "block": LingBotVideoBlock,
        "rms_norm": LingBotVideoRMSNorm,
        "sparse_moe": LingBotVideoSparseMoeBlock,
        "transformer": LingBotVideoTransformer3DModel,
        "make_joint_position_ids": make_joint_position_ids,
    }


def install_fixed_conditioning(
    transformer: nn.Module,
    timestep_values: Sequence[float],
) -> None:
    """Precompute LingBot's intentionally-FP32 timestep subgraph."""
    if not timestep_values:
        raise ValueError("timestep_values cannot be empty")
    parameter = next(transformer.time_embedder.parameters())
    timesteps = torch.tensor(list(timestep_values), device=parameter.device, dtype=torch.float32)
    with torch.no_grad():
        timestep_embeds = transformer.time_embedder(transformer.time_proj(timesteps))
        temb6 = transformer.time_modulation(timestep_embeds)
        final_shift, final_scale = transformer.norm_out_modulation(timestep_embeds).chunk(2, dim=-1)

    for name, value in {
        "lingbot_fixed_timestep_values": timesteps,
        "lingbot_fixed_temb6": temb6,
        "lingbot_fixed_final_shift": final_shift,
        "lingbot_fixed_final_scale": final_scale,
    }.items():
        value = value.detach()
        if name in transformer._buffers:
            transformer._buffers[name] = value
        else:
            transformer.register_buffer(name, value, persistent=False)


def install_export_geometry(
    transformer: nn.Module,
    hidden_state_shape: Sequence[int],
    padded_text_length: int,
) -> None:
    if len(hidden_state_shape) != 5:
        raise ValueError(f"Expected BCTHW hidden state shape, got {hidden_state_shape}")
    batch_size, channels, frames, height, width = map(int, hidden_state_shape)
    if batch_size != 1:
        raise ValueError("LingBot Merak export currently supports batch_size=1 only.")
    patch_frames, patch_height, patch_width = map(int, transformer.config.patch_size)
    if frames % patch_frames or height % patch_height or width % patch_width:
        raise ValueError(
            f"Hidden shape {hidden_state_shape} is not divisible by patch size {transformer.config.patch_size}."
        )
    grid_frames = frames // patch_frames
    grid_height = height // patch_height
    grid_width = width // patch_width
    video_token_count = grid_frames * grid_height * grid_width
    sequence_length = video_token_count + int(padded_text_length)
    value_scale = flash_attention_value_scale(sequence_length)
    geometry = {
        "lingbot_export_batch_size": batch_size,
        "lingbot_export_channels": channels,
        "lingbot_export_frames": frames,
        "lingbot_export_height": height,
        "lingbot_export_width": width,
        "lingbot_export_grid_frames": grid_frames,
        "lingbot_export_grid_height": grid_height,
        "lingbot_export_grid_width": grid_width,
        "lingbot_export_video_token_count": video_token_count,
        "lingbot_export_text_length": int(padded_text_length),
        "lingbot_export_sequence_length": sequence_length,
        "lingbot_flash_attention_value_scale": value_scale,
    }
    for name, value in geometry.items():
        setattr(transformer, name, value)
    for module in transformer.modules():
        if type(module).__name__ == "LingBotVideoAttention":
            module.lingbot_export_batch_size = batch_size
            module.lingbot_export_sequence_length = geometry["lingbot_export_sequence_length"]
            module.lingbot_flash_attention_value_scale = value_scale


def register_transformer_wrap_modules(
    transformer: nn.Module,
    *,
    flash_attention_bits: Dict[str, int],
) -> nn.Module:
    symbols = _lingbot_symbols()
    if int(getattr(transformer.config, "num_experts", 0)) != 0:
        raise NotImplementedError("The first Merak implementation supports LingBot Video Dense only.")
    if any(isinstance(module, symbols["sparse_moe"]) for module in transformer.modules()):
        raise NotImplementedError("Sparse MoE blocks are not supported by this exporter.")
    if not hasattr(transformer, "lingbot_export_sequence_length"):
        raise RuntimeError("Call install_export_geometry() before wrapping the transformer.")
    required_bits = {"q_bits", "k_bits", "v_bits", "s_bits", "p_bits"}
    if set(flash_attention_bits) != required_bits:
        raise ValueError(
            f"flash_attention_bits must define exactly {sorted(required_bits)}, got {sorted(flash_attention_bits)}."
        )
    invalid_bits = {name: bits for name, bits in flash_attention_bits.items() if bits not in (8, 16)}
    if invalid_bits:
        raise ValueError(f"LingBot FlashAttention bits must be 8 or 16, got {invalid_bits}.")

    # DynamicModule replaces the Diffusers class, including ConfigMixin's property.
    transformer.lingbot_patch_size = tuple(transformer.config.patch_size)
    transformer.lingbot_out_channels = int(transformer.config.out_channels)

    transformer = accelerate.hooks.remove_hook_from_module(transformer, recurse=True)
    wrapped_classes = {
        symbols["transformer"]: _LingBotVideoTransformer3DModel,
        symbols["block"]: _LingBotVideoBlock,
        symbols["attention"]: _LingBotVideoAttention,
        symbols["rms_norm"]: _LingBotVideoRMSNorm,
    }
    for _, module in list(transformer.named_modules()):
        dynamic_cls = wrapped_classes.get(type(module))
        if dynamic_cls is not None and not isinstance(module, DynamicModule):
            if dynamic_cls is _LingBotVideoAttention:
                module.lingbot_flash_attention_bits = dict(flash_attention_bits)
            dynamic_cls.convert(module)
    return transformer


class _LingBotVideoRMSNorm(DynamicModule):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.norm(hidden_states)

    def _setup(self, cfg: Optional[Dict] = None):
        del cfg
        self.norm = xhnn.RMSNorm(self.weight.shape[0], self.variance_epsilon)
        self.norm.weight = self.weight
        return self


class _LingBotVideoAttention(DynamicModule):
    def forward(
        self,
        x: torch.Tensor,
        rotary_emb: Tuple[torch.Tensor, torch.Tensor],
        current_input_length: Optional[torch.Tensor] = None,
        packed_indices=None,
        parallel_config=None,
    ) -> torch.Tensor:
        if packed_indices is not None or parallel_config is not None:
            raise NotImplementedError("Packed/context-parallel attention is outside this static HMONNX graph.")

        batch_size = self.lingbot_export_batch_size
        sequence_length = self.lingbot_export_sequence_length
        query = self.to_q(x).reshape(batch_size, sequence_length, self.num_heads, self.head_dim)
        key = self.to_k(x).reshape(batch_size, sequence_length, self.num_heads, self.head_dim)
        value = self.to_v(x).reshape(batch_size, sequence_length, self.num_heads, self.head_dim)
        value_scale = self.lingbot_flash_attention_value_scale
        value = value * (1.0 / value_scale)
        query = self._apply_rotary(self.norm_q(query), rotary_emb)
        key = self._apply_rotary(self.norm_k(key), rotary_emb)

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        output = self.flash_attn(
            query,
            key,
            value,
            current_input_length=current_input_length,
        )
        output = output.transpose(1, 2).contiguous()
        output = output * value_scale
        output = output.reshape(batch_size, sequence_length, self.num_heads * self.head_dim)
        return self.to_out(output.to(x.dtype))

    @staticmethod
    def _apply_rotary(
        hidden_states: torch.Tensor,
        rotary_emb: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        rotary_cos, rotary_sin = rotary_emb
        real = hidden_states[..., ::2]
        imaginary = hidden_states[..., 1::2]
        rotated = torch.stack((-imaginary, real), dim=-1).flatten(3)
        return hidden_states * rotary_cos + rotated * rotary_sin

    def _setup(self):
        self.attention_scale = self.head_dim**-0.5
        bits = self.lingbot_flash_attention_bits
        self.flash_attn = xhnn.FlashAttention(
            num_heads=self.num_heads,
            num_kv_heads=self.num_heads,
            is_causal=False,
            scale=self.attention_scale,
            q_bits=bits["q_bits"],
            k_bits=bits["k_bits"],
            v_bits=bits["v_bits"],
            s_bits=bits["s_bits"],
            p_bits=bits["p_bits"],
        )
        return self


class _LingBotVideoBlock(DynamicModule):
    def forward(
        self,
        x: torch.Tensor,
        temb6: torch.Tensor,
        rotary_emb: Tuple[torch.Tensor, torch.Tensor],
        current_input_length: Optional[torch.Tensor] = None,
        moe_padding_mask=None,
        packed_indices=None,
        parallel_config=None,
    ) -> torch.Tensor:
        if moe_padding_mask is not None or packed_indices is not None or parallel_config is not None:
            raise NotImplementedError("Packed, MoE, and context-parallel execution are not exportable here.")

        modulation = temb6 + self.scale_shift_table.unsqueeze(0)
        shift_msa = self.mod_slices[0](modulation)
        scale_msa = self.mod_slices[1](modulation)
        gate_msa = torch.tanh(self.mod_slices[2](modulation))
        shift_mlp = self.mod_slices[3](modulation)
        scale_mlp = self.mod_slices[4](modulation)
        gate_mlp = torch.tanh(self.mod_slices[5](modulation))

        bulk_dtype = self.attn.to_q.weight.dtype
        attention_input = (self.norm1(x) * (1.0 + scale_msa) + shift_msa).to(bulk_dtype)
        attention_output = self.attn(attention_input, rotary_emb, current_input_length)
        x = x + (gate_msa * self.norm_post_attn(attention_output)).to(x.dtype)

        ffn_input = (self.norm2(x) * (1.0 + scale_mlp) + shift_mlp).to(bulk_dtype)
        ffn_output = self.ffn(ffn_input)
        return x + (gate_mlp * self.norm_post_ffn(ffn_output)).to(x.dtype)

    def _setup(self):
        hidden_size = self.scale_shift_table.shape[-1] // 6
        self.mod_slices = nn.ModuleList(
            [
                xhnn.Slice(
                    [index * hidden_size],
                    [(index + 1) * hidden_size],
                    [2],
                    [1],
                )
                for index in range(6)
            ]
        )
        return self


class _LingBotVideoTransformer3DModel(DynamicModule):
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep_index: torch.Tensor,
        rotary_cos: torch.Tensor,
        rotary_sin: torch.Tensor,
        current_input_length: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = self.lingbot_export_batch_size
        channels = self.lingbot_export_channels
        frames = self.lingbot_export_frames
        height = self.lingbot_export_height
        width = self.lingbot_export_width
        patch_frames, patch_height, patch_width = self.lingbot_patch_size
        grid_frames = self.lingbot_export_grid_frames
        grid_height = self.lingbot_export_grid_height
        grid_width = self.lingbot_export_grid_width
        video_token_count = self.lingbot_export_video_token_count

        patches = hidden_states.reshape(
            batch_size,
            channels,
            grid_frames,
            patch_frames,
            grid_height,
            patch_height,
            grid_width,
            patch_width,
        )
        patches = patches.permute(0, 2, 4, 6, 3, 5, 7, 1).reshape(
            batch_size,
            video_token_count,
            patch_frames * patch_height * patch_width * channels,
        )
        video = self.patch_embedder(patches)
        text = self.text_embedder(encoder_hidden_states)
        joint = torch.cat((video, text), dim=1)

        temb6 = self.lingbot_fixed_temb6[timestep_index]
        temb6 = temb6.unsqueeze(1).expand(batch_size, self.lingbot_export_sequence_length, -1)
        rotary_emb = (rotary_cos, rotary_sin)
        for block in self.blocks:
            joint = block(joint, temb6, rotary_emb, current_input_length)

        final_shift = self.lingbot_fixed_final_shift[timestep_index]
        final_scale = self.lingbot_fixed_final_scale[timestep_index]
        final_shift = final_shift.unsqueeze(1).expand(batch_size, self.lingbot_export_sequence_length, -1)
        final_scale = final_scale.unsqueeze(1).expand(batch_size, self.lingbot_export_sequence_length, -1)
        final_hidden = self.norm_out(joint) * (1.0 + final_scale) + final_shift
        projected = self.proj_out(final_hidden.to(self.proj_out.weight.dtype))
        video = projected[:, :video_token_count]

        output_channels = self.lingbot_out_channels
        video = video.reshape(
            batch_size,
            grid_frames,
            grid_height,
            grid_width,
            patch_frames,
            patch_height,
            patch_width,
            output_channels,
        )
        return video.permute(0, 7, 1, 4, 2, 5, 3, 6).reshape(batch_size, output_channels, frames, height, width)

    def _setup(self):
        required = (
            "lingbot_fixed_temb6",
            "lingbot_fixed_final_shift",
            "lingbot_fixed_final_scale",
        )
        missing = [name for name in required if not hasattr(self, name)]
        if missing:
            raise RuntimeError(f"Call install_fixed_conditioning() before wrapping; missing {missing}.")
        return self


class LingBotVideoTransformerExportWrapper(nn.Module):
    def __init__(self, transformer: nn.Module):
        super().__init__()
        self.transformer = transformer

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep_index: torch.Tensor,
        rotary_cos: torch.Tensor,
        rotary_sin: torch.Tensor,
        current_input_length: torch.Tensor,
    ) -> torch.Tensor:
        return self.transformer(
            hidden_states,
            encoder_hidden_states,
            timestep_index,
            rotary_cos,
            rotary_sin,
            current_input_length,
        )

    @staticmethod
    def build_rotary_inputs(
        transformer: nn.Module,
        hidden_states: torch.Tensor,
        valid_text_length: int,
        padded_text_length: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if hidden_states.shape[0] != 1:
            raise ValueError("LingBot HMONNX export currently supports batch_size=1 only.")
        if valid_text_length > padded_text_length:
            raise ValueError(f"valid_text_length={valid_text_length} exceeds padded_text_length={padded_text_length}.")

        symbols = _lingbot_symbols()
        _, _, frames, height, width = hidden_states.shape
        patch_frames, patch_height, patch_width = transformer.config.patch_size
        grid_frames = frames // patch_frames
        grid_height = height // patch_height
        grid_width = width // patch_width
        video_token_count = grid_frames * grid_height * grid_width
        position_ids = symbols["make_joint_position_ids"](
            valid_text_length,
            grid_frames,
            grid_height,
            grid_width,
            hidden_states.device,
        )
        rotary = transformer.rope(position_ids)
        rotary_parts = torch.view_as_real(rotary)
        rotary_cos = rotary_parts[..., 0].repeat_interleave(2, dim=-1)
        rotary_sin = rotary_parts[..., 1].repeat_interleave(2, dim=-1)

        padding = padded_text_length - valid_text_length
        if padding:
            head_dim = rotary_cos.shape[-1]
            rotary_cos = torch.cat(
                (
                    rotary_cos,
                    torch.ones(
                        padding,
                        head_dim,
                        device=rotary_cos.device,
                        dtype=rotary_cos.dtype,
                    ),
                ),
                dim=0,
            )
            rotary_sin = torch.cat(
                (
                    rotary_sin,
                    torch.zeros(
                        padding,
                        head_dim,
                        device=rotary_sin.device,
                        dtype=rotary_sin.dtype,
                    ),
                ),
                dim=0,
            )

        current_input_length = torch.tensor(
            [video_token_count + valid_text_length],
            device=hidden_states.device,
            dtype=torch.int32,
        )
        return (
            rotary_cos.unsqueeze(0).unsqueeze(2).to(hidden_states.dtype),
            rotary_sin.unsqueeze(0).unsqueeze(2).to(hidden_states.dtype),
            current_input_length,
        )
