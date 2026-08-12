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

import os
from typing import Dict, Optional, Sequence, Tuple

import accelerate
import torch
import torch.nn as nn
import torch.nn.functional as F

from xhquant import nn as xhnn
from xhquant.nn.modules.moeblock import MoeBlock
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
    token_indices = torch.arange(sequence_length, dtype=torch.int32, device=next(transformer.parameters()).device)
    if "lingbot_valid_token_indices" in transformer._buffers:
        transformer._buffers["lingbot_valid_token_indices"] = token_indices
    else:
        transformer.register_buffer("lingbot_valid_token_indices", token_indices, persistent=False)
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
        symbols["sparse_moe"]: _LingBotVideoSparseMoeBlock,
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


def _move_parameter_to_meta(module: nn.Module, parameter_name: str) -> None:
    parameter = getattr(module, parameter_name)
    setattr(
        module,
        parameter_name,
        nn.Parameter(
            torch.empty_like(parameter.data, device="meta"),
            requires_grad=parameter.requires_grad,
        ),
    )


class _LingBotVideoSparseMoeBlock(DynamicModule):
    def _group_limited_topk(self, scores_for_choice: torch.Tensor) -> torch.Tensor:
        seq_len = scores_for_choice.shape[0]
        if self.n_group is None or self.n_group <= 1:
            return torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)[1]

        experts_per_group = self.num_experts // self.n_group
        grouped = scores_for_choice.view(seq_len, self.n_group, experts_per_group)
        group_scores = grouped.topk(2, dim=-1)[0].sum(dim=-1)
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = group_mask.unsqueeze(-1).repeat(1, 1, experts_per_group).reshape(seq_len, -1)
        if getattr(self, "router_choice_mode", "float") == "fixed_int32":
            masked = scores_for_choice.masked_fill(score_mask == 0, -65504)
        else:
            masked = scores_for_choice + (1.0 - score_mask) * -65504.0
        return torch.topk(masked, k=self.top_k, dim=-1, sorted=False)[1]

    def forward(
        self,
        hidden_states: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = hidden_states.shape[0]
        tokens = hidden_states.reshape(-1, self.hidden_size)
        router_precision = os.environ.get("LINGBOT_MOE_ROUTER_PRECISION", "fp32")
        router_choice_scale = float(os.environ.get("LINGBOT_MOE_ROUTER_CHOICE_SCALE", "1.0"))
        router_fixed_frac_bits = int(os.environ.get("LINGBOT_MOE_ROUTER_FIXED_FRAC_BITS", "24"))
        router_logits = self.gate(tokens).float()

        if router_precision in ("half_sigmoid", "half_sigmoid_add"):
            router_logits = router_logits.to(dtype=tokens.dtype)

        if self.score_func == "softmax":
            scores = F.softmax(router_logits, dim=-1)
        else:
            scores = router_logits.sigmoid()

        if router_precision == "half_add":
            scores_for_choice = scores.to(dtype=tokens.dtype) + self.router_bias.to(dtype=tokens.dtype).unsqueeze(0)
        elif router_precision == "half_sigmoid":
            scores_for_choice = scores.float() + self.router_bias.unsqueeze(0)
        elif router_precision == "half_sigmoid_add":
            scores_for_choice = scores + self.router_bias.to(dtype=tokens.dtype).unsqueeze(0)
        elif router_precision == "half_scaled_add":
            scores_for_choice = scores.to(dtype=tokens.dtype) * router_choice_scale
            scaled_router_bias = (self.router_bias * router_choice_scale).to(dtype=tokens.dtype)
            scores_for_choice = scores_for_choice + scaled_router_bias.unsqueeze(0)
        elif router_precision == "fixed_int32":
            if router_fixed_frac_bits != self.router_fixed_frac_bits:
                raise ValueError(
                    f"LINGBOT_MOE_ROUTER_FIXED_FRAC_BITS={router_fixed_frac_bits} does not match "
                    f"the exported module frac_bits={self.router_fixed_frac_bits}."
                )
            score_int = self.score_float_to_fixed_int32(scores)
            selected_experts = self.group_limited_topk_int32(score_int, self.router_bias_fixed_int32)
        else:
            scores_for_choice = scores + self.router_bias.unsqueeze(0)
        if router_precision != "fixed_int32":
            selected_experts = self._group_limited_topk(scores_for_choice)
        routing_weights = scores.to(dtype=tokens.dtype)
        moe_out = self.moeblock(
            tokens.reshape(batch_size, -1, self.hidden_size),
            routing_weights.reshape(batch_size, -1, self.num_experts),
            selected_experts=selected_experts.reshape(batch_size, -1, self.top_k),
            fast_mode=False,
        )
        moe_out = moe_out * self.route_scale
        if self.shared_experts is not None:
            shared_out = self.shared_experts(hidden_states)
            if padding_mask is not None:
                shared_out = shared_out * padding_mask.reshape(batch_size, -1, 1)
            moe_out = moe_out + shared_out
        if padding_mask is not None:
            moe_out = moe_out * padding_mask.reshape(batch_size, -1, 1)
        return moe_out

    def _setup(self, cfg: Optional[Dict] = None):
        del cfg
        self.device = self.router.weight.device
        router_weight = self.router.weight.data.clone().half()
        router_bias = self.router.e_score_correction_bias.data.clone().float()
        self.num_experts = int(self.router.num_experts)
        self.hidden_size = int(self.hidden_size)
        self.score_func = str(self.router.score_func)
        self.n_group = int(self.router.n_group) if self.router.n_group is not None else None
        self.topk_group = int(self.router.topk_group)
        self.top_k = int(self.router.top_k)
        self.route_scale = float(self.router.route_scale)
        self.router_choice_mode = os.environ.get("LINGBOT_MOE_ROUTER_PRECISION", "fp32")
        self.router_fixed_frac_bits = int(os.environ.get("LINGBOT_MOE_ROUTER_FIXED_FRAC_BITS", "24"))
        self.score_float_to_fixed_int32 = xhnn.FloatToFixedInt32(self.router_fixed_frac_bits)
        self.group_limited_topk_int32 = xhnn.GroupLimitedTopkInt32(
            self.num_experts,
            self.n_group or 1,
            self.topk_group,
            self.top_k,
        )
        router_bias_fixed_int32 = torch.round(router_bias.float() * float(1 << self.router_fixed_frac_bits)).to(
            torch.int32
        )
        self.register_buffer("router_bias", router_bias, persistent=True)
        self.register_buffer("router_bias_fixed_int32", router_bias_fixed_int32, persistent=True)
        self.gate = nn.Linear(
            router_weight.shape[1],
            router_weight.shape[0],
            bias=False,
            device=self.device,
            dtype=router_weight.dtype,
        )
        self.gate.weight.data.copy_(router_weight)
        self.moeblock = MoeBlock(
            "silu",
            self.top_k,
            bool(self.router.norm_topk_prob),
            topk_outside=True,
        )

        self.moeblock.expert_gate_proj_weight = nn.Parameter(self.experts.w1.data.contiguous().half())
        self.moeblock.expert_up_proj_weight = nn.Parameter(self.experts.w3.data.contiguous().half())
        self.moeblock.expert_down_proj_weight = nn.Parameter(self.experts.w2.data.contiguous().half())
        self.moeblock.expert_gate_proj_bias = None
        self.moeblock.expert_up_proj_bias = None
        self.moeblock.expert_down_proj_bias = None

        _move_parameter_to_meta(self.experts, "w1")
        _move_parameter_to_meta(self.experts, "w2")
        _move_parameter_to_meta(self.experts, "w3")
        del self.experts
        del self.router
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
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
        output = output.transpose(1, 2).reshape(batch_size, sequence_length, self.num_heads * self.head_dim)
        return self.to_out(output)

    @staticmethod
    def _apply_rotary(
        hidden_states: torch.Tensor,
        rotary_emb: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        rotary_cos, rotary_sin = rotary_emb
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        rotary_cos = rotary_cos.float()
        rotary_sin = rotary_sin.float()
        real = hidden_states[..., ::2]
        imaginary = hidden_states[..., 1::2]
        rotated = torch.stack((-imaginary, real), dim=-1).flatten(3)
        return (hidden_states * rotary_cos + rotated * rotary_sin).to(dtype=input_dtype)

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
        if packed_indices is not None or parallel_config is not None:
            raise NotImplementedError("Packed, MoE, and context-parallel execution are not exportable here.")

        token_mask = None
        if moe_padding_mask is not None:
            token_mask = moe_padding_mask.reshape(x.shape[0], x.shape[1], 1)
            x = x * token_mask

        modulation = temb6 + self.scale_shift_table.unsqueeze(0).float()
        shift_msa = self.mod_slices[0](modulation)
        scale_msa = self.mod_slices[1](modulation)
        gate_msa = torch.tanh(self.mod_slices[2](modulation))
        shift_mlp = self.mod_slices[3](modulation)
        scale_mlp = self.mod_slices[4](modulation)
        gate_mlp = torch.tanh(self.mod_slices[5](modulation))

        if token_mask is not None:
            shift_msa = shift_msa * token_mask
            gate_msa = gate_msa * token_mask
            shift_mlp = shift_mlp * token_mask
            gate_mlp = gate_mlp * token_mask

        attention_input = (self.norm1(x).float() * (1.0 + scale_msa) + shift_msa).to(dtype=x.dtype)
        if token_mask is not None:
            attention_input = attention_input * token_mask
        attention_output = self.attn(attention_input, rotary_emb, current_input_length)
        if token_mask is not None:
            attention_output = attention_output * token_mask
        x = (x.float() + gate_msa * self.norm_post_attn(attention_output).float()).to(dtype=x.dtype)
        if token_mask is not None:
            x = x * token_mask

        ffn_input = (self.norm2(x).float() * (1.0 + scale_mlp) + shift_mlp).to(dtype=x.dtype)
        if token_mask is not None:
            ffn_input = ffn_input * token_mask
        if moe_padding_mask is not None and hasattr(self.ffn, "moeblock"):
            ffn_output = self.ffn(ffn_input, padding_mask=moe_padding_mask)
        else:
            ffn_output = self.ffn(ffn_input)
        if token_mask is not None:
            ffn_output = ffn_output * token_mask
        x = (x.float() + gate_mlp * self.norm_post_ffn(ffn_output).float()).to(dtype=x.dtype)
        if token_mask is not None:
            x = x * token_mask
        return x

    def _setup(self):
        hidden_size = self.scale_shift_table.shape[-1] // 6
        self.scale_shift_table = nn.Parameter(
            self.scale_shift_table.data.half(),
            requires_grad=self.scale_shift_table.requires_grad,
        )
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
        valid_token_mask: torch.Tensor,
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
        valid_token_mask = valid_token_mask.reshape(batch_size, self.lingbot_export_sequence_length, 1)
        moe_padding_mask = valid_token_mask.reshape(
            batch_size,
            self.lingbot_export_sequence_length,
        )

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
        joint = joint * valid_token_mask

        temb6 = self.lingbot_fixed_temb6[timestep_index]
        temb6 = temb6.unsqueeze(1).repeat(1, self.lingbot_export_sequence_length, 1)
        rotary_emb = (rotary_cos, rotary_sin)
        for block in self.blocks:
            joint = block(joint, temb6, rotary_emb, current_input_length, moe_padding_mask=moe_padding_mask)
            joint = joint * valid_token_mask

        final_shift = self.lingbot_fixed_final_shift[timestep_index]
        final_scale = self.lingbot_fixed_final_scale[timestep_index]
        final_shift = final_shift.unsqueeze(1).repeat(1, self.lingbot_export_sequence_length, 1)
        final_scale = final_scale.unsqueeze(1).repeat(1, self.lingbot_export_sequence_length, 1)
        final_hidden = (self.norm_out(joint).float() * (1.0 + final_scale) + final_shift).to(dtype=joint.dtype)
        projected = self.proj_out(final_hidden)
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
        valid_token_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.transformer(
            hidden_states,
            encoder_hidden_states,
            timestep_index,
            rotary_cos,
            rotary_sin,
            current_input_length,
            valid_token_mask,
        )

    @staticmethod
    def build_rotary_inputs(
        transformer: nn.Module,
        hidden_states: torch.Tensor,
        valid_text_length: int,
        padded_text_length: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
        valid_token_indices = torch.arange(
            video_token_count + padded_text_length,
            device=hidden_states.device,
            dtype=torch.int32,
        )
        valid_token_mask = (valid_token_indices.unsqueeze(0) < current_input_length.reshape(1, 1)).unsqueeze(-1)
        return (
            rotary_cos.unsqueeze(0).unsqueeze(2).to(hidden_states.dtype),
            rotary_sin.unsqueeze(0).unsqueeze(2).to(hidden_states.dtype),
            current_input_length,
            valid_token_mask.to(hidden_states.dtype),
        )
