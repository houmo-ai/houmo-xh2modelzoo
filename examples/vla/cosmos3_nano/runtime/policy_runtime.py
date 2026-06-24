# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0

"""Reusable host/runtime wrappers for Cosmos3-Nano policy/action graphs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from runtime.generator_runtime import (
    OfficialCosmos3StackWrapper,
    resolve_text_special_tokens,
    tokenize_generator_prompt,
)


class OfficialCosmos3ActionDenoiserBackboneWrapper(nn.Module):
    """Action denoiser backbone ending before action_proj_out/action head."""

    def __init__(self, transformer: nn.Module, start_layer: int, num_layers: int, und_seq: int, action_seq: int):
        super().__init__()
        if not bool(transformer.config.action_gen):
            raise ValueError("Cosmos3 transformer was not configured with action_gen=True")
        self.action_proj_in = transformer.action_proj_in
        self.action_modality_embed = transformer.action_modality_embed
        self.time_embedder = transformer.time_embedder
        self.stack = OfficialCosmos3StackWrapper(transformer, start_layer, num_layers, und_seq, action_seq)
        self.timestep_scale = float(transformer.config.timestep_scale)
        self.und_seq = int(und_seq)
        self.action_seq = int(action_seq)

    def forward(
        self,
        und_hidden_states: torch.Tensor,
        noisy_actions: torch.Tensor,
        timesteps: torch.Tensor,
        position_ids: torch.Tensor,
        domain_id: torch.Tensor,
    ) -> torch.Tensor:
        if domain_id.ndim == 0 or domain_id.numel() == 1:
            per_token_domain_id = domain_id.reshape(1).expand(noisy_actions.shape[0])
        else:
            per_token_domain_id = domain_id.reshape(-1)
        action_hidden_states = self.action_proj_in(noisy_actions, per_token_domain_id)
        action_hidden_states = action_hidden_states + self.action_modality_embed.view(1, -1)

        scaled_timesteps = timesteps.reshape(-1).to(dtype=torch.float32) * self.timestep_scale
        if scaled_timesteps.shape[0] == 1:
            scaled_timesteps = scaled_timesteps.expand(self.action_seq)
        timestep_embeds = self.time_embedder(scaled_timesteps).to(dtype=action_hidden_states.dtype)
        action_hidden_states = action_hidden_states + timestep_embeds

        hidden_states = torch.cat([und_hidden_states, action_hidden_states], dim=0)
        _, action_output = self.stack(hidden_states, position_ids)
        return action_output


class OfficialCosmos3ActionDenoiserBoundaryWrapper(nn.Module):
    """Action denoiser step: action_proj_in + time/modality + MoT stack + action_proj_out.

    Scheduler integration, action de-normalization, raw-action unpadding, and safety
    clamps stay in host runtime. The action head is kept as a clear boundary so first
    pass quantization can leave it FP16.
    """

    def __init__(self, transformer: nn.Module, start_layer: int, num_layers: int, und_seq: int, action_seq: int):
        super().__init__()
        if not bool(transformer.config.action_gen):
            raise ValueError("Cosmos3 transformer was not configured with action_gen=True")
        self.backbone = OfficialCosmos3ActionDenoiserBackboneWrapper(
            transformer, start_layer=start_layer, num_layers=num_layers, und_seq=und_seq, action_seq=action_seq
        )
        self.action_proj_out = transformer.action_proj_out

    def forward(
        self,
        und_hidden_states: torch.Tensor,
        noisy_actions: torch.Tensor,
        timesteps: torch.Tensor,
        position_ids: torch.Tensor,
        domain_id: torch.Tensor,
    ) -> torch.Tensor:
        if domain_id.ndim == 0 or domain_id.numel() == 1:
            per_token_domain_id = domain_id.reshape(1).expand(noisy_actions.shape[0])
        else:
            per_token_domain_id = domain_id.reshape(-1)
        action_output = self.backbone(und_hidden_states, noisy_actions, timesteps, position_ids, domain_id)
        return self.action_proj_out(action_output, per_token_domain_id)


class OfficialCosmos3ActionDenoiserInitSegmentWrapper(nn.Module):
    """Initial policy segment: action embedding/projection plus a stack slice.

    The output is the full packed hidden sequence, not only action tokens, so later
    stack-only segments can continue from the same transformer state.
    """

    def __init__(self, transformer: nn.Module, start_layer: int, num_layers: int, und_seq: int, action_seq: int):
        super().__init__()
        if not bool(transformer.config.action_gen):
            raise ValueError("Cosmos3 transformer was not configured with action_gen=True")
        self.action_proj_in = transformer.action_proj_in
        self.action_modality_embed = transformer.action_modality_embed
        self.time_embedder = transformer.time_embedder
        self.stack = OfficialCosmos3StackWrapper(transformer, start_layer, num_layers, und_seq, action_seq)
        self.timestep_scale = float(transformer.config.timestep_scale)
        self.action_seq = int(action_seq)

    def forward(
        self,
        und_hidden_states: torch.Tensor,
        noisy_actions: torch.Tensor,
        timesteps: torch.Tensor,
        position_ids: torch.Tensor,
        domain_id: torch.Tensor,
    ) -> torch.Tensor:
        if domain_id.ndim == 0 or domain_id.numel() == 1:
            per_token_domain_id = domain_id.reshape(1).expand(noisy_actions.shape[0])
        else:
            per_token_domain_id = domain_id.reshape(-1)
        action_hidden_states = self.action_proj_in(noisy_actions, per_token_domain_id)
        action_hidden_states = action_hidden_states + self.action_modality_embed.view(1, -1)

        scaled_timesteps = timesteps.reshape(-1).to(dtype=torch.float32) * self.timestep_scale
        if scaled_timesteps.shape[0] == 1:
            scaled_timesteps = scaled_timesteps.expand(self.action_seq)
        timestep_embeds = self.time_embedder(scaled_timesteps).to(dtype=action_hidden_states.dtype)
        action_hidden_states = action_hidden_states + timestep_embeds

        packed_hidden_states = torch.cat([und_hidden_states, action_hidden_states], dim=0)
        und_output, action_output = self.stack(packed_hidden_states, position_ids)
        return torch.cat([und_output, action_output], dim=0)


class OfficialCosmos3ActionDenoiserStackSegmentWrapper(nn.Module):
    """Stack-only policy segment for continuing a packed hidden sequence."""

    def __init__(self, transformer: nn.Module, start_layer: int, num_layers: int, und_seq: int, action_seq: int):
        super().__init__()
        self.stack = OfficialCosmos3StackWrapper(transformer, start_layer, num_layers, und_seq, action_seq)

    def forward(self, packed_hidden_states: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        und_output, action_output = self.stack(packed_hidden_states, position_ids)
        return torch.cat([und_output, action_output], dim=0)


def make_policy_real_packed_action_boundary_inputs(
    *,
    transformer: nn.Module,
    model_root: str | Path,
    prompt: str,
    action_seq: int,
    raw_action_dim: int,
    domain_id: int,
    seed: int,
    dtype: torch.dtype,
    device: torch.device,
    timestep: float = 0.5,
    fps_action: float = 4.0,
) -> tuple[
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    list[str],
    list[str],
    dict[str, Any],
]:
    """Create official-packed host inputs for a fixed-shape action denoiser step."""
    from diffusers_cosmos3.sequence_packing import GenerationDataClean, SequencePlan, pack_input_sequence

    max_action_dim = int(transformer.config.max_action_dim)
    if raw_action_dim > max_action_dim:
        raise ValueError(f"raw_action_dim={raw_action_dim} exceeds max_action_dim={max_action_dim}")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    noisy_actions = torch.zeros((action_seq, max_action_dim), dtype=torch.float32)
    noisy_actions[:, :raw_action_dim] = torch.randn(
        (action_seq, raw_action_dim), generator=generator, dtype=torch.float32
    )
    noisy_actions = noisy_actions.to(device=device, dtype=dtype)

    text_ids = tokenize_generator_prompt(model_root, prompt)
    special_tokens = resolve_text_special_tokens(model_root)
    domain_tensor = torch.tensor([int(domain_id)], dtype=torch.long, device=device)
    raw_dim_tensor = torch.tensor(int(raw_action_dim), dtype=torch.long, device=device)
    gen_data = GenerationDataClean(
        batch_size=1,
        is_image_batch=False,
        x0_tokens_action=[noisy_actions],
        fps_action=torch.tensor([float(fps_action)], dtype=torch.float32),
        action_domain_id=[domain_tensor.cpu()],
        raw_action_dim=[raw_dim_tensor.cpu()],
    )
    packed_seq = pack_input_sequence(
        sequence_plans=[SequencePlan(has_text=True, has_action=True)],
        input_text_indexes=[text_ids],
        gen_data_clean=gen_data,
        input_timesteps=torch.tensor([float(timestep)], dtype=torch.float32),
        special_tokens=special_tokens,
        include_end_of_generation_token=False,
        position_embedding_type=str(transformer.config.position_embedding_type),
        unified_3d_mrope_reset_spatial_ids=bool(transformer.config.unified_3d_mrope_reset_spatial_ids),
        unified_3d_mrope_temporal_modality_margin=int(transformer.config.unified_3d_mrope_temporal_modality_margin),
        enable_fps_modulation=bool(transformer.config.enable_fps_modulation),
        base_fps=float(transformer.config.base_fps),
        temporal_compression_factor=4,
        video_temporal_causal=bool(transformer.config.video_temporal_causal),
        action_dim=max_action_dim,
    )
    if packed_seq.action is None:
        raise ValueError("official pack_input_sequence did not create action metadata")
    if int(packed_seq.action.mse_loss_indexes.numel()) != int(packed_seq.action.sequence_indexes.numel()):
        raise ValueError("conditioned actions are not supported by the fixed action boundary smoke yet")

    und_hidden_states = transformer.embed_tokens(packed_seq.text_ids.to(device)).to(dtype=dtype)
    timesteps = packed_seq.action.timesteps.to(device=device, dtype=torch.float32)
    position_ids = packed_seq.position_ids.to(device=device)
    metadata = {
        "model_root": str(Path(model_root)),
        "prompt": prompt,
        "text_token_count": int(packed_seq.text_ids.numel()),
        "text_ids": [int(item) for item in packed_seq.text_ids.tolist()],
        "und_seq": int(packed_seq.text_ids.numel()),
        "action_seq": int(packed_seq.action.sequence_indexes.numel()),
        "raw_action_dim": int(raw_action_dim),
        "max_action_dim": max_action_dim,
        "domain_id": int(domain_id),
        "sequence_length": int(packed_seq.sequence_length),
        "split_lens": list(packed_seq.split_lens),
        "attn_modes": list(packed_seq.attn_modes),
        "sample_lens": list(packed_seq.sample_lens),
        "position_ids_shape": list(position_ids.shape),
        "action_sequence_indexes": [int(item) for item in packed_seq.action.sequence_indexes.tolist()],
        "action_mse_loss_indexes": [int(item) for item in packed_seq.action.mse_loss_indexes.tolist()],
        "action_token_shapes": [list(item) for item in packed_seq.action.token_shapes],
        "special_tokens": special_tokens,
        "notes": [
            "Scheduler loop, action de-normalization, raw_action_dim unpadding, and safety clamp stay on host.",
            "First-pass quantization should keep action_proj_out/action head in FP16.",
        ],
    }
    return (
        und_hidden_states,
        noisy_actions,
        timesteps,
        position_ids,
        domain_tensor,
    ), ["und_hidden_states", "noisy_actions", "timesteps", "position_ids", "domain_id"], ["action_velocity"], metadata
