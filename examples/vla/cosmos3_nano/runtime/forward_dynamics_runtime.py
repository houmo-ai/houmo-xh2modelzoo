# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0

"""Reusable host/runtime wrappers for Cosmos3-Nano forward dynamics graphs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import json
import torch
import torch.nn as nn

from runtime.generator_runtime import (
    OfficialCosmos3StackWrapper,
    patchify_latents,
    resolve_text_special_tokens,
    tokenize_generator_prompt,
    unpatchify_latents,
)


def load_forward_dynamics_action_chunk(path: str | Path, *, chunk_index: int = 0) -> tuple[str, torch.Tensor, dict[str, Any]]:
    """Load one normalized action chunk from a Cosmos3 forward-dynamics asset JSON."""
    action_path = Path(path)
    payload = json.loads(action_path.read_text(encoding="utf-8"))
    chunks = payload.get("action_chunks")
    if not isinstance(chunks, list) or not chunks:
        raise ValueError(f"No action_chunks found in {action_path}")
    if chunk_index < 0 or chunk_index >= len(chunks):
        raise IndexError(f"chunk_index={chunk_index} is out of range for {len(chunks)} chunks")
    actions = torch.tensor(chunks[chunk_index], dtype=torch.float32)
    prompt = str(payload.get("prompt") or "Pickup items in the supermarket")
    metadata = {
        "action_json": str(action_path),
        "chunk_index": int(chunk_index),
        "num_chunks": int(len(chunks)),
        "prompt": prompt,
        "domain_name": payload.get("domain_name"),
        "view_point": payload.get("view_point"),
        "fps": payload.get("fps"),
        "action_chunk_size": payload.get("action_chunk_size"),
        "action_shape_per_chunk": payload.get("action_shape_per_chunk"),
        "description": payload.get("description"),
    }
    return prompt, actions, metadata


class OfficialCosmos3ForwardDynamicsBoundaryWrapper(nn.Module):
    """Action-conditioned future-observation denoiser boundary.

    Forward dynamics reuses the shared Cosmos3 omni transformer. The generation
    segment is packed as future vision latent tokens followed by action tokens;
    only the future vision output is projected back to latent velocity.
    """

    def __init__(
        self,
        transformer: nn.Module,
        start_layer: int,
        num_layers: int,
        und_seq: int,
        vision_seq: int,
        action_seq: int,
        frames: int,
        latent_height: int,
        latent_width: int,
    ):
        super().__init__()
        if not bool(transformer.config.action_gen):
            raise ValueError("Cosmos3 transformer was not configured with action_gen=True")
        self.proj_in = transformer.proj_in
        self.proj_out = transformer.proj_out
        self.action_proj_in = transformer.action_proj_in
        self.action_modality_embed = transformer.action_modality_embed
        self.time_embedder = transformer.time_embedder
        self.stack = OfficialCosmos3StackWrapper(
            transformer,
            start_layer=start_layer,
            num_layers=num_layers,
            und_seq=und_seq,
            gen_seq=int(vision_seq) + int(action_seq),
        )
        self.timestep_scale = float(transformer.config.timestep_scale)
        self.latent_patch_size = int(transformer.config.latent_patch_size)
        self.latent_channels = int(transformer.config.latent_channel)
        self.vision_seq = int(vision_seq)
        self.action_seq = int(action_seq)
        self.frames = int(frames)
        self.latent_height = int(latent_height)
        self.latent_width = int(latent_width)

    def _time_embedding(self, timesteps: torch.Tensor, seq_len: int, dtype: torch.dtype) -> torch.Tensor:
        # Keep embedding computation in the same dtype as downstream hidden states.
        t = timesteps.reshape(-1).to(dtype=dtype)
        t = t * self.timestep_scale
        half_dim = self.time_embedder.frequency_embedding_size // 2
        freqs = torch.exp(
            -torch.log(torch.tensor(10000.0, dtype=dtype, device=t.device))
            * torch.arange(half_dim, dtype=dtype, device=t.device)
            / float(half_dim)
        )
        args = t[:, None] * freqs[None, :]
        t_freq = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.time_embedder.frequency_embedding_size % 2 == 1:
            t_freq = torch.cat([t_freq, torch.zeros_like(t_freq[:, :1])], dim=-1)
        if t_freq.shape[0] == 1:
            t_freq = t_freq.expand(seq_len, -1)
        else:
            t_freq = t_freq.view(-1, t_freq.shape[-1])
        if int(t_freq.shape[0]) != int(seq_len):
            raise ValueError(f"Expected {seq_len} timestep values, got {t_freq.shape[0]}")
        return self.time_embedder.linear_2(self.time_embedder.act(self.time_embedder.linear_1(t_freq))).to(dtype=dtype)

    def forward(
        self,
        und_hidden_states: torch.Tensor,
        noisy_future_latents: torch.Tensor,
        actions: torch.Tensor,
        vision_timesteps: torch.Tensor,
        action_timesteps: torch.Tensor,
        position_ids: torch.Tensor,
        domain_id: torch.Tensor,
    ) -> torch.Tensor:
        latent_patches = patchify_latents(noisy_future_latents, self.latent_patch_size)
        vision_hidden_states = self.proj_in(latent_patches)
        vision_hidden_states = vision_hidden_states + self._time_embedding(
            vision_timesteps,
            self.vision_seq,
            vision_hidden_states.dtype,
        )

        if domain_id.ndim == 0 or domain_id.numel() == 1:
            per_token_domain_id = domain_id.reshape(1).expand(actions.shape[0])
        else:
            per_token_domain_id = domain_id.reshape(-1)
        action_hidden_states = self.action_proj_in(actions, per_token_domain_id)
        action_hidden_states = action_hidden_states + self.action_modality_embed.view(1, -1)
        action_hidden_states = action_hidden_states + self._time_embedding(
            action_timesteps,
            self.action_seq,
            action_hidden_states.dtype,
        )

        hidden_states = torch.cat([und_hidden_states, vision_hidden_states, action_hidden_states], dim=0)
        _, gen_output = self.stack(hidden_states, position_ids)
        vision_output = gen_output[: self.vision_seq]
        pred_patches = self.proj_out(vision_output)
        return unpatchify_latents(
            pred_patches,
            latent_channels=self.latent_channels,
            frames=self.frames,
            latent_height=self.latent_height,
            latent_width=self.latent_width,
            latent_patch_size=self.latent_patch_size,
        )


class OfficialCosmos3ForwardDynamicsInitSegmentWrapper(nn.Module):
    """Initial FD segment: input projection/embeddings plus a stack slice.

    The output is the full packed hidden sequence. Later stack-only segments can
    continue from the same sequence without re-running token projection.
    """

    def __init__(
        self,
        transformer: nn.Module,
        start_layer: int,
        num_layers: int,
        und_seq: int,
        vision_seq: int,
        action_seq: int,
    ):
        super().__init__()
        if not bool(transformer.config.action_gen):
            raise ValueError("Cosmos3 transformer was not configured with action_gen=True")
        self.proj_in = transformer.proj_in
        self.action_proj_in = transformer.action_proj_in
        self.action_modality_embed = transformer.action_modality_embed
        self.time_embedder = transformer.time_embedder
        self.stack = OfficialCosmos3StackWrapper(
            transformer,
            start_layer=start_layer,
            num_layers=num_layers,
            und_seq=und_seq,
            gen_seq=int(vision_seq) + int(action_seq),
        )
        self.timestep_scale = float(transformer.config.timestep_scale)
        self.latent_patch_size = int(transformer.config.latent_patch_size)
        self.vision_seq = int(vision_seq)
        self.action_seq = int(action_seq)

    def _time_embedding(self, timesteps: torch.Tensor, seq_len: int, dtype: torch.dtype) -> torch.Tensor:
        # Keep embedding computation in the same dtype as downstream hidden states.
        t = timesteps.reshape(-1).to(dtype=dtype)
        t = t * self.timestep_scale
        half_dim = self.time_embedder.frequency_embedding_size // 2
        freqs = torch.exp(
            -torch.log(torch.tensor(10000.0, dtype=dtype, device=t.device))
            * torch.arange(half_dim, dtype=dtype, device=t.device)
            / float(half_dim)
        )
        args = t[:, None] * freqs[None, :]
        t_freq = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.time_embedder.frequency_embedding_size % 2 == 1:
            t_freq = torch.cat([t_freq, torch.zeros_like(t_freq[:, :1])], dim=-1)
        if t_freq.shape[0] == 1:
            t_freq = t_freq.expand(seq_len, -1)
        else:
            t_freq = t_freq.view(-1, t_freq.shape[-1])
        if int(t_freq.shape[0]) != int(seq_len):
            raise ValueError(f"Expected {seq_len} timestep values, got {t_freq.shape[0]}")
        return self.time_embedder.linear_2(self.time_embedder.act(self.time_embedder.linear_1(t_freq))).to(dtype=dtype)

    def forward(
        self,
        und_hidden_states: torch.Tensor,
        noisy_future_latents: torch.Tensor,
        actions: torch.Tensor,
        vision_timesteps: torch.Tensor,
        action_timesteps: torch.Tensor,
        position_ids: torch.Tensor,
        domain_id: torch.Tensor,
    ) -> torch.Tensor:
        latent_patches = patchify_latents(noisy_future_latents, self.latent_patch_size)
        vision_hidden_states = self.proj_in(latent_patches)
        vision_hidden_states = vision_hidden_states + self._time_embedding(
            vision_timesteps,
            self.vision_seq,
            vision_hidden_states.dtype,
        )

        if domain_id.ndim == 0 or domain_id.numel() == 1:
            per_token_domain_id = domain_id.reshape(1).expand(actions.shape[0])
        else:
            per_token_domain_id = domain_id.reshape(-1)
        action_hidden_states = self.action_proj_in(actions, per_token_domain_id)
        action_hidden_states = action_hidden_states + self.action_modality_embed.view(1, -1)
        action_hidden_states = action_hidden_states + self._time_embedding(
            action_timesteps,
            self.action_seq,
            action_hidden_states.dtype,
        )

        packed_hidden_states = torch.cat([und_hidden_states, vision_hidden_states, action_hidden_states], dim=0)
        und_output, gen_output = self.stack(packed_hidden_states, position_ids)
        return torch.cat([und_output, gen_output], dim=0)


class OfficialCosmos3ForwardDynamicsStackSegmentWrapper(nn.Module):
    """Stack-only FD segment for continuing a packed hidden sequence."""

    def __init__(self, transformer: nn.Module, start_layer: int, num_layers: int, und_seq: int, gen_seq: int):
        super().__init__()
        self.stack = OfficialCosmos3StackWrapper(transformer, start_layer, num_layers, und_seq, gen_seq)

    def forward(self, packed_hidden_states: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        und_output, gen_output = self.stack(packed_hidden_states, position_ids)
        return torch.cat([und_output, gen_output], dim=0)


class OfficialCosmos3ForwardDynamicsHeadWrapper(nn.Module):
    """FD final projection from packed hidden sequence to future latent velocity."""

    def __init__(
        self,
        transformer: nn.Module,
        und_seq: int,
        vision_seq: int,
        frames: int,
        latent_height: int,
        latent_width: int,
    ):
        super().__init__()
        self.proj_out = transformer.proj_out
        self.und_seq = int(und_seq)
        self.vision_seq = int(vision_seq)
        self.frames = int(frames)
        self.latent_height = int(latent_height)
        self.latent_width = int(latent_width)
        self.latent_patch_size = int(transformer.config.latent_patch_size)
        self.latent_channels = int(transformer.config.latent_channel)

    def forward(self, packed_hidden_states: torch.Tensor) -> torch.Tensor:
        gen_hidden_states = packed_hidden_states[self.und_seq :]
        vision_output = gen_hidden_states[: self.vision_seq]
        pred_patches = self.proj_out(vision_output)
        return unpatchify_latents(
            pred_patches,
            latent_channels=self.latent_channels,
            frames=self.frames,
            latent_height=self.latent_height,
            latent_width=self.latent_width,
            latent_patch_size=self.latent_patch_size,
        )


def make_forward_dynamics_real_packed_boundary_inputs(
    *,
    transformer: nn.Module,
    model_root: str | Path,
    prompt: str,
    actions: torch.Tensor | None,
    frames: int,
    latent_height: int,
    latent_width: int,
    raw_action_dim: int,
    domain_id: int,
    seed: int,
    dtype: torch.dtype,
    device: torch.device,
    timestep: float = 0.5,
    fps_vision: float = 10.0,
    fps_action: float = 10.0,
) -> tuple[
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    list[str],
    list[str],
    dict[str, Any],
]:
    """Create official-packed host inputs for a fixed-shape forward dynamics step."""
    from diffusers_cosmos3.sequence_packing import GenerationDataClean, SequencePlan, pack_input_sequence

    max_action_dim = int(transformer.config.max_action_dim)
    if raw_action_dim > max_action_dim:
        raise ValueError(f"raw_action_dim={raw_action_dim} exceeds max_action_dim={max_action_dim}")
    latent_channels = int(transformer.config.latent_channel)
    latent_patch_size = int(transformer.config.latent_patch_size)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    noisy_future_latents = torch.randn(
        (1, latent_channels, frames, latent_height, latent_width),
        generator=generator,
        dtype=torch.float32,
    )
    if actions is None:
        action_seq = 4
        action_tensor = torch.zeros((action_seq, max_action_dim), dtype=torch.float32)
        action_tensor[:, :raw_action_dim] = torch.randn(
            (action_seq, raw_action_dim),
            generator=generator,
            dtype=torch.float32,
        )
    else:
        if actions.ndim != 2:
            raise ValueError(f"actions must be [seq, dim], got {tuple(actions.shape)}")
        action_seq = int(actions.shape[0])
        raw_action_dim = int(actions.shape[1])
        if raw_action_dim > max_action_dim:
            raise ValueError(f"actions dim={raw_action_dim} exceeds max_action_dim={max_action_dim}")
        action_tensor = torch.zeros((action_seq, max_action_dim), dtype=torch.float32)
        action_tensor[:, :raw_action_dim] = actions.detach().cpu().float()

    text_ids = tokenize_generator_prompt(model_root, prompt)
    special_tokens = resolve_text_special_tokens(model_root)
    domain_tensor = torch.tensor([int(domain_id)], dtype=torch.long, device=device)
    raw_dim_tensor = torch.tensor(int(raw_action_dim), dtype=torch.long)
    gen_data = GenerationDataClean(
        batch_size=1,
        is_image_batch=frames == 1,
        x0_tokens_vision=[noisy_future_latents],
        fps_vision=torch.tensor([float(fps_vision)], dtype=torch.float32),
        x0_tokens_action=[action_tensor],
        fps_action=torch.tensor([float(fps_action)], dtype=torch.float32),
        action_domain_id=[domain_tensor.cpu()],
        raw_action_dim=[raw_dim_tensor.cpu()],
    )
    packed_seq = pack_input_sequence(
        sequence_plans=[SequencePlan(has_text=True, has_vision=True, has_action=True)],
        input_text_indexes=[text_ids],
        gen_data_clean=gen_data,
        input_timesteps=torch.tensor([float(timestep)], dtype=torch.float32),
        special_tokens=special_tokens,
        latent_patch_size=latent_patch_size,
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
    if packed_seq.vision is None or packed_seq.action is None:
        raise ValueError("official pack_input_sequence did not create both vision and action metadata")
    if int(packed_seq.vision.mse_loss_indexes.numel()) != int(packed_seq.vision.sequence_indexes.numel()):
        raise ValueError("conditioned future frames are not supported by this fixed forward dynamics boundary")
    if int(packed_seq.action.mse_loss_indexes.numel()) != int(packed_seq.action.sequence_indexes.numel()):
        raise ValueError("conditioned actions are not supported by this fixed forward dynamics boundary")

    und_hidden_states = transformer.embed_tokens(packed_seq.text_ids.to(device)).to(dtype=dtype)
    metadata = {
        "model_root": str(Path(model_root)),
        "prompt": prompt,
        "text_token_count": int(packed_seq.text_ids.numel()),
        "text_ids": [int(item) for item in packed_seq.text_ids.tolist()],
        "und_seq": int(packed_seq.text_ids.numel()),
        "vision_seq": int(packed_seq.vision.sequence_indexes.numel()),
        "action_seq": int(packed_seq.action.sequence_indexes.numel()),
        "raw_action_dim": int(raw_action_dim),
        "max_action_dim": max_action_dim,
        "domain_id": int(domain_id),
        "sequence_length": int(packed_seq.sequence_length),
        "split_lens": list(packed_seq.split_lens),
        "attn_modes": list(packed_seq.attn_modes),
        "sample_lens": list(packed_seq.sample_lens),
        "position_ids_shape": list(packed_seq.position_ids.shape),
        "vision_sequence_indexes": [int(item) for item in packed_seq.vision.sequence_indexes.tolist()],
        "vision_mse_loss_indexes": [int(item) for item in packed_seq.vision.mse_loss_indexes.tolist()],
        "vision_token_shapes": [list(item) for item in packed_seq.vision.token_shapes],
        "action_sequence_indexes": [int(item) for item in packed_seq.action.sequence_indexes.tolist()],
        "action_mse_loss_indexes": [int(item) for item in packed_seq.action.mse_loss_indexes.tolist()],
        "action_token_shapes": [list(item) for item in packed_seq.action.token_shapes],
        "special_tokens": special_tokens,
        "notes": [
            "Forward dynamics predicts future vision latent velocity; action tokens are conditioning tokens.",
            "Scheduler loop and VAE decode stay in host/runtime.",
        ],
    }
    return (
        und_hidden_states,
        noisy_future_latents.to(device=device, dtype=dtype),
        action_tensor.to(device=device, dtype=dtype),
        packed_seq.vision.timesteps.to(device=device, dtype=torch.float32),
        packed_seq.action.timesteps.to(device=device, dtype=torch.float32),
        packed_seq.position_ids.to(device=device),
        domain_tensor,
    ), [
        "und_hidden_states",
        "noisy_future_latents",
        "actions",
        "vision_timesteps",
        "action_timesteps",
        "position_ids",
        "domain_id",
    ], ["future_latent_velocity"], metadata
