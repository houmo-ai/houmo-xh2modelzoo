# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0

"""Reusable host/runtime wrappers for Cosmos3-Nano generator graphs."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    seq, num_key_value_heads, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :].expand(seq, num_key_value_heads, n_rep, head_dim)
    return hidden_states.reshape(seq, num_key_value_heads * n_rep, head_dim)


class ExportableTwoWayAttentionProcessor:
    """ONNX-exportable equivalent of official single-sample two-way attention."""

    def __call__(
        self,
        packed_query_states: dict[str, Any],
        packed_key_states: dict[str, Any],
        packed_value_states: dict[str, Any],
    ) -> dict[str, Any]:
        from diffusers_cosmos3.sequence_packing import (
            from_mode_splits,
            get_all_seq,
            get_causal_seq,
            get_full_only_seq,
        )

        causal_q, _ = get_causal_seq(packed_query_states)
        causal_k, _ = get_causal_seq(packed_key_states)
        causal_v, _ = get_causal_seq(packed_value_states)
        full_q, _ = get_full_only_seq(packed_query_states)
        all_k = get_all_seq(packed_key_states)
        all_v = get_all_seq(packed_value_states)

        n_rep = int(causal_q.shape[1] // causal_k.shape[1])
        causal_k = repeat_kv(causal_k, n_rep)
        causal_v = repeat_kv(causal_v, n_rep)
        all_k = repeat_kv(all_k, n_rep)
        all_v = repeat_kv(all_v, n_rep)
        scale = causal_q.shape[-1] ** -0.5

        causal_scores = torch.matmul(causal_q.transpose(0, 1), causal_k.transpose(0, 1).transpose(-2, -1)) * scale
        causal_len = causal_q.shape[0]
        causal_mask = torch.triu(
            torch.full((causal_len, causal_len), -10000.0, dtype=causal_scores.dtype, device=causal_scores.device),
            diagonal=1,
        )
        causal_scores = causal_scores + causal_mask.unsqueeze(0)
        causal_out = torch.matmul(torch.softmax(causal_scores, dim=-1), causal_v.transpose(0, 1))
        causal_out = causal_out.transpose(0, 1).contiguous().flatten(-2, -1)

        full_scores = torch.matmul(full_q.transpose(0, 1), all_k.transpose(0, 1).transpose(-2, -1)) * scale
        full_out = torch.matmul(torch.softmax(full_scores, dim=-1), all_v.transpose(0, 1))
        full_out = full_out.transpose(0, 1).contiguous().flatten(-2, -1)
        return from_mode_splits(causal_out, full_out, packed_query_states)


class OfficialCosmos3StackWrapper(nn.Module):
    """Fixed-shape official Cosmos3 generator transformer stack."""

    def __init__(self, transformer: nn.Module, start_layer: int, num_layers: int, und_seq: int, gen_seq: int):
        super().__init__()
        self.config = transformer.config
        self.layers = nn.ModuleList(transformer.layers[start_layer : start_layer + num_layers])
        for layer in self.layers:
            layer.self_attn.dispatch_attention_fn = ExportableTwoWayAttentionProcessor()
        self.rotary_emb = transformer.rotary_emb
        self.start_layer = int(start_layer)
        self.num_layers = int(num_layers)
        self.und_seq = int(und_seq)
        self.gen_seq = int(gen_seq)
        self.total_seq = self.und_seq + self.gen_seq

    def _build_pack(self, hidden_states: torch.Tensor) -> dict[str, Any]:
        from diffusers_cosmos3.sequence_packing import build_packed_sequence

        und_indexes = torch.arange(self.und_seq, dtype=torch.long, device=hidden_states.device)
        gen_indexes = torch.arange(self.und_seq, self.total_seq, dtype=torch.long, device=hidden_states.device)
        pack, _, _ = build_packed_sequence(
            "two_way",
            packed_sequence=hidden_states,
            attn_modes=["causal", "full"],
            split_lens=[self.und_seq, self.gen_seq],
            sample_lens=[self.total_seq],
            packed_und_token_indexes=und_indexes,
            packed_gen_token_indexes=gen_indexes,
            num_heads=int(self.config.num_attention_heads),
            head_dim=int(self.config.head_dim),
            num_layers=int(self.config.num_hidden_layers),
        )
        return pack

    def _position_embeddings(self, position_ids: torch.Tensor, pack: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        from diffusers_cosmos3.sequence_packing import from_joint, get_device_and_dtype

        device, dtype = get_device_and_dtype(pack)
        meta_tensor = torch.tensor([], dtype=dtype, device=device)
        if position_ids.ndim == 1:
            rope_position_ids = position_ids.unsqueeze(0)
        else:
            rope_position_ids = position_ids.unsqueeze(1)
        cos, sin = self.rotary_emb(meta_tensor, position_ids=rope_position_ids)
        cos = cos.squeeze(0)
        sin = sin.squeeze(0)
        return from_joint(cos, pack), from_joint(sin, pack)

    def forward(self, hidden_states: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        from diffusers_cosmos3.sequence_packing import get_gen_seq, get_und_seq

        pack = self._build_pack(hidden_states)
        position_embeddings = self._position_embeddings(position_ids, pack)
        for layer in self.layers:
            pack = layer(pack, None, position_embeddings, None, None, natten_metadata=None)
        return get_und_seq(pack), get_gen_seq(pack)


class OfficialCosmos3LayerWrapper(OfficialCosmos3StackWrapper):
    """Backward-compatible one-layer wrapper."""

    def __init__(self, transformer: nn.Module, layer: int, und_seq: int, gen_seq: int):
        super().__init__(transformer, layer, 1, und_seq, gen_seq)


class OfficialCosmos3DenoiserStepWrapper(nn.Module):
    """Fixed-shape generator denoiser step: proj_in + time + stack + proj_out."""

    def __init__(self, transformer: nn.Module, start_layer: int, num_layers: int, und_seq: int, gen_seq: int):
        super().__init__()
        self.proj_in = transformer.proj_in
        self.proj_out = transformer.proj_out
        self.time_embedder = transformer.time_embedder
        self.stack = OfficialCosmos3StackWrapper(transformer, start_layer, num_layers, und_seq, gen_seq)
        self.timestep_scale = float(transformer.config.timestep_scale)
        self.und_seq = int(und_seq)
        self.gen_seq = int(gen_seq)

    def forward(
        self,
        und_hidden_states: torch.Tensor,
        latent_patches: torch.Tensor,
        timesteps: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        gen_hidden_states = self.proj_in(latent_patches)
        scaled_timesteps = timesteps.reshape(-1).to(dtype=torch.float32) * self.timestep_scale
        if scaled_timesteps.shape[0] == 1:
            scaled_timesteps = scaled_timesteps.expand(self.gen_seq)
        timestep_embeds = self.time_embedder(scaled_timesteps).to(dtype=gen_hidden_states.dtype)
        gen_hidden_states = gen_hidden_states + timestep_embeds
        hidden_states = torch.cat([und_hidden_states, gen_hidden_states], dim=0)
        _, gen_output = self.stack(hidden_states, position_ids)
        return self.proj_out(gen_output)


def patchify_latents(latents: torch.Tensor, latent_patch_size: int) -> torch.Tensor:
    """Patchify [1, C, T, H, W] latents to [T * H/p * W/p, p * p * C]."""
    p = int(latent_patch_size)
    batch, channels, frames, height, width = latents.shape
    if batch != 1:
        raise ValueError(f"Only batch=1 is supported for fixed-shape generator boundary, got {batch}")
    if height % p != 0 or width % p != 0:
        raise ValueError(f"Latent height/width must be divisible by patch size {p}, got {(height, width)}")
    h_patches = height // p
    w_patches = width // p
    latents = latents.squeeze(0).reshape(channels, frames, h_patches, p, w_patches, p)
    return torch.einsum("cthpwq->thwpqc", latents).reshape(frames * h_patches * w_patches, p * p * channels)


def unpatchify_latents(
    latent_patches: torch.Tensor,
    *,
    latent_channels: int,
    frames: int,
    latent_height: int,
    latent_width: int,
    latent_patch_size: int,
) -> torch.Tensor:
    """Unpatchify [T * H/p * W/p, p * p * C] patches to [1, C, T, H, W]."""
    p = int(latent_patch_size)
    h_patches = int(latent_height) // p
    w_patches = int(latent_width) // p
    latents = latent_patches.reshape(int(frames), h_patches, w_patches, p, p, int(latent_channels))
    latents = torch.einsum("thwpqc->cthpwq", latents)
    latents = latents.reshape(int(latent_channels), int(frames), int(latent_height), int(latent_width))
    return latents.unsqueeze(0)


class OfficialCosmos3LatentDenoiserBoundaryWrapper(nn.Module):
    """Generator denoiser boundary with latent patchify/unpatchify in the runtime graph."""

    def __init__(
        self,
        transformer: nn.Module,
        start_layer: int,
        num_layers: int,
        und_seq: int,
        frames: int,
        latent_height: int,
        latent_width: int,
    ):
        super().__init__()
        self.latent_patch_size = int(transformer.config.latent_patch_size)
        self.latent_channels = int(transformer.config.latent_channel)
        self.frames = int(frames)
        self.latent_height = int(latent_height)
        self.latent_width = int(latent_width)
        if self.latent_height % self.latent_patch_size != 0 or self.latent_width % self.latent_patch_size != 0:
            raise ValueError(
                "latent_height and latent_width must be divisible by latent_patch_size "
                f"{self.latent_patch_size}, got {(self.latent_height, self.latent_width)}"
            )
        gen_seq = self.frames * (self.latent_height // self.latent_patch_size) * (
            self.latent_width // self.latent_patch_size
        )
        self.denoiser = OfficialCosmos3DenoiserStepWrapper(
            transformer,
            start_layer=start_layer,
            num_layers=num_layers,
            und_seq=und_seq,
            gen_seq=gen_seq,
        )

    @property
    def gen_seq(self) -> int:
        return self.denoiser.gen_seq

    def forward(
        self,
        und_hidden_states: torch.Tensor,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        latent_patches = patchify_latents(noisy_latents, self.latent_patch_size)
        pred_patches = self.denoiser(und_hidden_states, latent_patches, timesteps, position_ids)
        return unpatchify_latents(
            pred_patches,
            latent_channels=self.latent_channels,
            frames=self.frames,
            latent_height=self.latent_height,
            latent_width=self.latent_width,
            latent_patch_size=self.latent_patch_size,
        )


def make_generator_stack_inputs(
    *,
    hidden_size: int,
    und_seq: int,
    gen_seq: int,
    seed: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[tuple[torch.Tensor, torch.Tensor], list[str], list[str]]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    total_seq = int(und_seq) + int(gen_seq)
    hidden_states = torch.randn((total_seq, hidden_size), generator=generator, dtype=torch.float32).to(
        device=device,
        dtype=dtype,
    )
    position_ids = torch.arange(total_seq, dtype=torch.long, device=device)
    return (hidden_states, position_ids), ["hidden_states", "position_ids"], ["und_hidden_states", "gen_hidden_states"]


def make_generator_denoiser_step_inputs(
    *,
    hidden_size: int,
    patch_latent_dim: int,
    und_seq: int,
    gen_seq: int,
    seed: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], list[str], list[str]]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    und_hidden_states = torch.randn((und_seq, hidden_size), generator=generator, dtype=torch.float32).to(
        device=device,
        dtype=dtype,
    )
    latent_patches = torch.randn((gen_seq, patch_latent_dim), generator=generator, dtype=torch.float32).to(
        device=device,
        dtype=dtype,
    )
    timesteps = torch.full((gen_seq,), 0.5, dtype=torch.float32, device=device)
    position_ids = torch.arange(und_seq + gen_seq, dtype=torch.long, device=device)
    return (
        und_hidden_states,
        latent_patches,
        timesteps,
        position_ids,
    ), ["und_hidden_states", "latent_patches", "timesteps", "position_ids"], ["latent_pred"]


def make_generator_latent_boundary_inputs(
    *,
    hidden_size: int,
    latent_channels: int,
    frames: int,
    latent_height: int,
    latent_width: int,
    latent_patch_size: int,
    und_seq: int,
    seed: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], list[str], list[str]]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    h_patches = int(latent_height) // int(latent_patch_size)
    w_patches = int(latent_width) // int(latent_patch_size)
    gen_seq = int(frames) * h_patches * w_patches
    und_hidden_states = torch.randn((und_seq, hidden_size), generator=generator, dtype=torch.float32).to(
        device=device,
        dtype=dtype,
    )
    noisy_latents = torch.randn(
        (1, latent_channels, frames, latent_height, latent_width),
        generator=generator,
        dtype=torch.float32,
    ).to(device=device, dtype=dtype)
    timesteps = torch.full((gen_seq,), 0.5, dtype=torch.float32, device=device)
    position_ids = torch.arange(und_seq + gen_seq, dtype=torch.long, device=device)
    return (
        und_hidden_states,
        noisy_latents,
        timesteps,
        position_ids,
    ), ["und_hidden_states", "noisy_latents", "timesteps", "position_ids"], ["latent_velocity"]




def resolve_text_special_tokens(model_root: str | Path) -> dict[str, int]:
    """Resolve Cosmos3 text/generation special token ids from the local tokenizer."""
    from pathlib import Path

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(Path(model_root) / "text_tokenizer"), trust_remote_code=True)
    return {
        "start_of_generation": int(tokenizer.convert_tokens_to_ids("<|vision_start|>")),
        "end_of_generation": int(tokenizer.convert_tokens_to_ids("<|vision_end|>")),
        "eos_token_id": int(tokenizer.eos_token_id),
    }


def tokenize_generator_prompt(model_root: str | Path, prompt: str) -> list[int]:
    """Tokenize a prompt for generator packing without pulling in the full pipeline."""
    from pathlib import Path

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(Path(model_root) / "text_tokenizer"), trust_remote_code=True)
    token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not token_ids:
        token_ids = [int(tokenizer.eos_token_id)]
    return [int(item) for item in token_ids]


def make_generator_real_packed_latent_boundary_inputs(
    *,
    transformer: nn.Module,
    model_root: str | Path,
    prompt: str,
    frames: int,
    latent_height: int,
    latent_width: int,
    seed: int,
    dtype: torch.dtype,
    device: torch.device,
    timestep: float = 0.5,
    condition_frame_indexes: list[int] | None = None,
) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], list[str], list[str], dict[str, Any]]:
    """Create real official-packed host inputs for the latent denoiser boundary.

    This keeps scheduler/tokenizer/control flow on host, but uses official sequence packing
    for text indexes, vision indexes, noisy-frame indexes, timestep expansion, and mRoPE
    position ids. Returned tensors match OfficialCosmos3LatentDenoiserBoundaryWrapper.
    """
    from pathlib import Path

    from diffusers_cosmos3.sequence_packing import GenerationDataClean, SequencePlan, pack_input_sequence

    generator = torch.Generator(device="cpu").manual_seed(seed)
    latent_channels = int(transformer.config.latent_channel)
    latent_patch_size = int(transformer.config.latent_patch_size)
    noisy_latents = torch.randn(
        (1, latent_channels, frames, latent_height, latent_width),
        generator=generator,
        dtype=torch.float32,
    ).to(device=device, dtype=dtype)
    text_ids = tokenize_generator_prompt(model_root, prompt)
    special_tokens = resolve_text_special_tokens(model_root)
    gen_data = GenerationDataClean(
        batch_size=1,
        is_image_batch=frames == 1,
        x0_tokens_vision=[noisy_latents],
        fps_vision=torch.tensor([24.0], dtype=torch.float32),
    )
    packed_seq = pack_input_sequence(
        sequence_plans=[
            SequencePlan(
                has_text=True,
                has_vision=True,
                condition_frame_indexes_vision=list(condition_frame_indexes or []),
            )
        ],
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
        action_dim=int(transformer.config.max_action_dim),
    )
    if packed_seq.vision is None:
        raise ValueError("official pack_input_sequence did not create vision metadata")
    if int(packed_seq.vision.mse_loss_indexes.numel()) != int(packed_seq.vision.sequence_indexes.numel()):
        raise ValueError("conditioned frames are not supported by the fixed latent boundary smoke yet")
    und_hidden_states = transformer.embed_tokens(packed_seq.text_ids.to(device)).to(dtype=dtype)
    timesteps = packed_seq.vision.timesteps.to(device=device, dtype=torch.float32)
    position_ids = packed_seq.position_ids.to(device=device)
    metadata = {
        "model_root": str(Path(model_root)),
        "prompt": prompt,
        "text_token_count": int(packed_seq.text_ids.numel()),
        "text_ids": [int(item) for item in packed_seq.text_ids.tolist()],
        "und_seq": int(packed_seq.text_ids.numel()),
        "gen_seq": int(packed_seq.vision.sequence_indexes.numel()),
        "sequence_length": int(packed_seq.sequence_length),
        "split_lens": list(packed_seq.split_lens),
        "attn_modes": list(packed_seq.attn_modes),
        "sample_lens": list(packed_seq.sample_lens),
        "position_ids_shape": list(position_ids.shape),
        "vision_sequence_indexes": [int(item) for item in packed_seq.vision.sequence_indexes.tolist()],
        "vision_mse_loss_indexes": [int(item) for item in packed_seq.vision.mse_loss_indexes.tolist()],
        "vision_token_shapes": [list(item) for item in packed_seq.vision.token_shapes],
        "special_tokens": special_tokens,
    }
    return (
        und_hidden_states,
        noisy_latents,
        timesteps,
        position_ids,
    ), ["und_hidden_states", "noisy_latents", "timesteps", "position_ids"], ["latent_velocity"], metadata
