# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0

"""Segmented Cosmos3-Nano generator runtimes for mixed-precision validation."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import torch
from xhquant.api import HMONNXInference, xhquant_init

from runtime.generator_runtime import OfficialCosmos3StackWrapper, patchify_latents, unpatchify_latents


class GeneratorLateStackHmonnxDenoiserRuntime:
    """Run early generator layers in PyTorch and the late stack with HMONNX.

    This runtime is a diagnostic bridge for validating late-layer precision
    choices under a real prefix activation handoff, without re-exporting the
    large full denoiser boundary graph.
    """

    def __init__(
        self,
        transformer: torch.nn.Module,
        *,
        late_stack_hmonnx: str | Path,
        late_start_layer: int,
        late_num_layers: int,
        und_seq: int,
        frames: int,
        latent_height: int,
        latent_width: int,
        device: str = "cuda",
        hmonnx_input_dtype: str = "float16",
        init_xhquant: bool = True,
        debug: bool = False,
    ) -> None:
        path = Path(late_stack_hmonnx)
        if not path.exists():
            raise FileNotFoundError(f"Missing late-stack HMONNX file: {path}")
        if int(late_start_layer) <= 0:
            raise ValueError("late_start_layer must be > 0 for segmented runtime")
        self.transformer = transformer.eval()
        self.device = torch.device(device if torch.cuda.is_available() or not device.startswith("cuda") else "cpu")
        self.transformer.to(self.device)
        self.input_dtype = next(self.transformer.parameters()).dtype
        self.hmonnx_dtype = torch.float16 if hmonnx_input_dtype == "float16" else torch.float32

        self.latent_patch_size = int(transformer.config.latent_patch_size)
        self.latent_channels = int(transformer.config.latent_channel)
        self.frames = int(frames)
        self.latent_height = int(latent_height)
        self.latent_width = int(latent_width)
        self.und_seq = int(und_seq)
        self.gen_seq = self.frames * (self.latent_height // self.latent_patch_size) * (
            self.latent_width // self.latent_patch_size
        )
        self.timestep_scale = float(transformer.config.timestep_scale)

        self.early_stack = OfficialCosmos3StackWrapper(
            transformer,
            start_layer=0,
            num_layers=int(late_start_layer),
            und_seq=self.und_seq,
            gen_seq=self.gen_seq,
        ).eval()
        self.late_start_layer = int(late_start_layer)
        self.late_num_layers = int(late_num_layers)
        if self.late_start_layer + self.late_num_layers != int(transformer.config.num_hidden_layers):
            raise ValueError(
                "late stack must end at the final transformer layer for this runtime: "
                f"{self.late_start_layer}+{self.late_num_layers} != {transformer.config.num_hidden_layers}"
            )

        if init_xhquant:
            xhquant_init(None, debug=debug)
        self.late_session = HMONNXInference(str(path)).to(self.device)

    @torch.no_grad()
    def denoise(
        self,
        und_hidden_states: torch.Tensor,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        und_hidden_states = und_hidden_states.to(self.device, dtype=self.input_dtype)
        noisy_latents = noisy_latents.to(self.device, dtype=self.input_dtype)
        position_ids = position_ids.to(self.device)

        autocast_context = (
            torch.autocast(device_type="cuda", dtype=self.input_dtype)
            if self.device.type == "cuda" and self.input_dtype == torch.float16
            else nullcontext()
        )
        with autocast_context:
            latent_patches = patchify_latents(noisy_latents, self.latent_patch_size)
            gen_hidden_states = self.transformer.proj_in(latent_patches)
            scaled_timesteps = timesteps.reshape(-1).to(self.device, dtype=torch.float32) * self.timestep_scale
            if scaled_timesteps.shape[0] == 1:
                scaled_timesteps = scaled_timesteps.expand(self.gen_seq)
            timestep_embeds = self.transformer.time_embedder(scaled_timesteps).to(dtype=gen_hidden_states.dtype)
            gen_hidden_states = gen_hidden_states + timestep_embeds

            hidden_states = torch.cat([und_hidden_states, gen_hidden_states], dim=0)
            early_und, early_gen = self.early_stack(hidden_states, position_ids)
        late_input = torch.cat([early_und, early_gen], dim=0)

        raw = self.late_session.forward(
            late_input.to(self.device, dtype=self.hmonnx_dtype),
            position_ids.to(self.device, dtype=torch.int32),
        )
        if isinstance(raw, torch.Tensor):
            raise TypeError("late stack HMONNX must return und and gen hidden states")
        late_outputs = tuple(item.detach().to(self.device, dtype=self.input_dtype) for item in raw)
        if len(late_outputs) != 2:
            raise ValueError(f"late stack HMONNX returned {len(late_outputs)} outputs, expected 2")
        _, late_gen = late_outputs
        pred_patches = self.transformer.proj_out(late_gen)
        return unpatchify_latents(
            pred_patches,
            latent_channels=self.latent_channels,
            frames=self.frames,
            latent_height=self.latent_height,
            latent_width=self.latent_width,
            latent_patch_size=self.latent_patch_size,
        ).detach().cpu()
