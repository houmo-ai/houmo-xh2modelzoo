# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0

"""Host-side CFG scheduler runtime for Cosmos3-Nano generator HMONNX graphs.

The denoiser network is executed by HMONNX. Tokenization, classifier-free
guidance, scheduler stepping, and latent buffer ownership intentionally remain
in Python host code. This mirrors the production boundary: neural network
forward graphs are quantized, while sampling/control flow stays inspectable and
replaceable.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from diffusers.schedulers import UniPCMultistepScheduler
from xhquant.api import HMONNXInference, xhquant_init

from runtime.generator_runtime import make_generator_real_packed_latent_boundary_inputs


def _as_tuple(value: Any) -> tuple[torch.Tensor, ...]:
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return (value,)


def _require_path(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing HMONNX file: {path}")


@dataclass(frozen=True)
class GeneratorDenoiserHmonnxPath:
    denoiser: Path

    def all(self) -> list[Path]:
        return [self.denoiser]

    def as_dict(self) -> dict[str, str]:
        return {"denoiser": str(self.denoiser)}


@dataclass(frozen=True)
class GeneratorPackedCondition:
    und_hidden_states: torch.Tensor
    position_ids: torch.Tensor
    pack_meta: dict[str, Any]


@dataclass(frozen=True)
class GeneratorSchedulerInputs:
    cond: GeneratorPackedCondition
    uncond: GeneratorPackedCondition
    initial_latents: torch.Tensor


@dataclass(frozen=True)
class GeneratorSchedulerStepTrace:
    step_index: int
    timestep: float
    cond_velocity: torch.Tensor
    uncond_velocity: torch.Tensor
    cfg_velocity: torch.Tensor
    latents: torch.Tensor


@dataclass(frozen=True)
class GeneratorSchedulerOutput:
    final_latents: torch.Tensor
    steps: tuple[GeneratorSchedulerStepTrace, ...]


def load_generator_scheduler(model_root: str | Path) -> UniPCMultistepScheduler:
    return UniPCMultistepScheduler.from_pretrained(str(Path(model_root) / "scheduler"))


def make_generator_cfg_scheduler_inputs(
    *,
    transformer: torch.nn.Module,
    model_root: str | Path,
    prompt: str,
    negative_prompt: str,
    frames: int,
    latent_height: int,
    latent_width: int,
    seed: int,
    dtype: torch.dtype,
    device: torch.device,
    timestep: float = 0.5,
) -> GeneratorSchedulerInputs:
    cond_tensors, _, _, cond_meta = make_generator_real_packed_latent_boundary_inputs(
        transformer=transformer,
        model_root=model_root,
        prompt=prompt,
        frames=frames,
        latent_height=latent_height,
        latent_width=latent_width,
        seed=seed,
        dtype=dtype,
        device=device,
        timestep=timestep,
    )
    uncond_tensors, _, _, uncond_meta = make_generator_real_packed_latent_boundary_inputs(
        transformer=transformer,
        model_root=model_root,
        prompt=negative_prompt,
        frames=frames,
        latent_height=latent_height,
        latent_width=latent_width,
        seed=seed,
        dtype=dtype,
        device=device,
        timestep=timestep,
    )
    cond_hidden, initial_latents, _, cond_position_ids = cond_tensors
    uncond_hidden, _, _, uncond_position_ids = uncond_tensors
    _validate_condition_compatibility(cond_meta, uncond_meta)
    return GeneratorSchedulerInputs(
        cond=GeneratorPackedCondition(cond_hidden, cond_position_ids, cond_meta),
        uncond=GeneratorPackedCondition(uncond_hidden, uncond_position_ids, uncond_meta),
        initial_latents=initial_latents,
    )


def _validate_condition_compatibility(cond_meta: dict[str, Any], uncond_meta: dict[str, Any]) -> None:
    keys = ("und_seq", "gen_seq", "sequence_length", "split_lens", "attn_modes", "sample_lens", "position_ids_shape")
    mismatches = [key for key in keys if cond_meta.get(key) != uncond_meta.get(key)]
    if mismatches:
        detail = ", ".join(f"{key}: {cond_meta.get(key)!r} != {uncond_meta.get(key)!r}" for key in mismatches)
        raise ValueError(f"cond/uncond packed shapes are incompatible with one fixed HMONNX graph: {detail}")


def flatten_latents(latents: torch.Tensor) -> torch.Tensor:
    return latents.reshape(-1)


def unflatten_latents(flat: torch.Tensor, shape: torch.Size | tuple[int, ...]) -> torch.Tensor:
    return flat.reshape(tuple(shape))


def make_timestep_vector(timestep: torch.Tensor, gen_seq: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    scalar = float(timestep.detach().cpu().reshape(-1)[0].item())
    return torch.full((int(gen_seq),), scalar, device=device, dtype=dtype)


def cfg_combine(cond_velocity: torch.Tensor, uncond_velocity: torch.Tensor, guidance_scale: float) -> torch.Tensor:
    cond = cond_velocity.float()
    uncond = uncond_velocity.float()
    return uncond + float(guidance_scale) * (cond - uncond)


class GeneratorTorchDenoiserRuntime:
    """PyTorch reference denoiser runtime with the same host contract as HMONNX."""

    def __init__(self, wrapper: torch.nn.Module, *, device: str = "cuda") -> None:
        self.wrapper = wrapper.eval()
        self.device = torch.device(device if torch.cuda.is_available() or not device.startswith("cuda") else "cpu")
        self.wrapper.to(self.device)
        self.input_dtype = next(self.wrapper.parameters()).dtype

    @torch.no_grad()
    def denoise(
        self,
        condition: GeneratorPackedCondition,
        latents: torch.Tensor,
        timestep_vector: torch.Tensor,
    ) -> torch.Tensor:
        autocast_context = (
            torch.autocast(device_type="cuda", dtype=self.input_dtype)
            if self.device.type == "cuda" and self.input_dtype == torch.float16
            else nullcontext()
        )
        with autocast_context:
            output = self.wrapper(
                condition.und_hidden_states.to(self.device, dtype=self.input_dtype),
                latents.to(self.device, dtype=self.input_dtype),
                timestep_vector.to(self.device, dtype=self.input_dtype),
                condition.position_ids.to(self.device),
            )
        return output.detach().cpu()


class GeneratorHmonnxDenoiserRuntime:
    """HMONNX denoiser runtime for one fixed-shape generator latent boundary graph."""

    def __init__(
        self,
        path: GeneratorDenoiserHmonnxPath | Path,
        *,
        device: str = "cuda",
        input_dtype: str = "float16",
        debug: bool = False,
        init_xhquant: bool = True,
    ) -> None:
        if isinstance(path, Path):
            path = GeneratorDenoiserHmonnxPath(path)
        _require_path(path.denoiser)
        if init_xhquant:
            xhquant_init(None, debug=debug)
        self.path = path
        self.device = torch.device(device if torch.cuda.is_available() or not device.startswith("cuda") else "cpu")
        self.hmonnx_dtype = torch.float16 if input_dtype == "float16" else torch.float32
        self.session = HMONNXInference(str(path.denoiser)).to(self.device)

    @torch.no_grad()
    def denoise(
        self,
        condition: GeneratorPackedCondition,
        latents: torch.Tensor,
        timestep_vector: torch.Tensor,
    ) -> torch.Tensor:
        raw = self.session.forward(
            condition.und_hidden_states.to(self.device, dtype=self.hmonnx_dtype),
            latents.to(self.device, dtype=self.hmonnx_dtype),
            timestep_vector.to(self.device, dtype=self.hmonnx_dtype),
            condition.position_ids.to(self.device, dtype=self.hmonnx_dtype if condition.position_ids.is_floating_point() else torch.int32),
        )
        return _as_tuple(raw)[0].detach().cpu()


class GeneratorCfgSchedulerRuntime:
    """Host CFG + scheduler loop around a denoiser runtime."""

    def __init__(
        self,
        denoiser_runtime: GeneratorTorchDenoiserRuntime | GeneratorHmonnxDenoiserRuntime,
        scheduler: UniPCMultistepScheduler,
        *,
        guidance_scale: float,
        num_inference_steps: int,
        device: str = "cuda",
        scheduler_dtype: torch.dtype = torch.float32,
    ) -> None:
        self.denoiser_runtime = denoiser_runtime
        self.scheduler = scheduler
        self.guidance_scale = float(guidance_scale)
        self.num_inference_steps = int(num_inference_steps)
        self.device = torch.device(device if torch.cuda.is_available() or not device.startswith("cuda") else "cpu")
        self.scheduler_dtype = scheduler_dtype

    @torch.no_grad()
    def run(self, inputs: GeneratorSchedulerInputs, *, trace_steps: bool = True) -> GeneratorSchedulerOutput:
        gen_seq = int(inputs.cond.pack_meta["gen_seq"])
        latent_shape = tuple(inputs.initial_latents.shape)
        latents = inputs.initial_latents.detach().cpu().float()
        self.scheduler.set_timesteps(self.num_inference_steps, device=self.device)
        traces: list[GeneratorSchedulerStepTrace] = []
        for step_index, timestep in enumerate(self.scheduler.timesteps):
            timestep_vector = make_timestep_vector(
                timestep,
                gen_seq,
                device=torch.device("cpu"),
                dtype=torch.float32,
            )
            cond_velocity = self.denoiser_runtime.denoise(inputs.cond, latents, timestep_vector).float()
            uncond_velocity = self.denoiser_runtime.denoise(inputs.uncond, latents, timestep_vector).float()
            cfg_velocity = cfg_combine(cond_velocity, uncond_velocity, self.guidance_scale)
            flat_next = self.scheduler.step(
                model_output=flatten_latents(cfg_velocity).to(self.device, dtype=self.scheduler_dtype),
                timestep=timestep,
                sample=flatten_latents(latents).unsqueeze(0).to(self.device, dtype=self.scheduler_dtype),
                return_dict=False,
            )[0].squeeze(0).detach().cpu().float()
            latents = unflatten_latents(flat_next, latent_shape)
            if trace_steps:
                traces.append(
                    GeneratorSchedulerStepTrace(
                        step_index=step_index,
                        timestep=float(timestep.detach().cpu().item()),
                        cond_velocity=cond_velocity.detach().cpu(),
                        uncond_velocity=uncond_velocity.detach().cpu(),
                        cfg_velocity=cfg_velocity.detach().cpu(),
                        latents=latents.detach().cpu(),
                    )
                )
        return GeneratorSchedulerOutput(final_latents=latents.detach().cpu(), steps=tuple(traces))
