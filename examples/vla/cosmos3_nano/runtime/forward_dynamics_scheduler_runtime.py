# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0

"""Host-side scheduler runtime for Cosmos3-Nano forward dynamics graphs."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from diffusers.schedulers import UniPCMultistepScheduler
from xhquant.api import HMONNXInference, xhquant_init

from runtime.forward_dynamics_runtime import (
    OfficialCosmos3ForwardDynamicsHeadWrapper,
    make_forward_dynamics_real_packed_boundary_inputs,
)
from runtime.generator_scheduler_runtime import flatten_latents, load_generator_scheduler, make_timestep_vector, unflatten_latents


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
class ForwardDynamicsDenoiserHmonnxPath:
    denoiser: Path

    def all(self) -> list[Path]:
        return [self.denoiser]

    def as_dict(self) -> dict[str, str]:
        return {"denoiser": str(self.denoiser)}


@dataclass(frozen=True)
class ForwardDynamicsPackedCondition:
    und_hidden_states: torch.Tensor
    actions: torch.Tensor
    action_timesteps: torch.Tensor
    position_ids: torch.Tensor
    domain_id: torch.Tensor
    pack_meta: dict[str, Any]


@dataclass(frozen=True)
class ForwardDynamicsSchedulerInputs:
    condition: ForwardDynamicsPackedCondition
    initial_latents: torch.Tensor


@dataclass(frozen=True)
class ForwardDynamicsSchedulerStepTrace:
    step_index: int
    timestep: float
    velocity: torch.Tensor
    latents: torch.Tensor


@dataclass(frozen=True)
class ForwardDynamicsSchedulerOutput:
    final_latents: torch.Tensor
    steps: tuple[ForwardDynamicsSchedulerStepTrace, ...]


def make_forward_dynamics_scheduler_inputs(
    *,
    transformer: torch.nn.Module,
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
) -> ForwardDynamicsSchedulerInputs:
    tensors, _, _, pack_meta = make_forward_dynamics_real_packed_boundary_inputs(
        transformer=transformer,
        model_root=model_root,
        prompt=prompt,
        actions=actions,
        frames=frames,
        latent_height=latent_height,
        latent_width=latent_width,
        raw_action_dim=raw_action_dim,
        domain_id=domain_id,
        seed=seed,
        dtype=dtype,
        device=device,
        timestep=timestep,
        fps_vision=fps_vision,
        fps_action=fps_action,
    )
    (
        und_hidden_states,
        initial_latents,
        padded_actions,
        _vision_timesteps,
        action_timesteps,
        position_ids,
        domain_tensor,
    ) = tensors
    return ForwardDynamicsSchedulerInputs(
        condition=ForwardDynamicsPackedCondition(
            und_hidden_states=und_hidden_states,
            actions=padded_actions,
            action_timesteps=action_timesteps,
            position_ids=position_ids,
            domain_id=domain_tensor,
            pack_meta=pack_meta,
        ),
        initial_latents=initial_latents,
    )


class ForwardDynamicsTorchDenoiserRuntime:
    """PyTorch reference denoiser runtime with the same host contract as HMONNX."""

    def __init__(self, wrapper: torch.nn.Module, *, device: str = "cuda") -> None:
        self.wrapper = wrapper.eval()
        self.device = torch.device(device if torch.cuda.is_available() or not device.startswith("cuda") else "cpu")
        self.wrapper.to(self.device)
        self.input_dtype = next(self.wrapper.parameters()).dtype

    @torch.no_grad()
    def denoise(
        self,
        condition: ForwardDynamicsPackedCondition,
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
                condition.actions.to(self.device, dtype=self.input_dtype),
                timestep_vector.to(self.device, dtype=torch.float32),
                condition.action_timesteps.to(self.device, dtype=torch.float32),
                condition.position_ids.to(self.device),
                condition.domain_id.to(self.device),
            )
        return output.detach().cpu()


class ForwardDynamicsHmonnxDenoiserRuntime:
    """HMONNX denoiser runtime for one fixed-shape forward dynamics graph."""

    def __init__(
        self,
        path: ForwardDynamicsDenoiserHmonnxPath | Path,
        *,
        device: str = "cuda",
        input_dtype: str = "float16",
        debug: bool = False,
        init_xhquant: bool = True,
    ) -> None:
        if isinstance(path, Path):
            path = ForwardDynamicsDenoiserHmonnxPath(path)
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
        condition: ForwardDynamicsPackedCondition,
        latents: torch.Tensor,
        timestep_vector: torch.Tensor,
    ) -> torch.Tensor:
        raw = self.session.forward(
            condition.und_hidden_states.to(self.device, dtype=self.hmonnx_dtype),
            latents.to(self.device, dtype=self.hmonnx_dtype),
            condition.actions.to(self.device, dtype=self.hmonnx_dtype),
            timestep_vector.to(self.device, dtype=self.hmonnx_dtype),
            condition.action_timesteps.to(self.device, dtype=self.hmonnx_dtype),
            condition.position_ids.to(
                self.device,
                dtype=self.hmonnx_dtype if condition.position_ids.is_floating_point() else torch.int32,
            ),
            condition.domain_id.to(self.device, dtype=torch.int32),
        )
        return _as_tuple(raw)[0].detach().cpu()


@dataclass(frozen=True)
class ForwardDynamicsSegmentedDenoiserHmonnxPath:
    segments: tuple[Path, ...]

    def all(self) -> list[Path]:
        return list(self.segments)

    def as_dict(self) -> dict[str, list[str]]:
        return {"segments": [str(item) for item in self.segments]}


class ForwardDynamicsSegmentedHmonnxDenoiserRuntime:
    """HMONNX denoiser runtime for segmented fixed-shape FD stack graphs."""

    def __init__(
        self,
        transformer: torch.nn.Module,
        path: ForwardDynamicsSegmentedDenoiserHmonnxPath | list[Path] | tuple[Path, ...],
        *,
        und_seq: int,
        vision_seq: int,
        frames: int,
        latent_height: int,
        latent_width: int,
        device: str = "cuda",
        input_dtype: str = "float16",
        debug: bool = False,
        init_xhquant: bool = True,
    ) -> None:
        if not isinstance(path, ForwardDynamicsSegmentedDenoiserHmonnxPath):
            path = ForwardDynamicsSegmentedDenoiserHmonnxPath(tuple(Path(item) for item in path))
        if not path.segments:
            raise ValueError("at least one segment HMONNX path is required")
        for item in path.segments:
            _require_path(item)
        if init_xhquant:
            xhquant_init(None, debug=debug)
        self.path = path
        self.device = torch.device(device if torch.cuda.is_available() or not device.startswith("cuda") else "cpu")
        self.hmonnx_dtype = torch.float16 if input_dtype == "float16" else torch.float32
        self.sessions = [HMONNXInference(str(item)).to(self.device) for item in path.segments]
        self.head = OfficialCosmos3ForwardDynamicsHeadWrapper(
            transformer,
            und_seq=und_seq,
            vision_seq=vision_seq,
            frames=frames,
            latent_height=latent_height,
            latent_width=latent_width,
        ).eval()

    @torch.no_grad()
    def denoise(
        self,
        condition: ForwardDynamicsPackedCondition,
        latents: torch.Tensor,
        timestep_vector: torch.Tensor,
    ) -> torch.Tensor:
        packed_hidden_states: torch.Tensor | None = None
        position_ids = condition.position_ids.to(self.device, dtype=self.hmonnx_dtype)
        for index, session in enumerate(self.sessions):
            if index == 0:
                raw = session.forward(
                    condition.und_hidden_states.to(self.device, dtype=self.hmonnx_dtype),
                    latents.to(self.device, dtype=self.hmonnx_dtype),
                    condition.actions.to(self.device, dtype=self.hmonnx_dtype),
                    timestep_vector.to(self.device, dtype=self.hmonnx_dtype),
                    condition.action_timesteps.to(self.device, dtype=self.hmonnx_dtype),
                    position_ids,
                    condition.domain_id.to(self.device, dtype=torch.int32),
                )
            else:
                if packed_hidden_states is None:
                    raise RuntimeError("internal error: missing packed hidden state before stack segment")
                raw = session.forward(packed_hidden_states.to(self.device, dtype=self.hmonnx_dtype), position_ids)
            packed_hidden_states = _as_tuple(raw)[0]
        if packed_hidden_states is None:
            raise RuntimeError("segmented HMONNX runtime produced no output")
        head_device = next(self.head.parameters()).device
        head_dtype = next(self.head.parameters()).dtype
        head_input = packed_hidden_states.detach().to(device=head_device, dtype=head_dtype)
        return self.head(head_input).detach().cpu()


class ForwardDynamicsSchedulerRuntime:
    """Host scheduler loop around a forward dynamics denoiser runtime."""

    def __init__(
        self,
        denoiser_runtime: ForwardDynamicsTorchDenoiserRuntime | ForwardDynamicsHmonnxDenoiserRuntime,
        scheduler: UniPCMultistepScheduler,
        *,
        num_inference_steps: int,
        device: str = "cuda",
        scheduler_dtype: torch.dtype = torch.float32,
    ) -> None:
        self.denoiser_runtime = denoiser_runtime
        self.scheduler = scheduler
        self.num_inference_steps = int(num_inference_steps)
        self.device = torch.device(device if torch.cuda.is_available() or not device.startswith("cuda") else "cpu")
        self.scheduler_dtype = scheduler_dtype

    @torch.no_grad()
    def run(
        self,
        inputs: ForwardDynamicsSchedulerInputs,
        *,
        trace_steps: bool = True,
    ) -> ForwardDynamicsSchedulerOutput:
        vision_seq = int(inputs.condition.pack_meta["vision_seq"])
        latent_shape = tuple(inputs.initial_latents.shape)
        latents = inputs.initial_latents.detach().cpu().float()
        self.scheduler.set_timesteps(self.num_inference_steps, device=self.device)
        traces: list[ForwardDynamicsSchedulerStepTrace] = []
        for step_index, timestep in enumerate(self.scheduler.timesteps):
            timestep_vector = make_timestep_vector(
                timestep,
                vision_seq,
                device=torch.device("cpu"),
                dtype=torch.float32,
            )
            velocity = self.denoiser_runtime.denoise(inputs.condition, latents, timestep_vector).float()
            flat_next = self.scheduler.step(
                model_output=flatten_latents(velocity).to(self.device, dtype=self.scheduler_dtype),
                timestep=timestep,
                sample=flatten_latents(latents).unsqueeze(0).to(self.device, dtype=self.scheduler_dtype),
                return_dict=False,
            )[0].squeeze(0).detach().cpu().float()
            latents = unflatten_latents(flat_next, latent_shape)
            if trace_steps:
                traces.append(
                    ForwardDynamicsSchedulerStepTrace(
                        step_index=step_index,
                        timestep=float(timestep.detach().cpu().item()),
                        velocity=velocity.detach().cpu(),
                        latents=latents.detach().cpu(),
                    )
                )
        return ForwardDynamicsSchedulerOutput(final_latents=latents.detach().cpu(), steps=tuple(traces))


__all__ = [
    "ForwardDynamicsDenoiserHmonnxPath",
    "ForwardDynamicsHmonnxDenoiserRuntime",
    "ForwardDynamicsPackedCondition",
    "ForwardDynamicsSegmentedDenoiserHmonnxPath",
    "ForwardDynamicsSegmentedHmonnxDenoiserRuntime",
    "ForwardDynamicsSchedulerInputs",
    "ForwardDynamicsSchedulerOutput",
    "ForwardDynamicsSchedulerRuntime",
    "ForwardDynamicsSchedulerStepTrace",
    "ForwardDynamicsTorchDenoiserRuntime",
    "load_generator_scheduler",
    "make_forward_dynamics_scheduler_inputs",
]
