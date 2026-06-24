# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0

"""Host pipeline runtime that stitches generator scheduler output into VAE decode."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from runtime.generator_scheduler_runtime import (
    GeneratorCfgSchedulerRuntime,
    GeneratorSchedulerInputs,
    GeneratorSchedulerOutput,
)
from runtime.vae_decoder_runtime import VaeDecoderFullHmonnxRuntime


@dataclass(frozen=True)
class GeneratorVaePipelineOutput:
    scheduler: GeneratorSchedulerOutput
    decoded_sample: torch.Tensor

    @property
    def final_latents(self) -> torch.Tensor:
        return self.scheduler.final_latents


class GeneratorVaePipelineRuntime:
    """Run host CFG scheduler and decode its final latents with VAE HMONNX."""

    def __init__(
        self,
        scheduler_runtime: GeneratorCfgSchedulerRuntime,
        vae_runtime: VaeDecoderFullHmonnxRuntime,
    ) -> None:
        self.scheduler_runtime = scheduler_runtime
        self.vae_runtime = vae_runtime

    @torch.no_grad()
    def run(self, inputs: GeneratorSchedulerInputs, *, trace_steps: bool = True) -> GeneratorVaePipelineOutput:
        scheduler_output = self.scheduler_runtime.run(inputs, trace_steps=trace_steps)
        decoded_sample = self.vae_runtime.decode(scheduler_output.final_latents)
        return GeneratorVaePipelineOutput(scheduler=scheduler_output, decoded_sample=decoded_sample)
