"""Learned CSA/HCA compressor wrappers built on the fixed state transitions."""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor, nn

from ._trace import is_fx_proxy
from .attention import InterleavedPartialRope
from .norm import build_xh_rms_norm, rms_norm_from_hf
from .static_cache import (
    CompressorStepOutput,
    NonOverlappingCompressorStep,
    OverlappingCompressorStep,
)


class LearnedCompressorOutput(NamedTuple):
    compressed: Tensor
    compressed_valid: Tensor
    next_kv_state: Tensor
    next_score_state: Tensor
    new_count: Tensor


class _LearnedCompressorBase(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        head_dim: int,
        input_sequence_length: int,
        ratio: int,
        projection_factor: int,
        rms_norm_eps: float,
        rope_dim: int,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.head_dim = int(head_dim)
        self.input_sequence_length = int(input_sequence_length)
        self.ratio = int(ratio)
        self.projection_factor = int(projection_factor)
        self.rope_dim = int(rope_dim)
        self.partial_rope = InterleavedPartialRope(self.rope_dim)
        projection_width = self.projection_factor * self.head_dim
        self.kv_proj = nn.Linear(self.hidden_size, projection_width, bias=False, dtype=torch.float16)
        self.gate_proj = nn.Linear(self.hidden_size, projection_width, bias=False, dtype=torch.float16)
        # ``from_hf`` replaces this tensor with the checkpoint parameter.  The
        # plain constructor is also used by graph/unit smoke tests, however,
        # and an uninitialised bias can randomly push every real candidate
        # below the finite fp16 invalid sentinel.  That makes a completed
        # compressed row pool only the zero-valued history state.  Zero is the
        # neutral, deterministic fallback until checkpoint weights are bound.
        self.position_bias = nn.Parameter(torch.zeros(self.ratio, projection_width, dtype=torch.float16))
        self.kv_norm: nn.Module = build_xh_rms_norm(
            self.head_dim,
            rms_norm_eps,
            weight=torch.ones(self.head_dim, dtype=torch.float16),
        )

    def _project_and_finish(
        self,
        hidden_states: Tensor,
        step: nn.Module,
        kv_state: Tensor,
        score_state: Tensor,
        offset: Tensor,
        phase_indices: Tensor,
        current_length: Tensor,
        compressor_validity: Tensor,
        compressor_new_count: Tensor,
        compressed_cos: Tensor,
        compressed_sin: Tensor,
    ) -> LearnedCompressorOutput:
        if not is_fx_proxy(hidden_states):
            expected = (
                hidden_states.shape[0],
                self.input_sequence_length,
                self.hidden_size,
            )
            if tuple(hidden_states.shape) != expected:
                raise ValueError(f"hidden_states must have shape {expected}")
        kv = self.kv_proj(hidden_states)
        score = self.gate_proj(hidden_states)
        result: CompressorStepOutput = step(
            kv,
            score,
            kv_state,
            score_state,
            offset,
            phase_indices,
            current_length,
            compressor_validity,
            compressor_new_count,
            self.position_bias,
        )
        pooled = self.kv_norm(result.pooled)
        pooled = self.partial_rope(
            pooled.unsqueeze(2),
            compressed_cos,
            compressed_sin,
        ).squeeze(2)
        return LearnedCompressorOutput(
            pooled,
            result.pooled_valid,
            result.next_kv_state,
            result.next_score_state,
            result.new_count,
        )

    @torch.no_grad()
    def load_hf_weights(self, compressor: nn.Module) -> None:
        self.kv_proj.load_state_dict(compressor.kv_proj.state_dict())
        self.gate_proj.load_state_dict(compressor.gate_proj.state_dict())
        self.position_bias.copy_(compressor.position_bias)
        self.kv_norm.load_state_dict(compressor.kv_norm.state_dict())


class LearnedOverlappingCompressor(_LearnedCompressorBase):
    """CSA C4 compressor producing [B,ceil(P/4),D]."""

    def __init__(
        self,
        *,
        hidden_size: int,
        head_dim: int,
        input_sequence_length: int,
        ratio: int = 4,
        rms_norm_eps: float = 1e-6,
        rope_dim: int = 64,
    ) -> None:
        super().__init__(
            hidden_size=hidden_size,
            head_dim=head_dim,
            input_sequence_length=input_sequence_length,
            ratio=ratio,
            projection_factor=2,
            rms_norm_eps=rms_norm_eps,
            rope_dim=rope_dim,
        )
        self.step = OverlappingCompressorStep(
            input_sequence_length=input_sequence_length,
            head_dim=head_dim,
            ratio=ratio,
        )

    @classmethod
    def from_hf(
        cls,
        compressor: nn.Module,
        *,
        input_sequence_length: int,
    ) -> "LearnedOverlappingCompressor":
        config = compressor.rotary_emb.config
        module = cls(
            hidden_size=config.hidden_size,
            head_dim=config.head_dim,
            input_sequence_length=input_sequence_length,
            ratio=compressor.compress_rate,
            rms_norm_eps=config.rms_norm_eps,
            rope_dim=config.qk_rope_head_dim,
        )
        module.kv_proj = compressor.kv_proj
        module.gate_proj = compressor.gate_proj
        module.position_bias = compressor.position_bias
        module.kv_norm = rms_norm_from_hf(compressor.kv_norm)
        return module

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float16,
    ) -> tuple[Tensor, Tensor]:
        return self.step.initial_state(batch_size, device=device, dtype=dtype)

    def forward(
        self,
        hidden_states: Tensor,
        kv_state: Tensor,
        score_state: Tensor,
        offset: Tensor,
        phase_indices: Tensor,
        current_length: Tensor,
        compressor_validity: Tensor,
        compressor_new_count: Tensor,
        compressed_cos: Tensor,
        compressed_sin: Tensor,
    ) -> LearnedCompressorOutput:
        return self._project_and_finish(
            hidden_states,
            self.step,
            kv_state,
            score_state,
            offset,
            phase_indices,
            current_length,
            compressor_validity,
            compressor_new_count,
            compressed_cos,
            compressed_sin,
        )


class LearnedNonOverlappingCompressor(_LearnedCompressorBase):
    """HCA C128 compressor producing [B,ceil(P/128),D]."""

    def __init__(
        self,
        *,
        hidden_size: int,
        head_dim: int,
        input_sequence_length: int,
        ratio: int = 128,
        rms_norm_eps: float = 1e-6,
        rope_dim: int = 64,
    ) -> None:
        super().__init__(
            hidden_size=hidden_size,
            head_dim=head_dim,
            input_sequence_length=input_sequence_length,
            ratio=ratio,
            projection_factor=1,
            rms_norm_eps=rms_norm_eps,
            rope_dim=rope_dim,
        )
        self.step = NonOverlappingCompressorStep(
            input_sequence_length=input_sequence_length,
            head_dim=head_dim,
            ratio=ratio,
        )

    @classmethod
    def from_hf(
        cls,
        compressor: nn.Module,
        *,
        input_sequence_length: int,
    ) -> "LearnedNonOverlappingCompressor":
        config = compressor.rotary_emb.config
        module = cls(
            hidden_size=config.hidden_size,
            head_dim=config.head_dim,
            input_sequence_length=input_sequence_length,
            ratio=compressor.compress_rate,
            rms_norm_eps=config.rms_norm_eps,
            rope_dim=config.qk_rope_head_dim,
        )
        module.kv_proj = compressor.kv_proj
        module.gate_proj = compressor.gate_proj
        module.position_bias = compressor.position_bias
        module.kv_norm = rms_norm_from_hf(compressor.kv_norm)
        return module

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float16,
    ) -> tuple[Tensor, Tensor]:
        return self.step.initial_state(batch_size, device=device, dtype=dtype)

    def forward(
        self,
        hidden_states: Tensor,
        kv_state: Tensor,
        score_state: Tensor,
        offset: Tensor,
        phase_indices: Tensor,
        current_length: Tensor,
        compressor_validity: Tensor,
        compressor_new_count: Tensor,
        compressed_cos: Tensor,
        compressed_sin: Tensor,
    ) -> LearnedCompressorOutput:
        return self._project_and_finish(
            hidden_states,
            self.step,
            kv_state,
            score_state,
            offset,
            phase_indices,
            current_length,
            compressor_validity,
            compressor_new_count,
            compressed_cos,
            compressed_sin,
        )


__all__ = [
    "LearnedCompressorOutput",
    "LearnedNonOverlappingCompressor",
    "LearnedOverlappingCompressor",
]
