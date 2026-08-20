"""Pure MiniCPM Token2Wav graph input/output operations."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


def _values(output: Tensor | Sequence[Tensor]) -> tuple[Tensor, ...]:
    return (output,) if isinstance(output, Tensor) else tuple(output)


@dataclass(frozen=True, slots=True)
class FlowFrontendInputs:
    tokens: Tensor
    token_length: Tensor
    prompt_feat: Tensor
    prompt_feat_length: Tensor
    embedding: Tensor
    output_mel_length: int


@dataclass(slots=True)
class Token2WavExecutionReport:
    flow_frontend_count: int = 0
    flow_decoder_count: int = 0
    hift_count: int = 0
    stream_flow_count: int = 0
    stream_flow_final_count: int = 0
    stream_estimator_step_count: int = 0
    stream_hift_count: int = 0
    stream_hift_final_count: int = 0
    host_initialization_count: int = 0
    host_initialization_backend: str = "official_host"
    streaming_attached: bool = False

    @property
    def backends(self) -> dict[str, str]:
        return {
            "token2wav_flow_frontend": "hmonnx",
            "token2wav_flow_decoder": "hmonnx",
            "token2wav_hift": "hmonnx",
        }

    @property
    def execution_counts(self) -> dict[str, int]:
        return {
            "token2wav_flow_frontend": self.flow_frontend_count,
            "token2wav_flow_decoder": self.flow_decoder_count,
            "token2wav_hift": self.hift_count,
        }

    @property
    def streaming_backends(self) -> dict[str, str]:
        return {
            "token2wav_flow_stream": self.full_backends["token2wav_flow_stream"],
            "token2wav_flow_stream_final": self.full_backends["token2wav_flow_stream_final"],
            "token2wav_hift_stream": self.full_backends["token2wav_hift_stream"],
            "token2wav_hift_stream_final": self.full_backends["token2wav_hift_stream_final"],
        }

    @property
    def streaming_execution_counts(self) -> dict[str, int]:
        return {
            "token2wav_flow_stream": self.stream_flow_count,
            "token2wav_flow_stream_final": self.stream_flow_final_count,
            "token2wav_hift_stream": self.stream_hift_count,
            "token2wav_hift_stream_final": self.stream_hift_final_count,
        }

    def record_flow_frontend(self) -> None:
        self.flow_frontend_count += 1

    def record_flow_decoder(self) -> None:
        self.flow_decoder_count += 1

    def record_hift(self) -> None:
        self.hift_count += 1

    def record_stream_flow(self, final: bool) -> None:
        if final:
            self.stream_flow_final_count += 1
        else:
            self.stream_flow_count += 1

    def record_stream_estimator_step(self) -> None:
        self.stream_estimator_step_count += 1

    def record_host_initialization(self) -> None:
        self.host_initialization_count += 1

    def record_stream_hift(self, final: bool) -> None:
        if final:
            self.stream_hift_final_count += 1
        else:
            self.stream_hift_count += 1

    @property
    def full_backends(self) -> dict[str, str]:
        def stream_backend(count: int) -> str:
            return "hmonnx" if count > 0 else "uninitialized"

        return {
            **{name: "hmonnx" for name in self.execution_counts},
            "token2wav_flow_stream": stream_backend(self.stream_flow_count),
            "token2wav_flow_stream_final": stream_backend(self.stream_flow_final_count),
            "token2wav_hift_stream": stream_backend(self.stream_hift_count),
            "token2wav_hift_stream_final": stream_backend(self.stream_hift_final_count),
            "token2wav_initialization": self.host_initialization_backend,
        }

    @property
    def full_execution_counts(self) -> dict[str, int]:
        return {
            **self.execution_counts,
            "token2wav_flow_stream": self.stream_flow_count,
            "token2wav_flow_stream_final": self.stream_flow_final_count,
            "token2wav_hift_stream": self.stream_hift_count,
            "token2wav_hift_stream_final": self.stream_hift_final_count,
            "token2wav_initialization": self.host_initialization_count,
        }

    def require_streaming_components(self, names: Sequence[str]) -> None:
        """Require each requested streaming graph role to have executed."""
        missing = [name for name in names if self.full_execution_counts.get(name, 0) <= 0]
        if missing:
            raise RuntimeError(f"Token2Wav HMONNX streaming roles did not execute: {', '.join(missing)}")

    def require_full_hmonnx_execution(self) -> None:
        """Validate the active non-streaming or streaming HMONNX execution path."""
        if self.streaming_attached:
            if self.stream_flow_count + self.stream_flow_final_count <= 0:
                raise RuntimeError("Token2Wav HMONNX streaming Flow role did not execute")
            if self.stream_hift_count + self.stream_hift_final_count <= 0:
                raise RuntimeError("Token2Wav HMONNX streaming HiFT role did not execute")
            return
        missing = [name for name, count in self.execution_counts.items() if count <= 0]
        if missing:
            raise RuntimeError(f"Token2Wav HMONNX roles did not execute: {', '.join(missing)}")


def prepare_flow_frontend_inputs(
    token: Tensor,
    prompt_token: Tensor,
    prompt_feat: Tensor,
    embedding: Tensor,
    *,
    token_capacity: int,
    mel_capacity: int,
    up_rate: int,
) -> FlowFrontendInputs:
    tokens = torch.cat((prompt_token, token), dim=1)
    if torch.any(tokens < 0):
        raise RuntimeError("Token2Wav token ids must be non-negative")
    token_length = int(tokens.shape[1])
    if token_length > token_capacity:
        raise RuntimeError(f"Token2Wav token length {token_length} exceeds exported capacity {token_capacity}")
    if prompt_feat.shape[1] > mel_capacity:
        raise RuntimeError(
            f"Token2Wav prompt mel length {prompt_feat.shape[1]} exceeds exported capacity {mel_capacity}"
        )
    output_mel_length = token_length * up_rate - int(prompt_feat.shape[1])
    if output_mel_length <= 0:
        raise RuntimeError("Token2Wav generated mel length must be positive")
    padded_tokens = F.pad(tokens.to(torch.int32), (0, token_capacity - token_length))
    padded_prompt = F.pad(prompt_feat.to(torch.float16), (0, 0, 0, mel_capacity - prompt_feat.shape[1]))
    return FlowFrontendInputs(
        tokens=padded_tokens,
        token_length=torch.tensor([token_length], dtype=torch.int32, device=tokens.device),
        prompt_feat=padded_prompt,
        prompt_feat_length=torch.tensor([prompt_feat.shape[1]], dtype=torch.int32, device=prompt_feat.device),
        embedding=embedding.to(torch.float16),
        output_mel_length=output_mel_length,
    )


def run_cfm(
    decoder: Callable[..., Tensor | Sequence[Tensor]],
    *,
    mu: Tensor,
    mask: Tensor,
    spks: Tensor,
    cond: Tensor,
    noise: Tensor,
    n_timesteps: int,
    cfg_rate: float,
    on_decoder_call: Callable[[], None] | None = None,
) -> Tensor:
    x = noise
    t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device, dtype=mu.dtype)
    t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
    t = t_span[0].unsqueeze(0)
    dt = t_span[1] - t_span[0]
    mask_in = torch.cat((mask, mask), dim=0)
    mu_in = torch.cat((mu, torch.zeros_like(mu)), dim=0)
    spks_in = torch.cat((spks, torch.zeros_like(spks)), dim=0)
    cond_in = torch.cat((cond, torch.zeros_like(cond)), dim=0)
    for step in range(1, len(t_span)):
        output = decoder(
            torch.cat((x, x), dim=0),
            mask_in,
            mu_in,
            torch.cat((t, t), dim=0),
            spks_in,
            cond_in,
        )
        if on_decoder_call is not None:
            on_decoder_call()
        derivative = output if isinstance(output, Tensor) else output[0]
        conditional, unconditional = derivative.chunk(2, dim=0)
        x = x + dt * ((1.0 + cfg_rate) * conditional - cfg_rate * unconditional)
        t = t + dt
        if step < len(t_span) - 1:
            dt = t_span[step + 1] - t
    return x


def prepare_hift_input(mel: Tensor, *, frame_capacity: int) -> Tensor:
    if mel.shape[2] > frame_capacity:
        raise RuntimeError(f"HiFT mel length {mel.shape[2]} exceeds exported capacity {frame_capacity}")
    return F.pad(mel, (0, frame_capacity - mel.shape[2]))


def crop_hift_waveform(waveform: Tensor, *, mel_frames: int, hop_length: int = 480) -> Tensor:
    return waveform[:, : mel_frames * hop_length]
