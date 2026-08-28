from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def cordic_atan2(y: Tensor, x: Tensor, iterations: int = 16) -> Tensor:
    """atan2 approximation using a fixed, export-friendly CORDIC graph."""

    if iterations <= 0:
        raise ValueError("iterations must be positive")
    original_y = y
    negative_x = x < 0
    x = torch.where(negative_x, -x, x)
    y = torch.where(negative_x, -y, y)
    angle = torch.where(
        negative_x,
        torch.where(
            original_y >= 0,
            torch.full_like(x, math.pi),
            torch.full_like(x, -math.pi),
        ),
        torch.zeros_like(x),
    )
    for index in range(iterations):
        direction = torch.where(
            y > 0,
            torch.ones_like(y),
            torch.where(y < 0, -torch.ones_like(y), torch.zeros_like(y)),
        )
        scale = 2.0**-index
        previous_x = x
        x = x + direction * y * scale
        y = y - direction * previous_x * scale
        angle = angle + direction * math.atan(scale)
    return angle


class StaticSTFT20(nn.Module):
    """Fixed-shape n_fft=20/hop=5 STFT expressed as two Conv1d calls."""

    N_FFT = 20
    HOP_LENGTH = 5
    FREQUENCY_BINS = N_FFT // 2 + 1

    def __init__(
        self,
        pad_mode: str = "constant",
        waveform_length: int | None = None,
        phase_mode: str = "atan2",
    ) -> None:
        super().__init__()
        if pad_mode not in {"constant", "reflect", "replicate", "length_aware_reflect"}:
            raise ValueError("pad_mode must be constant, reflect, replicate, or length_aware_reflect")
        if pad_mode == "length_aware_reflect" and (waveform_length is None or waveform_length <= 0):
            raise ValueError("positive waveform_length is required for length_aware_reflect")
        self.pad_mode = pad_mode
        self.waveform_length = waveform_length
        if phase_mode not in {"atan2", "cordic"}:
            raise ValueError("phase_mode must be atan2 or cordic")
        self.phase_mode = phase_mode
        window = torch.hann_window(self.N_FFT, periodic=True, dtype=torch.float64)
        frequency = torch.arange(self.FREQUENCY_BINS, dtype=torch.float64).unsqueeze(1)
        time = torch.arange(self.N_FFT, dtype=torch.float64).unsqueeze(0)
        angle = 2.0 * math.pi * frequency * time / self.N_FFT
        self.register_buffer(
            "real_weight",
            (torch.cos(angle) * window).float().unsqueeze(1),
        )
        self.register_buffer(
            "imag_weight",
            (-torch.sin(angle) * window).float().unsqueeze(1),
        )
        if waveform_length is not None:
            padding = self.N_FFT // 2
            self.register_buffer(
                "padded_positions",
                torch.arange(-padding, waveform_length + padding, dtype=torch.int64),
            )

    def forward(self, waveform: Tensor, valid_samples: Tensor | None = None) -> Tensor:
        if waveform.ndim == 3:
            if not torch.jit.is_tracing() and waveform.shape[-1] != 1:
                raise ValueError("3-D waveform must have shape [B,N,1]")
            waveform = waveform.transpose(1, 2)
        elif waveform.ndim == 2:
            waveform = waveform.unsqueeze(1)
        else:
            raise ValueError("waveform must have shape [B,N] or [B,N,1]")
        padding = self.N_FFT // 2
        if self.pad_mode == "length_aware_reflect":
            if not torch.jit.is_tracing() and waveform.shape[-1] != self.waveform_length:
                raise ValueError(f"waveform length must be {self.waveform_length}, got {waveform.shape[-1]}")
            if valid_samples is None:
                raise ValueError("valid_samples is required for length_aware_reflect")
            length = valid_samples.reshape(-1)[0].to(dtype=torch.int64)
            positions = self.padded_positions
            indices = torch.where(
                positions < 0,
                -positions,
                torch.where(positions < length, positions, 2 * length - 2 - positions),
            )
            maximum_index = int(self.waveform_length) - 1
            indices = torch.where(indices < 0, torch.zeros_like(indices), indices)
            indices = torch.where(
                indices > maximum_index,
                torch.full_like(indices, maximum_index),
                indices,
            )
            waveform = torch.index_select(waveform, 2, indices)
            active = (positions < length + padding).reshape(1, 1, -1)
            waveform = torch.where(active, waveform, torch.zeros_like(waveform))
        elif self.pad_mode == "constant":
            waveform = F.pad(waveform, (padding, padding), mode="constant", value=0.0)
        else:
            waveform = F.pad(waveform, (padding, padding), mode=self.pad_mode)
        real = F.conv1d(waveform, self.real_weight, stride=self.HOP_LENGTH)
        imag = F.conv1d(waveform, self.imag_weight, stride=self.HOP_LENGTH)
        power = real.square() + imag.square()
        magnitude = torch.sqrt(power)
        raw_phase = torch.atan2(imag, real) if self.phase_mode == "atan2" else cordic_atan2(imag, real)
        phase = torch.where(power > 1e-12, raw_phase, torch.zeros_like(real))
        return torch.cat([magnitude, phase], dim=1)


class StaticISTFT20(nn.Module):
    """Exact fixed-length real iSTFT expressed as linear overlap-add."""

    N_FFT = StaticSTFT20.N_FFT
    HOP_LENGTH = StaticSTFT20.HOP_LENGTH
    FREQUENCY_BINS = StaticSTFT20.FREQUENCY_BINS

    def __init__(self, frame_max_length: int) -> None:
        super().__init__()
        if frame_max_length <= 0:
            raise ValueError("frame_max_length must be positive")
        self.frame_max_length = int(frame_max_length)
        self.waveform_length = 600 * self.frame_max_length
        self.spectral_frames = self.waveform_length // self.HOP_LENGTH + 1

        frequency = torch.arange(self.FREQUENCY_BINS, dtype=torch.float64).unsqueeze(1)
        time = torch.arange(self.N_FFT, dtype=torch.float64).unsqueeze(0)
        angle = 2.0 * math.pi * frequency * time / self.N_FFT
        scale = torch.full(
            (self.FREQUENCY_BINS, 1),
            2.0 / self.N_FFT,
            dtype=torch.float64,
        )
        scale[0] = 1.0 / self.N_FFT
        scale[-1] = 1.0 / self.N_FFT
        window = torch.hann_window(self.N_FFT, periodic=True, dtype=torch.float64)
        real_weight = scale * torch.cos(angle) * window
        imag_weight = -scale * torch.sin(angle) * window
        self.register_buffer("real_weight", real_weight.float().unsqueeze(1))
        self.register_buffer("imag_weight", imag_weight.float().unsqueeze(1))

        envelope = F.conv_transpose1d(
            torch.ones(1, 1, self.spectral_frames, dtype=torch.float64),
            window.square().reshape(1, 1, -1),
            stride=self.HOP_LENGTH,
        )
        center = self.N_FFT // 2
        envelope = envelope[:, :, center : center + self.waveform_length]
        self.register_buffer("envelope", envelope.clamp_min(1e-11).float())

    def forward(self, spec_phase: Tensor, waveform_mask: Tensor | None = None) -> Tensor:
        if spec_phase.ndim != 3:
            raise ValueError("spec_phase must have shape [B,22,120F+1]")
        if not torch.jit.is_tracing() and spec_phase.shape[1] != 2 * self.FREQUENCY_BINS:
            raise ValueError(f"spec_phase channel count must be {2 * self.FREQUENCY_BINS}")
        if not torch.jit.is_tracing() and spec_phase.shape[2] != self.spectral_frames:
            raise ValueError(f"spec_phase frame count must be {self.spectral_frames}")
        magnitude = spec_phase[:, : self.FREQUENCY_BINS]
        phase = spec_phase[:, self.FREQUENCY_BINS :]
        real = magnitude * torch.cos(phase)
        imag = magnitude * torch.sin(phase)
        waveform = F.conv_transpose1d(
            real,
            self.real_weight,
            stride=self.HOP_LENGTH,
        )
        waveform = waveform + F.conv_transpose1d(
            imag,
            self.imag_weight,
            stride=self.HOP_LENGTH,
        )
        center = self.N_FFT // 2
        waveform = waveform[:, :, center : center + self.waveform_length]
        waveform = waveform / self.envelope
        if waveform_mask is not None:
            waveform = waveform * waveform_mask
        # ``squeeze(1)`` lowers to an ONNX If because the legacy exporter
        # guards the channel dimension dynamically.  This channel is fixed to
        # one by construction, so an explicit index is exactly equivalent and
        # keeps the deployment graph purely static.
        return waveform[:, 0, :]


__all__ = ["StaticISTFT20", "StaticSTFT20", "cordic_atan2"]
