"""MiniCPM Token2Wav streaming cache state and fixed-graph transforms.

These rules are specific to MiniCPM's exported Flow/HiFT streaming ABI.  They
are isolated from graph ownership so the runtime class only orchestrates roles.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from .runtime_token2wav_ops import _values


EstimatorStepSession = Callable[..., Tensor | Sequence[Tensor]]


FLOW_CACHE_NAMES = (
    "conformer_cnn_cache",
    "conformer_att_cache",
    "estimator_cnn_cache",
    "estimator_att_cache",
)
FLOW_CACHE_AXES = {"conformer_att_cache": 3, "estimator_att_cache": 4}
# Official Token2wav right-pads speech tokens with this silence token; the fixed-capacity
# streaming Flow graph reuses it for short final chunks.
STREAM_SILENCE_TOKEN = 4218


@dataclass(slots=True)
class StreamState:
    """Runtime-owned mutable state for one official Token2wav stream."""

    flow_cache: dict[str, Tensor]
    flow_valid_lengths: dict[str, int]
    hift_cache: dict[str, Tensor]
    hift_valid_lengths: dict[str, int]
    prompt_mel_length: int
    estimator_cnn_banks: Tensor
    estimator_att_banks: Tensor
    att_cache_capacity: int
    append_capacity: int
    host_timestep_cache_banks: int

    @classmethod
    def from_base_cache(
        cls,
        base_cache: Mapping[str, Tensor],
        prompt_mel_length: int,
        *,
        att_cache_capacity: int | None = None,
        append_capacity: int = 56,
        host_timestep_cache_banks: int = 10,
    ) -> StreamState:
        flow_cache = {name: value.detach().clone() for name, value in base_cache.items()}
        return cls(
            flow_cache=flow_cache,
            flow_valid_lengths={name: int(base_cache[name].shape[FLOW_CACHE_AXES[name]]) for name in FLOW_CACHE_AXES},
            hift_cache={
                "mel": next(iter(base_cache.values())).new_zeros((1, 80, 0)),
                "source": next(iter(base_cache.values())).new_zeros((1, 1, 0)),
                "speech": next(iter(base_cache.values())).new_zeros((1, 0)),
            },
            hift_valid_lengths={"mel": 0, "source": 0, "speech": 0},
            prompt_mel_length=prompt_mel_length,
            estimator_cnn_banks=base_cache["estimator_cnn_cache"][:host_timestep_cache_banks].detach().clone(),
            estimator_att_banks=base_cache["estimator_att_cache"][:host_timestep_cache_banks].detach().clone(),
            att_cache_capacity=int(att_cache_capacity) if att_cache_capacity is not None else 0,
            append_capacity=int(append_capacity),
            host_timestep_cache_banks=int(host_timestep_cache_banks),
        )

    def reset(self, base_cache: Mapping[str, Tensor]) -> None:
        """Restore the prompt cache and official zero-length HiFT tails."""
        self.flow_cache = {name: value.detach().clone() for name, value in base_cache.items()}
        self.flow_valid_lengths = {name: int(base_cache[name].shape[FLOW_CACHE_AXES[name]]) for name in FLOW_CACHE_AXES}
        device = next(iter(self.flow_cache.values())).device
        dtype = next(iter(self.flow_cache.values())).dtype
        self.hift_cache = {
            "mel": torch.zeros((1, 80, 0), device=device, dtype=dtype),
            "source": torch.zeros((1, 1, 0), device=device, dtype=dtype),
            "speech": torch.zeros((1, 0), device=device, dtype=dtype),
        }
        self.hift_valid_lengths = {"mel": 0, "source": 0, "speech": 0}
        bank_count = self.host_timestep_cache_banks
        self.estimator_cnn_banks = base_cache["estimator_cnn_cache"][:bank_count].detach().clone()
        self.estimator_att_banks = base_cache["estimator_att_cache"][:bank_count].detach().clone()

    def _check_lengths(self) -> None:
        for name, length in self.flow_valid_lengths.items():
            if length < 0 or length > self.flow_cache[name].shape[FLOW_CACHE_AXES[name]]:
                raise RuntimeError(f"invalid Flow cache valid length for {name}: {length}")
        for name, length in self.hift_valid_lengths.items():
            axis = {"mel": 2, "source": 2, "speech": 1}[name]
            if length < 0 or length > self.hift_cache[name].shape[axis]:
                raise RuntimeError(f"invalid HiFT cache valid length for {name}: {length}")


def apply_streaming_waveform_overlap(
    raw_waveform: Tensor,
    *,
    raw_valid_length: int,
    past_speech: Tensor,
    past_speech_valid_length: int,
    speech_window: Tensor,
    overlap: int,
    final: bool,
) -> tuple[Tensor, Tensor, int]:
    if raw_waveform.ndim != 2 or raw_waveform.shape[0] != 1:
        raise RuntimeError(f"streaming HiFT raw waveform must have shape [1, samples], got {tuple(raw_waveform.shape)}")
    if not 0 < raw_valid_length <= raw_waveform.shape[1]:
        raise RuntimeError(
            f"streaming HiFT raw waveform valid length {raw_valid_length} exceeds capacity {raw_waveform.shape[1]}"
        )
    if tuple(past_speech.shape) != (1, overlap):
        raise RuntimeError(f"streaming HiFT speech tail shape {tuple(past_speech.shape)} != {(1, overlap)}")
    if not 0 <= past_speech_valid_length <= overlap:
        raise RuntimeError(f"invalid streaming HiFT speech tail valid length: {past_speech_valid_length}")
    if raw_valid_length < overlap:
        raise RuntimeError(f"streaming HiFT raw waveform length {raw_valid_length} is shorter than overlap {overlap}")
    if speech_window.numel() != overlap * 2:
        raise RuntimeError(f"streaming HiFT overlap window length {speech_window.numel()} != {overlap * 2}")

    raw = raw_waveform[:, :raw_valid_length]
    if past_speech_valid_length == 0:
        if final:
            return raw, torch.zeros_like(past_speech), 0
        emitted = torch.cat((raw.new_zeros((1, overlap)), raw[:, :-overlap]), dim=1)
        return emitted, raw[:, -overlap:], overlap
    past = past_speech
    if past_speech_valid_length < overlap:
        past = past.clone()
        past[:, : overlap - past_speech_valid_length] = 0
    window = speech_window.to(device=raw.device, dtype=raw.dtype)
    blended_head = raw[:, :overlap] * window[:overlap].reshape(1, -1) + past * window[overlap:].reshape(1, -1)
    blended = torch.cat((blended_head, raw[:, overlap:]), dim=1)
    if final:
        return blended, torch.zeros_like(past_speech), 0
    return blended[:, :-overlap], blended[:, -overlap:], overlap


def _truncate_streaming_cache(cache: Tensor, *, prompt_length: int, tail: int, capacity: int) -> Tensor:
    """Official token2wav.stream() cache truncation: keep the prompt in full plus
    the most recent ``tail`` frames, bounded by the exported graph capacity.

    Conformer caches are 5-D [..., frames, dim] (axis 3); estimator banks are
    6-D [steps, ..., frames, dim] (axis 4)."""
    axis = 4 if cache.dim() == 6 else 3
    length = int(cache.shape[axis])
    if length <= capacity:
        return cache
    prompt = cache.narrow(axis, 0, prompt_length)
    recent = cache.narrow(axis, length - tail, tail)
    bounded = torch.cat((prompt, recent), dim=axis)
    return bounded


def pack_streaming_attention_cache(
    cache: Tensor,
    *,
    valid_length: int,
    capacity: int,
    axis: int,
) -> Tensor:
    """Right-align a compact logical cache in a fixed-capacity graph input."""
    if valid_length < 0 or valid_length > capacity:
        raise RuntimeError(_token2wav_capacity_guide(valid_length, capacity))
    if cache.shape[axis] < valid_length:
        raise RuntimeError(
            f"Token2Wav cache axis {axis} has {cache.shape[axis]} frames, shorter than valid length {valid_length}"
        )
    valid = cache.narrow(axis, 0, valid_length)
    padding_shape = list(valid.shape)
    padding_shape[axis] = capacity - valid_length
    padding = valid.new_zeros(padding_shape)
    return torch.cat((padding, valid), dim=axis)


def pack_conformer_attention_cache(
    cache: Tensor,
    *,
    valid_length: int,
    capacity: int,
    base_layer_count: int,
) -> Tensor:
    """Pack the mixed-resolution Conformer cache for the fixed streaming graph.

    Base Conformer layers store two copies of a half-rate cache. Up-Conformer
    layers store one full-rate mel cache. Both are right-aligned so graph-side
    masks can hide leading fixed-capacity padding without changing distances
    between valid keys and the current chunk.
    """
    if valid_length % 2 != 0 or capacity % 2 != 0:
        raise RuntimeError(f"Conformer cache lengths must be even, got valid={valid_length}, capacity={capacity}")
    if not 0 < base_layer_count < cache.shape[0]:
        raise RuntimeError(f"invalid base Conformer layer count: {base_layer_count}")
    base_half_valid = valid_length // 2
    base_half_capacity = capacity // 2
    base_half = cache[:base_layer_count, :, :, :base_half_valid, :]
    packed_base_half = pack_streaming_attention_cache(
        base_half,
        valid_length=base_half_valid,
        capacity=base_half_capacity,
        axis=3,
    )
    packed_base = packed_base_half.repeat(1, 1, 1, 2, 1)
    packed_up = pack_streaming_attention_cache(
        cache[base_layer_count:],
        valid_length=valid_length,
        capacity=capacity,
        axis=3,
    )
    return torch.cat((packed_base, packed_up), dim=0)


def compact_conformer_attention_cache(
    cache: Tensor,
    *,
    past_valid_length: int,
    current_valid_length: int,
    input_capacity: int,
    base_layer_count: int,
) -> Tensor:
    """Remove fixed input/current padding from a raw Conformer graph present."""
    if past_valid_length % 2 != 0 or current_valid_length % 2 != 0 or input_capacity % 2 != 0:
        raise RuntimeError(
            "Conformer past/current/capacity lengths must all be even: "
            f"{past_valid_length}/{current_valid_length}/{input_capacity}"
        )
    raw_half_length = cache.shape[3] // 2
    base_input_capacity = input_capacity // 2
    past_half_length = past_valid_length // 2
    current_half_length = current_valid_length // 2
    base_half = cache[:base_layer_count, :, :, :raw_half_length, :]
    base_past = base_half.narrow(3, base_input_capacity - past_half_length, past_half_length)
    base_current = base_half.narrow(3, base_input_capacity, current_half_length)
    compact_base_half = torch.cat((base_past, base_current), dim=3)
    compact_base = compact_base_half.repeat(1, 1, 1, 2, 1)

    up = cache[base_layer_count:]
    up_past = up.narrow(3, input_capacity - past_valid_length, past_valid_length)
    up_current = up.narrow(3, input_capacity, current_valid_length)
    compact_up = torch.cat((up_past, up_current), dim=3)
    return torch.cat((compact_base, compact_up), dim=0)


def compact_estimator_attention_cache(
    cache: Tensor,
    *,
    past_valid_length: int,
    current_valid_length: int,
    input_capacity: int,
    frame_capacity: int,
) -> Tensor:
    """Restore the official estimator cache order: current valid frames, then past."""
    current = cache.narrow(3, 0, current_valid_length)
    past = cache.narrow(3, frame_capacity + input_capacity - past_valid_length, past_valid_length)
    return torch.cat((current, past), dim=3)


def bound_streaming_attention_cache(
    cache: Tensor,
    *,
    valid_length: int,
    capacity: int,
    prompt_length: int,
) -> Tensor:
    if valid_length > capacity:
        raise RuntimeError(_token2wav_capacity_guide(valid_length, capacity))
    valid = cache.narrow(3, 0, min(valid_length, cache.shape[3]))
    prompt = valid.narrow(3, 0, min(prompt_length, valid.shape[3]))
    tail_length = min(capacity - prompt.shape[3], max(valid.shape[3] - prompt.shape[3], 0))
    tail = valid.narrow(3, valid.shape[3] - tail_length, tail_length)
    bounded = torch.cat((prompt, tail), dim=3)
    return F.pad(bounded, (0, 0, 0, capacity - bounded.shape[3]))


def _token2wav_capacity_guide(required_length: int, capacity: int) -> str:
    """Build the re-export guidance for a Token2Wav streaming capacity overflow."""
    return (
        f"Token2Wav streaming cache length {required_length} exceeds exported "
        f"capacity {capacity} (reference audio too long or stream grew past the "
        "exported window).\n"
        "To support a longer reference prompt, increase the Token2Wav streaming "
        "capacities in the workflow config and re-export:\n"
        "  configs_merak/workflows/xh2a/llm_models/minicpm_o_4_5/<target-config>.yaml\n"
        f"    streaming.prompt_token_capacity:  {max(required_length // 2 + 3, 424)} (tokens)\n"
        f"    streaming.prompt_mel_capacity:    {max(required_length, 842)} (mel)\n"
        f"    streaming.cache_shapes.conformer_att_cache[3]: {max(required_length + 100, 942)} "
        "(mel, = prompt + 100 tail)\n"
        "    streaming.cache_shapes.estimator_att_cache[4]: use the same cache capacity\n"
        "    token2wav_flow_decoder.streaming.estimator_step_cache_shapes.input_att[3]: "
        "use the same cache capacity\n"
        "    token2wav_flow_decoder.streaming.estimator_step_cache_shapes.output_att[3]: "
        "cache capacity + append_capacity\n"
        "then re-run the export workflow and re-verify with the new artifacts."
    )


def _read_present_valid_length(value: Tensor, *, name: str, capacity: int) -> int:
    if value.numel() != 1:
        raise RuntimeError(f"Token2Wav {name} must be a scalar tensor, got shape {tuple(value.shape)}")
    valid_length = int(value.reshape(()).item())
    if valid_length < 0:
        raise RuntimeError(f"Token2Wav {name}={valid_length} must be non-negative")
    # The graph reports its pre-truncation logical length. The Host immediately
    # keeps only prompt + recent-tail frames, so the stored valid length must be
    # bounded to the capacity of that truncated cache.
    return min(valid_length, capacity)


def run_stream_cfm(
    estimator_step: EstimatorStepSession,
    *,
    mu: Tensor,
    spks: Tensor,
    cond: Tensor,
    noise: Tensor,
    n_timesteps: int,
    cfg_rate: float,
    estimator_cnn_banks: Tensor,
    estimator_att_banks: Tensor,
    frame_capacity: int | None = None,
    estimator_att_capacity: int | None = None,
    on_estimator_call: Callable[[], None] | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    output_length = mu.shape[2]
    capacity = output_length if frame_capacity is None else frame_capacity
    if output_length > capacity:
        raise RuntimeError(f"streaming Flow length {output_length} exceeds estimator capacity {capacity}")
    if output_length < capacity:
        padding = (0, capacity - output_length)
        mu = F.pad(mu, padding)
        cond = F.pad(cond, padding)
        noise = F.pad(noise, padding)
    x = noise
    present_cnn_banks: list[Tensor] = []
    present_att_banks: list[Tensor] = []
    past_cnn_banks = estimator_cnn_banks.to(mu.dtype)
    past_att_banks = estimator_att_banks.to(mu.dtype)
    past_att_valid_length = int(past_att_banks.shape[4])
    att_capacity = past_att_valid_length if estimator_att_capacity is None else estimator_att_capacity
    t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device, dtype=mu.dtype)
    t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
    t = t_span[0].reshape(1)
    dt = t_span[1] - t_span[0]
    mu_cfg = torch.cat((mu, torch.zeros_like(mu)), dim=0)
    spks_cfg = torch.cat((spks, torch.zeros_like(spks)), dim=0)
    cond_cfg = torch.cat((cond, torch.zeros_like(cond)), dim=0)
    for index in range(n_timesteps):
        values = _values(
            estimator_step(
                torch.cat((x, x), dim=0),
                mu_cfg,
                t.repeat(2),
                spks_cfg,
                cond_cfg,
                past_cnn_banks[index],
                pack_streaming_attention_cache(
                    past_att_banks[index],
                    valid_length=past_att_valid_length,
                    capacity=att_capacity,
                    axis=3,
                ),
                torch.tensor([past_att_valid_length], dtype=torch.int32, device=mu.device),
                torch.tensor([output_length], dtype=torch.int32, device=mu.device),
            )
        )
        if len(values) != 3:
            raise RuntimeError(f"streaming estimator step returned {len(values)} tensors, expected 3")
        derivative, present_cnn, present_att = values
        present_cnn_banks.append(present_cnn)
        present_att_banks.append(
            compact_estimator_attention_cache(
                present_att,
                past_valid_length=past_att_valid_length,
                current_valid_length=output_length,
                input_capacity=att_capacity,
                frame_capacity=capacity,
            )
        )
        conditional, unconditional = derivative.chunk(2, dim=0)
        x = x + dt * ((1.0 + cfg_rate) * conditional - cfg_rate * unconditional)
        t = t + dt
        if on_estimator_call is not None:
            on_estimator_call()
        if index + 1 < n_timesteps:
            dt = t_span[index + 2] - t
    return x[:, :, :output_length], torch.stack(present_cnn_banks), torch.stack(present_att_banks)
