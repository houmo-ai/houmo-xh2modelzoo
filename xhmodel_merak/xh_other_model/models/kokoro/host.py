from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F


SAMPLE_RATE = 24_000
HARMONICS = 9
F0_UPSAMPLE = 300
WAVEFORM_SAMPLES_PER_FRAME = 600
ATTENTION_MASK_MIN = -65_504.0

# Official v1.1-zh ZHG2P(version="1.1") output for
# "千里之行，始于足下。", including Kokoro's leading/trailing boundary token.
DEFAULT_TEXT = "千里之行，始于足下。"
DEFAULT_PHONEMES = "ㄑ言1ㄌㄧ3ㄓ十1ㄒ应2, ㄕ十3ㄩ2ㄗㄨ2ㄒ压4."
DEFAULT_INPUT_IDS = (
    0,
    91,
    146,
    171,
    74,
    127,
    169,
    23,
    144,
    171,
    93,
    152,
    172,
    3,
    16,
    95,
    144,
    169,
    137,
    172,
    96,
    134,
    172,
    93,
    145,
    173,
    4,
    0,
)


@dataclass(frozen=True)
class StaticSample:
    input_ids: Tensor
    valid_len: Tensor
    style: Tensor
    speed: Tensor
    text: str
    phonemes: str
    voice: str


def default_static_sample(
    voice_path: str | Path,
    text_max_length: int,
) -> StaticSample:
    if text_max_length < len(DEFAULT_INPUT_IDS):
        raise ValueError(
            f"text_max_length={text_max_length} is smaller than the default golden length {len(DEFAULT_INPUT_IDS)}"
        )
    input_ids = torch.zeros(1, text_max_length, dtype=torch.int32)
    input_ids[0, : len(DEFAULT_INPUT_IDS)] = torch.tensor(DEFAULT_INPUT_IDS, dtype=torch.int32)
    return StaticSample(
        input_ids=input_ids,
        valid_len=torch.tensor([len(DEFAULT_INPUT_IDS)], dtype=torch.int32),
        style=load_voice_style(voice_path, phoneme_count=len(DEFAULT_INPUT_IDS) - 2),
        speed=torch.tensor([1.0], dtype=torch.float32),
        text=DEFAULT_TEXT,
        phonemes=DEFAULT_PHONEMES,
        voice="zf_001",
    )


def load_voice_pack(path: str | Path) -> Tensor:
    """Load an upstream ``.pt`` voice pack or an exported NumPy pack."""

    source = Path(path)
    if source.suffix.lower() == ".npy":
        voice = torch.from_numpy(np.load(source, allow_pickle=False))
    else:
        voice = torch.load(source, map_location="cpu", weights_only=True)
    if not isinstance(voice, Tensor):
        raise TypeError(f"voice file must contain a Tensor: {path}")
    if voice.ndim not in {1, 2, 3}:
        raise ValueError(f"unsupported voice tensor shape: {tuple(voice.shape)}")
    return voice.detach().float().cpu().contiguous()


def save_voice_pack_numpy(source_path: str | Path, destination: str | Path) -> Path:
    """Materialize a safe, framework-independent runtime voice asset."""

    output = Path(destination)
    if output.suffix.lower() != ".npy":
        raise ValueError(f"Kokoro runtime voice pack must use .npy: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, load_voice_pack(source_path).numpy(), allow_pickle=False)
    return output


def load_voice_style(path: str | Path, phoneme_count: int) -> Tensor:
    if phoneme_count <= 0:
        raise ValueError("phoneme_count must be positive")
    voice = load_voice_pack(path)
    if voice.ndim == 3:
        index = min(phoneme_count, int(voice.shape[0])) - 1
        style = voice[index]
    elif voice.ndim == 2:
        style = voice
    elif voice.ndim == 1:
        style = voice.unsqueeze(0)
    if tuple(style.shape) != (1, 256):
        raise ValueError(f"expected voice style [1,256], got {tuple(style.shape)}")
    return style.float().contiguous()


def make_reverse_idx(text_max_length: int, valid_len: Tensor) -> Tensor:
    length = _scalar_length(valid_len, text_max_length, "valid_len")
    index = torch.arange(text_max_length, dtype=torch.int32)
    index[:length] = torch.arange(length - 1, -1, -1, dtype=torch.int32)
    return index


def make_text_mask(text_max_length: int, valid_len: Tensor) -> Tensor:
    length = _scalar_length(valid_len, text_max_length, "valid_len")
    mask = torch.zeros(1, text_max_length, 1, dtype=torch.float32)
    mask[:, :length] = 1.0
    return mask


def make_attention_mask(text_max_length: int, valid_len: Tensor) -> Tensor:
    """Return an additive ``[1,1,T,T]`` ALBERT mask using finite FP16 values."""

    length = _scalar_length(valid_len, text_max_length, "valid_len")
    key_mask = torch.full(
        (1, 1, 1, text_max_length),
        ATTENTION_MASK_MIN,
        dtype=torch.float32,
    )
    key_mask[..., :length] = 0.0
    return key_mask.expand(1, 1, text_max_length, text_max_length).contiguous()


def prepare_lstm_inputs(values: Tensor, valid_len: Tensor) -> tuple[Tensor, Tensor]:
    if values.ndim != 3 or values.shape[0] != 1:
        raise ValueError("LSTM values must be [1,T,C]")
    length = _scalar_length(valid_len, int(values.shape[1]), "valid_len")
    forward = torch.zeros_like(values)
    backward = torch.zeros_like(values)
    forward[:, :length] = values[:, :length]
    backward[:, :length] = torch.flip(values[:, :length], dims=[1])
    return forward, backward


def restore_bidirectional_outputs(
    forward: Tensor,
    backward_reversed: Tensor,
    valid_len: Tensor,
) -> Tensor:
    if forward.shape != backward_reversed.shape or forward.ndim != 3:
        raise ValueError("forward and backward LSTM outputs must have the same [1,T,C] shape")
    length = _scalar_length(valid_len, int(forward.shape[1]), "valid_len")
    restored_forward = torch.zeros_like(forward)
    restored_backward = torch.zeros_like(backward_reversed)
    restored_forward[:, :length] = forward[:, :length]
    restored_backward[:, :length] = torch.flip(backward_reversed[:, :length], dims=[1])
    return torch.cat([restored_forward, restored_backward], dim=-1)


def duration_from_logits(
    logits: Tensor,
    speed: Tensor,
    valid_len: Tensor,
) -> Tensor:
    if logits.ndim != 3 or logits.shape[0] != 1:
        raise ValueError("duration logits must be [1,T,max_duration]")
    speed_value = float(speed.reshape(-1)[0].item())
    if speed_value <= 0:
        raise ValueError("speed must be positive")
    length = _scalar_length(valid_len, int(logits.shape[1]), "valid_len")
    duration = torch.round(torch.sigmoid(logits.float()).sum(dim=-1) / speed_value)
    duration = duration.clamp(min=1).to(torch.int64)
    duration[:, length:] = 0
    return duration


def _valid_duration_values(
    duration: Tensor,
    frame_max_length: int,
    valid_len: Tensor,
) -> tuple[Tensor, int]:
    if duration.ndim != 2 or duration.shape[0] != 1:
        raise ValueError("duration must be [1,T]")
    length = _scalar_length(valid_len, int(duration.shape[1]), "valid_len")
    values = duration[0, :length].to(torch.int64)
    if torch.any(values < 1):
        raise ValueError("every valid token must have duration >= 1")
    frame_count = int(values.sum().item())
    if frame_count > frame_max_length:
        raise ValueError(
            f"predicted frame length {frame_count} exceeds Fmax={frame_max_length}; "
            "split the phoneme sequence or select a larger frame bucket"
        )
    return values, frame_count


def duration_to_frame_indices(
    duration: Tensor,
    frame_max_length: int,
    valid_len: Tensor,
) -> tuple[Tensor, Tensor]:
    """Return the valid frame-to-token Gather index without a ``[T,F]`` matrix."""

    values, frame_count = _valid_duration_values(duration, frame_max_length, valid_len)
    token_indices = torch.arange(values.numel(), dtype=torch.int64, device=values.device)
    frame_indices = torch.repeat_interleave(token_indices, values)
    valid_frames = torch.tensor([frame_count], dtype=torch.int32, device=duration.device)
    return frame_indices.contiguous(), valid_frames


def duration_to_alignment(
    duration: Tensor,
    frame_max_length: int,
    valid_len: Tensor,
) -> tuple[Tensor, Tensor]:
    values, frame_count = _valid_duration_values(duration, frame_max_length, valid_len)
    alignment = torch.zeros(1, duration.shape[1], frame_max_length, dtype=torch.float32)
    cursor = 0
    for token_index, value in enumerate(values.tolist()):
        alignment[0, token_index, cursor : cursor + value] = 1.0
        cursor += value
    return alignment, torch.tensor([frame_count], dtype=torch.int32)


def make_frame_masks(valid_frames: Tensor, frame_max_length: int) -> tuple[Tensor, ...]:
    valid = _scalar_length(valid_frames, frame_max_length, "valid_frames")

    def prefix(total: int, effective: int) -> Tensor:
        mask = torch.zeros(1, 1, total, dtype=torch.float32)
        mask[:, :, :effective] = 1.0
        return mask

    return (
        prefix(frame_max_length, valid),
        prefix(2 * frame_max_length, 2 * valid),
        prefix(20 * frame_max_length, 20 * valid),
        prefix(120 * frame_max_length + 1, 120 * valid + 1),
    )


def make_rmsnorm_scales(valid_frames: Tensor, frame_max_length: int) -> Tensor:
    """Return Host-computed masked RMSNorm scales as ``[1, 2, 1]``.

    The first value is ``sqrt(T/L)`` for the RMSNorm input and the second is
    ``sqrt(L/T)`` for its output.  Supplying both values avoids computing a
    reciprocal from an already-rounded accelerator scalar.
    """

    valid = _scalar_length(valid_frames, frame_max_length, "valid_frames")
    scale = math.sqrt(float(frame_max_length) / float(valid))
    inverse_scale = math.sqrt(float(valid) / float(frame_max_length))
    return torch.tensor(
        [[[scale], [inverse_scale]]],
        dtype=torch.float32,
    )


def prepare_shared_lstm_inputs(encoded: Tensor, valid_frames: Tensor) -> tuple[Tensor, Tensor]:
    if encoded.ndim != 3 or encoded.shape[0] != 1:
        raise ValueError("encoded features must be [1,C,F]")
    values = encoded.transpose(1, 2).contiguous()
    return prepare_lstm_inputs(values, valid_frames)


def build_sine_wavs(
    f0_prediction: Tensor,
    valid_frames: Tensor,
    frame_max_length: int,
    *,
    seed: int,
) -> Tensor:
    valid = _scalar_length(valid_frames, frame_max_length, "valid_frames")
    f0 = f0_prediction.float()[:, : 2 * valid]
    f0_up = F.interpolate(f0[:, None], scale_factor=F0_UPSAMPLE).transpose(1, 2)
    harmonics = torch.arange(1, HARMONICS + 1, dtype=f0_up.dtype).reshape(1, 1, -1)
    radians = torch.remainder(f0_up * harmonics / SAMPLE_RATE, 1.0)
    # fork_rng keeps the caller's global stream untouched while reproducing
    # torch.rand + torch.randn_like exactly.  The latter's preserve-format
    # iteration order matters because `sine` is non-contiguous.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        initial = torch.rand((1, HARMONICS), dtype=radians.dtype)
        initial[:, 0] = 0
        radians[:, 0, :] += initial
        compact = F.interpolate(
            radians.transpose(1, 2),
            scale_factor=1 / F0_UPSAMPLE,
            mode="linear",
        ).transpose(1, 2)
        phase = torch.cumsum(compact, dim=1) * (2 * torch.pi)
        phase = F.interpolate(
            phase.transpose(1, 2) * F0_UPSAMPLE,
            scale_factor=F0_UPSAMPLE,
            mode="linear",
        ).transpose(1, 2)
        sine = torch.sin(phase) * 0.1
        uv = (f0_up > 10).to(torch.float32)
        noise_amplitude = uv * 0.003 + (1 - uv) * (0.1 / 3)
        noise = torch.randn_like(sine)
        sine = sine * uv + noise_amplitude * noise
    padded = torch.zeros(1, WAVEFORM_SAMPLES_PER_FRAME * frame_max_length, HARMONICS)
    padded[:, : sine.shape[1]] = sine
    return padded.contiguous()


def harmonic_spectrogram(
    harmonic_source: Tensor,
    valid_frames: Tensor,
    frame_max_length: int,
) -> Tensor:
    valid = _scalar_length(valid_frames, frame_max_length, "valid_frames")
    valid_samples = WAVEFORM_SAMPLES_PER_FRAME * valid
    source = harmonic_source[:, :valid_samples].transpose(1, 2).squeeze(1).float()
    window = torch.hann_window(20, periodic=True, dtype=source.dtype)
    complex_spec = torch.stft(
        source,
        n_fft=20,
        hop_length=5,
        win_length=20,
        window=window,
        center=True,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    valid_result = torch.cat([complex_spec.abs(), torch.angle(complex_spec)], dim=1)
    result = torch.zeros(1, 22, 120 * frame_max_length + 1, dtype=torch.float32)
    result[:, :, : valid_result.shape[-1]] = valid_result
    return result.contiguous()


def istft_waveform(spec_phase: Tensor, valid_frames: Tensor) -> Tensor:
    valid = int(valid_frames.reshape(-1)[0].item())
    magnitude = spec_phase[:, :11].float()
    phase = spec_phase[:, 11:].float()
    complex_spec = torch.polar(magnitude, phase)
    window = torch.hann_window(20, periodic=True, dtype=magnitude.dtype)
    waveform = torch.istft(
        complex_spec,
        n_fft=20,
        hop_length=5,
        win_length=20,
        window=window,
        center=True,
        length=WAVEFORM_SAMPLES_PER_FRAME * int(spec_phase.shape[-1] - 1) // 120,
    )
    return waveform[:, : WAVEFORM_SAMPLES_PER_FRAME * valid].contiguous()


def as_numpy(values: dict[str, Tensor]) -> dict[str, np.ndarray]:
    return {name: value.detach().cpu().numpy() for name, value in values.items()}


def _scalar_length(value: Tensor, maximum: int, name: str) -> int:
    length = int(value.reshape(-1)[0].item())
    if not 1 <= length <= maximum:
        raise ValueError(f"{name} must be in [1,{maximum}], got {length}")
    return length


__all__ = [
    "ATTENTION_MASK_MIN",
    "DEFAULT_INPUT_IDS",
    "DEFAULT_PHONEMES",
    "DEFAULT_TEXT",
    "SAMPLE_RATE",
    "StaticSample",
    "WAVEFORM_SAMPLES_PER_FRAME",
    "as_numpy",
    "build_sine_wavs",
    "default_static_sample",
    "duration_from_logits",
    "duration_to_frame_indices",
    "duration_to_alignment",
    "harmonic_spectrogram",
    "istft_waveform",
    "load_voice_pack",
    "load_voice_style",
    "make_frame_masks",
    "make_attention_mask",
    "make_rmsnorm_scales",
    "make_reverse_idx",
    "make_text_mask",
    "prepare_lstm_inputs",
    "prepare_shared_lstm_inputs",
    "restore_bidirectional_outputs",
    "save_voice_pack_numpy",
]
