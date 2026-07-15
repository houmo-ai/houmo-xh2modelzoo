from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AudioChunk:
    waveform: np.ndarray
    valid_samples: int
    chunk_index: int


def validate_audio_inputs(waveform: np.ndarray, sampling_rate: int, expected_sampling_rate: int = 16000) -> None:
    if int(sampling_rate) != int(expected_sampling_rate):
        raise ValueError(f"emotion2vec requires 16 kHz audio, got {sampling_rate}")
    if waveform.ndim != 1:
        raise ValueError(f"emotion2vec expects mono waveform, got shape {waveform.shape}")
    if waveform.size == 0:
        raise ValueError("emotion2vec requires a non-empty waveform")


def chunk_waveform(
    waveform: np.ndarray,
    sampling_rate: int,
    window_samples: int = 256000,
    max_valid_samples: int | None = None,
) -> list[AudioChunk]:
    validate_audio_inputs(waveform, sampling_rate)
    window_samples = int(window_samples)
    chunk_samples = window_samples if max_valid_samples is None else min(window_samples, int(max_valid_samples))
    if chunk_samples <= 0:
        raise ValueError("max_valid_samples must be positive")
    chunks: list[AudioChunk] = []
    for chunk_index, start in enumerate(range(0, waveform.size, chunk_samples)):
        stop = min(start + chunk_samples, waveform.size)
        valid_samples = stop - start
        padded = np.zeros(window_samples, dtype=np.float32)
        padded[:valid_samples] = waveform[start:stop].astype(np.float32, copy=False)
        chunks.append(AudioChunk(waveform=padded, valid_samples=valid_samples, chunk_index=chunk_index))
    return chunks


def normalize_padded_waveform(waveform: np.ndarray, valid_samples: int, eps: float = 1e-5) -> np.ndarray:
    waveform = np.asarray(waveform, dtype=np.float32)
    valid_samples = int(valid_samples)
    if waveform.ndim != 1:
        raise ValueError(f"expected one padded waveform, got shape {waveform.shape}")
    if valid_samples <= 0 or valid_samples > waveform.size:
        raise ValueError(f"valid_samples must be in [1, {waveform.size}], got {valid_samples}")
    normalized = np.zeros_like(waveform, dtype=np.float32)
    valid = waveform[:valid_samples]
    mean = np.mean(valid, dtype=np.float32)
    variance = np.mean(np.square(valid - mean, dtype=np.float32), dtype=np.float32)
    normalized[:valid_samples] = (valid - mean) / np.sqrt(variance + np.float32(eps))
    return normalized


def conv_output_length(length: int, kernel_size: int, stride: int, padding: int = 0, dilation: int = 1) -> int:
    numerator = length + 2 * padding - dilation * (kernel_size - 1) - 1
    return int(np.floor(numerator / stride + 1))


def emotion2vec_frame_count(valid_samples: int) -> int:
    length = int(valid_samples)
    for kernel_size, stride in ((10, 5), (3, 2), (3, 2), (3, 2), (3, 2), (2, 2), (2, 2)):
        length = conv_output_length(length, kernel_size=kernel_size, stride=stride)
    return max(length, 0)


def trim_frame_padding_mask(frame_padding_mask: np.ndarray) -> np.ndarray:
    return np.asarray(frame_padding_mask, dtype=bool)


def masked_mean(features: np.ndarray, frame_padding_mask: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    mask = np.asarray(frame_padding_mask, dtype=bool)
    if features.ndim != 3:
        raise ValueError(f"expected features to have shape [batch, frames, hidden], got {features.shape}")
    if mask.ndim != 2:
        raise ValueError(f"expected frame mask to have shape [batch, frames], got {mask.shape}")
    if features.shape[:2] != mask.shape:
        raise ValueError("features and mask must agree on batch/frame dimensions")
    weights = (~mask).astype(np.float32)[..., None]
    total = np.sum(features * weights, axis=1)
    denom = np.clip(np.sum(weights, axis=1), 1.0, None)
    return total / denom


def sample_mask(valid_samples: int, window_samples: int) -> np.ndarray:
    mask = np.zeros(int(window_samples), dtype=bool)
    mask[: int(valid_samples)] = True
    return mask


def trim_valid_frames(features: np.ndarray, frame_padding_mask: np.ndarray) -> np.ndarray:
    features = np.asarray(features)
    frame_padding_mask = np.asarray(frame_padding_mask, dtype=bool)
    if features.ndim != 3 or frame_padding_mask.ndim != 2:
        raise ValueError("expected batched frame features and padding mask")
    if features.shape[:2] != frame_padding_mask.shape or features.shape[0] != 1:
        raise ValueError("features and padding mask must describe one matching batch")
    return features[0, ~frame_padding_mask[0]]
