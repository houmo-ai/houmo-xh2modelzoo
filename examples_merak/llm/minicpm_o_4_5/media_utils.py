from __future__ import annotations

import math
import wave
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def video_chunks(path: Path, sample_rate: int) -> list[tuple[np.ndarray, list[Any]]]:
    """Decode a video into per-second (audio_chunk, [frame]) pairs.

    Frames are sampled at the mid-second point (index + 0.5) for the
    streaming demos; videos without an audio track yield silence chunks.
    """
    from moviepy import VideoFileClip

    chunks: list[tuple[np.ndarray, list[Any]]] = []
    with VideoFileClip(str(path)) as video:
        duration = max(float(video.duration or 0.0), 0.001)
        chunk_count = max(1, math.ceil(duration))
        if video.audio is None:
            audio = np.zeros(chunk_count * sample_rate, dtype=np.float32)
        else:
            audio = np.asarray(video.audio.to_soundarray(fps=sample_rate), dtype=np.float32)
            if audio.ndim == 2:
                audio = audio.mean(axis=1)
        for index in range(chunk_count):
            start = index * sample_rate
            chunk = audio[start : start + sample_rate]
            if len(chunk) < sample_rate:
                chunk = np.pad(chunk, (0, sample_rate - len(chunk)))
            frame_time = min(index + 0.5, max(duration - 1e-6, 0.0))
            frame = Image.fromarray(video.get_frame(frame_time).astype(np.uint8))
            chunks.append((chunk, [frame]))
    return chunks


def flatten_audio_chunks(chunks: Sequence[np.ndarray]) -> np.ndarray:
    return np.concatenate([np.asarray(chunk).reshape(-1) for chunk in chunks])


def write_wav(path: Path, waveform: np.ndarray, sample_rate: int) -> None:
    """Write float32 mono audio as a 16-bit PCM WAV (clip to [-1, 1])."""
    mono = np.asarray(waveform, dtype=np.float32).reshape(-1)
    pcm = (np.clip(mono, -1.0, 1.0) * 32767.0).astype(np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())


__all__ = ["flatten_audio_chunks", "video_chunks", "write_wav"]
