from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import soundfile as sf
except ImportError:
    sf = None


def save_wav(audio: np.ndarray, sample_rate: int, output: str) -> None:
    output_path = Path(output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    waveform = np.clip(
        np.asarray(audio, dtype=np.float32).reshape(-1),
        -1.0,
        1.0,
    )
    if sf is not None:
        sf.write(output_path, waveform, sample_rate)
        return

    from scipy.io import wavfile

    wavfile.write(output_path, sample_rate, (waveform * 32767).astype(np.int16))


def calc_audio_metrics(hmonnx_audio: np.ndarray, torch_audio: np.ndarray) -> dict[str, float | int]:
    a = np.asarray(hmonnx_audio, dtype=np.float32).reshape(-1)
    b = np.asarray(torch_audio, dtype=np.float32).reshape(-1)
    min_len = min(a.size, b.size)
    if min_len == 0:
        return {
            "hmonnx_len": int(a.size),
            "torch_len": int(b.size),
            "len_ratio": float("inf") if b.size == 0 else float(a.size / max(1, b.size)),
            "max_abs": float("inf"),
            "mean_abs": float("inf"),
            "cosine": 0.0,
        }

    ax = a[:min_len]
    bx = b[:min_len]
    abs_diff = np.abs(ax - bx)
    cosine = float(np.dot(ax, bx) / ((np.linalg.norm(ax) * np.linalg.norm(bx)) + 1e-12))
    return {
        "hmonnx_len": int(a.size),
        "torch_len": int(b.size),
        "len_ratio": float(a.size / max(1, b.size)),
        "max_abs": float(abs_diff.max()),
        "mean_abs": float(abs_diff.mean()),
        "cosine": cosine,
    }
