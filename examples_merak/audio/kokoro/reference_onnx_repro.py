from __future__ import annotations

import argparse
import hashlib
import json
import time
import wave
from pathlib import Path

import numpy as np
import onnxruntime as ort

from xhmodel_merak.xh_other_model.models.kokoro.assets import resolve_model_assets
from xhmodel_merak.xh_other_model.models.kokoro.host import (
    DEFAULT_INPUT_IDS,
    DEFAULT_PHONEMES,
    DEFAULT_TEXT,
    SAMPLE_RATE,
    load_voice_style,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce the released dynamic Kokoro ONNX")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-wav", default="work_dirs/kokoro_reference.wav")
    parser.add_argument("--repeat", type=int, default=4)
    args = parser.parse_args()
    if args.repeat < 1:
        raise ValueError("--repeat must be positive")

    assets = resolve_model_assets(args.model_dir)
    ids = np.asarray(DEFAULT_INPUT_IDS, dtype=np.int64)[None]
    style = load_voice_style(
        assets.voice,
        phoneme_count=len(DEFAULT_INPUT_IDS) - 2,
    ).numpy()
    feed = {
        "input_ids": ids,
        "style": style,
        "speed": np.asarray([1], dtype=np.int32),
    }
    started = time.perf_counter()
    session = ort.InferenceSession(
        str(assets.reference_onnx),
        providers=["CPUExecutionProvider"],
    )
    initialization_seconds = time.perf_counter() - started
    runs = []
    waveforms = []
    duration = None
    for _ in range(args.repeat):
        started = time.perf_counter()
        waveform, duration = session.run(None, feed)
        runs.append(time.perf_counter() - started)
        waveforms.append(np.asarray(waveform, dtype=np.float32).reshape(-1))
    assert duration is not None
    output = Path(args.output_wav).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_wav(output, waveforms[0])
    seconds = waveforms[0].size / SAMPLE_RATE
    print(
        json.dumps(
            {
                "text": DEFAULT_TEXT,
                "phonemes": DEFAULT_PHONEMES,
                "input_ids": list(DEFAULT_INPUT_IDS),
                "duration": np.asarray(duration).reshape(-1).tolist(),
                "duration_sum": int(np.asarray(duration).sum()),
                "samples": int(waveforms[0].size),
                "audio_seconds": seconds,
                "session_initialization_seconds": initialization_seconds,
                "inference_seconds": runs,
                "warm_rtf": runs[-1] / seconds,
                "waveform_sha256": hashlib.sha256(waveforms[0].tobytes()).hexdigest(),
                "repeatability": [_error_stats(waveforms[0], value) for value in waveforms[1:]],
                "output_wav": str(output),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def _error_stats(reference: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    delta = reference.astype(np.float64) - actual.astype(np.float64)
    denominator = max(
        float(np.linalg.norm(reference) * np.linalg.norm(actual)),
        np.finfo(np.float64).eps,
    )
    return {
        "max_abs": float(np.max(np.abs(delta))),
        "mean_abs": float(np.mean(np.abs(delta))),
        "cosine": float(np.dot(reference, actual) / denominator),
    }


def _write_wav(path: Path, waveform: np.ndarray) -> None:
    pcm = np.clip(waveform, -1.0, 1.0)
    pcm = np.round(pcm * 32767).astype("<i2")
    with wave.open(str(path), "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(SAMPLE_RATE)
        target.writeframes(pcm.tobytes())


if __name__ == "__main__":
    main()
