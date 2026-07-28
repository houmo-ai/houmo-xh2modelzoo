from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

import numpy as np
import onnx
import onnxruntime as ort
import torch
from scipy.signal import resample_poly

from xhquant.api import HMONNXInference

from .graph import RATE_CONFIG


class Runner(Protocol):
    def run(self, feed: dict[str, np.ndarray]) -> dict[str, np.ndarray]: ...


class OrtRunner:
    def __init__(self, model: str | Path, *, source_graph: bool = False):
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        self.session = ort.InferenceSession(str(model), options, providers=["CPUExecutionProvider"])
        self.output_names = [value.name for value in self.session.get_outputs()]
        self.source_graph = bool(source_graph)

    def run(self, feed: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        if not self.source_graph:
            feed = {key: value for key, value in feed.items() if key != "sr"}
        values = self.session.run(None, feed)
        return dict(zip(self.output_names, values, strict=True))


class HmonnxRunner:
    def __init__(self, model: str | Path):
        model_proto = onnx.load(str(model), load_external_data=False)
        self.input_names = [value.name for value in model_proto.graph.input]
        self.output_names = [value.name for value in model_proto.graph.output]
        self.session = HMONNXInference(str(model))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.session.to(self.device)

    @torch.no_grad()
    def run(self, feed: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        tensors = [
            torch.from_numpy(np.asarray(feed[name], dtype=np.float16)).to(self.device) for name in self.input_names
        ]
        values = self.session(*tensors)
        if isinstance(values, torch.Tensor):
            values = [values]
        return {name: value.detach().cpu().numpy() for name, value in zip(self.output_names, values, strict=True)}


def load_audio(path: str | Path, sample_rate: int, seconds: float = 0) -> np.ndarray:
    import soundfile as sf

    waveform, source_rate = sf.read(str(path), dtype="float32", always_2d=True)
    waveform = waveform.mean(axis=1)
    if source_rate != sample_rate:
        waveform = resample_poly(waveform, sample_rate, source_rate).astype(np.float32)
    if seconds > 0:
        waveform = waveform[: int(round(seconds * sample_rate))]
    return np.ascontiguousarray(waveform, dtype=np.float32)


def run_audio(runner: Runner, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
    config = RATE_CONFIG[sample_rate]
    state = np.zeros((2, 1, 128), dtype=np.float32)
    context = np.zeros((1, config["context"]), dtype=np.float32)
    probabilities = []
    for offset in range(0, waveform.size, config["frame"]):
        chunk = waveform[offset : offset + config["frame"]]
        if chunk.size < config["frame"]:
            chunk = np.pad(chunk, (0, config["frame"] - chunk.size))
        model_input = np.concatenate([context, chunk.reshape(1, -1)], axis=1).astype(np.float32)
        output = runner.run(
            {
                "input": model_input,
                "state": state,
                "sr": np.asarray(sample_rate, dtype=np.int64),
            }
        )
        probabilities.append(float(output["output"][0, 0]))
        state = output["stateN"].astype(np.float32)
        context = model_input[:, -config["context"] :]
    return np.asarray(probabilities, dtype=np.float32)


def error_stats(reference: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    expected = np.asarray(reference, dtype=np.float64).reshape(-1)
    observed = np.asarray(actual, dtype=np.float64).reshape(-1)
    delta = expected - observed
    eps = np.finfo(np.float64).eps
    return {
        "max_abs": float(np.max(np.abs(delta))),
        "mean_abs": float(np.mean(np.abs(delta))),
        "mse": float(np.mean(delta * delta)),
        "cosine": float(
            np.dot(expected, observed)
            / max(
                float(np.linalg.norm(expected) * np.linalg.norm(observed)),
                eps,
            )
        ),
    }


def decision_stats(reference: np.ndarray, actual: np.ndarray, threshold: float = 0.5) -> dict[str, float | int]:
    expected = reference >= threshold
    observed = actual >= threshold
    return {
        "threshold": threshold,
        "agreement": float(np.mean(expected == observed)),
        "different_frames": int(np.count_nonzero(expected != observed)),
        "false_speech_frames": int(np.count_nonzero(~expected & observed)),
        "missed_speech_frames": int(np.count_nonzero(expected & ~observed)),
    }


def load_export_meta(
    export_dir: str | Path,
) -> tuple[Path, dict[str, object]]:
    root = Path(export_dir).expanduser().resolve()
    meta = json.loads((root / "export_meta_info.json").read_text(encoding="utf-8"))
    return root, meta


__all__ = [
    "HmonnxRunner",
    "OrtRunner",
    "decision_stats",
    "error_stats",
    "load_audio",
    "load_export_meta",
    "run_audio",
]
