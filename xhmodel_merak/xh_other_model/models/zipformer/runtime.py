from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np
import onnx
import onnxruntime as ort
import torch

from xhquant.api import HMONNXInference


def concrete_shape(shape: list[int | str | None]) -> tuple[int, ...]:
    return tuple(int(dimension) if isinstance(dimension, int) else 1 for dimension in shape)


def numpy_dtype(type_name: str) -> np.dtype:
    if type_name == "tensor(int64)":
        return np.dtype(np.int64)
    if type_name == "tensor(int32)":
        return np.dtype(np.int32)
    return np.dtype(np.float32)


class Runner(Protocol):
    input_specs: dict[str, tuple[tuple[int, ...], np.dtype]]

    def run(self, feed: dict[str, np.ndarray]) -> dict[str, np.ndarray]: ...


class OrtRunner:
    def __init__(self, model: str | Path):
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        self.session = ort.InferenceSession(str(model), options, providers=["CPUExecutionProvider"])
        self.output_names = [value.name for value in self.session.get_outputs()]
        self.input_specs = {
            value.name: (
                concrete_shape(value.shape),
                numpy_dtype(value.type),
            )
            for value in self.session.get_inputs()
        }

    def run(self, feed: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        values = self.session.run(None, feed)
        return dict(zip(self.output_names, values, strict=True))


class HmonnxRunner:
    def __init__(self, contract_model: str | Path, model: str | Path):
        contract = ort.InferenceSession(str(contract_model), providers=["CPUExecutionProvider"])
        self.input_specs = {
            value.name: (
                concrete_shape(value.shape),
                numpy_dtype(value.type),
            )
            for value in contract.get_inputs()
        }
        model_proto = onnx.load(str(model), load_external_data=False)
        self.input_names = [value.name for value in model_proto.graph.input]
        self.output_names = [value.name for value in model_proto.graph.output]
        self.session = HMONNXInference(str(model))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.session.to(self.device)

    @torch.no_grad()
    def run(self, feed: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        tensors = []
        for name in self.input_names:
            value = np.asarray(feed[name])
            if np.issubdtype(value.dtype, np.integer):
                tensor = torch.from_numpy(value.astype(np.int32))
            else:
                tensor = torch.from_numpy(value.astype(np.float16))
            tensors.append(tensor.to(self.device))
        values = self.session(*tensors)
        if isinstance(values, torch.Tensor):
            values = [values]
        return {name: value.detach().cpu().numpy() for name, value in zip(self.output_names, values, strict=True)}


class StreamingEncoder:
    """Owns and feeds back every cache for one Zipformer stream."""

    def __init__(self, runner: Runner):
        self.runner = runner
        self.state: dict[str, np.ndarray] = {}
        self.reset()

    def reset(self) -> None:
        self.state = {
            name: np.zeros(shape, dtype=dtype)
            for name, (shape, dtype) in self.runner.input_specs.items()
            if name != "x"
        }

    def run(self, features: np.ndarray) -> np.ndarray:
        value = np.asarray(features, dtype=np.float32)
        if value.ndim == 2:
            value = value.reshape(1, 39, 80)
        if value.shape != (1, 39, 80):
            raise ValueError(f"expected Zipformer fbank [1,39,80], got {value.shape}")
        output = self.runner.run({"x": value, **self.state})
        for name in self.state:
            self.state[name] = output[f"new_{name}"]
        encoder_out = np.asarray(output["encoder_out"], dtype=np.float32)
        if encoder_out.shape != (1, 8, 320):
            raise RuntimeError(f"expected encoder_out [1,8,320], got {encoder_out.shape}")
        return encoder_out


def make_features(waveform: np.ndarray, sample_rate: int) -> np.ndarray:
    import kaldi_native_fbank as knf

    if sample_rate != 16000:
        raise ValueError("Zipformer expects 16 kHz mono audio")
    options = knf.FbankOptions()
    options.frame_opts.dither = 0
    options.frame_opts.snip_edges = False
    options.frame_opts.samp_freq = 16000
    options.mel_opts.num_bins = 80
    options.mel_opts.high_freq = -400
    fbank = knf.OnlineFbank(options)
    padded = np.concatenate(
        [
            np.asarray(waveform, dtype=np.float32).reshape(-1),
            np.zeros(4800, dtype=np.float32),
        ]
    )
    fbank.accept_waveform(16000, padded)
    fbank.input_finished()
    return np.stack([fbank.get_frame(index) for index in range(fbank.num_frames_ready)]).astype(np.float32)


def feature_windows(features: np.ndarray) -> list[np.ndarray]:
    value = np.asarray(features, dtype=np.float32)
    if value.ndim != 2 or value.shape[1] != 80:
        raise ValueError("fbank features must have shape [T,80]")
    return [
        value[offset : offset + 39].reshape(1, 39, 80)
        for offset in range(0, max(value.shape[0] - 38, 0), 32)
        if value[offset : offset + 39].shape[0] == 39
    ]


def error_stats(reference: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    expected = np.asarray(reference, dtype=np.float64).reshape(-1)
    observed = np.asarray(actual, dtype=np.float64).reshape(-1)
    delta = expected - observed
    eps = np.finfo(np.float64).eps
    signal = float(np.mean(expected * expected))
    noise = float(np.mean(delta * delta))
    return {
        "max_abs": float(np.max(np.abs(delta))),
        "mean_abs": float(np.mean(np.abs(delta))),
        "mse": noise,
        "cosine": float(
            np.dot(expected, observed)
            / max(
                float(np.linalg.norm(expected) * np.linalg.norm(observed)),
                eps,
            )
        ),
        "snr_db": float(10 * np.log10(max(signal, eps) / max(noise, eps))),
    }


__all__ = [
    "HmonnxRunner",
    "OrtRunner",
    "StreamingEncoder",
    "concrete_shape",
    "error_stats",
    "feature_windows",
    "make_features",
    "numpy_dtype",
]
