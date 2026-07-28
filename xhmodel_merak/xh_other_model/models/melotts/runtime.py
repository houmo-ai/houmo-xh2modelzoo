from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np
import onnx
import onnxruntime as ort
import torch

from xhquant.api import HMONNXInference


class Runner(Protocol):
    def run(self, feed: dict[str, np.ndarray]) -> dict[str, np.ndarray]: ...


class OrtRunner:
    def __init__(self, model: str | Path):
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        self.session = ort.InferenceSession(str(model), options, providers=["CPUExecutionProvider"])
        self.output_names = [value.name for value in self.session.get_outputs()]

    def run(self, feed: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        values = self.session.run(None, feed)
        return dict(zip(self.output_names, values, strict=True))


class HmonnxRunner:
    def __init__(self, model: str | Path):
        proto = onnx.load(str(model), load_external_data=False)
        self.input_names = [value.name for value in proto.graph.input]
        self.output_names = [value.name for value in proto.graph.output]
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


def prepare_decoder_inputs(
    encoder_outputs: dict[str, np.ndarray],
    acoustic_max_length: int,
    *,
    length_scale: float = 1.0,
    noise_scale: float = 0.0,
    seed: int | None = None,
) -> tuple[dict[str, np.ndarray], int, np.ndarray]:
    means = np.asarray(encoder_outputs["m_p"], dtype=np.float32)
    log_scales = np.asarray(encoder_outputs["logs_p"], dtype=np.float32)
    logw = np.asarray(encoder_outputs["logw"], dtype=np.float32)
    x_mask = np.asarray(encoder_outputs["x_mask"], dtype=np.float32)
    durations = np.ceil(np.exp(logw) * x_mask * np.float32(length_scale)).astype(np.int64)[0, 0]
    raw_length = int(durations.sum())
    acoustic_length = max(raw_length, 1)
    if acoustic_length > acoustic_max_length:
        raise ValueError(f"predicted acoustic length {acoustic_length} exceeds Tmax={acoustic_max_length}")
    if raw_length:
        indices = np.repeat(np.arange(durations.size, dtype=np.int64), durations)
        expanded_mean = means[:, :, indices]
        expanded_logs = log_scales[:, :, indices]
    else:
        expanded_mean = np.zeros((1, means.shape[1], 1), dtype=np.float32)
        expanded_logs = np.zeros_like(expanded_mean)
    if noise_scale:
        rng = np.random.default_rng(seed)
        epsilon = rng.standard_normal(expanded_mean.shape).astype(np.float32)
    else:
        epsilon = np.zeros_like(expanded_mean)
    z_valid = expanded_mean + (epsilon * np.exp(expanded_logs) * np.float32(noise_scale))
    z_p = np.zeros((1, means.shape[1], acoustic_max_length), dtype=np.float32)
    y_mask = np.zeros((1, 1, acoustic_max_length), dtype=np.float32)
    z_p[:, :, :acoustic_length] = z_valid
    y_mask[:, :, :acoustic_length] = 1
    return (
        {
            "z_p": z_p,
            "y_mask": y_mask,
            "g": np.asarray(encoder_outputs["g"], dtype=np.float32),
        },
        acoustic_length,
        durations,
    )


class MeloTTSRuntime:
    """Batch-1 two-graph runtime with the exact CPU duration bridge."""

    def __init__(
        self,
        encoder: Runner,
        decoder: Runner,
        *,
        text_max_length: int = 32,
        acoustic_max_length: int = 64,
        hop_length: int = 512,
    ):
        self.encoder = encoder
        self.decoder = decoder
        self.text_max_length = int(text_max_length)
        self.acoustic_max_length = int(acoustic_max_length)
        self.hop_length = int(hop_length)

    def synthesize(
        self,
        token_ids: np.ndarray,
        tone_ids: np.ndarray,
        *,
        speaker_id: int = 1,
        length_scale: float = 1.0,
        noise_scale: float = 0.0,
        seed: int | None = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        tokens = np.asarray(token_ids, dtype=np.int32).reshape(-1)
        tones = np.asarray(tone_ids, dtype=np.int32).reshape(-1)
        if tokens.shape != tones.shape:
            raise ValueError("token_ids and tone_ids must have equal length")
        if not 0 < tokens.size <= self.text_max_length:
            raise ValueError(f"text length must be [1,{self.text_max_length}]")
        padded_tokens = np.zeros((1, self.text_max_length), dtype=np.int32)
        padded_tones = np.zeros_like(padded_tokens)
        padded_tokens[0, : tokens.size] = tokens
        padded_tones[0, : tones.size] = tones
        encoded = self.encoder.run(
            {
                "x": padded_tokens,
                "x_lengths": np.asarray([tokens.size], dtype=np.int32),
                "tones": padded_tones,
                "sid": np.asarray([speaker_id], dtype=np.int32),
            }
        )
        decoder_feed, acoustic_length, durations = prepare_decoder_inputs(
            encoded,
            self.acoustic_max_length,
            length_scale=length_scale,
            noise_scale=noise_scale,
            seed=seed,
        )
        waveform = self.decoder.run(decoder_feed)["y"]
        valid_samples = acoustic_length * self.hop_length
        return (
            np.asarray(waveform[0, 0, :valid_samples], dtype=np.float32),
            {
                "text_length": int(tokens.size),
                "acoustic_length": acoustic_length,
                "valid_samples": valid_samples,
                "durations": durations.tolist(),
            },
        )


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
    "MeloTTSRuntime",
    "OrtRunner",
    "error_stats",
    "prepare_decoder_inputs",
]
