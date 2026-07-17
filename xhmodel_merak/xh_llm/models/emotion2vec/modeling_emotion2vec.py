from __future__ import annotations

from typing import Any

import torch
from torch import nn


def extract_x_from_result(result: Any) -> torch.Tensor:
    if isinstance(result, dict):
        for key in ("x", "frame_features", "hidden_states", "features"):
            value = result.get(key)
            if torch.is_tensor(value):
                return value
    if isinstance(result, (tuple, list)):
        for value in result:
            if torch.is_tensor(value):
                return value
            if isinstance(value, dict):
                nested = extract_x_from_result(value)
                if torch.is_tensor(nested):
                    return nested
    if torch.is_tensor(result):
        return result
    raise TypeError(f"Unsupported emotion2vec result type: {type(result)!r}")


def extract_mask_from_result(result: Any, *, frame_count: int) -> torch.Tensor:
    if isinstance(result, dict):
        for key in ("frame_padding_mask", "padding_mask", "output_mask"):
            value = result.get(key)
            if torch.is_tensor(value):
                return value.to(torch.bool)
    if isinstance(result, (tuple, list)):
        for value in result:
            if torch.is_tensor(value) and value.ndim == 2:
                return value.to(torch.bool)
    return torch.zeros(1, frame_count, dtype=torch.bool)


class Emotion2vecReferenceModel(nn.Module):
    """Official FunASR FP32 feature reference; not used to build the HMONNX graph."""

    def __init__(self, native_model: Any, *, sampling_rate: int = 16000, window_samples: int = 256000):
        super().__init__()
        self.native_model = native_model
        self.sampling_rate = int(sampling_rate)
        self.window_samples = int(window_samples)

    def forward(self, waveform: torch.Tensor, valid_samples: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if not torch.jit.is_tracing():
            if waveform.ndim != 2 or waveform.shape[0] != 1 or waveform.shape[1] != self.window_samples:
                raise ValueError("emotion2vec reference model expects batch size 1 and fixed waveform length")
            if valid_samples.ndim != 1 or valid_samples.shape[0] != 1:
                raise ValueError("emotion2vec reference model expects valid_samples with shape [1]")

        valid = valid_samples.to(torch.int64)
        positions = torch.arange(self.window_samples, device=waveform.device).unsqueeze(0)
        valid_mask = positions < valid.unsqueeze(1)
        valid_weights = valid_mask.to(waveform.dtype)
        denominator = valid.to(waveform.dtype).unsqueeze(1)
        mean = (waveform * valid_weights).sum(dim=1, keepdim=True) / denominator
        centered = (waveform - mean) * valid_weights
        variance = centered.square().sum(dim=1, keepdim=True) / denominator
        source = centered * torch.rsqrt(variance + 1e-5)
        padding_mask = ~valid_mask

        result = self.native_model.forward(
            source=source,
            padding_mask=padding_mask,
            mask=False,
            features_only=True,
            remove_extra_tokens=True,
        )

        frame_features = extract_x_from_result(result)
        frame_padding_mask = extract_mask_from_result(result, frame_count=frame_features.shape[1])
        # Keep the mask graph-connected so ONNX constant folding does not drop
        # the named output before xhquant imports the graph.
        frame_padding_mask = frame_padding_mask | torch.isnan(frame_features[..., 0])
        valid_frame_weights = (~frame_padding_mask).unsqueeze(-1).to(frame_features.dtype)
        utterance_feature = (frame_features * valid_frame_weights).sum(dim=1) / valid_frame_weights.sum(
            dim=1
        ).clamp_min(1.0)
        return frame_features, frame_padding_mask, utterance_feature


def classify_utterance_feature(
    utterance_feature: torch.Tensor,
    projection: nn.Linear,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the official classification head outside the feature/HMONNX graph."""
    logits = projection(utterance_feature)
    return logits, torch.softmax(logits, dim=-1)


def load_funasr_emotion2vec_model(model_dir: str):
    try:
        from funasr import AutoModel
    except ImportError as exc:  # pragma: no cover - optional dependency guard
        raise ImportError("funasr is required to load emotion2vec checkpoints") from exc

    model = AutoModel(model=model_dir, hub="ms", disable_update=True, device="cpu")
    return getattr(model, "model", model)
