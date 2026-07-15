from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .audio_utils import (
    chunk_waveform,
    emotion2vec_frame_count,
    normalize_padded_waveform,
    trim_valid_frames,
    validate_audio_inputs,
)
from .configuration_emotion2vec import Emotion2vecModelMeta


class Emotion2vecHMONNXModel:
    LLM_MODEL_CLS = None

    def __init__(self, meta_info: Emotion2vecModelMeta, **kwargs):
        self.meta_info = meta_info
        self._session = None
        self._hmonnx_model = None
        if meta_info.hmonnx and Path(meta_info.hmonnx).exists():
            if os.getenv("ENABLE_HMINFERENCE_V2", "").lower() in {"1", "true", "yes", "on"}:
                from xhquant.xhonnxruntime.hmonnx_inference_v2 import HMONNXInferenceConfig, HMONNXInferenceV2

                config = HMONNXInferenceConfig()
                config.enable_golden = bool(kwargs.get("enable_golden", False))
                config.enable_auto_offload = bool(kwargs.get("enable_auto_offload", False))
                config.exec_devices = kwargs.get("device_map", [])
                self._hmonnx_model = HMONNXInferenceV2(meta_info.hmonnx, config)
            else:
                from xhquant.api import HMONNXGoldenInference

                self._hmonnx_model = HMONNXGoldenInference(meta_info.hmonnx)
                self._hmonnx_model.to("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def session(self):
        return self._session or self._hmonnx_model

    @session.setter
    def session(self, value):
        self._session = value

    def _run_session(self, waveform: torch.Tensor, valid_samples: torch.Tensor):
        session = self.session
        if session is None:
            raise ValueError("emotion2vec HMONNX session is not initialized")
        if self._session is None and torch.cuda.is_available():
            waveform = waveform.cuda()
            valid_samples = valid_samples.cuda()
        if hasattr(session, "forward"):
            return session.forward(waveform, valid_samples)
        return session(waveform, valid_samples)

    def extract_waveform(self, waveform: np.ndarray, sampling_rate: int) -> dict[str, Any]:
        validate_audio_inputs(waveform, sampling_rate, expected_sampling_rate=self.meta_info.sampling_rate)
        chunks = chunk_waveform(
            waveform,
            sampling_rate=sampling_rate,
            window_samples=self.meta_info.window_samples,
        )

        valid_features: list[torch.Tensor] = []
        for chunk in chunks:
            normalized = normalize_padded_waveform(chunk.waveform, chunk.valid_samples)
            chunk_waveform_tensor = torch.from_numpy(normalized).unsqueeze(0).to(torch.float16)
            valid_samples = torch.tensor([chunk.valid_samples], dtype=torch.int32)
            result = self._run_session(chunk_waveform_tensor, valid_samples)
            if isinstance(result, dict):
                frame_features = result["frame_features"]
                frame_mask = result["frame_padding_mask"]
            else:
                frame_features, frame_mask = result
            frame_features_numpy = frame_features.cpu().numpy()
            frame_mask_numpy = frame_mask.to(torch.bool).cpu().numpy()
            if chunk.valid_samples < self.meta_info.window_samples and not frame_mask_numpy.any():
                valid_frame_count = min(emotion2vec_frame_count(chunk.valid_samples), frame_features_numpy.shape[1])
                trimmed = frame_features_numpy[0, :valid_frame_count]
            else:
                trimmed = trim_valid_frames(frame_features_numpy, frame_mask_numpy)
            valid_features.append(torch.from_numpy(trimmed))

        frame_features = torch.cat(valid_features, dim=0)
        frame_padding_mask = torch.zeros(frame_features.shape[0], dtype=torch.bool)
        utterance_feature = frame_features.float().mean(dim=0)
        return {
            "frame_features": frame_features,
            "frame_padding_mask": frame_padding_mask,
            "utterance_feature": utterance_feature,
            "chunk_count": len(chunks),
            "valid_frame_count": int(frame_features.shape[0]),
        }

    def extract_file(self, audio_file: str | Path) -> dict[str, Any]:
        try:
            import soundfile as sf
        except ImportError as exc:  # pragma: no cover - optional dependency guard
            raise ImportError("soundfile is required for file-based emotion2vec inference") from exc

        waveform, sampling_rate = sf.read(str(audio_file), always_2d=False)
        waveform = np.asarray(waveform, dtype=np.float32)
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=-1)
        return self.extract_waveform(waveform, sampling_rate=sampling_rate)
