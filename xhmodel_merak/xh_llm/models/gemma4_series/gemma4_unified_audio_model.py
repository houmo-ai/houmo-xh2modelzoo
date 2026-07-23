"""Encoder-free Gemma4 Unified audio projection subgraph."""

from __future__ import annotations

from typing import cast

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForImageTextToText
from transformers.models.gemma4_unified.modeling_gemma4_unified import (
    Gemma4UnifiedForConditionalGeneration,
)

import xhquant.nn.modules as xhnn
from xhquant.api import to_frontend_graph

from ...base_vision_model import BaseVisionModel
from ...llm_data_processor import BaseVisualProcessor
from .gemma4_series_processor import XHGemma4Processor
from .xh_gemma4_series_config import (
    Gemma4SeriesAudioModelMeta,
    XHGemma4UnifiedAudioConfig,
)


class Gemma4UnifiedAudioAdapter(nn.Module):
    def __init__(self, embed_audio: nn.Module):
        super().__init__()
        hidden_size = int(embed_audio.embedding_projection.in_features)
        self.pre_projection_norm = xhnn.RMSNorm(hidden_size, eps=embed_audio.embedding_pre_projection_norm.eps)
        self.pre_projection_norm.weight.data.fill_(1.0)
        self.pre_projection_norm.weight.requires_grad = False
        self.embedding_projection = embed_audio.embedding_projection

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        hidden_states = self.pre_projection_norm(input_features.to(self.embedding_projection.weight.dtype))
        return self.embedding_projection(hidden_states)


class _Gemma4UnifiedAudioProcessor(BaseVisualProcessor):
    def forward(self, data: dict) -> tuple[torch.Tensor]:
        features = data.get("input_features")
        if features is None:
            raise ValueError("Gemma4 Unified audio graph requires input_features.")
        return (features,)


class XHGemma4UnifiedAudioModel(BaseVisionModel):
    transformers_min_version = "5.13.0"
    HF_MODEL_CLS = Gemma4UnifiedForConditionalGeneration
    HF_AUTO_MODEL_CLS = AutoModelForImageTextToText
    META_CLS = Gemma4SeriesAudioModelMeta
    CONFIG_CLS = XHGemma4UnifiedAudioConfig

    def __init__(self, config: XHGemma4UnifiedAudioConfig):
        super().__init__(config)
        self.config = cast(XHGemma4UnifiedAudioConfig, self.config)

    def init_wrap_model(self, hf_model: Gemma4UnifiedForConditionalGeneration | None = None):
        if hf_model is None:
            hf_model = self.get_native_model()
        adapter = Gemma4UnifiedAudioAdapter(hf_model.model.embed_audio)
        return super().init_wrap_model(adapter)

    def get_tf_processor(self):
        return XHGemma4Processor.from_pretrained(self.hf_model_dir, trust_remote_code=True)

    def _get_data_preprocessor(self):
        return _Gemma4UnifiedAudioProcessor()

    def get_dummy_inputs(self):
        processor = self.get_tf_processor()
        audio = np.zeros(self.config.sampling_rate, dtype=np.float32)
        inputs = processor(
            text=[processor.audio_token],
            audio=[audio],
            sampling_rate=self.config.sampling_rate,
            return_tensors="pt",
        )
        return {
            "input_features": inputs["input_features"],
        }

    def _to_fronted(self, wrap_model):
        dummy = self.get_dummy_inputs()
        return to_frontend_graph(
            wrap_model.float().cpu(),
            "TorchFX",
            [dummy["input_features"].float().cpu()],
        )

    def get_export_cfg(self) -> dict[str, list[str]]:
        return {
            "input_names": ["input_features"],
            "output_names": ["audio_embeds"],
        }

    def create_export_metadata(self, output_dir: str) -> Gemma4SeriesAudioModelMeta:  # noqa: ARG002
        meta = cast(Gemma4SeriesAudioModelMeta, self.get_export_metadata_cls()())
        meta.sampling_rate = self.config.sampling_rate
        meta.feature_size = self.config.feature_size
        meta.input_feature_length = self.config.input_feature_length
        meta.frontend_kind = self.config.frontend_kind
        return meta

    def export_hmonnx(self, output_dir: str) -> Gemma4SeriesAudioModelMeta:
        meta = self.create_export_metadata(output_dir)
        meta.hmonnx = str(super()._export_hmonnx(output_dir))
        return meta


__all__ = ["Gemma4UnifiedAudioAdapter", "XHGemma4UnifiedAudioModel"]
