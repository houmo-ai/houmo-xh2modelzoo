"""Encoder-free Gemma4 Unified image and per-frame video subgraphs."""

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModelForImageTextToText
from transformers.models.gemma4_unified.modeling_gemma4_unified import (
    Gemma4UnifiedForConditionalGeneration,
)

import xhquant.nn.modules as xhnn
from xhquant.api import to_frontend_graph

from ...base_vision_model import BaseVisionModel
from ...llm_data_processor import BaseVisualProcessor
from ...types import VisualModelMeta
from .gemma4_series_processor import XHGemma4Processor
from .xh_gemma4_series_config import XHGemma4UnifiedVisualConfig


class Gemma4UnifiedVisionAdapter(nn.Module):
    """Trace-friendly equivalent of HF's combined Unified vision embedder."""

    def __init__(self, embed_vision: nn.Module):
        super().__init__()
        self.patch_ln1 = embed_vision.patch_ln1
        self.patch_dense = embed_vision.patch_dense
        self.patch_ln2 = embed_vision.patch_ln2
        self.pos_embedding = embed_vision.pos_embedding
        self.pos_norm = embed_vision.pos_norm
        multimodal = embed_vision.multimodal_embedder
        hidden_size = int(multimodal.embedding_projection.in_features)
        self.pre_projection_norm = xhnn.RMSNorm(hidden_size, eps=multimodal.embedding_pre_projection_norm.eps)
        self.pre_projection_norm.weight.data.fill_(1.0)
        self.pre_projection_norm.weight.requires_grad = False
        self.embedding_projection = multimodal.embedding_projection

    def forward(
        self,
        pixel_values: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        target_dtype = self.patch_dense.weight.dtype
        hidden_states = self.patch_ln1(pixel_values.to(target_dtype))
        hidden_states = self.patch_dense(hidden_states)
        hidden_states = self.patch_ln2(hidden_states)

        # Position normalization and padding detection are host preprocessing.
        # The remaining operators are token-independent, so padded rows can be
        # discarded after this graph without exposing a graph-level mask.
        pos_embeds = torch.zeros_like(hidden_states)
        for axis in range(2):
            axis_embed = F.embedding(position_ids[..., axis], self.pos_embedding[:, axis, :])
            pos_embeds = pos_embeds + axis_embed
        hidden_states = self.pos_norm(hidden_states + pos_embeds)
        hidden_states = self.pre_projection_norm(hidden_states)
        hidden_states = self.embedding_projection(hidden_states)

        return hidden_states


class _Gemma4UnifiedVisualProcessor(BaseVisualProcessor):
    def __init__(self, input_modality: str):
        self.input_modality = input_modality

    @staticmethod
    def normalize_position_inputs(position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        valid = ~(position_ids == -1).all(dim=-1, keepdim=True)
        normalized = torch.where(valid, position_ids, torch.zeros_like(position_ids))
        return normalized.to(torch.int32), valid

    def forward(self, data: dict) -> tuple[torch.Tensor, torch.Tensor]:
        if self.input_modality == "video":
            pixel_values = data.get("pixel_values", data.get("pixel_values_videos", data.get("image")))
            position_ids = data.get("video_position_ids", data.get("position_ids"))
        else:
            pixel_values = data.get("pixel_values", data.get("image"))
            position_ids = data.get("image_position_ids", data.get("position_ids"))
        if pixel_values is None or position_ids is None:
            raise ValueError(f"Gemma4 Unified {self.input_modality} graph requires pixel_values and position_ids.")
        normalized, _ = self.normalize_position_inputs(position_ids)
        return pixel_values, normalized


class XHGemma4UnifiedVisionModel(BaseVisionModel):
    transformers_min_version = "5.13.0"
    HF_MODEL_CLS = Gemma4UnifiedForConditionalGeneration
    HF_AUTO_MODEL_CLS = AutoModelForImageTextToText
    META_CLS = VisualModelMeta
    CONFIG_CLS = XHGemma4UnifiedVisualConfig

    def __init__(self, config: XHGemma4UnifiedVisualConfig):
        super().__init__(config)
        self.config = cast(XHGemma4UnifiedVisualConfig, self.config)

    def init_wrap_model(self, hf_model: Gemma4UnifiedForConditionalGeneration | None = None):
        if hf_model is None:
            hf_model = self.get_native_model()
        adapter = Gemma4UnifiedVisionAdapter(hf_model.model.embed_vision)
        return super().init_wrap_model(adapter)

    def get_tf_processor(self):
        return XHGemma4Processor.from_pretrained(self.hf_model_dir, trust_remote_code=True)

    def _get_data_preprocessor(self):
        return _Gemma4UnifiedVisualProcessor(self.config.input_modality)

    def _get_dummy_inputs(self) -> dict[str, torch.Tensor]:
        processor = self.get_tf_processor()
        if self.config.input_modality == "video":
            frames = [Image.new("RGB", (320, 240), color="white") for _ in range(4)]
            inputs = processor(
                text=[processor.video_token],
                videos=[frames],
                video_metadata=[
                    {
                        "fps": 2.0,
                        "duration": 2.0,
                        "total_num_frames": len(frames),
                        "frames_indices": list(range(len(frames))),
                        "video_backend": "synthetic",
                    }
                ],
                do_sample_frames=False,
                return_tensors="pt",
            )
            return {
                "pixel_values": inputs["pixel_values_videos"][:, 0],
                "position_ids": inputs["video_position_ids"][:, 0],
            }
        inputs = processor(
            text=[processor.image_token],
            images=[[Image.new("RGB", (640, 480), color="white")]],
            return_tensors="pt",
        )
        return {
            "pixel_values": inputs["pixel_values"],
            "position_ids": inputs["image_position_ids"],
        }

    def get_dummy_inputs(self) -> dict[str, torch.Tensor]:
        return self._get_dummy_inputs()

    def _to_fronted(self, wrap_model):
        dummy = self._get_dummy_inputs()
        positions, _ = _Gemma4UnifiedVisualProcessor.normalize_position_inputs(dummy["position_ids"])
        return to_frontend_graph(
            wrap_model.float().cpu(),
            "TorchFX",
            [
                dummy["pixel_values"].float().cpu(),
                positions.cpu(),
            ],
        )

    def get_export_cfg(self) -> dict[str, list[str]]:
        return {
            "input_names": ["pixel_values", "position_ids"],
            "output_names": ["image_embeds"],
        }

    def create_export_metadata(self, output_dir: str) -> VisualModelMeta:  # noqa: ARG002
        meta = cast(VisualModelMeta, self.get_export_metadata_cls()())
        meta.image_size_w = 0
        meta.image_size_h = 0
        meta.num_image_tokens = self.config.image_seq_length
        meta.input_dim = self.config.input_dim
        meta.position_capacity = self.config.position_capacity
        meta.input_modality = self.config.input_modality
        meta.frontend_kind = self.config.frontend_kind
        return meta

    def export_hmonnx(self, output_dir: str) -> VisualModelMeta:
        meta = self.create_export_metadata(output_dir)
        meta.hmonnx = str(super()._export_hmonnx(output_dir))
        return meta


__all__ = ["Gemma4UnifiedVisionAdapter", "XHGemma4UnifiedVisionModel"]
