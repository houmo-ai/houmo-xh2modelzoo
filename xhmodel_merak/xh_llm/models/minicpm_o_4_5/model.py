# ruff: noqa: E402

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from transformers import AutoModel, AutoProcessor

from ...builder import register_llm_model
from .minicpmo_base_model import XHMiniCPMOBaseModel


def _load_host(model_dir: str, device_map: str) -> nn.Module:
    model = AutoModel.from_pretrained(
        model_dir,
        trust_remote_code=True,
        attn_implementation="sdpa",
        torch_dtype=torch.float16,
        init_vision=True,
        init_audio=True,
        init_tts=True,
        device_map=device_map,
    ).eval()
    model.processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    return model


class MiniCPMO45BaseModel(XHMiniCPMOBaseModel):
    def get_hf_model(self, device_map: str = "cpu", **kwargs: Any) -> nn.Module:
        del kwargs
        return _load_host(str(self.hf_model_dir), device_map)


@register_llm_model("MiniCPMO45Model")
class MiniCPMO45Model(MiniCPMO45BaseModel):
    WORKFLOW_CLS = "xhmodel_merak.xh_llm.models.minicpm_o_4_5.workflow:MiniCPMO45Workflow"


from .minicpmo_audio_model import XHMiniCPMOAudioModel as MiniCPMO45AudioModel
from .minicpmo_llm_model import XHMiniCPMOLLMModel as MiniCPMO45LLMModel
from .minicpmo_tts_model import XHMiniCPMOTTSModel as MiniCPMO45TTSModel
from .minicpmo_vision_model import XHMiniCPMOVisionModel as MiniCPMO45VisionModel


__all__ = [
    "MiniCPMO45AudioModel",
    "MiniCPMO45LLMModel",
    "MiniCPMO45Model",
    "MiniCPMO45TTSModel",
    "MiniCPMO45VisionModel",
]
