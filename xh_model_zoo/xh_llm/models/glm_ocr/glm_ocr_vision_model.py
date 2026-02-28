from __future__ import annotations

import torch
import torch.nn as nn

from ..base_model import BaseModel
from ..builder import MODELS
from .modeling_glm_ocr import GlmOcrForConditionalGeneration


@MODELS.register_module()
class XHGlmOcrVisionModel(BaseModel):
    """GLM-OCR Vision model with wrap/quant/export support (mirrors XHQwen2_5_VLVisionModel)."""

    def __init__(
        self,
        hf_model: str,
        wrap_cfg,
        quant_config,
        frontend_type="TorchFX",
        allow_quant=True,
        export_cfg=None,
    ):
        super().__init__(
            hf_model=hf_model,
            wrap_cfg=wrap_cfg,
            quant_config=quant_config,
            allow_quant=allow_quant,
            export_cfg=export_cfg,
            frontend_type=frontend_type,
        )

    def get_hf_model(self, device_map="cpu", **kwargs) -> GlmOcrForConditionalGeneration:
        assert self.hf_model_dir is not None
        hf_model = GlmOcrForConditionalGeneration.from_pretrained(
            self.hf_model_dir,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="cpu",
            attn_implementation="eager",
        ).eval()
        return hf_model

    def init_wrap_model(self, hf_model=None):
        if hf_model is None:
            hf_model = self.get_hf_model()

        from ._vision_model_impl import register_wrap_cls as vision_register_wrap_cls

        vision_register_wrap_cls(hf_model)
        visual = hf_model.model.visual
        self.config = visual.config
        wraped_model = super().init_wrap_model(visual)
        wraped_model.to(torch.float16)
        return wraped_model
