from typing import Optional

import torch
from transformers import AutoModelForCausalLM

from .modeling_qwen3_omni_moe import Qwen3OmniMoeForConditionalGeneration
from ..base_model import BaseModel
from ..base_llm_model import LLMBaseModel
from ..builder import MODELS


@MODELS.register_module()
class XHQwen3OmniMoeVisionEncoderModel(LLMBaseModel):
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

    def init_wrap_model(self, hf_model: Optional[Qwen3OmniMoeForConditionalGeneration] = None):
        from ._vision_model import register_wrap_modules as vision_register_wrap_modules  # noqa: F401
        vision_register_wrap_modules()

        wraped_model = super().init_wrap_model(hf_model)
        wraped_model.to(torch.float16)
        return wraped_model

    def forward(self, *args, **kwargs):
        
        return super().forward(*args, **kwargs)




