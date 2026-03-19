from typing import Optional

import torch
from transformers import AutoModelForCausalLM

from .modeling_qwen3_omni_moe import Qwen3OmniMoeForConditionalGeneration
from ..base_model import BaseModel
from ..base_llm_model import LLMBaseModel
from ..builder import MODELS


@MODELS.register_module()
class XHQwen3OmniMoeTextModel(LLMBaseModel):
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
        # from xhquant_llm.models.qwen3moe._moe_model import register_wrap_modules as qwen3moe_register_wrap_modules
        from ._text_model import register_wrap_modules as qwen3moe_register_wrap_modules
        qwen3moe_register_wrap_modules()

        self.token_embedding = hf_model.embed_tokens.to(self.dtype)

        super().init_wrap_model(hf_model)
        model = self.wrap_model

        model.to(torch.float16)
        return model

    def forward(self, *args, **kwargs):
        
        return super().forward(*args, **kwargs)
    
    def get_input_embeddings(self):
        return self.token_embedding




