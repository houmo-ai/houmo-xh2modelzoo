from typing import Any

from transformers import AutoModelForCausalLM
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

from ...builder import register_llm_model
from ...text_llm_model import TextLLMModel, TextLLMModelConfig
from .qwen3_hmonnx_inference import XHQwen3LegacyOptHMONNXModel


class XHQwen3LegacyOptModelConfig(TextLLMModelConfig):
    pass


@register_llm_model("Qwen3ForCausalLM_legacy_opt")
class XHQwen3LegacyOptModel(TextLLMModel):
    transformers_min_version = "4.51.0"
    HF_MODEL_CLS = Qwen3ForCausalLM
    HF_AUTO_MODEL_CLS = AutoModelForCausalLM
    HMONNXINFERENCE_CLS = XHQwen3LegacyOptHMONNXModel
    CONFIG_CLS = XHQwen3LegacyOptModelConfig

    def __init__(self, config: XHQwen3LegacyOptModelConfig):
        super().__init__(config)

    def init_wrap_model(self, hf_model: Any) -> Any:
        from ._model import register_wrap_modules as qwen3_register_wrap_modules

        qwen3_register_wrap_modules()

        super().init_wrap_model(hf_model)
