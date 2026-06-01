from typing import Any

from transformers import AutoModelForCausalLM
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeForCausalLM

from ...builder import register_llm_model
from ...text_llm_model import TextLLMModel, TextLLMModelConfig
from .qwen3moe_hmonnx_inference import XHQwen3MoeLegacyHMONNXModel


class XHQwen3MoeLegacyModelConfig(TextLLMModelConfig):
    pass


@register_llm_model("Qwen3MoeForCausalLM")
class XHQwen3MoeLegacyModel(TextLLMModel):
    transformers_min_version = "4.51.0"
    transformers_max_version = "4.57.6"
    HF_MODEL_CLS = Qwen3MoeForCausalLM
    HF_AUTO_MODEL_CLS = AutoModelForCausalLM
    HMONNXINFERENCE_CLS = XHQwen3MoeLegacyHMONNXModel
    CONFIG_CLS = XHQwen3MoeLegacyModelConfig

    def __init__(self, config: XHQwen3MoeLegacyModelConfig):
        super().__init__(config)

    def init_wrap_model(self, hf_model: Any) -> Any:
        from ._model import register_wrap_modules as qwen3moe_register_wrap_modules

        qwen3moe_register_wrap_modules()
        super().init_wrap_model(hf_model)
