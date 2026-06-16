from typing import Any

from transformers import AutoModelForCausalLM
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

from ...builder import register_llm_model
from ...text_llm_model import TextLLMModel, TextLLMModelConfig
from .qwen3_hmonnx_inference import XHQwen3HMONNXModel


class XHQwen3ModelConfig(TextLLMModelConfig):
    pass


@register_llm_model("Qwen3ForCausalLM")
class XHQwen3Model(TextLLMModel):
    transformers_min_version = "4.51.0"
    HF_MODEL_CLS = Qwen3ForCausalLM
    HF_AUTO_MODEL_CLS = AutoModelForCausalLM
    HMONNXINFERENCE_CLS = XHQwen3HMONNXModel
    CONFIG_CLS = XHQwen3ModelConfig
    WORKFLOW_CLS = "xhmodel_merak.xh_llm.models.qwen3.workflow:XHQwen3HMONNXWorkflow"

    def __init__(self, config: XHQwen3ModelConfig):
        super().__init__(config)

    def init_wrap_model(self, hf_model: Any) -> Any:
        from ._model import register_wrap_modules as qwen3_register_wrap_modules

        qwen3_register_wrap_modules()

        super().init_wrap_model(hf_model)
