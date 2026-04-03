from typing import Any

from transformers import AutoModelForCausalLM
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssForCausalLM

from ...builder import register_llm_model
from ...text_llm_model import TextLLMModel, TextLLMModelConfig
from .gpt_oss_hmonnx_inference import XHGptOssHMONNXModel
# from ._model import GptOssKVCacheMixin, build_gpt_oss_cache_specs


class XHGptOssModelConfig(TextLLMModelConfig):
    def __init__(self, *, num_experts_per_tok: int | None = None, **kwargs):
        super().__init__(**kwargs)
        self.num_experts_per_tok = num_experts_per_tok


@register_llm_model("GptOssForCausalLM")
class XHGptOssModel(TextLLMModel):
    transformers_min_version = "4.57.0"
    transformers_max_version = "4.57.6"
    HF_MODEL_CLS = GptOssForCausalLM
    HF_AUTO_MODEL_CLS = AutoModelForCausalLM
    HMONNXINFERENCE_CLS = XHGptOssHMONNXModel
    CONFIG_CLS = XHGptOssModelConfig

    def __init__(self, config: XHGptOssModelConfig):
        super().__init__(config)

    def init_wrap_model(self, hf_model: Any) -> Any:
        from ._model import register_wrap_modules as gpt_oss_register_wrap_modules

        gpt_oss_register_wrap_modules()
        super().init_wrap_model(hf_model)
