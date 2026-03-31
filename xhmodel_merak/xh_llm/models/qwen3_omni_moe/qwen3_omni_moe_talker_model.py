from typing import Any

from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeTalkerForConditionalGeneration

from ...builder import register_llm_model
from ...text_llm_model import TextLLMModel
from .qwen3_omni_common import build_empty_omni_root_model, load_omni_root_model
from .qwen3_omni_hmonnx_inference import XHQwen3OmniTalkerHMONNXModel
from .xh_qwen3_omni_config import XHQwen3OmniTalkerModelConfig


@register_llm_model("Qwen3OmniMoeTalkerForConditionalGeneration")
class XHQwen3OmniMoeTalkerModel(TextLLMModel):
    transformers_min_version = "4.57.0"
    HF_MODEL_CLS = Qwen3OmniMoeTalkerForConditionalGeneration
    HF_AUTO_MODEL_CLS = Qwen3OmniMoeTalkerForConditionalGeneration
    HMONNXINFERENCE_CLS = XHQwen3OmniTalkerHMONNXModel
    CONFIG_CLS = XHQwen3OmniTalkerModelConfig

    @classmethod
    def get_hf_model(cls, hf_model_dir: str, quant_weight=None, **kwargs) -> Any:
        root_model = load_omni_root_model(hf_model_dir, cls.HF_MODEL_DTYPE, **kwargs)
        return root_model.talker

    @classmethod
    def get_empty_hf_model(cls, hf_model_dir, **kwargs) -> Any:
        return build_empty_omni_root_model(hf_model_dir).talker

    def init_wrap_model(self, hf_model: Any) -> Any:
        from ._talker_model import register_wrap_modules

        register_wrap_modules()
        super().init_wrap_model(hf_model)
