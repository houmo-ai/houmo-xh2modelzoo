from typing import Any

from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeTalkerCodePredictorModelForConditionalGeneration,
)

from ...builder import register_llm_model
from ...text_llm_model import TextLLMModel
from .qwen3_omni_common import build_empty_omni_root_model, load_omni_root_model
from .qwen3_omni_hmonnx_inference import XHQwen3OmniTalkerPredictionHMONNXModel
from .xh_qwen3_omni_config import XHQwen3OmniTalkerPredictionModelConfig


@register_llm_model("Qwen3OmniMoeTalkerCodePredictorModelForConditionalGeneration")
class XHQwen3OmniMoeTalkerPrediction(TextLLMModel):
    transformers_min_version = "4.57.0"
    HF_MODEL_CLS = Qwen3OmniMoeTalkerCodePredictorModelForConditionalGeneration
    HF_AUTO_MODEL_CLS = Qwen3OmniMoeTalkerCodePredictorModelForConditionalGeneration
    HMONNXINFERENCE_CLS = XHQwen3OmniTalkerPredictionHMONNXModel
    CONFIG_CLS = XHQwen3OmniTalkerPredictionModelConfig

    @classmethod
    def get_hf_model(cls, hf_model_dir: str, quant_weight=None, **kwargs) -> Any:
        root_model = load_omni_root_model(hf_model_dir, cls.HF_MODEL_DTYPE, **kwargs)
        return root_model.talker.code_predictor

    @classmethod
    def get_empty_hf_model(cls, hf_model_dir, **kwargs) -> Any:
        return build_empty_omni_root_model(hf_model_dir).talker.code_predictor

    def init_wrap_model(self, hf_model: Any) -> Any:
        from ._talker_prediction import register_wrap_modules

        register_wrap_modules()
        super().init_wrap_model(hf_model)
