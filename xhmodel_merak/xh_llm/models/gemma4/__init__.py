from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoModelForImageTextToText
from transformers.models.gemma4.configuration_gemma4 import Gemma4AudioConfig, Gemma4Config, Gemma4TextConfig, Gemma4VisionConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM, Gemma4ForConditionalGeneration, Gemma4TextModel, Gemma4VisionModel

from .gemma4_hmonnx_inference import XHGemma4HMONNXModel
from .gemma4_llm_model import XHGemma4Model
from .gemma4_processor import XHGemma4Processor
from .gemma4_visual_model import XHGemma4VisionModel
from .xh_gemma4_config import XHGemma4ModelConfig, XHGemma4VisualConfig

for fn in [
    lambda: AutoConfig.register("gemma4", Gemma4Config),
    lambda: AutoConfig.register("gemma4_text", Gemma4TextConfig),
    lambda: AutoConfig.register("gemma4_vision", Gemma4VisionConfig),
    lambda: AutoConfig.register("gemma4_audio", Gemma4AudioConfig),
    lambda: AutoModel.register(Gemma4TextConfig, Gemma4TextModel),
    lambda: AutoModel.register(Gemma4VisionConfig, Gemma4VisionModel),
    lambda: AutoModelForCausalLM.register(Gemma4TextConfig, Gemma4ForCausalLM),
    lambda: AutoModelForImageTextToText.register(Gemma4Config, Gemma4ForConditionalGeneration),
]:
    try:
        fn()
    except Exception:
        pass

__all__ = [
    "Gemma4AudioConfig",
    "Gemma4Config",
    "Gemma4TextConfig",
    "Gemma4VisionConfig",
    "Gemma4ForConditionalGeneration",
    "Gemma4ForCausalLM",
    "Gemma4TextModel",
    "Gemma4VisionModel",
    "XHGemma4Model",
    "XHGemma4HMONNXModel",
    "XHGemma4VisionModel",
    "XHGemma4ModelConfig",
    "XHGemma4VisualConfig",
    "XHGemma4Processor",
]
