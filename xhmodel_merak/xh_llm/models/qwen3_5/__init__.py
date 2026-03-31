from transformers import AutoConfig

# from .configuration_qwen3_5 import Qwen3_5Config
from .modeling_qwen3_5 import Qwen3_5ForConditionalGeneration
from .qwen3_5_hmonnx_inference import XHQwen3_5_HMONNXModel
from .qwen3_5_llm_model import XHQwen3_5Model
from .qwen3_5_vision_model import XHQwen3_5VisionModel
from .xh_qwen3_5_config import XHQwen3_5_VisualConfig, XHQwen3_5ModelConfig


# try:
#     AutoConfig.register("qwen3_5", Qwen3_5Config)
# except ValueError:
#     pass


__all__ = [
    # "Qwen3_5Config",
    "Qwen3_5ForConditionalGeneration",
    "XHQwen3_5Model",
    "XHQwen3_5ModelConfig",
    "XHQwen3_5_HMONNXModel",
    "XHQwen3_5VisionModel",
    "XHQwen3_5_VisualConfig",
]
