"""DeepSeek-V4 Flash model registration."""

from .deepseek_v4_hmonnx_inference import XHDeepSeekV4HMONNXModel
from .deepseek_v4_model import XHDeepSeekV4Model
from .workflow import DeepSeekV4Workflow
from .xh_deepseek_v4_config import XHDeepSeekV4ModelConfig


__all__ = [
    "XHDeepSeekV4HMONNXModel",
    "XHDeepSeekV4Model",
    "XHDeepSeekV4ModelConfig",
    "DeepSeekV4Workflow",
]
