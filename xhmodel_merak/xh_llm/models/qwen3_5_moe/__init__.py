from transformers import AutoConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeForConditionalGeneration

# from .configuration_qwen3_5 import Qwen3_5Config
from .qwen3_5_moe_hmonnx_inference import XHQwen3_5MoeHMONNXModel
from .qwen3_5_moe_model import XHQwen3_5MoeModel
from .qwen3_5_moe_vision_model import XHQwen3_5MoeVisionModel
from .xh_qwen3_5_moe_config import XHQwen3_5Moe_VisualConfig, XHQwen3_5MoeModelConfig


__all__ = [
    # "Qwen3_5Config",
    "Qwen3_5MoeForConditionalGeneration",
    "XHQwen3_5MoeModel",
    "XHQwen3_5MoeModelConfig",
    "XHQwen3_5MoeHMONNXModel",
    "XHQwen3_5MoeVisionModel",
    "XHQwen3_5Moe_VisualConfig",
]
