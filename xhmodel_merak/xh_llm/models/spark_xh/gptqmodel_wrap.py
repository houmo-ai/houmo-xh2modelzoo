import copy
from typing import Any, Dict

from gptqmodel.models import auto
from gptqmodel.models.base import BaseQModel


class IPTGPTQModel(BaseQModel):
    # allow dynamic expert index for layer_modules
    # config.num_routed_experts contains the actual expert count
    dynamic_expert_index = "num_routed_experts"  # 修复：使用 config 中的正确属性名
    require_trust_remote_code = False
    layer_modules_strict = False
    module_tree = [
        "model",
        "transformer",
        "layers",
        "#",
        {
            "layer_norm": ("layer_norm:!",),
            "attention": ("q_up_proj:0", "kv_down_proj_with_mqa:0", "kv_up_proj:1", "o_proj:2"),
            "final_layer_norm": ("final_layer_norm:!",),
            "mlp:?": {
                "router": ("gating:!",),
                "routed_experts": ("fc1:0", "fc2:1"),
                "shared_experts": ("fc1:!", "fc2:!"),
                # Standard MLP for dense layers (non-MoE)
                "fc1": ("fc1:0",),
                "fc2": ("fc2:1",),
            },
        },
    ]


auto.MODEL_MAP["ipt"] = IPTGPTQModel

auto.SUPPORTED_MODELS = list(auto.MODEL_MAP.keys())
