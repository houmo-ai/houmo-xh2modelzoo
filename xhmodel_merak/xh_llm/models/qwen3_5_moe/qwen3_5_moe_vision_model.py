"""
XHQwen3_5MoeVisionModel — Vision model wrapper for Qwen3.5 xhquant export.

Follows the same pattern as xhquant_llm/models/qwen3_vl/qwen3_vl_vision_model.py,
but adapted for Qwen3.5 (no deepstack features).
"""

import types

import torch
from transformers import AutoModelForImageTextToText
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeForConditionalGeneration as XHQwen3_5MoeForConditionalGeneration,
)
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeVisionModel as HFQwen3_5MoeVisionModel

from xhquant.utils.registry import DynamicModule, _DMRegistryCls

from ...base_vision_model import BaseVisionModel
from ...builder import register_llm_model
from ...types import VisualModelMeta
from ..qwen3_5.qwen3_5_vision_model import XHQwen3_5VisionModel
from .xh_qwen3_5_moe_config import XHQwen3_5Moe_VisualConfig


class _Qwen3_5Moe_VisualHFCompatible(DynamicModule):  # noqa: N801
    def _setup(self: HFQwen3_5MoeVisionModel, xh_visual_model: "XHQwen3_5MoeVisionModel"):
        dtype = self.dtype
        device = self.device

        del self.blocks
        del self.pos_embed
        del self.patch_embed
        del self.merger
        del self.rotary_pos_emb

        # 没有这行，self.dtype时会报错，因为所有参数都被移除了。
        self._dummy_param = torch.nn.Parameter(torch.empty(0, dtype=dtype, device=device))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.visual = xh_visual_model

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.visual.forward(hidden_states, **kwargs)


def build_qwen3_5_moe_visual_hf_compatible_model(
    hf_model: XHQwen3_5MoeForConditionalGeneration, xh_visual_model: "XHQwen3_5MoeVisionModel"
):
    # 构建一个兼容HF的视觉模型，主要用于导出ONNX
    LLM_COMPATIBLE_MODULES = _DMRegistryCls("XHCompatible")  # noqa: N806
    hf_model_cls = type(hf_model.model.visual)
    if hf_model_cls not in LLM_COMPATIBLE_MODULES:
        LLM_COMPATIBLE_MODULES.register_module(
            {
                hf_model_cls: hf_model_cls.__name__,
            },
            _Qwen3_5Moe_VisualHFCompatible,
        )
    LLM_COMPATIBLE_MODULES.convert(hf_model.model.visual, xh_visual_model=xh_visual_model)
    from ..qwen3_5.modeling_qwen3_5_patch import get_image_features

    hf_model.model.get_image_features = types.MethodType(get_image_features, hf_model.model)

    return hf_model


@register_llm_model("Qwen3_5MoeForConditionalGeneration_visual", master=False)
class XHQwen3_5MoeVisionModel(XHQwen3_5VisionModel):  # noqa: N801
    transformers_min_version = "5.2.0"
    HF_MODEL_CLS = XHQwen3_5MoeForConditionalGeneration
    HF_AUTO_MODEL_CLS = AutoModelForImageTextToText
    VISUAL_HF_MODEL_CLS = HFQwen3_5MoeVisionModel
    # HMONNXINFERENCE_CLS = Qwen3HMONNXModel
    BUILD_HF_COMPATIBLE_FUNC = build_qwen3_5_moe_visual_hf_compatible_model

    META_CLS = VisualModelMeta
    CONFIG_CLS = XHQwen3_5Moe_VisualConfig
    WORKFLOW_CLS = "xhmodel_merak.xh_llm.models.qwen3_5.workflow:Qwen35Workflow"

    def __init__(self, config: XHQwen3_5Moe_VisualConfig):
        super().__init__(config)
        if self.config.model_type is None:
            self.config.model_type = "Qwen3_5MoeForConditionalGeneration_visual"

    def init_wrap_model(self, hf_model: XHQwen3_5MoeForConditionalGeneration = None):
        from ._vision_model_impl import register_wrap_cls as vision_register_wrap_cls

        vision_register_wrap_cls(hf_model)
        if isinstance(hf_model, self.VISUAL_HF_MODEL_CLS):
            visual = hf_model
        else:
            visual = hf_model.model.visual
        # self.config = visual.config
        wraped_model = super(BaseVisionModel, self).init_wrap_model(visual)
        return wraped_model
