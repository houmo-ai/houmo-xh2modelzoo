from functools import wraps
from typing import Any

import torch
from accelerate import init_empty_weights
from transformers import AutoModelForImageTextToText
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeCausalLMOutputWithPast,
    Qwen3_5MoeForConditionalGeneration,
)

from xhmodel_merak.xh_llm.models.qwen3_5_moe.qwen3_5_moe_vision_model import XHQwen3_5MoeVisionModel
from xhquant.utils.registry import _DMRegistryCls

from ...builder import register_llm_model
from ...text_llm_hf_compatible import TextLLMHFCompatible
from ...vision_llm_model import VisionLLMModel
from ..qwen3_5.qwen3_5_llm_model import (
    Qwen3_5_ModelMeta,
    XHQwen3_5Model,
    _enforce_split_conv_cache_wrap_cfg,
)
from ..qwen3_5.qwen3_5_llm_model import (
    _Qwen3_5HFCompatible as _Qwen3_5DenseHFCompatible,
)
from .qwen3_5_moe_hmonnx_inference import XHQwen3_5MoeHMONNXModel
from .xh_qwen3_5_moe_config import XHQwen3_5MoeModelConfig


try:
    from transformers.modeling_utils import no_init_weights
except ImportError:
    no_init_weights = init_empty_weights


class _Qwen3_5HFCompatible(TextLLMHFCompatible):  # noqa: N801
    def _setup(self: Qwen3_5MoeForConditionalGeneration, text_llm_model: "XHQwen3_5MoeModel"):
        model = super()._setup(text_llm_model)
        if model is not None:
            # if hasattr(model, "model"):
            #     del model.model
            del model.model.visual
            del model.model.language_model
            if hasattr(model, "lm_head"):
                del model.lm_head
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return model

    def set_experts_implementation(self, experts_implementation):  # noqa: D401
        """No-op override for Transformers decode optimization hooks."""
        self.config.experts_implementation = experts_implementation

    def get_correct_experts_implementation(self, experts_implementation):
        return experts_implementation

    def _grouped_mm_can_dispatch(self):
        return False

    @wraps(_Qwen3_5DenseHFCompatible.forward)
    def forward(self, *args, **kwargs) -> Qwen3_5MoeCausalLMOutputWithPast:
        # Dense and MoE share patch-token routing, preprocessing and cache I/O.
        output = _Qwen3_5DenseHFCompatible.forward(self, *args, **kwargs)
        return Qwen3_5MoeCausalLMOutputWithPast(**output)

    @wraps(forward)
    def _sample_forward(self, *args, **kwargs):
        # GenerationMixin validates kwargs after generate() replaces forward.
        # Preserve the multimodal signature while the sampling hook is active.
        return super()._sample_forward(*args, **kwargs)


def build_qwen3_5_moe_hf_compatible_model(
    hf_model: Qwen3_5MoeForConditionalGeneration,
    xh_model: "XHQwen3_5MoeModel",
):
    llm_compatible_modules = _DMRegistryCls("XHCompatible")
    hf_model_cls = type(hf_model)
    if hf_model_cls not in llm_compatible_modules:
        llm_compatible_modules.register_module({hf_model_cls: hf_model_cls.__name__}, _Qwen3_5HFCompatible)
    return llm_compatible_modules.convert(hf_model, text_llm_model=xh_model)


class Qwen3_5Moe_ModelMeta(Qwen3_5_ModelMeta):  # noqa: N801
    pass


@register_llm_model("Qwen3_5MoeForConditionalGeneration")
class XHQwen3_5MoeModel(XHQwen3_5Model):  # noqa: N801
    HF_MODEL_CLS = Qwen3_5MoeForConditionalGeneration
    HF_AUTO_MODEL_CLS = AutoModelForImageTextToText
    META_CLS = Qwen3_5Moe_ModelMeta
    HMONNXINFERENCE_CLS = XHQwen3_5MoeHMONNXModel
    CONFIG_CLS = XHQwen3_5MoeModelConfig
    BUILD_HF_COMPATIBLE_FUNC = staticmethod(build_qwen3_5_moe_hf_compatible_model)
    WORKFLOW_CLS = "xhmodel_merak.xh_llm.models.qwen3_5.workflow:Qwen35Workflow"

    def __init__(self, config: XHQwen3_5MoeModelConfig):
        super().__init__(config)

        if hasattr(config, "visual_config") and config.visual_config is not None and config.visual_config.enable:
            self.visual = XHQwen3_5MoeVisionModel(config.visual_config)
            self.visual.config.model_name = f"{self.config.model_name}_visual"
        else:
            self.visual = None
        # self.full_attention_layer_indices: list[int] = []
        # self.linear_attention_layer_indices: list[int] = []
        # self._kvcache_config = KVCacheWithLinearConfig()
        # self._kvcache_mixin = KVCacheWithLinearMixin(self.kvcache_config)
        # self.config = cast(XHQwen3_5MoeModelConfig, self.config)
        # self.wrap_cfg["linear_attention_mode"] = "auto"

        # self.wrap_cfg["linear_attention_mode"] = "chunk"  # for prefill
        # self.wrap_cfg["linear_attention_mode"] = "recurrent"  # for decode

    def _wraped_post(self, hf_model: Qwen3_5MoeForConditionalGeneration):
        super()._wraped_post(hf_model)

        # The MoE wrapper follows the dense Qwen3.5 cache contract: external
        # export/runtime signatures are flat q/k/v conv-cache tensors, while
        # trace-time linear attention consumes per-layer (q, k, v) tuples.
        # Re-apply the split flag after MoE wrapping so child DynamicModules do
        # not silently trace the merged qkv path when the cache mixin/export
        # side is already split.
        language_model = self._get_language_model(self._wrap_model)
        _enforce_split_conv_cache_wrap_cfg(language_model, self.wrap_cfg)

        split_conv_cache = bool(self.wrap_cfg.get("split_conv_cache", False))
        self._kvcache_mixin.split_conv_cache = split_conv_cache
        if split_conv_cache and self.linear_attention_layer_indices:
            linear_attn = language_model.layers[self.linear_attention_layer_indices[0]].linear_attn
            self._kvcache_mixin._linear_key_dim = linear_attn.key_dim
            self._kvcache_mixin._linear_value_dim = linear_attn.value_dim

    def _get_big_language_placeholder_export_components(self):
        # Import the real MoE wrappers before entering
        # ``traceable_module_placeholder_context``.  The context temporarily
        # replaces these registrations with lightweight placeholder wrappers;
        # importing ``_moe_model`` for the first time from inside that context
        # would make its decorators register the same HF classes twice.
        from ._moe_model import register_wrap_modules
        from ._qwen3_5_moe_big_export import Qwen3_5_MOE_BigHFModel

        register_wrap_modules()
        return Qwen3_5_MOE_BigHFModel, Qwen3_5_MOE_BigHFModel.PLACEHOLDER_TYPES

    def _check_big_language_placeholder_export_supported(self, empty_hf_model: Any) -> None:
        hf_model_type = str(getattr(empty_hf_model.config, "model_type", "")).lower()
        model_type_name = type(empty_hf_model).__name__.lower()
        if "moe" not in hf_model_type and "moe" not in model_type_name:
            raise NotImplementedError(
                "Qwen3.5 dense big-model placeholder export must use the dense placeholder components."
            )

        language_model = self._get_language_model(empty_hf_model)
        if not hasattr(language_model, "modules"):
            raise NotImplementedError("Qwen3.5 MoE big-model placeholder export requires an nn.Module language model.")

        found_types = {type(module).__name__ for module in language_model.modules()}

        _, placeholder_types = self._get_big_language_placeholder_export_components()
        required_hf_types = {
            "Qwen3_5MoeAttention",
            "Qwen3_5MoeGatedDeltaNet",
            "Qwen3_5MoeSparseMoeBlock",
        }
        missing_types = sorted(required_hf_types - found_types)
        if missing_types:
            raise NotImplementedError(
                "Qwen3.5 MoE big-model placeholder export requires placeholder modules "
                f"{placeholder_types}, but missing {missing_types}."
            )

    def init_wrap_model(self, hf_model: Qwen3_5MoeForConditionalGeneration) -> Any:
        from ._moe_model import register_wrap_modules
        
        register_wrap_modules()
        wrap_model = super(VisionLLMModel, self).init_wrap_model(hf_model)
        return wrap_model
