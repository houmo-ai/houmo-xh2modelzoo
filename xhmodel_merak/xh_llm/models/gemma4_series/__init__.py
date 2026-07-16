"""Unified Gemma4 Series model package.

Public ownership for E2B/E4B, 31B dense, and 26B-A4B MoE lives here.  Legacy
``gemma4``, ``gemma4e``, and ``gemma4_moe`` modules are historical
compatibility surfaces, not implementation sources for new Gemma4 Series work.
"""

import importlib

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoModelForImageTextToText
from transformers.models.gemma4.configuration_gemma4 import (
    Gemma4AudioConfig,
    Gemma4Config,
    Gemma4TextConfig,
    Gemma4VisionConfig,
)
from transformers.models.gemma4.modeling_gemma4 import (
    Gemma4ForCausalLM,
    Gemma4ForConditionalGeneration,
    Gemma4TextModel,
    Gemma4VisionModel,
)

from .export_plan import Gemma4SeriesExportPlan, build_gemma4_series_export_plan
from .gemma4_series_audio_model import XHGemma4AudioModel, XHGemma4SeriesAudioModel
from .gemma4_series_hmonnx_inference import XHGemma4HMONNXModel, XHGemma4SeriesHMONNXModel
from .gemma4_series_llm_model import XHGemma4Model, XHGemma4SeriesModel
from .gemma4_series_processor import XHGemma4Processor, XHGemma4SeriesProcessor
from .gemma4_series_vision_model import XHGemma4SeriesVisionModel, XHGemma4VisionModel
from .quant_adapter import build_gptqmodel_recipe_kwargs, quantize_with_gptqmodel_recipe
from .variants import Gemma4SeriesVariantSpec, resolve_gemma4_series_variant
from .workflow import (
    Gemma4SeriesWorkflow,
    XHGemma4HMONNXWorkflow,
    XHGemma4SeriesHMONNXWorkflow,
    dump_autoround_mode1_quant_config_template,
    dump_autoround_moe_mode1_quant_config_template,
    dump_export_config_template,
    dump_quant_config_template,
    export,
    get_export_config_help,
    get_quant_config_help,
    list_recommended_configs,
    quant,
)
from .xh_gemma4_series_config import (
    Gemma4AudioModelMeta,
    Gemma4ModelMeta,
    Gemma4SeriesAudioModelMeta,
    Gemma4SeriesModelMeta,
    XHGemma4AudioConfig,
    XHGemma4ModelConfig,
    XHGemma4SeriesAudioConfig,
    XHGemma4SeriesModelConfig,
    XHGemma4SeriesVisualConfig,
    XHGemma4VisualConfig,
)


_LAZY_MTP_EXPORTS = {
    "XHGemma4AssistantDraftModel",
    "XHGemma4SeriesAssistantDraftModel",
}


def __getattr__(name: str):
    if name not in _LAZY_MTP_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    mtp_model = importlib.import_module(f"{__name__}.gemma4_series_mtp_model")
    value = getattr(mtp_model, name)
    globals()[name] = value
    return value


def _register_transformers_once(register_fn) -> None:
    """Register Gemma4 HF classes while allowing idempotent imports only."""
    try:
        register_fn()
    except ValueError as exc:
        message = str(exc).lower()
        if "already" not in message and "exist" not in message:
            raise


for fn in [
    lambda: AutoConfig.register("gemma4", Gemma4Config),
    lambda: AutoConfig.register("gemma4_text", Gemma4TextConfig),
    lambda: AutoConfig.register("gemma4_vision", Gemma4VisionConfig),
    lambda: AutoConfig.register("gemma4_audio", Gemma4AudioConfig),
    lambda: AutoModel.register(Gemma4TextConfig, Gemma4TextModel),
    lambda: AutoModelForCausalLM.register(Gemma4TextConfig, Gemma4ForCausalLM),
    lambda: AutoModelForImageTextToText.register(Gemma4Config, Gemma4ForConditionalGeneration),
]:
    _register_transformers_once(fn)


__all__ = [
    "Gemma4AudioConfig",
    "Gemma4AudioModelMeta",
    "Gemma4Config",
    "Gemma4ForCausalLM",
    "Gemma4ForConditionalGeneration",
    "Gemma4ModelMeta",
    "Gemma4SeriesAudioModelMeta",
    "Gemma4SeriesExportPlan",
    "Gemma4SeriesModelMeta",
    "Gemma4SeriesVariantSpec",
    "Gemma4TextConfig",
    "Gemma4TextModel",
    "Gemma4VisionConfig",
    "Gemma4VisionModel",
    "XHGemma4AudioConfig",
    "XHGemma4AudioModel",
    "XHGemma4HMONNXModel",
    "XHGemma4Model",
    "XHGemma4ModelConfig",
    "XHGemma4Processor",
    "XHGemma4SeriesAudioConfig",
    "XHGemma4SeriesAudioModel",
    "XHGemma4SeriesHMONNXModel",
    "XHGemma4SeriesModel",
    "XHGemma4SeriesAssistantDraftModel",
    "XHGemma4AssistantDraftModel",
    "XHGemma4SeriesModelConfig",
    "XHGemma4SeriesProcessor",
    "XHGemma4SeriesVisionModel",
    "XHGemma4SeriesVisualConfig",
    "XHGemma4VisionModel",
    "XHGemma4VisualConfig",
    "build_gemma4_series_export_plan",
    "build_gptqmodel_recipe_kwargs",
    "quantize_with_gptqmodel_recipe",
    "resolve_gemma4_series_variant",
    "Gemma4SeriesWorkflow",
    "XHGemma4SeriesHMONNXWorkflow",
    "XHGemma4HMONNXWorkflow",
    "dump_autoround_moe_mode1_quant_config_template",
    "dump_autoround_mode1_quant_config_template",
    "dump_export_config_template",
    "dump_quant_config_template",
    "export",
    "get_export_config_help",
    "get_quant_config_help",
    "list_recommended_configs",
    "quant",
]
