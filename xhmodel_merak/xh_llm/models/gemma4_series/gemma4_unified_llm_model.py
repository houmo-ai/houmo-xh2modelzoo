"""Gemma4 12B Unified model entry.

Transformers added the ``gemma4_unified`` implementation after the original
Gemma4 Series models.  Keep this class in a separate module and resolve its HF
class lazily so the model registry remains usable in a Transformers 5.5
environment.
"""

from xhmodel_merak.xh_llm.builder import register_llm_model

from .gemma4_series_llm_model import XHGemma4SeriesModel


@register_llm_model("Gemma4UnifiedForConditionalGeneration", force=True)
class XHGemma4UnifiedModel(XHGemma4SeriesModel):
    """Gemma4 12B entry backed by the Transformers 5.13 Unified classes."""

    transformers_min_version = "5.13.0"
    HF_MODEL_CLS = None

    def __init__(self, config):
        # AutoLLMModel performs its common version check after construction,
        # but this subclass needs the check before the inherited constructor
        # lazily imports the Unified vision/audio modules.
        self.check_transformer_version()
        super().__init__(config)

    @classmethod
    def get_hf_model_cls(cls):
        from transformers.models.gemma4_unified.modeling_gemma4_unified import (
            Gemma4UnifiedForConditionalGeneration,
        )

        return Gemma4UnifiedForConditionalGeneration


__all__ = ["XHGemma4UnifiedModel"]
