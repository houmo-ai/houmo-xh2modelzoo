"""Patch entry points for the localized Unlimited-OCR HF implementation.

Issue 003 only localizes the original HF remote code. Operator replacement and
runtime-specific class rewrites are intentionally added in later issues.
"""

from __future__ import annotations

from typing import Dict, Type

from .modeling_deepseekv2 import (
    DeepseekV2Attention,
    DeepseekV2DecoderLayer,
    DeepseekV2ForCausalLM,
    DeepseekV2MLP,
    DeepseekV2Model,
    DeepseekV2RMSNorm,
    DeepseekV2RotaryEmbedding,
)
from .modeling_unlimitedocr import UnlimitedOCRForCausalLM, UnlimitedOCRModel


PATCHABLE_CLASSES: Dict[Type[object], Type[object]] = {
    UnlimitedOCRForCausalLM: UnlimitedOCRForCausalLM,
    UnlimitedOCRModel: UnlimitedOCRModel,
    DeepseekV2ForCausalLM: DeepseekV2ForCausalLM,
    DeepseekV2Model: DeepseekV2Model,
    DeepseekV2DecoderLayer: DeepseekV2DecoderLayer,
    DeepseekV2Attention: DeepseekV2Attention,
    DeepseekV2MLP: DeepseekV2MLP,
    DeepseekV2RMSNorm: DeepseekV2RMSNorm,
    DeepseekV2RotaryEmbedding: DeepseekV2RotaryEmbedding,
}


def unlimited_ocr_patch(hf_model: UnlimitedOCRForCausalLM) -> UnlimitedOCRForCausalLM:
    """Return a localized Unlimited-OCR HF model.

    The mapping is identity for now because issue 003 preserves original HF
    behavior. Later wrap/operator issues will replace entries in
    ``PATCHABLE_CLASSES`` with XH-specific subclasses and reuse this traversal.
    """

    for module in hf_model.modules():
        patched_cls = PATCHABLE_CLASSES.get(type(module))
        if patched_cls is not None:
            module.__class__ = patched_cls
    return hf_model
