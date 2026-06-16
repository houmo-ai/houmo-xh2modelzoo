"""Apply FunASR-Nano audio warp replacements."""

from __future__ import annotations

import torch.nn as nn

from funasr.models.llm_asr.adaptor import Transformer as FunASRAdaptorTransformer
from funasr.models.sense_voice.model import MultiHeadedAttentionSANM, SenseVoiceEncoderSmall
from funasr.models.transformer.attention import MultiHeadedAttention
from funasr.models.transformer.encoder import EncoderLayer as TransformerEncoderLayer

from .adaptor_ctc_warp import XHAdaptorTransformer, XHMultiHeadedAttention, XHTransformerEncoderLayer
from .encoder_warp import XHMultiHeadedAttentionSANM, XHSenseVoiceEncoderLayer, XHSenseVoiceEncoderSmall


def apply_funasr_audio_warp(module: nn.Module) -> nn.Module:
    """Replace FunASR audio modules recursively in-place."""
    for name, child in list(module.named_children()):
        if isinstance(child, MultiHeadedAttentionSANM):
            setattr(module, name, XHMultiHeadedAttentionSANM(child))
        elif isinstance(child, MultiHeadedAttention):
            setattr(module, name, XHMultiHeadedAttention(child))
        else:
            setattr(module, name, apply_funasr_audio_warp(child))

    if isinstance(module, SenseVoiceEncoderSmall):
        return XHSenseVoiceEncoderSmall(module)
    if isinstance(module, FunASRAdaptorTransformer):
        return XHAdaptorTransformer(module)
    if isinstance(module, TransformerEncoderLayer):
        if isinstance(getattr(module, "self_attn", None), XHMultiHeadedAttention):
            return XHTransformerEncoderLayer(module)
    if isinstance(getattr(module, "self_attn", None), XHMultiHeadedAttentionSANM):
        return XHSenseVoiceEncoderLayer(module)
    return module
