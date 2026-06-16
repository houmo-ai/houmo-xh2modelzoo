"""Backward-compatible imports for FunASR-Nano audio warp helpers.

Warp implementations are split by component:
- ``encoder_warp.py`` for SenseVoice encoder/SANM.
- ``adaptor_ctc_warp.py`` for audio adaptor and CTC decoder transformer blocks.
- ``mask_utils.py`` for externally constructed masks.
- ``warp.py`` for recursive replacement entry point.
"""

from .adaptor_ctc_warp import XHAdaptorTransformer, XHMultiHeadedAttention, XHTransformerEncoderLayer
from .encoder_warp import XHMultiHeadedAttentionSANM, XHSenseVoiceEncoderLayer, XHSenseVoiceEncoderSmall
from .mask_utils import attention_additive_mask, downsample_lengths, downsample_time, sequence_mask
from .warp import apply_funasr_audio_warp

__all__ = [
    "XHAdaptorTransformer",
    "XHMultiHeadedAttention",
    "XHTransformerEncoderLayer",
    "XHMultiHeadedAttentionSANM",
    "XHSenseVoiceEncoderLayer",
    "XHSenseVoiceEncoderSmall",
    "apply_funasr_audio_warp",
    "attention_additive_mask",
    "downsample_lengths",
    "downsample_time",
    "sequence_mask",
]
