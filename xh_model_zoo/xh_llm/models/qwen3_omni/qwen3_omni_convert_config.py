from dataclasses import dataclass
from typing import Optional

from ...llm_convert_config import LLMConvertConfig


@dataclass
class Qwen3OmniMoeConvertConfig(LLMConvertConfig):
    """Minimal convert config for Qwen3OmniMoe routing compatibility."""

    mix_search: Optional[str] = None
    accept_hidden_layer: Optional[int] = None
    use_multimodal_position_ids: bool = True
    prefill_full_accept_hidden: bool = True
    num_logits_to_keep: Optional[int] = 1
    export_audio_encoder: bool = False
    export_vision_encoder: bool = False
    export_talker_model: bool = False
    export_talker_prediction: bool = False
