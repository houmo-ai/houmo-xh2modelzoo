from dataclasses import dataclass
from typing import Optional

from ...llm_convert_config import LLMConvertConfig


@dataclass
class Qwen3OmniMoeConvertConfig(LLMConvertConfig):
    """Minimal convert config for Qwen3OmniMoe routing compatibility."""

    mix_search: Optional[str] = None
    num_logits_to_keep: Optional[int] = 1
    export_audio_encoder: bool = False
    export_vision_encoder: bool = False
    export_talker_model: bool = False
    export_talker_prediction: bool = False
