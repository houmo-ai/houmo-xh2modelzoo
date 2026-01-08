from dataclasses import dataclass
from typing import Optional

from ...llm_convert_config import LLMConvertConfig


@dataclass
class Qwen3MoeConvertConfig(LLMConvertConfig):
    mix_search: Optional[str] = None
    num_logits_to_keep: Optional[str] = None
