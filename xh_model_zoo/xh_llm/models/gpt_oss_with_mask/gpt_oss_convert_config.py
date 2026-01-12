from dataclasses import dataclass
from typing import Optional

from ...llm_convert_config import LLMConvertConfig


@dataclass
class GptOssWithMaskConvertConfig(LLMConvertConfig):
    sliding_window: Optional[int] = None
    num_experts_per_tok: Optional[int] = None
    num_logits_to_keep: int = 1

