from dataclasses import dataclass
from typing import Optional

from ...llm_convert_config import LLMConvertConfig


@dataclass
class Qwen3LegacyLoRAConvertConfig(LLMConvertConfig):
    mix_search: str = None
    num_logits_to_keep: Optional[str] = None
    lora_checkpoint: str = None  # LoRA权重文件路径
