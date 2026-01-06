from dataclasses import dataclass

from ...llm_convert_config import LLMConvertConfig


@dataclass
class Qwen2LegacyConvertConfig(LLMConvertConfig):
    mix_search: str = None
    pass
