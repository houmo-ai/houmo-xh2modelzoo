from dataclasses import dataclass

from ...llm_convert_config import LLMConvertConfig


@dataclass
class Qwen2ConvertConfig(LLMConvertConfig):
    update_cfg: str = None
    gptqmodel_cfg: bool = False
    pass
