import dataclasses
from dataclasses import dataclass

from ...llm_convert_config import LLMConvertConfig


@dataclass
class VisualConfig:
    image_max_size_h: int = 364
    image_max_size_w: int = 644
    patch_size: int = 14

    def to_dict(self):
        return dataclasses.asdict(self)


@dataclass
class Qwen2_5_VLConvertConfig(LLMConvertConfig):
    visual_config: VisualConfig = VisualConfig()
