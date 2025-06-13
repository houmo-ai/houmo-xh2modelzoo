import dataclasses
from dataclasses import dataclass

from ...llm_convert_config import LLMConvertConfig


@dataclass
class VisionConfig:
    image_size: int = 1344
    patch_size: int = 14

    def to_dict(self):
        return dataclasses.asdict(self)


@dataclass
class CogVLM2ConvertConfig(LLMConvertConfig):
    vision_config: VisionConfig = VisionConfig()
