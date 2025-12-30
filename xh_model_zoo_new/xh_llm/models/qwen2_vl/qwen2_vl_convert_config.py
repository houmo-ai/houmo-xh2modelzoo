import dataclasses
from dataclasses import dataclass

from xh_model_zoo_new.xh_llm.base_llm_converter_config import BaseLLMConverterConfig


@dataclass
class VisualConfig:
    image_max_size: int = 1204
    patch_size: int = 14

    def to_dict(self):
        return dataclasses.asdict(self)


@dataclass
class Qwen2VLConvertConfig(BaseLLMConverterConfig):
    visual_config: VisualConfig = VisualConfig()
