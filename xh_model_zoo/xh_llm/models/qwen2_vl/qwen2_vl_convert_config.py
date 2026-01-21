import dataclasses
from traitlets import default
from ...llm_convert_config import LLMConvertConfig
from dataclasses import dataclass, field

@dataclass
class VisualConfig:
    image_max_size: int = 1204
    patch_size: int = 14

    def to_dict(self):
        return dataclasses.asdict(self)


@dataclass
class Qwen2VLConvertConfig(LLMConvertConfig):
    visual_config: VisualConfig = field(default=VisualConfig())
