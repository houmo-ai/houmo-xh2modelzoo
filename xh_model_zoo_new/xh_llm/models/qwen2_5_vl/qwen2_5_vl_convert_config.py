import dataclasses
from dataclasses import dataclass, field

from xh_model_zoo_new.xh_llm.base_llm_converter_config import BaseLLMConverterConfig


@dataclass
class VisualConfig:
    image_max_size_h: int = 364
    image_max_size_w: int = 644
    image_max_size_t: int = 2
    patch_size: int = 14
    temporal_patch_size: int = 2

    sample_image_path: str = field(default_factory=str)

    def to_dict(self):
        return dataclasses.asdict(self)


@dataclass
class Qwen2_5_VLConvertConfig(BaseLLMConverterConfig):
    visual_config: VisualConfig = field(default_factory=VisualConfig)
    gptqmodel_cfg: str = field(default_factory=str)
    quant_weight: str = field(default_factory=str)
    max_pe_length: int = 32768
