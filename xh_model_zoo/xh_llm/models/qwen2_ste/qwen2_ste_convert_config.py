import dataclasses
from dataclasses import dataclass, field

from xhquant.api import QuantScheme

from ...llm_convert_config import BaseConvertConfig


@dataclass
class SteQwen2ConvertConfig(BaseConvertConfig):
    batch_size: int = 10
    context_length: int = 512  # 上下文长度
    quant_scheme: QuantScheme = field(default_factory=QuantScheme)
    input_sequence_length: int = 256

    def to_dict(self):
        return dataclasses.asdict(self)
