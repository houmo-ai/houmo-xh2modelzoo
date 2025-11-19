import dataclasses
from dataclasses import dataclass, field

from torch import mode

from xhquant.api import QuantScheme

from ...llm_convert_config import BaseConvertConfig


@dataclass
class BGERerankerConvertConfig(BaseConvertConfig):
    batch_size: int = 10
    context_length: int = 512  # 上下文长度
    quant_scheme: QuantScheme = field(default_factory=QuantScheme)
    mode: str = "reranker"

    def to_dict(self):
        return dataclasses.asdict(self)
