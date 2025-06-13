import dataclasses
from dataclasses import dataclass
from typing import Optional

from xhquant.api import QuantScheme


@dataclass
class LLMConvertConfig:
    batch_size: int = 1
    context_length: int = 2048  # 上下文长度
    input_sequence_length: int = 256  # prefill阶段的输入的最大序列长度
    quant_scheme: QuantScheme = QuantScheme()
    quant_weight: Optional[str] = None

    def to_dict(self):
        return dataclasses.asdict(self)
