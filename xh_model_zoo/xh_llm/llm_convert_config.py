import dataclasses
from dataclasses import dataclass, field
from typing import Optional

from sympy import false

from xhquant.api import QuantScheme


@dataclass
class BaseConvertConfig:
    pass


@dataclass
class LLMConvertConfig(BaseConvertConfig):
    batch_size: int = 1
    context_length: int = 2048  # 上下文长度
    input_sequence_length: int = 256  # prefill阶段的输入的最大序列长度
    quant_scheme: QuantScheme = field(default_factory=QuantScheme)
    quant_weight: Optional[str] = None
    eval_ppl:bool = false

    def to_dict(self):
        return dataclasses.asdict(self)
