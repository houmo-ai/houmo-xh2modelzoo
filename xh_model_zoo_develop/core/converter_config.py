import dataclasses
from dataclasses import dataclass, field
from typing import Dict, Type, Any, List, Union
from torch import nn
from xhquant.api import QuantScheme, QuantGraph, FrontendGraph, ConfigDict, create_quant_config
from abc import abstractmethod


@dataclass
class ConverterConfig:
    """General Config"""

    def get_quant_cfg(self) -> "ConfigDict":
        return ConfigDict(create_quant_config(self.quant_scheme))

    def to_dict(self):
        return dataclasses.asdict(self)

    @classmethod
    def from_dict_or_other(cls, other: Union[dict, "ConverterConfig", Any]) -> "ConverterConfig":
        if isinstance(other, (dict, Dict)):
            return cls(**other)
        elif isinstance(other, ConverterConfig):
            # 检查 other 是否是 cls 的实例（包括子类）
            if isinstance(other, cls):  # type: ignore
                return other
            else:
                return cls(**other.to_dict())
        else:
            raise ValueError(f"Invalid type: {type(other)}")
