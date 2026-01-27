# Copyright 2025 HOUMO AI
#
# File: llm_convert_config.py
# Description:
#   Configuration classes for model conversion.
#   This module provides BaseConvertConfig and LLMConvertConfig dataclasses
#   for managing conversion parameters and settings.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
import dataclasses
from dataclasses import dataclass, field
from typing import Optional,Union,Any,Dict

from xhquant.api import QuantScheme


@dataclass
class BaseConvertConfig:
    """Base configuration class for model conversion."""
    @classmethod
    def from_dict_or_other(cls, other: Union[dict, "BaseConvertConfig", Any]) -> "BaseConvertConfig":
        if isinstance(other, (dict, Dict)):
            return cls(**other)
        elif isinstance(other, BaseConvertConfig):
            # 检查 other 是否是 cls 的实例（包括子类）
            if isinstance(other, cls):  # type: ignore
                return other
            else:
                return cls(**other.to_dict())
        else:
            raise ValueError(f"Invalid type: {type(other)}")


@dataclass
class LLMConvertConfig(BaseConvertConfig):
    """Configuration class for Large Language Model conversion.

    Attributes:
        batch_size: Batch size for inference
        context_length: Maximum context length
        input_sequence_length: Maximum input sequence length for prefill stage
        quant_scheme: Quantization scheme
        quant_weight: Path to quantization weight file
        eval_ppl: Whether to evaluate perplexity
    """

    batch_size: int = 1
    context_length: int = 2048
    input_sequence_length: int = 256
    quant_scheme: QuantScheme = field(default_factory=QuantScheme)
    quant_weight: Optional[str] = None
    eval_ppl: bool = False

    def to_dict(self):
        """Convert configuration to dictionary format."""
        return dataclasses.asdict(self)
