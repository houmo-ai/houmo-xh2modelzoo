# Copyright 2025 HOUMO AI
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import Optional

from xhquant.api import QuantScheme


@dataclass
class LLMConvertConfig:
    batch_size: int = 1
    context_length: int = 2048
    input_sequence_length: int = 256
    quant_scheme: QuantScheme = field(default_factory=QuantScheme)
    quant_weight: Optional[str] = None
    eval_ppl: bool = False


@dataclass
class HyMT2ConvertConfig(LLMConvertConfig):
    mix_search: Optional[str] = None
    num_logits_to_keep: Optional[int] = 1
