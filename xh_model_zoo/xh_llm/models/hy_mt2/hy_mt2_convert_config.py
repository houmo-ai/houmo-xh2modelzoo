# Copyright 2025 HOUMO AI
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Optional

from ...llm_convert_config import LLMConvertConfig


@dataclass
class HyMT2ConvertConfig(LLMConvertConfig):
    mix_search: str = None
    num_logits_to_keep: Optional[int] = 1
