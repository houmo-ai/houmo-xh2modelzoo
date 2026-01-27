# Copyright 2025 HOUMO AI
#
# File: qwen2_ste_convert_config.py
# Description:
#   Qwen2 Ste Convert Config configuration.
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
