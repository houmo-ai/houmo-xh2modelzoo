# Copyright 2025 HOUMO AI
#
# File: bge_reranker_convert_config.py
# Description:
#   Bge Reranker Convert Config configuration.
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
