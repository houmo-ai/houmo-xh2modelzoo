# Copyright 2025 HOUMO AI
#
# File: qwen3_next_convert_config.py
# Description:
#   Qwen3 Next Convert Config configuration.
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

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ...llm_convert_config import LLMConvertConfig


@dataclass
class Qwen3NextConvertConfig(LLMConvertConfig):
    mix_search: Optional[str] = None
    num_logits_to_keep: Optional[int] = 1
    linear_attention_mode: str = "auto"
    linear_chunk_size: int = 64
    enable_rope: bool = True
    alpha_scaling_layers: List[int] = field(default_factory=lambda: [8, 20])
    chunk_inverse_alpha: float = 0.5
    cumsum_matmul_quant_config: Optional[Dict] = None
    normalize_force_fp32: bool = False
    split_conv_cache: bool = True
    use_manual_depthwise_conv1d: bool = False
    fuse_gdr_ops: bool = False
