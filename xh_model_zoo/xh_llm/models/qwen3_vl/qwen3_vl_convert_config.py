# Copyright 2025 HOUMO AI
#
# File: qwen3_vl_convert_config.py
# Description:
#   Configuration classes for Qwen3 VL model conversion.
#   This module provides VisualConfig and Qwen3_VLConvertConfig dataclasses
#   for managing Qwen3 VL conversion parameters.
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

from ...llm_convert_config import LLMConvertConfig


@dataclass
class VisualConfig:
    image_max_size_h: int = 448
    image_max_size_w: int = 448
    image_max_size_t: int = 2
    patch_size: int = 16
    temporal_patch_size: int = 2

    sample_image_path: str = field(default_factory=str)

    def to_dict(self):
        return dataclasses.asdict(self)


@dataclass
class Qwen3_VLConvertConfig(LLMConvertConfig):
    visual_config: VisualConfig = field(default_factory=VisualConfig)
    gptqmodel_cfg: str = field(default_factory=str)
    quant_weight: str = field(default_factory=str)
    max_pe_length: int = 32768
