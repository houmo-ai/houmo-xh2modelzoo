# Copyright 2025 HOUMO AI
#
# File: qwen2_vl_convert_config.py
# Description:
#   Configuration classes for Qwen2 VL model conversion.
#   This module provides VisualConfig and Qwen2VLConvertConfig dataclasses
#   for managing Qwen2 VL conversion parameters.
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
from traitlets import default
from ...llm_convert_config import LLMConvertConfig
from dataclasses import dataclass, field

@dataclass
class VisualConfig:
    image_max_size: int = 1204
    patch_size: int = 14

    def to_dict(self):
        return dataclasses.asdict(self)


@dataclass
class Qwen2VLConvertConfig(LLMConvertConfig):
    visual_config: VisualConfig = field(default=VisualConfig)
