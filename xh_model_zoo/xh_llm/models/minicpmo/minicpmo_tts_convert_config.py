# Copyright 2025 HOUMO AI
#
# File: minicpmo_tts_convert_config.py
# Description:
#   Minicpmo Tts Convert Config configuration.
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

from dataclasses import dataclass
from typing import Tuple

from ...llm_convert_config import LLMConvertConfig


@dataclass
class MinicpmoTTSConvertConfig(LLMConvertConfig):
    audio: str = ""
    video: str = ""
    debug: bool = False
    valid: bool = False
    image_slice_max_size: Tuple[int, int] = (40, 40)

