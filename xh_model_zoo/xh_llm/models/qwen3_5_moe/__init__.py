# Copyright 2025 HOUMO AI
#
# File: __init__.py
# Description:
#   Qwen3.5-MoE module initialization.
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

from .inference import Qwen3_5MoeInference
from .qwen3_5_moe_convert_config import Qwen3_5MoeConvertConfig
from .qwen3_5_moe_converter import Qwen3_5MoeConverterXH2a
from .qwen3_5_moe_hf_compatible import Qwen3_5MoeHFCompatible
from .qwen3_5_moe_vision_model import XHQwen3_5MoeVisionModel

__all__ = [
    "Qwen3_5MoeInference",
    "Qwen3_5MoeConvertConfig",
    "Qwen3_5MoeConverterXH2a",
    "Qwen3_5MoeHFCompatible",
    "XHQwen3_5MoeVisionModel",
]
