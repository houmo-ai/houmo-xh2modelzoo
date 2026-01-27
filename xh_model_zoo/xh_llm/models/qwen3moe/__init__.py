# Copyright 2025 HOUMO AI
#
# File: __init__.py
# Description:
#   Qwen3Moe module initialization.
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

from .inference import Qwen3MoeInference
from .qwen_moe_convert_config import Qwen3MoeConvertConfig
from .qwen_moe_converter import Qwen3MoeConverterXH2a
from .qwen_moe_hf_compatible import Qwen3MoeHFCompatible

__all__ = [
    "Qwen3MoeHFCompatible",
    "Qwen3MoeInference",
    "Qwen3MoeConverterXH2a",
    "Qwen3MoeConvertConfig",
]
