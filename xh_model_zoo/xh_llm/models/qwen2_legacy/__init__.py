# Copyright 2025 HOUMO AI
#
# File: __init__.py
# Description:
#   Qwen2 Legacy module initialization.
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

from .inference import Qwen2LegacyInference
from .qwen2_convert_config import Qwen2LegacyConvertConfig
from .qwen2_converter import Qwen2LegacyConverterXH2a
from .qwen2_hf_compatible import Qwen2LegacyHFCompatible
from .qwen_llm_model import XHQwen2LegacyModel

# Keep old symbol name used by migration scripts.
Qwen2_HFCompatible = Qwen2LegacyHFCompatible

__all__ = [
    "XHQwen2LegacyModel",
    "Qwen2LegacyConvertConfig",
    "Qwen2LegacyConverterXH2a",
    "Qwen2LegacyInference",
    "Qwen2LegacyHFCompatible",
    "Qwen2_HFCompatible",
]
