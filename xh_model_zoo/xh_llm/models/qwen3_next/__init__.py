# Copyright 2025 HOUMO AI
#
# File: __init__.py
# Description:
#   Qwen3 Next module initialization.
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

from .qwen3_next_convert_config import Qwen3NextConvertConfig
from .qwen3_next_converter import Qwen3NextConverterXH2a
from .qwen3_next_hf_compatible import Qwen3NextHFCompatible
from .qwen3_next_llm_model import XHQwen3NextModel
from .qwen3_next_onnx_model import Qwen3NextONNXModel

__all__ = [
    "Qwen3NextConvertConfig",
    "Qwen3NextConverterXH2a",
    "Qwen3NextHFCompatible",
    "XHQwen3NextModel",
    "Qwen3NextONNXModel",
]
