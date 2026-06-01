# Copyright 2025 HOUMO AI
#
# File: __init__.py
# Description:
#   Qwen3 Vl Moe module initialization.
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

from .modeling_qwen3moe_vl import Qwen3VLMoeForConditionalGeneration
from .qwen3_vl_convert_config import Qwen3_VLConvertConfig, VisualConfig
from .processing_qwen3moe_vl import Qwen3VLProcessor
from .qwen3moe_vl_onnx_model import Qwen3VLMoeONNXModel
from .qwen3_vl_moe_converter import Qwen3_VL_MOEConverterXH2a
from .postprocess import VLLMPresencePenaltyLogitsProcessor

__all__ = [
    "Qwen3VLProcessor",
    "Qwen3VLMoeForConditionalGeneration",
    "Qwen3VLMoeONNXModel",
    "Qwen3_VLConvertConfig",
    "VisualConfig",
    "Qwen3_VL_MOEConverterXH2a",
    "VLLMPresencePenaltyLogitsProcessor",
]