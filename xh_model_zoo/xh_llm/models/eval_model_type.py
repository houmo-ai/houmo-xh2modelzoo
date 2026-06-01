# Copyright 2025 HOUMO AI
#
# File: eval_model_type.py
# Description:
#   Evaluation model type enumeration.
#   This module provides EvalModelType enum for different
#   model evaluation modes.
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
from enum import Enum, auto


class EvalModelType(Enum):
    NONE = auto()
    WRAPED = auto()  # 由改写后的模型
    FRONTEND = auto()  # 由原始模型得到的计算图, 在nn.Module中运行
    QUANTED_DISABLED = auto()  # 在QModule中运行,但禁用量化计算，浮点计算，
    QUANTED_ALIGNED = auto()  # 运行真正量化计算，在QModule中运行，可做Golden
    QUANTED_FAST = auto()  # 在QModule中运行, 准确度介于Fake和Aligned之间
    EXPORTED = auto()  # 用于导出的计算图，做golden data
    CALIBRATION = auto  # 标定QModule

    def __str__(self):
        # return self.name.capitalize()
        return self.name

    def __repr__(self) -> str:
        return self.value
