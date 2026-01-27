# Copyright 2025 HOUMO AI
#
# File: process_model.py
# Description:
#   Example script: cv/zte/process_model.py
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

import onnx
from onnx import TensorProto, helper

# 加载 ONNX 模型
model = onnx.load("data/models/model-chenjunbo-shape.onnx")

# 找到目标 Gather 节点（这里假设通过节点名称查找，你也可以根据其他特征定位）
target_gather_node = None
for node in model.graph.node:
    if node.name == "/model/embedding_model/embedding/Gather":
        target_gather_node = node
        break

if target_gather_node is None:
    raise ValueError("未找到目标 Gather 节点")

# 假设要将 indices 输入从固定值改为来自名为 new_indices_input 的输入
# 首先，确保模型中有 new_indices_input 这个输入（如果没有，需要添加）
new_input_exists = False
for input in model.graph.input:
    if input.name == "new_indices_input":
        new_input_exists = True
        break

if not new_input_exists:
    # 添加新的输入
    new_indices_input = helper.make_tensor_value_info(
        "new_indices_input", TensorProto.INT64, [422, 39]  # 根据实际形状设置
    )
    model.graph.input.append(new_indices_input)

# 修改 Gather 节点的 indices 输入为 new_indices_input
for i, input_name in enumerate(target_gather_node.input):
    if input_name == "onnx::Gather_2416":  # 原来的固定 indices 输入名称
        target_gather_node.input[i] = "new_indices_input"

# 保存修改后的模型
onnx.save(model, "data/models/model-chenjunbo-shape-md.onnx")
