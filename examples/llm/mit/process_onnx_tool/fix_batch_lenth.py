# Copyright 2025 HOUMO AI
#
# File: fix_batch_lenth.py
# Description:
#   Example script: llm/mit/process_onnx_tool/fix_batch_lenth.py
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
from onnx import helper, TensorProto

# 加载模型
model = onnx.load("/data01/home/xuchen/xh2/xh2_model_zoo/examples/llm/mit/fp/model_all.sim.onnx")
graph = model.graph

# 1. 找到x_length输入，并从输入列表中移除
x_length_input = None
for inp in graph.input:
    if inp.name == "x_length":
        x_length_input = inp
        break
if x_length_input:
    graph.input.remove(x_length_input)

# 2. 创建Constant节点，固定x_length的值（比如设为260）
fixed_x_length = 256  # 根据你的模型实际值填写
const_node = helper.make_node(
    "Constant",
    inputs=[],
    outputs=["x_length"],  # 保持原张量名，不影响下游节点
    name="fixed_x_length_const",
    value=helper.make_tensor(
        name="const_val",
        data_type=TensorProto.INT32,  # 匹配原x_length的类型
        dims=[1],  # 匹配原x_length的形状（图中是1）
        vals=[fixed_x_length]
    )
)
# 将Constant节点插入图的开头
graph.node.insert(0, const_node)

from onnxsim import simplify

# 固定其他输入的形状（如x、language等）
# input_shapes = {
#     "x": [1, 256, 560],  # 匹配x_length的固定值260
#     "language": [1],
#     "text_norm": [1]
# }

# 简化模型（此时x_length是常量，Range的limit会被推断为固定值）
model_simp, check = simplify(
    model,
    # input_shapes=input_shapes,
    # skip_fuse=False  # 此时可以开启算子融合，因为形状都固定了
)
assert check, "简化失败"

# 保存优化后的模型
onnx.save(model_simp, "/data01/home/xuchen/xh2/xh2_model_zoo/examples/llm/mit/fp/model_fix_lenth.sim.onnx")