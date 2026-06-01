# Copyright 2025 HOUMO AI
#
# File: remove_reshape_mul.py
# Description:
#   Example script: llm/mit/process_onnx_tool/remove_reshape_mul.py
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
from onnx import helper

def remove_reshape_and_mul(model_path, output_path):
    # 1. 加载模型
    model = onnx.load(model_path)
    graph = model.graph
    nodes = graph.node

    # 2. 定位目标Reshape节点
    reshape_node = None
    for node in nodes:
        if node.name == "/encoder/encoders0.0/self_attn/Reshape_3":
            reshape_node = node
            break
    if not reshape_node:
        raise ValueError("未找到/encoder/encoders0.0/self_attn/Reshape_3节点")

    # 3. 找到Reshape的所有输出节点（Mul）
    reshape_output = reshape_node.output[0]
    mul_nodes = []
    for node in nodes:
        if node.op_type == "Mul" and reshape_output in node.input:
            mul_nodes.append(node)
    if not mul_nodes:
        print("未找到Reshape对应的Mul节点")
        onnx.save(model, output_path)
        return

    # 4. 记录Mul的前后连接：Mul的另一个输入 → Mul的输出（后续节点的输入）
    # 说明：Mul的输入是 [Reshape输出, 另一个张量]；删除Mul后，用“另一个张量”替代Mul的输出
    mul_connections = []
    for mul_node in mul_nodes:
        # 找到Mul中不是Reshape输出的那个输入（即要保留的张量）
        for inp in mul_node.input:
            if inp != reshape_output:
                mul_input_keep = inp
                break
        mul_output = mul_node.output[0]  # Mul的输出（后续节点的输入）
        mul_connections.append((mul_input_keep, mul_output))

    # 5. 删除Reshape和所有Mul节点
    nodes_to_remove = {reshape_node.name}
    nodes_to_remove.update([node.name for node in mul_nodes])
    new_nodes = [node for node in nodes if node.name not in nodes_to_remove]

    # 6. 重新连接链路：用“Mul的保留输入”替换“Mul的输出”
    for (keep_input, mul_output) in mul_connections:
        for node in new_nodes:
            for idx, inp in enumerate(node.input):
                if inp == mul_output:
                    node.input[idx] = keep_input  # 替换引用

    # 7. 更新图节点
    graph.ClearField("node")
    graph.node.extend(new_nodes)

    # 8. 保存修改后的模型
    onnx.save(model, output_path)
    print(f"已删除Reshape节点和{len(mul_nodes)}个Mul节点，并重新连接链路")


# 调用（替换为你的模型路径）
remove_reshape_and_mul(
    model_path="/data01/home/xuchen/xh2/xh2_model_zoo/examples/llm/mit/fp/model_all_del.sim.onnx",
    output_path="/data01/home/xuchen/xh2/xh2_model_zoo/examples/llm/mit/fp/model_all_del_mul.sim.onnx"
)