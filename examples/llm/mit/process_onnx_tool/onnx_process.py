# Copyright 2025 HOUMO AI
#
# File: onnx_process.py
# Description:
#   Example script: llm/mit/process_onnx_tool/onnx_process.py
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
from collections import Counter
import argparse

def count_onnx_operators(onnx_path):
    """
    统计ONNX模型中的算子类型及数量
    
    Args:
        onnx_path (str): ONNX模型文件路径
    
    Returns:
        dict: 算子类型到数量的映射
    """
    # 加载ONNX模型
    try:
        model = onnx.load(onnx_path)
    except Exception as e:
        raise ValueError(f"加载ONNX模型失败: {e}")
    
    # 提取所有节点的算子类型
    operator_types = []
    for node in model.graph.node:
        operator_types.append(node.op_type)
    
    # 统计算子数量
    operator_count = Counter(operator_types)
    
    return operator_count

def print_operator_stats(operator_count):
    """
    格式化打印算子统计结果
    
    Args:
        operator_count (dict): 算子统计结果
    """
    print("=" * 50)
    print("ONNX模型算子统计结果")
    print("=" * 50)
    # 按数量降序排序
    sorted_ops = sorted(operator_count.items(), key=lambda x: x[1], reverse=True)
    
    total_ops = sum(operator_count.values())
    print(f"总算子数量: {total_ops}")
    print(f"算子类型数: {len(operator_count)}")
    print("\n算子类型 | 数量 | 占比")
    print("-" * 30)
    
    for op_type, count in sorted_ops:
        percentage = (count / total_ops) * 100
        print(f"{op_type:<10} | {count:<6} | {percentage:.2f}%")

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='统计ONNX模型中的算子类型及数量')
    parser.add_argument('--onnx_path', type=str, help='ONNX模型文件路径', default="/data01/home/xuchen/xh2/xh2_model_zoo/examples/llm/mit/fp/model_all_del_final.sim_final.onnx")
    args = parser.parse_args()
    
    # 统计算子
    try:
        op_count = count_onnx_operators(args.onnx_path)
        print_operator_stats(op_count)
    except Exception as e:
        print(f"统计失败: {e}")

# 
# ------------------------------
# Constant   | 2908   | 28.49%
# Mul        | 1057   | 10.36%
# Add        | 847    | 8.30%
# Cast       | 645    | 6.32%
# Unsqueeze  | 569    | 5.58%
# Reshape    | 493    | 4.83%
# Transpose  | 490    | 4.80%
# Concat     | 425    | 4.16%
# Gather     | 287    | 2.81%
# Shape      | 285    | 2.79%
# ReduceMean | 284    | 2.78%
# DynamicQuantizeLinear | 281    | 2.75%
# MatMulInteger | 281    | 2.75%
# Div        | 144    | 1.41%
# Sub        | 143    | 1.40%
# Pow        | 142    | 1.39%
# Sqrt       | 142    | 1.39%
# MatMul     | 140    | 1.37%
# Where      | 140    | 1.37%
# ConstantOfShape | 71     | 0.70%
# Slice      | 70     | 0.69%
# Split      | 70     | 0.69%
# Softmax    | 70     | 0.69%
# Pad        | 70     | 0.69%
# Conv       | 70     | 0.69%
# Relu       | 70     | 0.69%
# Range      | 3      | 0.03%
# ReduceMax  | 1      | 0.01%
# Less       | 1      | 0.01%
# Expand     | 1      | 0.01%
# Tile       | 1      | 0.01%
# Equal      | 1      | 0.01%
# Neg        | 1      | 0.01%
# Exp        | 1      | 0.01%
# Sin        | 1      | 0.01%
# Cos        | 1      | 0.01%


# 3506
# ------------------------------
# Add        | 845    | 24.10%
# MatMul     | 421    | 12.01%
# Transpose  | 420    | 11.98%
# ReduceMean | 284    | 8.10%
# Reshape    | 280    | 7.99%
# Mul        | 213    | 6.08%
# Sub        | 142    | 4.05%
# Pow        | 142    | 4.05%
# Sqrt       | 142    | 4.05%
# Div        | 142    | 4.05%
# Split      | 70     | 2.00%
# Softmax    | 70     | 2.00%
# Conv       | 70     | 2.00%
# Relu       | 70     | 2.00%
# Gather     | 59     | 1.68%
# Unsqueeze  | 59     | 1.68%
# Concat     | 39     | 1.11%
# Shape      | 38     | 1.08%