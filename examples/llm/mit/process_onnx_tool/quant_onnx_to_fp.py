# Copyright 2025 HOUMO AI
#
# File: quant_onnx_to_fp.py
# Description:
#   Example script: llm/mit/process_onnx_tool/quant_onnx_to_fp.py
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

from collections import defaultdict

import numpy as np
import onnx
from onnx import TensorProto, helper
from onnxsim import simplify


def parse_raw_data(initializer):
    """
    解析initializer的raw_data字段（兼容int8/fp32等类型）
    返回：numpy数组（对应数据类型）
    """
    dtype_map = {
        TensorProto.UINT8: np.uint8,
        TensorProto.INT8: np.int8,
        TensorProto.INT32: np.int32,
        TensorProto.FLOAT: np.float32,
    }
    dtype = dtype_map[initializer.data_type]

    # 解析raw_data为numpy数组
    if len(initializer.raw_data) == 0:
        if len(initializer.float_data) != 0:
            data = np.array(initializer.float_data).reshape(-1)
        else:
            data = np.array(initializer.int32_data).reshape(-1)
    else:
        data = np.frombuffer(initializer.raw_data, dtype=dtype)
        # 重塑为指定维度
    if initializer.dims:
        data = data.reshape(initializer.dims)
    return data


def find_node_connections(nodes):
    """
    构建节点输入输出映射，快速定位节点关联关系
    返回：
        input_to_nodes: {输入张量名: [使用该输入的节点列表]}
        output_to_node: {输出张量名: 生成该输出的节点}
    """
    input_to_nodes = defaultdict(list)
    output_to_node = {}

    for node in nodes:
        # 输出映射：张量名 -> 节点
        for output in node.output:
            output_to_node[output] = node
        # 输入映射：张量名 -> 节点列表
        for inp in node.input:
            input_to_nodes[inp].append(node)

    return input_to_nodes, output_to_node


def replace_single_matmul_integer(
    model, graph, initializers, nodes, input_to_nodes, output_to_node
):
    """
    替换单个MatMulInteger链路（返回是否替换成功）
    """
    # 1. 找第一个MatMulInteger节点
    matmul_int_node = None
    insert_id = 0
    for idx, node in enumerate(nodes):
        if node.op_type == "MatMulInteger":
            matmul_int_node = node
            insert_id = idx
            break
    if not matmul_int_node:
        return False  # 无MatMulInteger，退出循环

    # 2. 定位对应的DynamicQuantizeLinear节点（通过MatMulInteger的输入回溯）
    dq_node = None
    # MatMulInteger输入0：INT8输入张量 -> 找到生成该张量的DynamicQuantizeLinear
    int8_input_tensor = matmul_int_node.input[0]
    if int8_input_tensor in output_to_node:
        dq_node = output_to_node[int8_input_tensor]
        if dq_node.op_type != "DynamicQuantizeLinear":
            # 若不是直接关联，递归找上游的DynamicQuantizeLinear
            for inp in dq_node.input:
                if (
                    inp in output_to_node
                    and output_to_node[inp].op_type == "DynamicQuantizeLinear"
                ):
                    dq_node = output_to_node[inp]
                    break

    if not dq_node or dq_node.op_type != "DynamicQuantizeLinear":
        raise ValueError(
            f"未找到MatMulInteger[{matmul_int_node.name}]对应的DynamicQuantizeLinear"
        )

    # 3. 激活量化参数 提取量化参数
    # 3.1 DynamicQuantizeLinear的原始FP32输入
    orig_fp32_input = dq_node.input[0]
    # # 3.2 DynamicQuantizeLinear的scale输出（用于反量化）
    # scale_tensor = dq_node.output[1]
    # zero_point_tensor = dq_node.output[2]

    # # 3.3 提取scale值（常量/初始值）
    # scale_val = None
    # if scale_tensor in initializers:
    #     scale_val = np.array(initializers[scale_tensor].float_data).reshape(-1)[0]
    # else:
    #     # 若scale不是初始值，找生成scale的节点（Constant/Mul）
    #     scale_node = output_to_node.get(scale_tensor)
    #     if scale_node and scale_node.op_type == "Constant":
    #         scale_val = np.array(scale_node.attribute[0].t.float_data).reshape(-1)[0]

    # if scale_val is None:
    #     raise ValueError(f"无法提取MatMulInteger[{matmul_int_node.name}]的scale值")
    Mul_scale_node = input_to_nodes[dq_node.output[1]][0]
    # 提取b_zero_point（从属性或输入）
    scale_val = 0
    scale_tensor = Mul_scale_node.input[1]
    if scale_tensor in initializers:
        scale_val = parse_raw_data(initializers[scale_tensor])
    else:
        raise NotImplementedError

    # 3.4 提取MatMulInteger的INT8权重和b_zero_point
    int8_weight_name = matmul_int_node.input[1]
    if int8_weight_name not in initializers:
        raise ValueError(
            f"MatMulInteger[{matmul_int_node.name}]的权重不是初始值，暂不支持动态权重"
        )

    # 提取b_zero_point（从属性或输入）
    b_zero_point = 0
    # 方式1：从属性取
    for attr in matmul_int_node.attribute:
        if attr.name in ["b_zero_point", "zero_point"]:
            b_zero_point = attr.i
            break
    # 方式2：从输入取（部分模型将zero_point作为输入）
    if b_zero_point == 0 and len(matmul_int_node.input) >= 3:
        zp_tensor = matmul_int_node.input[3]
        if zp_tensor in initializers:
            b_zero_point = parse_raw_data(initializers[zp_tensor])

    # 4. 反量化INT8权重为FP32
    int8_weight_init = initializers[int8_weight_name]
    int8_weight_data = parse_raw_data(int8_weight_init)
    # 反量化公式：fp32 = (int8 - zero_point) * scale
    fp32_weight_data = (int8_weight_data - b_zero_point) * scale_val

    # 创建FP32权重初始值
    fp32_weight_name = f"{int8_weight_name}_fp32_{matmul_int_node.name}"
    # fp32_weight_init = helper.make_tensor(
    #     name=fp32_weight_name,
    #     data_type=TensorProto.FLOAT,
    #     dims=int8_weight_init.dims,
    #     vals=fp32_weight_data.flatten().tolist()
    # )

    fp32_weight_init = TensorProto()
    fp32_weight_init.name = fp32_weight_name
    fp32_weight_init.data_type = TensorProto.FLOAT
    fp32_weight_init.dims.extend(int8_weight_init.dims)  # 复制维度
    # 将numpy数组转为二进制raw_data（关键：替代vals的列表存储）
    fp32_weight_init.raw_data = fp32_weight_data.tobytes()

    # fp32_weight_init.raw_data = fp32_weight_data.astype(np.float32).tobytes()
    graph.initializer.append(fp32_weight_init)
    initializers[fp32_weight_name] = fp32_weight_init  # 更新初始值映射

    # 5. 定位MatMulInteger的输出下游节点（替换其输入）
    matmul_int_output = input_to_nodes[Mul_scale_node.output[0]][0].output[0]
    downstream_nodes = input_to_nodes.get(matmul_int_output, [])

    # 6. 创建普通MatMul节点
    matmul_fp32_output = f"{matmul_int_output}_fp32"
    matmul_fp32_node = helper.make_node(
        op_type="MatMul",
        inputs=[orig_fp32_input, fp32_weight_name],
        outputs=[matmul_fp32_output],
        name=f"MatMul_FP32_{matmul_int_node.name}",
    )

    # 7. 替换下游节点的输入（将原MatMulInteger输出替换为新FP32输出） # /encoder/encoders0.0/self_attn/linear_q_k_v/MatMul_output_0_fp32
    for downstream_node in downstream_nodes:
        for idx, inp in enumerate(downstream_node.input):
            if inp == matmul_int_output:
                downstream_node.input[idx] = matmul_fp32_output

    # 8. 标记需要删除的节点（量化相关）
    nodes_to_remove = {
        dq_node.name,
        matmul_int_node.name,
        Mul_scale_node.name,
        input_to_nodes[Mul_scale_node.output[0]][0].name,
        input_to_nodes[matmul_int_node.output[0]][0].name,
    }
    # # 额外删除zero_point相关的Cast节点（若有）
    # for node in nodes:
    #     if (node.op_type == "Cast") or \
    #        (node.op_type == "Mul" and scale_tensor in node.input) or \
    #         nodes_to_remove.add(node.name)

    # 9. 清理节点并添加新MatMul
    new_nodes = []
    for idx, node in enumerate(nodes):
        if idx == insert_id:
            # print(f"insert matmul node:{len()}")
            new_nodes.append(matmul_fp32_node)
        if node.name not in nodes_to_remove:
            new_nodes.append(node)
    # new_nodes.append(matmul_fp32_node)

    # 更新图节点
    graph.ClearField("node")
    graph.node.extend(new_nodes)

    return True


def batch_dequantize_matmul(quant_onnx_path, output_onnx_path):
    """
    批量替换所有MatMulInteger，直到无量化算子
    """
    # 1. 加载模型
    model = onnx.load(quant_onnx_path)
    graph = model.graph
    initializers = {init.name: init for init in graph.initializer}

    # 2. 循环替换MatMulInteger，直到全部替换完成
    replace_count = 0
    while True:
        # 每次循环重新构建节点关联（因为节点结构会变化）
        nodes = graph.node
        input_to_nodes, output_to_node = find_node_connections(nodes)

        # 尝试替换一个MatMulInteger
        replaced = replace_single_matmul_integer(
            model, graph, initializers, nodes, input_to_nodes, output_to_node
        )

        if not replaced:
            break
        replace_count += 1
        print(f"已替换 {replace_count} 个MatMulInteger算子")

    if replace_count == 0:
        print("模型中无MatMulInteger算子，无需转换")
        onnx.save(model, output_onnx_path)
        return

    # 3. 简化模型（清理冗余节点/张量）
    print("开始简化模型...")
    model_simp, check = simplify(model)
    assert check, "模型简化验证失败"

    # 4. 最终检查：确保无量化算子
    final_nodes = model_simp.graph.node
    quant_ops = [
        n.op_type
        for n in final_nodes
        if n.op_type in ["MatMulInteger", "DynamicQuantizeLinear"]
    ]
    if quant_ops:
        print(
            f"警告：仍残留量化算子 {set(quant_ops)}，但已替换所有可处理的MatMulInteger"
        )
    else:
        print("✅ 所有量化算子已替换完成")

    # 5. 保存最终FP32模型
    onnx.save(
        model_simp,
        output_onnx_path,
        save_as_external_data=True,
        size_threshold=1,
        all_tensors_to_one_file=True,
    )
    print(
        f"\n转换完成！共替换 {replace_count} 个MatMulInteger，FP32模型保存至：{output_onnx_path}"
    )


# 调用示例
if __name__ == "__main__":
    # 替换为你的模型路径
    QUANTIZED_ONNX = "/data02/datasets/MIT/demo_model/ASR/model.int8.onnx"  # 量化模型
    FP32_ONNX = "/data01/home/xuchen/xh2/xh2_model_zoo/examples/llm/mit/asr_raw_data/model.fp_all.onnx"  # 输出FP32模型

    batch_dequantize_matmul(QUANTIZED_ONNX, FP32_ONNX)
