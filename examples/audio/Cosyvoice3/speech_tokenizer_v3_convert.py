# -*- coding: utf-8 -*-
# Copyright 2025 HOUMO AI
#
# File: speech_tokenizer_v3_convert.py
# Description:
#   CosyVoice3 speech-tokenizer-v3 ONNX shape-fix converter (HOUMO tooling).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import onnx
import onnxsim
import onnxruntime as ort
import numpy as np
from onnx import helper, TensorProto, shape_inference
from onnxsim import simplify
from pathlib import Path

model_path = "/data01/home/she.gao/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B/speech_tokenizer_v3.onnx"
fixed_model_path = "/data01/home/she.gao/xh2modelzoo/examples/audio/Cosyvoice3/onnx/speech_tokenizer_v3_3000.onnx"
output_onnx = "/data01/home/she.gao/xh2modelzoo/examples/audio/Cosyvoice3/onnx/speech_tokenizer_v3_3000_1.onnx"
output_onnx_mask = "/data01/home/she.gao/xh2modelzoo/examples/audio/Cosyvoice3/onnx/speech_tokenizer_v3_3000_2.onnx"
input_model = output_onnx_mask  # 上一步生成的模型作为输入
# 移除中间文件路径定义
output_model = "/data01/home/she.gao/xh2modelzoo/examples/audio/Cosyvoice3/onnx/speech_tokenizer_v3_3000_3.onnx"

fixed_d = {
    'T': 3000,
}
feat_length_name = "feats_length"
fixed_feat_length = 3000
BLOCK_IDS = list(range(6))
MASK_INPUT_NAME = "mask1"
MASK_SHAPE = [1, 750, 1280]
MASK_DTYPE = TensorProto.FLOAT

# ---------------------- 步骤1：固定输入形状 ----------------------
# 加载模型
model = onnx.load(model_path)
providers = ['CPUExecutionProvider']
session = ort.InferenceSession(model_path, providers=providers)
print("模型输入节点:")
for input_node in session.get_inputs():
    print(f"  {input_node.name}: {input_node.type}, 形状: {input_node.shape}")
print("模型输出节点:")
for output_node in session.get_outputs():
    print(f"  {output_node.name}: {output_node.type}, 形状: {output_node.shape}")

# 修改输入节点的shape（将符号维度替换为固定值）
for input_node in model.graph.input:
    dims = [d.dim_value if d.dim_value != 0 else d.dim_param for d in input_node.type.tensor_type.shape.dim]
    new_dims = []
    for dim in dims:
        if dim in fixed_d:
            new_dims.append(fixed_d[dim])
        else:
            new_dims.append(dim if isinstance(dim, int) else 0)
    input_node.type.tensor_type.shape.dim.clear()
    for dim_val in new_dims:
        input_node.type.tensor_type.shape.dim.add(dim_value=dim_val)

slimmed_model, _ = onnxsim.simplify(model)
# 移除中间保存：onnx.save(slimmed_model, fixed_model_path)
print(f"✅ 固定形状完成（内存中处理）")

# ---------------------- 步骤2：替换feat_length为常量 ----------------------
# 直接使用步骤1的内存模型，无需从文件加载
model = slimmed_model
graph = model.graph

# 移除原 feat_length 输入
feat_length_input = next(inp for inp in graph.input if inp.name == feat_length_name)
graph.input.remove(feat_length_input)

# 创建 Constant 节点替代输入
const_node = helper.make_node(
    op_type="Constant",
    inputs=[],
    outputs=[feat_length_name],
    value=helper.make_tensor(
        name="fixed_feat_val",
        data_type=TensorProto.INT32,
        dims=[1],
        vals=[fixed_feat_length]
    )
)
graph.node.insert(0, const_node)

# 简化模型
simplified_model, check = simplify(
    model,
    check_n=0,
    skip_fuse_bn=False,
    dynamic_input_shape=False
)
assert check, "Simplified model is invalid!"
# 移除中间保存：onnx.save(simplified_model, output_onnx)
print(f"✅ 替换feat_length完成（内存中处理）")

# ---------------------- 步骤3：添加mask输入并插入Add节点 ----------------------
# 修改process_model函数，接收内存模型并返回处理后的模型
def process_model(model_in):
    model = model_in  # 直接使用输入模型，无需从路径加载
    add_mask_input(model)
    model = insert_add_before_softmax(model)
    model = simplify_model(model)
    return model  # 返回处理后的模型，不保存到磁盘

# 直接在内存中处理模型
masked_model = process_model(simplified_model)
print(f"✅ 添加mask输入完成（内存中处理）")

def add_mask_input(model, shape=[1, 20, 750, 750]):
    input_names = [i.name for i in model.graph.input]
    if "mask" in input_names:
        print("mask input already存在")
        return
    mask_input = helper.make_tensor_value_info(
        "mask", TensorProto.FLOAT, shape
    )
    model.graph.input.append(mask_input)
    print(f"Added mask input with shape {shape}")

def insert_add_before_softmax(model):
    softmax_nodes = [n for n in model.graph.node if n.op_type == "Softmax"]
    count = 0
    for node in softmax_nodes:
        original_in = node.input[0]
        new_in = original_in + "_masked"
        add_node = helper.make_node(
            "Add",
            inputs=[original_in, "mask"],
            outputs=[new_in],
            name=f"Add_mask_{count}"
        )
        node.input[0] = new_in
        idx = list(model.graph.node).index(node)
        model.graph.node.insert(idx, add_node)
        count += 1
    print(f"Inserted {count} Add nodes before Softmax ops.")
    return model

def simplify_model(model):
    print("Running onnx-simplify ...")
    model_simp, check = simplify(model)
    assert check, "onnx-simplify check failed"
    print("onnx-simplify done")
    return model_simp

def process_model(path_in, path_out):
    model = load_model(path_in)
    add_mask_input(model)
    model = insert_add_before_softmax(model)
    model = simplify_model(model)
    save_model(model, path_out)

process_model(output_onnx, output_onnx_mask)

# ---------------------- 步骤4：添加mask1输入并插入Mul节点 ----------------------
def add_mask_input_if_missing(graph, name, dtype, shape):
    existing_names = {i.name for i in graph.input}
    if name in existing_names:
        print(f"Model already has input '{name}', skip adding.")
        return
    vi = helper.make_tensor_value_info(name, dtype, shape)
    graph.input.append(vi)
    print(f"Added model input '{name}' with shape {shape}.")

def find_target_add_nodes(graph, block_id):
    name1 = f"/blocks.{block_id}/attn/value/Add"
    name2 = f"/blocks.{block_id}/attn/Add"
    return [node for node in graph.node if node.name in (name1, name2)]

def insert_mul_after_node(graph, target_node, mask_name):
    if len(target_node.output) == 0:
        raise RuntimeError(f"Target node {target_node.name} has no outputs")
    old_output = target_node.output[0]
    downstream_nodes = [node for node in graph.node if old_output in node.input]
    mul_inserted = False
    for node in downstream_nodes:
        new_mul_output = old_output + f"_mul_{node.name}"
        mul_name = f"{target_node.name}_Mul_{node.name}"
        mul_node = helper.make_node(
            "Mul",
            inputs=[old_output, mask_name],
            outputs=[new_mul_output],
            name=mul_name
        )
        try:
            idx = list(graph.node).index(target_node)
        except ValueError:
            graph.node.append(mul_node)
        else:
            graph.node.insert(idx + 1, mul_node)
        for i, inp in enumerate(node.input):
            if inp == old_output:
                node.input[i] = new_mul_output
        mul_inserted = True
        print(f"Inserted Mul '{mul_name}' for branch '{node.name}'.")
    if mul_inserted:
        for out_vi in graph.output:
            if out_vi.name == old_output:
                out_vi.name = old_output + "_mul"
    return mul_inserted

def fix_reducemean_axes_to_input(graph):
    for node in graph.node:
        if node.op_type == "ReduceMean":
            axes_attr = next((a for a in node.attribute if a.name == "axes"), None)
            if axes_attr is None:
                continue
            axes_name = node.name + "_axes"
            axes_tensor = helper.make_tensor(
                name=axes_name,
                data_type=TensorProto.INT64,
                dims=[len(axes_attr.ints)],
                vals=np.array(axes_attr.ints, dtype=np.int64)
            )
            graph.initializer.append(axes_tensor)
            node.input.append(axes_name)
            node.attribute.remove(axes_attr)
            print(f"Fixed ReduceMean '{node.name}': axes attr -> input")

def process_model_final(model_in):
    model = model_in  # 使用内存中的输入模型，无需从文件加载
    graph = model.graph
    add_mask_input_if_missing(graph, MASK_INPUT_NAME, MASK_DTYPE, MASK_SHAPE)
    total_inserted = 0
    for bid in BLOCK_IDS:
        matched = find_target_add_nodes(graph, bid)
        print(f"[block {bid}] found {len(matched)} target nodes")
        for node in matched:
            insert_mul_after_node(graph, node, MASK_INPUT_NAME)
            total_inserted += 1
    print(f"Total Mul nodes inserted: {total_inserted}")
    inferred_model = shape_inference.infer_shapes(onnx.helper.make_model(graph, producer_name="mask_inserter"))
    fix_reducemean_axes_to_input(inferred_model.graph)
    print("Running onnx-simplify ...")
    try:
        simplified_model, check = simplify(inferred_model)
    except Exception as e:
        tmp_path = output_model + ".pre_simplify.onnx"
        onnx.save(inferred_model, tmp_path)
        print(f"onnx-simplify error: {e}, saved pre-simplify model to {tmp_path}")
        raise
    if not check:
        failed_path = output_model + ".simplify_failed.onnx"
        onnx.save(simplified_model, failed_path)
        raise RuntimeError(f"onnx-simplify check failed, saved to {failed_path}")
    onnx.save(simplified_model, output_model)
    print("Done! 最终模型保存到:", output_model)

# 使用步骤3的内存模型作为输入
process_model_final(masked_model)

# ---------------------- 推理测试 ----------------------
np.random.seed(42)
# 生成原始输入
input1 = np.random.rand(1, 128, 348).astype(np.float32)
input2 = np.array([348], dtype=np.int32)

# 原始模型推理
print("\n===== 原始模型推理 =====")
onnx_model_path = model_path
session = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])
input_names = [inp.name for inp in session.get_inputs()]
inputs = {input_names[0]: input1, input_names[1]: input2}
outputs = session.run(None, inputs)
for i, out in enumerate(outputs):
    print(f"Output {i} shape: {out.shape}")

# 处理后模型推理
print("\n===== 处理后模型推理 =====")
seq_len = 348
target_len = 3000
padded_input1 = np.zeros((1, 128, target_len), dtype=np.float32)
padded_input1[:, :, :seq_len] = input1

# 构建mask
mask_shape = (1, 20, 750, 750)
mask = np.full(mask_shape, -1e9, dtype=np.float32)
mask[:, :, :, :87] = 0  # 有效区域

mask1 = np.zeros((1, 750, 1280), dtype=np.float32)
mask1[:, 0:87, :] = 1.0

# 加载最终模型
onnx_model_path = output_model
session = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])
input_names = [inp.name for inp in session.get_inputs()]
print("处理后模型输入名:", input_names)

# 构建输入字典
inputs = {
    input_names[0]: padded_input1,
    input_names[1]: mask,
    input_names[2]: mask1,
}

# 推理并打印结果
outputs = session.run(None, inputs)
for i, out in enumerate(outputs):
    print(f"Output {i} shape: {out.shape}")