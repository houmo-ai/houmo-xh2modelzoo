import os.path as osp
import argparse
import numpy as np
import torch
import onnx
import onnxruntime as ort
import onnxsim
from onnxsim import simplify
from onnx import helper, TensorProto, numpy_helper

# ==============================
# 路径配置
# ==============================
MODEL_PATH = "/data01/home/she.gao/xh2modelzoo/examples/audio/Cosyvoice3/onnx/hift.onnx"
MODEL_SIMPLIFY = "/data01/home/she.gao/xh2modelzoo/examples/audio/Cosyvoice3/onnx/hift_simplify.onnx"
REFLECT_CONSTANT = "/data01/home/she.gao/xh2modelzoo/examples/audio/Cosyvoice3/onnx/hift_simplify_reflect_replaced_constant.onnx"
FINAL_ONNX = "/data01/home/she.gao/xh2modelzoo/examples/audio/Cosyvoice3/onnx/hift_simplify_final.onnx"
FINAL_ONNX_1 = "/data01/home/she.gao/xh2modelzoo/examples/audio/Cosyvoice3/onnx/hift_simplify_final_1.onnx"
OUTPUT_PATH = "/data01/home/she.gao/xh2modelzoo/examples/audio/Cosyvoice3/hmonnx"

FIXED_DIMS = {
    "batch_size": 1,
    "seq_len": 1024,
}

# ==============================
# 工具函数
# ==============================

def print_model_io(model_path):
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

    print("模型输入节点:")
    for node in session.get_inputs():
        print(f"{node.name} | {node.type} | {node.shape}")

    print("模型输出节点:")
    for node in session.get_outputs():
        print(f"{node.name} | {node.type} | {node.shape}")


def fix_input_shape(model):
    for inp in model.graph.input:
        dims = [
            d.dim_value if d.dim_value != 0 else d.dim_param
            for d in inp.type.tensor_type.shape.dim
        ]

        new_dims = []
        for d in dims:
            if d in FIXED_DIMS:
                new_dims.append(FIXED_DIMS[d])
            else:
                new_dims.append(d if isinstance(d, int) else 0)

        inp.type.tensor_type.shape.dim.clear()
        for v in new_dims:
            inp.type.tensor_type.shape.dim.add(dim_value=v)


def simplify_model(model, save_path):
    model_s, check = simplify(model)
    assert check
    onnx.save(model_s, save_path)
    return model_s


# ==============================
# Resize 修改
# ==============================

def replace_resize_with_sizes(model):

    graph = model.graph

    sizes = numpy_helper.from_array(
        np.array([1, 9, 1024], dtype=np.int64),
        name="sizes",
    )

    graph.initializer.append(sizes)

    for node in graph.node:
        if node.name == "/m_source/l_sin_gen/Resize":
            node.input[2] = ""
            node.input.append("sizes")


# ==============================
# ScatterElements 分解
# ==============================

def transform_scatter_add(model_path, output_path, win=16, hop=4):

    model = onnx.load(model_path)
    graph = model.graph
    K = win // hop

    const_map = {init.name: numpy_helper.to_array(init) for init in graph.initializer}

    new_graph_nodes = []
    transform_count = 0

    for node in graph.node:
        if node.op_type == 'ScatterElements':
            reduction = 'none'
            axis = 0
            for attr in node.attribute:
                if attr.name == 'reduction':
                    reduction = attr.s.decode() if attr.s else 'none'
                elif attr.name == 'axis':
                    axis = attr.i

            if reduction == 'add':
                print(f"\n分解节点: {node.name}")
                transform_count += 1

                data = node.input[0]
                indices_name = node.input[1]
                updates = node.input[2]
                output = node.output[0]

                print(f"  data: {data}")
                print(f"  indices: {indices_name}")
                print(f"  updates: {updates}")

                # 读取 indices 常量（或动态生成的值）
                if indices_name in const_map:
                    indices = const_map[indices_name]
                    is_indices_const = True
                else:
                    print(f"  警告: indices 不是常量，尝试追溯...")
                    # 如果 indices 是动态生成的，暂不支持
                    new_graph_nodes.append(node)
                    continue

                print(f"  indices shape: {indices.shape}")
                print(f"  indices dtype: {indices.dtype}")

                # 处理扁平化的 indices [1, total] 或 [total]
                if indices.ndim == 2 and indices.shape[0] == 1:
                    indices = indices[0]  # 去除单维度
                    print(f"  展平后 shape: {indices.shape}")

                total_len = len(indices)
                num_frames = total_len // win

                print(f"  总长度: {total_len}")
                print(f"  每帧长度 (win): {win}")
                print(f"  帧数: {num_frames}")
                print(f"  分组数 K: {K}")

                # 创建零张量
                zero_tensor = f"{output}_zero"
                shape_name = f"{output}_shape"

                new_graph_nodes.append(helper.make_node(
                    'Shape',
                    [data],
                    [shape_name],
                    f"{output}_Shape"
                ))

                zero_value = numpy_helper.from_array(np.array([0.0], dtype=np.float32))
                const_of_shape_node = helper.make_node(
                    'ConstantOfShape',
                    [shape_name],
                    [zero_tensor],
                    f"{output}_ConstShape"
                )
                const_of_shape_node.attribute.append(
                    helper.make_attribute('value', zero_value)
                )
                new_graph_nodes.append(const_of_shape_node)

                # 按 q 分组处理
                outs = []

                for q in range(K):
                    print(f"  处理组 q={q}")

                    # 生成新的 indices_q
                    indices_q_list = []
                    for t in range(num_frames):
                        for r in range(hop):
                            idx = (t + q) * hop + r
                            indices_q_list.append(idx)

                    indices_q = np.array(indices_q_list, dtype=np.int64)
                    # 保持与原始 indices 相同的维度（二维）
                    indices_q = indices_q.reshape(1, -1)
                    indices_q_name = f"{output}_indices_q{q}"
                    graph.initializer.append(
                        numpy_helper.from_array(indices_q, name=indices_q_name)
                    )

                    updates_q_name = f"{output}_updates_q{q}"

                    # Slice 参数
                    starts = f"{output}_starts_q{q}"
                    ends = f"{output}_ends_q{q}"
                    axes = f"{output}_axes_q{q}"
                    steps = f"{output}_steps_q{q}"

                    reshape_name = f"{output}_reshape_q{q}"
                    reshape_shape = f"{output}_reshape_shape_q{q}"

                    graph.initializer.append(
                        helper.make_tensor(reshape_shape, TensorProto.INT64, [2],
                                          [num_frames, win])
                    )

                    new_graph_nodes.append(helper.make_node(
                        'Reshape',
                        [updates, reshape_shape],
                        [reshape_name],
                        f"{output}_Reshape_q{q}"
                    ))

                    slice_out_name = f"{output}_slice_q{q}"
                    graph.initializer.extend([
                        helper.make_tensor(f"{output}_s{q}_0", TensorProto.INT64, [2], [0, q*hop]),
                        helper.make_tensor(f"{output}_e{q}_0", TensorProto.INT64, [2], [num_frames, (q+1)*hop]),
                        helper.make_tensor(f"{output}_a{q}_0", TensorProto.INT64, [2], [0, 1]),
                    ])

                    new_graph_nodes.append(helper.make_node(
                        'Slice',
                        [reshape_name, f"{output}_s{q}_0", f"{output}_e{q}_0", f"{output}_a{q}_0"],
                        [slice_out_name],
                        f"{output}_Slice_q{q}"
                    ))

                    shape_flat_name = f"{output}_shape_flat_q{q}"
                    graph.initializer.append(
                        helper.make_tensor(shape_flat_name, TensorProto.INT64, [2], [1, -1])
                    )

                    new_graph_nodes.append(helper.make_node(
                        'Reshape',
                        [slice_out_name, shape_flat_name],
                        [updates_q_name],
                        f"{output}_Reshape_back_q{q}"
                    ))

                    out_q_name = f"{output}_out_q{q}"
                    scatter = helper.make_node(
                        'ScatterElements',
                        [zero_tensor, indices_q_name, updates_q_name],
                        [out_q_name],
                        f"{output}_Sct_q{q}",
                        axis=axis
                    )
                    scatter.attribute.append(helper.make_attribute('reduction', 'none'))
                    new_graph_nodes.append(scatter)
                    outs.append(out_q_name)

                # === 累加所有组 ===
                if K == 1:
                    new_graph_nodes.append(helper.make_node(
                        'Identity',
                        outs,
                        [output],
                        f"{output}_Id"
                    ))
                elif K == 2:
                    new_graph_nodes.append(helper.make_node(
                        'Add',
                        outs,
                        [output],
                        f"{output}_Add"
                    ))
                else:
                    current = outs[0]
                    for i in range(1, len(outs)):
                        next_out = f"{output}_add_{i}"
                        new_graph_nodes.append(helper.make_node(
                            'Add',
                            [current, outs[i]],
                            [next_out],
                            f"{output}_Add_{i}"
                        ))
                        current = next_out
                    new_graph_nodes[-1].output[0] = output

            else:
                new_graph_nodes.append(node)
        else:
            new_graph_nodes.append(node)

    # 替换节点
    graph.ClearField('node')
    graph.node.extend(new_graph_nodes)

    onnx.save(model, output_path)
    
    try:
        onnx.checker.check_model(output_path)
        print("✓ 模型检查通过")
    except Exception as e:
        print(f"✗ 模型检查失败: {e}")
        import traceback
        traceback.print_exc()

    return output_path


# ==============================
# ONNX → HMONNX
# ==============================

def convert_hmonnx(model_path):

    from xhquant.api import (
        convert_onnx_to_hmonnx,
        QuantScheme,
        create_quant_config,
        DeviceType,
    )

    inp = torch.randn(1, 80, 1024)

    quant_type = "w8a16_sefp"
    scheme = QuantScheme(
        target_device=DeviceType.XH2a,
        quant_type=quant_type,
    )

    config = create_quant_config(scheme)

    convert_onnx_to_hmonnx(
        model_path,
        (inp,),
        out_hmonnx_file=osp.join(OUTPUT_PATH, "hift_1024.onnx"),
        device_type="XH2A",
        quant_config=config,
    )


# ==============================
# 主流程
# ==============================

def main():

    model = onnx.load(MODEL_PATH)

    print_model_io(MODEL_PATH)

    fix_input_shape(model)
    model = simplify_model(model, MODEL_SIMPLIFY)

    for node in model.graph.node:
        if node.op_type == "Pad":
            for a in node.attribute:
                if a.name == "mode":
                    a.s = b"constant"

    simplify_model(model, REFLECT_CONSTANT)
    compare_model_outputs(MODEL_SIMPLIFY, REFLECT_CONSTANT)

    model = onnx.load(REFLECT_CONSTANT)

    replace_resize_with_sizes(model)

    simplify_model(model, FINAL_ONNX)
    compare_model_outputs(FINAL_ONNX, REFLECT_CONSTANT)


    transform_scatter_add(FINAL_ONNX, FINAL_ONNX_1)
    
    compare_model_outputs(FINAL_ONNX, FINAL_ONNX_1)

    convert_hmonnx(FINAL_ONNX_1)

def compare_model_outputs(original_model, new_model):
    """简单比较模型输出"""
    print(f"\n{'='*60}")
    print("验证模型等价性")
    print(f"{'='*60}")

    try:
        sess_orig = ort.InferenceSession(original_model, providers=['CPUExecutionProvider'])
        sess_new = ort.InferenceSession(new_model, providers=['CPUExecutionProvider'])
        print("✓ 推理会话创建成功\n")
    except Exception as e:
        print(f"✗ 创建推理会话失败: {e}")
        return False

    # 获取输入
    input_name = sess_orig.get_inputs()[0].name
    input_shape = sess_orig.get_inputs()[0].shape

    # 生成随机输入
    batch_size = input_shape[0] if input_shape[0] > 0 else 1
    seq_len = input_shape[1] if input_shape[1] > 0 else 80

    test_input = np.random.randn(batch_size, seq_len, 1024).astype(np.float32)
    print(f"测试输入形状: {test_input.shape}\n")

    try:
        outputs_orig = sess_orig.run(None, {input_name: test_input})
        outputs_new = sess_new.run(None, {input_name: test_input})

        for i, (out_orig, out_new) in enumerate(zip(outputs_orig, outputs_new)):
            if out_orig.shape != out_new.shape:
                print(f"✗ 输出 {i} 形状不匹配: {out_orig.shape} vs {out_new.shape}")
                return False

            max_diff = np.max(np.abs(out_orig - out_new))
            if max_diff > 1e-4:
                print(f"✗ 输出 {i} 不匹配，最大误差: {max_diff:.6e}")
                return False
            else:
                print(f"✓ 输出 {i} 匹配，最大误差: {max_diff:.6e}")

    except Exception as e:
        print(f"✗ 推理失败: {e}")
        return False

    print(f"\n{'='*60}")
    print("✓ 验证通过！")
    print(f"{'='*60}")

    return True

if __name__ == "__main__":
    main()