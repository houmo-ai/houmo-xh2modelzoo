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
UPDATE_SCALE = "/data01/home/she.gao/xh2modelzoo/examples/audio/Cosyvoice3/onnx/hift_simplify_update_scale.onnx"
REFLECT_REPLACED = "/data01/home/she.gao/xh2modelzoo/examples/audio/Cosyvoice3/onnx/hift_simplify_reflect_replaced.onnx"
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


def update_scale_constant(model):
    for init in model.graph.initializer:
        if init.name == "/m_source/l_sin_gen/Constant_23_output_0":
            data = numpy_helper.to_array(init).copy()
            data[2] = np.float32(0.002084)
            init.CopyFrom(numpy_helper.from_array(data, init.name))
            break


# ==============================
# Pad reflect 替换
# ==============================

def replace_pad_reflect(graph, pad_name, L, pad):
    pad_node = None
    idx = None

    for i, node in enumerate(graph.node):
        if node.name == pad_name:
            pad_node = node
            idx = i
            break

    assert pad_node is not None

    inp = pad_node.input[0]
    out = pad_node.output[0]

    graph.initializer.extend([
        helper.make_tensor(f"{pad_name}_l_starts", TensorProto.INT64, [3], [0, 0, pad]),
        helper.make_tensor(f"{pad_name}_l_ends", TensorProto.INT64, [3], [1, 1, 0]),
        helper.make_tensor(f"{pad_name}_l_axes", TensorProto.INT64, [3], [0, 1, 2]),
        helper.make_tensor(f"{pad_name}_l_steps", TensorProto.INT64, [3], [1, 1, -1]),
    ])

    left = helper.make_node(
        "Slice",
        [
            inp,
            f"{pad_name}_l_starts",
            f"{pad_name}_l_ends",
            f"{pad_name}_l_axes",
            f"{pad_name}_l_steps",
        ],
        [f"{pad_name}_left"],
    )

    nodes = [left]

    if pad_name == "/Pad":

        graph.initializer.extend([
            helper.make_tensor(f"{pad_name}_r_starts", TensorProto.INT64, [3], [0, 0, L - 2]),
            helper.make_tensor(f"{pad_name}_r_ends", TensorProto.INT64, [3], [1, 1, L - pad - 2]),
            helper.make_tensor(f"{pad_name}_r_axes", TensorProto.INT64, [3], [0, 1, 2]),
            helper.make_tensor(f"{pad_name}_r_steps", TensorProto.INT64, [3], [1, 1, -1]),
        ])

        right = helper.make_node(
            "Slice",
            [
                inp,
                f"{pad_name}_r_starts",
                f"{pad_name}_r_ends",
                f"{pad_name}_r_axes",
                f"{pad_name}_r_steps",
            ],
            [f"{pad_name}_right"],
        )

        concat = helper.make_node(
            "Concat",
            [f"{pad_name}_left", inp, f"{pad_name}_right"],
            [out],
            axis=2,
        )

        nodes.extend([right, concat])

    else:

        concat = helper.make_node(
            "Concat",
            [f"{pad_name}_left", inp],
            [out],
            axis=2,
        )

        nodes.append(concat)

    graph.node.remove(pad_node)

    for i, n in enumerate(nodes):
        graph.node.insert(idx + i, n)


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
    const_map = {i.name: numpy_helper.to_array(i) for i in graph.initializer}

    new_nodes = []

    for node in graph.node:

        if node.op_type != "ScatterElements":
            new_nodes.append(node)
            continue

        reduction = "none"
        axis = 0

        for a in node.attribute:
            if a.name == "reduction":
                reduction = a.s.decode()
            if a.name == "axis":
                axis = a.i

        if reduction != "add":
            new_nodes.append(node)
            continue

        data, indices_name, updates = node.input
        output = node.output[0]

        if indices_name not in const_map:
            new_nodes.append(node)
            continue

        indices = const_map[indices_name]
        if indices.ndim == 2:
            indices = indices[0]

        total = len(indices)
        frames = total // win

        shape_name = f"{output}_shape"
        zero_name = f"{output}_zero"

        new_nodes.append(helper.make_node("Shape", [data], [shape_name]))

        zero_val = numpy_helper.from_array(np.array([0.0], dtype=np.float32))
        cos = helper.make_node("ConstantOfShape", [shape_name], [zero_name])
        cos.attribute.append(helper.make_attribute("value", zero_val))

        new_nodes.append(cos)

        outs = []

        for q in range(K):

            idx_list = []

            for t in range(frames):
                for r in range(hop):
                    idx_list.append((t + q) * hop + r)

            idx_arr = np.array(idx_list, dtype=np.int64).reshape(1, -1)
            idx_name = f"{output}_indices_q{q}"

            graph.initializer.append(
                numpy_helper.from_array(idx_arr, idx_name)
            )

            scatter_out = f"{output}_out_q{q}"

            scatter = helper.make_node(
                "ScatterElements",
                [zero_name, idx_name, updates],
                [scatter_out],
                axis=axis,
            )

            scatter.attribute.append(helper.make_attribute("reduction", "none"))

            new_nodes.append(scatter)
            outs.append(scatter_out)

        cur = outs[0]
        for i in range(1, len(outs)):
            add_out = f"{output}_add_{i}"
            new_nodes.append(
                helper.make_node("Add", [cur, outs[i]], [add_out])
            )
            cur = add_out

        new_nodes[-1].output[0] = output

    graph.ClearField("node")
    graph.node.extend(new_nodes)

    onnx.save(model, output_path)

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
        out_hmonnx_file=osp.join(OUTPUT_PATH, "hift_1024_conv.onnx"),
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

    update_scale_constant(model)
    onnx.save(model, UPDATE_SCALE)

    model = onnx.load(UPDATE_SCALE)
    graph = model.graph

    replace_pad_reflect(graph, "/Pad", L=491520, pad=8)
    replace_pad_reflect(graph, "/reflection_pad/Pad", L=122880, pad=1)

    model = onnx.shape_inference.infer_shapes(model)
    simplify_model(model, REFLECT_REPLACED)

    model = onnx.load(UPDATE_SCALE)

    for node in model.graph.node:
        if node.op_type == "Pad":
            for a in node.attribute:
                if a.name == "mode":
                    a.s = b"constant"

    simplify_model(model, REFLECT_CONSTANT)

    model = onnx.load(REFLECT_CONSTANT)

    replace_resize_with_sizes(model)

    simplify_model(model, FINAL_ONNX)

    transform_scatter_add(FINAL_ONNX, FINAL_ONNX_1)

    convert_hmonnx(FINAL_ONNX_1)


if __name__ == "__main__":
    main()