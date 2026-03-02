
import onnx
import torch
import onnxsim
import onnxruntime as ort
import numpy as np
import os.path as osp
from onnxsim import simplify
from onnx import numpy_helper
from onnx import helper, TensorProto

model_path = "/data01/home/she.gao/xh2modelzoo/examples/audio/Cosyvoice3/onnx/hift.onnx"
model_path_simplify = "/data01/home/she.gao/xh2modelzoo/examples/audio/Cosyvoice3/onnx/hift_simplify.onnx"
update_scale_simplify = "/data01/home/she.gao/xh2modelzoo/examples/audio/Cosyvoice3/onnx/hift_simplify_update_scale.onnx"
replace_path = "/data01/home/she.gao/xh2modelzoo/examples/audio/Cosyvoice3/onnx/hift_simplify_reflect_replaced.onnx"
replace_reflect_path = "/data01/home/she.gao/xh2modelzoo/examples/audio/Cosyvoice3/onnx/hift_simplify_reflect_replaced_constant.onnx"
output_path = "/data01/home/she.gao/xh2modelzoo/examples/audio/Cosyvoice3/hmonnx"

fix shape
fixed_d = {
   'batch_size': 1,
   'seq_len' : 1024,
}

# 加载模型
model = onnx.load(model_path)
providers = ['CPUExecutionProvider']  # 替换为 ['CUDAExecutionProvider'] 启用 GPU
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
   # 根据fixed_dims替换符号维度
   new_dims = []
   for dim in dims:
      if dim in fixed_d:
            new_dims.append(fixed_d[dim])  # 替换为固定值
      else:
            # 如果是数字维度（如80），保持不变
            new_dims.append(dim if isinstance(dim, int) else 0)  # 0表示动态，这里应避免
   
   # 更新输入节点的shape
   input_node.type.tensor_type.shape.dim.clear()
   for dim_val in new_dims:
      input_node.type.tensor_type.shape.dim.add(dim_value=dim_val)

slimmed_model, _ = onnxsim.simplify(model)
onnx.save(slimmed_model, model_path_simplify)

model = onnx.load(model_path_simplify)

# 遍历 initializer，找到 Constant_23_output_0
for init in model.graph.initializer:
    if init.name == "/m_source/l_sin_gen/Constant_23_output_0":
        data = numpy_helper.to_array(init).copy()  # 关键：copy 避免只读
        data[2] = np.float32(0.002084)
        new_init = numpy_helper.from_array(data, init.name)
        init.CopyFrom(new_init)
        break

# 保存新模型
onnx.save(model, update_scale_simplify)

model = onnx.load(update_scale_simplify)
graph = model.graph

def replace_pad_reflect(graph, pad_name, L, pad):
    pad_node = None
    pad_index = None

    for i, node in enumerate(graph.node):
        if node.name == pad_name:
            pad_node = node
            pad_index = i
            break

    assert pad_node is not None, f"未找到 {pad_name}"

    input_name = pad_node.input[0]
    output_name = pad_node.output[0]

    # reflect 左
    graph.initializer.extend([
        helper.make_tensor(f"{pad_name}_l_starts", TensorProto.INT64, [3], [0, 0, pad]),
        helper.make_tensor(f"{pad_name}_l_ends",   TensorProto.INT64, [3], [1, graph.input[0].type.tensor_type.shape.dim[1].dim_value, 0]),
        helper.make_tensor(f"{pad_name}_l_axes",   TensorProto.INT64, [3], [0, 1, 2]),
        helper.make_tensor(f"{pad_name}_l_steps",  TensorProto.INT64, [3], [1, 1, -1]),
    ])

    left_node = helper.make_node(
        "Slice",
        [input_name,
         f"{pad_name}_l_starts",
         f"{pad_name}_l_ends",
         f"{pad_name}_l_axes",
         f"{pad_name}_l_steps"],
        [f"{pad_name}_left"]
    )

    nodes_to_insert = [left_node]

    # 如果是双侧 padding
    if pad_name == "/Pad":
        graph.initializer.extend([
            helper.make_tensor(f"{pad_name}_r_starts", TensorProto.INT64, [3], [0, 0, L - 2]),
            helper.make_tensor(f"{pad_name}_r_ends",   TensorProto.INT64, [3], [1, 1, L - pad - 2]),
            helper.make_tensor(f"{pad_name}_r_axes",   TensorProto.INT64, [3], [0, 1, 2]),
            helper.make_tensor(f"{pad_name}_r_steps",  TensorProto.INT64, [3], [1, 1, -1]),
        ])

        right_node = helper.make_node(
            "Slice",
            [input_name,
             f"{pad_name}_r_starts",
             f"{pad_name}_r_ends",
             f"{pad_name}_r_axes",
             f"{pad_name}_r_steps"],
            [f"{pad_name}_right"]
        )

        concat_node = helper.make_node(
            "Concat",
            [f"{pad_name}_left", input_name, f"{pad_name}_right"],
            [output_name],
            axis=2
        )

        nodes_to_insert.extend([right_node, concat_node])
    else:
        # 单侧
        concat_node = helper.make_node(
            "Concat",
            [f"{pad_name}_left", input_name],
            [output_name],
            axis=2
        )
        nodes_to_insert.append(concat_node)

    graph.node.remove(pad_node)

    for offset, n in enumerate(nodes_to_insert):
        graph.node.insert(pad_index + offset, n)

replace_pad_reflect(graph, "/Pad", L=491520, pad=8)
replace_pad_reflect(graph, "/reflection_pad/Pad", L=122880, pad=1)
model = onnx.shape_inference.infer_shapes(model)

model_simplified, check = simplify(model)
assert check, "onnx simplify 校验失败"
onnx.save(model_simplified, replace_path)

model = onnx.load(update_scale_simplify)
for node in model.graph.node:
    if node.op_type == "Pad":
        for attr in node.attribute:
            if attr.name == "mode":
                attr.s = b"constant"
                break

model_simplified, check = simplify(model)
assert check, "onnx simplify 校验失败"
onnx.save(model_simplified, replace_reflect_path)

#convert hmonnx
from xhquant.api import convert_fx_model_to_hmonnx, convert_onnx_to_hmonnx, QuantScheme, create_quant_config, DeviceType
input = torch.randn(1, 80, 1024)
quant_type = "w8a16_sefp"
quant_scheme = QuantScheme(target_device=DeviceType.XH2a, quant_type=quant_type)
quant_config = create_quant_config(quant_scheme)
convert_onnx_to_hmonnx(replace_reflect_path, (input,), out_hmonnx_file=osp.join(output_path,"hift_1024.onnx"), device_type="XH2A", quant_config=quant_config)