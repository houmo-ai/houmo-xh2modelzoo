      
import onnx
from onnx import helper, TensorProto
import numpy as np
import argparse

def convert_onnx_fp16_to_fp32(input_onnx_path, output_onnx_path):
    """
    将 FP16/FP64 ONNX 模型转换为 FP32 格式（删除Cast+全量修正张量类型）
    :param input_onnx_path: 输入模型路径
    :param output_onnx_path: 输出FP32模型路径
    """
    # 1. 加载模型（保留原始结构）
    model = onnx.load(input_onnx_path, load_external_data=False)
    graph = model.graph

    # ===================== 步骤1：转换初始器（权重）为FP32 =====================
    for initializer in graph.initializer:
        # 处理FP16权重
        if initializer.data_type == TensorProto.FLOAT16:
            fp16_array = np.frombuffer(initializer.raw_data, dtype=np.float16)
            fp16_array = fp16_array.reshape(tuple(initializer.dims))
            fp32_array = fp16_array.astype(np.float32)
            initializer.data_type = TensorProto.FLOAT
            initializer.raw_data = fp32_array.tobytes()
        # 处理FP64权重
        elif initializer.data_type == TensorProto.DOUBLE:
            fp64_array = np.frombuffer(initializer.raw_data, dtype=np.float64)
            fp64_array = fp64_array.reshape(tuple(initializer.dims))
            fp32_array = fp64_array.astype(np.float32)
            initializer.data_type = TensorProto.FLOAT
            initializer.raw_data = fp32_array.tobytes()
        
        # 清理冗余属性
        if hasattr(initializer, 'float_data'):
            del initializer.float_data[:]
        if hasattr(initializer, 'int32_data'):
            del initializer.int32_data[:]

    # ===================== 步骤2：删除所有Cast节点 + 修复数据流 =====================
    cast_nodes_info = []
    for idx, node in enumerate(graph.node):
        if node.op_type == "Cast":
            cast_nodes_info.append({
                "idx": idx,
                "input": node.input[0],
                "output": node.output[0]
            })

    # 反向删除Cast节点（避免索引错乱）
    cast_nodes_info.sort(key=lambda x: x["idx"], reverse=True)
    for info in cast_nodes_info:
        del graph.node[info["idx"]]
        print(f"🗑️ 删除 Cast 节点：{info['input']} → {info['output']}")

    # 修复节点输入依赖
    cast_output_to_input = {info["output"]: info["input"] for info in cast_nodes_info}
    for node in graph.node:
        new_inputs = [cast_output_to_input.get(inp, inp) for inp in node.input]
        node.input[:] = new_inputs

    # 修复模型输出依赖
    for output in graph.output:
        if output.name in cast_output_to_input:
            output.name = cast_output_to_input[output.name]

    # ===================== 步骤3：全量修正所有张量类型为FP32 =====================
    # 3.1 修正图的输入/输出张量类型
    all_io_tensors = list(graph.input) + list(graph.output)
    for tensor_info in all_io_tensors:
        tensor_type = tensor_info.type.tensor_type
        if tensor_type.elem_type in [TensorProto.FLOAT16, TensorProto.DOUBLE]:
            tensor_type.elem_type = TensorProto.FLOAT
            print(f"🔧 修正IO张量 {tensor_info.name} 类型：{tensor_type.elem_type} → FP32")

    # 3.2 修正ValueInfo（节点输入/输出的类型标注）
    # ValueInfo是模型中张量的类型/形状定义，必须和实际数据类型一致
    for value_info in graph.value_info:
        tensor_type = value_info.type.tensor_type
        if tensor_type.elem_type in [TensorProto.FLOAT16, TensorProto.DOUBLE]:
            tensor_type.elem_type = TensorProto.FLOAT
            print(f"🔧 修正ValueInfo {value_info.name} 类型：{tensor_type.elem_type} → FP32")

    # 3.3 修正节点输出的类型标注（部分模型的节点输出有显式类型）
    for node in graph.node:
        # 遍历节点的输出张量，匹配ValueInfo并修正
        for output_name in node.output:
            # 查找该输出对应的ValueInfo
            for value_info in graph.value_info:
                if value_info.name == output_name:
                    tensor_type = value_info.type.tensor_type
                    if tensor_type.elem_type in [TensorProto.FLOAT16, TensorProto.DOUBLE]:
                        tensor_type.elem_type = TensorProto.FLOAT
                        break

    # ===================== 步骤4：验证并保存模型 =====================
    try:
        onnx.checker.check_model(model)
        onnx.save(model, output_onnx_path)
        print(f"\n✅ 转换完成！FP32模型已保存至：{output_onnx_path}")
    except onnx.checker.ValidationError as e:
        print(f"\n❌ 模型验证失败：{e}")
        return

    # 验证模型加载
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(output_onnx_path)
        # 打印所有输入节点的类型（确认无FP16/FP64）
        for inp in sess.get_inputs():
            print(f"✅ 输入节点 {inp.name}：类型={inp.type}，形状={inp.shape}")
        for out in sess.get_outputs():
            print(f"✅ 输出节点 {out.name}：类型={out.type}，形状={out.shape}")
    except Exception as e:
        print(f"⚠️ 推理验证警告：{e}")

# ===================== 运行转换 =====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将 FP16/FP64 ONNX 模型转换为 FP32 格式")
    parser.add_argument("input_model", type=str, default="/data01/home/xuchen/xh2/xh2_model_zoo/data/models/L1csi-zgb_b8.sim.onnx", help="输入FP16/FP64模型路径")
    parser.add_argument("output_model", type=str, default="/data01/home/xuchen/xh2/xh2_model_zoo/data/models/L1csi-zgb_b8_fp32.sim.onnx", help="输出FP32模型路径")
    
    args = parser.parse_args()
    convert_onnx_fp16_to_fp32(args.input_model, args.output_model)
