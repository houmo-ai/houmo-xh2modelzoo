import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper


def replace_reshape_minus_one_by_infer(onnx_path, output_path, input_shape):
    """
    实际运行onnx模型记录每个节点的输出shape，然后遍历reshape节点替换-1
    """

    model = onnx.load(onnx_path)
    graph = model.graph

    # 第一步：实际运行onnx模型，记录每个节点的输出shape
    shape_map = {}
    input_name = graph.input[0].name
    shape_map[input_name] = tuple(input_shape)
    print(f"初始输入shape: {input_shape}")

    # 创建onnxruntime session
    session = ort.InferenceSession(onnx_path)

    # 准备输入数据
    input_data = np.random.randn(*input_shape).astype(np.float32)
    input_feed = {input_name: input_data}

    # 使用onnx的shape inference获取所有节点的输出shape
    try:
        from onnx import shape_inference

        inferred_model = shape_inference.infer_shapes(model)

        # 从inferred model中提取所有节点的输出shape
        for node in inferred_model.graph.node:
            for output_name in node.output:
                # 查找对应的value_info
                for value_info in inferred_model.graph.value_info:
                    if value_info.name == output_name:
                        shape = tuple(dim.dim_value for dim in value_info.type.tensor_type.shape.dim)
                        shape_map[output_name] = shape
                        print(f"节点 {node.op_type} 输出 {output_name}: {shape}")
                        break

        # 处理输出节点
        for output in inferred_model.graph.output:
            shape = tuple(dim.dim_value for dim in output.type.tensor_type.shape.dim)
            shape_map[output.name] = shape
            print(f"最终输出 {output.name}: {shape}")

    except Exception as e:
        print(f"Shape inference失败: {e}")
        print("尝试使用onnxruntime运行模型...")

        try:
            # 运行模型并获取所有中间节点的输出
            outputs = session.run(None, input_feed)

            # 获取所有输出节点的shape
            for i, output_name in enumerate(session.get_outputs()):
                shape_map[output_name.name] = tuple(outputs[i].shape)
                print(f"输出节点 {output_name.name}: {shape_map[output_name.name]}")

        except Exception as e2:
            print(f"运行模型也失败: {e2}")
            print("使用最基本的shape传递方法...")

            # 最后的备选方案：简单传递shape
            for node in graph.node:
                if len(node.input) > 0 and node.input[0] in shape_map:
                    shape_map[node.output[0]] = shape_map[node.input[0]]

    print(f"shape_map包含 {len(shape_map)} 个节点的shape信息")

    # 第二步：遍历所有reshape节点，根据实际shape替换-1
    replaced_count = 0

    for node in graph.node:
        if node.op_type == "Reshape":
            data_input_name = node.input[0]  # reshape的数据输入
            shape_input_name = node.input[1]  # reshape的shape输入
            output_name = node.output[0]

            print(f"\n处理Reshape节点: {output_name}")
            print(f"  数据输入: {data_input_name}")
            print(f"  shape输入: {shape_input_name}")

            # 从字典中获取输入节点的实际shape信息
            if data_input_name in shape_map:
                input_shape = shape_map[data_input_name]
                input_size = np.prod(input_shape)
                print(f"  输入shape: {input_shape}, 总元素数: {input_size}")

                # 获取reshape的shape参数
                shape_array = None

                # 查找initializer
                for init in graph.initializer:
                    if init.name == shape_input_name:
                        shape_array = numpy_helper.to_array(init).copy()
                        print(f"  从initializer获取shape: {shape_array}")
                        break

                # 如果没找到initializer，查找Constant节点
                if shape_array is None:
                    for const_node in graph.node:
                        if const_node.op_type == "Constant" and const_node.output[0] == shape_input_name:
                            for attr in const_node.attribute:
                                if attr.name == "value":
                                    shape_array = numpy_helper.to_array(attr.t).copy()
                                    print(f"  从Constant节点获取shape: {shape_array}")
                                    break
                            break

                # 如果有-1，计算实际值并替换
                if shape_array is not None and -1 in shape_array:
                    # 计算-1的实际值
                    known_dims = np.where(shape_array != -1, shape_array, 1)
                    minus_one_indices = np.where(shape_array == -1)[0]

                    for idx in minus_one_indices:
                        known_prod = np.prod(known_dims)
                        if known_prod > 0:
                            shape_array[idx] = int(input_size // known_prod)
                        else:
                            shape_array[idx] = input_size

                    print(f"  计算得到的实际shape: {shape_array}")

                    # 替换initializer中的-1
                    for init in graph.initializer:
                        if init.name == shape_input_name:
                            init.CopyFrom(numpy_helper.from_array(shape_array.astype(np.int64), name=init.name))
                            replaced_count += 1
                            print(f"  替换initializer中的-1: {shape_array}")
                            break

                    # 替换Constant节点中的-1
                    for const_node in graph.node:
                        if const_node.op_type == "Constant" and const_node.output[0] == shape_input_name:
                            for attr in const_node.attribute:
                                if attr.name == "value":
                                    new_const_init = numpy_helper.from_array(shape_array.astype(np.int64))
                                    attr.t.CopyFrom(new_const_init)
                                    replaced_count += 1
                                    print(f"  替换Constant节点中的-1: {shape_array}")
                                    break
                            break
                else:
                    print(f"  没有找到-1或无法获取shape参数")

            else:
                print(f"  警告: 找不到输入 {data_input_name} 的shape信息")

    onnx.save(model, output_path)
    print(f"已保存新模型到: {output_path}")
    print(f"总共替换了 {replaced_count} 个Reshape节点的-1")
