import onnx


def remove_equal_and_where(model_path, output_path):
    # 1. 加载模型
    model = onnx.load(model_path)
    graph = model.graph
    nodes = graph.node
    initializers = {init.name: init for init in graph.initializer}

    # 2. 定位目标Equal节点
    equal_node = None
    for node in nodes:
        if node.name == "/encoder/encoders0.0/self_attn/Equal":
            equal_node = node
            break
    if not equal_node:
        raise ValueError("未找到/encoder/encoders0.0/self_attn/Equal节点")

    # 3. 找到Equal的所有Where输出节点
    # (Equal的输出是Where的第一个输入)
    equal_output = equal_node.output[0]
    where_nodes = []
    for node in nodes:
        if node.op_type == "Where" and equal_output in node.input:
            where_nodes.append(node)
    if not where_nodes:
        print("未找到Equal对应的Where节点")
        onnx.save(model, output_path)
        return

    # 4. 记录Where的前后连接关系
    # - Where的输入：[0]=Equal输出, [1]=X(如-Infinity/0), [2]=MatMul输出
    # - Where的输出：连接到后续节点(如Softmax)
    where_connections = []
    for where_node in where_nodes:
        # 记录：Where的输入2（MatMul输出） → Where的输出（后续节点的输入）
        where_input_matmul = where_node.input[2]  # Where的第3个输入（MatMul的输出）
        where_output = where_node.output[0]  # Where的输出（后续节点的输入）
        where_connections.append((where_input_matmul, where_output))

    # 5. 删除Equal和所有Where节点
    nodes_to_remove = {equal_node.name}
    nodes_to_remove.update([node.name for node in where_nodes])
    new_nodes = [node for node in nodes if node.name not in nodes_to_remove]

    # 6. 重新连接前后节点：用MatMul的输出替换Where的输出
    for matmul_output, where_output in where_connections:
        for node in new_nodes:
            # 遍历所有节点的输入，将引用Where输出的地方替换为MatMul输出
            for idx, inp in enumerate(node.input):
                if inp == where_output:
                    node.input[idx] = matmul_output

    # 7. 更新图节点
    graph.ClearField("node")
    graph.node.extend(new_nodes)

    # 8. 保存修改后的模型
    onnx.save(
        model,
        output_path,
        save_as_external_data=true,
        all_tensors_to_one_file=True,
        size_threshold=1,
    )
    print(f"已删除Equal节点和{len(where_nodes)}个Where节点，并重新连接链路")


# 调用（替换为你的模型路径）
remove_equal_and_where(
    model_path="/data01/home/xuchen/xh2/xh2_model_zoo/examples/llm/mit/fp/model_all.sim.onnx",
    output_path="/data01/home/xuchen/xh2/xh2_model_zoo/examples/llm/mit/fp/model_all_del_sim.sim.onnx",
)
