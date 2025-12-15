import onnx
from onnx import helper
from tqdm import tqdm

def clean_unsqueeze_branch(model_path, output_path):
    model = onnx.load(model_path)
    graph = model.graph
    nodes = list(graph.node)  # 转列表便于遍历
    node_name_map = {node.name: node for node in nodes}

    # --------------------------
    # 步骤1：定位所有待删除的节点
    # --------------------------
    # 1.1 定位目标Unsqueeze_1
    unsqueeze_node = node_name_map.get("/encoder/Unsqueeze_1")
    if not unsqueeze_node:
        raise ValueError("未找到/encoder/Unsqueeze_1节点")
    unsqueeze_output = unsqueeze_node.output[0]

    # 1.2 找到Unsqueeze_1的输出节点（Reshape）
    reshape_nodes = [n for n in nodes if unsqueeze_output in n.input and n.op_type == "Reshape"]
    if not reshape_nodes:
        print("未找到Unsqueeze_1对应的Reshape节点")
        onnx.save(model, output_path)
        return

    # # 1.3 找到每个Reshape的输出节点（两个Mul）
    # mul_nodes = []
    # for reshape_node in reshape_nodes:
    #     reshape_output = reshape_node.output[0]
    #     mul_nodes.extend([n for n in nodes if reshape_output in n.input and n.op_type == "Mul"])



    # 1.2 找到Reshape的第二个输入（Concat的输出），并定位Concat节点
    concat_node = None
    for reshape in tqdm(reshape_nodes):
        # Reshape的输入格式：[数据输入, 形状输入(Concat输出)]
        if len(reshape.input) >= 2:
            shape_input = reshape.input[1]  # 第二个输入是形状张量（来自Concat）

            # 找到输出为shape_input的Concat节点
            concat_candidate = next((n for n in nodes if n.op_type == "Concat" and shape_input in n.output), None)
            if concat_candidate:
                concat_node = concat_candidate
        
            if concat_node:    
                concat_output = concat_node.output[0]

                # 1.3 追溯Concat的上游分支（Unsqueeze → Gather → Shape → Split）
                # Concat的输入是两个Unsqueeze的输出
                unsqueeze_nodes = [n for n in nodes if any(inp in n.output for inp in concat_node.input)]
                # if len(unsqueeze_nodes) != 2:
                #     raise ValueError("Concat应依赖2个Unsqueeze节点")

                # Unsqueeze的输入是Gather的输出
                gather_nodes = [n for n in nodes if any(inp in n.output for inp in [u.input[0] for u in unsqueeze_nodes])]
                # if len(gather_nodes) != 2:
                #     raise ValueError("每个Unsqueeze应依赖1个Gather节点")

                # Gather的输入是Shape的输出
                shape_node = next((n for n in nodes if any(inp in n.output for inp in [g.input[0] for g in gather_nodes])), None)
                if not shape_node:
                    raise ValueError("Gather应依赖Shape节点")

                # Shape的输入是Split的某个输出
                split_node = next((n for n in nodes if shape_node.input[0] in n.output), None)
                if not split_node:
                    raise ValueError("Shape应依赖Split节点")
                split_branch_output = shape_node.input[0]  # Split用于该分支的输出张量


                # --------------------------
                # 步骤2：定位所有待删除节点
                # --------------------------
                # 分支节点：Split分支（Shape→Gather→Unsqueeze→Concat）
                branch_nodes = [shape_node] + gather_nodes + unsqueeze_nodes + [concat_node]
            else:
                branch_nodes = []
        else:
            branch_nodes = []

        # 关联节点：Reshape及其输出节点Mul
        mul_nodes = []
        reshape_output = reshape.output[0]
        mul_nodes.extend([n for n in nodes if reshape_output in n.input and n.op_type == "Mul"])
        # 汇总待删除节点
        nodes_to_remove = [reshape]
        nodes_to_remove.extend(mul_nodes)
        nodes_to_remove.extend(branch_nodes)
        nodes_to_remove_name = {n.name for n in nodes_to_remove}


        # --------------------------
        # 步骤3：记录连接关系（重新布线）
        # --------------------------
        connections = []
        # 2.1 记录Reshape的输入 → Mul的另一输入（删除Reshape/Mul后，用Reshape的输入连Mul的后续节点）
        for mul_node in mul_nodes:
            mul_other_input = [inp for inp in mul_node.input if inp != reshape.output[0]][0]
            # 记录：Reshape输入 → Mul输出（后续节点的输入）
            connections.append((mul_other_input, mul_node.output[0]))

        # --------------------------
        # 步骤4：清理节点+重新连接链路
        # --------------------------
        # 4.1 过滤保留的节点
        print(nodes_to_remove_name)
        new_nodes = [node for node in nodes if node.name not in nodes_to_remove_name]

        # 4.2 替换输入引用（用保留的张量替换被删除节点的输出）
        for (keep_input, old_output) in connections:
            for node in new_nodes:
                for idx, inp in enumerate(node.input):
                    if inp == old_output:
                        node.input[idx] = keep_input

        nodes = new_nodes
    
    new_nodes = nodes

    # --------------------------
    # 步骤5：保存模型
    # --------------------------
    graph.ClearField("node")
    graph.node.extend(new_nodes)
    onnx.save(model, output_path)
    print(f"已删除节点数：{len(nodes_to_remove)}，并完成链路重连")


# 调用（替换为你的模型路径）
clean_unsqueeze_branch(
    model_path="/data01/home/xuchen/xh2/xh2_model_zoo/examples/llm/mit/fp/model_all_del_mul.sim.onnx",
    output_path="/data01/home/xuchen/xh2/xh2_model_zoo/examples/llm/mit/fp/model_all_del_mul_reshape.sim.onnx"
)