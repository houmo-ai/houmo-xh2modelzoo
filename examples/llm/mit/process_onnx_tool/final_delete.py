import onnx
from onnx import helper

def remove_xlength_branch(model_path, output_path):
    model = onnx.load(model_path)
    graph = model.graph
    nodes = list(graph.node)

    # 1. 定位x_length输入节点
    xlength_input = None
    for inp in graph.input:
        if inp.name == "x_length":
            xlength_input = inp
            break
    if not xlength_input:
        raise ValueError("未找到x_length输入节点")
    xlength_name = xlength_input.name

    # 2. 递归找到所有依赖x_length的节点（下游节点）
    def find_downstream_nodes(start_tensor):
        downstream = []
        # 找到所有以start_tensor为输入的节点
        next_nodes = [n for n in nodes if start_tensor in n.input]
        for node in next_nodes:
            if node not in downstream:
                downstream.append(node)
                # 递归找该节点输出的下游节点
                for out_tensor in node.output:
                    downstream.extend(find_downstream_nodes(out_tensor))
        return downstream

    # 获取x_length的所有下游节点
    xlength_downstream = find_downstream_nodes(xlength_name)

    # 3. 删除x_length输入和所有下游节点
    # 移除输入
    if xlength_input in graph.input:
        graph.input.remove(xlength_input)
    # 移除下游节点
    nodes_to_keep = [n for n in nodes if n not in xlength_downstream]
    graph.ClearField("node")
    graph.node.extend(nodes_to_keep)

    # 4. 保存模型
    onnx.save(model, output_path)
    print(f"已删除x_length及{len(xlength_downstream)}个下游节点")


# 调用
remove_xlength_branch(
    model_path="/data01/home/xuchen/xh2/xh2_model_zoo/examples/llm/mit/fp/model_all_del_mul_reshape.sim.onnx",
    output_path="/data01/home/xuchen/xh2/xh2_model_zoo/examples/llm/mit/fp/model_all_del_final.sim.onnx"
)