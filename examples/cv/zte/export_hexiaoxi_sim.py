import onnx
from google.protobuf import text_format
from copy import deepcopy

model = onnx.load("weights/zte/model_hexiaoxi_sim.onnx")
model = deepcopy(model)  # 防止污染原模型


def clear_raw_data_from_graph(graph):
    for tensor in graph.initializer:
        if tensor.raw_data:
            tensor.raw_data = b""  # 清空
    for sparse in getattr(graph, "sparse_initializer", []):
        if sparse.values.raw_data:
            sparse.values.raw_data = b""  # 清空
    for node in graph.node:
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.GRAPH:
                clear_raw_data_from_graph(attr.g)
            elif attr.type == onnx.AttributeProto.GRAPHS:
                for subgraph in attr.graphs:
                    clear_raw_data_from_graph(subgraph)


clear_raw_data_from_graph(model.graph)
        
with open("weights/zte/model_full_sim.txt", "w") as f:
    f.write(text_format.MessageToString(model))
