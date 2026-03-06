import onnx
from google.protobuf import text_format
from copy import deepcopy

model = onnx.load("weights/zte/hift.onnx")
model = deepcopy(model)

def strip_tensor_data(t):
    # 清掉所有可能携带数据的字段，而不是只清 raw_data
    t.raw_data = b""
    del t.float_data[:]
    del t.int32_data[:]
    del t.int64_data[:]
    del t.double_data[:]
    del t.uint64_data[:]
    del t.string_data[:]
    del t.external_data[:]   # 如果是 external_data，也会在 txt 里很烦
    t.data_location = 0      # DEFAULT

def strip_weights_from_graph(g, drop_constant_nodes=True):
    # 1) 删 initializer（权重最主要的载体）
    del g.initializer[:]  # 直接移除条目（比“清空字段”小很多）
    if hasattr(g, "sparse_initializer"):
        del g.sparse_initializer[:]

    # 2) 处理 Constant 节点里携带的 Tensor（很多导出会用 Constant 存权重）
    new_nodes = []
    for node in g.node:
        if drop_constant_nodes and node.op_type == "Constant":
            # 直接丢掉 Constant 节点（最省）
            continue

        # 否则就把 Constant 的 attribute 里的 tensor 数据清空
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.TENSOR:
                strip_tensor_data(attr.t)
            elif attr.type == onnx.AttributeProto.TENSORS:
                for tt in attr.tensors:
                    strip_tensor_data(tt)
            elif attr.type == onnx.AttributeProto.SPARSE_TENSOR:
                if attr.sparse_tensor.values is not None:
                    strip_tensor_data(attr.sparse_tensor.values)

            # 递归子图（If/Loop/Scan 等）
            elif attr.type == onnx.AttributeProto.GRAPH:
                strip_weights_from_graph(attr.g, drop_constant_nodes=drop_constant_nodes)
            elif attr.type == onnx.AttributeProto.GRAPHS:
                for sg in attr.graphs:
                    strip_weights_from_graph(sg, drop_constant_nodes=drop_constant_nodes)

        new_nodes.append(node)

    # 覆盖 node 列表
    del g.node[:]
    g.node.extend(new_nodes)

strip_weights_from_graph(model.graph, drop_constant_nodes=True)

with open("weights/zte/model_stripped_hift.txt", "w") as f:
    f.write(text_format.MessageToString(model))
    