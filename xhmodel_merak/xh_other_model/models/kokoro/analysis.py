from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import onnx
from onnx import AttributeProto, TensorProto

from .assets import sha256


def analyze_onnx(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    model = onnx.load(source, load_external_data=False)
    nodes = _walk_graph(model.graph)
    op_counts = Counter(node.op_type for _, node in nodes)
    lstms = []
    for scope, node in nodes:
        if node.op_type != "LSTM":
            continue
        attributes = {attribute.name: onnx.helper.get_attribute_value(attribute) for attribute in node.attribute}
        for name, value in list(attributes.items()):
            if isinstance(value, bytes):
                attributes[name] = value.decode("utf-8")
        lstms.append(
            {
                "scope": scope,
                "name": node.name,
                "inputs": list(node.input),
                "outputs": list(node.output),
                "direction": attributes.get("direction", "forward"),
                "hidden_size": attributes.get("hidden_size"),
                "has_sequence_lens": len(node.input) >= 5 and bool(node.input[4]),
            }
        )
    return {
        "path": str(source),
        "sha256": sha256(source),
        "file_bytes": source.stat().st_size,
        "ir_version": model.ir_version,
        "opsets": [{"domain": item.domain or "ai.onnx", "version": item.version} for item in model.opset_import],
        "producer": {
            "name": model.producer_name,
            "version": model.producer_version,
        },
        "inputs": [_value_contract(value) for value in model.graph.input],
        "outputs": [_value_contract(value) for value in model.graph.output],
        "main_nodes": len(model.graph.node),
        "recursive_nodes": len(nodes),
        "initializers": len(model.graph.initializer),
        "initializer_raw_bytes": sum(len(value.raw_data) for value in model.graph.initializer),
        "op_counts": dict(sorted(op_counts.items())),
        "lstm_nodes": lstms,
        "unsupported_xh2_frontend_ops": {
            name: op_counts[name]
            for name in (
                "SplitToSequence",
                "ConcatFromSequence",
                "Loop",
                "If",
                "SequenceEmpty",
                "SequenceAt",
                "SequenceInsert",
                "ReduceProd",
            )
            if op_counts[name]
        },
    }


def _walk_graph(graph: onnx.GraphProto, scope: str = "main") -> list[tuple[str, onnx.NodeProto]]:
    result: list[tuple[str, onnx.NodeProto]] = []
    for node in graph.node:
        result.append((scope, node))
        for attribute in node.attribute:
            if attribute.type == AttributeProto.GRAPH:
                result.extend(_walk_graph(attribute.g, f"{scope}/{node.op_type}:{attribute.name}"))
            elif attribute.type == AttributeProto.GRAPHS:
                for index, child in enumerate(attribute.graphs):
                    result.extend(
                        _walk_graph(
                            child,
                            f"{scope}/{node.op_type}:{attribute.name}[{index}]",
                        )
                    )
    return result


def _value_contract(value: onnx.ValueInfoProto) -> dict[str, Any]:
    tensor = value.type.tensor_type
    shape: list[int | str | None] = []
    for dimension in tensor.shape.dim:
        if dimension.HasField("dim_value"):
            shape.append(int(dimension.dim_value))
        elif dimension.HasField("dim_param"):
            shape.append(dimension.dim_param)
        else:
            shape.append(None)
    return {
        "name": value.name,
        "dtype": TensorProto.DataType.Name(tensor.elem_type),
        "shape": shape,
    }


__all__ = ["analyze_onnx"]
