# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0

"""Patch ONNX ScatterND index tensors to int64 for ONNX Runtime compatibility."""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnx import TensorProto, helper, shape_inference


def patch_scatternd_indices_to_int64(input_path: Path, output_path: Path | None = None) -> tuple[Path, int]:
    """Insert Cast(to=int64) before ScatterND index inputs that infer as int32.

    Some PyTorch-exported Cosmos sequence-packing graphs contain constant int32
    ScatterND indices. HMONNX accepts those graphs, but ONNX Runtime rejects them
    because ScatterND indices are specified as int64. This patch leaves data and
    update tensors unchanged and only casts the indices input.
    """
    input_path = input_path.resolve()
    if output_path is None:
        output_path = input_path.with_name(input_path.stem + ".scatter_i64.onnx")
    else:
        output_path = output_path.resolve()

    model = onnx.load(input_path, load_external_data=False)
    inferred = shape_inference.infer_shapes(onnx.load(input_path, load_external_data=False))
    value_types: dict[str, int] = {}
    graph_collections = (
        inferred.graph.input,
        inferred.graph.value_info,
        inferred.graph.output,
        inferred.graph.initializer,
    )
    for collection in graph_collections:
        for value in collection:
            if hasattr(value, "type"):
                value_types[value.name] = value.type.tensor_type.elem_type
            else:
                value_types[value.name] = value.data_type

    new_nodes = []
    patched = 0
    for node in model.graph.node:
        if node.op_type == "ScatterND" and value_types.get(node.input[1]) == TensorProto.INT32:
            safe_node_name = node.name.replace("/", "_") or f"scatternd_{patched}"
            cast_name = f"{node.input[1]}_cast_i64_for_{safe_node_name}"
            new_nodes.append(
                helper.make_node("Cast", [node.input[1]], [cast_name], name=cast_name, to=TensorProto.INT64)
            )
            node.input[1] = cast_name
            patched += 1
        new_nodes.append(node)

    model.graph.ClearField("node")
    model.graph.node.extend(new_nodes)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, output_path)
    onnx.checker.check_model(output_path)
    return output_path, patched


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path, patched = patch_scatternd_indices_to_int64(args.input, args.output)
    print({"output": str(output_path), "scatternd_indices_casts": patched})


if __name__ == "__main__":
    main()
