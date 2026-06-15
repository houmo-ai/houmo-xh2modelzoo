# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0

"""Patch HMONNX RMSNorm decomposition to preserve fp32 variance computation."""

from __future__ import annotations

import sys
from pathlib import Path

_VLA_JEPA_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _VLA_JEPA_ROOT.parents[2]
for _path in (_VLA_JEPA_ROOT, _REPO_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import argparse
from pathlib import Path

import onnx
from onnx import TensorProto, helper, numpy_helper

XH2A_DOMAIN = "ai.houmo.xh2a"


def get_shape_by_name(model: onnx.ModelProto, name: str) -> list[int | str]:
    for value in list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output):
        if value.name != name:
            continue
        shape = []
        for dim in value.type.tensor_type.shape.dim:
            shape.append(dim.dim_value if dim.dim_value else dim.dim_param)
        return shape
    return []


def patch_rmsnorm_casts(model: onnx.ModelProto) -> int:
    nodes = list(model.graph.node)
    patched = 0
    new_nodes = []
    skip_names: set[str] = set()

    for idx, node in enumerate(nodes):
        if node.name in skip_names:
            continue
        if node.op_type != "Mul" or len(node.input) != 2 or node.input[0] != node.input[1]:
            new_nodes.append(node)
            continue
        if idx + 6 >= len(nodes):
            new_nodes.append(node)
            continue

        reduce_mean = nodes[idx + 1]
        add = nodes[idx + 2]
        sqrt = nodes[idx + 3]
        reciprocal = nodes[idx + 4]
        mul_norm = nodes[idx + 5]
        mul_weight = nodes[idx + 6]
        if not (
            reduce_mean.op_type == "ReduceMean"
            and add.op_type == "Add"
            and sqrt.op_type == "Sqrt"
            and reciprocal.op_type == "Reciprocal"
            and mul_norm.op_type == "Mul"
            and mul_weight.op_type == "Mul"
            and reduce_mean.input[0] == node.output[0]
            and add.input[0] == reduce_mean.output[0]
            and sqrt.input[0] == add.output[0]
            and reciprocal.input[0] == sqrt.output[0]
            and reciprocal.output[0] in mul_norm.input
            and mul_norm.output[0] in mul_weight.input
        ):
            new_nodes.append(node)
            continue

        hidden_name = node.input[0]
        hidden_fp32 = f"{hidden_name}_rmsnorm_fp32_{patched}"
        norm_fp16 = f"{mul_norm.output[0]}_rmsnorm_fp16_{patched}"

        cast_to_fp32 = helper.make_node(
            "Cast",
            inputs=[hidden_name],
            outputs=[hidden_fp32],
            name=f"patch_rmsnorm_cast_to_fp32_{patched}",
            to=TensorProto.FLOAT,
        )
        node.input[0] = hidden_fp32
        node.input[1] = hidden_fp32
        mul_norm.input[:] = [hidden_fp32 if item == hidden_name else item for item in mul_norm.input]

        cast_to_fp16 = helper.make_node(
            "Cast",
            inputs=[mul_norm.output[0]],
            outputs=[norm_fp16],
            name=f"patch_rmsnorm_cast_to_fp16_{patched}",
            to=TensorProto.FLOAT16,
        )
        mul_weight.input[:] = [norm_fp16 if item == mul_norm.output[0] else item for item in mul_weight.input]

        shape = get_shape_by_name(model, hidden_name)
        if shape:
            model.graph.value_info.append(helper.make_tensor_value_info(hidden_fp32, TensorProto.FLOAT, shape))
            model.graph.value_info.append(helper.make_tensor_value_info(norm_fp16, TensorProto.FLOAT16, shape))

        new_nodes.extend([cast_to_fp32, node, reduce_mean, add, sqrt, reciprocal, mul_norm, cast_to_fp16, mul_weight])
        skip_names.update(n.name for n in nodes[idx + 1 : idx + 7])
        patched += 1

    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    return patched



def get_scalar_initializer(model: onnx.ModelProto, name: str) -> float | None:
    for initializer in model.graph.initializer:
        if initializer.name == name:
            array = numpy_helper.to_array(initializer)
            return float(array.reshape(-1)[0])
    return None


def _consumer_map(nodes: list[onnx.NodeProto]) -> dict[str, list[onnx.NodeProto]]:
    result: dict[str, list[onnx.NodeProto]] = {}
    for node in nodes:
        for item in node.input:
            result.setdefault(item, []).append(node)
    return result


def _single_consumer(consumers: dict[str, list[onnx.NodeProto]], name: str, op_type: str) -> onnx.NodeProto | None:
    matched = [node for node in consumers.get(name, []) if node.op_type == op_type]
    if len(matched) != 1:
        return None
    return matched[0]


def _match_rmsnorm_from_square(
    model: onnx.ModelProto,
    node: onnx.NodeProto,
    consumers: dict[str, list[onnx.NodeProto]],
) -> tuple[onnx.NodeProto, set[str]] | None:
    if node.op_type != "Mul" or len(node.input) != 2 or node.input[0] != node.input[1]:
        return None

    hidden_name = node.input[0]
    reduce_mean = _single_consumer(consumers, node.output[0], "ReduceMean")
    if reduce_mean is None:
        return None
    add = _single_consumer(consumers, reduce_mean.output[0], "Add")
    if add is None:
        return None
    sqrt = _single_consumer(consumers, add.output[0], "Sqrt")
    if sqrt is None:
        return None
    reciprocal = _single_consumer(consumers, sqrt.output[0], "Reciprocal")
    if reciprocal is None:
        return None

    mul_norm_candidates = [
        item
        for item in consumers.get(reciprocal.output[0], [])
        if item.op_type == "Mul" and hidden_name in item.input
    ]
    if len(mul_norm_candidates) != 1:
        return None
    mul_norm = mul_norm_candidates[0]

    mul_weight_candidates = [item for item in consumers.get(mul_norm.output[0], []) if item.op_type == "Mul"]
    if len(mul_weight_candidates) != 1:
        return None
    mul_weight = mul_weight_candidates[0]

    eps_name = next((item for item in add.input if item != reduce_mean.output[0]), "")
    eps = get_scalar_initializer(model, eps_name)
    if eps is None:
        return None

    weight_name = next((item for item in mul_weight.input if item != mul_norm.output[0]), "")
    if not weight_name:
        return None

    rmsnorm = helper.make_node(
        "RMSNorm",
        inputs=[hidden_name, weight_name],
        outputs=list(mul_weight.output),
        name="fused_rmsnorm_topo",
        domain=XH2A_DOMAIN,
        eps=float(eps),
        axis=-1,
        compute_mode="normal",
        pre_sub_exp=0,
        force_fp32=1,
        keep_fp32_variance=1,
        rsqrt_type="default",
        sum_dtype="float32",
    )
    matched_names = {node.name, reduce_mean.name, add.name, sqrt.name, reciprocal.name, mul_norm.name, mul_weight.name}
    return rmsnorm, matched_names


def fuse_rmsnorm(model: onnx.ModelProto) -> int:
    nodes = list(model.graph.node)
    consumers = _consumer_map(nodes)
    fused = 0
    new_nodes = []
    skip_names: set[str] = set()

    for node in nodes:
        if node.name in skip_names:
            continue
        matched = _match_rmsnorm_from_square(model, node, consumers)
        if matched is None:
            new_nodes.append(node)
            continue
        rmsnorm, matched_names = matched
        rmsnorm.name = f"fused_rmsnorm_{fused}"
        new_nodes.append(rmsnorm)
        skip_names.update(name for name in matched_names if name)
        fused += 1

    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    return fused


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--mode", choices=("cast", "fuse"), default="cast")
    args = parser.parse_args()

    model = onnx.load(args.input, load_external_data=False)
    if args.mode == "cast":
        patched = patch_rmsnorm_casts(model)
    else:
        patched = fuse_rmsnorm(model)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.output)
    print(f"{args.mode}_rmsnorm_patterns={patched}")


if __name__ == "__main__":
    main()
