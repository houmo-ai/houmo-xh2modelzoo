# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0

"""Patch selected VLA-JEPA context decoder-layer HMONNX precision.

The fused context HMONNX produced by xhquant uses generic node names such as
``node_linear_123``. For the Qwen context decoder, the exported order is stable:
each layer contributes 7 Linear nodes and 2 MatMul nodes.

Linear order inside one layer:

0. q_proj
1. k_proj
2. v_proj
3. o_proj
4. gate_proj
5. up_proj
6. down_proj

MatMul order inside one layer:

0. attention score matmul
1. attention value matmul
"""

from __future__ import annotations

import sys
from pathlib import Path

_VLA_JEPA_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _VLA_JEPA_ROOT.parents[2]
for _path in (_VLA_JEPA_ROOT, _REPO_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import argparse
import json
from pathlib import Path

import onnx


LINEAR_NAMES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
ATTN_LINEAR_NAMES = {"q_proj", "k_proj", "v_proj", "o_proj"}
MLP_LINEAR_NAMES = {"gate_proj", "up_proj", "down_proj"}


def set_int_attr(node: onnx.NodeProto, name: str, value: int) -> None:
    for attr in node.attribute:
        if attr.name == name:
            attr.i = int(value)
            return
    node.attribute.append(onnx.helper.make_attribute(name, int(value)))


def parse_layers(value: str, num_layers: int) -> set[int]:
    if value == "all":
        return set(range(num_layers))

    result: set[int] = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start, end = item.split("-", maxsplit=1)
            result.update(range(int(start), int(end) + 1))
        else:
            result.add(int(item))

    invalid = sorted(idx for idx in result if idx < 0 or idx >= num_layers)
    if invalid:
        raise ValueError(f"Invalid layer indices {invalid}; expected 0..{num_layers - 1}")
    return result


def selected_linear_names(mode: str) -> set[str]:
    if mode == "all":
        return set(LINEAR_NAMES)
    if mode == "attention":
        return set(ATTN_LINEAR_NAMES)
    if mode == "mlp":
        return set(MLP_LINEAR_NAMES)
    raise ValueError(f"Unsupported linear mode: {mode}")


def patch_context_layer_precision(
    model: onnx.ModelProto,
    layers: set[int],
    linear_mode: str,
    linear_weight_man_bit: int | None,
    linear_act_man_bit: int | None,
    matmul_man_bit: int | None,
    num_layers: int,
) -> dict[str, object]:
    linear_names = selected_linear_names(linear_mode)
    linear_idx = 0
    matmul_idx = 0
    patched_linear: list[dict[str, object]] = []
    patched_matmul: list[dict[str, object]] = []

    for node in model.graph.node:
        if node.domain != "ai.houmo.xh2a":
            continue

        if node.op_type == "Linear":
            layer_idx = linear_idx // len(LINEAR_NAMES)
            linear_name = LINEAR_NAMES[linear_idx % len(LINEAR_NAMES)]
            linear_idx += 1
            if layer_idx >= num_layers:
                continue
            if layer_idx not in layers or linear_name not in linear_names:
                continue
            if linear_weight_man_bit is not None:
                set_int_attr(node, "hmfp_weight_man_bit", linear_weight_man_bit)
            if linear_act_man_bit is not None:
                set_int_attr(node, "hmfp_act_man_bit", linear_act_man_bit)
            patched_linear.append({"layer": layer_idx, "linear": linear_name, "node": node.name})

        elif node.op_type == "MatMul":
            layer_idx = matmul_idx // 2
            matmul_name = "attn_score" if matmul_idx % 2 == 0 else "attn_value"
            matmul_idx += 1
            if layer_idx >= num_layers or layer_idx not in layers:
                continue
            if matmul_man_bit is not None:
                set_int_attr(node, "hmfp_act_man_bit", matmul_man_bit)
                set_int_attr(node, "hmfp_weight_man_bit", matmul_man_bit)
                patched_matmul.append({"layer": layer_idx, "matmul": matmul_name, "node": node.name})

    return {
        "num_layers": num_layers,
        "layers": sorted(layers),
        "linear_mode": linear_mode,
        "linear_weight_man_bit": linear_weight_man_bit,
        "linear_act_man_bit": linear_act_man_bit,
        "matmul_man_bit": matmul_man_bit,
        "total_linear_nodes": linear_idx,
        "total_matmul_nodes": matmul_idx,
        "patched_linear_count": len(patched_linear),
        "patched_matmul_count": len(patched_matmul),
        "patched_linear": patched_linear,
        "patched_matmul": patched_matmul,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--layers", default="all", help="Layer list/ranges, e.g. all, 27, 24-27, 16-27")
    parser.add_argument("--num-layers", type=int, default=28)
    parser.add_argument("--linear-mode", choices=("all", "attention", "mlp"), default="all")
    parser.add_argument("--linear-weight-man-bit", type=int, default=None)
    parser.add_argument("--linear-act-man-bit", type=int, default=None)
    parser.add_argument("--matmul-man-bit", type=int, default=None)
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    layers = parse_layers(args.layers, args.num_layers)
    model = onnx.load(args.input, load_external_data=False)
    result = patch_context_layer_precision(
        model,
        layers=layers,
        linear_mode=args.linear_mode,
        linear_weight_man_bit=args.linear_weight_man_bit,
        linear_act_man_bit=args.linear_act_man_bit,
        matmul_man_bit=args.matmul_man_bit,
        num_layers=args.num_layers,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(output))

    result = {"input": args.input, "output": str(output), **result}
    if args.report:
        report = Path(args.report)
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
