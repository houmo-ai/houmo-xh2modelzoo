# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0

"""Patch context HMONNX precision to match the useful parts of w8a16.

For VLA-JEPA context graph, w8a16_sefp differs from w8a8_sefp in two
important ways:

* ai.houmo.xh2a::Linear keeps weight mantissa at 8 bits, but raises
  activation mantissa to 16 bits.
* ai.houmo.xh2a::MatMul is used for activation-activation attention
  products, so both sides use 16-bit mantissa.
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
from pathlib import Path

import onnx


def set_int_attr(node: onnx.NodeProto, name: str, value: int) -> None:
    for attr in node.attribute:
        if attr.name == name:
            attr.i = int(value)
            return
    node.attribute.append(onnx.helper.make_attribute(name, int(value)))


def patch_context_precision(
    model: onnx.ModelProto,
    linear_act_man_bit: int = 16,
    matmul_man_bit: int = 16,
) -> dict[str, int]:
    patched_linear = 0
    patched_matmul = 0

    for node in model.graph.node:
        if node.domain != "ai.houmo.xh2a":
            continue

        if node.op_type == "Linear":
            set_int_attr(node, "hmfp_act_man_bit", linear_act_man_bit)
            patched_linear += 1
        elif node.op_type == "MatMul":
            set_int_attr(node, "hmfp_act_man_bit", matmul_man_bit)
            set_int_attr(node, "hmfp_weight_man_bit", matmul_man_bit)
            patched_matmul += 1

    return {
        "patched_linear": patched_linear,
        "patched_matmul": patched_matmul,
        "linear_act_man_bit": linear_act_man_bit,
        "matmul_man_bit": matmul_man_bit,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--linear-act-man-bit", type=int, default=16)
    parser.add_argument("--matmul-man-bit", type=int, default=16)
    args = parser.parse_args()

    model = onnx.load(args.input, load_external_data=False)
    result = patch_context_precision(
        model,
        linear_act_man_bit=args.linear_act_man_bit,
        matmul_man_bit=args.matmul_man_bit,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(output))

    print({"input": args.input, "output": str(output), **result})


if __name__ == "__main__":
    main()
