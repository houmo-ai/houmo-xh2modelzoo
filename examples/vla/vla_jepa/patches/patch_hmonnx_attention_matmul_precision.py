# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0

"""Patch HMONNX attention-core MatMul mantissa precision."""

from __future__ import annotations

import sys
from pathlib import Path

_VLA_JEPA_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _VLA_JEPA_ROOT.parents[2]
for _path in (_VLA_JEPA_ROOT, _REPO_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import argparse
import re
from pathlib import Path

import onnx


def set_int_attr(node: onnx.NodeProto, name: str, value: int) -> bool:
    for attr in node.attribute:
        if attr.name == name:
            attr.i = int(value)
            return True
    node.attribute.append(onnx.helper.make_attribute(name, int(value)))
    return False


def patch_matmul_precision(model: onnx.ModelProto, man_bit: int = 16, name_regex: str | None = None) -> int:
    pattern = re.compile(name_regex) if name_regex else None
    patched = 0
    for node in model.graph.node:
        if node.op_type != "MatMul" or node.domain != "ai.houmo.xh2a":
            continue
        if pattern is not None and not pattern.search(node.name):
            continue
        set_int_attr(node, "hmfp_act_man_bit", man_bit)
        set_int_attr(node, "hmfp_weight_man_bit", man_bit)
        patched += 1
    return patched


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--man-bit", type=int, default=16)
    parser.add_argument("--name-regex", default=None)
    args = parser.parse_args()

    model = onnx.load(args.input, load_external_data=False)
    patched = patch_matmul_precision(model, args.man_bit, args.name_regex)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(output))
    print({"input": args.input, "output": str(output), "patched_matmul": patched, "man_bit": args.man_bit})


if __name__ == "__main__":
    main()
