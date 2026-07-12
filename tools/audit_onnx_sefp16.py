#!/usr/bin/env python3
"""Audit exported ONNX graphs for all-SEFP16 activation precision."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import onnx


FLASH_BITS = ("q_bits", "k_bits", "v_bits", "s_bits", "p_bits")
ACTIVATION_OPS = {"Linear", "MatMul"}
SEFP16_ACTIVATION_ATTRIBUTES = {
    "hmfp_act_man_bit": 16,
    "hmfp_act_exp_bit": 5,
    "hmfp_act_hidden_bit": 1,
    "hmfp_act_nshare": 64,
}


def _attribute_value(attribute: onnx.AttributeProto) -> Any:
    value = onnx.helper.get_attribute_value(attribute)
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, tuple):
        return list(value)
    return value


def _attributes(node: onnx.NodeProto) -> dict[str, Any]:
    return {attribute.name: _attribute_value(attribute) for attribute in node.attribute}


def audit_model(path: Path) -> dict[str, Any]:
    model = onnx.load(path, load_external_data=False)
    activation_nodes: list[dict[str, Any]] = []
    flash_nodes: list[dict[str, Any]] = []
    violations: list[str] = []

    for index, node in enumerate(model.graph.node):
        attributes = _attributes(node)
        identity = node.name or f"{node.op_type}[{index}]"
        if node.op_type in ACTIVATION_OPS:
            record = {
                "name": identity,
                "op_type": node.op_type,
                "domain": node.domain,
                "mode": attributes.get("mode"),
                "hmfp_act_man_bit": attributes.get("hmfp_act_man_bit"),
                "hmfp_act_exp_bit": attributes.get("hmfp_act_exp_bit"),
                "hmfp_act_hidden_bit": attributes.get("hmfp_act_hidden_bit"),
                "hmfp_act_nshare": attributes.get("hmfp_act_nshare"),
            }
            activation_nodes.append(record)
            # ``mode`` follows the weight encoding on Linear nodes (and can
            # legitimately be ssfp). Activation SEFP is encoded independently
            # by the full activation precision tuple.
            invalid_attributes = {
                name: record[name]
                for name, expected in SEFP16_ACTIVATION_ATTRIBUTES.items()
                if record[name] != expected
            }
            if invalid_attributes:
                actual = ", ".join(
                    f"{name}={record[name]!r}" for name in SEFP16_ACTIVATION_ATTRIBUTES
                )
                violations.append(
                    f"{identity}: {node.op_type} activation is not SEFP16 ({actual})"
                )

        if "flashattention" in node.op_type.lower():
            bits = {name: attributes.get(name) for name in FLASH_BITS}
            flash_nodes.append(
                {"name": identity, "op_type": node.op_type, "domain": node.domain, **bits}
            )
            for name, value in bits.items():
                if value != 16:
                    violations.append(
                        f"{identity}: {node.op_type} {name} must be explicit 16, got {value!r}"
                    )

    if not activation_nodes:
        violations.append("graph has no Linear or MatMul nodes")
    if not flash_nodes:
        violations.append("graph has no FlashAttention nodes")

    precision_inventory = Counter(
        (
            record["op_type"],
            record["mode"],
            record["hmfp_act_man_bit"],
            record["hmfp_act_exp_bit"],
            record["hmfp_act_hidden_bit"],
            record["hmfp_act_nshare"],
        )
        for record in activation_nodes
    )
    return {
        "path": str(path.resolve()),
        "node_count": len(model.graph.node),
        "activation_node_count": len(activation_nodes),
        "flash_attention_node_count": len(flash_nodes),
        "precision_inventory": [
            {
                "op_type": key[0],
                "mode": key[1],
                "hmfp_act_man_bit": key[2],
                "hmfp_act_exp_bit": key[3],
                "hmfp_act_hidden_bit": key[4],
                "hmfp_act_nshare": key[5],
                "count": count,
            }
            for key, count in sorted(precision_inventory.items(), key=lambda item: repr(item[0]))
        ],
        "activation_nodes": activation_nodes,
        "flash_attention_nodes": flash_nodes,
        "violations": violations,
        "passed": not violations,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("onnx", nargs="+", type=Path, help="prefill/decode ONNX paths")
    parser.add_argument("--json", action="store_true", help="emit full per-node JSON evidence")
    args = parser.parse_args()

    reports = [audit_model(path) for path in args.onnx]
    if args.json:
        print(json.dumps(reports, indent=2, sort_keys=True))
    else:
        for report in reports:
            status = "PASS" if report["passed"] else "FAIL"
            print(f"{status}: {report['path']}")
            print(
                f"  graph nodes={report['node_count']} Linear/MatMul={report['activation_node_count']} "
                f"FlashAttention={report['flash_attention_node_count']}"
            )
            for item in report["precision_inventory"]:
                print(
                    f"  {item['op_type']} mode={item['mode']!r} "
                    f"hmfp_act_man_bit={item['hmfp_act_man_bit']!r} "
                    f"hmfp_act_exp_bit={item['hmfp_act_exp_bit']!r} "
                    f"hmfp_act_hidden_bit={item['hmfp_act_hidden_bit']!r} "
                    f"hmfp_act_nshare={item['hmfp_act_nshare']!r}: {item['count']}"
                )
            for node in report["flash_attention_nodes"]:
                bits = " ".join(f"{name}={node[name]!r}" for name in FLASH_BITS)
                print(f"  FlashAttention {node['name']}: {bits}")
            for violation in report["violations"]:
                print(f"  ERROR: {violation}")
    return 0 if all(report["passed"] for report in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
