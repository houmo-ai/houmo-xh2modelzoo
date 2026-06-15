# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0

"""Standard VLA-JEPA XH2a export/quantization entry.

This script is the VLA-JEPA counterpart of the Qwen3-VL export entry:

    HF/LeRobot policy
        -> stable graph boundaries
        -> ONNX export
        -> HMONNX quantization
        -> optional post-process patches
        -> golden/compare reports
        -> export meta

VLA-JEPA is not a normal autoregressive chat model at inference time. The
runtime boundary used here is therefore split into two graphs:

1. context graph: Qwen3-VL text/context stack -> conditioning_tokens
2. action head: one deterministic denoise step -> pred_velocity

The visual preprocessing and full LIBERO rollout evaluation are kept in the
existing capture/eval scripts because they own environment rendering and task
success metrics.
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
import os
import time
from pathlib import Path
from typing import Any

import torch

from compare.compare_context_graph_hmonnx import diff_summary, run_hmonnx_context
from export.export_action_head import (
    ActionHeadStep,
    DEFAULT_MODEL,
    convert_to_hmonnx as convert_action_to_hmonnx,
    load_action_head,
    make_dummy_inputs,
    run_hmonnx_golden as run_action_hmonnx_golden,
    tensor_summary,
    validate_onnx as validate_action_onnx,
)
from common.paths import output_str, set_default_libero_config_path
from export.export_context_graph_wrapper import (
    DEFAULT_SAMPLE as DUMMY_CONTEXT_SAMPLE,
    GRAPH_INPUT_NAMES,
    build_graph_inputs,
    build_wrapper,
    convert_to_hmonnx as convert_context_to_hmonnx,
    export_onnx as export_context_onnx,
    fuse_hmonnx_rmsnorm,
)

DEFAULT_CONTEXT_SAMPLE = output_str("context_encoder", "live_eval", "task0_seed1000_context_graph_sample.pt")
DEFAULT_OUT_DIR = output_str("standard_export")


def setup_runtime_env() -> None:
    set_default_libero_config_path()
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def torch_dtype(name: str) -> torch.dtype:
    table = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    return table[name]


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def path_info(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "size_mb": path.stat().st_size / 1024 / 1024 if path.exists() else None,
    }


def context_quant_suffix(args: argparse.Namespace) -> str:
    suffix = args.context_quant_type
    if args.context_linear_quant_type:
        suffix = f"{suffix}_linear_{args.context_linear_quant_type}"
    if args.qwen3vl_matmul_ops:
        suffix = f"{suffix}_qwenmatmul"
    return suffix.replace("/", "_")


def reference_info(reference: Any) -> tuple[dict[str, Any], bool]:
    info: dict[str, Any] = {
        "present": torch.is_tensor(reference),
        "valid_for_compare": False,
    }
    if not torch.is_tensor(reference):
        info["warning"] = "conditioning_tokens reference is absent from the sample"
        return info, False

    nonzero = bool(torch.count_nonzero(reference.detach().cpu()).item())
    info["summary"] = tensor_summary(reference)
    info["valid_for_compare"] = nonzero
    if not nonzero:
        info["warning"] = "conditioning_tokens reference is all zeros; skip original_vs_* comparisons"
    return info, nonzero


def run_context_hmonnx_golden(
    hmonnx_path: Path,
    inputs: tuple[torch.Tensor, ...],
    golden_dir: Path,
    device: str,
) -> dict[str, Any]:
    from xhquant.api import HMONNXGoldenInference

    input_by_name = dict(zip(GRAPH_INPUT_NAMES, inputs))
    golden_dir.mkdir(parents=True, exist_ok=True)

    session = HMONNXGoldenInference(str(hmonnx_path))
    session.save_golden = True
    session.golden_dir = str(golden_dir)
    session.step = 0
    session.to(device)

    run_inputs = []
    for input_info in session.inputs:
        item = input_by_name[input_info.name].detach().cpu().to(dtype=input_info.dtype)
        run_inputs.append(item.to(device))

    with torch.no_grad():
        output = session(*run_inputs)
    if isinstance(output, (tuple, list)):
        output = output[0]

    return {
        "golden_dir": str(golden_dir),
        "device": device,
        "output": tensor_summary(output),
        "input_names": [item.name for item in session.inputs],
    }


def export_context(args: argparse.Namespace, out_dir: Path) -> dict[str, Any]:
    dtype = torch_dtype(args.dtype)
    context_dir = out_dir / "context"
    context_dir.mkdir(parents=True, exist_ok=True)

    onnx_path = context_dir / f"vla_jepa_context_graph_wrapper_{args.dtype}.onnx"
    quant_suffix = context_quant_suffix(args)
    hmonnx_path = context_dir / f"vla_jepa_context_graph_wrapper_{args.dtype}_{quant_suffix}.hmonnx.onnx"
    fused_hmonnx_path = context_dir / f"vla_jepa_context_graph_wrapper_{args.dtype}_{quant_suffix}.fused_rmsnorm.hmonnx.onnx"
    compare_report_path = out_dir / "reports" / "context_compare_report.json"

    sample = torch.load(args.context_sample, map_location="cpu", weights_only=False)
    inputs = build_graph_inputs(sample, args.device, dtype, args.context_length)
    wrapper = build_wrapper(args.model, args.device, dtype, int(inputs[0].shape[1]))

    with torch.no_grad():
        torch_output = wrapper(*inputs)

    reference = sample.get("conditioning_tokens")
    sample_reference, reference_valid = reference_info(reference)
    report: dict[str, Any] = {
        "graph": "context",
        "model": args.model,
        "sample": args.context_sample,
        "device": args.device,
        "hmonnx_device": args.hmonnx_device,
        "dtype": args.dtype,
        "opset": args.context_opset,
        "context_length": args.context_length,
        "quant_type": args.context_quant_type,
        "context_linear_quant_type": args.context_linear_quant_type,
        "qwen3vl_matmul_ops": bool(args.qwen3vl_matmul_ops),
        "fuse_rmsnorm": bool(args.fuse_rmsnorm),
        "onnx": path_info(onnx_path),
        "hmonnx": path_info(hmonnx_path),
        "fused_hmonnx": path_info(fused_hmonnx_path),
        "inputs": {name: tensor_summary(value) for name, value in zip(GRAPH_INPUT_NAMES, inputs)},
        "wrapped_classes": getattr(wrapper, "wrapped_class_report", {}),
        "sample_reference": sample_reference,
        "torch_output": tensor_summary(torch_output),
    }
    if reference_valid:
        report["original_vs_torch_wrapper"] = diff_summary(reference.to(args.device), torch_output)

    if args.overwrite or not onnx_path.exists():
        export_context_onnx(wrapper, inputs, onnx_path, args.context_opset)
    report["onnx"] = path_info(onnx_path)

    if args.overwrite or not hmonnx_path.exists():
        report["convert_hmonnx"] = convert_context_to_hmonnx(
            onnx_path,
            hmonnx_path,
            inputs,
            args.context_quant_type,
            use_qwen3vl_matmul_ops=bool(args.qwen3vl_matmul_ops),
            linear_quant_type=args.context_linear_quant_type,
        )
    report["hmonnx"] = path_info(hmonnx_path)

    final_hmonnx_path = hmonnx_path
    if args.fuse_rmsnorm:
        if args.overwrite or not fused_hmonnx_path.exists():
            report["rmsnorm_fusion"] = fuse_hmonnx_rmsnorm(hmonnx_path, fused_hmonnx_path)
        final_hmonnx_path = fused_hmonnx_path
    report["final_hmonnx"] = path_info(final_hmonnx_path)

    if not args.skip_context_compare:
        hmonnx_output, onnx_input_names = run_hmonnx_context(
            str(final_hmonnx_path), str(onnx_path), inputs, args.hmonnx_device
        )
        report["onnx_input_names"] = onnx_input_names
        report["torch_vs_hmonnx"] = diff_summary(torch_output, hmonnx_output.to(args.device))
        if reference_valid:
            report["original_vs_hmonnx"] = diff_summary(reference.to(args.device), hmonnx_output.to(args.device))

    if args.run_golden:
        report["hmonnx_golden"] = run_context_hmonnx_golden(
            final_hmonnx_path,
            inputs,
            out_dir / "golden" / "context",
            args.hmonnx_device,
        )

    write_json(compare_report_path, report)
    return report


def export_action(args: argparse.Namespace, out_dir: Path) -> dict[str, Any]:
    action_dir = out_dir / "action_head"
    action_dir.mkdir(parents=True, exist_ok=True)

    onnx_path = action_dir / "vla_jepa_action_head_step.onnx"
    hmonnx_path = action_dir / f"vla_jepa_action_head_step_{args.action_quant_type}.hmonnx.onnx"
    report_path = out_dir / "reports" / "action_head_report.json"

    inputs = make_dummy_inputs(args.device)
    report: dict[str, Any] = {
        "graph": "action_head",
        "model": args.model,
        "device": args.device,
        "hmonnx_device": args.hmonnx_device,
        "opset": args.action_opset,
        "quant_type": args.action_quant_type,
        "onnx": path_info(onnx_path),
        "hmonnx": path_info(hmonnx_path),
        "inputs": {
            "conditioning_tokens": tensor_summary(inputs[0]),
            "actions": tensor_summary(inputs[1]),
            "state": tensor_summary(inputs[2]),
            "timesteps": tensor_summary(inputs[3]),
        },
    }

    torch_output = None
    if args.overwrite or not onnx_path.exists():
        action_head = load_action_head(args.model, args.device)
        model = ActionHeadStep(action_head).eval().to(args.device)
        with torch.no_grad():
            torch_output = model(*inputs)
        torch.onnx.export(
            model,
            inputs,
            str(onnx_path),
            export_params=True,
            opset_version=args.action_opset,
            do_constant_folding=True,
            input_names=["conditioning_tokens", "actions", "state", "timesteps"],
            output_names=["pred_velocity"],
        )
    report["onnx"] = path_info(onnx_path)

    if torch_output is None:
        action_head = load_action_head(args.model, args.device)
        model = ActionHeadStep(action_head).eval().to(args.device)
        with torch.no_grad():
            torch_output = model(*inputs)
    report["torch_output"] = tensor_summary(torch_output)

    if not args.skip_action_ort:
        report["onnxruntime"] = validate_action_onnx(onnx_path, inputs, torch_output)

    if args.overwrite or not hmonnx_path.exists():
        report["convert_hmonnx"] = convert_action_to_hmonnx(
            onnx_path,
            hmonnx_path,
            inputs,
            args.action_quant_type,
            simplify=not args.no_action_simplify,
        )
    report["hmonnx"] = path_info(hmonnx_path)
    report["final_hmonnx"] = path_info(hmonnx_path)

    if args.run_golden:
        report["hmonnx_golden"] = run_action_hmonnx_golden(
            hmonnx_path,
            inputs,
            out_dir / "golden" / "action_head",
            args.hmonnx_device,
        )

    write_json(report_path, report)
    return report


def write_meta(args: argparse.Namespace, out_dir: Path, reports: dict[str, dict[str, Any]]) -> Path:
    graphs = {}
    for name, report in reports.items():
        graphs[name] = {
            "onnx": report.get("onnx"),
            "hmonnx": report.get("hmonnx"),
            "final_hmonnx": report.get("final_hmonnx"),
            "quant_type": report.get("quant_type"),
            "report": str(out_dir / "reports" / ("context_compare_report.json" if name == "context" else "action_head_report.json")),
        }

    meta = {
        "format": "vla_jepa_xh2a_standard_export",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "model": args.model,
        "out_dir": str(out_dir),
        "device": args.device,
        "hmonnx_device": args.hmonnx_device,
        "dtype": args.dtype,
        "context_sample": args.context_sample,
        "context_length": args.context_length,
        "graphs": graphs,
        "notes": [
            "context graph exports Qwen/context stack to conditioning_tokens",
            "action_head graph exports one deterministic denoise step",
            "full LIBERO success-rate evaluation is still run by eval_full_hmonnx_libero.py",
        ],
    }
    meta_path = out_dir / "export_meta_info.json"
    write_json(meta_path, meta)
    return meta_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export VLA-JEPA context/action graphs to XH2a HMONNX.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--context-sample", default=DEFAULT_CONTEXT_SAMPLE)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--hmonnx-device", default="cpu")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--context-opset", type=int, default=12)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--action-opset", type=int, default=18)
    parser.add_argument("--context-quant-type", default="w8a16_sefp")
    parser.add_argument("--context-linear-quant-type", default="w16a16_sefp")
    parser.add_argument("--action-quant-type", default="w8a8_sefp")
    parser.add_argument("--context-only", action="store_true")
    parser.add_argument("--action-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--qwen3vl-matmul-ops", action="store_true")
    parser.add_argument("--no-fuse-rmsnorm", dest="fuse_rmsnorm", action="store_false")
    parser.set_defaults(fuse_rmsnorm=True)
    parser.add_argument("--skip-context-compare", action="store_true")
    parser.add_argument("--skip-action-ort", action="store_true")
    parser.add_argument("--no-action-simplify", action="store_true")
    parser.add_argument("--run-golden", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_runtime_env()
    torch.manual_seed(43)

    context_sample = Path(args.context_sample)
    if not context_sample.exists() and args.context_sample == DEFAULT_CONTEXT_SAMPLE:
        args.context_sample = DUMMY_CONTEXT_SAMPLE

    if args.context_only and args.action_only:
        raise ValueError("--context-only and --action-only cannot be used together")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    reports: dict[str, dict[str, Any]] = {}

    if not args.action_only:
        reports["context"] = export_context(args, out_dir)
    if not args.context_only:
        reports["action_head"] = export_action(args, out_dir)

    meta_path = write_meta(args, out_dir, reports)
    print(json.dumps({"meta": str(meta_path), "graphs": list(reports)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
