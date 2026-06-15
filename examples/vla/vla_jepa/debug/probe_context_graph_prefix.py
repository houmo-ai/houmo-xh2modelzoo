# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0

"""Export and compare prefix-layer VLA-JEPA context graph probes."""

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
from pathlib import Path
from typing import Any

import onnx
import torch
from torch import nn

from compare.compare_context_graph_hmonnx import diff_summary, run_hmonnx_context
from export.export_action_head import tensor_summary
from patches.patch_hmonnx_rmsnorm_casts import fuse_rmsnorm

from common.paths import output_str, set_default_libero_config_path
from export.export_context_graph_wrapper import (
    DEFAULT_MODEL,
    GRAPH_INPUT_NAMES,
    build_graph_inputs,
    build_wrapper,
    convert_to_hmonnx,
    export_onnx,
)

DEFAULT_SAMPLE = output_str("context_encoder", "live_eval", "task0_seed1000_context_graph_sample.pt")
DEFAULT_OUT_DIR = output_str("context_encoder", "prefix_probe")


def truncate_layers(wrapper: nn.Module, num_layers: int) -> None:
    layers = wrapper.wrapped_qwen.language_model.layers
    total_layers = len(layers)
    if num_layers < 0 or num_layers > total_layers:
        raise ValueError(f"num_layers must be in [0, {total_layers}], got {num_layers}")
    wrapper.wrapped_qwen.language_model.layers = nn.ModuleList(list(layers)[:num_layers])
    report = getattr(wrapper, "wrapped_class_report", {})
    report["prefix_num_layers"] = num_layers
    report["original_num_layers"] = total_layers
    wrapper.wrapped_class_report = report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--sample", default=DEFAULT_SAMPLE)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--hmonnx-device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--num-layers", type=int, required=True)
    parser.add_argument("--quant-type", default="w16a16_sefp")
    parser.add_argument("--opset", type=int, default=12)
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--skip-convert", action="store_true")
    parser.add_argument("--skip-compare", action="store_true")
    parser.add_argument("--fuse-rmsnorm", action="store_true")
    args = parser.parse_args()

    set_default_libero_config_path()
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.manual_seed(53)

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = f"context_prefix_layers{args.num_layers}_{args.dtype}"
    onnx_path = out_dir / f"{stem}.onnx"
    hmonnx_path = out_dir / f"{stem}_{args.quant_type}.hmonnx.onnx"
    compare_hmonnx_path = hmonnx_path
    report_suffix = args.quant_type
    if args.fuse_rmsnorm:
        compare_hmonnx_path = out_dir / f"{stem}_{args.quant_type}.fused_rmsnorm.hmonnx.onnx"
        report_suffix = f"{args.quant_type}_fused_rmsnorm"
    report_path = out_dir / f"{stem}_{report_suffix}_compare_report.json"

    sample = torch.load(args.sample, map_location="cpu", weights_only=False)
    inputs = build_graph_inputs(sample, args.device, dtype)
    wrapper = build_wrapper(args.model, args.device, dtype, int(inputs[0].shape[1]))
    truncate_layers(wrapper, args.num_layers)
    wrapper.eval()

    with torch.no_grad():
        torch_output = wrapper(*inputs)

    report: dict[str, Any] = {
        "model": args.model,
        "sample": args.sample,
        "num_layers": args.num_layers,
        "device": args.device,
        "hmonnx_device": args.hmonnx_device,
        "dtype": args.dtype,
        "quant_type": args.quant_type,
        "onnx": str(onnx_path),
        "hmonnx": str(hmonnx_path),
        "compare_hmonnx": str(compare_hmonnx_path),
        "fuse_rmsnorm": bool(args.fuse_rmsnorm),
        "inputs": {str(idx): tensor_summary(item) for idx, item in enumerate(inputs)},
        "wrapped_classes": getattr(wrapper, "wrapped_class_report", {}),
        "torch_output": tensor_summary(torch_output),
    }

    if not args.skip_export:
        export_onnx(wrapper, inputs, onnx_path, args.opset)
        report["onnx_size_mb"] = onnx_path.stat().st_size / 1024 / 1024

    if not args.skip_convert:
        if not onnx_path.exists():
            export_onnx(wrapper, inputs, onnx_path, args.opset)
            report["onnx_size_mb"] = onnx_path.stat().st_size / 1024 / 1024
        report["hmonnx_export"] = convert_to_hmonnx(onnx_path, hmonnx_path, inputs, args.quant_type)

    if args.fuse_rmsnorm and not args.skip_convert:
        model = onnx.load(str(hmonnx_path), load_external_data=False)
        fused_count = fuse_rmsnorm(model)
        onnx.save(model, str(compare_hmonnx_path))
        report["fused_rmsnorm_patterns"] = fused_count

    if not args.skip_compare:
        if args.fuse_rmsnorm and not compare_hmonnx_path.exists():
            model = onnx.load(str(hmonnx_path), load_external_data=False)
            fused_count = fuse_rmsnorm(model)
            onnx.save(model, str(compare_hmonnx_path))
            report["fused_rmsnorm_patterns"] = fused_count
        hmonnx_output, onnx_input_names = run_hmonnx_context(
            str(compare_hmonnx_path),
            str(onnx_path),
            inputs,
            args.hmonnx_device,
        )
        report["onnx_input_names"] = onnx_input_names
        report["torch_vs_hmonnx"] = diff_summary(torch_output, hmonnx_output.to(args.device))

    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
