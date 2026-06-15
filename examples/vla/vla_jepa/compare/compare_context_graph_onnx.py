# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0

"""Compare VLA-JEPA context graph PyTorch wrapper and ONNX output."""

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

import numpy as np
import onnx
import onnxruntime as ort
import torch

from export.export_action_head import tensor_summary
from common.paths import output_str, set_default_libero_config_path
from export.export_context_graph_wrapper import (
    DEFAULT_MODEL,
    GRAPH_INPUT_NAMES,
    build_graph_inputs,
    build_wrapper,
    select_onnx_example_inputs,
)

DEFAULT_SAMPLE = output_str("context_encoder", "live_eval", "task0_seed1000_context_graph_sample.pt")
DEFAULT_ONNX = output_str("context_encoder", "live_eval_227", "vla_jepa_context_graph_wrapper_float16.onnx")
DEFAULT_REPORT = output_str("context_encoder", "live_eval_227", "onnx_compare_context_graph_report.json")


def diff_summary(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    ref = reference.detach().cpu().float().numpy()
    cand = candidate.detach().cpu().float().numpy()
    diff = np.abs(ref - cand)
    return {
        "reference": tensor_summary(reference),
        "candidate": tensor_summary(candidate),
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "p95_abs_diff": float(np.percentile(diff, 95)),
        "allclose_rtol_1e-2_atol_1e-2": bool(np.allclose(ref, cand, rtol=1e-2, atol=1e-2)),
        "allclose_rtol_5e-2_atol_5e-2": bool(np.allclose(ref, cand, rtol=5e-2, atol=5e-2)),
        "allclose_rtol_1e-1_atol_1e-1": bool(np.allclose(ref, cand, rtol=1e-1, atol=1e-1)),
    }


def _cuda_device_id(device: str) -> int:
    if device.startswith("cuda:"):
        return int(device.split(":", 1)[1])
    return 0


def _providers(provider: str, device: str) -> list[Any]:
    if provider == "cuda":
        return [
            ("CUDAExecutionProvider", {"device_id": _cuda_device_id(device)}),
            "CPUExecutionProvider",
        ]
    if provider == "cpu":
        return ["CPUExecutionProvider"]
    return ort.get_available_providers()


def run_onnx(
    onnx_path: Path,
    inputs: tuple[torch.Tensor, ...],
    provider: str,
    device: str,
) -> tuple[torch.Tensor, list[str], list[str]]:
    _, onnx_input_names = select_onnx_example_inputs(onnx_path, inputs)
    input_by_name = dict(zip(GRAPH_INPUT_NAMES, inputs, strict=True))

    sess = ort.InferenceSession(str(onnx_path), providers=_providers(provider, device))
    ort_inputs = {}
    for input_info in sess.get_inputs():
        tensor = input_by_name[input_info.name].detach().cpu()
        ort_inputs[input_info.name] = tensor.numpy()

    outputs = sess.run(None, ort_inputs)
    if len(outputs) != 1:
        raise RuntimeError(f"Expected 1 ONNX output, got {len(outputs)}")
    return torch.from_numpy(outputs[0]), onnx_input_names, sess.get_providers()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--sample", default=DEFAULT_SAMPLE)
    parser.add_argument("--onnx", default=DEFAULT_ONNX)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--ort-provider", choices=("cuda", "cpu", "auto"), default="cuda")
    parser.add_argument("--report", default=DEFAULT_REPORT)
    args = parser.parse_args()

    set_default_libero_config_path()
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.manual_seed(47)

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    sample = torch.load(args.sample, map_location="cpu", weights_only=False)
    inputs = build_graph_inputs(sample, args.device, dtype)

    onnx_path = Path(args.onnx)
    onnx.checker.check_model(str(onnx_path))

    wrapper = build_wrapper(args.model, args.device, dtype, int(inputs[0].shape[1]))
    with torch.no_grad():
        torch_output = wrapper(*inputs)

    onnx_output, onnx_input_names, providers = run_onnx(onnx_path, inputs, args.ort_provider, args.device)
    onnx_output = onnx_output.to(args.device)

    report = {
        "model": args.model,
        "sample": args.sample,
        "onnx": args.onnx,
        "device": args.device,
        "dtype": args.dtype,
        "ort_provider_arg": args.ort_provider,
        "ort_providers": providers,
        "onnx_input_names": onnx_input_names,
        "wrapped_classes": getattr(wrapper, "wrapped_class_report", {}),
        "inputs": {str(idx): tensor_summary(item) for idx, item in enumerate(inputs)},
        "torch_vs_onnx": diff_summary(torch_output, onnx_output),
    }

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
