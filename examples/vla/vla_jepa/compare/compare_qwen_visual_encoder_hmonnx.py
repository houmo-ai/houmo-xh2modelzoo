# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0

"""Compare VLA-JEPA Qwen visual encoder PyTorch wrapper and HMONNX."""

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
import torch

from export.export_action_head import tensor_summary
from export.export_context_encoder import DEFAULT_SAMPLE, load_sample
from export.export_qwen_visual_encoder import (
    QwenVisualEncoder,
    configure_visual_for_export,
    replace_visual_attention_for_export,
    replace_visual_patch_embed_for_export,
)
from common.inspect_policy import DEFAULT_MODEL, load_policy
from common.paths import DEFAULT_VISUAL_ENCODER_HMONNX, output_str, set_default_libero_config_path


DEFAULT_HMONNX = DEFAULT_VISUAL_ENCODER_HMONNX
DEFAULT_REPORT = output_str("context_encoder", "qwen_visual_encoder", "hmonnx_compare_report.json")
OUTPUT_NAMES = (
    "image_embeds",
    "deepstack_feature_0",
    "deepstack_feature_1",
    "deepstack_feature_2",
)


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


def build_torch_wrapper(model_path: str, device: str, image_grid_thw: torch.Tensor) -> QwenVisualEncoder:
    policy = load_policy(model_path, device, disable_world_model=True)
    policy.eval()
    qwen_model = policy.model.qwen.model
    configure_visual_for_export(qwen_model)
    replace_visual_patch_embed_for_export(qwen_model)
    replace_visual_attention_for_export(qwen_model, image_grid_thw)
    return QwenVisualEncoder(qwen_model, image_grid_thw).eval().float().to(device)


def normalize_outputs(outputs: Any) -> tuple[torch.Tensor, ...]:
    if isinstance(outputs, torch.Tensor):
        return (outputs,)
    return tuple(outputs)


def run_hmonnx(hmonnx_path: str, pixel_values: torch.Tensor, device: str, input_dtype: str) -> tuple[torch.Tensor, ...]:
    from xhquant.api import HMONNXInference

    session = HMONNXInference(hmonnx_path)
    session.to(device)
    if input_dtype == "float16":
        hmonnx_input = pixel_values.half()
    elif input_dtype == "bfloat16":
        hmonnx_input = pixel_values.bfloat16()
    else:
        hmonnx_input = pixel_values.float()
    hmonnx_input = hmonnx_input.to(device)
    with torch.no_grad():
        return normalize_outputs(session(hmonnx_input))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--sample", default=DEFAULT_SAMPLE)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--hmonnx", default=DEFAULT_HMONNX)
    parser.add_argument("--hmonnx-device", default="cpu")
    parser.add_argument("--hmonnx-input-dtype", choices=("float32", "float16", "bfloat16"), default="float16")
    parser.add_argument("--report", default=DEFAULT_REPORT)
    args = parser.parse_args()

    set_default_libero_config_path()
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.manual_seed(43)

    sample = load_sample(args.sample, args.device)
    pixel_values = sample["qwen_inputs"]["pixel_values"]
    image_grid_thw = sample["qwen_inputs"]["image_grid_thw"]

    wrapper = build_torch_wrapper(args.model, args.device, image_grid_thw)
    with torch.no_grad():
        torch_outputs = normalize_outputs(wrapper(pixel_values.float().to(args.device)))

    hmonnx_outputs = run_hmonnx(args.hmonnx, pixel_values.detach().cpu(), args.hmonnx_device, args.hmonnx_input_dtype)

    if len(torch_outputs) != len(hmonnx_outputs):
        raise RuntimeError(f"Output count mismatch: torch={len(torch_outputs)} hmonnx={len(hmonnx_outputs)}")

    output_reports = {
        name: diff_summary(reference, candidate)
        for name, reference, candidate in zip(OUTPUT_NAMES, torch_outputs, hmonnx_outputs, strict=True)
    }
    report = {
        "model": args.model,
        "sample": args.sample,
        "torch_device": args.device,
        "hmonnx": args.hmonnx,
        "hmonnx_device": args.hmonnx_device,
        "hmonnx_input_dtype": args.hmonnx_input_dtype,
        "inputs": {
            "pixel_values": tensor_summary(pixel_values),
            "image_grid_thw": tensor_summary(image_grid_thw),
        },
        "summary": {
            "max_abs_diff_over_outputs": max(item["max_abs_diff"] for item in output_reports.values()),
            "mean_abs_diff_over_outputs": float(np.mean([item["mean_abs_diff"] for item in output_reports.values()])),
        },
        "outputs": output_reports,
    }

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
