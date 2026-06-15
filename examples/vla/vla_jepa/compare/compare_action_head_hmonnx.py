# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0

"""Compare VLA-JEPA ActionHead PyTorch and HMONNX on captured inference steps."""

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
from types import MethodType
from typing import Any

import numpy as np
import torch

from export.export_action_head import ActionHeadStep, tensor_summary
from common.inspect_policy import DEFAULT_MODEL, load_policy, make_dummy_batch
from common.paths import DEFAULT_ACTION_HEAD_HMONNX, output_str, set_default_libero_config_path


DEFAULT_HMONNX = DEFAULT_ACTION_HEAD_HMONNX
DEFAULT_REPORT = output_str("action_head", "hmonnx_compare_report.json")
DEFAULT_CAPTURE_DIR = output_str("action_head", "captured_steps")


def capture_action_head_steps(model_path: str, device: str, capture_dir: Path) -> tuple[Any, list[dict[str, torch.Tensor]]]:
    policy = load_policy(model_path, device, disable_world_model=True)
    policy.eval()
    policy.reset()

    captured: list[dict[str, torch.Tensor]] = []
    action_head = policy.model.action_model
    original_build_inputs = action_head._build_inputs

    def build_inputs_wrapper(self, conditioning_tokens, actions, state, timesteps):
        captured.append(
            {
                "conditioning_tokens": conditioning_tokens.detach().float().cpu(),
                "actions": actions.detach().float().cpu(),
                "state": state.detach().float().cpu() if state is not None else None,
                "timesteps": timesteps.detach().cpu(),
            }
        )
        return original_build_inputs(conditioning_tokens, actions, state, timesteps)

    action_head._build_inputs = MethodType(build_inputs_wrapper, action_head)
    batch = make_dummy_batch(policy.config, device)
    with torch.no_grad():
        policy.predict_action_chunk(batch)
    action_head._build_inputs = original_build_inputs

    capture_dir.mkdir(parents=True, exist_ok=True)
    for idx, item in enumerate(captured):
        save_item = {key: value for key, value in item.items() if value is not None}
        torch.save(save_item, capture_dir / f"step_{idx}.pt")
    return policy.model.action_model, captured


def run_hmonnx_step(session: Any, step: dict[str, torch.Tensor], device: str) -> torch.Tensor:
    conditioning_tokens = step["conditioning_tokens"].half().to(device)
    actions = step["actions"].half().to(device)
    state = step["state"].half().to(device)
    timesteps = step["timesteps"].int().to(device)
    with torch.no_grad():
        return session(conditioning_tokens, actions, state, timesteps)


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
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--hmonnx", default=DEFAULT_HMONNX)
    parser.add_argument("--hmonnx-device", default="cpu")
    parser.add_argument("--report", default=DEFAULT_REPORT)
    parser.add_argument("--capture-dir", default=DEFAULT_CAPTURE_DIR)
    args = parser.parse_args()

    set_default_libero_config_path()
    os.environ.setdefault("MUJOCO_GL", "egl")
    torch.manual_seed(13)

    action_head, captured = capture_action_head_steps(args.model, args.device, Path(args.capture_dir))
    action_head_step = ActionHeadStep(action_head).eval().float().to(args.device)

    from xhquant.api import HMONNXInference

    hmonnx_session = HMONNXInference(args.hmonnx)
    hmonnx_session.to(args.hmonnx_device)

    step_reports = []
    for idx, step in enumerate(captured):
        torch_inputs = (
            step["conditioning_tokens"].to(args.device),
            step["actions"].to(args.device),
            step["state"].to(args.device),
            step["timesteps"].long().to(args.device),
        )
        with torch.no_grad():
            torch_out = action_head_step(*torch_inputs)
        hmonnx_out = run_hmonnx_step(hmonnx_session, step, args.hmonnx_device)
        step_reports.append(
            {
                "step": idx,
                "inputs": {
                    "conditioning_tokens": tensor_summary(step["conditioning_tokens"]),
                    "actions": tensor_summary(step["actions"]),
                    "state": tensor_summary(step["state"]),
                    "timesteps": tensor_summary(step["timesteps"]),
                },
                "diff": diff_summary(torch_out, hmonnx_out),
            }
        )

    max_abs = max(item["diff"]["max_abs_diff"] for item in step_reports)
    mean_abs = float(np.mean([item["diff"]["mean_abs_diff"] for item in step_reports]))
    report = {
        "model": args.model,
        "hmonnx": args.hmonnx,
        "capture_dir": args.capture_dir,
        "num_steps": len(step_reports),
        "summary": {
            "max_abs_diff_over_steps": max_abs,
            "mean_abs_diff_over_steps": mean_abs,
        },
        "steps": step_reports,
    }

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
