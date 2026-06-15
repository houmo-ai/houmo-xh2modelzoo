# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0

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
from common.inspect_policy import DEFAULT_MODEL, load_policy
from common.paths import DEFAULT_ACTION_HEAD_HMONNX, output_str, set_default_libero_config_path

DEFAULT_HMONNX = DEFAULT_ACTION_HEAD_HMONNX
DEFAULT_REPORT = output_str("action_head", "hmonnx_compare_libero_reset_report.json")
DEFAULT_CAPTURE_DIR = output_str("action_head", "captured_steps_libero_reset")


def install_capture(action_head: Any, captured: list[dict[str, torch.Tensor]]):
    original = action_head._build_inputs

    def wrapper(self, conditioning_tokens, actions, state, timesteps):
        captured.append({
            "conditioning_tokens": conditioning_tokens.detach().float().cpu(),
            "actions": actions.detach().float().cpu(),
            "state": state.detach().float().cpu() if state is not None else None,
            "timesteps": timesteps.detach().cpu(),
        })
        return original(conditioning_tokens, actions, state, timesteps)

    action_head._build_inputs = MethodType(wrapper, action_head)
    return original


def save_steps(captured: list[dict[str, torch.Tensor]], capture_dir: Path) -> None:
    capture_dir.mkdir(parents=True, exist_ok=True)
    for idx, item in enumerate(captured):
        torch.save({k: v for k, v in item.items() if v is not None}, capture_dir / f"step_{idx}.pt")


def capture_libero_steps(model_path: str, device: str, capture_dir: Path, task_id: int):
    from lerobot.envs import close_envs, make_env, make_env_pre_post_processors, preprocess_observation
    from lerobot.envs.configs import LiberoEnv
    from lerobot.policies import make_pre_post_processors

    env_cfg = LiberoEnv(task="libero_10", task_ids=[task_id])
    envs = make_env(env_cfg, n_envs=1, use_async_envs=False)
    env = envs["libero_10"][task_id]

    policy = load_policy(model_path, device, disable_world_model=True)
    policy.eval()
    policy.reset()

    preprocessor, _ = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=model_path,
        preprocessor_overrides={"device_processor": {"device": str(policy.config.device)}},
    )
    env_preprocessor, _ = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy.config)

    captured: list[dict[str, torch.Tensor]] = []
    action_head = policy.model.action_model
    original = install_capture(action_head, captured)
    try:
        observation, _info = env.reset(seed=[0])
        observation = preprocess_observation(observation)
        observation["task"] = list(env.call("task_description"))
        observation = env_preprocessor(observation)
        observation = preprocessor(observation)
        with torch.no_grad():
            policy.predict_action_chunk(observation)
    finally:
        action_head._build_inputs = original
        close_envs(envs)

    save_steps(captured, capture_dir)
    return policy.model.action_model, captured


def run_hmonnx_step(session: Any, step: dict[str, torch.Tensor], device: str) -> torch.Tensor:
    inputs = [step["conditioning_tokens"].half(), step["actions"].half(), step["state"].half(), step["timesteps"].int()]
    inputs = [item.to(device) for item in inputs]
    with torch.no_grad():
        return session(*inputs)


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
    parser.add_argument("--libero-task-id", type=int, default=0)
    args = parser.parse_args()

    set_default_libero_config_path()
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.manual_seed(17)

    action_head, captured = capture_libero_steps(args.model, args.device, Path(args.capture_dir), args.libero_task_id)
    action_head_step = ActionHeadStep(action_head).eval().float().to(args.device)

    from xhquant.api import HMONNXInference
    hmonnx_session = HMONNXInference(args.hmonnx)
    hmonnx_session.to(args.hmonnx_device)

    steps = []
    for idx, step in enumerate(captured):
        torch_inputs = (step["conditioning_tokens"].to(args.device), step["actions"].to(args.device), step["state"].to(args.device), step["timesteps"].long().to(args.device))
        with torch.no_grad():
            torch_out = action_head_step(*torch_inputs)
        hmonnx_out = run_hmonnx_step(hmonnx_session, step, args.hmonnx_device)
        steps.append({
            "step": idx,
            "inputs": {k: tensor_summary(v) for k, v in step.items() if v is not None},
            "diff": diff_summary(torch_out, hmonnx_out),
        })

    report = {
        "model": args.model,
        "hmonnx": args.hmonnx,
        "source": "libero_reset",
        "libero_task_id": args.libero_task_id,
        "capture_dir": args.capture_dir,
        "num_steps": len(steps),
        "summary": {
            "max_abs_diff_over_steps": max(item["diff"]["max_abs_diff"] for item in steps),
            "mean_abs_diff_over_steps": float(np.mean([item["diff"]["mean_abs_diff"] for item in steps])),
        },
        "steps": steps,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
