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
from lerobot.utils.constants import ACTION

DEFAULT_HMONNX = DEFAULT_ACTION_HEAD_HMONNX
DEFAULT_REPORT = output_str("action_head", "hmonnx_compare_libero_sweep_report.json")
DEFAULT_CAPTURE_DIR = output_str("action_head", "captured_steps_libero_sweep")


def log(message: str) -> None:
    print(f"[sweep] {message}", flush=True)


def parse_task_ids(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def install_capture(action_head: Any, captured: list[dict[str, Any]]):
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


def save_sample_steps(sample_dir: Path, steps: list[dict[str, torch.Tensor]]) -> None:
    sample_dir.mkdir(parents=True, exist_ok=True)
    for idx, item in enumerate(steps):
        torch.save({k: v for k, v in item.items() if v is not None}, sample_dir / f"denoise_{idx}.pt")


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
    parser.add_argument("--task-ids", default="0,1,2")
    parser.add_argument("--rollout-steps", type=int, default=2)
    args = parser.parse_args()

    set_default_libero_config_path()
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.manual_seed(23)

    from lerobot.envs import close_envs, make_env, make_env_pre_post_processors, preprocess_observation
    from lerobot.envs.configs import LiberoEnv
    from lerobot.policies import make_pre_post_processors
    from xhquant.api import HMONNXInference

    task_ids = parse_task_ids(args.task_ids)
    log(f"loading policy: {args.model}")
    policy = load_policy(args.model, args.device, disable_world_model=True)
    policy.eval()
    log("building policy pre/post processors")
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=args.model,
        preprocessor_overrides={"device_processor": {"device": str(policy.config.device)}},
    )
    action_head_step = ActionHeadStep(policy.model.action_model).eval().float().to(args.device)
    log(f"loading HMONNX: {args.hmonnx}")
    hmonnx_session = HMONNXInference(args.hmonnx)
    hmonnx_session.to(args.hmonnx_device)
    log(f"start sweep task_ids={task_ids}, rollout_steps={args.rollout_steps}")

    all_reports = []
    capture_root = Path(args.capture_dir)

    for task_id in task_ids:
        log(f"create LIBERO env task_id={task_id}")
        env_cfg = LiberoEnv(task="libero_10", task_ids=[task_id])
        envs = make_env(env_cfg, n_envs=1, use_async_envs=False)
        env = envs["libero_10"][task_id]
        env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy.config)
        policy.reset()
        log(f"reset env task_id={task_id}")
        observation, _info = env.reset(seed=[task_id])
        try:
            for env_step in range(args.rollout_steps):
                log(f"prepare observation task_id={task_id} env_step={env_step}")
                obs = preprocess_observation(observation)
                obs["task"] = list(env.call("task_description"))
                obs = env_preprocessor(obs)
                obs = preprocessor(obs)

                captured: list[dict[str, torch.Tensor]] = []
                original = install_capture(policy.model.action_model, captured)
                log(f"predict action chunk task_id={task_id} env_step={env_step}")
                with torch.no_grad():
                    action_chunk = policy.predict_action_chunk(obs)
                policy.model.action_model._build_inputs = original
                log(f"captured {len(captured)} denoise steps task_id={task_id} env_step={env_step}")

                sample_dir = capture_root / f"task_{task_id}" / f"env_step_{env_step}"
                save_sample_steps(sample_dir, captured)

                denoise_reports = []
                for denoise_idx, step in enumerate(captured):
                    log(f"compare denoise task_id={task_id} env_step={env_step} denoise_step={denoise_idx}")
                    torch_inputs = (
                        step["conditioning_tokens"].to(args.device),
                        step["actions"].to(args.device),
                        step["state"].to(args.device),
                        step["timesteps"].long().to(args.device),
                    )
                    with torch.no_grad():
                        torch_out = action_head_step(*torch_inputs)
                    hmonnx_out = run_hmonnx_step(hmonnx_session, step, args.hmonnx_device)
                    denoise_reports.append({
                        "denoise_step": denoise_idx,
                        "inputs": {k: tensor_summary(v) for k, v in step.items() if v is not None},
                        "diff": diff_summary(torch_out, hmonnx_out),
                    })

                action = action_chunk[:, 0, :]
                action = postprocessor(action)
                action_transition = env_postprocessor({ACTION: action})
                action_numpy = action_transition[ACTION].detach().cpu().numpy()
                log(f"env step task_id={task_id} env_step={env_step}")
                observation, reward, terminated, truncated, info = env.step(action_numpy)

                all_reports.append({
                    "task_id": task_id,
                    "env_step": env_step,
                    "reward": reward.tolist() if hasattr(reward, "tolist") else reward,
                    "terminated": terminated.tolist() if hasattr(terminated, "tolist") else terminated,
                    "truncated": truncated.tolist() if hasattr(truncated, "tolist") else truncated,
                    "capture_dir": str(sample_dir),
                    "denoise_steps": denoise_reports,
                })

                done = np.asarray(terminated) | np.asarray(truncated)
                if bool(done.all()):
                    log(f"task_id={task_id} finished at env_step={env_step}")
                    break
        finally:
            log(f"close LIBERO env task_id={task_id}")
            close_envs(envs)

    flat_diffs = [d["diff"] for sample in all_reports for d in sample["denoise_steps"]]
    report = {
        "model": args.model,
        "hmonnx": args.hmonnx,
        "source": "libero_sweep",
        "task_ids": task_ids,
        "rollout_steps": args.rollout_steps,
        "num_observations": len(all_reports),
        "num_denoise_steps": len(flat_diffs),
        "summary": {
            "max_abs_diff_over_steps": max(item["max_abs_diff"] for item in flat_diffs),
            "mean_abs_diff_over_steps": float(np.mean([item["mean_abs_diff"] for item in flat_diffs])),
            "allclose_5e-2_count": int(sum(item["allclose_rtol_5e-2_atol_5e-2"] for item in flat_diffs)),
        },
        "samples": all_reports,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    log(f"wrote report: {report_path}")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
