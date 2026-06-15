# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0

"""Run LIBERO rollout eval with VLA-JEPA ActionHead HMONNX plugged into policy.

This is an end-to-end task-level evaluator: environment reset/step,
pre/post-processing, Qwen/context encoding, action queue, and LIBERO success
metrics all run through the normal LeRobot path. The optional quantized boundary
is the ActionHead denoise step HMONNX, which replaces the PyTorch denoise body
inside ``VLAJEPAActionHead.predict_action``.
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
from types import MethodType
from typing import Any

import torch

from common.inspect_policy import DEFAULT_MODEL, load_policy
from common.paths import DEFAULT_ACTION_HEAD_HMONNX, output_str, set_default_libero_config_path

DEFAULT_HMONNX = DEFAULT_ACTION_HEAD_HMONNX
DEFAULT_REPORT = output_str("eval", "action_head_hmonnx_libero_eval_report.json")


def log(message: str) -> None:
    print(f"[eval_action_head_hmonnx] {message}", flush=True)


def configure_hmonnx_progress(enabled: bool) -> None:
    if not enabled:
        return
    import xhquant.xhonnxruntime.config as hmonnx_config

    hmonnx_config.disable_progress = False
    hmonnx_config.verbose_progress = True
    log("enabled HMONNXInference node progress")


def tensor_summary(name: str, tensor: torch.Tensor) -> str:
    return f"{name}:shape={tuple(tensor.shape)},dtype={tensor.dtype},device={tensor.device}"


def parse_task_ids(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def as_jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): as_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [as_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [as_jsonable(v) for v in value]
    return value


class ActionHeadHMONNXRuntime:
    def __init__(self, hmonnx_path: str, hmonnx_device: str, trace: bool = False) -> None:
        from xhquant.api import HMONNXInference

        self.hmonnx_path = hmonnx_path
        self.hmonnx_device = hmonnx_device
        self.trace = trace
        self.session = HMONNXInference(hmonnx_path)
        self.session.to(hmonnx_device)
        self.num_predict_calls = 0
        self.num_denoise_steps = 0
        self.input_schema = [
            {"name": item.name, "dtype": str(item.dtype), "shape": [int(dim) for dim in item.shape]}
            for item in self.session.inputs
        ]
        if self.trace:
            log(f"[hmonnx_trace] action session ready path={self.hmonnx_path} inputs={self.input_schema}")

    def step(
        self,
        conditioning_tokens: torch.Tensor,
        actions: torch.Tensor,
        state: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        input_by_name = {
            "conditioning_tokens": conditioning_tokens,
            "actions": actions,
            "state": state,
            "timesteps": timesteps,
        }
        hmonnx_inputs = []
        for input_info in self.session.inputs:
            tensor = input_by_name[input_info.name].detach().to(device=self.hmonnx_device, dtype=input_info.dtype)
            hmonnx_inputs.append(tensor)
        if self.trace:
            input_text = ", ".join(
                tensor_summary(input_info.name, tensor)
                for input_info, tensor in zip(self.session.inputs, hmonnx_inputs, strict=True)
            )
            log(f"[hmonnx_trace] action step={self.num_denoise_steps + 1} path={self.hmonnx_path} inputs=[{input_text}]")
        with torch.no_grad():
            output = self.session(*hmonnx_inputs)
        if isinstance(output, (tuple, list)):
            output = output[0]
        if self.trace:
            log(f"[hmonnx_trace] action output={tensor_summary('pred_velocity', output)}")
        self.num_denoise_steps += 1
        return output

    def stats(self) -> dict[str, Any]:
        return {
            "hmonnx_path": self.hmonnx_path,
            "hmonnx_device": self.hmonnx_device,
            "trace": self.trace,
            "input_schema": self.input_schema,
            "num_predict_calls": self.num_predict_calls,
            "num_denoise_steps": self.num_denoise_steps,
        }


def install_action_head_hmonnx(action_head: Any, runtime: ActionHeadHMONNXRuntime) -> None:
    def predict_action_hmonnx(self: Any, conditioning_tokens: torch.Tensor, state: torch.Tensor | None = None) -> torch.Tensor:
        if state is None:
            raise ValueError("ActionHead HMONNX export expects non-None state input")
        runtime.num_predict_calls += 1
        if runtime.trace:
            log(
                "[hmonnx_trace] action predict="
                f"{runtime.num_predict_calls} path={runtime.hmonnx_path} "
                f"{tensor_summary('conditioning_tokens', conditioning_tokens)} "
                f"{tensor_summary('state', state)}"
            )
        batch_size = conditioning_tokens.shape[0]
        actions = torch.randn(
            batch_size,
            self.action_horizon,
            self.config.action_dim,
            dtype=conditioning_tokens.dtype,
            device=conditioning_tokens.device,
        )
        dt = 1.0 / max(self.num_inference_timesteps, 1)
        for step in range(self.num_inference_timesteps):
            t_cont = step / float(max(self.num_inference_timesteps, 1))
            t_value = int(t_cont * self.config.action_num_timestep_buckets)
            timesteps = torch.full((batch_size,), t_value, device=conditioning_tokens.device, dtype=torch.long)
            pred_velocity = runtime.step(conditioning_tokens, actions, state, timesteps)
            pred_velocity = pred_velocity.to(device=actions.device, dtype=actions.dtype)
            actions = actions + dt * pred_velocity
        return actions

    action_head.predict_action = MethodType(predict_action_hmonnx, action_head)


def build_policy_and_processors(args: argparse.Namespace):
    from lerobot.envs import make_env_pre_post_processors
    from lerobot.envs.configs import LiberoEnv
    from lerobot.policies import make_pre_post_processors

    policy = load_policy(args.model, args.device, disable_world_model=True)
    policy.eval()

    runtime = None
    if args.mode == "action_head_hmonnx":
        log(f"loading ActionHead HMONNX: {args.hmonnx}")
        runtime = ActionHeadHMONNXRuntime(args.hmonnx, args.hmonnx_device, trace=args.trace_hmonnx)
        install_action_head_hmonnx(policy.model.action_model, runtime)

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=args.model,
        preprocessor_overrides={"device_processor": {"device": str(policy.config.device)}},
    )
    env_cfg = LiberoEnv(task=args.task, task_ids=parse_task_ids(args.task_ids))
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy.config)
    return policy, env_cfg, env_preprocessor, env_postprocessor, preprocessor, postprocessor, runtime


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--mode", choices=("original", "action_head_hmonnx"), default="action_head_hmonnx")
    parser.add_argument("--hmonnx", default=DEFAULT_HMONNX)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--hmonnx-device", default="cuda:0")
    parser.add_argument("--task", default="libero_10")
    parser.add_argument("--task-ids", default="0")
    parser.add_argument("--n-episodes", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--max-episodes-rendered", type=int, default=0)
    parser.add_argument("--report", default=DEFAULT_REPORT)
    parser.add_argument("--trace-hmonnx", action="store_true")
    parser.add_argument("--trace-hmonnx-progress", action="store_true")
    args = parser.parse_args()

    set_default_libero_config_path()
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    configure_hmonnx_progress(args.trace_hmonnx_progress)
    torch.manual_seed(args.seed)

    from lerobot.envs import close_envs, make_env
    from lerobot.scripts.lerobot_eval import eval_policy_all

    start = time.time()
    log(f"loading policy mode={args.mode}: {args.model}")
    policy, env_cfg, env_preprocessor, env_postprocessor, preprocessor, postprocessor, runtime = build_policy_and_processors(args)

    log(f"creating env task={args.task} task_ids={args.task_ids} batch_size={args.batch_size}")
    envs = make_env(env_cfg, n_envs=args.batch_size, use_async_envs=False)
    try:
        log("starting eval rollout")
        info = eval_policy_all(
            envs=envs,
            policy=policy,
            env_preprocessor=env_preprocessor,
            env_postprocessor=env_postprocessor,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            n_episodes=args.n_episodes,
            max_episodes_rendered=args.max_episodes_rendered,
            videos_dir=Path(args.report).parent / "videos" if args.max_episodes_rendered > 0 else None,
            start_seed=args.seed,
            max_parallel_tasks=1,
        )
    finally:
        close_envs(envs)

    report = {
        "model": args.model,
        "mode": args.mode,
        "task": args.task,
        "task_ids": parse_task_ids(args.task_ids),
        "n_episodes_per_task": args.n_episodes,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "device": args.device,
        "hmonnx_device": args.hmonnx_device,
        "runtime": runtime.stats() if runtime is not None else None,
        "elapsed_s": time.time() - start,
        "eval": as_jsonable(info),
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    log(f"wrote report: {report_path}")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
