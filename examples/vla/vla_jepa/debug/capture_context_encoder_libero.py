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

import torch

from debug.capture_context_encoder import capture_context, detach_to_cpu, summarize_value
from common.inspect_policy import DEFAULT_MODEL, load_policy
from common.paths import output_str, set_default_libero_config_path

DEFAULT_OUT_DIR = output_str("context_encoder", "libero")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--task", default="libero_10")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--sample-name", default="task0_seed1000_context_sample")
    args = parser.parse_args()

    set_default_libero_config_path()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.manual_seed(args.seed)

    from lerobot.envs import close_envs, make_env, make_env_pre_post_processors, preprocess_observation
    from lerobot.envs.configs import LiberoEnv
    from lerobot.policies import make_pre_post_processors

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_path = out_dir / f"{args.sample_name}.pt"
    report_path = out_dir / f"{args.sample_name}_report.json"

    policy = load_policy(args.model, args.device, disable_world_model=True)
    policy.eval()
    policy.reset()

    preprocessor, _postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=args.model,
        preprocessor_overrides={"device_processor": {"device": str(policy.config.device)}},
    )
    env_cfg = LiberoEnv(task=args.task, task_ids=[args.task_id])
    env_preprocessor, _env_postprocessor = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy.config)
    envs = make_env(env_cfg, n_envs=1, use_async_envs=False)
    env = envs[args.task][args.task_id]

    try:
        observation, _info = env.reset(seed=[args.seed])
        observation = preprocess_observation(observation)
        observation["task"] = list(env.call("task_description"))
        observation = env_preprocessor(observation)
        batch = preprocessor(observation)
        captured = capture_context(policy, batch)
    finally:
        close_envs(envs)

    serializable = detach_to_cpu(captured)
    torch.save(serializable, sample_path)

    report = {
        "model": args.model,
        "device": args.device,
        "task": args.task,
        "task_id": args.task_id,
        "seed": args.seed,
        "sample_path": str(sample_path),
        "batch": summarize_value(batch),
        "prepared_examples": captured["prepared_examples_summary"],
        "qwen_inputs": summarize_value(captured["qwen_inputs"]),
        "embodied_indices": summarize_value(captured["embodied_indices"]),
        "last_hidden": summarize_value(captured["last_hidden"]),
        "conditioning_tokens": summarize_value(captured["conditioning_tokens"]),
        "state": summarize_value(captured["state"]),
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
