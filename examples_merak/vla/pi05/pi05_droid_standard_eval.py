#!/usr/bin/env python3
"""Evaluate PI0.5 on real episodes streamed from the standard DROID dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import load_dataset

from examples_merak.vla.pi05.pi05_droid_hmonnx_validation import (
    CompactHMONNX,
    cosine,
    load_config,
)


DEFAULT_REPO_ID = "lerobot/droid_1.0.1"
DEFAULT_REVISION = "v3.0"
EXTERIOR_KEY = "observation.images.exterior_1_left"
WRIST_KEY = "observation.images.wrist_left"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def first_instruction(row: dict[str, Any], episode_info: dict[str, Any]) -> str:
    candidates = [
        *(episode_info.get("tasks") or []),
        row.get("language_instruction"),
        row.get("language_instruction_2"),
        row.get("language_instruction_3"),
    ]
    for value in candidates:
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ValueError(f"DROID episode {row['episode_index']} has no language instruction")


def select_episode_ids(meta: Any, num_samples: int, horizon: int, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    selected = []
    tasks = set()
    for episode_id in rng.permutation(meta.total_episodes):
        episode_info = meta.episodes[int(episode_id)]
        task = next(
            (value.strip() for value in episode_info.get("tasks", []) if isinstance(value, str) and value.strip()),
            None,
        )
        if task is None or task.casefold() in tasks or int(episode_info["length"]) < horizon:
            continue
        selected.append(int(episode_id))
        tasks.add(task.casefold())
        if len(selected) == num_samples:
            return selected
    raise ValueError(f"Could not select {num_samples} distinct DROID tasks")


def load_episode_window(
    *,
    repo_id: str,
    revision: str,
    data_path: str,
    episode_id: int,
    start_frame: int,
    horizon: int,
) -> list[dict[str, Any]]:
    dataset = load_dataset(
        repo_id,
        split="train",
        streaming=True,
        data_files=data_path,
        revision=revision,
    )
    rows = []
    for row in dataset:
        row_episode = int(row["episode_index"])
        if row_episode < episode_id:
            continue
        if row_episode > episode_id:
            break
        frame_index = int(row["frame_index"])
        if start_frame <= frame_index < start_frame + horizon:
            rows.append(row)
            if len(rows) == horizon:
                break

    expected_frames = list(range(start_frame, start_frame + horizon))
    actual_frames = [int(row["frame_index"]) for row in rows]
    if actual_frames != expected_frames:
        raise ValueError(
            f"DROID episode {episode_id} action window is incomplete: expected {expected_frames}, got {actual_frames}"
        )
    return rows


def decode_observation_images(
    dataset: Any,
    episode_id: int,
    frame_index: int,
) -> dict[str, torch.Tensor]:
    episode_info = dataset.meta.episodes[episode_id]
    timestamps = {
        key: [float(episode_info[f"videos/{key}/from_timestamp"]) + frame_index / dataset.fps]
        for key in (EXTERIOR_KEY, WRIST_KEY)
    }
    return dataset._query_videos(timestamps, episode_id)


def action_array(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    return array


def to_droid_relative_actions(
    actions: np.ndarray,
    state: np.ndarray,
    action_names: list[str],
) -> np.ndarray:
    if actions.shape[-1] != len(action_names) or state.shape[-1] < actions.shape[-1]:
        raise ValueError(
            f"DROID action/state schema mismatch: actions={actions.shape}, state={state.shape}, names={action_names}"
        )
    relative = actions.copy()
    for index, name in enumerate(action_names):
        if "gripper" not in name.casefold():
            relative[..., index] -= state[index]
    return relative


def metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    if reference.shape != candidate.shape:
        raise ValueError(f"Action shape mismatch: {reference.shape} != {candidate.shape}")
    difference = candidate - reference
    return {
        "cosine": cosine(reference, candidate),
        "mae": float(np.abs(difference).mean()),
        "rmse": float(np.sqrt(np.square(difference).mean())),
        "max_abs": float(np.abs(difference).max()),
    }


def aggregate(records: list[dict[str, Any]], key: str) -> dict[str, Any]:
    result: dict[str, Any] = {"num_samples": len(records)}
    for metric_name in ("cosine", "mae", "rmse", "max_abs"):
        values = np.asarray([record[key][metric_name] for record in records], dtype=np.float64)
        result[metric_name] = {
            "mean": float(values.mean()),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return result


def parse_episode_ids(value: str | None) -> list[int] | None:
    if value is None:
        return None
    episode_ids = list(dict.fromkeys(int(item.strip()) for item in value.split(",") if item.strip()))
    if not episode_ids or any(episode_id < 0 for episode_id in episode_ids):
        raise ValueError("--episode-ids must contain non-negative integers")
    return episode_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--dataset-revision", default=DEFAULT_REVISION)
    parser.add_argument("--episode-ids", default=None, help="Comma-separated official DROID episode IDs.")
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=445)
    parser.add_argument("--frame-fraction", type=float, default=0.5)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--runtime-config-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--min-cosine-mean", type=float, default=0.99)
    parser.add_argument("--max-mae-mean", type=float, default=0.02)
    args = parser.parse_args()
    if args.num_samples <= 0:
        parser.error("--num-samples must be positive")
    if not 0.0 <= args.frame_fraction <= 1.0:
        parser.error("--frame-fraction must be between 0 and 1")
    return args


def main() -> int:
    from lerobot.datasets.streaming_dataset import StreamingLeRobotDataset
    from lerobot.datasets.video_utils import VideoDecoderCache
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.pi05 import PI05Policy
    from lerobot.utils.constants import ACTION

    args = parse_args()
    model_dir = args.model_dir.resolve()
    runtime_config_dir = args.runtime_config_dir.resolve()
    tokenizer_dir = args.tokenizer_dir.resolve()
    export_dir = args.export_dir.resolve()
    output = args.output.resolve()
    for path in (model_dir / "config.json", runtime_config_dir / "config.json", tokenizer_dir, export_dir):
        if not path.exists():
            raise FileNotFoundError(path)

    device_name = args.device
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)

    dataset = StreamingLeRobotDataset(
        args.dataset_repo_id,
        revision=args.dataset_revision,
        streaming=True,
        shuffle=False,
        max_num_shards=1,
    )
    dataset.video_decoder_cache = VideoDecoderCache()

    export_meta = json.loads((export_dir / "export_meta_info.json").read_text(encoding="utf-8"))
    action_horizon = int(export_meta["compact_prefix"]["action_horizon"])
    episode_ids = parse_episode_ids(args.episode_ids)
    if episode_ids is None:
        episode_ids = select_episode_ids(dataset.meta, args.num_samples, action_horizon, args.seed)

    config = load_config(model_dir, device_name, action_horizon, runtime_config_dir)
    policy = PI05Policy.from_pretrained(model_dir, config=config, strict=True).to(device).eval()
    preprocessor, postprocessor = make_pre_post_processors(
        config,
        pretrained_path=str(runtime_config_dir),
        preprocessor_overrides={
            "tokenizer_processor": {"tokenizer_name": str(tokenizer_dir)},
            "device_processor": {"device": device_name, "float_dtype": None},
        },
        postprocessor_overrides={"device_processor": {"device": "cpu", "float_dtype": None}},
    )
    hmonnx = CompactHMONNX(policy, export_dir, device)
    action_names = list(dataset.meta.info["features"]["action"]["names"]["axes"])

    records = []
    for episode_id in episode_ids:
        episode_info = dataset.meta.episodes[episode_id]
        episode_length = int(episode_info["length"])
        if episode_length < action_horizon:
            raise ValueError(f"DROID episode {episode_id} has {episode_length} frames, below horizon {action_horizon}")
        max_start = episode_length - action_horizon
        start_frame = min(int(max_start * args.frame_fraction), max_start)
        data_path = str(dataset.meta.get_data_file_path(episode_id))
        rows = load_episode_window(
            repo_id=args.dataset_repo_id,
            revision=args.dataset_revision,
            data_path=data_path,
            episode_id=episode_id,
            start_frame=start_frame,
            horizon=action_horizon,
        )
        first_row = rows[0]
        if not bool(first_row["is_episode_successful"]):
            raise ValueError(f"DROID episode {episode_id} is not marked successful")
        task = first_instruction(first_row, episode_info)
        images = decode_observation_images(dataset, episode_id, start_frame)
        state = np.asarray(first_row["observation.state"], dtype=np.float32)
        absolute_ground_truth = np.asarray([row["action"] for row in rows], dtype=np.float32)
        ground_truth = to_droid_relative_actions(absolute_ground_truth, state, action_names)
        sample = {
            "observation.images.base_0_rgb": images[EXTERIOR_KEY].float(),
            "observation.images.left_wrist_0_rgb": images[WRIST_KEY].float(),
            "observation.state": torch.from_numpy(state.copy()),
            "action": torch.zeros(config.output_features[ACTION].shape[0], dtype=torch.float32),
            "task": task,
        }
        batch = preprocessor(sample)
        noise_rng = np.random.default_rng(np.random.SeedSequence([args.seed, episode_id]))
        noise_array = noise_rng.standard_normal((action_horizon, int(config.max_action_dim)), dtype=np.float32)
        noise = torch.from_numpy(noise_array)[None].to(device)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        with torch.inference_mode():
            normalized_reference = policy.predict_action_chunk(batch, noise=noise.clone())
            reference = action_array(postprocessor(normalized_reference).detach().cpu().numpy())
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        reference_ms = (time.perf_counter() - started) * 1000.0

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        normalized_hmonnx, contract = hmonnx.predict(batch, noise.clone())
        hmonnx_actions = action_array(postprocessor(normalized_hmonnx).detach().cpu().numpy())
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        hmonnx_ms = (time.perf_counter() - started) * 1000.0

        record = {
            "episode_id": episode_id,
            "frame_index": start_frame,
            "task": task,
            "task_index": int(first_row["task_index"]),
            "building": str(first_row["building"]),
            "collector_id": str(first_row["collector_id"]),
            "is_episode_successful": True,
            "data_path": data_path,
            "state_sha256": sha256_bytes(np.asarray(first_row["observation.state"], dtype=np.float32).tobytes()),
            "dataset_absolute_action_sha256": sha256_bytes(absolute_ground_truth.tobytes()),
            "dataset_relative_action_sha256": sha256_bytes(ground_truth.tobytes()),
            "noise_sha256": sha256_bytes(noise_array.tobytes()),
            "action_shape": list(reference.shape),
            "runtime_contract": contract,
            "latency_ms": {"lerobot": reference_ms, "hmonnx": hmonnx_ms},
            "hmonnx_vs_lerobot": metrics(reference, hmonnx_actions),
            "lerobot_vs_dataset_relative_action": metrics(ground_truth, reference),
            "hmonnx_vs_dataset_relative_action": metrics(ground_truth, hmonnx_actions),
        }
        records.append(record)
        print(
            json.dumps(
                {
                    "episode_id": episode_id,
                    "frame_index": start_frame,
                    "task": task,
                    **record["hmonnx_vs_lerobot"],
                }
            ),
            flush=True,
        )

    parity = aggregate(records, "hmonnx_vs_lerobot")
    acceptance = {
        "min_cosine_mean": args.min_cosine_mean,
        "max_mae_mean": args.max_mae_mean,
        "passed": (parity["cosine"]["mean"] >= args.min_cosine_mean and parity["mae"]["mean"] <= args.max_mae_mean),
    }
    report = {
        "schema_version": 1,
        "dataset": {
            "repo_id": args.dataset_repo_id,
            "requested_revision": args.dataset_revision,
            "resolved_revision": dataset.revision,
            "total_episodes": dataset.num_episodes,
            "total_frames": dataset.num_frames,
            "fps": dataset.fps,
            "episode_ids": episode_ids,
            "selection_seed": args.seed,
            "frame_fraction": args.frame_fraction,
        },
        "model_dir": str(model_dir),
        "runtime_config_dir": str(runtime_config_dir),
        "tokenizer_dir": str(tokenizer_dir),
        "export_dir": str(export_dir),
        "action_horizon": action_horizon,
        "aggregates": {
            "hmonnx_vs_lerobot": parity,
            "lerobot_vs_dataset_relative_action": aggregate(records, "lerobot_vs_dataset_relative_action"),
            "hmonnx_vs_dataset_relative_action": aggregate(records, "hmonnx_vs_dataset_relative_action"),
        },
        "acceptance": acceptance,
        "records": records,
        "note": (
            "DROID v3 absolute joint targets are converted to OpenPI-style deltas for the first "
            "seven joints while gripper remains absolute. Dataset-action metrics are offline "
            "diagnostics only and do not replace closed-loop robot success evaluation."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output}")
    return 0 if acceptance["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
