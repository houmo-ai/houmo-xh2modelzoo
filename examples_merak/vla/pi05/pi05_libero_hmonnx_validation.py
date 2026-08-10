#!/usr/bin/env python3
"""Validate FP16 PI0.5 LIBERO HMONNX against LeRobot on real simulator observations."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path
from types import MethodType
from typing import Any

import numpy as np
import torch

from examples_merak.vla.pi05.pi05_droid_fae_pipeline import verify_masked_add_graph
from examples_merak.vla.pi05.pi05_droid_hmonnx_validation import (
    HMONNX_RUNTIME_CHOICES,
    CompactHMONNX,
    cosine,
    load_config,
    sha256_file,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_DIR = REPO_ROOT / "weights/pi05_libero_finetuned"
DEFAULT_TOKENIZER_DIR = REPO_ROOT / "weights/paligemma-3b-pt-224"
DEFAULT_EXPORT_DIR = REPO_ROOT / "work_dirs/pi05_libero_lerobot_h50_compact_maskadd2_XH2a"


def _parse_task_ids(value: str | None) -> list[int] | None:
    if value is None:
        return None
    task_ids = list(dict.fromkeys(int(item.strip()) for item in value.split(",") if item.strip()))
    if not task_ids or any(task_id < 0 for task_id in task_ids):
        raise ValueError("--task-ids must contain non-negative integers")
    return task_ids


def _action_array(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    return array


def _metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    if reference.shape != candidate.shape:
        raise ValueError(f"Action shape mismatch: {reference.shape} != {candidate.shape}")
    difference = candidate - reference
    return {
        "cosine": cosine(reference, candidate),
        "mae": float(np.abs(difference).mean()),
        "rmse": float(np.sqrt(np.square(difference).mean())),
        "max_abs": float(np.abs(difference).max()),
    }


def _aggregate(records: list[dict[str, Any]], key: str) -> dict[str, Any]:
    result: dict[str, Any] = {"num_samples": len(records)}
    for metric in ("cosine", "mae", "rmse", "max_abs"):
        values = np.asarray([record[key][metric] for record in records], dtype=np.float64)
        result[metric] = {
            "mean": float(values.mean()),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return result


def _graph_paths(export_dir: Path) -> dict[str, Path]:
    return {
        "vision": export_dir / "Vision/hmonnx/vision_XH2a_w8a8h1_sefp.onnx",
        "action_in": export_dir / "Other/hmonnx/action_in_proj_XH2a_w8a8h1_sefp.onnx",
        "action_out": export_dir / "Other/hmonnx/action_out_proj_XH2a_w8a8h1_sefp.onnx",
        "time_mlp": export_dir / "Other/hmonnx/time_mlp_XH2a_w8a8h1_sefp.onnx",
        "gemma_prefill": (export_dir / "Gemma2B/prefill_onnx/pi05_gemma_2b_XH2a_w8a8h1_sefp_prefill.onnx"),
        "gemma_decode": (export_dir / "Gemma2B/decode_onnx/pi05_gemma_2b_XH2a_w8a8h1_sefp_decode.onnx"),
        "expert_prefill": (
            export_dir / "GemmaExpert/prefill_onnx/pi05_gemma_expert_300m_XH2a_w8a8h1_sefp_prefill.onnx"
        ),
        "expert_decode": (export_dir / "GemmaExpert/decode_onnx/pi05_gemma_expert_300m_XH2a_w8a8h1_sefp_decode.onnx"),
    }


def _verify_fp16_graph(path: Path) -> dict[str, Any]:
    import onnx

    model = onnx.load(str(path), load_external_data=False)
    io_dtypes = {}
    tensor_types: list[tuple[str, int]] = []
    for kind, values in (
        ("input", model.graph.input),
        ("output", model.graph.output),
        ("value_info", model.graph.value_info),
    ):
        for value in values:
            if not value.type.HasField("tensor_type"):
                continue
            elem_type = value.type.tensor_type.elem_type
            tensor_types.append((f"{kind}:{value.name}", elem_type))
            if kind != "value_info":
                io_dtypes[value.name] = onnx.TensorProto.DataType.Name(elem_type)
    tensor_types.extend((f"initializer:{value.name}", value.data_type) for value in model.graph.initializer)
    for node in model.graph.node:
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.TENSOR:
                tensor_types.append((f"attribute:{node.name}:{attribute.name}", attribute.t.data_type))
            elif attribute.type == onnx.AttributeProto.TENSORS:
                tensor_types.extend(
                    (f"attribute:{node.name}:{attribute.name}", tensor.data_type) for tensor in attribute.tensors
                )

    bfloat16_tensors = [name for name, elem_type in tensor_types if elem_type == onnx.TensorProto.BFLOAT16]
    if bfloat16_tensors:
        raise ValueError(f"{path} contains BF16 tensors: {bfloat16_tensors[:8]}")

    floating_types = {
        onnx.TensorProto.FLOAT,
        onnx.TensorProto.FLOAT16,
        onnx.TensorProto.DOUBLE,
        onnx.TensorProto.BFLOAT16,
    }
    invalid_floating_io = [
        name
        for name, dtype_name in io_dtypes.items()
        if onnx.TensorProto.DataType.Value(dtype_name) in floating_types and dtype_name != "FLOAT16"
    ]
    if invalid_floating_io:
        raise ValueError(f"{path} has non-FP16 floating I/O: {invalid_floating_io}")

    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "io_dtypes": io_dtypes,
        "bfloat16_tensor_count": 0,
    }


def _verify_export(export_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    meta_path = export_dir / "export_meta_info.json"
    if not meta_path.is_file():
        raise FileNotFoundError(meta_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    expected_contract = {
        "selected_image_indices": [0, 1],
        "text_max_length": 200,
        "prefix_sequence_length": 712,
        "action_horizon": 50,
        "cache_length": 1024,
    }
    if meta.get("compact_prefix") != expected_contract:
        raise ValueError(
            f"LIBERO compact-prefix contract mismatch: {meta.get('compact_prefix')} != {expected_contract}"
        )
    if set(meta.get("components", {})) != {"vision", "other", "gemma", "expert"}:
        raise ValueError(f"Incomplete PI0.5 export components: {meta.get('components')}")

    graph_paths = _graph_paths(export_dir)
    for path in graph_paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    dtype_checks = {name: _verify_fp16_graph(path) for name, path in graph_paths.items()}

    graph_checks = {
        name: verify_masked_add_graph(graph_paths[name], num_hidden_layers=18)
        for name in ("gemma_prefill", "gemma_decode", "expert_prefill", "expert_decode")
    }
    golden_dirs = {
        "vision": graph_paths["vision"].parent / "golden",
        "action_in": graph_paths["action_in"].parent / "golden/action_in_proj",
        "action_out": graph_paths["action_out"].parent / "golden/action_out_proj",
        "time_mlp": graph_paths["time_mlp"].parent / "golden/time_mlp",
        "gemma_prefill": graph_paths["gemma_prefill"].parent / "golden",
        "gemma_decode": graph_paths["gemma_decode"].parent / "golden",
        "expert_prefill": graph_paths["expert_prefill"].parent / "golden",
        "expert_decode": graph_paths["expert_decode"].parent / "golden",
    }
    for path in golden_dirs.values():
        if not path.is_dir() or not any(path.iterdir()):
            raise FileNotFoundError(f"Missing HMONNX golden data: {path}")

    return meta, {
        "dtype_checks": dtype_checks,
        "masked_add_checks": graph_checks,
        "golden_dirs": {name: str(path) for name, path in golden_dirs.items()},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--tokenizer-dir", type=Path, default=DEFAULT_TOKENIZER_DIR)
    parser.add_argument("--export-dir", type=Path, default=DEFAULT_EXPORT_DIR)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--suite", default="libero_object")
    parser.add_argument("--task-ids", default=None, help="Comma-separated task IDs; default is all tasks.")
    parser.add_argument("--samples-per-task", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--noise-seed", type=int, default=20260810)
    parser.add_argument("--closed-loop-task-id", type=int, default=None)
    parser.add_argument("--closed-loop-episodes", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--hmonnx-runtime",
        choices=HMONNX_RUNTIME_CHOICES,
        default="v2-cuda-graph",
    )
    parser.add_argument("--min-cosine-mean", type=float, default=0.99)
    parser.add_argument("--max-mae-mean", type=float, default=0.02)
    args = parser.parse_args()
    if args.samples_per_task <= 0:
        parser.error("--samples-per-task must be positive")
    if args.closed_loop_episodes < 0:
        parser.error("--closed-loop-episodes must be non-negative")
    if args.closed_loop_episodes > 0 and args.closed_loop_task_id is None:
        parser.error("--closed-loop-task-id is required when --closed-loop-episodes is positive")
    return args


def main() -> int:
    from lerobot.envs.configs import LiberoEnv
    from lerobot.envs.factory import make_env, make_env_pre_post_processors
    from lerobot.envs.utils import add_envs_task, close_envs, preprocess_observation
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.pi05 import PI05Policy
    from lerobot.scripts.lerobot_eval import eval_policy

    args = parse_args()
    model_dir = args.model_dir.expanduser().resolve()
    tokenizer_dir = args.tokenizer_dir.expanduser().resolve()
    export_dir = args.export_dir.expanduser().resolve()
    results_dir = (
        args.results_dir.expanduser().resolve() if args.results_dir is not None else export_dir / "validation_results"
    )
    for path in (model_dir / "config.json", model_dir / "model.safetensors", tokenizer_dir, export_dir):
        if not path.exists():
            raise FileNotFoundError(path)
    results_dir.mkdir(parents=True, exist_ok=True)
    records_path = results_dir / "libero_h50_fp_vs_hmonnx.jsonl"
    summary_path = results_dir / "libero_h50_summary.json"

    export_meta, artifact_checks = _verify_export(export_dir)
    horizon = int(export_meta["compact_prefix"]["action_horizon"])
    device_name = args.device if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    config = load_config(model_dir, device_name, horizon)
    policy = PI05Policy.from_pretrained(model_dir, config=config, strict=True).to(device).eval()
    preprocessor, postprocessor = make_pre_post_processors(
        config,
        pretrained_path=str(model_dir),
        preprocessor_overrides={
            "tokenizer_processor": {"tokenizer_name": str(tokenizer_dir)},
            "device_processor": {"device": device_name, "float_dtype": None},
        },
        postprocessor_overrides={"device_processor": {"device": "cpu", "float_dtype": None}},
    )

    requested_task_ids = _parse_task_ids(args.task_ids)
    env_cfg = LiberoEnv(task=args.suite, task_ids=requested_task_ids)
    envs = make_env(env_cfg, n_envs=1, use_async_envs=False)
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=config)
    hmonnx = CompactHMONNX(policy, export_dir, device, runtime_mode=args.hmonnx_runtime)
    suite_envs = envs.get(args.suite)
    if not suite_envs:
        close_envs(envs)
        raise ValueError(f"LIBERO suite not created: {args.suite}")
    task_ids = requested_task_ids or sorted(suite_envs)
    missing_task_ids = [task_id for task_id in task_ids if task_id not in suite_envs]
    if missing_task_ids:
        close_envs(envs)
        raise ValueError(f"LIBERO task IDs not created: {missing_task_ids}")

    records: list[dict[str, Any]] = []
    closed_loop = None
    try:
        with records_path.open("w", encoding="utf-8") as records_file:
            for task_id in task_ids:
                env = suite_envs[task_id]
                for sample_index in range(args.samples_per_task):
                    env_seed = args.seed + sample_index
                    raw_observation, _ = env.reset(seed=[env_seed])
                    observation = preprocess_observation(raw_observation)
                    observation = add_envs_task(env, observation)
                    observation = env_preprocessor(observation)
                    batch = preprocessor(observation)

                    noise_rng = np.random.default_rng(np.random.SeedSequence([args.noise_seed, task_id, sample_index]))
                    noise_array = noise_rng.standard_normal((1, horizon, int(config.max_action_dim)), dtype=np.float32)
                    noise = torch.from_numpy(noise_array.copy()).to(device)

                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    started = time.perf_counter()
                    with torch.inference_mode():
                        normalized_reference = policy.predict_action_chunk(batch, noise=noise.clone())
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    reference_ms = (time.perf_counter() - started) * 1000.0

                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    started = time.perf_counter()
                    normalized_hmonnx, contract = hmonnx.predict(batch, noise.clone())
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    hmonnx_ms = (time.perf_counter() - started) * 1000.0

                    normalized_reference_array = _action_array(normalized_reference.detach().cpu().numpy())
                    normalized_hmonnx_array = _action_array(normalized_hmonnx.detach().cpu().numpy())
                    reference_actions = _action_array(postprocessor(normalized_reference).detach().cpu().numpy())
                    hmonnx_actions = _action_array(postprocessor(normalized_hmonnx).detach().cpu().numpy())
                    expected_shape = (horizon, int(config.output_features["action"].shape[0]))
                    if reference_actions.shape != expected_shape or hmonnx_actions.shape != expected_shape:
                        raise ValueError(
                            f"Task {task_id} action shape mismatch: reference={reference_actions.shape}, "
                            f"hmonnx={hmonnx_actions.shape}, expected={expected_shape}"
                        )

                    record = {
                        "sample_id": len(records),
                        "suite": args.suite,
                        "task_id": task_id,
                        "task": observation["task"][0],
                        "sample_index": sample_index,
                        "env_seed": env_seed,
                        "noise_seed": args.noise_seed,
                        "noise_sha256": hashlib.sha256(noise_array.tobytes()).hexdigest(),
                        "action_shape": list(expected_shape),
                        "valid_prefix_length": contract["valid_prefix_length"],
                        "valid_language_length": contract["valid_language_length"],
                        "selected_image_indices": contract["selected_image_indices"],
                        "vision_runs": contract["vision_runs"],
                        "latency_ms": {"lerobot": reference_ms, "hmonnx": hmonnx_ms},
                        "normalized_hmonnx_vs_lerobot": _metrics(normalized_reference_array, normalized_hmonnx_array),
                        "hmonnx_vs_lerobot": _metrics(reference_actions, hmonnx_actions),
                        "lerobot_actions": reference_actions.tolist(),
                        "hmonnx_actions": hmonnx_actions.tolist(),
                    }
                    records.append(record)
                    records_file.write(json.dumps(record) + "\n")
                    records_file.flush()
                    print(
                        json.dumps(
                            {
                                "sample_id": record["sample_id"],
                                "task_id": task_id,
                                "valid_prefix_length": contract["valid_prefix_length"],
                                "cosine": record["hmonnx_vs_lerobot"]["cosine"],
                                "mae": record["hmonnx_vs_lerobot"]["mae"],
                                "hmonnx_ms": hmonnx_ms,
                            }
                        ),
                        flush=True,
                    )
        if args.closed_loop_episodes > 0:
            closed_loop_task_id = int(args.closed_loop_task_id)
            if closed_loop_task_id not in suite_envs:
                raise ValueError(f"LIBERO closed-loop task ID not created: {closed_loop_task_id}")

            def predict_action_chunk_with_hmonnx(self, batch, **kwargs):
                noise = kwargs.get("noise")
                if noise is None:
                    batch_size = int(batch["observation.language.tokens"].shape[0])
                    noise = self.model.sample_noise(
                        (batch_size, horizon, int(self.config.max_action_dim)),
                        batch["observation.language.tokens"].device,
                    )
                return hmonnx.predict(batch, noise)[0]

            policy.predict_action_chunk = MethodType(predict_action_chunk_with_hmonnx, policy)
            try:
                closed_loop = eval_policy(
                    env=suite_envs[closed_loop_task_id],
                    policy=policy,
                    env_preprocessor=env_preprocessor,
                    env_postprocessor=env_postprocessor,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    n_episodes=args.closed_loop_episodes,
                    max_episodes_rendered=1,
                    videos_dir=results_dir / f"closed_loop_task_{closed_loop_task_id}",
                    return_episode_data=False,
                    start_seed=args.seed,
                )
            finally:
                del policy.predict_action_chunk
    finally:
        close_envs(envs)

    normalized_aggregate = _aggregate(records, "normalized_hmonnx_vs_lerobot")
    action_aggregate = _aggregate(records, "hmonnx_vs_lerobot")
    per_task = {}
    grouped_records: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped_records[int(record["task_id"])].append(record)
    for task_id, task_records in sorted(grouped_records.items()):
        per_task[str(task_id)] = {
            "task": task_records[0]["task"],
            "aggregate": _aggregate(task_records, "hmonnx_vs_lerobot"),
        }

    valid_prefix_lengths = [int(record["valid_prefix_length"]) for record in records]
    valid_language_lengths = [int(record["valid_language_length"]) for record in records]
    reference_latencies = np.asarray([record["latency_ms"]["lerobot"] for record in records], dtype=np.float64)
    hmonnx_latencies = np.asarray([record["latency_ms"]["hmonnx"] for record in records], dtype=np.float64)
    acceptance = {
        "min_cosine_mean": args.min_cosine_mean,
        "max_mae_mean": args.max_mae_mean,
        "passed": (
            action_aggregate["cosine"]["mean"] >= args.min_cosine_mean
            and action_aggregate["mae"]["mean"] <= args.max_mae_mean
        ),
    }
    summary = {
        "schema_version": 1,
        "model": {
            "path": str(model_dir),
            "weights_sha256": sha256_file(model_dir / "model.safetensors"),
            "reference_dtype": str(next(policy.parameters()).dtype),
        },
        "hmonnx": {
            "path": str(export_dir),
            "runtime": args.hmonnx_runtime,
            "floating_io_dtype": "torch.float16",
            "quant_type": "w8a8h1_sefp",
            "artifact_checks": artifact_checks,
        },
        "evaluation": {
            "suite": args.suite,
            "task_ids": task_ids,
            "samples_per_task": args.samples_per_task,
            "num_samples": len(records),
            "env_seed_start": args.seed,
            "noise_seed": args.noise_seed,
            "action_shape": [horizon, int(config.output_features["action"].shape[0])],
            "valid_prefix_length": {
                "min": min(valid_prefix_lengths),
                "max": max(valid_prefix_lengths),
            },
            "valid_language_length": {
                "min": min(valid_language_lengths),
                "max": max(valid_language_lengths),
            },
        },
        "aggregates": {
            "normalized_hmonnx_vs_lerobot": normalized_aggregate,
            "hmonnx_vs_lerobot": action_aggregate,
            "latency_ms": {
                "lerobot": {
                    "mean": float(reference_latencies.mean()),
                    "min": float(reference_latencies.min()),
                    "max": float(reference_latencies.max()),
                },
                "hmonnx": {
                    "mean": float(hmonnx_latencies.mean()),
                    "min": float(hmonnx_latencies.min()),
                    "max": float(hmonnx_latencies.max()),
                },
            },
        },
        "per_task": per_task,
        "closed_loop_hmonnx": closed_loop,
        "acceptance": acceptance,
        "records_file": str(records_path),
        "note": (
            "LeRobot is the float32 software reference. All HMONNX floating inputs and outputs are "
            "FP16 with no BF16 tensors; W8A8 quantization means parity is tolerance-based, not bit-exact."
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {records_path}")
    print(f"Wrote {summary_path}")
    print(json.dumps({"acceptance": acceptance, "aggregate": action_aggregate}, indent=2))
    return 0 if acceptance["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
