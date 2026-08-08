#!/usr/bin/env python3
"""Download, verify, and smoke-test the public ModelScope PI0.5 DROID mirror."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
MODELSCOPE_REPO_ID = "lerobot/pi05_droid"
MODELSCOPE_REVISION = "master"
MODELSCOPE_OBSERVED_FILE_REVISION = "0fd787830f979307d217c9523eef28231e9d7e3e"
DEFAULT_MODEL_DIR = (
    REPO_ROOT / "weights" / "pi05_result_check_droid" / "models" / "modelscope_lerobot_pi05_droid"
)
DEFAULT_RUNTIME_CONFIG_DIR = (
    REPO_ROOT / "weights" / "pi05_result_check_droid" / "models" / "pi05_droid_openpi_to_lerobot"
)
DEFAULT_INPUTS_DIR = REPO_ROOT / "weights" / "pi05_result_check_droid" / "inputs"
DEFAULT_NOISE_FILE = (
    REPO_ROOT / "weights" / "pi05_result_check_droid" / "noise" / "noise_droid_100_h50_a32_seed20260707.npz"
)

# ModelScope publishes SHA-256 and size metadata. MD5 values are recorded here
# as a convenient local mirror identity, not as a replacement for SHA-256.
EXPECTED_FILES = {
    "config.json": {
        "size_bytes": 2406,
        "sha256": "bb8d260332ad7206ab80accb739da60309d2b8ea23120363c30a4dcd0c9a91e5",
        "md5": "2025b6860656591d730b44a2bc0712c4",
    },
    "model.safetensors": {
        "size_bytes": 16573760032,
        "sha256": "684c8b2033dcfacee9c6a83f38810ecc77df2c81aeb171bb87319da01fafcfaa",
        "md5": "b653cdd10007defc65adea3b892ce741",
    },
    "policy_preprocessor.json": {
        "size_bytes": 1241,
        "sha256": "be2cd1acc33229d42be0640cb55b5d5a4bb3e8d8e27ddfcdcf3f99f1fe3beaef",
        "md5": "c42d985ffd0787681550c77c008b70af",
    },
    "policy_postprocessor.json": {
        "size_bytes": 567,
        "sha256": "142c8b622aeac5c14631e057279df058d3005f616d909cfe30a04524de0b0891",
        "md5": "c83b1d525df6c97f142f4c996b84d08e",
    },
}


def hash_file(path: Path) -> dict[str, str]:
    md5 = hashlib.md5()
    sha256 = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            md5.update(chunk)
            sha256.update(chunk)
    return {"md5": md5.hexdigest(), "sha256": sha256.hexdigest()}


def verify_snapshot(
    model_dir: Path,
    expected_files: dict[str, dict[str, Any]] = EXPECTED_FILES,
) -> dict[str, Any]:
    records: dict[str, Any] = {}
    passed = True
    for relative_path, expected in expected_files.items():
        path = model_dir / relative_path
        record: dict[str, Any] = {
            "path": str(path),
            "expected": expected,
        }
        if not path.is_file():
            record["status"] = "missing"
            passed = False
        else:
            actual = {
                "size_bytes": path.stat().st_size,
                **hash_file(path),
            }
            matches = {
                name: actual[name] == expected[name]
                for name in ("size_bytes", "sha256", "md5")
            }
            record.update(actual=actual, matches=matches, status="passed" if all(matches.values()) else "mismatch")
            passed = passed and all(matches.values())
        records[relative_path] = record
    return {
        "passed": passed,
        "source": {
            "repo_id": MODELSCOPE_REPO_ID,
            "revision": MODELSCOPE_REVISION,
            "observed_file_revision": MODELSCOPE_OBSERVED_FILE_REVISION,
        },
        "model_dir": str(model_dir),
        "files": records,
    }


def download_snapshot(model_dir: Path, max_workers: int) -> None:
    command = [
        "modelscope",
        "download",
        "--model",
        MODELSCOPE_REPO_ID,
        "--revision",
        MODELSCOPE_REVISION,
        "--local_dir",
        str(model_dir),
        "--max-workers",
        str(max_workers),
    ]
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def _resolve_device(device: str) -> str:
    if device.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return device


def _load_runtime_config(model_dir: Path, runtime_config_dir: Path, device: str, horizon: int):
    from xhmodel_merak.xh_other_model.models.pi05._export_utils import _load_local_pi05_config_compat

    config = _load_local_pi05_config_compat(str(runtime_config_dir))
    if config is None:
        raise ValueError(f"DROID runtime config is incompatible: {runtime_config_dir}")
    config.pretrained_path = model_dir
    config.device = device
    config.dtype = "float32"
    config.compile_model = False
    config.chunk_size = horizon
    config.n_action_steps = horizon
    return config


def _load_fixed_droid_sample(inputs_dir: Path, sample_id: int, action_dim: int) -> tuple[dict[str, Any], Path]:
    from examples_merak.vla.pi05.pi05_droid_hmonnx_validation import chw_float, find_sample

    input_file = find_sample(inputs_dir, sample_id)
    with np.load(input_file, allow_pickle=False) as sample_file:
        sample = {
            "observation.images.base_0_rgb": chw_float(sample_file["exterior_image_1_left"]),
            "observation.images.left_wrist_0_rgb": chw_float(sample_file["wrist_image_left"]),
            "observation.state": torch.from_numpy(np.asarray(sample_file["state"], dtype=np.float32)),
            "action": torch.zeros(action_dim, dtype=torch.float32),
            "task": str(sample_file["task"].item()),
        }
    return sample, input_file


def _noise_for_sample(noise_file: Path, sample_id: int, horizon: int, device: str) -> torch.Tensor:
    key = f"sample_{sample_id:04d}"
    with np.load(noise_file, allow_pickle=False) as noises:
        if key not in noises:
            raise KeyError(f"Noise archive is missing {key}: {noise_file}")
        noise = np.asarray(noises[key], dtype=np.float32)
    if noise.ndim != 2 or noise.shape[0] < horizon or noise.shape[1] != 32:
        raise ValueError(f"Noise {key} has shape {noise.shape}; expected at least [{horizon}, 32]")
    return torch.from_numpy(noise[:horizon].copy())[None].to(device)


def _action_digest(actions: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(actions, dtype=np.float32).tobytes()).hexdigest()


def run_demo(
    *,
    model_dir: Path,
    runtime_config_dir: Path,
    tokenizer_dir: Path,
    inputs_dir: Path,
    noise_file: Path,
    sample_id: int,
    horizon: int,
    device: str,
) -> tuple[dict[str, Any], np.ndarray]:
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.pi05 import PI05Policy
    from lerobot.utils.constants import ACTION

    required_paths = (
        model_dir / "model.safetensors",
        runtime_config_dir / "config.json",
        tokenizer_dir,
        inputs_dir,
        noise_file,
    )
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(path)

    device = _resolve_device(device)
    config = _load_runtime_config(model_dir, runtime_config_dir, device, horizon)
    policy = PI05Policy.from_pretrained(model_dir, config=config, strict=True).to(device).eval()
    preprocessor, postprocessor = make_pre_post_processors(
        config,
        pretrained_path=str(runtime_config_dir),
        preprocessor_overrides={
            "tokenizer_processor": {"tokenizer_name": str(tokenizer_dir)},
            "device_processor": {"device": device, "float_dtype": None},
        },
        postprocessor_overrides={"device_processor": {"device": "cpu", "float_dtype": None}},
    )
    sample, input_file = _load_fixed_droid_sample(
        inputs_dir,
        sample_id,
        int(config.output_features[ACTION].shape[0]),
    )
    noise = _noise_for_sample(noise_file, sample_id, horizon, device)
    batch = preprocessor(sample)
    if device.startswith("cuda"):
        torch.cuda.synchronize(torch.device(device))
    started = time.perf_counter()
    with torch.inference_mode():
        actions = postprocessor(policy.predict_action_chunk(batch, noise=noise))
    if device.startswith("cuda"):
        torch.cuda.synchronize(torch.device(device))
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    actions_np = actions.detach().cpu().numpy().astype(np.float32)
    result = {
        "model_dir": str(model_dir),
        "runtime_config_dir": str(runtime_config_dir),
        "sample_id": sample_id,
        "input_file": str(input_file),
        "input_sha256": hash_file(input_file)["sha256"],
        "noise_file": str(noise_file),
        "noise_sha256": hash_file(noise_file)["sha256"],
        "device": device,
        "horizon": horizon,
        "actions_shape": list(actions_np.shape),
        "actions_sha256": _action_digest(actions_np),
        "actions_mean": float(actions_np.mean()),
        "actions_std": float(actions_np.std()),
        "actions_min": float(actions_np.min()),
        "actions_max": float(actions_np.max()),
        "latency_ms_wall": elapsed_ms,
    }
    del policy
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result, actions_np


def compare_actions(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    if reference.shape != candidate.shape:
        raise ValueError(f"Action shape mismatch: reference={reference.shape}, candidate={candidate.shape}")
    difference = candidate - reference
    denominator = max(float(np.linalg.norm(reference) * np.linalg.norm(candidate)), 1e-12)
    return {
        "cosine": float(np.dot(reference.reshape(-1), candidate.reshape(-1)) / denominator),
        "mae": float(np.abs(difference).mean()),
        "max_abs": float(np.abs(difference).max()),
    }


def _resolve_path(path: Path | None, default: Path | None = None) -> Path:
    if path is None and default is None:
        raise ValueError("A path must be provided")
    value = path if path is not None else default
    assert value is not None
    if not value.is_absolute():
        value = REPO_ROOT / value
    return value.expanduser().resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--runtime-config-dir", type=Path, default=None)
    parser.add_argument("--tokenizer-dir", type=Path, default=None)
    parser.add_argument("--inputs-dir", type=Path, default=None)
    parser.add_argument("--noise-file", type=Path, default=None)
    parser.add_argument("--download", action="store_true", help="Download the ModelScope snapshot before verification.")
    parser.add_argument("--verify", action="store_true", help="Verify source SHA-256 and local MD5 identities.")
    parser.add_argument("--demo", action="store_true", help="Run one fixed DROID LeRobot inference sample.")
    parser.add_argument("--compare-model-dir", type=Path, default=None)
    parser.add_argument("--sample-id", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON report path.")
    args = parser.parse_args()
    if args.sample_id < 0:
        parser.error("--sample-id must be non-negative")
    if args.horizon <= 0:
        parser.error("--horizon must be positive")
    if args.max_workers <= 0:
        parser.error("--max-workers must be positive")
    if args.demo and args.tokenizer_dir is None:
        parser.error("--tokenizer-dir is required with --demo")
    return args


def main() -> int:
    args = parse_args()
    model_dir = _resolve_path(args.model_dir, DEFAULT_MODEL_DIR)
    runtime_config_dir = _resolve_path(args.runtime_config_dir, DEFAULT_RUNTIME_CONFIG_DIR)
    inputs_dir = _resolve_path(args.inputs_dir, DEFAULT_INPUTS_DIR)
    noise_file = _resolve_path(args.noise_file, DEFAULT_NOISE_FILE)
    output_path = _resolve_path(args.output, args.output) if args.output is not None else None

    report: dict[str, Any] = {}
    if args.download:
        download_snapshot(model_dir, args.max_workers)
        args.verify = True
    if args.verify or not args.demo:
        report["verification"] = verify_snapshot(model_dir)
        if not report["verification"]["passed"]:
            print(json.dumps(report, indent=2), flush=True)
            return 1
    if args.demo:
        tokenizer_dir = _resolve_path(args.tokenizer_dir)
        demo, actions = run_demo(
            model_dir=model_dir,
            runtime_config_dir=runtime_config_dir,
            tokenizer_dir=tokenizer_dir,
            inputs_dir=inputs_dir,
            noise_file=noise_file,
            sample_id=args.sample_id,
            horizon=args.horizon,
            device=args.device,
        )
        report["demo"] = demo
        if args.compare_model_dir is not None:
            reference_dir = _resolve_path(args.compare_model_dir, args.compare_model_dir)
            reference_demo, reference_actions = run_demo(
                model_dir=reference_dir,
                runtime_config_dir=runtime_config_dir,
                tokenizer_dir=tokenizer_dir,
                inputs_dir=inputs_dir,
                noise_file=noise_file,
                sample_id=args.sample_id,
                horizon=args.horizon,
                device=args.device,
            )
            report["comparison"] = {
                "reference": reference_demo,
                "metrics": compare_actions(reference_actions, actions),
            }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
        print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
