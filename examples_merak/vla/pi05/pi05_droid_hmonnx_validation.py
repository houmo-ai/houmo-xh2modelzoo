#!/usr/bin/env python3
"""Validate compact-prefix PI0.5 DROID HMONNX against LeRobot."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import re
import time
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROJECT_ROOT = REPO_ROOT / "weights" / "pi05_result_check_droid"
DEFAULT_MODEL_DIR = "pi05_droid_openpi_to_lerobot"
DEFAULT_EXPORT_DIR = "pi05_droid_customer_h50_compact_maskadd2_XH2a"
DEFAULT_NOISE = "noise_droid_100_h50_a32_seed20260707.npz"
MASK_VALUE = torch.finfo(torch.float16).min
HMONNX_RUNTIME_CHOICES = ("legacy", "v2", "v2-cuda-graph")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sample_key(sample_id: int) -> str:
    return f"sample_{sample_id:04d}"


def find_sample(inputs_dir: Path, sample_id: int) -> Path:
    expected = inputs_dir / f"{sample_key(sample_id)}_idx_{sample_id * 100:06d}.npz"
    if expected.exists():
        return expected
    matches = sorted(glob.glob(str(inputs_dir / f"{sample_key(sample_id)}_idx_*.npz")))
    if not matches:
        raise FileNotFoundError(f"No input found for sample_id={sample_id} in {inputs_dir}")
    return Path(matches[0])


def parse_dataset_index(path: Path) -> int:
    match = re.search(r"_idx_(\d+)\.npz$", path.name)
    return int(match.group(1)) if match else -1


def chw_float(image_hwc: np.ndarray) -> torch.Tensor:
    image = np.asarray(image_hwc, dtype=np.float32) / 255.0
    return torch.from_numpy(np.transpose(image, (2, 0, 1))).contiguous()


def create_time_embedding(timestep: torch.Tensor, dimension: int = 1024) -> torch.Tensor:
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=torch.float64, device=timestep.device)
    period = 0.004 * (4.0 / 0.004) ** fraction
    sin_input = (2 * math.pi / period)[None, :] * timestep.to(torch.float64)[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1).float()


def load_records(path: Path | None) -> dict[int, dict]:
    if path is None or not path.exists():
        return {}
    with path.open() as file:
        return {int(record["sample_id"]): record for line in file if line.strip() for record in [json.loads(line)]}


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    left = left.reshape(-1).astype(np.float32)
    right = right.reshape(-1).astype(np.float32)
    denominator = max(float(np.linalg.norm(left) * np.linalg.norm(right)), 1e-12)
    return float(np.dot(left, right) / denominator)


def action_array(record: dict) -> np.ndarray:
    actions = np.asarray(record["actions"], dtype=np.float32)
    if actions.ndim == 3 and actions.shape[0] == 1:
        actions = actions[0]
    return actions


def write_summary(
    hmonnx_output: Path,
    reference_output: Path,
    external_reference: Path | None,
    summary_output: Path,
) -> bool:
    hmonnx_records = load_records(hmonnx_output)
    reference_records = load_records(external_reference)
    reference_records.update(load_records(reference_output))
    matched_ids = sorted(hmonnx_records.keys() & reference_records.keys())
    if not matched_ids:
        return False

    per_sample = []
    for sample_id in matched_ids:
        reference = action_array(reference_records[sample_id])
        hmonnx = action_array(hmonnx_records[sample_id])
        if reference.shape != hmonnx.shape:
            raise ValueError(f"sample {sample_id} shape mismatch: reference={reference.shape}, hmonnx={hmonnx.shape}")
        difference = hmonnx - reference
        per_sample.append(
            {
                "sample_id": sample_id,
                "dataset_index": hmonnx_records[sample_id]["dataset_index"],
                "shape": list(reference.shape),
                "cosine": cosine(reference, hmonnx),
                "mae": float(np.mean(np.abs(difference))),
                "rmse": float(np.sqrt(np.mean(np.square(difference)))),
                "max_abs": float(np.max(np.abs(difference))),
            }
        )

    aggregate = {"num_samples": len(per_sample), "shape": per_sample[0]["shape"]}
    for metric in ("cosine", "mae", "rmse", "max_abs"):
        values = np.asarray([record[metric] for record in per_sample], dtype=np.float64)
        aggregate[metric] = {
            "mean": float(values.mean()),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    latencies = np.asarray(
        [hmonnx_records[sample_id]["latency_ms_wall"] for sample_id in matched_ids],
        dtype=np.float64,
    )
    aggregate["hmonnx_latency_ms_wall"] = {
        "mean": float(latencies.mean()),
        "min": float(latencies.min()),
        "max": float(latencies.max()),
    }

    first_record = hmonnx_records[matched_ids[0]]
    valid_prefix_lengths = [hmonnx_records[sample_id]["valid_prefix_length"] for sample_id in matched_ids]
    valid_language_lengths = [hmonnx_records[sample_id]["valid_language_length"] for sample_id in matched_ids]
    runtime_contract = {
        "static_prefix_length": first_record["physical_prefix_length"],
        "valid_prefix_length": {
            "min": min(valid_prefix_lengths),
            "max": max(valid_prefix_lengths),
        },
        "valid_language_length": {
            "min": min(valid_language_lengths),
            "max": max(valid_language_lengths),
        },
        "selected_image_indices": first_record["selected_image_indices"],
        "vision_runs": first_record["vision_runs"],
        "horizon": first_record["horizon"],
        "noise_sha256": first_record["noise_sha256"],
    }

    summary = {
        "contract": "compact_prefix_valid_kv",
        "backend": "hmonnx_software",
        "hmonnx_file": str(hmonnx_output),
        "reference_file": str(external_reference or reference_output),
        "matched_sample_ids": matched_ids,
        "runtime_contract": runtime_contract,
        "aggregate": aggregate,
        "per_sample": per_sample,
    }
    summary_output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return True


def load_config(
    model_dir: Path,
    device: str,
    action_horizon: int,
    runtime_config_dir: Path | None = None,
):
    from xhmodel_merak.xh_other_model.models.pi05._export_utils import _load_local_pi05_config_compat

    config_source = runtime_config_dir or model_dir
    config = _load_local_pi05_config_compat(str(config_source))
    if config is None:
        from lerobot.configs.policies import PreTrainedConfig

        config = PreTrainedConfig.from_pretrained(config_source)
    config.pretrained_path = model_dir
    config.device = device
    config.dtype = "float32"
    config.compile_model = False
    config.chunk_size = action_horizon
    config.n_action_steps = action_horizon
    return config


def _create_hmonnx_session(path: Path, device: torch.device, runtime_mode: str):
    if runtime_mode not in HMONNX_RUNTIME_CHOICES:
        raise ValueError(f"Unsupported HMONNX runtime mode: {runtime_mode}")
    if runtime_mode == "legacy":
        from xhquant.api import HMONNXInference

        return HMONNXInference(str(path)).to(device)

    from xhquant.xhonnxruntime.hmonnx_inference_v2 import HMONNXInferenceConfig, HMONNXInferenceV2

    session_config = HMONNXInferenceConfig(
        enable_cuda_graph=runtime_mode == "v2-cuda-graph",
        exec_devices=[device],
    )
    return HMONNXInferenceV2(str(path), session_config)


class CompactHMONNX:
    def __init__(
        self,
        policy,
        export_dir: Path,
        device: torch.device,
        runtime_mode: str = "legacy",
    ):
        from xhquant.core import CacheTensor

        self.policy = policy
        self.device = device
        self.runtime_mode = runtime_mode
        self._active_runtime_mode = runtime_mode
        self._cuda_graph_primed = runtime_mode != "v2-cuda-graph"
        self._cuda_graph_prewarming = False
        self._cuda_graph_prewarm_ms: float | None = None
        self.CacheTensor = CacheTensor
        export_meta = json.loads((export_dir / "export_meta_info.json").read_text(encoding="utf-8"))
        contract = export_meta["compact_prefix"]
        self.selected_image_indices = [int(index) for index in contract["selected_image_indices"]]
        self.prefix_sequence_length = int(contract["prefix_sequence_length"])
        self.action_horizon = int(contract["action_horizon"])
        self.cache_length = int(contract["cache_length"])
        self.action_dim = int(policy.config.output_features["action"].shape[0])
        self.session_paths = {
            "vision": export_dir / "Vision/hmonnx/vision_XH2a_w8a8h1_sefp.onnx",
            "action_in": export_dir / "Other/hmonnx/action_in_proj_XH2a_w8a8h1_sefp.onnx",
            "action_out": export_dir / "Other/hmonnx/action_out_proj_XH2a_w8a8h1_sefp.onnx",
            "time_mlp": export_dir / "Other/hmonnx/time_mlp_XH2a_w8a8h1_sefp.onnx",
            "gemma": export_dir / "Gemma2B/prefill_onnx/pi05_gemma_2b_XH2a_w8a8h1_sefp_prefill.onnx",
            "expert": export_dir / "GemmaExpert/decode_onnx/pi05_gemma_expert_300m_XH2a_w8a8h1_sefp_decode.onnx",
        }
        self.sessions = self._create_sessions(self._active_runtime_mode)

    def _create_sessions(self, runtime_mode: str):
        return {
            name: _create_hmonnx_session(path, self.device, runtime_mode) for name, path in self.session_paths.items()
        }

    def _prime_cuda_graph(self, batch: dict[str, torch.Tensor], noise: torch.Tensor) -> None:
        started = time.perf_counter()
        self._cuda_graph_prewarming = True
        try:
            # Keep the warmed graph module so Triton autotuning survives until capture.
            self._predict(batch, noise)
        finally:
            self._cuda_graph_prewarming = False
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self._cuda_graph_primed = True
        self._cuda_graph_prewarm_ms = (time.perf_counter() - started) * 1000.0

    def _run_session(self, name: str, *args: torch.Tensor):
        session = self.sessions[name]
        if self._cuda_graph_prewarming:
            eager_warmup = getattr(session, "warmup_without_cuda_graph", None)
            if callable(eager_warmup):
                return eager_warmup(*args)
        return session(*args)

    def _runtime_status(self) -> dict[str, object]:
        sessions = {}
        for name, session in self.sessions.items():
            interpreter = getattr(session, "interpreter", None)
            sessions[name] = {
                "session": type(session).__name__,
                "interpreter": type(interpreter).__name__ if interpreter is not None else None,
                "cuda_graph_captured": bool(getattr(interpreter, "has_captured_graph", False)),
                "cuda_graph_capture_disabled_reason": getattr(interpreter, "capture_disabled_reason", None),
            }
        return {
            "requested": self.runtime_mode,
            "active": self._active_runtime_mode,
            "cuda_graph_prewarm_ms": self._cuda_graph_prewarm_ms,
            "sessions": sessions,
        }

    def _require_cuda_graph_capture(self) -> None:
        failed_sessions = {
            name: status["cuda_graph_capture_disabled_reason"] or "capture did not complete"
            for name, status in self._runtime_status()["sessions"].items()
            if not status["cuda_graph_captured"]
        }
        if failed_sessions:
            raise RuntimeError(f"CUDA Graph capture failed: {failed_sessions}")

    @torch.inference_mode()
    def predict(self, batch: dict[str, torch.Tensor], noise: torch.Tensor):
        if not self._cuda_graph_primed:
            self._prime_cuda_graph(batch, noise)
        result = self._predict(batch, noise)
        if self.runtime_mode == "v2-cuda-graph":
            self._require_cuda_graph_capture()
        return result

    @torch.inference_mode()
    def _predict(self, batch: dict[str, torch.Tensor], noise: torch.Tensor):
        images, image_masks = self.policy._preprocess_images(batch)
        vision_outputs = []
        for image_index in self.selected_image_indices:
            if not bool(image_masks[image_index].item()):
                raise ValueError(f"Selected compact-prefix image {image_index} is not valid")
            vision_outputs.append(self._run_session("vision", images[image_index].half()))

        tokens = batch["observation.language.tokens"]
        token_masks = batch["observation.language.attention_mask"]
        language_embs = (
            self.policy.model.paligemma_with_expert.embed_language_tokens(tokens)
            * math.sqrt(self.policy.model.paligemma_with_expert.paligemma.config.text_config.hidden_size)
        ).half()
        prefix_embs = torch.cat([*vision_outputs, language_embs], dim=1)
        prefix_pad_masks = torch.cat(
            [
                image_masks[image_index][:, None].expand(image_masks[image_index].shape[0], output.shape[1])
                for image_index, output in zip(
                    self.selected_image_indices,
                    vision_outputs,
                    strict=True,
                )
            ]
            + [token_masks],
            dim=1,
        )
        physical_prefix_length = int(prefix_embs.shape[1])
        valid_prefix_length = int(prefix_pad_masks.sum().item())
        if physical_prefix_length != self.prefix_sequence_length:
            raise ValueError(
                f"Compact prefix has {physical_prefix_length} tokens, expected {self.prefix_sequence_length}"
            )
        if not bool(prefix_pad_masks[:, :valid_prefix_length].all()) or bool(
            prefix_pad_masks[:, valid_prefix_length:].any()
        ):
            raise ValueError("Compact prefix contains a hole before language padding")
        prefix_attention = torch.full(
            (1, 1, 1, self.cache_length),
            MASK_VALUE,
            dtype=torch.float16,
            device=self.device,
        )
        prefix_attention[..., :valid_prefix_length] = 0

        caches = [
            self.CacheTensor(torch.zeros(1, 1, self.cache_length, 256, dtype=torch.float16, device=self.device))
            for _ in range(36)
        ]
        self._run_session(
            "gemma",
            prefix_embs,
            torch.zeros(1, dtype=torch.int32, device=self.device),
            torch.tensor([valid_prefix_length], dtype=torch.int32, device=self.device),
            prefix_attention,
            *caches,
        )

        expert_attention = torch.full(
            (1, 1, 1, self.cache_length),
            MASK_VALUE,
            dtype=torch.float16,
            device=self.device,
        )
        expert_attention[..., : valid_prefix_length + self.action_horizon] = 0

        current = noise.float()
        for step in range(10):
            timestep = torch.tensor(
                [1.0 - step / 10.0],
                dtype=torch.float32,
                device=self.device,
            )
            suffix_embs = self._run_session("action_in", current.half())
            cond = self._run_session("time_mlp", create_time_embedding(timestep).half())
            expert_hidden = self._run_session(
                "expert",
                suffix_embs,
                torch.tensor([valid_prefix_length], dtype=torch.int32, device=self.device),
                torch.tensor([self.action_horizon], dtype=torch.int32, device=self.device),
                cond,
                expert_attention,
                *caches,
            )
            velocity = self._run_session("action_out", expert_hidden).float()
            current = current - velocity / 10.0

        return current[:, :, : self.action_dim], {
            "physical_prefix_length": physical_prefix_length,
            "valid_prefix_length": valid_prefix_length,
            "valid_language_length": valid_prefix_length - len(self.selected_image_indices) * 256,
            "image_masks": [bool(mask.item()) for mask in image_masks],
            "selected_image_indices": self.selected_image_indices,
            "vision_runs": len(vision_outputs),
            "runtime": self._runtime_status(),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--workspace-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument(
        "--runtime-config-dir",
        type=Path,
        default=None,
        help="Optional LeRobot config/processor bundle used with --model-dir weights.",
    )
    parser.add_argument("--export-dir", type=Path, default=None)
    parser.add_argument(
        "--tokenizer-dir",
        type=Path,
        required=True,
    )
    parser.add_argument("--inputs-dir", type=Path, default=None)
    parser.add_argument("--noise-file", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--hmonnx-output", default="compact_h50_hmonnx_droid.jsonl")
    parser.add_argument("--reference-output", default="compact_h50_lerobot_droid.jsonl")
    parser.add_argument("--summary-output", default="compact_h50_summary.json")
    parser.add_argument("--reference-jsonl", type=Path, default=None)
    parser.add_argument("--start-sample-id", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument(
        "--sample-ids",
        default=None,
        help="Comma-separated sample IDs; overrides --start-sample-id/--num-samples.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--hmonnx-runtime",
        choices=HMONNX_RUNTIME_CHOICES,
        default="v2-cuda-graph",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--hmonnx-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.pi05 import PI05Policy
    from lerobot.utils.constants import ACTION

    args = parse_args()
    project_root = args.project_root.resolve()
    workspace_root = args.workspace_root.resolve()
    model_dir = (args.model_dir or project_root / "models" / DEFAULT_MODEL_DIR).resolve()
    runtime_config_dir = (args.runtime_config_dir or model_dir).resolve()
    export_dir = (args.export_dir or workspace_root / "work_dirs" / DEFAULT_EXPORT_DIR).resolve()
    tokenizer_dir = args.tokenizer_dir.resolve()
    inputs_dir = (args.inputs_dir or project_root / "inputs").resolve()
    noise_file = (args.noise_file or project_root / "noise" / DEFAULT_NOISE).resolve()
    results_dir = (args.results_dir or export_dir / "validation_results").resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    hmonnx_output = results_dir / args.hmonnx_output
    reference_output = results_dir / args.reference_output
    summary_output = results_dir / args.summary_output

    for path in (
        model_dir / "config.json",
        runtime_config_dir / "config.json",
        export_dir,
        tokenizer_dir,
        inputs_dir,
        noise_file,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    device_name = args.device
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)

    export_meta = json.loads((export_dir / "export_meta_info.json").read_text(encoding="utf-8"))
    action_horizon = int(export_meta["compact_prefix"]["action_horizon"])
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
    hmonnx = CompactHMONNX(policy, export_dir, device, runtime_mode=args.hmonnx_runtime)
    noise_sha = sha256_file(noise_file)

    reference_records = load_records(args.reference_jsonl)
    if args.resume:
        reference_records.update(load_records(reference_output))
    completed = set(load_records(hmonnx_output)) if args.resume else set()
    mode = "a" if args.resume else "w"
    reference_file = None if args.hmonnx_only else reference_output.open(mode)
    try:
        with np.load(noise_file, allow_pickle=False) as noises, hmonnx_output.open(mode) as hmonnx_file:
            if args.sample_ids:
                sample_ids = list(
                    dict.fromkeys(int(value.strip()) for value in args.sample_ids.split(",") if value.strip())
                )
            else:
                sample_ids = range(args.start_sample_id, args.start_sample_id + args.num_samples)
            for sample_id in sample_ids:
                if sample_id in completed:
                    continue

                input_file = find_sample(inputs_dir, sample_id)
                with np.load(input_file, allow_pickle=False) as sample_file:
                    sample = {
                        "observation.images.base_0_rgb": chw_float(sample_file["exterior_image_1_left"]),
                        "observation.images.left_wrist_0_rgb": chw_float(sample_file["wrist_image_left"]),
                        "observation.state": torch.from_numpy(np.asarray(sample_file["state"], dtype=np.float32)),
                        "action": torch.zeros(
                            config.output_features[ACTION].shape[0],
                            dtype=torch.float32,
                        ),
                        "task": str(sample_file["task"].item()),
                    }
                key = sample_key(sample_id)
                noise_array = np.asarray(noises[key], dtype=np.float32)
                if noise_array.ndim != 2 or noise_array.shape[0] < action_horizon:
                    raise ValueError(
                        f"Noise {key} has shape {noise_array.shape}, expected at least [{action_horizon}, action_dim]"
                    )
                noise = torch.from_numpy(noise_array[:action_horizon].copy())[None].to(device)
                batch = preprocessor(sample)

                reference_record = reference_records.get(sample_id)
                reference_actions = None
                if reference_record is not None:
                    reference_actions = np.asarray(reference_record["actions"], dtype=np.float32)
                elif not args.hmonnx_only:
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    started = time.perf_counter()
                    with torch.inference_mode():
                        normalized_reference = policy.predict_action_chunk(batch, noise=noise.clone())
                        reference_actions = (
                            postprocessor(normalized_reference).detach().cpu().numpy().astype(np.float32)
                        )
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    reference_latency = (time.perf_counter() - started) * 1000.0
                    reference_record = {
                        "source": "lerobot",
                        "sample_id": sample_id,
                        "dataset_index": parse_dataset_index(input_file),
                        "input_file": str(input_file),
                        "input_sha256": sha256_file(input_file),
                        "task": sample["task"],
                        "model_dir": str(model_dir),
                        "runtime_config_dir": str(runtime_config_dir),
                        "horizon": action_horizon,
                        "noise_file": str(noise_file),
                        "noise_key": key,
                        "noise_sha256": noise_sha,
                        "device": device_name,
                        "latency_ms_wall": reference_latency,
                        "actions_shape": list(reference_actions.shape),
                        "actions": reference_actions.tolist(),
                    }
                    reference_file.write(json.dumps(reference_record) + "\n")
                    reference_file.flush()

                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                started = time.perf_counter()
                normalized_hmonnx, contract = hmonnx.predict(batch, noise.clone())
                hmonnx_actions = postprocessor(normalized_hmonnx).detach().cpu().numpy().astype(np.float32)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                hmonnx_latency = (time.perf_counter() - started) * 1000.0

                hmonnx_record = {
                    "source": "hmm",
                    "backend": "hmonnx_software",
                    "runtime": args.hmonnx_runtime,
                    "sample_id": sample_id,
                    "dataset_index": parse_dataset_index(input_file),
                    "input_file": str(input_file),
                    "input_sha256": sha256_file(input_file),
                    "task": sample["task"],
                    "config_dir": str(model_dir),
                    "runtime_config_dir": str(runtime_config_dir),
                    "hmm_dir": str(export_dir),
                    "mode": "compact_prefix_valid_kv",
                    "horizon": action_horizon,
                    "noise_file": str(noise_file),
                    "noise_key": key,
                    "noise_sha256": noise_sha,
                    "latency_ms_wall": hmonnx_latency,
                    **contract,
                    "actions_shape": list(hmonnx_actions.shape),
                    "actions": hmonnx_actions.tolist(),
                }
                hmonnx_file.write(json.dumps(hmonnx_record) + "\n")
                hmonnx_file.flush()

                progress = {
                    "sample_id": sample_id,
                    "valid_prefix_length": contract["valid_prefix_length"],
                    "hmonnx_ms": hmonnx_latency,
                }
                if reference_actions is not None:
                    progress.update(
                        cosine=cosine(reference_actions, hmonnx_actions),
                        mae=float(np.mean(np.abs(reference_actions - hmonnx_actions))),
                    )
                print(json.dumps(progress), flush=True)
    finally:
        if reference_file is not None:
            reference_file.close()

    print(f"Wrote {hmonnx_output}")
    if not args.hmonnx_only:
        print(f"Wrote {reference_output}")
    if write_summary(hmonnx_output, reference_output, args.reference_jsonl, summary_output):
        print(f"Wrote {summary_output}")


if __name__ == "__main__":
    main()
