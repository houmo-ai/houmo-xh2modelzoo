#!/usr/bin/env python3
"""Run and verify the PI0.5 DROID export-to-accuracy handoff."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROJECT_ROOT = REPO_ROOT / "weights" / "pi05_result_check_droid"
DEFAULT_MODEL_NAME = "pi05_droid_openpi_to_lerobot"
DEFAULT_NOISE_NAME = "noise_droid_100_h50_a32_seed20260707.npz"
EXPECTED_NOISE_SHA256 = "29d1b136a58a1ee2a16609cbc3b03ed977f773e4fcf574b61e72f6f5e21a1ae4"


@dataclass(frozen=True)
class VariantSpec:
    workflow_variant: str
    metadata_variant: str
    config_path: str
    export_dir: str
    horizon: int
    hmonnx_output: str
    reference_output: str
    summary_output: str

    @property
    def contract(self) -> dict[str, Any]:
        return {
            "selected_image_indices": [0, 1],
            "text_max_length": 200,
            "prefix_sequence_length": 712,
            "action_horizon": self.horizon,
            "cache_length": 1024,
        }


VARIANTS = {
    "droid-customer-h50": VariantSpec(
        workflow_variant="droid-customer-h50",
        metadata_variant="droid_customer_h50",
        config_path=(
            "configs_merak/workflows/xh2a/other_models/pi05/droid/"
            "pi05_droid_customer_h50.yaml"
        ),
        export_dir="work_dirs/pi05_droid_customer_h50_compact_maskadd2_XH2a",
        horizon=50,
        hmonnx_output="compact_h50_hmonnx_droid.jsonl",
        reference_output="compact_h50_lerobot_droid.jsonl",
        summary_output="compact_h50_summary.json",
    ),
    "droid-openpi-h15": VariantSpec(
        workflow_variant="droid-openpi-h15",
        metadata_variant="droid_openpi_h15",
        config_path=(
            "configs_merak/workflows/xh2a/other_models/pi05/droid/"
            "pi05_droid_openpi_h15.yaml"
        ),
        export_dir="work_dirs/pi05_droid_openpi_h15_compact_maskadd2_XH2a",
        horizon=15,
        hmonnx_output="official_h15_hmonnx_droid.jsonl",
        reference_output="official_h15_lerobot_droid.jsonl",
        summary_output="official_h15_summary.json",
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return data


def resolve_path(value: Path | None, default: Path | str | None = None) -> Path:
    if value is None and default is None:
        raise ValueError("A path must be provided")
    path = value if value is not None else Path(default)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.expanduser().resolve()


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")


def require_directory(path: Path) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"Required directory not found: {path}")


def package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def expected_sample_ids(args: argparse.Namespace) -> list[int]:
    return list(range(args.start_sample_id, args.start_sample_id + args.num_samples))


def model_identity(model_dir: Path, hash_weights: bool) -> dict[str, Any]:
    identity: dict[str, Any] = {"path": str(model_dir), "files": {}}
    for name in ("config.json", "norm_stats.json"):
        path = model_dir / name
        if path.is_file():
            identity["files"][name] = {
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }

    weight_files = sorted(model_dir.glob("*.safetensors"))
    if not weight_files:
        raise FileNotFoundError(f"No *.safetensors weights found in {model_dir}")
    identity["weights"] = []
    for path in weight_files:
        item: dict[str, Any] = {
            "path": str(path),
            "size_bytes": path.stat().st_size,
        }
        if hash_weights:
            item["sha256"] = sha256_file(path)
        identity["weights"].append(item)
    return identity


def check_python_environment(device: str, require_device: bool = True) -> dict[str, Any]:
    modules = ("lerobot", "numpy", "onnx", "torch", "xhquant")
    missing = [name for name in modules if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError(f"Missing Python module(s): {', '.join(missing)}")

    import torch

    if require_device and device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {device}")
    if require_device and device.startswith("cuda:"):
        device_index = int(device.split(":", 1)[1])
        if device_index >= torch.cuda.device_count():
            raise RuntimeError(
                f"CUDA device {device} is out of range; visible device count is {torch.cuda.device_count()}"
            )

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "requested_device": device,
        "device_required": require_device,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "packages": {
            name: package_version(name)
            for name in ("lerobot", "numpy", "onnx", "xhquant")
        },
    }


def preflight(
    args: argparse.Namespace,
    spec: VariantSpec,
    paths: dict[str, Path],
) -> dict[str, Any]:
    require_file(paths["config_path"])
    require_directory(paths["model_dir"])
    require_file(paths["model_dir"] / "config.json")
    require_directory(paths["tokenizer_dir"])
    if not any((paths["tokenizer_dir"] / name).is_file() for name in ("tokenizer.json", "tokenizer.model")):
        raise FileNotFoundError(
            f"Tokenizer directory has neither tokenizer.json nor tokenizer.model: {paths['tokenizer_dir']}"
        )

    require_device = not args.skip_export or (
        not args.skip_validation and not args.reuse_validation
    )
    environment = check_python_environment(args.device, require_device=require_device)
    checks: dict[str, Any] = {
        "environment": environment,
        "model": model_identity(paths["model_dir"], args.hash_model),
        "config_path": str(paths["config_path"]),
        "config_sha256": sha256_file(paths["config_path"]),
        "tokenizer_dir": str(paths["tokenizer_dir"]),
    }

    if not args.skip_validation:
        require_directory(paths["inputs_dir"])
        require_file(paths["noise_file"])
        if args.reference_jsonl is not None:
            require_file(paths["reference_jsonl"])

        sample_ids = expected_sample_ids(args)
        missing_samples = [
            sample_id
            for sample_id in sample_ids
            if not list(paths["inputs_dir"].glob(f"sample_{sample_id:04d}_idx_*.npz"))
        ]
        if missing_samples:
            raise FileNotFoundError(
                f"Missing DROID input NPZ files for sample IDs: {missing_samples[:10]}"
            )

        noise_sha256 = sha256_file(paths["noise_file"])
        if args.expected_noise_sha256 and noise_sha256 != args.expected_noise_sha256:
            raise ValueError(
                f"Noise SHA256 mismatch: expected {args.expected_noise_sha256}, got {noise_sha256}"
            )

        import numpy as np

        with np.load(paths["noise_file"], allow_pickle=False) as noises:
            for sample_id in sample_ids:
                key = f"sample_{sample_id:04d}"
                if key not in noises:
                    raise KeyError(f"Noise archive is missing {key}")
                shape = noises[key].shape
                if len(shape) != 2 or shape[0] < spec.horizon or shape[1] != 32:
                    raise ValueError(
                        f"Noise {key} has shape {shape}; expected at least [{spec.horizon}, 32]"
                    )

        checks["dataset"] = {
            "inputs_dir": str(paths["inputs_dir"]),
            "sample_ids": sample_ids,
            "num_samples": len(sample_ids),
            "noise_file": str(paths["noise_file"]),
            "noise_sha256": noise_sha256,
        }

    if args.skip_export:
        require_directory(paths["export_dir"])
    elif paths["export_dir"].exists() and not args.overwrite and not args.dry_run and not args.check_only:
        raise FileExistsError(
            f"Export directory already exists: {paths['export_dir']}. "
            "Use --overwrite for a clean export or --skip-export to verify it."
        )

    return checks


def command_text(command: list[str]) -> str:
    return shlex.join(command)


def run_command(command: list[str]) -> None:
    print(f"+ {command_text(command)}", flush=True)
    environment = os.environ.copy()
    python_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = str(REPO_ROOT) + (os.pathsep + python_path if python_path else "")
    subprocess.run(command, cwd=REPO_ROOT, env=environment, check=True)


def build_export_command(
    args: argparse.Namespace,
    spec: VariantSpec,
    paths: dict[str, Path],
) -> list[str]:
    command = [
        sys.executable,
        str(REPO_ROOT / "examples_merak/vla/pi05/pi05_workflow.py"),
        "--variant",
        spec.workflow_variant,
        "--model-dir",
        str(paths["model_dir"]),
        "--config-path",
        str(paths["config_path"]),
        "--config-dir",
        str(paths["tokenizer_dir"]),
        "--quant-output-dir",
        str(paths["quant_output_dir"]),
        "--export-output-dir",
        str(paths["export_dir"]),
        "--device",
        args.device,
        "--dump-golden",
    ]
    if args.overwrite:
        command.append("--overwrite")
    return command


def build_validation_command(
    args: argparse.Namespace,
    spec: VariantSpec,
    paths: dict[str, Path],
) -> list[str]:
    command = [
        sys.executable,
        str(REPO_ROOT / "examples_merak/vla/pi05/pi05_droid_hmonnx_validation.py"),
        "--project-root",
        str(paths["project_root"]),
        "--workspace-root",
        str(REPO_ROOT),
        "--model-dir",
        str(paths["model_dir"]),
        "--export-dir",
        str(paths["export_dir"]),
        "--tokenizer-dir",
        str(paths["tokenizer_dir"]),
        "--inputs-dir",
        str(paths["inputs_dir"]),
        "--noise-file",
        str(paths["noise_file"]),
        "--results-dir",
        str(paths["results_dir"]),
        "--hmonnx-output",
        spec.hmonnx_output,
        "--reference-output",
        spec.reference_output,
        "--summary-output",
        spec.summary_output,
        "--start-sample-id",
        str(args.start_sample_id),
        "--num-samples",
        str(args.num_samples),
        "--device",
        args.device,
    ]
    if args.reference_jsonl is not None:
        command.extend(["--reference-jsonl", str(paths["reference_jsonl"])])
    if args.resume:
        command.append("--resume")
    return command


def verify_masked_add_graph(path: Path, num_hidden_layers: int) -> dict[str, Any]:
    import onnx

    model = onnx.load(str(path), load_external_data=False)
    add_nodes = [node for node in model.graph.node if node.op_type == "Add"]
    mask_add_nodes = [node for node in add_nodes if "attention_mask" in node.input]
    mask_add_outputs = {output for node in mask_add_nodes for output in node.output}
    chained_adds = sum(
        any(input_name in mask_add_outputs for input_name in node.input if input_name != "attention_mask")
        for node in mask_add_nodes
    )
    expected_mask_add_nodes = num_hidden_layers * 2
    if len(mask_add_nodes) != expected_mask_add_nodes or chained_adds != num_hidden_layers:
        raise ValueError(
            f"{path} does not contain two chained attention-mask Add nodes per layer: "
            f"layers={num_hidden_layers}, mask_add_nodes={len(mask_add_nodes)}, "
            f"chained_second_adds={chained_adds}"
        )
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "total_add_nodes": len(add_nodes),
        "attention_mask_add_nodes": len(mask_add_nodes),
        "chained_second_add_nodes": chained_adds,
        "num_hidden_layers": num_hidden_layers,
    }


def require_nonempty_directory(path: Path) -> None:
    require_directory(path)
    if not any(path.iterdir()):
        raise FileNotFoundError(f"Required directory is empty: {path}")


def artifact_record(path: Path) -> dict[str, Any]:
    require_file(path)
    record: dict[str, Any] = {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    external_data = path.with_name(f"{path.stem}_external_data")
    if external_data.is_file():
        record["external_data"] = {
            "path": str(external_data),
            "size_bytes": external_data.stat().st_size,
        }
    return record


def jsonl_record(path: Path) -> dict[str, Any]:
    require_file(path)
    with path.open(encoding="utf-8") as file:
        num_records = sum(1 for line in file if line.strip())
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "num_records": num_records,
    }


def verify_export(spec: VariantSpec, export_dir: Path) -> dict[str, Any]:
    meta_path = export_dir / "export_meta_info.json"
    require_file(meta_path)
    meta = read_json(meta_path)
    if meta.get("variant") != spec.metadata_variant:
        raise ValueError(
            f"Export variant mismatch: expected {spec.metadata_variant}, got {meta.get('variant')}"
        )
    if meta.get("target_device") != "XH2a":
        raise ValueError(f"Expected XH2a export, got {meta.get('target_device')}")
    if meta.get("compact_prefix") != spec.contract:
        raise ValueError(
            f"Compact-prefix contract mismatch: expected {spec.contract}, got {meta.get('compact_prefix')}"
        )

    components = meta.get("components")
    if not isinstance(components, dict) or set(components) != {"vision", "other", "gemma", "expert"}:
        raise ValueError("Export must contain vision, other, gemma, and expert components")

    artifacts = []
    golden_dirs = []

    vision = components["vision"]
    vision_path = export_dir / vision["component_dir"] / vision["hmonnx_file"]
    artifacts.append(artifact_record(vision_path))
    golden_dirs.append(vision_path.parent / "golden")

    other = components["other"]
    other_dir = export_dir / other["component_dir"]
    other_graphs = other.get("graphs", [])
    expected_other_graphs = {"action_in_proj", "action_out_proj", "time_mlp"}
    if {graph.get("name") for graph in other_graphs} != expected_other_graphs:
        raise ValueError(
            f"Other-component graph set mismatch: expected {sorted(expected_other_graphs)}"
        )
    for graph in other_graphs:
        graph_path = other_dir / graph["hmonnx_file"]
        artifacts.append(artifact_record(graph_path))
        golden_dirs.append(graph_path.parent / "golden" / graph["name"])

    masked_add_graphs = []
    for component_name in ("gemma", "expert"):
        component = components[component_name]
        component_dir = export_dir / component["component_dir"]
        component_meta = read_json(component_dir / component["meta_file"])
        num_hidden_layers = int(component_meta["num_hidden_layers"])
        for graph_key in ("prefill_onnx_file", "decode_onnx_file"):
            graph_path = component_dir / component_meta[graph_key]
            artifacts.append(artifact_record(graph_path))
            golden_dirs.append(graph_path.parent / "golden")
            graph_check = verify_masked_add_graph(graph_path, num_hidden_layers)
            graph_check["component"] = component_name
            graph_check["graph"] = graph_key.removesuffix("_onnx_file")
            masked_add_graphs.append(graph_check)

    for golden_dir in golden_dirs:
        require_nonempty_directory(golden_dir)
    if len(artifacts) != 8 or len(golden_dirs) != 8:
        raise ValueError(
            f"Expected 8 HMONNX artifacts and golden directories, got "
            f"{len(artifacts)} artifacts and {len(golden_dirs)} golden directories"
        )

    return {
        "export_dir": str(export_dir),
        "metadata_file": str(meta_path),
        "metadata_sha256": sha256_file(meta_path),
        "contract": spec.contract,
        "artifacts": artifacts,
        "golden_directories": [str(path) for path in golden_dirs],
        "masked_add_graphs": masked_add_graphs,
    }


def validate_summary(
    summary: dict[str, Any],
    spec: VariantSpec,
    sample_ids: list[int],
    noise_sha256: str,
    min_cosine_mean: float,
    max_mae_mean: float,
) -> dict[str, Any]:
    matched_ids = {int(value) for value in summary.get("matched_sample_ids", [])}
    missing_ids = sorted(set(sample_ids) - matched_ids)
    if missing_ids:
        raise ValueError(f"Validation summary is missing sample IDs: {missing_ids[:10]}")

    runtime_contract = summary.get("runtime_contract", {})
    expected_runtime = {
        "static_prefix_length": 712,
        "selected_image_indices": [0, 1],
        "vision_runs": 2,
        "horizon": spec.horizon,
        "noise_sha256": noise_sha256,
    }
    mismatches = {
        key: {"expected": expected, "actual": runtime_contract.get(key)}
        for key, expected in expected_runtime.items()
        if runtime_contract.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"Runtime contract mismatch: {mismatches}")

    aggregate = summary.get("aggregate", {})
    expected_shape = [spec.horizon, 8]
    if aggregate.get("shape") != expected_shape:
        raise ValueError(
            f"Validation action shape mismatch: expected {expected_shape}, got {aggregate.get('shape')}"
        )
    cosine_mean = float(aggregate.get("cosine", {}).get("mean", float("nan")))
    mae_mean = float(aggregate.get("mae", {}).get("mean", float("nan")))
    failures = []
    if not cosine_mean >= min_cosine_mean:
        failures.append(f"cosine mean {cosine_mean} < {min_cosine_mean}")
    if not mae_mean <= max_mae_mean:
        failures.append(f"MAE mean {mae_mean} > {max_mae_mean}")
    if failures:
        raise ValueError("Validation acceptance failed: " + "; ".join(failures))

    return {
        "passed": True,
        "criteria": {
            "min_cosine_mean": min_cosine_mean,
            "max_mae_mean": max_mae_mean,
            "required_sample_ids": sample_ids,
        },
        "observed": {
            "num_samples": int(aggregate["num_samples"]),
            "shape": aggregate["shape"],
            "cosine_mean": cosine_mean,
            "mae_mean": mae_mean,
        },
    }


def write_validation_report(
    spec: VariantSpec,
    export_verification: dict[str, Any],
    summary: dict[str, Any],
    summary_path: Path,
    manifest_path: Path,
    acceptance: dict[str, Any],
) -> Path:
    runtime = summary["runtime_contract"]
    aggregate = summary["aggregate"]
    export_dir = Path(export_verification["export_dir"])
    report_path = export_dir / "VALIDATION.md"
    summary_display = os.path.relpath(summary_path, export_dir)
    manifest_display = os.path.relpath(manifest_path, export_dir)
    title = (
        "PI0.5 DROID Customer H50 Validation"
        if spec.workflow_variant == "droid-customer-h50"
        else "PI0.5 Official OpenPI DROID H15 Validation"
    )
    lines = [
        f"# {title}",
        "",
        "## Status",
        "",
        "Passed on the fixed DROID samples using the compact-prefix XH2A HMONNX software runtime.",
        "",
        "## Contract",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| Variant | `{spec.metadata_variant}` |",
        f"| Selected image indices | `{runtime['selected_image_indices']}` |",
        f"| Static prefix | {runtime['static_prefix_length']} |",
        (
            "| Valid prefix range | "
            f"{runtime['valid_prefix_length']['min']} to {runtime['valid_prefix_length']['max']} |"
        ),
        (
            "| Valid language range | "
            f"{runtime['valid_language_length']['min']} to {runtime['valid_language_length']['max']} |"
        ),
        f"| Action horizon | {runtime['horizon']} |",
        f"| Action shape | `{aggregate['shape']}` |",
        f"| Cache capacity | {spec.contract['cache_length']} |",
        f"| Vision runs per sample | {runtime['vision_runs']} |",
        "| Flow-matching steps | 10 |",
        f"| Noise SHA256 | `{runtime['noise_sha256']}` |",
        "",
        (
            "The H15 run uses the first 15 rows of every fixed [50,32] noise entry."
            if spec.horizon == 15
            else "The customer graph overrides the source checkpoint action horizon from H15 to H50."
        ),
        "",
        "## Graph Checks",
        "",
        "| Component | Graph | Total Add | Mask Add | Chained second Add |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for graph in export_verification["masked_add_graphs"]:
        lines.append(
            f"| {graph['component']} | {graph['graph']} | {graph['total_add_nodes']} | "
            f"{graph['attention_mask_add_nodes']} | {graph['chained_second_add_nodes']} |"
        )

    lines.extend(
        [
            "",
            "Every Gemma/Expert graph contains two chained attention-mask Add nodes per hidden layer.",
            "",
            "## Results",
            "",
            "| Metric | Mean | Min | Max |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for metric, label in (
        ("cosine", "Cosine"),
        ("mae", "MAE"),
        ("rmse", "RMSE"),
        ("max_abs", "Max abs"),
        ("hmonnx_latency_ms_wall", "HMONNX wall latency (ms)"),
    ):
        values = aggregate[metric]
        lines.append(
            f"| {label} | {values['mean']:.8f} | {values['min']:.8f} | {values['max']:.8f} |"
        )

    criteria = acceptance["criteria"]
    lines.extend(
        [
            "",
            (
                f"Acceptance passed for {aggregate['num_samples']} samples: cosine mean "
                f"`>={criteria['min_cosine_mean']}`, MAE mean "
                f"`<={criteria['max_mae_mean']}`, and action shape `{aggregate['shape']}`."
            ),
            "",
            "Canonical files:",
            "",
            "```text",
            summary_display,
            manifest_display,
            "```",
            "",
            (
                "This report covers HMONNX software execution. Final XH2 HMM compilation and board "
                "execution require the separate compiler environment."
            ),
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def git_identity() -> dict[str, Any]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return {"revision": revision, "dirty": dirty}


def resolve_paths(args: argparse.Namespace, spec: VariantSpec) -> dict[str, Path]:
    project_root = resolve_path(args.project_root, DEFAULT_PROJECT_ROOT)
    export_dir = resolve_path(args.export_dir, spec.export_dir)
    results_dir = resolve_path(args.results_dir, export_dir / "validation_results")
    reference_jsonl = (
        resolve_path(args.reference_jsonl, args.reference_jsonl)
        if args.reference_jsonl is not None
        else Path()
    )
    return {
        "project_root": project_root,
        "model_dir": resolve_path(args.model_dir, project_root / "models" / DEFAULT_MODEL_NAME),
        "config_path": resolve_path(args.config_path, spec.config_path),
        "tokenizer_dir": resolve_path(args.tokenizer_dir),
        "inputs_dir": resolve_path(args.inputs_dir, project_root / "inputs"),
        "noise_file": resolve_path(args.noise_file, project_root / "noise" / DEFAULT_NOISE_NAME),
        "export_dir": export_dir,
        "quant_output_dir": resolve_path(args.quant_output_dir, "work_dirs/pi05_quant"),
        "results_dir": results_dir,
        "reference_jsonl": reference_jsonl,
        "manifest_output": resolve_path(args.manifest_output, export_dir / "fae_delivery_manifest.json"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=sorted(VARIANTS), default="droid-customer-h50")
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--config-path", type=Path, default=None)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--inputs-dir", type=Path, default=None)
    parser.add_argument("--noise-file", type=Path, default=None)
    parser.add_argument("--export-dir", type=Path, default=None)
    parser.add_argument("--quant-output-dir", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--manifest-output", type=Path, default=None)
    parser.add_argument("--reference-jsonl", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--start-sample-id", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--min-cosine-mean", type=float, default=0.99)
    parser.add_argument("--max-mae-mean", type=float, default=0.02)
    parser.add_argument("--expected-noise-sha256", default=EXPECTED_NOISE_SHA256)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument(
        "--reuse-validation",
        action="store_true",
        help="Verify existing JSONL/summary outputs without launching inference.",
    )
    parser.add_argument("--hash-model", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.start_sample_id < 0:
        parser.error("--start-sample-id must be non-negative")
    if args.num_samples <= 0:
        parser.error("--num-samples must be positive")
    if args.skip_validation and args.reuse_validation:
        parser.error("--skip-validation and --reuse-validation are mutually exclusive")
    return args


def main() -> int:
    args = parse_args()
    spec = VARIANTS[args.variant]
    paths = resolve_paths(args, spec)
    checks = preflight(args, spec, paths)
    export_command = build_export_command(args, spec, paths)
    validation_command = build_validation_command(args, spec, paths)

    if args.check_only:
        print(json.dumps({"status": "preflight_passed", **checks}, indent=2))
        return 0
    if args.dry_run:
        if not args.skip_export:
            print(f"+ {command_text(export_command)}")
        if not args.skip_validation and not args.reuse_validation:
            print(f"+ {command_text(validation_command)}")
        elif args.reuse_validation:
            print(f"# reuse validation: {paths['results_dir'] / spec.summary_output}")
        return 0

    commands = []
    if not args.skip_export:
        run_command(export_command)
        commands.append(command_text(export_command))
    export_verification = verify_export(spec, paths["export_dir"])

    validation = None
    if not args.skip_validation:
        if not args.reuse_validation:
            run_command(validation_command)
            commands.append(command_text(validation_command))
        summary_path = paths["results_dir"] / spec.summary_output
        require_file(summary_path)
        summary = read_json(summary_path)
        hmonnx_records_path = paths["results_dir"] / spec.hmonnx_output
        expected_hmonnx_path = hmonnx_records_path.resolve()
        actual_hmonnx_path = Path(summary.get("hmonnx_file", "")).resolve()
        if actual_hmonnx_path != expected_hmonnx_path:
            raise ValueError(
                f"Summary HMONNX records path mismatch: expected {expected_hmonnx_path}, "
                f"got {actual_hmonnx_path}"
            )
        reference_records_path = Path(summary.get("reference_file", "")).resolve()
        acceptance = validate_summary(
            summary=summary,
            spec=spec,
            sample_ids=expected_sample_ids(args),
            noise_sha256=checks["dataset"]["noise_sha256"],
            min_cosine_mean=args.min_cosine_mean,
            max_mae_mean=args.max_mae_mean,
        )
        report_path = write_validation_report(
            spec=spec,
            export_verification=export_verification,
            summary=summary,
            summary_path=summary_path,
            manifest_path=paths["manifest_output"],
            acceptance=acceptance,
        )
        validation = {
            "reused": args.reuse_validation,
            "summary_file": str(summary_path),
            "summary_sha256": sha256_file(summary_path),
            "hmonnx_records": jsonl_record(hmonnx_records_path),
            "reference_records": jsonl_record(reference_records_path),
            "report": artifact_record(report_path),
            "runtime_contract": summary["runtime_contract"],
            "aggregate": summary["aggregate"],
            "acceptance": acceptance,
        }

    manifest = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if validation is not None else "export_verified",
        "variant": spec.workflow_variant,
        "repository": git_identity(),
        "execution": {
            "export": "reused" if args.skip_export else "executed",
            "validation": (
                "skipped"
                if args.skip_validation
                else "reused"
                if args.reuse_validation
                else "executed"
            ),
        },
        "commands": commands,
        "preflight": checks,
        "export": export_verification,
        "validation": validation,
        "boundary": (
            "HMONNX software runtime only; final XH2 HMM compilation and board execution "
            "require the separate XH2 compiler environment."
        ),
    }
    paths["manifest_output"].parent.mkdir(parents=True, exist_ok=True)
    paths["manifest_output"].write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"FAE handoff passed: {paths['manifest_output']}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        FileNotFoundError,
        KeyError,
        RuntimeError,
        subprocess.CalledProcessError,
        TypeError,
        ValueError,
    ) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error
