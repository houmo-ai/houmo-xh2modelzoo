#!/usr/bin/env python3
"""Inventory Wan2.2 A14B model directories without loading weights."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


DEFAULT_T2V_MODEL_DIR = Path("/data01/datasets/Wan2.2-T2V-A14B")
I2V_PENDING_MESSAGE = "未提供本地 I2V，仅基于 README/ModelScope 待补充"
NOISE_MODEL_DIRS = ("high_noise_model", "low_noise_model")
TOP_LEVEL_WEIGHT_FILES = (
    "Wan2.1_VAE.pth",
    "models_t5_umt5-xxl-enc-bf16.pth",
)
SKIP_DIRS = {".git", "__pycache__", "assets"}


def human_size(num_bytes: int | None) -> str:
    if num_bytes is None:
        return "n/a"
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} TiB"


def read_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.exists():
        return None, "missing"
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception as exc:  # pragma: no cover - defensive CLI reporting
        return None, f"{exc.__class__.__name__}: {exc}"
    if not isinstance(data, dict):
        return None, "json root is not an object"
    return data, None


def file_record(path: Path, root: Path) -> dict[str, Any]:
    exists = path.exists()
    size_bytes = path.stat().st_size if exists else None
    try:
        relpath = str(path.relative_to(root))
    except ValueError:
        relpath = str(path)
    return {
        "path": relpath,
        "exists": exists,
        "size_bytes": size_bytes,
        "size": human_size(size_bytes),
    }


def collect_weight_files(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not root.exists():
        return records
    for current_root, dirnames, filenames in os.walk(root):
        dirnames[:] = [name for name in dirnames if name not in SKIP_DIRS]
        current = Path(current_root)
        for filename in sorted(filenames):
            if not filename.endswith((".safetensors", ".pth")):
                continue
            records.append(file_record(current / filename, root))
    records.sort(key=lambda item: item["path"])
    return records


def summarize_configuration(root: Path) -> dict[str, Any]:
    path = root / "configuration.json"
    data, error = read_json(path)
    summary: dict[str, Any] = {
        "path": "configuration.json",
        "exists": path.exists(),
        "error": error,
    }
    if data is None:
        return summary
    model_files = data.get("model_file_location")
    summary.update(
        {
            "framework": data.get("framework"),
            "task": data.get("task"),
            "aigc_model": data.get("aigc_model"),
            "model_file_location_count": len(model_files)
            if isinstance(model_files, list)
            else None,
            "top_level_keys": sorted(data.keys()),
        }
    )
    return summary


def summarize_safetensors_index(index_path: Path, component_root: Path) -> dict[str, Any]:
    data, error = read_json(index_path)
    summary: dict[str, Any] = {
        "path": index_path.name,
        "exists": index_path.exists(),
        "error": error,
    }
    if data is None:
        return summary

    metadata = data.get("metadata")
    weight_map = data.get("weight_map")
    total_size = None
    if isinstance(metadata, dict) and isinstance(metadata.get("total_size"), int):
        total_size = metadata["total_size"]
    shard_names = sorted(set(weight_map.values())) if isinstance(weight_map, dict) else []
    missing_shards = [
        shard_name for shard_name in shard_names if not (component_root / shard_name).exists()
    ]
    summary.update(
        {
            "metadata_total_size_bytes": total_size,
            "metadata_total_size": human_size(total_size),
            "weight_count": len(weight_map) if isinstance(weight_map, dict) else None,
            "shard_count": len(shard_names),
            "shards": shard_names,
            "missing_shards": missing_shards,
        }
    )
    return summary


def summarize_noise_model(root: Path, name: str) -> dict[str, Any]:
    component_root = root / name
    config_path = component_root / "config.json"
    config, config_error = read_json(config_path)
    shards = sorted(component_root.glob("*.safetensors")) if component_root.exists() else []
    shard_records = [file_record(path, root) for path in shards]
    shard_total = sum(item["size_bytes"] or 0 for item in shard_records)
    summary: dict[str, Any] = {
        "name": name,
        "path": name,
        "exists": component_root.exists(),
        "config": {
            "path": f"{name}/config.json",
            "exists": config_path.exists(),
            "error": config_error,
        },
        "safetensors_files": shard_records,
        "safetensors_total_bytes": shard_total,
        "safetensors_total": human_size(shard_total),
        "index": summarize_safetensors_index(
            component_root / "diffusion_pytorch_model.safetensors.index.json",
            component_root,
        ),
    }
    if config is not None:
        summary["config"].update(
            {
                "class_name": config.get("_class_name"),
                "diffusers_version": config.get("_diffusers_version"),
                "model_type": config.get("model_type"),
                "dim": config.get("dim"),
                "ffn_dim": config.get("ffn_dim"),
                "num_heads": config.get("num_heads"),
                "num_layers": config.get("num_layers"),
                "text_len": config.get("text_len"),
                "in_dim": config.get("in_dim"),
                "out_dim": config.get("out_dim"),
            }
        )
    return summary


def summarize_model_dir(root: Path, label: str, missing_message: str | None = None) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "label": label,
        "root": str(root),
        "exists": root.exists(),
        "message": None,
    }
    if not root.exists():
        summary["message"] = missing_message or "model directory does not exist"
        return summary

    components = [summarize_noise_model(root, name) for name in NOISE_MODEL_DIRS]
    top_level_files = [file_record(root / filename, root) for filename in TOP_LEVEL_WEIGHT_FILES]
    weight_files = collect_weight_files(root)
    summary.update(
        {
            "configuration": summarize_configuration(root),
            "components": components,
            "top_level_weight_files": top_level_files,
            "weight_file_count": len(weight_files),
            "weight_total_bytes": sum(item["size_bytes"] or 0 for item in weight_files),
            "weight_total": human_size(sum(item["size_bytes"] or 0 for item in weight_files)),
            "weight_files": weight_files,
        }
    )
    return summary


def build_report(model_dir: Path, i2v_model_dir: Path | None) -> dict[str, Any]:
    i2v_root = i2v_model_dir if i2v_model_dir is not None else Path("<not-provided>")
    return {
        "schema_version": 1,
        "note": "Inventory only. This script does not load weights or export HMONNX.",
        "t2v": summarize_model_dir(model_dir, "Wan2.2-T2V-A14B"),
        "i2v": summarize_model_dir(
            i2v_root,
            "Wan2.2-I2V-A14B",
            missing_message=I2V_PENDING_MESSAGE,
        ),
    }


def print_component_table(section: dict[str, Any]) -> None:
    print(f"\n[{section['label']}]")
    print(f"root: {section['root']}")
    if not section.get("exists"):
        print(f"status: {section.get('message')}")
        return

    config = section.get("configuration") or {}
    print(
        "configuration: "
        f"task={config.get('task') or 'n/a'}, "
        f"framework={config.get('framework') or 'n/a'}, "
        f"model_files={config.get('model_file_location_count') or 'n/a'}"
    )
    print(f"weights: count={section.get('weight_file_count')}, total={section.get('weight_total')}")
    print("component          exists  type  layers  heads  shards  index_total")
    print("-----------------  ------  ----  ------  -----  ------  -----------")
    for component in section.get("components", []):
        cfg = component.get("config") or {}
        index = component.get("index") or {}
        print(
            f"{component.get('name', 'n/a'):<17}  "
            f"{str(component.get('exists')):<6}  "
            f"{str(cfg.get('model_type') or 'n/a'):<4}  "
            f"{str(cfg.get('num_layers') or 'n/a'):<6}  "
            f"{str(cfg.get('num_heads') or 'n/a'):<5}  "
            f"{str(index.get('shard_count') or len(component.get('safetensors_files', []))):<6}  "
            f"{index.get('metadata_total_size') or 'n/a'}"
        )
    for record in section.get("top_level_weight_files", []):
        print(
            f"file: {record['path']:<38} exists={record['exists']} size={record['size']}"
        )


def print_report(report: dict[str, Any]) -> None:
    print("Wan2.2 A14B inventory summary")
    print(report["note"])
    print_component_table(report["t2v"])
    print_component_table(report["i2v"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze local Wan2.2 A14B directory structure without loading weights."
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_T2V_MODEL_DIR,
        help=f"T2V model directory. Default: {DEFAULT_T2V_MODEL_DIR}",
    )
    parser.add_argument(
        "--i2v-model-dir",
        type=Path,
        default=None,
        help="Optional local Wan2.2-I2V-A14B model directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path for the full inventory report.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_report(args.model_dir, args.i2v_model_dir)
    print_report(report)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        print(f"\nJSON written to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
