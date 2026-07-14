#!/usr/bin/env python3
"""Merak model card and model delivery flow utilities.

The module keeps shared implementation in one place. Dedicated scripts in this
directory import ``main`` with a fixed subcommand so each flow stage can also be
called independently from automation.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from core import MerakDeliveryStore, MerakEvaluator, MerakModelFlow, MerakModelFlowCLI
from core.bindings import configure as configure_core

ROOT = Path(__file__).resolve().parents[2]
DELIVERY_ROOT = ROOT / "merak_delivery"
WORK_DIRS_DELIVERY_ROOT = ROOT / "work_dirs" / "merak_delivery"
SCHEMA_PATH = DELIVERY_ROOT / "schemas/merak_model_card.schema.json"
REQUIRED_TOP_LEVEL_KEYS = ("schema_version", "model", "source", "workflow", "runtime", "frontend", "release")
EXPECTED_ACTIONS = ["quant", "export", "dump_golden", "eval"]


class DeliveryValidationError(ValueError):
    """Raised when a model card YAML file does not satisfy phase-1 rules."""


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise DeliveryValidationError(f"{path}: invalid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise DeliveryValidationError(f"{path}: expected a YAML mapping at document root")
    return data


def _require_mapping(data: dict[str, Any], key: str, path: Path) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise DeliveryValidationError(f"{path}: `{key}` must be a mapping")
    return value


def _require_non_empty_string(mapping: dict[str, Any], key: str, path: Path, prefix: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise DeliveryValidationError(f"{path}: `{prefix}.{key}` must be a non-empty string")
    return value


def _require_list(mapping: dict[str, Any], key: str, path: Path, prefix: str) -> list[Any]:
    value = mapping.get(key)
    if not isinstance(value, list) or not value:
        raise DeliveryValidationError(f"{path}: `{prefix}.{key}` must be a non-empty list")
    return value


def validate_delivery_file(path: Path) -> dict[str, Any]:
    """Validate one phase-1 Merak model card YAML file.

    This intentionally avoids a hard jsonschema dependency. The JSON schema file
    is still emitted as the public contract, while this script performs the
    minimal checks needed for the first working stage.
    """
    data = _load_yaml(path)

    for key in REQUIRED_TOP_LEVEL_KEYS:
        if key not in data:
            raise DeliveryValidationError(f"{path}: missing top-level key `{key}`")

    if data["schema_version"] != 1:
        raise DeliveryValidationError(f"{path}: `schema_version` must be 1")

    model = _require_mapping(data, "model", path)
    model_id = _require_non_empty_string(model, "id", path, "model")
    _require_non_empty_string(model, "family", path, "model")
    _require_non_empty_string(model, "display_name", path, "model")
    _require_list(model, "modality", path, "model")
    _require_list(model, "task", path, "model")
    tags = model.get("tags")
    if tags is not None and not isinstance(tags, list):
        raise DeliveryValidationError(f"{path}: `model.tags` must be a list")

    source = _require_mapping(data, "source", path)
    _require_non_empty_string(source, "provider", path, "source")
    _require_non_empty_string(source, "name", path, "source")
    _require_non_empty_string(source, "raw_model_path", path, "source")

    workflow = _require_mapping(data, "workflow", path)
    config_path = _require_non_empty_string(workflow, "config_path", path, "workflow")
    _require_non_empty_string(workflow, "model_dir", path, "workflow")
    workflow_class = _require_non_empty_string(workflow, "class", path, "workflow")
    if workflow_class != "auto":
        raise DeliveryValidationError(f"{path}: phase-1 only supports `workflow.class: auto`")
    actions = _require_list(workflow, "actions", path, "workflow")
    if actions != EXPECTED_ACTIONS:
        raise DeliveryValidationError(f"{path}: `workflow.actions` must be {EXPECTED_ACTIONS!r}")
    if not (ROOT / config_path).is_file():
        raise DeliveryValidationError(f"{path}: workflow config does not exist: {config_path}")

    runtime = _require_mapping(data, "runtime", path)
    _require_non_empty_string(runtime, "device", path, "runtime")
    if not isinstance(runtime.get("seed"), int) or runtime["seed"] < 0:
        raise DeliveryValidationError(f"{path}: `runtime.seed` must be a non-negative integer")
    work_dir = _require_non_empty_string(runtime, "work_dir", path, "runtime")
    if not work_dir.startswith("work_dirs/merak_delivery/"):
        raise DeliveryValidationError(f"{path}: `runtime.work_dir` must be under work_dirs/merak_delivery/")

    frontend = _require_mapping(data, "frontend", path)
    _require_list(frontend, "inputs", path, "frontend")
    _require_list(frontend, "outputs", path, "frontend")
    demo = _require_mapping(frontend, "demo", path)
    _require_non_empty_string(demo, "kind", path, "frontend.demo")
    limitations = frontend.get("limitations")
    if limitations is not None and not isinstance(limitations, list):
        raise DeliveryValidationError(f"{path}: `frontend.limitations` must be a list")

    release = _require_mapping(data, "release", path)
    version_id = _require_non_empty_string(release, "version_id", path, "release")
    if not version_id.startswith(model_id):
        raise DeliveryValidationError(f"{path}: `release.version_id` must start with model id `{model_id}`")
    _require_non_empty_string(release, "target_status", path, "release")

    return data


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _relative_or_str(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def _load_model_card(path: Path) -> dict[str, Any]:
    return validate_delivery_file(path)


def _extract_export_summary(workflow_config_path: Path, golden_meta_path: Path) -> dict[str, Any]:
    workflow_config = _load_yaml(workflow_config_path)
    export_model = workflow_config.get("export", {}).get("model", {})
    quant_scheme = export_model.get("quant_scheme", {}) if isinstance(export_model, dict) else {}
    summary: dict[str, Any] = {
        "chip_arch": export_model.get("chip_arch", ""),
        "model_type": export_model.get("model_type", ""),
        "model_name": export_model.get("model_name", ""),
        "context_max_length": export_model.get("context_max_length"),
        "prefill_chunk_length": export_model.get("prefill_chunk_length"),
        "max_pe_length": export_model.get("max_pe_length"),
        "quant_type": quant_scheme.get("quant_type", ""),
        "subgraphs": [],
    }
    if golden_meta_path.is_file():
        golden_meta = _read_json(golden_meta_path)
        subgraphs = golden_meta.get("subgraphs")
        if isinstance(subgraphs, list):
            summary["subgraphs"] = subgraphs
        model_config = golden_meta.get("model_config")
        if isinstance(model_config, dict):
            summary["context_max_length"] = model_config.get("context_max_length", summary["context_max_length"])
            summary["prefill_chunk_length"] = model_config.get(
                "prefill_chunk_length", summary["prefill_chunk_length"]
            )
    return summary


def _extract_quant_summary(workflow_config_path: Path) -> dict[str, Any]:
    workflow_config = _load_yaml(workflow_config_path)
    quant = workflow_config.get("quant")
    if quant is None:
        return {"skipped": True}
    if not isinstance(quant, dict):
        return {"skipped": False, "raw": quant}
    return {
        "skipped": False,
        "algorithm": quant.get("algorithm", ""),
        "method": quant.get("method", ""),
        "bits": quant.get("bits"),
        "group_size": quant.get("group_size"),
        "artifact_format": quant.get("artifact_format", ""),
        "output_format": quant.get("output_format", ""),
    }


def collect_manifest_command(args: argparse.Namespace) -> int:
    model_card_path = _resolve_path(args.model_card)
    card = _load_model_card(model_card_path)
    workflow_config_path = ROOT / card["workflow"]["config_path"]
    work_dir = _resolve_path(args.work_dir)
    export_dir = _resolve_path(args.export_dir)
    golden_meta_path = _resolve_path(args.golden_meta)
    eval_report_path = _resolve_path(args.eval_report) if args.eval_report else None

    manifest = {
        "schema_version": 1,
        "model_id": card["model"]["id"],
        "version_id": card["release"]["version_id"],
        "workflow": {
            "config_path": card["workflow"]["config_path"],
            "model_card": _relative_or_str(model_card_path),
            "workflow_class": card["workflow"]["class"],
            "actions": card["workflow"]["actions"],
            "seed": card["runtime"]["seed"],
        },
        "artifacts": {
            "raw_model_path": card["source"]["raw_model_path"],
            "quanted_model_path": args.quanted_model_path or "",
            "hmonnx_export_path": str(export_dir),
            "golden_meta_info": str(golden_meta_path),
        },
        "quant": _extract_quant_summary(workflow_config_path),
        "export": _extract_export_summary(workflow_config_path, golden_meta_path),
        "evaluation": {
            "status": "provided" if eval_report_path else "pending",
            "reports": [str(eval_report_path)] if eval_report_path else [],
        },
    }
    output = work_dir / "delivery_manifest.json"
    _write_json(output, manifest)
    print(f"Wrote {_relative_or_str(output)}")
    return 0


def _default_work_dir(card: dict[str, Any]) -> Path:
    return _resolve_path(card["runtime"]["work_dir"]) / card["release"]["version_id"]


def _default_release_root() -> Path:
    return WORK_DIRS_DELIVERY_ROOT / "releases" / "merak"


def _default_catalog_root() -> Path:
    return WORK_DIRS_DELIVERY_ROOT / "model_catalog"


def _default_quant_dir(work_dir: Path) -> Path:
    return work_dir / "quant"


def _default_export_dir(work_dir: Path) -> Path:
    return work_dir / "export"


def _find_golden_meta(export_dir: Path) -> Path:
    direct = export_dir / "golden_meta_info.json"
    if direct.is_file():
        return direct
    matches = sorted(export_dir.rglob("golden_meta_info.json")) if export_dir.is_dir() else []
    return matches[0] if matches else direct


def _default_eval_report(work_dir: Path, backend: str = "hmonnx") -> Path:
    filename = "eval_report_float.json" if backend == "float" else "eval_report.json"
    return work_dir / filename


def _default_eval_work_dir(work_dir: Path) -> Path:
    return work_dir / "hm_eval"


def _infer_eval_model_id(card: dict[str, Any], explicit_model: str = "") -> str:
    if explicit_model:
        return explicit_model
    evaluation = card.get("evaluation")
    if isinstance(evaluation, dict) and isinstance(evaluation.get("model"), str) and evaluation["model"]:
        return evaluation["model"]
    return card["model"]["id"]


def _infer_eval_datasets(card: dict[str, Any], explicit_datasets: list[str] | None = None) -> list[str]:
    if explicit_datasets:
        return explicit_datasets
    evaluation = card.get("evaluation")
    if isinstance(evaluation, dict) and isinstance(evaluation.get("datasets"), list) and evaluation["datasets"]:
        return [str(item) for item in evaluation["datasets"]]
    return ["ceval"]


def _dataset_statuses(results: dict[str, Any]) -> list[dict[str, Any]]:
    tasks = []
    datasets = results.get("datasets", {})
    if not isinstance(datasets, dict):
        return tasks
    for dataset, dataset_result in datasets.items():
        if not isinstance(dataset_result, dict):
            tasks.append({"dataset": dataset, "status": "failed", "error": "invalid dataset result"})
            continue
        tasks.append(
            {
                "dataset": dataset,
                "status": dataset_result.get("status", "unknown"),
                "metrics": dataset_result.get("metrics", {}),
                "subset_scores": dataset_result.get("subset_scores", {}),
                "elapsed_seconds": dataset_result.get("elapsed_seconds", 0),
                "report_files": dataset_result.get("report_files", []),
                "error": dataset_result.get("error", ""),
            }
        )
    return tasks


def _derive_eval_report_status(tasks: list[dict[str, Any]]) -> str:
    if not tasks:
        return "failed"
    return "passed" if all(task.get("status") == "completed" for task in tasks) else "failed"


def _configure_hmonnx_backend(
    model_config: Any,
    hmonnx_meta: Path | None,
    vision_hmonnx_meta: Path | None,
) -> None:
    from hm_eval.core.model_registry import BackendConfig

    backend_cfg = model_config.backends.get("hmonnx")
    if backend_cfg is None:
        if hmonnx_meta is None:
            raise DeliveryValidationError(
                f"hm_eval hmonnx backend is not configured for model: {model_config.config_id}"
            )
        backend_cfg = BackendConfig(type="hmonnx")
        model_config.backends["hmonnx"] = backend_cfg
    if hmonnx_meta is not None:
        backend_cfg.export_meta_info = str(hmonnx_meta)
    if vision_hmonnx_meta is not None:
        backend_cfg.vision_export_meta_info = str(vision_hmonnx_meta)


def _run_hm_eval(
    model_id: str,
    backend_type: str,
    datasets: list[str],
    dataset_hub: str,
    work_dir: Path,
    limit: int,
    max_tokens: int,
    hmonnx_meta: Path | None,
    vision_hmonnx_meta: Path | None,
) -> dict[str, Any]:
    from hm_eval.core.backends import create_backend
    from hm_eval.core.dataset_registry import DatasetRegistry
    from hm_eval.core.eval_runner import run_evaluation
    from hm_eval.core.model_registry import ModelRegistry

    registry = ModelRegistry()
    registry.scan()
    model_config = registry.get_model(model_id) or registry.get_model_by_display_name(model_id)
    if model_config is None:
        available = [model.config_id for model in registry.list_models()]
        raise DeliveryValidationError(f"hm_eval model config not found: {model_id}; available={available}")

    if backend_type == "hmonnx":
        _configure_hmonnx_backend(model_config, hmonnx_meta, vision_hmonnx_meta)

    backend = create_backend(backend_type, model_config)
    try:
        return run_evaluation(
            backend=backend,
            model_display_name=model_config.display_name,
            datasets=datasets,
            work_dir=str(work_dir),
            dataset_registry=DatasetRegistry(),
            dataset_hub=dataset_hub,
            limit=limit,
            max_tokens=max_tokens if max_tokens > 0 else model_config.max_tokens,
        )
    finally:
        backend.cleanup()


def run_eval_command(args: argparse.Namespace) -> int:
    return MerakEvaluator.from_args(args).run()


def _run_collect_manifest(
    model_card: Path,
    work_dir: Path,
    export_dir: Path,
    golden_meta: Path,
    eval_report: Path | None = None,
    quanted_model_path: str = "",
) -> int:
    return collect_manifest_command(
        argparse.Namespace(
            model_card=str(model_card),
            work_dir=str(work_dir),
            export_dir=str(export_dir),
            golden_meta=str(golden_meta),
            eval_report=str(eval_report) if eval_report else "",
            quanted_model_path=quanted_model_path,
        )
    )


def _run_check_artifact(manifest: Path) -> int:
    return check_artifact_command(argparse.Namespace(manifest=str(manifest)))


def _run_register_release(
    manifest: Path,
    artifact_check: Path,
    eval_report: Path | None = None,
    release_root: Path | None = None,
) -> int:
    return register_release_command(
        argparse.Namespace(
            manifest=str(manifest),
            artifact_check=str(artifact_check),
            eval_report=str(eval_report) if eval_report else "",
            release_root=str(release_root) if release_root else "",
        )
    )


def _run_build_catalog(release_root: Path | None = None, catalog_root: Path | None = None) -> int:
    return build_catalog_command(
        argparse.Namespace(
            release_root=str(release_root) if release_root else "",
            catalog_root=str(catalog_root) if catalog_root else "",
        )
    )


def _run_render_readme(
    model_id: str,
    release_root: Path | None = None,
    catalog_root: Path | None = None,
) -> int:
    return render_readme_command(
        argparse.Namespace(
            model_id=model_id,
            release_root=str(release_root) if release_root else "",
            catalog_root=str(catalog_root) if catalog_root else "",
        )
    )


def _run_eval(
    model_card: Path,
    work_dir: Path,
    eval_output: Path,
    eval_work_dir: Path,
    eval_model: str,
    eval_backend: str,
    eval_datasets: list[str],
    eval_dataset_hub: str,
    eval_limit: int,
    eval_max_tokens: int,
    hmonnx_meta: Path | None,
    vision_hmonnx_meta: Path | None,
) -> int:
    return run_eval_command(
        argparse.Namespace(
            model_card=str(model_card),
            work_dir=str(work_dir),
            output=str(eval_output),
            eval_work_dir=str(eval_work_dir),
            eval_model=eval_model,
            eval_backend=eval_backend,
            eval_datasets=eval_datasets,
            eval_dataset_hub=eval_dataset_hub,
            eval_limit=eval_limit,
            eval_max_tokens=eval_max_tokens,
            hmonnx_meta=str(hmonnx_meta) if hmonnx_meta else "",
            vision_hmonnx_meta=str(vision_hmonnx_meta) if vision_hmonnx_meta else "",
        )
    )


def _serialize_result_path(result: Any, default_dir: Path) -> str:
    if result is None:
        return ""
    quanted_model_dir = getattr(result, "quanted_model_dir", None)
    if quanted_model_dir:
        return str(quanted_model_dir)
    work_dir = getattr(result, "work_dir", None)
    if work_dir:
        return str(work_dir)
    return str(default_dir)


def run_workflow_command(args: argparse.Namespace) -> int:
    return MerakModelFlow(args).run()


def _check_file(name: str, path: Path) -> dict[str, str]:
    return {
        "name": name,
        "status": "passed" if path.is_file() else "failed",
        "evidence": str(path),
        "message": "" if path.is_file() else "file not found",
    }


def _check_dir(name: str, path: Path) -> dict[str, str]:
    return {
        "name": name,
        "status": "passed" if path.is_dir() else "failed",
        "evidence": str(path),
        "message": "" if path.is_dir() else "directory not found",
    }


def check_artifact_command(args: argparse.Namespace) -> int:
    manifest_path = _resolve_path(args.manifest)
    manifest = _read_json(manifest_path)
    export_path = Path(manifest["artifacts"]["hmonnx_export_path"])
    golden_meta = Path(manifest["artifacts"]["golden_meta_info"])
    checks = [_check_file("golden_meta_info", golden_meta)]
    for subgraph in manifest.get("export", {}).get("subgraphs", []):
        checks.append(_check_dir(f"{subgraph}_subgraph", golden_meta.parent / str(subgraph)))
    if not manifest.get("export", {}).get("subgraphs"):
        checks.append(_check_dir("export_dir", export_path))
    status = "passed" if all(item["status"] == "passed" for item in checks) else "failed"
    result = {
        "schema_version": 1,
        "model_id": manifest["model_id"],
        "version_id": manifest["version_id"],
        "status": status,
        "checks": checks,
    }
    output = manifest_path.parent / "artifact_check.json"
    _write_json(output, result)
    print(f"Wrote {_relative_or_str(output)}")
    return 0 if status == "passed" else 1


def _gate(status: str, evidence: str) -> dict[str, str]:
    return {"status": status, "evidence": evidence}


def _eval_status(eval_report_path: Path | None) -> str:
    if eval_report_path is None or not eval_report_path.is_file():
        return "pending"
    report = _read_json(eval_report_path)
    return "passed" if report.get("status") == "passed" else "failed"


def _derive_release_status(gates: dict[str, dict[str, str]]) -> str:
    if any(gate["status"] == "failed" for gate in gates.values()):
        return "blocked"
    if gates.get("accuracy_valid", {}).get("status") == "passed":
        return "accuracy_passed"
    if gates.get("artifact_valid", {}).get("status") == "passed":
        return "artifact_passed"
    if gates.get("golden_valid", {}).get("status") == "passed":
        return "golden_ready"
    if gates.get("workflow_exported", {}).get("status") == "passed":
        return "exported"
    return "ready"


def register_release_command(args: argparse.Namespace) -> int:
    manifest_path = _resolve_path(args.manifest)
    artifact_check_path = _resolve_path(args.artifact_check)
    eval_report_path = _resolve_path(args.eval_report) if args.eval_report else None
    manifest = _read_json(manifest_path)
    artifact_check = _read_json(artifact_check_path) if artifact_check_path.is_file() else {"status": "failed"}
    model_card = manifest["workflow"].get("model_card", "")
    gates = {
        "metadata_valid": _gate("passed" if model_card else "pending", model_card),
        "workflow_exported": _gate("passed", str(manifest_path)),
        "golden_valid": _gate("passed" if Path(manifest["artifacts"]["golden_meta_info"]).is_file() else "failed", manifest["artifacts"]["golden_meta_info"]),
        "artifact_valid": _gate("passed" if artifact_check.get("status") == "passed" else "failed", str(artifact_check_path)),
        "accuracy_valid": _gate(_eval_status(eval_report_path), str(eval_report_path) if eval_report_path else ""),
        "compiler_valid": _gate("pending", ""),
    }
    release_state = {
        "schema_version": 1,
        "model_id": manifest["model_id"],
        "version_id": manifest["version_id"],
        "status": _derive_release_status(gates),
        "artifacts": {
            "manifest": str(manifest_path),
            "artifact_check": str(artifact_check_path),
            "eval_report": str(eval_report_path) if eval_report_path else "",
        },
        "gates": gates,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    release_root = _resolve_path(getattr(args, "release_root", "")) if getattr(args, "release_root", "") else _default_release_root()
    release_dir = release_root / manifest["model_id"]
    output = release_dir / f"{manifest['version_id']}.yaml"
    latest = release_dir / "latest.yaml"
    _write_yaml(output, release_state)
    _write_yaml(latest, {"latest_version_id": manifest["version_id"], "latest_file": output.name})
    print(f"Wrote {_relative_or_str(output)}")
    return 0


def _find_model_cards() -> list[Path]:
    return sorted((DELIVERY_ROOT / "model_cards" / "merak").rglob("*.yaml"))


def _latest_release_for(model_id: str, release_root: Path | None = None) -> tuple[dict[str, Any] | None, Path | None]:
    release_dir = (release_root or _default_release_root()) / model_id
    latest = release_dir / "latest.yaml"
    if latest.is_file():
        latest_data = yaml.safe_load(latest.read_text(encoding="utf-8")) or {}
        release_file = release_dir / latest_data.get("latest_file", "")
        if release_file.is_file():
            return yaml.safe_load(release_file.read_text(encoding="utf-8")), release_file
    candidates = sorted(path for path in release_dir.glob("*.yaml") if path.name != "latest.yaml")
    if not candidates:
        return None, None
    release_file = candidates[-1]
    return yaml.safe_load(release_file.read_text(encoding="utf-8")), release_file


def build_catalog_command(args: argparse.Namespace) -> int:
    generated_at = datetime.now(timezone.utc).isoformat()
    release_root = _resolve_path(getattr(args, "release_root", "")) if getattr(args, "release_root", "") else _default_release_root()
    catalog_root = _resolve_path(getattr(args, "catalog_root", "")) if getattr(args, "catalog_root", "") else _default_catalog_root()
    data_dir = catalog_root / "data"
    models_dir = data_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    index_models = []
    all_models = []
    for card_path in _find_model_cards():
        card = _load_model_card(card_path)
        model_id = card["model"]["id"]
        release, release_path = _latest_release_for(model_id, release_root)
        versions = []
        latest_status = "draft"
        if release:
            latest_status = release.get("status", "draft")
            versions.append(release)
        detail = {
            "schema_version": 1,
            "generated_at": generated_at,
            "model_id": model_id,
            "model": card["model"],
            "source": card["source"],
            "workflow": card["workflow"],
            "frontend": card["frontend"],
            "latest_status": latest_status,
            "latest_release": _relative_or_str(release_path) if release_path else "",
            "versions": versions,
        }
        detail_path = models_dir / f"{model_id}.json"
        _write_json(detail_path, detail)
        index_models.append(
            {
                "model_id": model_id,
                "display_name": card["model"]["display_name"],
                "family": card["model"]["family"],
                "latest_status": latest_status,
                "detail": f"models/{model_id}.json",
            }
        )
        all_models.append(detail)
    _write_json(data_dir / "index.json", {"schema_version": 1, "generated_at": generated_at, "models": index_models})
    _write_json(data_dir / "models.json", {"schema_version": 1, "generated_at": generated_at, "models": all_models})
    overrides = data_dir / "models.overrides.json"
    if not overrides.is_file():
        _write_json(overrides, {"schema_version": 1, "overrides": {}})
    print(f"Wrote {_relative_or_str(data_dir / 'index.json')}")
    return 0


def render_readme_command(args: argparse.Namespace) -> int:
    model_id = args.model_id
    catalog_root = _resolve_path(getattr(args, "catalog_root", "")) if getattr(args, "catalog_root", "") else _default_catalog_root()
    release_root = _resolve_path(getattr(args, "release_root", "")) if getattr(args, "release_root", "") else _default_release_root()
    detail_path = catalog_root / "data" / "models" / f"{model_id}.json"
    if not detail_path.is_file():
        build_catalog_command(argparse.Namespace(release_root=str(release_root), catalog_root=str(catalog_root)))
    detail = _read_json(detail_path)
    versions = detail.get("versions", [])
    version = versions[0] if versions else {"version_id": "draft", "status": "draft", "gates": {}, "artifacts": {}}
    lines = [
        f"# {detail['model']['display_name']}",
        "",
        f"- Model ID: `{model_id}`",
        f"- Version: `{version.get('version_id', '')}`",
        f"- Status: `{version.get('status', '')}`",
        f"- Workflow config: `{detail['workflow']['config_path']}`",
        "",
        "## Summary",
        "",
        detail.get("frontend", {}).get("summary", ""),
        "",
        "## Gates",
        "",
    ]
    for gate_name, gate in version.get("gates", {}).items():
        lines.append(f"- `{gate_name}`: `{gate.get('status', '')}` — {gate.get('evidence', '')}")
    lines.append("")
    output_dir = release_root / model_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{version.get('version_id', 'draft')}.md"
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {_relative_or_str(output)}")
    return 0


def iter_delivery_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    if not root.is_dir():
        raise DeliveryValidationError(f"model card root does not exist: {root}")
    return sorted(path for path in root.rglob("*.yaml") if path.is_file())


def validate_command(args: argparse.Namespace) -> int:
    if not SCHEMA_PATH.is_file():
        raise DeliveryValidationError(f"schema file does not exist: {SCHEMA_PATH.relative_to(ROOT)}")
    # Ensure the schema itself is valid JSON for tooling consumption.
    json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

    root = (ROOT / args.root).resolve() if not Path(args.root).is_absolute() else Path(args.root)
    files = iter_delivery_files(root)
    if not files:
        raise DeliveryValidationError(f"no model card YAML files found under {root}")
    for delivery_file in files:
        validate_delivery_file(delivery_file)
        print(f"OK {delivery_file.relative_to(ROOT)}")
    print(f"Validated {len(files)} model card YAML file(s)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merak model card and delivery flow utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate", help="validate phase-1 model card YAML files")
    validate_parser.add_argument(
        "--root",
        default="merak_delivery/model_cards/merak",
        help="model card YAML file or directory",
    )
    validate_parser.set_defaults(func=validate_command)

    run_parser = subparsers.add_parser("run-workflow", help="run real Merak workflow and collect delivery metadata")
    run_parser.add_argument("--model-card", required=True, help="model card YAML path")
    run_parser.add_argument("--work-dir", default="", help="flow work directory; defaults to runtime.work_dir/version_id")
    run_parser.add_argument("--model-dir", default="", help="override workflow.model_dir")
    run_parser.add_argument("--quant-output-dir", default="", help="quant output directory")
    run_parser.add_argument("--export-output-dir", default="", help="export output directory")
    run_parser.add_argument("--golden-meta", default="", help="golden_meta_info.json path")
    run_parser.add_argument("--eval-report", default="", help="optional eval_report.json path")
    run_parser.add_argument("--run-eval", action="store_true", help="run hm_eval after export and write eval_report.json")
    run_parser.add_argument("--eval-output", default="", help="hm_eval delivery report path; defaults to work_dir/eval_report.json")
    run_parser.add_argument("--eval-work-dir", default="", help="hm_eval raw output directory; defaults to work_dir/hm_eval")
    run_parser.add_argument("--eval-model", default="", help="hm_eval model config_id or display name; defaults to model.id")
    run_parser.add_argument("--eval-backend", default="hmonnx", help="hm_eval backend, for example float or hmonnx")
    run_parser.add_argument("--eval-datasets", nargs="+", default=[], help="hm_eval dataset names; defaults to ceval")
    run_parser.add_argument("--eval-dataset-hub", choices=["modelscope", "huggingface"], default="modelscope", help="evalscope dataset hub; defaults to modelscope")
    run_parser.add_argument("--eval-limit", type=int, default=0, help="hm_eval per-subset sample limit; 0 means full")
    run_parser.add_argument("--eval-max-tokens", type=int, default=0, help="hm_eval max generation tokens; 0 uses model config")
    run_parser.add_argument("--hmonnx-meta", default="", help="hm_eval HMONNX meta path; defaults to generated golden_meta_info.json")
    run_parser.add_argument("--vision-hmonnx-meta", default="", help="hm_eval vision export_meta_info.json path")
    run_parser.add_argument("--device", default="", help="override runtime.device")
    run_parser.add_argument("--bits", type=int, default=None, help="override quant.bits")
    run_parser.add_argument("--skip-quant", action="store_true", help="set quant config override to None")
    run_parser.add_argument("--dump-golden", action="store_true", help="call workflow.dump_golden after export")
    run_parser.add_argument("--golden-prompt", default="用中文简单介绍这个模型。", help="text prompt for dump_golden")
    run_parser.add_argument("--register-release", action="store_true", help="register release after artifact check")
    run_parser.add_argument("--build-catalog", action="store_true", help="build frontend catalog after release/check")
    run_parser.add_argument("--render-readme", action="store_true", help="render release README after release/check")
    run_parser.add_argument("--release-root", default="", help="release root directory; defaults to work_dirs/merak_delivery/releases/merak")
    run_parser.add_argument("--catalog-root", default="", help="catalog root directory; defaults to work_dirs/merak_delivery/model_catalog")
    run_parser.add_argument("--debug", action="store_true", help="enable workflow debug mode")
    run_parser.add_argument("--dry-run", action="store_true", help="print workflow plan without running heavy steps")
    run_parser.set_defaults(func=run_workflow_command)

    eval_parser = subparsers.add_parser("run-eval", help="run hm_eval and write Merak eval_report.json")
    eval_parser.add_argument("--model-card", required=True, help="model card YAML path")
    eval_parser.add_argument("--work-dir", default="", help="flow work directory; defaults to runtime.work_dir/version_id")
    eval_parser.add_argument(
        "--output",
        default="",
        help="Merak eval report path; defaults to work_dir/eval_report.json or eval_report_float.json",
    )
    eval_parser.add_argument("--eval-work-dir", default="", help="hm_eval raw output directory; defaults to work_dir/hm_eval")
    eval_parser.add_argument("--eval-model", default="", help="hm_eval model config_id or display name; defaults to model.id")
    eval_parser.add_argument("--eval-backend", default="hmonnx", help="hm_eval backend, for example float or hmonnx")
    eval_parser.add_argument("--eval-datasets", nargs="+", default=[], help="hm_eval dataset names; defaults to ceval")
    eval_parser.add_argument("--eval-dataset-hub", choices=["modelscope", "huggingface"], default="modelscope", help="evalscope dataset hub; defaults to modelscope")
    eval_parser.add_argument("--eval-limit", type=int, default=0, help="hm_eval per-subset sample limit; 0 means full")
    eval_parser.add_argument("--eval-max-tokens", type=int, default=0, help="hm_eval max generation tokens; 0 uses model config")
    eval_parser.add_argument("--hmonnx-meta", default="", help="hm_eval HMONNX golden_meta_info.json/export_meta_info.json/meta.json path")
    eval_parser.add_argument("--vision-hmonnx-meta", default="", help="hm_eval vision export_meta_info.json path")
    eval_parser.set_defaults(func=run_eval_command)

    collect_parser = subparsers.add_parser("collect-manifest", help="collect a delivery manifest from run outputs")
    collect_parser.add_argument("--model-card", required=True, help="model card YAML path")
    collect_parser.add_argument("--work-dir", required=True, help="flow work directory")
    collect_parser.add_argument("--export-dir", required=True, help="HMONNX export directory")
    collect_parser.add_argument("--golden-meta", required=True, help="golden_meta_info.json path")
    collect_parser.add_argument("--eval-report", default="", help="optional eval_report.json path")
    collect_parser.add_argument("--quanted-model-path", default="", help="optional quanted model path")
    collect_parser.set_defaults(func=collect_manifest_command)

    artifact_parser = subparsers.add_parser("check-artifact", help="check HMONNX artifacts from a manifest")
    artifact_parser.add_argument("--manifest", required=True, help="delivery_manifest.json path")
    artifact_parser.set_defaults(func=check_artifact_command)

    release_parser = subparsers.add_parser("register-release", help="register or update release state")
    release_parser.add_argument("--manifest", required=True, help="delivery_manifest.json path")
    release_parser.add_argument("--artifact-check", required=True, help="artifact_check.json path")
    release_parser.add_argument("--eval-report", default="", help="optional eval_report.json path")
    release_parser.add_argument("--release-root", default="", help="release root directory")
    release_parser.set_defaults(func=register_release_command)

    catalog_parser = subparsers.add_parser("build-catalog", help="build frontend catalog files")
    catalog_parser.add_argument("--release-root", default="", help="release root directory")
    catalog_parser.add_argument("--catalog-root", default="", help="catalog root directory")
    catalog_parser.set_defaults(func=build_catalog_command)

    readme_parser = subparsers.add_parser("render-readme", help="render README from catalog/release data")
    readme_parser.add_argument("--model-id", required=True, help="model id to render")
    readme_parser.add_argument("--release-root", default="", help="release root directory")
    readme_parser.add_argument("--catalog-root", default="", help="catalog root directory")
    readme_parser.set_defaults(func=render_readme_command)
    return parser


configure_core(
    build_parser=build_parser,
    dataset_statuses=_dataset_statuses,
    default_catalog_root=_default_catalog_root,
    default_eval_report=_default_eval_report,
    default_eval_work_dir=_default_eval_work_dir,
    default_export_dir=_default_export_dir,
    default_quant_dir=_default_quant_dir,
    default_release_root=_default_release_root,
    default_work_dir=_default_work_dir,
    derive_eval_report_status=_derive_eval_report_status,
    find_golden_meta=_find_golden_meta,
    infer_eval_datasets=_infer_eval_datasets,
    infer_eval_model_id=_infer_eval_model_id,
    load_model_card=_load_model_card,
    relative_or_str=_relative_or_str,
    resolve_path=_resolve_path,
    run_build_catalog=lambda *args, **kwargs: _run_build_catalog(*args, **kwargs),
    run_check_artifact=lambda *args, **kwargs: _run_check_artifact(*args, **kwargs),
    run_collect_manifest=lambda *args, **kwargs: _run_collect_manifest(*args, **kwargs),
    run_hm_eval=lambda *args, **kwargs: _run_hm_eval(*args, **kwargs),
    run_register_release=lambda *args, **kwargs: _run_register_release(*args, **kwargs),
    run_render_readme=lambda *args, **kwargs: _run_render_readme(*args, **kwargs),
    serialize_result_path=_serialize_result_path,
    write_json=_write_json,
)


def main(argv: list[str] | None = None) -> int:
    try:
        return MerakModelFlowCLI().run(argv)
    except DeliveryValidationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except json.JSONDecodeError as exc:
        print(f"ERROR: invalid schema JSON: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
