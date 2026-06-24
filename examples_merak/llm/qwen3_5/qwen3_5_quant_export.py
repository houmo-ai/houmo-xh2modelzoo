"""Run Qwen3.5/Qwen3.6 Merak quantization and HMONNX export from one CLI.

The workflow YAML owns model topology and quant/export defaults.  This entrypoint
only receives paths and the source mode:

* default: run ``quant -> export`` from the base HF directory;
* ``--existing-hf-model-dir``: reuse an already quantized HF/GPTQModel directory
  and export it directly;
* ``--base``: skip quantization explicitly and export the base HF model.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


_QUANT_PLACEHOLDER_DIRNAME = "_workflow_existing_or_base_quant_placeholder"


def _is_unsafe_delete_target(path: Path) -> bool:
    resolved = path.resolve()
    cwd = Path.cwd().resolve()
    return resolved in {Path("/").resolve(), Path.home().resolve(), cwd}


def _remove_output_dir_if_needed(output_dir: str, force: bool, *, label: str) -> None:
    path = Path(output_dir)
    if not path.exists():
        return
    if not force:
        raise FileExistsError(f"{label} output_dir already exists: {path}; pass --force to overwrite")
    if path.is_symlink() or _is_unsafe_delete_target(path):
        raise ValueError(f"Refusing to delete unsafe {label} output_dir: {path.resolve()}")
    shutil.rmtree(path)


def _parse_override_value(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def _parse_dotted_overrides(items: list[str] | None) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Override must use PATH=JSON_VALUE format, got: {item!r}")
        path, raw_value = item.split("=", 1)
        if not path:
            raise ValueError(f"Override path must be non-empty, got: {item!r}")
        overrides[path] = _parse_override_value(raw_value)
    return overrides


def _build_quant_overrides(args: argparse.Namespace) -> dict[str, Any] | None:
    if args.base and args.existing_hf_model_dir:
        raise ValueError("--base and --existing-hf-model-dir are mutually exclusive")
    if args.base:
        return {"quant": None}
    if args.existing_hf_model_dir:
        return {
            "quant": {
                "algorithm": "existing_hf",
                "method": "autoround",
                "artifact_format": args.artifact_format,
                "existing_hf_model_dir": args.existing_hf_model_dir,
            }
        }
    return None


def _merge_overrides(*overrides: dict[str, Any] | None) -> dict[str, Any] | None:
    merged: dict[str, Any] = {}
    for item in overrides:
        if not item:
            continue
        overlap = set(merged).intersection(item)
        if overlap:
            keys = ", ".join(sorted(overlap))
            raise ValueError(f"Duplicate workflow override path(s): {keys}")
        merged.update(item)
    return merged or None


def _should_run_real_quant(args: argparse.Namespace) -> bool:
    return not args.base and not args.existing_hf_model_dir


def _resolve_quant_output_dir(args: argparse.Namespace) -> str:
    if args.quant_output_dir:
        return args.quant_output_dir
    if _should_run_real_quant(args):
        raise ValueError("--quant-output-dir is required when running real quantization")
    return str(Path(args.export_output_dir) / _QUANT_PLACEHOLDER_DIRNAME)


def _make_json_safe(value: Any) -> Any:
    """Return a JSON-serializable representation for CLI summaries.

    Workflow results may carry rich runtime objects in fields such as
    ``ExportResult.meta``.  The CLI summary is only an audit breadcrumb, so it
    must not fail the already-completed quant/export/golden run because such an
    object lacks a JSON encoder.
    """

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if is_dataclass(value):
        return _make_json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(key): _make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_make_json_safe(item) for item in value]
    return repr(value)


def _jsonable_dataclass(value: Any) -> dict[str, Any]:
    if is_dataclass(value):
        result = asdict(value)
    elif hasattr(value, "__dict__"):
        result = dict(value.__dict__)
    else:
        raise TypeError(f"Object is not JSON serializable as a workflow result: {type(value)!r}")
    safe_result = _make_json_safe(result)
    if not isinstance(safe_result, dict):
        raise TypeError(f"Workflow result did not serialize to a dict: {type(value)!r}")
    return safe_result


def main(args: argparse.Namespace) -> None:
    from xhmodel_merak.xh_llm.workflows import AutoLLMWorkflow

    quant_output_dir = _resolve_quant_output_dir(args)
    quant_overrides = _build_quant_overrides(args)
    dotted_overrides = _parse_dotted_overrides(args.override)
    config_overrides = _merge_overrides(quant_overrides, dotted_overrides)

    if _should_run_real_quant(args):
        _remove_output_dir_if_needed(quant_output_dir, args.force, label="quant")
    _remove_output_dir_if_needed(args.export_output_dir, args.force, label="export")

    workflow = AutoLLMWorkflow.from_config(
        hf_model_dir=args.hf_model_dir,
        config_path=args.config,
        seed=args.seed,
        debug=args.debug,
    )
    quant_result = workflow.quant(
        output_dir=quant_output_dir,
        device=args.quant_device or args.device,
        config_overrides=config_overrides,
    )
    export_result = workflow.export(
        quant_result=quant_result,
        output_dir=args.export_output_dir,
        device=args.export_device or args.device,
        config_overrides=config_overrides,
    )

    golden_meta_file = None
    if args.dump_golden:
        golden_meta_file = workflow.dump_golden(
            export_result=export_result,
            device=args.export_device or args.device,
            input_messages={"text": args.prompt},
        )

    summary = {
        "quant_result": _jsonable_dataclass(quant_result),
        "export_result": _jsonable_dataclass(export_result),
        "golden_meta_file": golden_meta_file,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Qwen3.5/Qwen3.6 workflow quant->HMONNX export from a YAML config.",
    )
    parser.add_argument("--hf-model-dir", required=True, help="Original HuggingFace model directory")
    parser.add_argument("--config", required=True, help="Workflow YAML containing quant and export sections")
    parser.add_argument(
        "--quant-output-dir",
        default="",
        help="Quant work/output directory. Required unless --base or --existing-hf-model-dir is used.",
    )
    parser.add_argument("--export-output-dir", required=True, help="HMONNX export work/output directory")
    parser.add_argument("--device", default="cuda", help="Default device for both quant and export")
    parser.add_argument("--quant-device", default="", help="Override quant device only, e.g. cuda:0")
    parser.add_argument("--export-device", default="", help="Override export/golden device only, e.g. cuda:0")
    parser.add_argument("--seed", type=int, default=1024, help="Workflow seed")
    parser.add_argument("--debug", action="store_true", help="Enable workflow debug logging")
    parser.add_argument("--force", action="store_true", help="Remove output directories before running")
    parser.add_argument(
        "--base",
        action="store_true",
        help="Skip quantization explicitly and export from --hf-model-dir",
    )
    parser.add_argument(
        "--existing-hf-model-dir",
        default="",
        help="Reuse an already quantized HF/GPTQModel directory and export it directly",
    )
    parser.add_argument(
        "--artifact-format",
        default="gptqmodel_hf",
        help="Artifact format metadata for --existing-hf-model-dir",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="PATH=JSON_VALUE",
        help=(
            "Workflow config override using dotted paths. Repeatable. "
            "Examples: --override export.model.fuse_gdr_ops=true "
            "--override export.model.context_max_length=32768"
        ),
    )
    parser.add_argument(
        "--dump-golden",
        action="store_true",
        help="Generate golden data after export using workflow.dump_golden()",
    )
    parser.add_argument("--prompt", default="用中文简单介绍这个模型。", help="Prompt used by --dump-golden")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
