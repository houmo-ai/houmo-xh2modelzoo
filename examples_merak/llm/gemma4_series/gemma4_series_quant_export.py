#!/usr/bin/env python3
"""Run Gemma4 Series quantization and HMONNX export from one CLI.

The workflow YAML owns model topology and quant/export defaults.  This
entrypoint only receives paths and source mode, mirroring the Qwen3.5
quant-export CLI:

* default: run ``quant -> export`` from the base HF directory;
* ``--existing-hf-model-dir``: reuse an already quantized HF/GPTQModel
  directory and export it directly;
* ``--base``: skip quantization explicitly and export the base HF model;
* ``--mtp-assistant-model-dir``: enable Gemma4 Series MTP target export;
  workflow.export() also exports the assistant draft ONNX into the same
  export directory.
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

from xhmodel_merak.xh_llm.models.gemma4_series.mtp_workflow import (  # noqa: E402
    find_mtp_draft_onnx as _find_mtp_draft_onnx,
)
from xhmodel_merak.xh_llm.models.gemma4_series.mtp_workflow import (  # noqa: E402
    update_manifest_with_draft as _update_manifest_with_draft,  # noqa: F401
)


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


def _target_model_dir_for_mtp(args: argparse.Namespace) -> str:
    return args.mtp_target_model_dir or args.existing_hf_model_dir or args.hf_model_dir


def _build_mtp_overrides(args: argparse.Namespace) -> dict[str, Any] | None:
    if not args.mtp_assistant_model_dir:
        return None
    return {
        "export.model.spec_decode_mode": "mtp",
        "export.model.mtp_config.assistant_hf_model": args.mtp_assistant_model_dir,
        "export.model.mtp_config.target_hf_model": _target_model_dir_for_mtp(args),
        "export.model.mtp_config.body_quant_type": args.mtp_body_quant_type,
        "export.model.mtp_config.lm_head_quant_type": args.mtp_lm_head_quant_type,
    }


def _validate_mtp_config_complete(args: argparse.Namespace, config_overrides: dict[str, Any] | None = None) -> None:
    if not args.mtp_assistant_model_dir:
        return

    from xhmodel_merak.xh_llm.workflows.config import WorkflowConfig

    try:
        cfg = WorkflowConfig.from_file(args.config).with_overrides(config_overrides).data
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "Gemma4 MTP export requires --config to point to a full_mtp YAML "
            "or another YAML with complete export.model.mtp_config when "
            f"--mtp-assistant-model-dir is passed. Config: {args.config}. "
            f"Invalid MTP override/config path: {exc}. Use a "
            "gemma4_*_full_mtp.yaml config or add the missing MTP fields."
        ) from exc

    model_cfg = ((cfg.get("export") or {}).get("model") or {}) if isinstance(cfg, dict) else {}
    mtp_cfg = model_cfg.get("mtp_config") or {}
    if not isinstance(mtp_cfg, dict):
        mtp_cfg = {}

    missing: list[str] = []
    if model_cfg.get("spec_decode_mode") != "mtp":
        missing.append("export.model.spec_decode_mode=mtp")
    if not model_cfg.get("num_draft_tokens"):
        missing.append("export.model.num_draft_tokens")

    required_mtp_keys = [
        "assistant_hf_model",
        "target_hf_model",
        "body_quant_type",
        "lm_head_quant_type",
        "batch_size",
        "input_sequence_length",
        "context_max_length",
        "use_cache",
        "num_draft_tokens",
        "assistant_num_hidden_layers",
        "assistant_layer_pattern",
        "assistant_hidden_size",
        "assistant_num_attention_heads",
        "assistant_num_key_value_heads",
        "head_dim",
        "shared_kv_inputs",
    ]
    missing.extend(f"export.model.mtp_config.{key}" for key in required_mtp_keys if not mtp_cfg.get(key))
    if mtp_cfg.get("num_draft_tokens") != model_cfg.get("num_draft_tokens"):
        missing.append("matching export.model and mtp_config num_draft_tokens")

    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(
            "Gemma4 MTP export requires --config to point to a full_mtp YAML "
            "or another YAML with complete export.model.mtp_config when "
            f"--mtp-assistant-model-dir is passed. Config: {args.config}. "
            f"Missing/inconsistent: {missing_text}. Use a "
            "gemma4_*_full_mtp.yaml config or add the missing MTP fields."
        )


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
    """Return a JSON-serializable representation for CLI summaries."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if is_dataclass(value):
        return _make_json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(key): _make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_make_json_safe(item) for item in value]
    if hasattr(value, "to_dict"):
        return _make_json_safe(value.to_dict())
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


def _find_single_golden_meta(work_dir: str | Path) -> Path:
    root = Path(work_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Export work_dir does not exist or is not a directory: {root}")
    candidates = sorted(root.rglob("golden_meta_info.json"))
    if not candidates:
        raise FileNotFoundError(f"No golden_meta_info.json found beneath {root}")
    if len(candidates) != 1:
        raise ValueError(f"Expected one golden_meta_info.json beneath {root}, found {len(candidates)}: {candidates}")
    return candidates[0]


def _validate_flash_attention_export_result(export_result: Any) -> dict[str, dict[str, int]] | None:
    from xhmodel_merak.xh_llm.models.gemma4_series.gemma4_series_llm_model import (
        validate_gemma4_flash_attention_graph,
    )

    meta_path = _find_single_golden_meta(export_result.work_dir)
    with meta_path.open(encoding="utf-8") as file:
        meta = json.load(file)
    if int(meta.get("attention_contract_version", 1)) < 2:
        return None

    facts: dict[str, dict[str, int]] = {}
    for graph_key in ("prefill_hmonnx", "decode_hmonnx"):
        graph_value = meta.get(graph_key)
        if not graph_value:
            raise ValueError(f"{meta_path}: contract-v2 metadata missing {graph_key}")
        graph_path = Path(graph_value)
        if not graph_path.is_absolute():
            graph_path = meta_path.parent / graph_path
        facts[graph_key] = validate_gemma4_flash_attention_graph(graph_path, meta)
    return facts


def main(args: argparse.Namespace) -> None:
    from xhmodel_merak.xh_llm.workflows import AutoLLMWorkflow

    quant_output_dir = _resolve_quant_output_dir(args)
    quant_overrides = _build_quant_overrides(args)
    mtp_overrides = _build_mtp_overrides(args)
    dotted_overrides = _parse_dotted_overrides(args.override)
    config_overrides = _merge_overrides(quant_overrides, mtp_overrides, dotted_overrides)
    _validate_mtp_config_complete(args, config_overrides)
    if args.skip_mtp_draft_export:
        raise ValueError(
            "--skip-mtp-draft-export is no longer supported: "
            "Gemma4 workflow.export() owns all target and MTP draft exports."
        )

    if _should_run_real_quant(args):
        _remove_output_dir_if_needed(quant_output_dir, args.force, label="quant")
    _remove_output_dir_if_needed(args.export_output_dir, args.force, label="export")

    workflow = AutoLLMWorkflow.from_config(
        model_dir=args.hf_model_dir,
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
    flash_attention_validation = _validate_flash_attention_export_result(export_result)

    draft_onnx = _find_mtp_draft_onnx(export_result)

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
        "mtp_draft_onnx": str(draft_onnx) if draft_onnx is not None else None,
        "golden_meta_file": golden_meta_file,
        "flash_attention_validation": flash_attention_validation,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Gemma4 Series workflow quant->HMONNX export from a YAML config.",
    )
    parser.add_argument("--hf-model-dir", required=True, help="Original HuggingFace target model directory")
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
        "--mtp-assistant-model-dir",
        "--assistant-model-dir",
        default="",
        help="Gemma4 assistant/MTP draft HF model directory. Passing it enables MTP export.",
    )
    parser.add_argument(
        "--mtp-target-model-dir",
        default="",
        help="Optional target HF/config directory recorded for draft export; defaults to existing/base target.",
    )
    parser.add_argument("--chip-arch", default="XH2a", help="Chip arch passed to xhquant for assistant draft export")
    parser.add_argument(
        "--mtp-body-quant-type",
        default="w8a8h1_sefp",
        help="Assistant draft non-head linear quant type; full_mtp YAMLs use W8 by default.",
    )
    parser.add_argument(
        "--mtp-lm-head-quant-type",
        default="w4a8h0_ssfp",
        help="Assistant draft logits head quant type; Gemma4 MTP uses W4 by default.",
    )
    parser.add_argument("--mtp-draft-dtype", default="float16", help="Assistant draft export dtype")
    parser.add_argument(
        "--skip-mtp-draft-export",
        action="store_true",
        help="Deprecated; workflow.export() always exports the MTP draft when MTP is enabled.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="PATH=JSON_VALUE",
        help=(
            "Workflow config override using dotted paths. Repeatable. "
            "Examples: --override export.model.context_max_length=8192 "
            "--override export.model.prefill_chunk_length=320"
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
