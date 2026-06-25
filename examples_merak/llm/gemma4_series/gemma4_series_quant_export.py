#!/usr/bin/env python3
"""Run Gemma4 Series quantization and HMONNX export from one CLI.

The workflow YAML owns model topology and quant/export defaults.  This
entrypoint only receives paths and source mode, mirroring the Qwen3.5
quant-export CLI:

* default: run ``quant -> export`` from the base HF directory;
* ``--existing-hf-model-dir``: reuse an already quantized HF/GPTQModel
  directory and export it directly;
* ``--base``: skip quantization explicitly and export the base HF model;
* ``--mtp-assistant-model-dir``: enable Gemma4 Series MTP target export and
  export the assistant draft ONNX into the same export directory.
"""

from __future__ import annotations

import argparse
import gc
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
    del config_overrides  # MTP topology must come from the YAML, not CLI path overrides.
    if not args.mtp_assistant_model_dir:
        return

    from xhmodel_merak.xh_llm.workflows.config import WorkflowConfig

    cfg = WorkflowConfig.from_file(args.config).data
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
            "Gemma4 MTP export requires a full_mtp YAML or complete "
            "export.model.mtp_config when --mtp-assistant-model-dir is passed; "
            f"missing/inconsistent: {missing_text}. Use a gemma4_*_full_mtp.yaml config."
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


def _resolve_exported_dir(export_result: Any) -> Path:
    exported_dir = getattr(export_result, "exported_dir", None)
    if exported_dir:
        return Path(exported_dir)

    work_dir = Path(getattr(export_result, "work_dir", ""))
    if not work_dir.is_dir():
        raise FileNotFoundError(f"Export work_dir does not exist: {work_dir}")

    prefill_hmonnx = getattr(getattr(export_result, "meta", None), "prefill_hmonnx", None)
    if prefill_hmonnx:
        matches = [
            path
            for path in work_dir.iterdir()
            if path.is_dir() and path.name.startswith("hmquant") and (path / str(prefill_hmonnx)).exists()
        ]
        if len(matches) == 1:
            return matches[0]

    meta_files = [
        path / "golden_meta_info.json"
        for path in work_dir.iterdir()
        if path.is_dir() and path.name.startswith("hmquant") and (path / "golden_meta_info.json").is_file()
    ]
    if len(meta_files) == 1:
        return meta_files[0].parent
    if not meta_files:
        raise RuntimeError(f"Cannot locate exported HMONNX directory under {work_dir}")
    meta_file_list = ", ".join(str(path) for path in meta_files)
    raise RuntimeError(f"Found multiple exported HMONNX directories under {work_dir}: {meta_file_list}")


def _quant_type_weight_bits(quant_type: str | None, default: int = 4) -> int:
    if not quant_type or not str(quant_type).startswith("w"):
        return default
    digits: list[str] = []
    for char in str(quant_type)[1:]:
        if not char.isdigit():
            break
        digits.append(char)
    return int("".join(digits)) if digits else default


def _model_config_dict(meta: dict[str, Any]) -> dict[str, Any]:
    model_config = meta.get("model_config") or {}
    return model_config if isinstance(model_config, dict) else {}


def _mtp_config_dict(meta: dict[str, Any]) -> dict[str, Any]:
    mtp_config = _model_config_dict(meta).get("mtp_config") or {}
    return mtp_config if isinstance(mtp_config, dict) else {}


def _resolve_model_path(path: str) -> Path:
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else _REPO_ROOT / candidate


def _infer_shared_cache_lengths(meta: dict[str, Any]) -> tuple[int | None, int | None]:
    spec = dict(meta.get("spec_decode") or {})
    sliding_len = spec.get("shared_sliding_cache_length")
    full_len = spec.get("shared_full_cache_length")
    if sliding_len and full_len:
        return int(sliding_len), int(full_len)

    model_cfg = _model_config_dict(meta)
    context_len = int(model_cfg.get("context_max_length") or meta.get("context_max_length") or 0)
    sliding_window = int(meta.get("sliding_window") or model_cfg.get("sliding_window") or 0)
    prefill_len = int(model_cfg.get("prefill_chunk_length") or model_cfg.get("input_sequence_length") or 0)
    if not sliding_len and sliding_window > 0 and prefill_len > 0:
        sliding_len = ((sliding_window + prefill_len + 15) // 16) * 16
    if not full_len and context_len > 0:
        full_len = context_len

    layer_shapes = meta.get("layer_kv_shapes") or meta.get("kv_cache_shapes_per_layer") or []
    lengths = sorted({int(shape[2]) for shape in layer_shapes if len(shape) > 2})
    if not sliding_len and lengths:
        candidates = [length for length in lengths if context_len <= 0 or length < context_len]
        sliding_len = max(candidates or lengths)
    if not full_len and lengths:
        full_len = max(lengths)
    return (int(sliding_len) if sliding_len else None, int(full_len) if full_len else None)


def _relative_to_export_dir(path: Path, export_dir: Path) -> str:
    try:
        return path.resolve().relative_to(export_dir.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _update_manifest_with_draft(
    meta_path: Path,
    draft_onnx: Path,
    *,
    lm_head_quant_type: str,
    shared_sliding_len: int | None,
    shared_full_len: int | None,
) -> None:
    export_dir = meta_path.parent
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    model_cfg = _model_config_dict(meta)
    spec_decode = dict(meta.get("spec_decode") or {})
    block_size = int(
        spec_decode.get("block_size")
        or meta.get("spec_decode_block_size")
        or model_cfg.get("num_draft_tokens")
        or 4
    )
    verify_length = int(spec_decode.get("verify_length") or meta.get("spec_decode_verify_length") or block_size + 1)
    draft_head_bits = _quant_type_weight_bits(lm_head_quant_type)
    rel_draft = _relative_to_export_dir(draft_onnx, export_dir)
    sliding_window = int(meta.get("sliding_window") or model_cfg.get("sliding_window") or 0)
    target_decode_sliding = None
    if sliding_window > 0:
        target_decode_sliding = ((sliding_window + verify_length - 1 + 15) // 16) * 16

    shared_sliding_len = shared_sliding_len or spec_decode.get("shared_sliding_cache_length")
    shared_full_len = shared_full_len or spec_decode.get("shared_full_cache_length")
    spec_decode.update(
        mode="mtp",
        block_size=block_size,
        verify_length=verify_length,
        hidden_output_name="target_hidden_state",
        draft_head_weight_bits=draft_head_bits,
        draft_decode_onnx=rel_draft,
        draft_onnx=rel_draft,
        shared_sliding_cache_length=shared_sliding_len,
        shared_full_cache_length=shared_full_len,
        target_decode_sliding_output_length=target_decode_sliding,
    )
    meta.update(
        spec_decode_mode="mtp",
        spec_decode_block_size=block_size,
        spec_decode_hidden_output_name="target_hidden_state",
        spec_decode_verify_length=verify_length,
        spec_decode_draft_head_weight_bits=draft_head_bits,
        draft_decode_onnx_file=rel_draft,
        draft_onnx_file=rel_draft,
        spec_decode=spec_decode,
    )
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=4), encoding="utf-8")


def _draft_quant_config(body_quant_type: str, lm_head_quant_type: str) -> Any:
    from xhquant.api import ConfigDict

    return ConfigDict(
        quant_type=body_quant_type,
        nodes=dict(
            lm_head=dict(quant_type=lm_head_quant_type),
            masked_embedding=dict(quant_type=lm_head_quant_type),
            masked_embedding_centroids=dict(quant_type=lm_head_quant_type),
            **{"masked_embedding.centroids": dict(quant_type=lm_head_quant_type)},
        ),
        ops={},
    )


def _export_mtp_draft(args: argparse.Namespace, export_result: Any) -> Path:
    exported_dir = _resolve_exported_dir(export_result)
    meta_path = exported_dir / "golden_meta_info.json"
    if not meta_path.exists():
        raise FileNotFoundError(meta_path)

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    mtp_cfg = _mtp_config_dict(meta)
    assistant_dir = args.mtp_assistant_model_dir or mtp_cfg.get("assistant_hf_model")
    target_dir = mtp_cfg.get("target_hf_model") or _target_model_dir_for_mtp(args)
    if not assistant_dir:
        raise ValueError("MTP draft export requires --mtp-assistant-model-dir or manifest assistant_hf_model")
    if not target_dir:
        raise ValueError("MTP draft export requires target_hf_model or --hf-model-dir")

    body_quant_type = str(mtp_cfg.get("body_quant_type") or args.mtp_body_quant_type)
    lm_head_quant_type = str(mtp_cfg.get("lm_head_quant_type") or args.mtp_lm_head_quant_type)
    shared_sliding_len, shared_full_len = _infer_shared_cache_lengths(meta)
    output_dir = exported_dir / "draft_onnx"
    output_dir.mkdir(parents=True, exist_ok=True)

    import torch
    from xhmodel_merak.xh_llm.models.gemma4_series.gemma4_series_mtp_model import (
        XHGemma4SeriesAssistantDraftModel,
    )
    from xhquant.api import ConfigDict, PrecisionMode, get_xhquant_logger, ptq_quantize, xhquant_init

    logger = get_xhquant_logger()
    logger.info("Exporting Gemma4 Series MTP assistant draft ONNX to %s", output_dir)
    xhquant_init(args.chip_arch.lower())
    model = XHGemma4SeriesAssistantDraftModel(
        assistant_model_dir=str(_resolve_model_path(str(assistant_dir))),
        target_model_dir=str(_resolve_model_path(str(target_dir))),
        wrap_cfg=ConfigDict(
            input_sequence_length=1,
            max_sequence_length=int(_model_config_dict(meta).get("context_max_length") or args.context_length),
            dtype=args.mtp_draft_dtype,
            cache_axis=2,
            shared_sliding_cache_length=shared_sliding_len,
            shared_full_cache_length=shared_full_len,
        ),
        quant_config=_draft_quant_config(body_quant_type, lm_head_quant_type),
    )
    model.init_wrap_model()
    dummy = model.prepare_inputs(None)
    model.convert_to_fronted_graph(dummy)
    model.convert_to_quant_graph(args.chip_arch.lower())
    ptq_quantize(model.quanted_model, [model.prepare_inputs(None)], PrecisionMode.ALIGNED, [torch.device("cpu")])
    model.convert_to_export_graph(dummy)
    onnx_file = Path(
        model.to_export_onnx(
            dummy,
            str(output_dir),
            prefix=f"gemma4_series_{Path(args.hf_model_dir).name}_assistant_decode",
        )[0]
    )
    model.release_exported_model()
    model.release_quanted_model()
    model.release_frontend_model()
    model.release_wraped_model()
    gc.collect()

    _update_manifest_with_draft(
        meta_path,
        onnx_file,
        lm_head_quant_type=lm_head_quant_type,
        shared_sliding_len=shared_sliding_len,
        shared_full_len=shared_full_len,
    )
    return onnx_file


def main(args: argparse.Namespace) -> None:
    from xhmodel_merak.xh_llm.workflows import AutoLLMWorkflow

    quant_output_dir = _resolve_quant_output_dir(args)
    quant_overrides = _build_quant_overrides(args)
    mtp_overrides = _build_mtp_overrides(args)
    dotted_overrides = _parse_dotted_overrides(args.override)
    config_overrides = _merge_overrides(quant_overrides, mtp_overrides, dotted_overrides)
    _validate_mtp_config_complete(args, config_overrides)

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

    draft_onnx = None
    if args.mtp_assistant_model_dir and not args.skip_mtp_draft_export:
        draft_onnx = _export_mtp_draft(args, export_result)

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
    parser.add_argument("--context-length", type=int, default=2048, help="Fallback MTP draft max sequence length")
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
        help="With --mtp-assistant-model-dir, export only target HMONNX and skip assistant draft ONNX.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="PATH=JSON_VALUE",
        help=(
            "Workflow config override using dotted paths. Repeatable. "
            "Examples: --override export.model.context_max_length=8192 "
            "--override export.model.prefill_chunk_length=256"
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
