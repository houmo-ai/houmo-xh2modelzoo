#!/usr/bin/env python3
"""Validate HM release/golden HMONNX package layout for Qwen3.5/Qwen3.6 exports.

The checks mirror the Feishu HM model release naming rules:
release directories stay lowercase, stage folders are named prefill/decode/visual
or the spec-decode draft stage names, and stage artifacts use
<release_prefix>_<stage>_{with_act.onnx,external_data}.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


STAGE_SUFFIXES = {
    "prefill": "prefill",
    "decode": "decode",
    "visual": "visual",
    "mtp_draft_prefill": "mtp_draft_prefill",
    "mtp_draft_decode": "mtp_draft_decode",
    "dflash_draft_context": "dflash_draft_context",
    "dflash_draft_context_decode": "dflash_draft_context_decode",
    "dflash_draft_decode": "dflash_draft_decode",
}
LEGACY_STAGE_DIRS = {"decoder", "vision", "mtp", "dflash"}
REQUIRED_TOP_LEVEL = {"prefill", "decode", "hf_config", "quant_embedding.pt", "golden_meta_info.json"}


def _failures_for_stage(stage_dir: Path, release_prefix: str) -> list[str]:
    failures: list[str] = []
    stage_suffix = STAGE_SUFFIXES.get(stage_dir.name)
    if stage_suffix is None:
        return [f"unsupported stage directory: {stage_dir.name}"]

    expected_onnx = stage_dir / f"{release_prefix}_{stage_suffix}_with_act.onnx"
    expected_external_data = stage_dir / f"{release_prefix}_{stage_suffix}_external_data"
    if not expected_onnx.is_file():
        failures.append(f"missing {expected_onnx.relative_to(stage_dir.parent)}")
    if not expected_external_data.is_file():
        failures.append(f"missing {expected_external_data.relative_to(stage_dir.parent)}")
    elif expected_onnx.is_file():
        failures.extend(_failures_for_onnx_external_data(expected_onnx, expected_external_data.name))

    for path in [*stage_dir.rglob("*.onnx"), *stage_dir.rglob("*external_data")]:
        if not path.name.startswith(release_prefix):
            failures.append(f"artifact does not start with release prefix: {path}")

    for step_dir in sorted(path for path in stage_dir.rglob("step_*") if path.is_dir()):
        step_onnx = list(step_dir.glob("*.onnx"))
        step_external_data = list(step_dir.glob("*external_data"))
        if not step_onnx:
            failures.append(f"missing step onnx under {step_dir}")
        if not step_external_data:
            failures.append(f"missing step external_data under {step_dir}")
        for path in step_onnx + step_external_data:
            if not (path.is_symlink() or path.exists()):
                failures.append(f"broken step artifact: {path}")
    return failures


def _failures_for_onnx_external_data(onnx_path: Path, expected_external_data_name: str) -> list[str]:
    """Verify ONNX protobuf external_data locations point at the HM-named artifact."""
    try:
        import onnx
    except ImportError as exc:
        return [f"onnx package is required to validate external_data locations: {exc}"]

    failures: list[str] = []
    try:
        model = onnx.load_model(str(onnx_path), load_external_data=False)
    except Exception as exc:  # pragma: no cover - defensive error formatting
        return [f"failed to read ONNX metadata for {onnx_path}: {exc}"]

    for tensor in model.graph.initializer:
        locations = [entry.value for entry in tensor.external_data if entry.key == "location"]
        for location in locations:
            if Path(location).name != expected_external_data_name:
                failures.append(
                    "ONNX external_data location mismatch: "
                    f"{onnx_path.relative_to(onnx_path.parent.parent)}/{tensor.name} "
                    f"references {location}, expected {expected_external_data_name}"
                )
    return failures


def ensure_step_artifact_links(export_dir: Path) -> None:
    """Ensure every existing step_* directory links to its stage HMONNX artifacts."""
    meta_path = export_dir / "golden_meta_info.json"
    if not meta_path.is_file():
        return
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    stage_models: dict[str, Path] = {}

    def _add_model(value) -> None:
        if not isinstance(value, str) or not value:
            return
        relative_path = Path(value)
        if relative_path.is_absolute() or len(relative_path.parts) < 2:
            return
        stage_name = relative_path.parts[0]
        if stage_name in STAGE_SUFFIXES:
            stage_models.setdefault(stage_name, export_dir / relative_path)

    _add_model(meta.get("prefill_hmonnx"))
    _add_model(meta.get("decode_hmonnx"))
    visual_config = meta.get("visual_config")
    if isinstance(visual_config, dict):
        _add_model(visual_config.get("hmonnx"))
    spec_decode = meta.get("spec_decode")
    if isinstance(spec_decode, dict):
        for key, value in spec_decode.items():
            if key.endswith("_onnx"):
                _add_model(value)

    for stage_name, onnx in stage_models.items():
        stage_dir = export_dir / stage_name
        if not stage_dir.is_dir() or not onnx.is_file():
            continue
        artifacts = [onnx, *sorted(path for path in stage_dir.glob("*external_data") if path.is_file())]
        for step_dir in sorted(path for path in stage_dir.rglob("step_*") if path.is_dir()):
            for artifact in artifacts:
                link = step_dir / artifact.name
                if link.exists() or link.is_symlink():
                    continue
                link.symlink_to(Path(os.path.relpath(artifact, step_dir)))


def validate_release_layout(export_dir: Path) -> list[str]:
    failures: list[str] = []
    release_prefix = export_dir.name
    if not export_dir.is_dir():
        return [f"not a directory: {export_dir}"]
    if release_prefix != release_prefix.lower():
        failures.append(f"release directory must be lowercase: {release_prefix}")

    for name in REQUIRED_TOP_LEVEL:
        path = export_dir / name
        if not path.exists():
            failures.append(f"missing top-level artifact: {name}")
    if (export_dir / "hf_config").is_dir() and not any((export_dir / "hf_config").iterdir()):
        failures.append("hf_config must be non-empty")
    for name in LEGACY_STAGE_DIRS:
        if (export_dir / name).exists():
            failures.append(f"legacy stage directory is forbidden: {name}")

    meta_path = export_dir / "golden_meta_info.json"
    meta = {}
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text())
        except json.JSONDecodeError as exc:
            failures.append(f"golden_meta_info.json is invalid JSON: {exc}")
    for key in ("prefill_hmonnx", "decode_hmonnx", "quant_embedding", "hf_config"):
        if meta_path.is_file() and key not in meta:
            failures.append(f"golden_meta_info.json missing key: {key}")

    for stage_name in ("prefill", "decode"):
        stage_dir = export_dir / stage_name
        if stage_dir.is_dir():
            failures.extend(_failures_for_stage(stage_dir, release_prefix))
    for stage_name in sorted(STAGE_SUFFIXES.keys() - {"prefill", "decode"}):
        stage_dir = export_dir / stage_name
        if stage_dir.exists():
            failures.extend(_failures_for_stage(stage_dir, release_prefix))

    if isinstance(meta.get("visual_config"), dict):
        visual_hmonnx = meta["visual_config"].get("hmonnx")
        if not isinstance(visual_hmonnx, str) or not visual_hmonnx.startswith("visual/"):
            failures.append("visual_config.hmonnx must be a visual/ relative path")
    spec_decode = meta.get("spec_decode", {})
    if isinstance(spec_decode, dict):
        for key, value in spec_decode.items():
            if key.endswith("_onnx") and isinstance(value, str):
                top_dir = value.split("/", 1)[0]
                if top_dir not in STAGE_SUFFIXES:
                    failures.append(f"spec_decode path uses unsupported stage dir: {key}={value}")
                if top_dir in LEGACY_STAGE_DIRS:
                    failures.append(f"spec_decode path uses legacy stage dir: {key}={value}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export_dir", type=Path, help="HM release directory containing golden_meta_info.json")
    parser.add_argument(
        "--repair-step-links",
        action="store_true",
        help="Create missing step_*/ ONNX and external_data symlinks before validating.",
    )
    args = parser.parse_args()

    if args.repair_step_links:
        ensure_step_artifact_links(args.export_dir)
    failures = validate_release_layout(args.export_dir)
    if failures:
        print("HM release layout validation failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print(f"HM release layout validation passed: {args.export_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
