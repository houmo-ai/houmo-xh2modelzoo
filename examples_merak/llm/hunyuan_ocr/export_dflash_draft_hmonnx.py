#!/usr/bin/env python3
# Copyright 2025 HOUMO AI
# SPDX-License-Identifier: Apache-2.0

"""Export HunyuanOCR DFlash draft graphs and atomically upgrade target metadata."""

from __future__ import annotations

import argparse
import copy
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class DraftExportPreflight:
    metadata: Any
    checkpoint: Any
    metadata_path: Path
    output_root: Path


def _cache_capacity(metadata: Any) -> int:
    value = getattr(metadata, "max_sequence_length", None)
    if value is None:
        model_config = metadata.model_config
        value = (
            model_config.get("context_max_length")
            if isinstance(model_config, Mapping)
            else getattr(model_config, "context_max_length", None)
        )
    if type(value) is not int or value <= 0:
        raise ValueError("HunyuanOCR target metadata must declare a positive context capacity")
    return value


def _generation_eos_token_id(metadata: Any) -> int | list[int]:
    value = getattr(metadata, "generation_eos_token_id", None)
    if value is None:
        model_config = metadata.model_config
        value = (
            model_config.get("generation_eos_token_id")
            if isinstance(model_config, Mapping)
            else getattr(model_config, "generation_eos_token_id", None)
        )
    if type(value) is int:
        return value
    if isinstance(value, (list, tuple)) and value and all(type(token_id) is int for token_id in value):
        return list(dict.fromkeys(value))
    raise ValueError("HunyuanOCR target metadata must declare generation_eos_token_id")


def preflight_draft_export(
    metadata_path: str | Path,
    target_model_dir: str | Path,
    draft_model_dir: str | Path,
) -> DraftExportPreflight:
    from xhmodel_merak.xh_llm.models.hunyuan_ocr import (
        HunyuanOCRModelMeta,
        HunyuanOCRTextExportMeta,
        load_hunyuan_ocr_dflash_checkpoint,
    )

    path = Path(metadata_path).resolve()
    raw = json.loads(path.read_text(encoding="utf-8"))
    class_name = raw.get("meta", {}).get("class_name")
    metadata_classes = {
        "HunyuanOCRTextExportMeta": HunyuanOCRTextExportMeta,
        "HunyuanOCRModelMeta": HunyuanOCRModelMeta,
    }
    if class_name not in metadata_classes:
        raise ValueError(f"Unsupported HunyuanOCR metadata class for draft export: {class_name!r}")
    metadata = metadata_classes[class_name].from_file(path)
    if metadata.schema_version != 2 or metadata.spec_decode.get("status") != "target_verify_ready":
        raise RuntimeError("DFlash draft export requires schema v2 target_verify_ready metadata")
    if metadata.spec_decode.get("capabilities", {}).get("target_verify") is not True:
        raise RuntimeError("DFlash draft export requires target verify capability")
    checkpoint = load_hunyuan_ocr_dflash_checkpoint(draft_model_dir, target_model_dir)
    expected_ids = metadata.spec_decode.get("target_layer_ids")
    if list(checkpoint.config.target_layer_ids) != list(expected_ids or ()):
        raise ValueError("DFlash draft target_layer_ids do not match target metadata")
    if checkpoint.config.hidden_size != int(metadata.spec_decode["target_hidden_size"]):
        raise ValueError("DFlash draft hidden_size does not match target metadata")
    return DraftExportPreflight(
        metadata=metadata,
        checkpoint=checkpoint,
        metadata_path=path,
        output_root=path.parent,
    )


def build_draft_model_config(
    preflight: DraftExportPreflight,
    *,
    mode: str,
    target_model_dir: str | Path,
    draft_model_dir: str | Path,
) -> dict[str, Any]:
    if mode not in ("context", "context_decode", "decode"):
        raise ValueError(f"Unsupported HunyuanOCR DFlash mode: {mode!r}")
    return {
        "model_name": f"hunyuan_ocr_dflash_{mode}",
        "model_type": "HunYuanOCR_DFlash_Draft",
        "chip_arch": "XH2a",
        "hf_model": str(Path(draft_model_dir).resolve()),
        "target_model_dir": str(Path(target_model_dir).resolve()),
        "mode": mode,
        "context_max_length": _cache_capacity(preflight.metadata),
    }


def _find_external_data(graph_path: Path) -> Path:
    candidates = (
        graph_path.with_name(f"{graph_path.stem.removesuffix('_with_act')}_external_data"),
        graph_path.with_name("model_external_data"),
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"HunyuanOCR DFlash external data does not exist beside {graph_path}")


def export_draft_graphs(
    preflight: DraftExportPreflight,
    *,
    target_model_dir: str | Path,
    draft_model_dir: str | Path,
) -> dict[str, tuple[Path, Path]]:
    from xhmodel_merak.xh_llm.models.hunyuan_ocr import (
        XHHunYuanOCRDFlashConfig,
        XHHunYuanOCRDFlashModel,
    )

    artifacts: dict[str, tuple[Path, Path]] = {}
    for mode in ("context", "context_decode", "decode"):
        output_dir = preflight.output_root / f"dflash_draft_{mode}"
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(f"Refusing to overwrite non-empty draft graph directory: {output_dir}")
        config = XHHunYuanOCRDFlashConfig(
            **build_draft_model_config(
                preflight,
                mode=mode,
                target_model_dir=target_model_dir,
                draft_model_dir=draft_model_dir,
            )
        )
        graph_meta = XHHunYuanOCRDFlashModel(config).export_hmonnx(str(output_dir))
        graph_path = Path(graph_meta.hmonnx).resolve()
        artifacts[mode] = (graph_path, _find_external_data(graph_path))
    return artifacts


def _relative_artifact(path: Path, root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(f"Draft artifact must stay inside metadata root {root}: {resolved}") from error


def _normalise_metadata_paths(metadata: Any, root: Path) -> None:
    for field in (
        "hf_config",
        "quant_embedding",
        "prefill_hmonnx",
        "prefill_external_data",
        "decode_hmonnx",
        "decode_external_data",
        "verify_hmonnx",
        "verify_external_data",
    ):
        value = getattr(metadata, field, "")
        if value:
            setattr(metadata, field, _relative_artifact(Path(value), root))
    visual_config = getattr(metadata, "visual_config", None)
    if visual_config is not None:
        for field in ("hmonnx", "external_data"):
            value = getattr(visual_config, field, "")
            if value:
                setattr(visual_config, field, _relative_artifact(Path(value), root))
    for visual_meta in getattr(metadata, "visual_buckets", {}).values():
        for field in ("hmonnx", "external_data"):
            value = getattr(visual_meta, field, "")
            if value:
                setattr(visual_meta, field, _relative_artifact(Path(value), root))


def _prepare_upgraded_metadata(
    preflight: DraftExportPreflight,
    *,
    artifacts: Mapping[str, tuple[Path, Path]],
) -> Path:
    from xhmodel_merak.xh_llm.models.hunyuan_ocr import build_hunyuan_ocr_dflash_contract

    expected_modes = {"context", "context_decode", "decode"}
    if set(artifacts) != expected_modes:
        raise ValueError(f"Draft artifacts must contain exactly {sorted(expected_modes)!r}")
    for graph_path, external_data in artifacts.values():
        if not graph_path.is_file() or not external_data.exists():
            raise FileNotFoundError(f"Draft graph artifact pair is incomplete: {graph_path}, {external_data}")
    metadata = copy.deepcopy(preflight.metadata)
    root = preflight.output_root
    _normalise_metadata_paths(metadata, root)
    for mode, (graph_path, external_data) in artifacts.items():
        setattr(metadata, f"dflash_{mode}_hmonnx", _relative_artifact(graph_path, root))
        setattr(metadata, f"dflash_{mode}_external_data", _relative_artifact(external_data, root))
    spec_decode = copy.deepcopy(dict(metadata.spec_decode))
    spec_decode["status"] = "speculative_runtime_ready"
    spec_decode["capabilities"] = {
        "target_hidden": True,
        "target_verify": True,
        "draft_graphs": True,
        "speculative_runtime": True,
    }
    spec_decode["draft"] = build_hunyuan_ocr_dflash_contract(
        preflight.checkpoint,
        generation_eos_token_id=_generation_eos_token_id(metadata),
        cache_capacity=_cache_capacity(metadata),
    )
    metadata.spec_decode = spec_decode
    temporary = preflight.metadata_path.with_name(f".{preflight.metadata_path.name}.tmp-{os.getpid()}")
    metadata.save(temporary)
    return temporary


def _atomic_upgrade_metadata(
    metadata_path: str | Path,
    preflight: DraftExportPreflight,
    *,
    artifacts: Mapping[str, tuple[Path, Path]],
) -> Path:
    destination = Path(metadata_path).resolve()
    temporary = _prepare_upgraded_metadata(preflight, artifacts=artifacts)
    try:
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def upgrade_metadata_with_draft_graphs(
    preflight: DraftExportPreflight,
    *,
    artifacts: Mapping[str, tuple[Path, Path]],
    meta_path: str | Path | None = None,
) -> Path:
    destination = Path(meta_path).resolve() if meta_path is not None else preflight.metadata_path
    if destination == preflight.metadata_path:
        return _atomic_upgrade_metadata(destination, preflight, artifacts=artifacts)
    adjusted = DraftExportPreflight(
        metadata=preflight.metadata,
        checkpoint=preflight.checkpoint,
        metadata_path=destination,
        output_root=destination.parent,
    )
    return _atomic_upgrade_metadata(destination, adjusted, artifacts=artifacts)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--target-model-dir", type=Path, required=True)
    parser.add_argument("--draft-model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    from xhquant.api import xhquant_init

    args = _parse_args()
    metadata_path = args.metadata.resolve()
    if args.output_dir.resolve() != metadata_path.parent:
        raise ValueError("--output-dir must be the existing target metadata artifact root")
    preflight = preflight_draft_export(metadata_path, args.target_model_dir, args.draft_model_dir)
    xhquant_init(str(args.output_dir / "export_dflash_draft_hmonnx.log"), False)
    artifacts = export_draft_graphs(
        preflight,
        target_model_dir=args.target_model_dir,
        draft_model_dir=args.draft_model_dir,
    )
    print(upgrade_metadata_with_draft_graphs(preflight, artifacts=artifacts))


if __name__ == "__main__":
    main()