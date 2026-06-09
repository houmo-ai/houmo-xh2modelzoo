"""Helpers for inspecting HMONNX meta files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


_EMBEDDED_VISION_MODEL_TYPES = {
    "Gemma4ForConditionalGeneration",
    "Qwen3_5ForConditionalGeneration",
    "Qwen3_5MoeForConditionalGeneration",
}


def _extract_visual_config(meta_info: dict[str, Any]) -> dict[str, Any] | None:
    model_config = meta_info.get("model_config", {})
    if not isinstance(model_config, dict):
        return None

    model_visual_config = model_config.get("visual_config")
    if isinstance(model_visual_config, dict) and model_visual_config:
        return model_visual_config

    meta_visual_config = meta_info.get("visual_config")
    if isinstance(meta_visual_config, dict) and meta_visual_config:
        return meta_visual_config

    return None


def hmonnx_meta_dict_has_embedded_vision(meta_info: dict[str, Any]) -> bool:
    model_config = meta_info.get("model_config", {})
    if not isinstance(model_config, dict):
        return False

    visual_config = _extract_visual_config(meta_info)
    if visual_config is None:
        return False

    model_type = str(model_config.get("model_type") or "")
    if model_type in _EMBEDDED_VISION_MODEL_TYPES:
        return True

    visual_model_type = str(visual_config.get("model_type") or "")
    return model_type.endswith("ForConditionalGeneration") and visual_model_type.endswith("_visual")


def resolve_hmonnx_embedded_vision_meta_path(meta_info_path: Path) -> Path | None:
    return _resolve_hmonnx_embedded_vision_meta_path(meta_info_path, set())


def hmonnx_meta_has_embedded_vision(meta_info_path: Path) -> bool:
    return resolve_hmonnx_embedded_vision_meta_path(meta_info_path) is not None


def _resolve_hmonnx_embedded_vision_meta_path(
    meta_info_path: Path,
    visited_paths: set[Path],
) -> Path | None:
    resolved_path = meta_info_path.expanduser().resolve()
    if resolved_path in visited_paths or not resolved_path.is_file():
        return None
    visited_paths.add(resolved_path)

    try:
        meta_info = json.loads(resolved_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    if hmonnx_meta_dict_has_embedded_vision(meta_info):
        return resolved_path

    exported_dir = meta_info.get("exported_dir")
    if not exported_dir:
        return None

    exported_path = Path(str(exported_dir))
    if not exported_path.is_absolute():
        exported_path = resolved_path.parent / exported_path
    return _resolve_hmonnx_embedded_vision_meta_path(
        exported_path / "golden_meta_info.json",
        visited_paths,
    )