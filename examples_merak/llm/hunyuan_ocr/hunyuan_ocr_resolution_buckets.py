# Copyright 2025 HOUMO AI
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from xhmodel_merak.xh_llm.models.hunyuan_ocr.resolution_bucket_routing import route_approved_image


_REQUIRED_TOP_LEVEL_FIELDS = {
    "schema_version",
    "status",
    "alignment",
    "input_policy",
    "routing",
    "resource_limits",
    "buckets",
    "multi_image_policy",
    "approval",
}
_REQUIRED_BUCKET_FIELDS = {
    "id",
    "width",
    "height",
    "image_grid_thw",
    "vit_patch_count",
    "merged_patch_hw",
    "image_token_count",
    "canonical_prefill_length",
    "vision_attention_pair_count",
    "legacy_baseline",
    "approval_rationale",
}


def load_resolution_bucket_manifest(path: str | Path, *, require_approved: bool = False) -> dict[str, Any]:
    manifest_path = Path(path)
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("HunyuanOCR resolution bucket manifest must be a JSON object")
    missing = sorted(_REQUIRED_TOP_LEVEL_FIELDS - set(value))
    if missing:
        raise ValueError(f"HunyuanOCR resolution bucket manifest is missing field: {missing[0]}")
    unknown = sorted(set(value) - _REQUIRED_TOP_LEVEL_FIELDS)
    if unknown:
        raise ValueError(f"HunyuanOCR resolution bucket manifest has unknown field: {unknown[0]}")
    if value["schema_version"] != 2:
        raise ValueError("HunyuanOCR resolution bucket manifest must use schema_version 2")
    if require_approved and value.get("status") != "approved":
        raise ValueError("HunyuanOCR resolution bucket manifest must be approved")
    buckets = value.get("buckets")
    if not isinstance(buckets, list) or not buckets:
        raise ValueError("HunyuanOCR resolution bucket manifest requires non-empty buckets")
    for bucket in buckets:
        if not isinstance(bucket, dict):
            raise ValueError("HunyuanOCR resolution bucket entries must be JSON objects")
        missing_bucket_fields = sorted(_REQUIRED_BUCKET_FIELDS - set(bucket))
        if missing_bucket_fields:
            raise ValueError(f"HunyuanOCR resolution bucket is missing field: {missing_bucket_fields[0]}")
        unknown_bucket_fields = sorted(set(bucket) - _REQUIRED_BUCKET_FIELDS)
        if unknown_bucket_fields:
            raise ValueError(f"HunyuanOCR resolution bucket has unknown field: {unknown_bucket_fields[0]}")
        if bucket["width"] <= 0 or bucket["height"] <= 0:
            raise ValueError("HunyuanOCR resolution bucket dimensions must be positive")
    return value


def route_image(source_width: int, source_height: int, manifest: dict[str, Any]) -> dict[str, Any]:
    return route_approved_image(source_width, source_height, manifest)


__all__ = ["load_resolution_bucket_manifest", "route_image"]
