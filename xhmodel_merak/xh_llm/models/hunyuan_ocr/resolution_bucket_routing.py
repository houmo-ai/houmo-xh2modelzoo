# Copyright 2025 HOUMO AI
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def route_approved_image(
    source_width: int,
    source_height: int,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Route one image through an already validated approved manifest."""

    if source_width <= 0 or source_height <= 0:
        raise ValueError("source image dimensions must be positive")
    if manifest.get("status") != "approved":
        raise ValueError("HunyuanOCR image routing requires an approved manifest")
    routing = manifest["routing"]
    resource_limits = manifest["resource_limits"]
    candidates = [
        _candidate_route(source_width, source_height, bucket, routing, resource_limits)
        for bucket in manifest["buckets"]
    ]
    feasible = [candidate for candidate in candidates if not candidate["failed"]]
    public_candidates = [
        {key: value for key, value in candidate.items() if not key.startswith("_")}
        for candidate in candidates
    ]
    if not feasible:
        return {
            "accepted": False,
            "source_size": [source_width, source_height],
            "reason": "no_feasible_bucket",
            "candidates": public_candidates,
        }
    selected = min(
        feasible,
        key=lambda item: (
            -item["_fidelity_scale"],
            -item["_content_ratio"],
            item["image_token_count"],
            item["bucket_id"],
        ),
    )
    return {
        "accepted": True,
        "source_size": [source_width, source_height],
        "bucket_id": selected["bucket_id"],
        "bucket_size": selected["bucket_size"],
        "geometry": selected["geometry"],
        "image_token_count": selected["image_token_count"],
        "candidates": public_candidates,
    }


def _candidate_route(
    source_width: int,
    source_height: int,
    bucket: Mapping[str, Any],
    routing: Mapping[str, Any],
    resource_limits: Mapping[str, Any],
) -> dict[str, Any]:
    geometry = compute_letterbox_geometry(source_width, source_height, int(bucket["width"]), int(bucket["height"]))
    raw_scale = min(bucket["width"] / source_width, bucket["height"] / source_height)
    raw_fidelity_scale = min(1.0, raw_scale)
    resized_width, resized_height = geometry["resized_size"]
    raw_content_ratio = (resized_width * resized_height) / (bucket["width"] * bucket["height"])
    exceeded_fields = [
        field
        for field, limit_field in (
            ("image_token_count", "max_image_tokens"),
            ("canonical_prefill_length", "max_canonical_prefill_length"),
            ("vision_attention_pair_count", "max_vision_attention_pair_count"),
        )
        if bucket[field] > resource_limits[limit_field]
    ]
    failures = []
    if raw_content_ratio < routing["min_content_ratio"]:
        failures.append("content_ratio_below_minimum")
    if raw_fidelity_scale < routing["min_downscale"]:
        failures.append("downscale_below_minimum")
    if exceeded_fields:
        failures.append("resource_limit_exceeded")
    return {
        "bucket_id": bucket["id"],
        "bucket_size": [bucket["width"], bucket["height"]],
        "geometry": geometry,
        "image_token_count": bucket["image_token_count"],
        "canonical_prefill_length": bucket["canonical_prefill_length"],
        "vision_attention_pair_count": bucket["vision_attention_pair_count"],
        "resource_limit_exceeded_fields": exceeded_fields,
        "failed": failures,
        "_fidelity_scale": raw_fidelity_scale,
        "_content_ratio": raw_content_ratio,
    }


def compute_letterbox_geometry(
    source_width: int,
    source_height: int,
    bucket_width: int,
    bucket_height: int,
) -> dict[str, Any]:
    if min(source_width, source_height, bucket_width, bucket_height) <= 0:
        raise ValueError("image dimensions must be positive")
    scale = min(bucket_width / source_width, bucket_height / source_height)
    fidelity_scale = min(1.0, scale)
    resized_width = max(1, min(bucket_width, round(source_width * scale)))
    resized_height = max(1, min(bucket_height, round(source_height * scale)))
    pad_x = bucket_width - resized_width
    pad_y = bucket_height - resized_height
    left = pad_x // 2
    top = pad_y // 2
    return {
        "source_size": [source_width, source_height],
        "bucket_size": [bucket_width, bucket_height],
        "raw_scale": round(scale, 6),
        "fidelity_scale": round(fidelity_scale, 6),
        "resized_size": [resized_width, resized_height],
        "padding_ltrb": [left, top, pad_x - left, pad_y - top],
        "content_ratio": round((resized_width * resized_height) / (bucket_width * bucket_height), 6),
    }


__all__ = ["compute_letterbox_geometry", "route_approved_image"]
