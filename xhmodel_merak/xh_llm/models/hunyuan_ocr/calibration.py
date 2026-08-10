# Copyright 2025 HOUMO AI
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def load_calibration_samples(calibration_config: Mapping[str, Any] | None) -> tuple[str, tuple[Path, ...]]:
    if not isinstance(calibration_config, Mapping):
        raise ValueError("HunyuanOCR W8 calibration requires a calibration configuration")

    prompt = calibration_config.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("HunyuanOCR calibration prompt must be a non-empty string")

    image_values = calibration_config.get("images")
    if not isinstance(image_values, Sequence) or isinstance(image_values, (str, bytes)) or not image_values:
        raise ValueError("HunyuanOCR calibration images must be a non-empty sequence")

    image_paths = tuple(Path(value).expanduser().resolve() for value in image_values)
    missing = [str(path) for path in image_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"HunyuanOCR calibration image does not exist: {missing[0]}")
    return prompt.strip(), image_paths


def build_calibration_messages(image_path: Path, prompt: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def load_reference_trajectories(
    calibration_config: Mapping[str, Any] | None,
    *,
    request_count: int,
    image_paths: Sequence[Path] | None = None,
) -> tuple[tuple[int, tuple[int, ...]], ...]:
    if not isinstance(calibration_config, Mapping):
        raise ValueError("HunyuanOCR W8 calibration requires a calibration configuration")
    values = calibration_config.get("reference_trajectories")
    manifest_value = calibration_config.get("reference_trajectory_manifest")
    validate_image_hashes = False
    if values is None and manifest_value is not None:
        manifest_path = Path(manifest_value).expanduser().resolve()
        if not manifest_path.is_file():
            raise FileNotFoundError(f"HunyuanOCR reference trajectory manifest does not exist: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, Mapping) or manifest.get("schema_version") != 1:
            raise ValueError("HunyuanOCR reference trajectory manifest must use schema_version 1")
        values = manifest.get("trajectories")
        validate_image_hashes = image_paths is not None
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or not values:
        raise ValueError("HunyuanOCR W8 calibration requires non-empty reference_trajectories")

    trajectories = []
    seen_request_indices = set()
    for value in values:
        if not isinstance(value, Mapping):
            raise ValueError("HunyuanOCR reference trajectory entries must be mappings")
        request_index = value.get("request_index")
        if type(request_index) is not int or not 0 <= request_index < request_count:
            raise ValueError(
                "HunyuanOCR reference trajectory request_index must identify a calibration image: "
                f"{request_index!r}"
            )
        if request_index in seen_request_indices:
            raise ValueError(f"HunyuanOCR reference trajectory request_index is duplicated: {request_index}")
        if validate_image_hashes:
            image_sha256 = value.get("image_sha256")
            if not isinstance(image_sha256, str) or len(image_sha256) != 64:
                raise ValueError("HunyuanOCR reference trajectory manifest requires image_sha256")
            actual_sha256 = hashlib.sha256(image_paths[request_index].read_bytes()).hexdigest()
            if actual_sha256 != image_sha256:
                raise ValueError(
                    "HunyuanOCR reference trajectory image SHA256 mismatch for "
                    f"request_index={request_index}: expected {image_sha256}, got {actual_sha256}"
                )
        token_values = value.get("token_ids")
        if (
            not isinstance(token_values, Sequence)
            or isinstance(token_values, (str, bytes))
            or not token_values
            or any(type(token_id) is not int or token_id < 0 for token_id in token_values)
        ):
            raise ValueError("HunyuanOCR reference trajectory token_ids must be a non-empty sequence of integers")
        seen_request_indices.add(request_index)
        trajectories.append((request_index, tuple(token_values)))
    return tuple(trajectories)


__all__ = [
    "build_calibration_messages",
    "load_calibration_samples",
    "load_reference_trajectories",
]
