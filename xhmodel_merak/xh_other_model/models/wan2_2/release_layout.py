# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: I001

# pyright: reportMissingImports=false

"""Build a release-ready Wan2.2 directory from workflow artifacts."""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping

import onnx
from onnx.external_data_helper import uses_external_data

_FRONTEND_RELEASE_NAMES = {
    "t5": "quant_embedding.pt",
    "low_noise_model": "low_noise_timestep_embedding.pt",
    "high_noise_model": "high_noise_timestep_embedding.pt",
}


def build_release_directory(
    source_work_dir: str | Path,
    output_dir: str | Path,
    *,
    release_date: str | None = None,
    release_prefix: str | None = None,
    overwrite: bool = False,
) -> Path:
    """Convert a completed Wan2.2 export into the HM release layout."""
    source = Path(source_work_dir).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    export_meta = _read_json(source / "export_meta_info.json")
    legacy_meta_path = source / str(export_meta.get("legacy_meta", "wan2_2_export_meta.json"))
    legacy_meta = _read_json(legacy_meta_path)
    components = tuple(legacy_meta.get("export_components") or ())
    if not components:
        raise ValueError(f"No Wan2.2 release components found under {source}")

    prefix = release_prefix or build_release_prefix(
        target_device=str(export_meta.get("target_device", "XH2a")),
        task=str(legacy_meta.get("task", "wan2_2")),
        quant_types=legacy_meta.get("quant_types") or {},
        release_date=release_date,
    )
    if destination.exists():
        if not overwrite:
            raise FileExistsError(f"Release directory already exists: {destination}")
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    temp_dir = Path(tempfile.mkdtemp(prefix=f".{prefix}.", dir=destination.parent))
    try:
        component_manifest: dict[str, dict[str, Any]] = {}
        updated_legacy = dict(legacy_meta)
        for component in components:
            component_meta_name = str(legacy_meta.get(f"{component}_meta", f"{component}_meta.json"))
            source_component_meta = _read_json(source / component_meta_name)
            released_meta, manifest_entry = _materialize_component(
                source,
                temp_dir,
                prefix,
                component,
                source_component_meta,
                str((legacy_meta.get("quant_types") or {}).get(component, "")),
            )
            released_meta_name = f"{component}_meta.json"
            _write_json(temp_dir / released_meta_name, released_meta)
            updated_legacy[f"{component}_meta"] = released_meta_name
            component_manifest[component] = manifest_entry

        updated_legacy["release_prefix"] = prefix
        updated_legacy["source_work_dir"] = str(source)
        _write_json(temp_dir / "wan2_2_export_meta.json", updated_legacy)
        config_file = _copy_config(source, export_meta, temp_dir, prefix)
        release_meta = {
            "format_version": 1,
            "model_name": "Wan2.2",
            "model_type": export_meta.get("model_type"),
            "release_prefix": prefix,
            "create_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "source_model_dir": export_meta.get("source_model_dir"),
            "source_work_dir": str(source),
            "target_device": export_meta.get("target_device", "XH2a"),
            "task": legacy_meta.get("task"),
            "config": config_file,
            "legacy_meta": "wan2_2_export_meta.json",
            "components": component_manifest,
            "wan2_2": updated_legacy,
        }
        _write_json(temp_dir / "export_meta_info.json", release_meta)
        _validate_release_directory(temp_dir, component_manifest)
        temp_dir.rename(destination)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    return destination


def build_release_prefix(
    *,
    target_device: str,
    task: str,
    quant_types: Mapping[str, str],
    release_date: str | None = None,
) -> str:
    """Create a lowercase Wan2.2 HM release prefix."""
    device = target_device.lower()
    if device.startswith("xh2"):
        device = "xh2"
    elif device.startswith("xh1"):
        device = "xh1"
    else:
        raise ValueError(f"Unsupported HM release target device: {target_device!r}")
    task_name = re.sub(r"[^a-z0-9]+", "_", task.lower()).strip("_")
    bit_widths = {_quant_bit_width(value) for value in quant_types.values() if value}
    quant_name = bit_widths.pop() if len(bit_widths) == 1 else "wmix_amix"
    date = release_date or time.strftime("%Y%m%d", time.localtime())
    if not re.fullmatch(r"\d{8}", date):
        raise ValueError(f"release_date must use YYYYMMDD, got {date!r}")
    return f"hmquant_{device}_wan2_2_{task_name}_{quant_name}_{date}"


def _materialize_component(
    source: Path,
    root: Path,
    prefix: str,
    component: str,
    source_meta: Mapping[str, Any],
    quant_type: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source_hmonnx = source / str(source_meta["hmonnx_file"])
    if not source_hmonnx.is_file():
        raise FileNotFoundError(f"Missing Wan2.2 HMONNX for {component}: {source_hmonnx}")
    component_dir = root / component
    component_dir.mkdir(parents=True)
    graph_stem = f"{_component_release_prefix(prefix, quant_type)}_{component}"
    hmonnx_file = component_dir / f"{graph_stem}_with_act.onnx"
    external_file = _materialize_onnx(source_hmonnx, hmonnx_file)

    frontend_value = source_meta.get("frontend_file")
    frontend_file = None
    if frontend_value:
        source_frontend = source / str(frontend_value)
        if not source_frontend.is_file():
            raise FileNotFoundError(f"Missing Wan2.2 frontend weights for {component}: {source_frontend}")
        frontend_name = _FRONTEND_RELEASE_NAMES.get(component, f"{component}_{source_frontend.name}")
        frontend_file = root / frontend_name
        _link_or_copy(source_frontend, frontend_file)

    released_meta = dict(source_meta)
    released_meta["hmonnx_file"] = hmonnx_file.relative_to(root).as_posix()
    released_meta["golden_dir"] = None
    if frontend_file is not None:
        released_meta["frontend_file"] = frontend_file.relative_to(root).as_posix()
    manifest = {
        "component_dir": component,
        "hmonnx_file": released_meta["hmonnx_file"],
        "external_data": external_file.relative_to(root).as_posix() if external_file is not None else None,
        "frontend_file": released_meta.get("frontend_file"),
        "golden_dir": None,
        "quant_type": quant_type,
        "runtime": released_meta,
    }
    return released_meta, manifest


def _materialize_onnx(source: Path, destination: Path) -> Path | None:
    """Copy the graph without changing its exported external-data location."""
    model = onnx.load_model(str(source), load_external_data=False)
    locations = _external_locations(model)
    if len(locations) > 1:
        raise ValueError(f"Expected at most one external data file in {source}, got {sorted(locations)}")
    _link_or_copy(source, destination)
    if not locations:
        return None

    location = next(iter(locations))
    relative_location = Path(location)
    if relative_location.is_absolute() or ".." in relative_location.parts:
        raise ValueError(f"Unsafe ONNX external data location in {source}: {location!r}")
    source_external = source.parent / relative_location
    if not source_external.is_file():
        raise FileNotFoundError(f"Missing ONNX external data: {source_external}")
    external_file = destination.parent / relative_location
    _link_or_copy(source_external, external_file)
    return external_file


def _copy_config(source: Path, export_meta: Mapping[str, Any], root: Path, prefix: str) -> str | None:
    config_value = export_meta.get("config")
    if not config_value:
        return None
    source_file = source / str(config_value)
    if not source_file.is_file():
        raise FileNotFoundError(f"Wan2.2 workflow config is missing: {source_file}")
    destination = root / f"{prefix}_export_config{source_file.suffix}"
    _link_or_copy(source_file, destination)
    return destination.relative_to(root).as_posix()


def _validate_release_directory(root: Path, components: Mapping[str, Mapping[str, Any]]) -> None:
    for component, values in components.items():
        for key in ("hmonnx_file", "external_data", "frontend_file"):
            value = values.get(key)
            if value and not (root / str(value)).is_file():
                raise FileNotFoundError(f"Incomplete Wan2.2 release component {component}: {key}={value}")
        hmonnx_file = root / str(values["hmonnx_file"])
        model = onnx.load_model(str(hmonnx_file), load_external_data=False)
        external_value = values.get("external_data")
        expected = (
            {(root / str(external_value)).relative_to(hmonnx_file.parent).as_posix()} if external_value else set()
        )
        if _external_locations(model) != expected:
            raise ValueError(f"Incorrect external_data reference for Wan2.2 component {component}")
    if not (root / "export_meta_info.json").is_file():
        raise FileNotFoundError("Wan2.2 release is missing export_meta_info.json")


def _component_release_prefix(release_prefix: str, quant_type: str) -> str:
    return re.sub(
        r"_(?:wmix_amix|w\d+a\d+)_",
        f"_{_quant_bit_width(quant_type)}_",
        release_prefix,
        count=1,
    )


def _quant_bit_width(value: str) -> str:
    match = re.match(r"^(w\d+a\d+)", value.lower())
    return match.group(1) if match else "wmix_amix"


def _iter_tensors(model: onnx.ModelProto):
    yield from model.graph.initializer
    for sparse in model.graph.sparse_initializer:
        yield sparse.values
        yield sparse.indices


def _external_locations(model: onnx.ModelProto) -> set[str]:
    locations: set[str] = set()
    for tensor in _iter_tensors(model):
        if not uses_external_data(tensor):
            continue
        for entry in tensor.external_data:
            if entry.key == "location":
                locations.add(entry.value)
    return locations


def _link_or_copy(source: Path, destination: Path) -> None:
    source = source.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, values: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(values, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
