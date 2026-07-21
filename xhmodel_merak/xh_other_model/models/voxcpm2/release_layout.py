"""Build a release-ready VoxCPM2 directory from workflow export artifacts."""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import onnx
from onnx.external_data_helper import uses_external_data


_COMPONENT_NAMES = {
    "locenc": "locenc",
    "locdit": "locdit",
    "audiovae_encoder": "audiovae_encoder_np128",
    "audiovae_decoder_stream": "audiovae_decoder_stream_np3",
    "audiovae_decoder_full": "audiovae_decoder_full_np128",
    "audiovae_decoder_stateful": "audiovae_decoder_stateful_np1",
}

_QUANT_EMBEDDING_PATH = Path("quant_embedding.pt")


@dataclass(frozen=True)
class ReleaseComponent:
    name: str
    source_onnx: Path
    golden_dir: Path | None
    source_meta: dict[str, Any]
    quant_type: str


def build_release_prefix(
    *,
    target_device: str,
    model_name: str,
    quant_types: list[str],
    prefill_length: int,
    context_length: int,
    release_date: str | None = None,
) -> str:
    """Create the lowercase HM release prefix."""
    device = target_device.lower()
    if device.startswith("xh2"):
        device = "xh2"
    elif device.startswith("xh1"):
        device = "xh1"
    else:
        raise ValueError(f"Unsupported HM release target device: {target_device!r}")

    model = re.sub(r"[^a-z0-9]+", "_", model_name.lower()).strip("_")
    if not model:
        raise ValueError("model_name must contain at least one alphanumeric character")

    bit_widths = {_quant_bit_width(value) for value in quant_types if value}
    quant_name = bit_widths.pop() if len(bit_widths) == 1 else "wmix_amix"
    date = release_date or time.strftime("%Y%m%d", time.localtime())
    if not re.fullmatch(r"\d{8}", date):
        raise ValueError(f"release_date must use YYYYMMDD, got {date!r}")

    return "_".join(
        (
            "hmquant",
            device,
            model,
            quant_name,
            str(int(prefill_length)),
            _format_context_length(context_length),
            date,
        )
    )


def build_release_directory(
    source_work_dir: str | Path,
    output_dir: str | Path,
    *,
    release_date: str | None = None,
    release_prefix: str | None = None,
    overwrite: bool = False,
) -> Path:
    """Convert a completed workflow export into the HM release layout."""
    source = Path(source_work_dir).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    export_meta = _read_json(source / "export_meta_info.json")
    source_golden_meta = (
        _read_json(source / "golden_meta_info.json")
        if (source / "golden_meta_info.json").is_file()
        else {}
    )
    lm_meta_path = source / "lm_export_meta_info.json"
    lm_meta = _read_json(lm_meta_path) if lm_meta_path.is_file() else {}

    components = _collect_components(source, export_meta, lm_meta)
    if not components:
        raise ValueError(f"No release components found under {source}")

    prefix = release_prefix or build_release_prefix(
        target_device=str(export_meta.get("target_device", "XH2a")),
        model_name=str(lm_meta.get("model_name", export_meta.get("model_name", source.name))),
        quant_types=[component.quant_type for component in components],
        prefill_length=int(lm_meta.get("prefill_length", export_meta["prefill_length"])),
        context_length=int(lm_meta.get("cache_length", export_meta["context_length"])),
        release_date=release_date,
    )
    if prefix != prefix.lower():
        raise ValueError(f"release_prefix must be lowercase, got {prefix!r}")

    if destination.exists():
        if not overwrite:
            raise FileExistsError(f"Release directory already exists: {destination}")
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    temp_dir = Path(tempfile.mkdtemp(prefix=f".{prefix}.", dir=destination.parent))
    try:
        component_manifest: dict[str, dict[str, Any]] = {}
        for component in components:
            component_manifest[component.name] = _materialize_component(temp_dir, prefix, component)

        hf_config_dir = _copy_hf_config(source, export_meta, lm_meta, temp_dir)
        host_meta = lm_meta if lm_meta.get("host_modules") else export_meta
        host_modules = _copy_host_modules(source, host_meta, temp_dir, prefix)
        config_file = _copy_export_config(source, export_meta, temp_dir, prefix)

        manifest = {
            "release_prefix": prefix,
            "create_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "source_work_dir": str(source),
            "hf_model": export_meta.get("hf_model"),
            "target_device": export_meta.get("target_device", "XH2a"),
            "dtype": export_meta.get("dtype", "float16"),
            "export_device": export_meta.get("export_device") or source_golden_meta.get("device"),
            "prefill_length": int(lm_meta.get("prefill_length", export_meta["prefill_length"])),
            "context_length": int(lm_meta.get("cache_length", export_meta["context_length"])),
            "hf_config": str(hf_config_dir.relative_to(temp_dir)),
            "quant_embedding": (
                _QUANT_EMBEDDING_PATH.as_posix()
                if (temp_dir / _QUANT_EMBEDDING_PATH).exists()
                else None
            ),
            "host_modules": host_modules,
            "components": component_manifest,
        }
        export_meta_info = {
            "format_version": 1,
            "model_name": "VoxCPM2",
            "release_prefix": prefix,
            "create_time": manifest["create_time"],
            "hf_model": manifest["hf_model"],
            "target_device": manifest["target_device"],
            "dtype": manifest["dtype"],
            "export_device": manifest["export_device"],
            "prefill_length": manifest["prefill_length"],
            "context_length": manifest["context_length"],
            "config": config_file,
            "hf_config": manifest["hf_config"],
            "quant_embedding": manifest["quant_embedding"],
            "host_modules": manifest["host_modules"],
            "components": {
                name: _canonical_component(values)
                for name, values in component_manifest.items()
            },
        }
        if any(values["step_dir"] is not None for values in component_manifest.values()):
            export_meta_info["golden"] = {
                "create_time": manifest["create_time"],
                "device": manifest["export_device"],
            }
        _write_json(temp_dir / "export_meta_info.json", export_meta_info)
        _validate_release_directory(temp_dir, prefix, components)
        temp_dir.rename(destination)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    return destination


def finalize_export_metadata(work_dir: str | Path) -> Path:
    """Index direct exporter outputs without copying or renaming graph files."""
    root = Path(work_dir).expanduser().resolve()
    export_meta = _read_json(root / "export_meta_info.json")
    lm_meta_path = root / "lm_export_meta_info.json"
    lm_meta = _read_json(lm_meta_path) if lm_meta_path.is_file() else {}
    components = _collect_components(root, export_meta, lm_meta)
    if not components:
        raise ValueError(f"No release components found under {root}")

    prefix = build_release_prefix(
        target_device=str(export_meta.get("target_device", "XH2a")),
        model_name=str(lm_meta.get("model_name", export_meta.get("model_name", root.name))),
        quant_types=[component.quant_type for component in components],
        prefill_length=int(lm_meta.get("prefill_length", export_meta["prefill_length"])),
        context_length=int(lm_meta.get("cache_length", export_meta["context_length"])),
    )
    component_manifest = {
        component.name: _direct_component(root, component)
        for component in components
    }
    hf_config_value = lm_meta.get("hf_config")
    if hf_config_value and (root / str(hf_config_value) / "config.json").is_file():
        hf_config_dir = root / str(hf_config_value)
    else:
        hf_config_dir = _copy_hf_config(root, export_meta, lm_meta, root)
    host_modules = {
        name: str(path)
        for name, path in lm_meta.get("host_modules", {}).items()
    }

    config_value = export_meta.get("config")
    if not config_value or not (root / str(config_value)).is_file():
        raise FileNotFoundError(f"Workflow config is missing from {root}: {config_value}")

    create_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    final_meta = {
        "format_version": 1,
        "model_name": str(export_meta.get("model_name", "VoxCPM2")),
        "release_prefix": prefix,
        "create_time": create_time,
        "hf_model": export_meta.get("hf_model"),
        "target_device": export_meta.get("target_device", "XH2a"),
        "dtype": export_meta.get("dtype", "float16"),
        "export_device": export_meta.get("export_device"),
        "prefill_length": int(lm_meta.get("prefill_length", export_meta["prefill_length"])),
        "context_length": int(lm_meta.get("cache_length", export_meta["context_length"])),
        "config": str(config_value),
        "hf_config": str(hf_config_dir.relative_to(root)),
        "quant_embedding": host_modules.get("token_embedding"),
        "host_modules": host_modules,
        "components": {
            name: values
            for name, values in component_manifest.items()
        },
    }
    _write_json(root / "export_meta_info.json", final_meta)
    _validate_direct_export_metadata(root, final_meta)
    return root


def _direct_component(root: Path, component: ReleaseComponent) -> dict[str, Any]:
    hmonnx_file = component.source_onnx.resolve()
    if not hmonnx_file.is_file():
        raise FileNotFoundError(f"Missing HMONNX for {component.name}: {hmonnx_file}")
    runtime = deepcopy(component.source_meta)
    source_onnx_value = runtime.pop("onnx_file", None)
    for key in (
        "hmonnx_file",
        "golden_dir",
        "prefill_onnx",
        "decode_onnx",
        "prefill_golden",
        "decode_golden",
        "hf_config",
        "host_modules",
    ):
        runtime.pop(key, None)
    calibration = runtime.get("calibration")
    if isinstance(calibration, dict):
        calibration.pop("wav_path", None)

    relative_hmonnx = hmonnx_file.relative_to(root)
    external_data = _direct_external_data(root, hmonnx_file)
    onnx_file = str(source_onnx_value) if source_onnx_value else None
    if onnx_file and not (root / onnx_file).is_file():
        onnx_file = None
    return {
        "component_dir": relative_hmonnx.parent.as_posix(),
        "onnx_file": onnx_file,
        "hmonnx_file": relative_hmonnx.as_posix(),
        "external_data": external_data,
        "golden_dir": None,
        "quant_type": component.quant_type,
        "runtime": runtime,
    }


def _direct_external_data(root: Path, hmonnx_file: Path) -> str | None:
    model = onnx.load_model(str(hmonnx_file), load_external_data=False)
    locations = _external_locations(model)
    if len(locations) > 1:
        raise ValueError(
            f"Expected at most one external data file in {hmonnx_file}, got {sorted(locations)}"
        )
    if not locations:
        return None
    external_file = hmonnx_file.parent / next(iter(locations))
    if not external_file.is_file():
        raise FileNotFoundError(f"Missing external data for {hmonnx_file}: {external_file}")
    return external_file.relative_to(root).as_posix()


def _validate_direct_export_metadata(root: Path, meta: dict[str, Any]) -> None:
    for key in ("config", "hf_config"):
        value = meta.get(key)
        if not value or not (root / str(value)).exists():
            raise FileNotFoundError(f"Export metadata path is missing: {key}={value}")
    for name, values in meta["components"].items():
        hmonnx_file = root / values["hmonnx_file"]
        if not hmonnx_file.is_file():
            raise FileNotFoundError(f"Missing HMONNX for {name}: {hmonnx_file}")
        external_data = values.get("external_data")
        if external_data and not (root / external_data).is_file():
            raise FileNotFoundError(f"Missing external data for {name}: {external_data}")
    for name, value in meta.get("host_modules", {}).items():
        if not (root / value).is_file():
            raise FileNotFoundError(f"Missing host module {name}: {value}")


def _collect_components(
    source: Path,
    export_meta: dict[str, Any],
    lm_meta: dict[str, Any],
) -> list[ReleaseComponent]:
    components: list[ReleaseComponent] = []
    exported = export_meta.get("components", {})
    if lm_meta:
        for lm_key, release_name in (("base_lm", "baselm"), ("residual_lm", "residuallm")):
            values = lm_meta[lm_key]
            for phase in ("prefill", "decode"):
                component_name = f"{release_name}_{phase}"
                canonical = exported.get(component_name, {})
                golden_value = canonical.get("golden_dir") or values.get(f"{phase}_golden")
                golden_dir = _resolve_golden_parent(source, golden_value)
                if canonical.get("hmonnx_file"):
                    source_onnx = source / canonical["hmonnx_file"]
                else:
                    source_onnx = (
                        _find_with_act_onnx(golden_dir)
                        if golden_dir is not None
                        else source / values[f"{phase}_onnx"]
                    )
                source_meta = deepcopy(canonical.get("runtime") or values)
                components.append(
                    ReleaseComponent(
                        name=component_name,
                        source_onnx=source_onnx,
                        golden_dir=golden_dir,
                        source_meta=source_meta,
                        quant_type=str(canonical.get("quant_type", lm_meta["quant_type"])),
                    )
                )

    for key, release_name in _COMPONENT_NAMES.items():
        values = exported.get(key)
        if not values:
            values = exported.get(release_name)
        if not values and "_np" in release_name:
            prefix = release_name.rsplit("_np", 1)[0] + "_np"
            matches = [
                (name, candidate)
                for name, candidate in exported.items()
                if name.startswith(prefix)
            ]
            if len(matches) == 1:
                release_name, values = matches[0]
        if not values or not values.get("exists", True):
            continue
        if values.get("hmonnx_file"):
            meta = deepcopy(values.get("runtime") or {})
            meta.update(
                {
                    "hmonnx_file": values["hmonnx_file"],
                    "golden_dir": values.get("golden_dir"),
                    "quant_type": values.get("quant_type", meta.get("quant_type", "")),
                }
            )
        else:
            meta_path = source / values["meta_file"]
            meta = _read_json(meta_path)
        hmonnx_value = meta.get("hmonnx_file")
        if not hmonnx_value:
            continue
        hmonnx_file = source / hmonnx_value
        golden_value = meta.get("golden_dir")
        golden_dir = (
            _resolve_golden_parent(source, golden_value)
            if golden_value
            else hmonnx_file.parent / "golden"
        )
        if not golden_dir.is_dir():
            golden_dir = None
        if key.startswith("audiovae_") and "num_patches" in meta:
            expected_suffix = f"np{int(meta['num_patches'])}"
            if not release_name.endswith(expected_suffix):
                release_name = re.sub(r"np\d+$", expected_suffix, release_name)
        components.append(
            ReleaseComponent(
                name=release_name,
                source_onnx=hmonnx_file,
                golden_dir=golden_dir,
                source_meta=meta,
                quant_type=str(meta.get("quant_type", "")),
            )
        )
    return components


def _resolve_golden_parent(source: Path, value: str | None) -> Path | None:
    if not value:
        return None
    golden_dir = source / value
    return golden_dir.parent if golden_dir.name == "step_0" else golden_dir


def _materialize_component(root: Path, prefix: str, component: ReleaseComponent) -> dict[str, Any]:
    if not component.source_onnx.exists():
        raise FileNotFoundError(f"Missing HMONNX file for {component.name}: {component.source_onnx}")
    relative_dir = _component_relative_dir(component.name)
    component_dir = root / relative_dir
    component_dir.mkdir(parents=True)
    file_prefix = _component_release_prefix(prefix, component.quant_type)
    graph_stem = f"{file_prefix}_{component.name}"
    onnx_name = f"{graph_stem}_with_act.onnx"
    external_name = f"{graph_stem}_external_data"
    onnx_file = component_dir / onnx_name
    external_file = component_dir / external_name
    _materialize_onnx(component.source_onnx, onnx_file, external_file)

    step_dir = None
    if component.golden_dir is not None:
        source_step = component.golden_dir / "step_0"
        if not source_step.is_dir():
            raise FileNotFoundError(f"Missing step_0 golden for {component.name}: {source_step}")
        step_dir = component_dir / "step_0"
        _copy_golden_step(source_step, step_dir, graph_stem)
        (step_dir / onnx_name).symlink_to(Path("..") / onnx_name)
        (step_dir / external_name).symlink_to(Path("..") / external_name)

    return {
        "directory": relative_dir.as_posix(),
        "file_prefix": file_prefix,
        "with_act_onnx": (relative_dir / onnx_name).as_posix(),
        "external_data": (relative_dir / external_name).as_posix(),
        "step_dir": (relative_dir / "step_0").as_posix() if step_dir is not None else None,
        "quant_type": component.quant_type,
        "source_meta": component.source_meta,
    }


def _materialize_onnx(source: Path, destination: Path, external_file: Path) -> None:
    """Save an ONNX model with one correctly referenced, release-named data file."""
    source = source.resolve()
    model = onnx.load_model(str(source), load_external_data=False)
    locations = _external_locations(model)
    if len(locations) > 1:
        raise ValueError(f"Expected at most one external data file in {source}, got {sorted(locations)}")

    if locations:
        old_location = next(iter(locations))
        source_external = (source.parent / old_location).resolve()
        if not source_external.is_file():
            raise FileNotFoundError(f"Missing ONNX external data: {source_external}")
        _link_or_copy(source_external, external_file)
        for tensor in _iter_tensors(model):
            if uses_external_data(tensor):
                for entry in tensor.external_data:
                    if entry.key == "location":
                        entry.value = external_file.name
        onnx.save_model(model, str(destination))
        return

    model = onnx.load_model(str(source), load_external_data=True)
    onnx.save_model(
        model,
        str(destination),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=external_file.name,
        size_threshold=0,
        convert_attribute=False,
    )
    if not external_file.exists():
        external_file.touch()


def _copy_golden_step(source: Path, destination: Path, graph_stem: str) -> None:
    destination.mkdir(parents=True)
    old_graph_stem = _golden_graph_stem(source)
    old_data_stem = old_graph_stem.removesuffix("_with_act")
    new_graph_stem = f"{graph_stem}_with_act"

    for path in source.rglob("*"):
        if path.is_symlink():
            continue
        relative = path.relative_to(source)
        parts = list(relative.parts)
        if parts and parts[0] == old_graph_stem:
            parts[0] = new_graph_stem
        elif len(parts) == 1 and parts[0].startswith(old_data_stem):
            parts[0] = graph_stem + parts[0][len(old_data_stem):]
        target = destination.joinpath(*parts)
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif path.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            _link_or_copy(path, target)


def _golden_graph_stem(step_dir: Path) -> str:
    graph_dirs = sorted(
        path for path in step_dir.glob("*_with_act") if path.is_dir()
    )
    if len(graph_dirs) == 1:
        return graph_dirs[0].name

    links = sorted(
        path for path in step_dir.glob("*.onnx") if path.is_symlink()
    )
    if len(links) != 1:
        raise ValueError(
            f"Expected one golden graph directory or ONNX link under {step_dir}, "
            f"got {len(graph_dirs)} directories and {len(links)} links"
        )
    return links[0].stem


def _copy_hf_config(source: Path, export_meta: dict[str, Any], lm_meta: dict[str, Any], root: Path) -> Path:
    destination = root / "hf_config"
    destination.mkdir()
    model_dir_value = export_meta.get("hf_model") or lm_meta.get("hf_model")
    model_dir = Path(model_dir_value).expanduser().resolve() if model_dir_value else None
    copied: set[str] = set()
    if model_dir and model_dir.is_dir():
        for path in sorted(model_dir.iterdir()):
            if (
                path.is_file()
                and path.suffix.lower() in {".json", ".md", ".txt"}
                and not path.name.lower().startswith("readme")
            ):
                _link_or_copy(path, destination / path.name)
                copied.add(path.name)

    legacy_dir = source / str(lm_meta.get("hf_config", "ConfigFiles"))
    if legacy_dir.is_dir():
        for path in sorted(legacy_dir.iterdir()):
            if (
                path.is_file()
                and path.name not in copied
                and not path.name.lower().startswith("readme")
            ):
                _link_or_copy(path, destination / path.name)
                copied.add(path.name)
    if "config.json" not in copied:
        raise FileNotFoundError("hf_config/config.json was not found in the model or export directory")
    return destination


def _copy_host_modules(source: Path, lm_meta: dict[str, Any], root: Path, prefix: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for name, relative in lm_meta.get("host_modules", {}).items():
        source_file = source / relative
        if name == "token_embedding":
            target = root / _QUANT_EMBEDDING_PATH
        else:
            target = root / f"{prefix}_{name}.pt"
        _link_or_copy(source_file, target)
        result[name] = str(target.relative_to(root))
    return result


def _copy_export_config(
    source: Path,
    export_meta: dict[str, Any],
    root: Path,
    prefix: str,
) -> str | None:
    relative = export_meta.get("config")
    if not relative:
        return None
    source_file = source / relative
    if source_file.is_file():
        destination = root / f"{prefix}_export_config{source_file.suffix}"
        _link_or_copy(source_file, destination)
        return destination.relative_to(root).as_posix()
    return None


def _canonical_component(values: dict[str, Any]) -> dict[str, Any]:
    """Build a release-root-relative component index without staging paths."""
    runtime = deepcopy(values.get("source_meta", {}))
    for key in (
        "onnx_file",
        "hmonnx_file",
        "golden_dir",
        "prefill_onnx",
        "decode_onnx",
        "prefill_golden",
        "decode_golden",
        "hf_config",
        "host_modules",
    ):
        runtime.pop(key, None)
    calibration = runtime.get("calibration")
    if isinstance(calibration, dict):
        calibration.pop("wav_path", None)
    return {
        "component_dir": values["directory"],
        "hmonnx_file": values["with_act_onnx"],
        "external_data": values["external_data"],
        "golden_dir": values["step_dir"],
        "quant_type": values["quant_type"],
        "runtime": runtime,
    }


def _validate_release_directory(root: Path, prefix: str, components: list[ReleaseComponent]) -> None:
    for component in components:
        component_dir = root / _component_relative_dir(component.name)
        graph_stem = f"{_component_release_prefix(prefix, component.quant_type)}_{component.name}"
        onnx_file = component_dir / f"{graph_stem}_with_act.onnx"
        external_file = component_dir / f"{graph_stem}_external_data"
        step_dir = component_dir / "step_0" if component.golden_dir is not None else None
        required_paths = (onnx_file, external_file) + ((step_dir,) if step_dir is not None else ())
        for path in required_paths:
            if not path.exists():
                raise FileNotFoundError(f"Incomplete release component {component.name}: {path}")
        model = onnx.load_model(str(onnx_file), load_external_data=False)
        if _external_locations(model) != {external_file.name}:
            raise ValueError(f"Incorrect external_data reference in {onnx_file}")
        if step_dir is not None:
            for name in (onnx_file.name, external_file.name):
                link = step_dir / name
                if not link.is_symlink() or os.readlink(link) != str(Path("..") / name):
                    raise ValueError(f"Incorrect step_0 symlink: {link}")
    if not (root / "hf_config" / "config.json").is_file():
        raise FileNotFoundError("Release is missing hf_config/config.json")
    export_meta_file = root / "export_meta_info.json"
    if not export_meta_file.is_file():
        raise FileNotFoundError("Release is missing export_meta_info.json")
    export_meta = _read_json(export_meta_file)
    config_file = export_meta.get("config")
    if config_file and not (root / config_file).is_file():
        raise FileNotFoundError(f"Release config is missing: {config_file}")


def _component_relative_dir(component_name: str) -> Path:
    """Use one release-root child directory for every exported component."""
    return Path(component_name)


def _find_with_act_onnx(directory: Path) -> Path:
    matches = sorted(path for path in directory.glob("*_with_act.onnx") if path.is_file())
    if len(matches) != 1:
        raise ValueError(f"Expected one with_act ONNX under {directory}, got {len(matches)}")
    return matches[0]


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


def _quant_bit_width(value: str) -> str:
    match = re.match(r"^(w\d+a\d+)", value.lower())
    return match.group(1) if match else "wmix_amix"


def _component_release_prefix(release_prefix: str, quant_type: str) -> str:
    """Use fixed component bit widths inside an overall mixed-precision release."""
    bit_width = _quant_bit_width(quant_type)
    return re.sub(
        r"_(?:wmix_amix|w\d+a\d+)_",
        f"_{bit_width}_",
        release_prefix,
        count=1,
    )


def _format_context_length(value: int) -> str:
    value = int(value)
    if value >= 1024 and value % 1024 == 0:
        return f"{value // 1024}k"
    return str(value)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, values: dict[str, Any]) -> None:
    path.write_text(json.dumps(values, indent=4, ensure_ascii=False) + "\n", encoding="utf-8")
