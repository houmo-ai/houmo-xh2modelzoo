from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

from xhmodel_merak.xh_other_model.models.kokoro.assets import sha256
from xhmodel_merak.xh_other_model.models.kokoro.bucketed_runtime import (
    BUCKETED_PRECISION_SPLIT_GRAPH_MODE,
)
from xhmodel_merak.xh_other_model.models.kokoro.buckets import (
    AUDIO_BUCKET_SECONDS,
    BUCKET_ROUTES,
    FRAME_BUCKETS,
    LSTM_VARIANTS,
    TOKEN_BUCKETS,
    frame_bucket_key,
    token_bucket_key,
)
from xhmodel_merak.xh_other_model.models.kokoro.independent_split import (
    FRAME_ACOUSTIC_ROLE,
    GENERATOR_ISTFT_ROLE,
    PHASE_CORE_ROLE,
    TEXT_DURATION_ROLE,
)
from xhmodel_merak.xh_other_model.models.kokoro.workflow import (
    _export_runtime_assets,
    _hmonnx_artifact_metadata,
    _validate_lstm_variant,
)


FINAL_COMPONENT_ORDER = (
    TEXT_DURATION_ROLE,
    FRAME_ACOUSTIC_ROLE,
    PHASE_CORE_ROLE,
    GENERATOR_ISTFT_ROLE,
)
NPU_COMPONENT_ORDER = (
    TEXT_DURATION_ROLE,
    FRAME_ACOUSTIC_ROLE,
    GENERATOR_ISTFT_ROLE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge Kokoro paired-route bucket shards into one routed artifact",
    )
    parser.add_argument("--shard-dir", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--copy", action="store_true", help="copy files instead of using same-filesystem hard links")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    shard_dirs = tuple(Path(value).expanduser().resolve() for value in args.shard_dir)
    config_path = Path(args.config_path).expanduser().resolve()
    metadata = merge_bucket_exports(
        shard_dirs,
        output_dir,
        config_path=config_path,
        hardlink=not args.copy,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "token_buckets": len(metadata["bucket_presets"]["token"]),
                "frame_buckets": len(metadata["bucket_presets"]["frame"]),
                "decomposed_failures": len(metadata["decomposed_failures"]),
                "metadata": str(output_dir / "export_meta_info.json"),
            },
            indent=2,
        )
    )


def merge_bucket_exports(
    shard_dirs: tuple[Path, ...],
    output_dir: Path,
    *,
    config_path: Path,
    hardlink: bool,
    overwrite: bool,
) -> dict[str, Any]:
    if not shard_dirs:
        raise ValueError("at least one shard directory is required")
    if not config_path.is_file():
        raise FileNotFoundError(f"missing workflow config: {config_path}")
    _prepare_output_dir(output_dir, overwrite=overwrite)

    shard_meta = [_load_shard(path) for path in shard_dirs]
    _validate_common_identity(shard_meta)
    for shard_dir, meta in zip(shard_dirs, shard_meta, strict=True):
        _install_artifact_tree(
            shard_dir,
            output_dir,
            hardlink=hardlink,
            include_hmonnx=bool(meta.get("hmonnx_converted")),
        )
    config_destination = output_dir / config_path.name
    shutil.copy2(config_path, config_destination)

    merged = copy.deepcopy(shard_meta[0])
    merged["create_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    merged["workflow_config"] = config_destination.name
    merged["runtime_assets"] = _merge_runtime_assets(shard_meta, output_dir)
    optimize_modes = {
        bool(meta.get("torch_onnx_internal_optimize", True)) for meta in shard_meta
    }
    if len(optimize_modes) != 1:
        raise ValueError("torch.onnx internal optimize mode differs across Kokoro shards")
    merged["torch_onnx_internal_optimize"] = optimize_modes.pop()
    merged["bucket_presets"] = {
        "policy": "duration_driven_paired_t_f",
        "routes": [route.as_dict() for route in BUCKET_ROUTES],
        "token": list(TOKEN_BUCKETS),
        "audio_seconds": list(AUDIO_BUCKET_SECONDS),
        "frame": list(FRAME_BUCKETS),
        "frame_to_audio_seconds": {
            str(frame): seconds for frame, seconds in zip(FRAME_BUCKETS, AUDIO_BUCKET_SECONDS, strict=True)
        },
        "export_count_per_lstm_variant": len(TOKEN_BUCKETS) + len(FRAME_BUCKETS),
        "cartesian_product_exported": False,
    }
    merged["components"] = _merge_components(shard_meta)
    merged["component_order"] = list(FINAL_COMPONENT_ORDER)
    merged["npu_component_order"] = list(NPU_COMPONENT_ORDER)
    merged["text_validation"] = _merge_bucket_mapping(shard_meta, "text_validation")
    merged["frame_validation"] = _merge_bucket_mapping(shard_meta, "frame_validation")
    merged["decomposed_failures"] = _merge_failures(shard_meta)
    merged["export_assembly"] = {
        "mode": "validated_paired_t_f_shards",
        "file_install": "hardlink" if hardlink else "copy",
        "shards": [str(path) for path in shard_dirs],
    }
    _refresh_and_validate_hmonnx_metadata(merged, output_dir)
    _validate_installed_artifacts(merged, output_dir)

    meta_path = output_dir / "export_meta_info.json"
    temporary = meta_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(merged, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, meta_path)
    return merged


def _prepare_output_dir(output_dir: Path, *, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"output directory exists: {output_dir}")
        cwd = Path.cwd().resolve()
        if output_dir in {Path("/").resolve(), cwd, cwd.parent}:
            raise ValueError(f"refusing to remove unsafe output directory: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)


def _load_shard(shard_dir: Path) -> dict[str, Any]:
    meta_path = shard_dir / "export_meta_info.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"shard has no export metadata: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("graph_mode") != BUCKETED_PRECISION_SPLIT_GRAPH_MODE:
        raise ValueError(f"not a bucketed Kokoro shard: {shard_dir}")
    if meta.get("bucket_presets", {}).get("policy") != "duration_driven_paired_t_f":
        raise ValueError(f"not a paired T/F Kokoro shard: {shard_dir}")
    if meta.get("phase_boundary", {}).get("mode") != "host_fp32":
        raise ValueError(f"shard does not use the final CPU FP32 phase boundary: {shard_dir}")
    if tuple(meta.get("component_order", ())) != FINAL_COMPONENT_ORDER:
        raise ValueError(f"shard has unexpected component order: {shard_dir}")
    return meta


def _validate_common_identity(shard_meta: list[dict[str, Any]]) -> None:
    keys = (
        "model_type",
        "asset_identity",
        "target_device",
        "graph_mode",
        "sample_rate",
        "samples_per_frame",
        "seed",
        "stft_pad_mode",
        "stft_phase_mode",
        "f0_norm_mode",
        "mask_contract",
        "lstm_contract",
        "phase_boundary",
        "source_level_compatibility_lowerings",
        "hmonnx_converted",
        "hmonnx_validated",
    )
    reference = shard_meta[0]
    for index, meta in enumerate(shard_meta[1:], start=1):
        mismatched = [key for key in keys if meta.get(key) != reference.get(key)]
        if mismatched:
            raise ValueError(f"shard {index} identity differs in {mismatched}")


def _install_artifact_tree(
    shard_dir: Path,
    output_dir: Path,
    *,
    hardlink: bool,
    include_hmonnx: bool,
) -> None:
    tree_names = ["onnx"]
    if include_hmonnx:
        tree_names.append("hmonnx")
    if (shard_dir / "assets").is_dir():
        tree_names.append("assets")
    for tree_name in tree_names:
        source_root = shard_dir / tree_name
        if not source_root.is_dir():
            raise FileNotFoundError(f"missing artifact tree: {source_root}")
        for source in source_root.rglob("*"):
            if not source.is_file():
                continue
            relative = source.relative_to(shard_dir)
            destination = output_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                if destination.stat().st_size != source.stat().st_size or sha256(destination) != sha256(source):
                    raise RuntimeError(f"conflicting shard artifact: {relative}")
                continue
            if hardlink:
                os.link(source, destination)
            else:
                shutil.copy2(source, destination)


def _merge_runtime_assets(
    shard_meta: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    available = [meta.get("runtime_assets") for meta in shard_meta if meta.get("runtime_assets")]
    if available:
        reference = available[0]
        if any(value != reference for value in available[1:]):
            raise ValueError("runtime asset metadata differs across Kokoro shards")
        relative = reference.get("voice_pack", {}).get("file")
        if relative and (output_dir / relative).is_file():
            return copy.deepcopy(reference)

    voice_path = Path(shard_meta[0]["source_paths"]["voice"])
    if not voice_path.is_file():
        raise FileNotFoundError(
            "shards have no embedded voice pack and the source voice is unavailable: "
            f"{voice_path}"
        )
    return _export_runtime_assets(voice_path, output_dir)


def _merge_components(shard_meta: list[dict[str, Any]]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for role in FINAL_COMPONENT_ORDER:
        reference = copy.deepcopy(shard_meta[0]["components"][role])
        reference["buckets"] = {}
        expected_common = {
            key: value for key, value in shard_meta[0]["components"][role].items() if key != "buckets"
        }
        for meta in shard_meta:
            component = meta["components"][role]
            common = {key: value for key, value in component.items() if key != "buckets"}
            if common != expected_common:
                raise ValueError(f"component contract differs across shards for {role}")
            for key, value in component["buckets"].items():
                previous = reference["buckets"].get(key)
                if previous is not None and _stable_metadata(previous) != _stable_metadata(value):
                    raise ValueError(f"conflicting {role} bucket across shards: {key}")
                if previous is None:
                    reference["buckets"][key] = copy.deepcopy(value)
        merged[role] = reference
    return merged


def _merge_bucket_mapping(shard_meta: list[dict[str, Any]], field: str) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for meta in shard_meta:
        for key, value in meta.get(field, {}).items():
            previous = merged.get(key)
            if previous is not None and _stable_metadata(previous) != _stable_metadata(value):
                raise ValueError(f"conflicting {field} bucket across shards: {key}")
            if previous is None:
                merged[key] = copy.deepcopy(value)
    return merged


def _stable_metadata(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _stable_metadata(item)
            for key, item in value.items()
            if key not in {"conversion_seconds", "lowering_seconds"}
        }
    if isinstance(value, list):
        return [_stable_metadata(item) for item in value]
    return value


def _merge_failures(shard_meta: list[dict[str, Any]]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    seen: set[str] = set()
    for meta in shard_meta:
        for failure in meta.get("decomposed_failures", []):
            identity = json.dumps(failure, sort_keys=True, ensure_ascii=False)
            if identity not in seen:
                seen.add(identity)
                failures.append(copy.deepcopy(failure))
    return failures


def _refresh_and_validate_hmonnx_metadata(meta: dict[str, Any], output_dir: Path) -> None:
    if not meta.get("hmonnx_converted"):
        return
    for role, expected_nodes in ((TEXT_DURATION_ROLE, 10), (FRAME_ACOUSTIC_ROLE, 2)):
        for bucket in meta["components"][role]["buckets"].values():
            for variant in LSTM_VARIANTS:
                entry = bucket["hmonnx_variants"][variant]
                if entry["status"] != "ok":
                    continue
                path = output_dir / entry["hmonnx_file"]
                entry.update(_hmonnx_artifact_metadata(path, output_dir))
                _validate_lstm_variant(entry, variant, expected_native_lstm_nodes=expected_nodes)
    for bucket in meta["components"][GENERATOR_ISTFT_ROLE]["buckets"].values():
        path = output_dir / bucket["hmonnx_file"]
        bucket.update(_hmonnx_artifact_metadata(path, output_dir))


def _validate_installed_artifacts(meta: dict[str, Any], output_dir: Path) -> None:
    voice = meta.get("runtime_assets", {}).get("voice_pack", {})
    voice_path = output_dir / str(voice.get("file", ""))
    if not voice.get("file") or not voice_path.is_file():
        raise FileNotFoundError(f"missing installed NumPy voice pack: {voice_path}")
    if voice.get("sha256") != sha256(voice_path):
        raise RuntimeError(f"NumPy voice pack SHA256 mismatch: {voice_path}")
    components = meta["components"]
    expected_keys = {
        TEXT_DURATION_ROLE: {token_bucket_key(value) for value in TOKEN_BUCKETS},
        FRAME_ACOUSTIC_ROLE: {frame_bucket_key(value) for value in FRAME_BUCKETS},
        PHASE_CORE_ROLE: {frame_bucket_key(value) for value in FRAME_BUCKETS},
        GENERATOR_ISTFT_ROLE: {frame_bucket_key(value) for value in FRAME_BUCKETS},
    }
    for role, keys in expected_keys.items():
        actual = set(components[role]["buckets"])
        if actual != keys:
            raise RuntimeError(
                f"{role} buckets differ from preset; missing={sorted(keys - actual)}, extra={sorted(actual - keys)}"
            )
        for bucket in components[role]["buckets"].values():
            for field in ("onnx_file", "reference_onnx_file"):
                relative = bucket.get(field)
                if relative and not (output_dir / relative).is_file():
                    raise FileNotFoundError(f"missing installed {field}: {output_dir / relative}")
    if meta.get("hmonnx_converted"):
        for role in (TEXT_DURATION_ROLE, FRAME_ACOUSTIC_ROLE):
            for bucket in components[role]["buckets"].values():
                for variant in bucket["hmonnx_variants"].values():
                    if variant["status"] == "ok" and not (output_dir / variant["hmonnx_file"]).is_file():
                        raise FileNotFoundError(f"missing installed HMONNX: {output_dir / variant['hmonnx_file']}")
        for bucket in components[GENERATOR_ISTFT_ROLE]["buckets"].values():
            if not (output_dir / bucket["hmonnx_file"]).is_file():
                raise FileNotFoundError(f"missing installed HMONNX: {output_dir / bucket['hmonnx_file']}")


if __name__ == "__main__":
    main()
