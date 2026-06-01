import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from input_utils import ensure_dir, resolve_path

ENCODER_CANDIDATES = (
    "encoder.onnx",
    "encoder_sim.onnx",
)
DECODER_CANDIDATES = (
    "decoder.onnx",
    "decoder_sim.onnx",
    "decoder_fix_mask.onnx",
    "decoder_fix_mask2.onnx",
    "decoder_fix_mask2_sim.onnx",
)
PREDICTOR_CANDIDATES = (
    "predictor.onnx",
    "predictor_sim.onnx",
)


def default_manifest_path() -> Path:
    return Path(__file__).with_name("exports") / "latest.json"


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return str(obj)


def _collect_paths(obj: Any, acc: List[str], key: str = "") -> None:
    if isinstance(obj, str):
        acc.append(obj)
        return
    if isinstance(obj, Path):
        acc.append(str(obj))
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            lower = str(k).lower()
            if isinstance(v, str) and v.endswith(".onnx"):
                acc.append(v)
            _collect_paths(v, acc, key=lower)
        return
    if isinstance(obj, (list, tuple)):
        for v in obj:
            _collect_paths(v, acc, key=key)


def _expand_existing(path_str: str) -> Optional[Path]:
    if not path_str:
        return None
    path = resolve_path(path_str)
    if path.exists():
        return path
    return None


def _find_in_files(files: Iterable[Path], candidates: Tuple[str, ...]) -> Optional[Path]:
    for cand in candidates:
        for fpath in files:
            if fpath.name == cand:
                return fpath
    return None


def _find_in_dirs(dirs: Iterable[Path], candidates: Tuple[str, ...]) -> Optional[Path]:
    for dpath in dirs:
        for cand in candidates:
            candidate = dpath / cand
            if candidate.exists():
                return candidate
    return None


def write_export_manifest(result: Any, manifest_path: Path, export_dir_hint: str = "") -> Dict[str, str]:
    raw_paths: List[str] = []
    _collect_paths(result, raw_paths)
    if export_dir_hint:
        raw_paths.append(export_dir_hint)

    files: List[Path] = []
    dirs: List[Path] = []
    for raw in raw_paths:
        path = _expand_existing(raw)
        if path is None:
            continue
        if path.is_dir():
            dirs.append(path)
        elif path.is_file():
            files.append(path)
            dirs.append(path.parent)

    encoder = _find_in_files(files, ENCODER_CANDIDATES) or _find_in_dirs(dirs, ENCODER_CANDIDATES)
    decoder = _find_in_files(files, DECODER_CANDIDATES) or _find_in_dirs(dirs, DECODER_CANDIDATES)
    predictor = _find_in_files(files, PREDICTOR_CANDIDATES) or _find_in_dirs(dirs, PREDICTOR_CANDIDATES)

    export_dir = None
    for dpath in dirs:
        if encoder and encoder.parent == dpath:
            export_dir = dpath
            break
        if decoder and decoder.parent == dpath:
            export_dir = dpath
            break
        if predictor and predictor.parent == dpath:
            export_dir = dpath
            break
    if export_dir is None and dirs:
        export_dir = dirs[0]

    manifest = {
        "export_dir": str(export_dir) if export_dir else "",
        "encoder_onnx": str(encoder) if encoder else "",
        "decoder_onnx": str(decoder) if decoder else "",
        "predictor_onnx": str(predictor) if predictor else "",
        "raw_result": _json_safe(result),
    }

    ensure_dir(manifest_path.parent)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True))
    return manifest


def load_manifest(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"invalid manifest: {path}")
    return data


def resolve_manifest(path_str: str = "") -> Optional[Path]:
    if path_str:
        return resolve_path(path_str)
    default = default_manifest_path()
    return default if default.exists() else None


def _resolve_from_dir(dir_path: Path, candidates: Tuple[str, ...]) -> Optional[Path]:
    for cand in candidates:
        path = dir_path / cand
        if path.exists():
            return path
    return None


def resolve_encoder_onnx(onnx_arg: str, model_dir: str, manifest_path: Optional[Path]) -> Path:
    if onnx_arg:
        return resolve_path(onnx_arg)
    if model_dir:
        resolved = _resolve_from_dir(resolve_path(model_dir), ENCODER_CANDIDATES)
        if resolved:
            return resolved
    if manifest_path and manifest_path.exists():
        manifest = load_manifest(manifest_path)
        if manifest.get("encoder_onnx"):
            return resolve_path(manifest["encoder_onnx"])
        if manifest.get("export_dir"):
            resolved = _resolve_from_dir(resolve_path(manifest["export_dir"]), ENCODER_CANDIDATES)
            if resolved:
                return resolved
    raise FileNotFoundError("encoder onnx not found; run funasr_export.py first or pass --onnx/--model-dir")


def resolve_decoder_onnx(onnx_arg: str, model_dir: str, manifest_path: Optional[Path]) -> Path:
    if onnx_arg:
        return resolve_path(onnx_arg)
    if model_dir:
        resolved = _resolve_from_dir(resolve_path(model_dir), DECODER_CANDIDATES)
        if resolved:
            return resolved
    if manifest_path and manifest_path.exists():
        manifest = load_manifest(manifest_path)
        if manifest.get("decoder_onnx"):
            return resolve_path(manifest["decoder_onnx"])
        if manifest.get("export_dir"):
            resolved = _resolve_from_dir(resolve_path(manifest["export_dir"]), DECODER_CANDIDATES)
            if resolved:
                return resolved
    raise FileNotFoundError("decoder onnx not found; run funasr_export.py first or pass --onnx/--model-dir")


def resolve_predictor_onnx(onnx_arg: str, model_dir: str, manifest_path: Optional[Path]) -> Path:
    if onnx_arg:
        return resolve_path(onnx_arg)
    if model_dir:
        resolved = _resolve_from_dir(resolve_path(model_dir), PREDICTOR_CANDIDATES)
        if resolved:
            return resolved
    if manifest_path and manifest_path.exists():
        manifest = load_manifest(manifest_path)
        if manifest.get("predictor_onnx"):
            return resolve_path(manifest["predictor_onnx"])
        if manifest.get("export_dir"):
            resolved = _resolve_from_dir(resolve_path(manifest["export_dir"]), PREDICTOR_CANDIDATES)
            if resolved:
                return resolved
    raise FileNotFoundError("predictor onnx not found; run funasr_export.py first or pass --predictor/--model-dir")


def default_hmonnx_path(onnx_path: Path, quant_type: str) -> Path:
    return Path("work_dirs") / onnx_path.stem / "hmonnx" / f"{onnx_path.stem}_{quant_type}_XH2a.onnx"
