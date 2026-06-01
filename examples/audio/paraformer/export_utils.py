import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from input_utils import ensure_dir, resolve_path

ENCODER_CANDIDATES = (
    "model.onnx",
    "encoder.onnx",
    "encoder_sim.onnx",
)
DECODER_CANDIDATES = (
    "decoder.onnx",
    "decoder_sim.onnx",
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


def _collect_paths(obj: Any, acc: List[str]) -> None:
    if isinstance(obj, str):
        acc.append(obj)
        return
    if isinstance(obj, Path):
        acc.append(str(obj))
        return
    if isinstance(obj, dict):
        for value in obj.values():
            _collect_paths(value, acc)
        return
    if isinstance(obj, (list, tuple)):
        for value in obj:
            _collect_paths(value, acc)


def _expand_existing(path_str: str) -> Optional[Path]:
    if not path_str:
        return None
    path = resolve_path(path_str)
    if path.exists():
        return path
    return None


def _find_in_files(files: Iterable[Path], candidates: Tuple[str, ...]) -> Optional[Path]:
    for candidate in candidates:
        for path in files:
            if path.name == candidate:
                return path
    return None


def _find_in_dirs(dirs: Iterable[Path], candidates: Tuple[str, ...]) -> Optional[Path]:
    for directory in dirs:
        for candidate in candidates:
            path = directory / candidate
            if path.exists():
                return path
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

    export_dir = None
    for directory in dirs:
        if encoder and encoder.parent == directory:
            export_dir = directory
            break
        if decoder and decoder.parent == directory:
            export_dir = directory
            break
    if export_dir is None and dirs:
        export_dir = dirs[0]

    manifest = {
        "export_dir": str(export_dir) if export_dir else "",
        "encoder_onnx": str(encoder) if encoder else "",
        "decoder_onnx": str(decoder) if decoder else "",
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
    for candidate in candidates:
        path = dir_path / candidate
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
    raise FileNotFoundError("encoder onnx not found; run paraformer_export.py first or pass --onnx/--model-dir")


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
    raise FileNotFoundError("decoder onnx not found; run paraformer_export.py first or pass --onnx/--model-dir")


def default_hmonnx_path(onnx_path: Path, quant_type: str) -> Path:
    return Path("work_dirs") / onnx_path.stem / "hmonnx" / f"{onnx_path.stem}_{quant_type}_XH2a.onnx"