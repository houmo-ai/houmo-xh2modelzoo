from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MeloAssets:
    root: Path
    source_root: Path
    config: Path
    checkpoint: Path
    release_package: Path


def resolve_model_assets(model_dir: str | Path) -> MeloAssets:
    root = Path(model_dir).expanduser().resolve()
    source_root = _one(
        root,
        (
            "source/MeloTTS",
            "MeloTTS",
            "source",
            ".",
        ),
        marker="melo/models.py",
        description="MeloTTS source checkout",
    )
    config = _one_file(
        root,
        (
            "melotts_weights/config.json",
            "weights/config.json",
            "config.json",
        ),
        "MeloTTS config.json",
    )
    checkpoint = _one_file(
        root,
        (
            "melotts_weights/checkpoint.pth",
            "weights/checkpoint.pth",
            "checkpoint.pth",
        ),
        "MeloTTS checkpoint.pth",
    )
    release_package = _one(
        root,
        (
            "packages/vits-melo-tts-zh_en",
            "vits-melo-tts-zh_en",
            "release",
            ".",
        ),
        marker="model.onnx",
        description="sherpa-onnx vits-melo-tts-zh_en package",
    )
    return MeloAssets(
        root=root,
        source_root=source_root,
        config=config,
        checkpoint=checkpoint,
        release_package=release_package,
    )


def _one(
    root: Path,
    candidates: tuple[str, ...],
    *,
    marker: str,
    description: str,
) -> Path:
    for relative in candidates:
        candidate = (root / relative).resolve()
        if (candidate / marker).is_file():
            return candidate
    raise FileNotFoundError(
        f"{root} does not contain {description}; expected one of "
        + ", ".join(str(root / relative / marker) for relative in candidates)
    )


def _one_file(root: Path, candidates: tuple[str, ...], description: str) -> Path:
    for relative in candidates:
        candidate = (root / relative).resolve()
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"{root} does not contain {description}; expected one of "
        + ", ".join(str(root / relative) for relative in candidates)
    )


__all__ = ["MeloAssets", "resolve_model_assets"]
