from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path


KOKORO_SOURCE_COMMIT = "dfb907a02bba8152ca444717ca5d78747ccb4bec"
KOKORO_CONFIG_SHA256 = "bc333efa5ce4ceff433c8c8e5d027a1eca0166001e4e4a62bea2d26ff7a46890"
KOKORO_CHECKPOINT_SHA256 = "b1d8410fa44dfb5c15471fd6c4225ea6b4e9ac7fa03c98e8bea47a9928476e2b"
KOKORO_REFERENCE_ONNX_SHA256 = "eefec708cbc7aba8e8129b5c2f7cb92e1fe7d281af1e1dd451592d9ff0714a0d"
KOKORO_ZF001_SHA256 = "9bdc9a87e13e9bb1ea3e7803259c2ecbfebaeeb2ff80b5d0c76df1a464c1c962"


@dataclass(frozen=True)
class KokoroAssets:
    root: Path
    source_root: Path
    config: Path
    checkpoint: Path
    voice: Path
    voice_pack: Path | None


def resolve_model_assets(model_dir: str | Path) -> KokoroAssets:
    root = Path(model_dir).expanduser().resolve()
    source_root = _one_dir(
        root,
        ("source/kokoro", "kokoro", "source", "."),
        marker="kokoro/model.py",
        description="the pinned hexgrad/kokoro source checkout",
    )
    config = _one_file(
        root,
        ("pytorch/config.json", "config.json", "onnx/config.json"),
        "Kokoro v1.1-zh config.json",
    )
    checkpoint = _one_file(
        root,
        ("pytorch/kokoro-v1_1-zh.pth", "kokoro-v1_1-zh.pth"),
        "Kokoro v1.1-zh PyTorch checkpoint",
    )
    voice = _one_file(
        root,
        ("pytorch/voices/zf_001.pt", "voices/zf_001.pt", "zf_001.pt"),
        "the zf_001 reference voice",
    )
    voice_pack = _optional_file(
        root,
        ("onnx/voices-v1.1-zh.bin", "voices-v1.1-zh.bin"),
    )
    return KokoroAssets(
        root=root,
        source_root=source_root,
        config=config,
        checkpoint=checkpoint,
        voice=voice,
        voice_pack=voice_pack,
    )


def verify_release_assets(assets: KokoroAssets) -> dict[str, str | None]:
    expected = {
        "config": (assets.config, KOKORO_CONFIG_SHA256),
        "checkpoint": (assets.checkpoint, KOKORO_CHECKPOINT_SHA256),
        "voice": (assets.voice, KOKORO_ZF001_SHA256),
    }
    hashes: dict[str, str | None] = {}
    for name, (path, wanted) in expected.items():
        actual = sha256(path)
        if actual != wanted:
            raise ValueError(f"Unexpected {name} SHA256 for {path}: expected {wanted}, got {actual}")
        hashes[f"{name}_sha256"] = actual
    commit = git_commit(assets.source_root)
    if commit != KOKORO_SOURCE_COMMIT:
        raise ValueError(
            f"Unexpected Kokoro source commit: expected {KOKORO_SOURCE_COMMIT}, got {commit or 'not a git checkout'}"
        )
    hashes["source_commit"] = commit
    return hashes


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit(path: str | Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def _one_dir(
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
    expected = ", ".join(str(root / relative / marker) for relative in candidates)
    raise FileNotFoundError(f"{root} does not contain {description}; expected one of {expected}")


def _one_file(root: Path, candidates: tuple[str, ...], description: str) -> Path:
    path = _optional_file(root, candidates)
    if path is not None:
        return path
    expected = ", ".join(str(root / relative) for relative in candidates)
    raise FileNotFoundError(f"{root} does not contain {description}; expected one of {expected}")


def _optional_file(root: Path, candidates: tuple[str, ...]) -> Path | None:
    for relative in candidates:
        candidate = (root / relative).resolve()
        if candidate.is_file():
            return candidate
    return None


__all__ = [
    "KOKORO_CHECKPOINT_SHA256",
    "KOKORO_CONFIG_SHA256",
    "KOKORO_REFERENCE_ONNX_SHA256",
    "KOKORO_SOURCE_COMMIT",
    "KOKORO_ZF001_SHA256",
    "KokoroAssets",
    "resolve_model_assets",
    "sha256",
    "verify_release_assets",
]
