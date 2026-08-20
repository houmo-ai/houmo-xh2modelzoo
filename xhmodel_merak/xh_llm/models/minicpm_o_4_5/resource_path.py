from __future__ import annotations

import os
from pathlib import Path


_REPO_RESOURCE_PREFIXES = ("xh2modelzoo://", "repo://")


def resolve_repo_resource(value: str, *, description: str) -> Path:
    """Resolve a repository resource independently of the caller's cwd.

    ``xh2modelzoo://`` is the stable form used in shipped YAML files. Plain
    absolute paths remain supported, and plain relative paths are resolved
    against the source checkout before falling back to the caller's cwd.
    ``XH2MODELZOO_ROOT`` can be set when the Python package is installed away
    from the repository; ``XH2MODELZOO_DATA_ROOT`` is also accepted for a
    checkout whose data directory is mounted separately.
    """
    raw = os.path.expandvars(str(value)).strip()
    if not raw:
        raise FileNotFoundError(f"{description} path is empty")

    path = Path(raw).expanduser()
    if path.is_absolute():
        candidates = [path]
    else:
        relative = raw
        for prefix in _REPO_RESOURCE_PREFIXES:
            if relative.startswith(prefix):
                relative = relative.removeprefix(prefix)
                break
        relative = relative.lstrip("/")
        candidates = _repo_resource_candidates(relative)

    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved.is_file():
            return resolved

    searched = ", ".join(str(candidate.expanduser()) for candidate in candidates)
    raise FileNotFoundError(
        f"{description} not found: {raw!r}; searched {searched}. "
        "Use an absolute path, xh2modelzoo://<repo-relative-path>, "
        "or set XH2MODELZOO_ROOT to the repository root."
    )


def _repo_resource_candidates(relative_path: str) -> list[Path]:
    candidates: list[Path] = []
    data_root = os.environ.get("XH2MODELZOO_DATA_ROOT")
    if data_root:
        root = Path(data_root).expanduser()
        candidates.append(root / relative_path)
        if relative_path.startswith("data/"):
            candidates.append(root / relative_path.removeprefix("data/"))

    env_root = os.environ.get("XH2MODELZOO_ROOT")
    if env_root:
        candidates.append(Path(env_root).expanduser() / relative_path)

    candidates.append(_repo_root() / relative_path)
    candidates.append(Path.cwd() / relative_path)
    return candidates


def _repo_root() -> Path:
    env_root = os.environ.get("XH2MODELZOO_ROOT")
    if env_root:
        root = Path(env_root).expanduser()
        if root.is_dir():
            return root.resolve()
    for parent in Path(__file__).resolve().parents:
        if (parent / "configs_merak").is_dir() and (parent / "xhmodel_merak").is_dir():
            return parent
    return Path(__file__).resolve().parents[4]


__all__ = ["resolve_repo_resource"]
