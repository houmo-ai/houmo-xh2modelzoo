"""Environment manager — handles transformers version switching."""

from __future__ import annotations

import logging
import os
from pathlib import Path
import re
import subprocess
import sys

logger = logging.getLogger(__name__)

_SEMVER_RE = re.compile(r"(\d+\.\d+\.\d+)")


def _normalize_transformers_version(version: str) -> str:
    version = (version or "").strip()
    match = _SEMVER_RE.search(version)
    if match:
        return match.group(1)
    return version


def transformers_version_matches(installed_version: str, target_version: str) -> bool:
    if not installed_version or not target_version:
        return False
    return _normalize_transformers_version(installed_version) == _normalize_transformers_version(target_version)


def _candidate_python_executables() -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set()

    def _add(path_str: str | None) -> None:
        if not path_str:
            return
        resolved = str(Path(path_str).expanduser().resolve())
        if resolved in seen:
            return
        if not Path(resolved).is_file():
            return
        seen.add(resolved)
        candidates.append(resolved)

    _add(sys.executable)

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        _add(str(Path(conda_prefix) / "bin" / "python"))

    try:
        result = subprocess.run(
            ["conda", "env", "list", "--json"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if result.returncode == 0 and result.stdout:
            import json

            data = json.loads(result.stdout)
            for prefix in data.get("envs", []):
                _add(str(Path(prefix) / "bin" / "python"))
    except Exception:
        logger.debug("Failed to enumerate conda envs for transformers lookup", exc_info=True)

    return candidates


def get_transformers_version_for_python(python_executable: str) -> str:
    try:
        result = subprocess.run(
            [
                python_executable,
                "-c",
                "import transformers; print(transformers.__version__)",
            ],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except Exception:
        return ""

    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def python_can_import_xhquant_hsum(python_executable: str) -> bool:
    try:
        result = subprocess.run(
            [
                python_executable,
                "-c",
                "from xhquant.lib import hsum; print(1)",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except Exception:
        return False

    return result.returncode == 0 and result.stdout.strip().endswith("1")


def _python_can_import_evalscope(python_executable: str) -> bool:
    try:
        result = subprocess.run(
            [python_executable, "-c", "import evalscope; print(1)"],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except Exception:
        return False
    return result.returncode == 0 and result.stdout.strip().endswith("1")


def find_python_with_transformers_version(
    target_version: str,
    require_xhquant_hsum: bool = False,
    require_evalscope: bool = False,
) -> tuple[str | None, str]:
    for python_executable in _candidate_python_executables():
        version = get_transformers_version_for_python(python_executable)
        if not transformers_version_matches(version, target_version):
            continue
        if require_xhquant_hsum and not python_can_import_xhquant_hsum(python_executable):
            continue
        if require_evalscope and not _python_can_import_evalscope(python_executable):
            continue
        return python_executable, version
    return None, ""


def python_has_mistral_reasoning_effort(python_executable: str) -> bool:
    try:
        result = subprocess.run(
            [
                python_executable,
                "-c",
                (
                    "from mistral_common.protocol.instruct import request; "
                    "print(int(hasattr(request, 'ReasoningEffort')))"
                ),
            ],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except Exception:
        return False

    return result.returncode == 0 and result.stdout.strip() == "1"


def ensure_qwen3_5_processor_deps(python_executable: str) -> tuple[bool, str]:
    if python_has_mistral_reasoning_effort(python_executable):
        return True, "mistral_common already provides ReasoningEffort"

    cmd = [
        python_executable,
        "-m",
        "pip",
        "install",
        "mistral_common>=1.11.0",
        "--quiet",
        "--no-warn-conflicts",
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, "pip install mistral_common timed out after 300s"
    except Exception as exc:
        return False, f"pip install mistral_common error: {exc}"

    if result.returncode != 0:
        return False, f"pip install mistral_common failed (rc={result.returncode}): {result.stderr.strip()}"

    if python_has_mistral_reasoning_effort(python_executable):
        return True, "Successfully upgraded mistral_common for Qwen3.5 AutoProcessor"

    return False, "mistral_common upgraded but ReasoningEffort is still unavailable"


def get_current_transformers_version() -> str:
    """Return the currently installed transformers version string."""
    try:
        import transformers
        return transformers.__version__
    except ImportError:
        return ""


def switch_transformers_version(target_version: str) -> tuple[bool, str]:
    """Switch transformers to the target version via pip install.

    Returns (success, message).
    """
    current = get_current_transformers_version()
    if transformers_version_matches(current, target_version):
        return True, f"transformers already at {target_version}"

    logger.info("Switching transformers: %s -> %s", current, target_version)
    cmd = [
        sys.executable, "-m", "pip", "install",
        f"transformers=={target_version}",
        "--quiet", "--no-warn-conflicts",
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode == 0:
            msg = f"Successfully switched transformers to {target_version}"
            logger.info(msg)
            return True, msg
        else:
            msg = f"pip install failed (rc={result.returncode}): {result.stderr.strip()}"
            logger.error(msg)
            return False, msg
    except subprocess.TimeoutExpired:
        msg = "pip install timed out after 300s"
        logger.error(msg)
        return False, msg
    except Exception as e:
        msg = f"pip install error: {e}"
        logger.error(msg)
        return False, msg


def check_gpu_availability() -> list[dict]:
    """Query GPU status via nvidia-smi. Returns list of GPU info dicts."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return []
        gpus = []
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 5:
                gpus.append({
                    "index": int(parts[0]),
                    "name": parts[1],
                    "used_mb": int(parts[2]),
                    "total_mb": int(parts[3]),
                    "util_pct": int(parts[4]),
                    "free_mb": int(parts[3]) - int(parts[2]),
                })
        return gpus
    except Exception:
        return []


def select_free_gpu(max_used_mb: int = 2048, max_util: int = 20) -> int | None:
    """Select the freest GPU below usage thresholds. Returns GPU index or None."""
    gpus = check_gpu_availability()
    if not gpus:
        return None
    free_candidates = [
        g for g in gpus if g["used_mb"] <= max_used_mb and g["util_pct"] <= max_util
    ]
    if free_candidates:
        best = max(free_candidates, key=lambda g: g["free_mb"])
        return best["index"]
    # Fallback: pick the freest GPU regardless
    best = max(gpus, key=lambda g: g["free_mb"])
    return best["index"]
