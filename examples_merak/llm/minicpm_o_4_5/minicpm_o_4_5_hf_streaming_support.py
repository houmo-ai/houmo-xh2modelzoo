from __future__ import annotations

import json
import platform
import sys
import traceback
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict


class FailurePacket(TypedDict):
    status: str
    timestamp: str
    environment_versions: Mapping[str, str]
    case: str
    command: list[str]
    asset_paths: list[str]
    api_stage: str
    exception_type: str
    exception_message: str
    traceback: str
    last_successful_stage: str
    output_paths: list[str]


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def environment_versions() -> dict[str, str]:
    import librosa
    import torch
    import transformers

    return {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "librosa": librosa.__version__,
        "cuda_available": str(torch.cuda.is_available()),
        "cuda_runtime": str(torch.version.cuda),
    }


def make_failure_packet(
    *,
    case_name: str,
    command: Sequence[str],
    asset_paths: Sequence[str],
    api_stage: str,
    error: BaseException,
    last_successful_stage: str,
    output_paths: Sequence[Path],
    environment: Mapping[str, str],
    timestamp: str,
) -> FailurePacket:
    return {
        "status": "failed",
        "timestamp": timestamp,
        "environment_versions": dict(environment),
        "case": case_name,
        "command": list(command),
        "asset_paths": list(asset_paths),
        "api_stage": api_stage,
        "exception_type": type(error).__name__,
        "exception_message": str(error),
        "traceback": "".join(traceback.format_exception(error)),
        "last_successful_stage": last_successful_stage,
        "output_paths": [str(path) for path in output_paths],
    }


def write_json(path: Path, payload: Mapping[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def write_failure_packet(output_dir: Path, packet: FailurePacket) -> Path:
    return write_json(output_dir / "failure_packet.json", packet)
