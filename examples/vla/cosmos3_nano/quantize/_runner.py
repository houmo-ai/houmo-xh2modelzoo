# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class QuantStage:
    name: str
    command: list[str]
    expected: tuple[Path, ...] = ()
    reports: tuple[Path, ...] = ()
    optional: bool = False
    note: str = ""


@dataclass
class StageResult:
    name: str
    status: str
    command: list[str]
    expected: list[str]
    reports: list[str]
    elapsed_sec: float = 0.0
    returncode: int | None = None
    log: str | None = None
    note: str = ""


@dataclass
class QuantRunConfig:
    dry_run: bool = False
    force: bool = False
    continue_on_error: bool = False
    env: dict[str, str] = field(default_factory=dict)
    log_dir: Path | None = None


def cosmos3_root_from_file(path: Path) -> Path:
    current = path.resolve()
    for parent in current.parents:
        if parent.name == "cosmos3_nano":
            return parent
    raise RuntimeError(f"Could not find cosmos3_nano root from {path}")


def all_exist(paths: tuple[Path, ...]) -> bool:
    return bool(paths) and all(path.exists() for path in paths)


def command_to_string(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def run_stages(stages: list[QuantStage], config: QuantRunConfig) -> list[StageResult]:
    results: list[StageResult] = []
    env = os.environ.copy()
    env.update(config.env)
    if config.log_dir is not None:
        config.log_dir.mkdir(parents=True, exist_ok=True)

    for stage in stages:
        expected = [str(path) for path in stage.expected]
        reports = [str(path) for path in stage.reports]
        if all_exist(stage.expected) and not config.force:
            results.append(
                StageResult(
                    name=stage.name,
                    status="skipped",
                    command=stage.command,
                    expected=expected,
                    reports=reports,
                    note=stage.note,
                )
            )
            print(f"[skip] {stage.name}")
            continue

        print(f"[run] {stage.name}")
        print(command_to_string(stage.command))
        if config.dry_run:
            results.append(
                StageResult(
                    name=stage.name,
                    status="dry_run",
                    command=stage.command,
                    expected=expected,
                    reports=reports,
                    note=stage.note,
                )
            )
            continue

        started = time.monotonic()
        log_path = config.log_dir / f"{stage.name}.log" if config.log_dir is not None else None
        stdout_target: Any = None
        try:
            if log_path is not None:
                log_path.parent.mkdir(parents=True, exist_ok=True)
                with log_path.open("w", encoding="utf-8") as log_file:
                    process = subprocess.run(
                        stage.command,
                        cwd=str(cosmos3_root_from_file(Path(__file__)).parents[3]),
                        env=env,
                        stdout=log_file,
                        stderr=subprocess.STDOUT,
                        text=True,
                        check=False,
                    )
            else:
                process = subprocess.run(
                    stage.command,
                    cwd=str(cosmos3_root_from_file(Path(__file__)).parents[3]),
                    env=env,
                    stdout=stdout_target,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )
            elapsed = time.monotonic() - started
            status = "done" if process.returncode == 0 else "failed"
            result = StageResult(
                name=stage.name,
                status=status,
                command=stage.command,
                expected=expected,
                reports=reports,
                elapsed_sec=elapsed,
                returncode=process.returncode,
                log=str(log_path) if log_path else None,
                note=stage.note,
            )
            results.append(result)
            if process.returncode != 0 and not (config.continue_on_error or stage.optional):
                raise RuntimeError(f"Stage {stage.name!r} failed with code {process.returncode}; log={log_path}")
        except Exception:
            if not (config.continue_on_error or stage.optional):
                raise
            elapsed = time.monotonic() - started
            results.append(
                StageResult(
                    name=stage.name,
                    status="failed",
                    command=stage.command,
                    expected=expected,
                    reports=reports,
                    elapsed_sec=elapsed,
                    log=str(log_path) if log_path else None,
                    note=stage.note,
                )
            )
    return results


def write_summary(path: Path, payload: dict[str, Any], results: list[StageResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        **payload,
        "stages": [
            {
                "name": item.name,
                "status": item.status,
                "command": item.command,
                "expected": item.expected,
                "reports": item.reports,
                "elapsed_sec": item.elapsed_sec,
                "returncode": item.returncode,
                "log": item.log,
                "note": item.note,
            }
            for item in results
        ],
    }
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_markdown_summary(path: Path, title: str, results: list[StageResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", "", "| Stage | Status | Log | Note |", "| --- | --- | --- | --- |"]
    for item in results:
        log = item.log or ""
        lines.append(f"| {item.name} | {item.status} | `{log}` | {item.note} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
