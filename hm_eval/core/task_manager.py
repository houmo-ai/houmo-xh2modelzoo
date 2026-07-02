"""Task manager — handles background evaluation tasks with process isolation."""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_HM_EVAL_ROOT = Path(__file__).resolve().parent.parent
_TASKS_DIR = _HM_EVAL_ROOT / "tasks"


class TaskStatus(str, Enum):
    PENDING = "pending"
    ENV_SETUP = "env_setup"
    EXPORTING = "exporting"
    RUNNING = "running"
    COMPLETED = "completed"
    ABNORMAL = "abnormal"
    FAILED = "failed"


@dataclass
class EvalTask:
    task_id: str
    model_config_id: str
    model_display_name: str
    backend: str
    datasets: List[str]
    limit: int
    max_tokens: int
    transformers_version: str
    # hmonnx export fields
    export_config: str = ""       # e.g. configs_merak/xh2a/llm_models/gemma4_moe/26b_a4b_it/...py
    export_script: str = ""       # e.g. examples_merak/llm/gemma4_moe/..._export_hmonnx.py
    export_meta_info_path: str = ""  # absolute path to user-provided LLM meta file
    vision_export_meta_info_path: str = ""  # absolute path to user-provided vision export_meta_info.json
    hf_model_dir: str = ""        # HF model dir for export
    cuda_devices: str = ""        # user-specified CUDA_VISIBLE_DEVICES (e.g. "0,1" or "auto")
    resume_from_cache: str = ""   # existing evalscope timestamp dir used for resume/use_cache
    # status fields
    status: TaskStatus = TaskStatus.PENDING
    created_at: str = ""
    started_at: str = ""
    finished_at: str = ""
    error: str = ""
    work_dir: str = ""
    report_path: str = ""
    log_file: str = ""
    pid: int = 0

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        if not self.work_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.work_dir = str(
                _HM_EVAL_ROOT / "outputs" / f"{self.model_config_id}_{self.backend}_{timestamp}"
            )
        if not self.log_file:
            self.log_file = str(Path(self.work_dir) / "eval.log")

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EvalTask":
        d = dict(d)
        d["status"] = TaskStatus(d.get("status", "pending"))
        return cls(**d)


_COMPLETED_LOG_ERROR_RE = re.compile(
    r"prediction failed|Traceback \(most recent call last\)|RuntimeError:|Exception:|\bFAILED:\b",
    flags=re.IGNORECASE,
)
_CHOICE_DATASET_LETTERS: dict[str, set[str]] = {
    "mmlu_pro": set("ABCDEFGHIJ"),
    "mmlu": set("ABCD"),
    "ceval": set("ABCD"),
    "cmmlu": set("ABCD"),
    "arc": set("ABCD"),
    "hellaswag": set("ABCD"),
    "winogrande": set("AB"),
    "truthfulqa": set("ABCD"),
    "gpqa": set("ABCD"),
}


def _load_task_report(task: EvalTask) -> Optional[Dict[str, Any]]:
    report_path = Path(task.report_path) if task.report_path else Path(task.work_dir) / "report.json"
    if not report_path.exists():
        return None
    try:
        return json.loads(report_path.read_text(encoding="utf-8"))
    except Exception:
        logger.exception("Failed to load report for task %s", task.task_id)
        return None


def _report_failed_datasets(report: Dict[str, Any]) -> list[str]:
    failed = []
    for dataset_name, summary in report.get("summary", {}).items():
        if not isinstance(summary, dict):
            continue
        if summary.get("status") != "completed":
            failed.append(dataset_name)
    return failed


def _missing_metric_datasets(report: Dict[str, Any]) -> list[str]:
    metric_keys = {"macro_acc", "accuracy", "score", "exact_match", "pass@1"}
    missing = []
    for dataset_name, summary in report.get("summary", {}).items():
        if not isinstance(summary, dict) or summary.get("status") != "completed":
            continue
        if not any(isinstance(summary.get(key), (int, float)) for key in metric_keys):
            missing.append(dataset_name)
    return missing


def _find_invalid_choice_predictions(task: EvalTask) -> Optional[str]:
    if len(task.datasets) != 1:
        return None

    dataset_name = task.datasets[0]
    valid_letters = _CHOICE_DATASET_LETTERS.get(dataset_name)
    if not valid_letters:
        return None

    work_dir = Path(task.work_dir)
    if not work_dir.exists():
        return None

    checked_count = 0
    invalid_predictions: list[str] = []
    for review_file in sorted(work_dir.rglob("reviews/**/*.jsonl")):
        try:
            for line in review_file.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                extracted = (
                    record.get("sample_score", {})
                    .get("score", {})
                    .get("extracted_prediction")
                )
                if extracted is None:
                    continue
                checked_count += 1
                prediction = str(extracted).strip().upper()
                if prediction not in valid_letters:
                    invalid_predictions.append(prediction or "<empty>")
        except Exception:
            logger.exception("Failed to parse review file for task %s: %s", task.task_id, review_file)

    if invalid_predictions:
        preview = ", ".join(invalid_predictions[:5])
        return f"选择题评测提取到非法选项: {preview}。"

    if checked_count == 0 and dataset_name == "mmlu_pro":
        return "MMLU-Pro 已完成但未生成可用 review 结果，报告不完整。"
    return None


def detect_abnormal_result(task: EvalTask, log_text: str = "") -> tuple[bool, str]:
    if task.status not in (TaskStatus.COMPLETED, TaskStatus.ABNORMAL):
        return False, ""

    report = _load_task_report(task)
    if report is not None:
        failed_datasets = _report_failed_datasets(report)
        if failed_datasets:
            return True, f"报告中存在失败数据集: {', '.join(failed_datasets)}"

        missing_metric_datasets = _missing_metric_datasets(report)
        if missing_metric_datasets:
            return True, f"报告缺少可用指标: {', '.join(missing_metric_datasets)}"

    invalid_choice_reason = _find_invalid_choice_predictions(task)
    if invalid_choice_reason:
        return True, invalid_choice_reason

    combined_log_text = f"{task.error}\n{log_text}".strip()
    if combined_log_text and _COMPLETED_LOG_ERROR_RE.search(combined_log_text):
        return True, "日志中包含 prediction failed / traceback 等异常记录。"

    return False, ""


def _find_latest_resume_cache_dir(work_dir: str) -> Optional[str]:
    work_path = Path(work_dir)
    if not work_path.exists():
        return None

    candidates = []
    for child in work_path.iterdir():
        if not child.is_dir():
            continue
        if (child / "predictions").exists() or (child / "reviews").exists():
            candidates.append(child)

    if not candidates:
        return None

    return str(max(candidates, key=lambda path: path.stat().st_mtime))


class TaskManager:
    """Manages evaluation tasks with persistence and process isolation."""

    def __init__(self) -> None:
        _TASKS_DIR.mkdir(parents=True, exist_ok=True)
        self._tasks: Dict[str, EvalTask] = {}
        self._load_tasks()

    def _load_tasks(self) -> None:
        """Load persisted tasks from disk."""
        for task_file in sorted(_TASKS_DIR.glob("*.json")):
            try:
                data = json.loads(task_file.read_text(encoding="utf-8"))
                task = EvalTask.from_dict(data)
                self._tasks[task.task_id] = task
            except Exception:
                logger.exception("Failed to load task: %s", task_file)

    def _save_task(self, task: EvalTask) -> None:
        """Persist a task to disk."""
        task_file = _TASKS_DIR / f"{task.task_id}.json"
        task_file.write_text(
            json.dumps(task.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def create_task(
        self,
        model_config_id: str,
        model_display_name: str,
        backend: str,
        datasets: List[str],
        limit: int,
        max_tokens: int,
        transformers_version: str,
        export_config: str = "",
        export_script: str = "",
        export_meta_info_path: str = "",
        vision_export_meta_info_path: str = "",
        hf_model_dir: str = "",
        cuda_devices: str = "",
        resume_from_cache: str = "",
    ) -> EvalTask:
        """Create a new evaluation task."""
        task = EvalTask(
            task_id=str(uuid.uuid4())[:8],
            model_config_id=model_config_id,
            model_display_name=model_display_name,
            backend=backend,
            datasets=datasets,
            limit=limit,
            max_tokens=max_tokens,
            transformers_version=transformers_version,
            export_config=export_config,
            export_script=export_script,
            export_meta_info_path=export_meta_info_path,
            vision_export_meta_info_path=vision_export_meta_info_path,
            hf_model_dir=hf_model_dir,
            cuda_devices=cuda_devices,
            resume_from_cache=resume_from_cache,
        )
        self._tasks[task.task_id] = task
        self._save_task(task)
        logger.info("Created task %s: %s %s %s", task.task_id, model_display_name, backend, datasets)
        return task

    def start_task(self, task_id: str) -> bool:
        """Start a task in a subprocess."""
        task = self._tasks.get(task_id)
        if task is None:
            return False

        Path(task.work_dir).mkdir(parents=True, exist_ok=True)

        # Build the subprocess command
        cmd = [
            sys.executable, "-u", "-m", "hm_eval.worker",
            "--task-id", task.task_id,
            "--task-file", str(_TASKS_DIR / f"{task.task_id}.json"),
        ]

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        # Ensure the repo root is in PYTHONPATH
        repo_root = str(_HM_EVAL_ROOT.parent)
        existing_path = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{repo_root}:{existing_path}" if existing_path else repo_root

        try:
            log_mode = "a" if task.resume_from_cache and Path(task.log_file).exists() else "w"
            log_file_handle = open(task.log_file, log_mode, encoding="utf-8", buffering=1)
            process = subprocess.Popen(
                cmd,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log_file_handle,
                stderr=subprocess.STDOUT,
                cwd=repo_root,
                start_new_session=True,
                close_fds=True,
            )
            log_file_handle.close()
            task.pid = process.pid
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.now().isoformat()
            self._save_task(task)
            logger.info("Started task %s (PID %d)", task_id, process.pid)
            return True
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            self._save_task(task)
            logger.exception("Failed to start task %s", task_id)
            return False

    def resume_task(self, task_id: str, cuda_devices: Optional[str] = None) -> bool:
        task = self.update_task_status(task_id)
        if task is None:
            return False

        resume_from_cache = _find_latest_resume_cache_dir(task.work_dir)
        if not resume_from_cache:
            logger.warning("Task %s has no resumable evalscope cache under %s", task_id, task.work_dir)
            return False

        task.resume_from_cache = resume_from_cache
        task.cuda_devices = cuda_devices if cuda_devices is not None else "auto"
        task.status = TaskStatus.PENDING
        task.error = ""
        task.started_at = ""
        task.finished_at = ""
        task.report_path = ""
        task.pid = 0
        self._save_task(task)
        logger.info("Resuming task %s from cache %s on CUDA_VISIBLE_DEVICES=%s", task_id, resume_from_cache, task.cuda_devices)
        return self.start_task(task_id)

    def update_task_status(self, task_id: str) -> Optional[EvalTask]:
        """Re-read task state from disk (worker writes updates) and check PID."""
        task = self._tasks.get(task_id)
        if task is None:
            return None

        # Re-read from disk — worker subprocess updates the JSON file directly
        task_file = _TASKS_DIR / f"{task_id}.json"
        if task_file.exists():
            try:
                data = json.loads(task_file.read_text(encoding="utf-8"))
                task = EvalTask.from_dict(data)
                self._tasks[task_id] = task
            except Exception:
                logger.exception("Failed to re-read task %s from disk", task_id)

        # If still marked as running, verify the process is alive
        if task.status in (TaskStatus.RUNNING, TaskStatus.EXPORTING, TaskStatus.ENV_SETUP) and task.pid > 0:
            try:
                os.kill(task.pid, 0)  # Check if process exists
            except OSError:
                # Process died without updating status — mark as failed or completed
                report_path = Path(task.work_dir) / "report.json"
                if report_path.exists():
                    task.status = TaskStatus.COMPLETED
                    task.report_path = str(report_path)
                else:
                    task.status = TaskStatus.FAILED
                    task.error = "进程异常退出"
                task.finished_at = datetime.now().isoformat()
                self._save_task(task)

        if task.status in (TaskStatus.COMPLETED, TaskStatus.ABNORMAL):
            log_text = ""
            if task.log_file and Path(task.log_file).exists():
                try:
                    log_text = "\n".join(_read_last_log_lines(Path(task.log_file), 400))
                except Exception:
                    log_text = ""

            is_abnormal, abnormal_reason = detect_abnormal_result(task, log_text)
            if is_abnormal and (task.status != TaskStatus.ABNORMAL or task.error != abnormal_reason):
                task.status = TaskStatus.ABNORMAL
                task.error = abnormal_reason
                self._save_task(task)
            elif not is_abnormal and task.status == TaskStatus.ABNORMAL:
                task.status = TaskStatus.COMPLETED
                task.error = ""
                self._save_task(task)

        return task

    def get_task(self, task_id: str) -> Optional[EvalTask]:
        return self._tasks.get(task_id)

    def list_tasks(self) -> List[EvalTask]:
        """Return all tasks, most recent first. Refreshes non-terminal tasks from disk."""
        # Also pick up any new task files created since the last load
        self._load_tasks()
        for task in list(self._tasks.values()):
            self.update_task_status(task.task_id)
        return sorted(self._tasks.values(), key=lambda t: t.created_at, reverse=True)

    def get_task_log(self, task_id: str, tail: int = 100) -> str:
        """Read the last N lines of a task's log file."""
        task = self._tasks.get(task_id)
        if task is None or not task.log_file:
            return ""
        log_path = Path(task.log_file)
        if not log_path.exists():
            return "(no log file yet)"
        try:
            lines = _read_last_log_lines(log_path, tail)
            return "\n".join(lines[-tail:])
        except Exception:
            return "(failed to read log)"

    def get_task_report(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Load the task's report JSON if completed."""
        task = self.update_task_status(task_id)
        if task is None:
            return None
        report_path = task.report_path or str(Path(task.work_dir) / "report.json")
        if Path(report_path).exists():
            return json.loads(Path(report_path).read_text(encoding="utf-8"))
        return None

    def delete_task(self, task_id: str) -> Optional[EvalTask]:
        """Delete a task and its local evaluation artifacts."""
        task = self.update_task_status(task_id)
        if task is None:
            return None

        self._terminate_task_process(task)

        task_file = _TASKS_DIR / f"{task_id}.json"
        if task_file.exists():
            task_file.unlink(missing_ok=True)

        work_dir = Path(task.work_dir).resolve() if task.work_dir else None
        outputs_root = (_HM_EVAL_ROOT / "outputs").resolve()
        if work_dir and work_dir.exists() and outputs_root in work_dir.parents:
            shutil.rmtree(work_dir, ignore_errors=True)

        log_path = Path(task.log_file).resolve() if task.log_file else None
        if log_path and log_path.exists() and not (work_dir and log_path.is_relative_to(work_dir)):
            log_path.unlink(missing_ok=True)

        self._tasks.pop(task_id, None)
        logger.info("Deleted task %s and cleaned local artifacts", task_id)
        return task

    def _terminate_task_process(self, task: EvalTask) -> None:
        if task.pid <= 0:
            return

        try:
            os.kill(task.pid, signal.SIGTERM)
        except OSError:
            return

        deadline = time.time() + 1.0
        while time.time() < deadline:
            try:
                os.kill(task.pid, 0)
            except OSError:
                return
            time.sleep(0.1)

        try:
            os.kill(task.pid, signal.SIGKILL)
        except OSError:
            return


def _read_last_log_lines(log_path: Path, tail: int) -> List[str]:
    if tail <= 0:
        return []

    chunk_size = 4096
    data = bytearray()
    newline_count = 0

    with log_path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        file_size = handle.tell()
        offset = file_size

        while offset > 0 and newline_count <= tail:
            read_size = min(chunk_size, offset)
            offset -= read_size
            handle.seek(offset)
            block = handle.read(read_size)
            data[:0] = block
            newline_count = data.count(b"\n")

    text = data.decode("utf-8", errors="replace")
    return text.splitlines()
