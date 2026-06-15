"""Dynamic scheduler for LIBERO-10 baseline and Full-HMONNX evaluation.

The script is intentionally conservative:
  * dry-run by default; pass --run to actually launch jobs;
  * one worker per visible GPU by default;
  * no video export;
  * auto-skips busy GPUs unless explicitly configured otherwise;
  * writes per-job reports/logs and an incremental summary.
"""

from __future__ import annotations

import sys
from pathlib import Path

_VLA_JEPA_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _VLA_JEPA_ROOT.parents[2]
for _path in (_VLA_JEPA_ROOT, _REPO_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import argparse
import csv
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from common.paths import (
    DEFAULT_CONTEXT_GRAPH_HMONNX,
    DEFAULT_LIBERO_CONFIG_PATH,
    DEFAULT_LOG_ROOT,
    DEFAULT_MODEL,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_PYTHON,
    DEFAULT_STANDARD_ACTION_HEAD_HMONNX,
    DEFAULT_VISUAL_ENCODER_HMONNX,
    REPO_ROOT,
)


ROOT = REPO_ROOT
PYTHON = DEFAULT_PYTHON
MODEL = Path(DEFAULT_MODEL)
BASELINE_EVAL = _VLA_JEPA_ROOT / "eval" / "eval_action_head_hmonnx_libero.py"
FULL_HMONNX_EVAL = _VLA_JEPA_ROOT / "eval" / "eval_full_hmonnx_libero.py"

DEFAULT_VISUAL_HMONNX = Path(DEFAULT_VISUAL_ENCODER_HMONNX)
DEFAULT_CONTEXT_HMONNX = Path(DEFAULT_CONTEXT_GRAPH_HMONNX)
DEFAULT_ACTION_HMONNX = Path(DEFAULT_STANDARD_ACTION_HEAD_HMONNX)


@dataclass(frozen=True)
class GpuInfo:
    index: int
    name: str
    total_mib: int
    used_mib: int
    free_mib: int
    util_gpu: int
    util_mem: int
    temperature: int


@dataclass(frozen=True)
class Job:
    job_id: str
    mode: str
    task_id: int
    seed_start: int
    n_episodes: int
    report: str
    log: str
    timeout_s: int

    @property
    def seed_end(self) -> int:
        return self.seed_start + self.n_episodes - 1



class Scheduler:
    def __init__(self, args: argparse.Namespace, jobs: list[Job], gpus: list[int]) -> None:
        self.args = args
        self.gpus = gpus
        self.stop_requested = False
        self.failure_count = 0
        self.full_queue = [job for job in jobs if job.mode == "full_hmonnx"]
        self.baseline_queue = [job for job in jobs if job.mode == "baseline"]
        self.active: dict[int, dict[str, Any]] = {}
        self.summary: dict[str, Any] = {
            "status": "running",
            "started_at": timestamp(),
            "config": summary_config(args, gpus),
            "jobs": [],
            "task_summary": {},
            "overall_summary": {},
            "failed_jobs": [],
            "remaining_jobs": len(jobs),
        }

    def run(self) -> int:
        self._install_signal_handlers()
        write_json_atomic(self.args.summary_json, self.summary)
        write_csv(self.args.summary_csv, self.summary["jobs"])

        try:
            while self.active or self.full_queue or self.baseline_queue:
                if self.stop_requested:
                    self._terminate_active_processes()
                    break
                self._launch_available_jobs()
                changed = self._poll_active_jobs()
                if changed:
                    self._write_progress()
                if not self.active and not self.full_queue and not self.baseline_queue:
                    break
                time.sleep(1.0)
        finally:
            if self.stop_requested:
                self._terminate_active_processes()
            self._write_final_summary()

        return 1 if self.summary["status"] == "failed" else 0

    def _install_signal_handlers(self) -> None:
        def handler(signum: int, _frame: Any) -> None:
            self.summary["stop_reason"] = f"received signal {signum}"
            self.stop_requested = True
            self._write_progress()

        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)

    def _launch_available_jobs(self) -> None:
        for slot_id, gpu in enumerate(self.gpus):
            if slot_id in self.active:
                continue
            if self.stop_requested:
                return
            job = self._pop_next_job()
            if job is None:
                return
            record = self._skip_existing_if_possible(slot_id, gpu, job)
            if record is not None:
                self._record_finished_job(record)
                continue
            self._launch_job(slot_id, gpu, job)

    def _pop_next_job(self) -> Job | None:
        active_by_mode = self._active_counts_by_mode()
        baseline_done = not self.baseline_queue and active_by_mode["baseline"] == 0
        full_limit = len(self.gpus) if baseline_done else self.args.full_hmonnx_workers

        if self.baseline_queue and active_by_mode["baseline"] < self.args.baseline_workers:
            return self.baseline_queue.pop(0)
        if self.full_queue and active_by_mode["full_hmonnx"] < full_limit:
            return self.full_queue.pop(0)
        if not self.full_queue and self.baseline_queue:
            return self.baseline_queue.pop(0)
        if not self.baseline_queue and self.full_queue and active_by_mode["full_hmonnx"] < len(self.gpus):
            return self.full_queue.pop(0)
        return None

    def _active_counts_by_mode(self) -> dict[str, int]:
        counts = defaultdict(int)
        for item in self.active.values():
            counts[item["job"].mode] += 1
        return counts

    def _skip_existing_if_possible(self, worker_id: int, gpu: int, job: Job) -> dict[str, Any] | None:
        report_path = ROOT / job.report
        if not self.args.resume or not is_ok_report(report_path, job):
            return None
        record: dict[str, Any] = {
            **asdict(job),
            "worker_id": worker_id,
            "gpu": gpu,
            "status": "skipped_existing",
            "started_at": timestamp(),
            "finished_at": timestamp(),
            "elapsed_s": 0.0,
        }
        record.update(parse_report(report_path, job))
        return record

    def _launch_job(self, worker_id: int, gpu: int, job: Job) -> None:
        report_path = ROOT / job.report
        log_path = ROOT / job.log
        report_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = build_command(self.args, job)
        record: dict[str, Any] = {
            **asdict(job),
            "worker_id": worker_id,
            "gpu": gpu,
            "status": "running",
            "started_at": timestamp(),
            "cmd": " ".join(cmd),
        }
        log_file = log_path.open("w")
        log_file.write(f"# started_at: {record['started_at']}\n")
        log_file.write(f"# worker_id: {worker_id}\n")
        log_file.write(f"# gpu: {gpu}\n")
        log_file.write(f"# command: {record['cmd']}\n")
        log_file.flush()

        proc = subprocess.Popen(
            cmd,
            cwd=ROOT,
            env=build_env(gpu),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        self.active[worker_id] = {
            "job": job,
            "record": record,
            "proc": proc,
            "log_file": log_file,
            "started_monotonic": time.time(),
        }

    def _poll_active_jobs(self) -> bool:
        changed = False
        for worker_id, item in list(self.active.items()):
            proc: subprocess.Popen[str] = item["proc"]
            job: Job = item["job"]
            runtime_s = time.time() - float(item["started_monotonic"])
            if proc.poll() is None and runtime_s <= job.timeout_s:
                continue
            if proc.poll() is None and runtime_s > job.timeout_s:
                self._terminate_process(proc)
                item["record"]["status"] = "timeout"
                item["record"]["returncode"] = proc.returncode
            else:
                item["record"]["returncode"] = proc.returncode
            record = self._finalize_active_item(worker_id, item)
            self._record_finished_job(record)
            changed = True
        return changed

    def _finalize_active_item(self, worker_id: int, item: dict[str, Any]) -> dict[str, Any]:
        job: Job = item["job"]
        record: dict[str, Any] = item["record"]
        log_file = item["log_file"]
        log_file.flush()
        log_file.close()
        self.active.pop(worker_id, None)

        record["elapsed_s"] = time.time() - float(item["started_monotonic"])
        record["finished_at"] = timestamp()

        if record.get("status") == "timeout":
            attach_tail(record, ROOT / job.log)
            return record
        if self.stop_requested:
            record["status"] = "stopped"
            attach_tail(record, ROOT / job.log)
            return record
        if record.get("returncode") != 0:
            record["status"] = "failed_process"
            attach_tail(record, ROOT / job.log)
            return record

        report_path = ROOT / job.report
        if not report_path.exists():
            record["status"] = "missing_report"
            attach_tail(record, ROOT / job.log)
            return record

        try:
            record.update(parse_report(report_path, job))
        except Exception as exc:  # noqa: BLE001 - keep parse failures inspectable.
            record["status"] = "bad_report"
            record["error_type"] = type(exc).__name__
            record["error"] = str(exc)
            attach_tail(record, ROOT / job.log)
            return record

        if record.get("report_status") not in {None, "ok"}:
            record["status"] = "failed_report"
            return record

        record["status"] = "ok"
        return record

    def _record_finished_job(self, record: dict[str, Any]) -> None:
        self.summary["jobs"].append(record)
        if record["status"] not in {"ok", "skipped_existing"}:
            self.failure_count += 1
            self.summary["failed_jobs"].append(record)
            if self.failure_count >= self.args.stop_after_failures:
                self.summary["stop_reason"] = f"stop_after_failures reached: {self.failure_count}"
                self.stop_requested = True
        self._refresh_rollups()
        self._write_progress()

    def _refresh_rollups(self) -> None:
        self.summary["task_summary"] = compute_task_summary(self.summary["jobs"])
        self.summary["overall_summary"] = compute_overall_summary(self.summary["task_summary"])
        self.summary["remaining_jobs"] = len(self.full_queue) + len(self.baseline_queue) + len(self.active)

    def _write_progress(self) -> None:
        self._refresh_rollups()
        write_json_atomic(self.args.summary_json, self.summary)
        write_csv(self.args.summary_csv, self.summary["jobs"])

    def _write_final_summary(self) -> None:
        self._refresh_rollups()
        if self.summary["failed_jobs"]:
            self.summary["status"] = "failed"
        elif self.stop_requested and self.summary.get("remaining_jobs", 0) > 0:
            self.summary["status"] = "stopped"
        else:
            self.summary["status"] = "ok"
        self.summary["finished_at"] = timestamp()
        write_json_atomic(self.args.summary_json, self.summary)
        write_csv(self.args.summary_csv, self.summary["jobs"])

    def _terminate_active_processes(self) -> None:
        for item in list(self.active.values()):
            self._terminate_process(item["proc"])
            try:
                item["log_file"].flush()
                item["log_file"].close()
            except Exception:
                pass
        self.active.clear()

    @staticmethod
    def _terminate_process(proc: subprocess.Popen[str]) -> None:
        if proc.poll() is not None:
            return
        proc.terminate()
        deadline = time.time() + 30
        while proc.poll() is None and time.time() < deadline:
            time.sleep(0.2)
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=30)

def timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def parse_int_list(value: str) -> list[int]:
    result: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start, end = int(start_s), int(end_s)
            if end < start:
                raise ValueError(f"bad range: {part}")
            result.extend(range(start, end + 1))
        else:
            result.append(int(part))
    return sorted(dict.fromkeys(result))


def query_gpus() -> list[GpuInfo]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,"
        "utilization.memory,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    gpus = []
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        fields = [item.strip() for item in line.split(",")]
        gpus.append(
            GpuInfo(
                index=int(fields[0]),
                name=fields[1],
                total_mib=int(fields[2]),
                used_mib=int(fields[3]),
                free_mib=int(fields[4]),
                util_gpu=int(fields[5]),
                util_mem=int(fields[6]),
                temperature=int(fields[7]),
            )
        )
    return gpus


def select_gpus(args: argparse.Namespace) -> tuple[list[int], list[GpuInfo]]:
    all_gpus = query_gpus()
    requested = None if args.gpus == "auto" else set(parse_int_list(args.gpus))
    selected = []
    for gpu in all_gpus:
        if requested is not None and gpu.index not in requested:
            continue
        if not args.include_busy_gpus:
            if gpu.free_mib < args.min_free_mib or gpu.util_gpu > args.max_gpu_util:
                continue
        selected.append(gpu.index)
    return selected, all_gpus


def expand_gpu_slots(gpus: list[int], workers_per_gpu: int) -> list[int]:
    if workers_per_gpu <= 0:
        raise ValueError("workers_per_gpu must be positive")
    return [gpu for gpu in gpus for _ in range(workers_per_gpu)]


def build_jobs(args: argparse.Namespace) -> list[Job]:
    jobs = []
    modes = parse_modes(args.modes)
    task_ids = parse_int_list(args.task_ids)
    seeds = list(range(args.seed_start, args.seed_start + args.num_seeds))

    if "full_hmonnx" in modes:
        jobs.extend(make_mode_jobs(args, "full_hmonnx", task_ids, seeds, args.full_chunk_size, args.full_timeout_s))
    if "baseline" in modes:
        jobs.extend(
            make_mode_jobs(
                args, "baseline", task_ids, seeds, args.baseline_chunk_size, args.baseline_timeout_s
            )
        )

    if args.schedule_order == "full_first":
        jobs.sort(key=lambda job: (0 if job.mode == "full_hmonnx" else 1, job.task_id, job.seed_start))
    elif args.schedule_order == "baseline_first":
        jobs.sort(key=lambda job: (0 if job.mode == "baseline" else 1, job.task_id, job.seed_start))
    else:
        jobs.sort(key=lambda job: (job.task_id, job.seed_start, job.mode))
    return jobs


def display_path(path: Path) -> str:
    return os.path.relpath(path, REPO_ROOT)


def make_mode_jobs(
    args: argparse.Namespace,
    mode: str,
    task_ids: list[int],
    seeds: list[int],
    chunk_size: int,
    timeout_s: int,
) -> list[Job]:
    jobs = []
    for task_id in task_ids:
        for chunk in chunks(seeds, chunk_size):
            seed_start = chunk[0]
            seed_end = chunk[-1]
            name = f"{mode}_task{task_id}_seed{seed_start}_{seed_end}"
            report = args.out_dir / mode / f"task{task_id}" / f"{name}_report.json"
            log = args.log_dir / f"{name}.log"
            jobs.append(
                Job(
                    job_id=name,
                    mode=mode,
                    task_id=task_id,
                    seed_start=seed_start,
                    n_episodes=len(chunk),
                    report=display_path(report),
                    log=display_path(log),
                    timeout_s=timeout_s,
                )
            )
    return jobs


def chunks(values: list[int], size: int) -> list[list[int]]:
    if size <= 0:
        raise ValueError("chunk size must be positive")
    return [values[index : index + size] for index in range(0, len(values), size)]


def parse_modes(value: str) -> set[str]:
    modes = {part.strip() for part in value.split(",") if part.strip()}
    allowed = {"baseline", "full_hmonnx"}
    unknown = modes - allowed
    if unknown:
        raise ValueError(f"unknown modes: {sorted(unknown)}")
    return modes


def build_command(args: argparse.Namespace, job: Job) -> list[str]:
    if job.mode == "baseline":
        return [
            args.python,
            str(BASELINE_EVAL),
            "--model",
            str(args.model),
            "--mode",
            "original",
            "--device",
            "cuda:0",
            "--task",
            args.task,
            "--task-ids",
            str(job.task_id),
            "--n-episodes",
            str(job.n_episodes),
            "--batch-size",
            str(args.batch_size),
            "--seed",
            str(job.seed_start),
            "--max-episodes-rendered",
            "0",
            "--report",
            job.report,
        ]

    cmd = [
        args.python,
        str(FULL_HMONNX_EVAL),
        "--model",
        str(args.model),
        "--visual-hmonnx",
        str(args.visual_hmonnx),
        "--context-hmonnx",
        str(args.context_hmonnx),
        "--action-hmonnx",
        str(args.action_hmonnx),
        "--device",
        "cuda:0",
        "--hmonnx-device",
        "cuda:0",
        "--context-dtype",
        args.context_dtype,
        "--task",
        args.task,
        "--task-ids",
        str(job.task_id),
        "--n-episodes",
        str(job.n_episodes),
        "--batch-size",
        str(args.batch_size),
        "--seed",
        str(job.seed_start),
        "--max-episodes-rendered",
        "0",
        "--report",
        job.report,
    ]
    if args.trace_hmonnx_progress:
        cmd.append("--trace-hmonnx-progress")
    return cmd


def build_env(gpu: int) -> dict[str, str]:
    env = os.environ.copy()
    conda_lib = str(Path(sys.prefix) / "lib")
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "MUJOCO_GL": "osmesa",
            "PYOPENGL_PLATFORM": "osmesa",
            "NUMBA_DISABLE_JIT": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "LIBERO_CONFIG_PATH": str(DEFAULT_LIBERO_CONFIG_PATH),
            "LD_LIBRARY_PATH": conda_lib + ":" + env.get("LD_LIBRARY_PATH", ""),
        }
    )
    return env


def wait_process(proc: subprocess.Popen[str], stop_event: threading.Event, timeout_s: int) -> int:
    deadline = time.time() + timeout_s
    while proc.poll() is None:
        if stop_event.is_set():
            proc.terminate()
            try:
                return proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                return proc.wait(timeout=30)
        if time.time() > deadline:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=30)
            raise subprocess.TimeoutExpired(proc.args, timeout_s)
        time.sleep(1)
    return int(proc.returncode)


def terminate_by_job_id(
    active_processes: dict[str, subprocess.Popen[str]],
    lock: threading.Lock,
    job_id: str,
) -> None:
    with lock:
        proc = active_processes.get(job_id)
    if proc is not None and proc.poll() is None:
        proc.terminate()


def is_ok_report(report_path: Path, job: Job) -> bool:
    if not report_path.exists():
        return False
    try:
        record = parse_report(report_path, job)
    except Exception:
        return False
    return record.get("report_status") in {None, "ok"} and record.get("n_episodes") == job.n_episodes


def parse_report(report_path: Path, job: Job) -> dict[str, Any]:
    data = json.loads(report_path.read_text())
    eval_info = data.get("eval") if isinstance(data.get("eval"), dict) else {}
    overall = eval_info.get("overall", {}) if isinstance(eval_info, dict) else {}
    runtime = data.get("runtime", {}) if isinstance(data.get("runtime"), dict) else {}

    qwen_context = runtime.get("qwen_context", {}) if isinstance(runtime, dict) else {}
    action = runtime.get("action", {}) if isinstance(runtime, dict) else {}

    n_episodes = int(overall.get("n_episodes") or data.get("n_episodes_per_task") or job.n_episodes)
    pc_success = maybe_float(overall.get("pc_success"))

    return {
        "report_status": data.get("status"),
        "report_elapsed_s": maybe_float(data.get("elapsed_s")),
        "n_episodes": n_episodes,
        "pc_success": pc_success,
        "success_count": None if pc_success is None else pc_success * n_episodes / 100.0,
        "avg_sum_reward": maybe_float(overall.get("avg_sum_reward")),
        "avg_max_reward": maybe_float(overall.get("avg_max_reward")),
        "eval_s": maybe_float(overall.get("eval_s")),
        "eval_ep_s": maybe_float(overall.get("eval_ep_s")),
        "qwen_context_calls": qwen_context.get("num_calls"),
        "context_selected_counts": qwen_context.get("selected_counts"),
        "action_predict_calls": action.get("num_predict_calls"),
        "action_denoise_steps": action.get("num_denoise_steps"),
        "error_type": data.get("error_type"),
        "error": data.get("error"),
    }


def maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def compute_task_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, dict[int, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        if record.get("status") not in {"ok", "skipped_existing"}:
            continue
        grouped[record["mode"]][int(record["task_id"])].append(record)

    summary: dict[str, Any] = {}
    for mode, by_task in grouped.items():
        summary[mode] = {}
        for task_id, task_records in sorted(by_task.items()):
            n_episodes = sum(int(item.get("n_episodes") or item.get("n_episodes", 0)) for item in task_records)
            success_count = sum(float(item.get("success_count") or 0.0) for item in task_records)
            elapsed_s = sum(float(item.get("elapsed_s") or 0.0) for item in task_records)
            summary[mode][str(task_id)] = {
                "jobs": len(task_records),
                "n_episodes": n_episodes,
                "success_count": success_count,
                "pc_success": None if n_episodes == 0 else success_count / n_episodes * 100.0,
                "elapsed_s": elapsed_s,
            }
    return summary


def compute_overall_summary(task_summary: dict[str, Any]) -> dict[str, Any]:
    overall = {}
    for mode, by_task in task_summary.items():
        n_episodes = 0
        success_count = 0.0
        task_rates = []
        elapsed_s = 0.0
        for item in by_task.values():
            n_episodes += int(item["n_episodes"])
            success_count += float(item["success_count"])
            elapsed_s += float(item["elapsed_s"])
            if item["pc_success"] is not None:
                task_rates.append(float(item["pc_success"]))
        overall[mode] = {
            "tasks_completed": len(by_task),
            "n_episodes": n_episodes,
            "success_count": success_count,
            "episode_weighted_pc_success": None if n_episodes == 0 else success_count / n_episodes * 100.0,
            "task_mean_pc_success": None if not task_rates else sum(task_rates) / len(task_rates),
            "elapsed_s_sum_over_jobs": elapsed_s,
        }
    return overall


def attach_tail(record: dict[str, Any], log_path: Path, lines: int = 80) -> None:
    if not log_path.exists():
        return
    content = log_path.read_text(errors="replace").splitlines()
    record["log_tail"] = "\n".join(content[-lines:])


def write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    tmp.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "status",
        "mode",
        "task_id",
        "seed_start",
        "seed_end",
        "n_episodes",
        "gpu",
        "elapsed_s",
        "returncode",
        "pc_success",
        "success_count",
        "avg_sum_reward",
        "avg_max_reward",
        "report",
        "log",
        "error_type",
        "error",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summary_config(args: argparse.Namespace, gpus: list[int]) -> dict[str, Any]:
    return {
        "model": str(args.model),
        "task": args.task,
        "task_ids": parse_int_list(args.task_ids),
        "seed_start": args.seed_start,
        "num_seeds": args.num_seeds,
        "modes": sorted(parse_modes(args.modes)),
        "worker_gpu_slots": gpus,
        "physical_gpus": sorted(set(gpus)),
        "workers_per_gpu": args.workers_per_gpu,
        "batch_size": args.batch_size,
        "baseline_chunk_size": args.baseline_chunk_size,
        "full_chunk_size": args.full_chunk_size,
        "baseline_workers": args.baseline_workers,
        "full_hmonnx_workers": args.full_hmonnx_workers,
        "max_episodes_rendered": 0,
        "visual_hmonnx": str(args.visual_hmonnx),
        "context_hmonnx": str(args.context_hmonnx),
        "action_hmonnx": str(args.action_hmonnx),
    }


def validate_paths(args: argparse.Namespace) -> None:
    required = [Path(args.python), args.model, BASELINE_EVAL]
    if "full_hmonnx" in parse_modes(args.modes):
        required.extend([FULL_HMONNX_EVAL, args.visual_hmonnx, args.context_hmonnx, args.action_hmonnx])
    missing = [str(path) for path in required if not Path(path).exists()]
    if missing:
        raise FileNotFoundError("missing required paths:\n" + "\n".join(missing))


def check_disk(args: argparse.Namespace) -> None:
    usage = shutil.disk_usage(args.out_dir.parent)
    free_gib = usage.free / 1024**3
    if free_gib < args.min_disk_free_gb and not args.allow_low_disk:
        raise RuntimeError(
            f"only {free_gib:.1f} GiB free under {args.out_dir.parent}; "
            f"need at least {args.min_disk_free_gb:.1f} GiB or pass --allow-low-disk"
        )


def print_dry_run(args: argparse.Namespace, jobs: list[Job], selected_gpus: list[int], all_gpus: list[GpuInfo]) -> None:
    total_episodes = sum(job.n_episodes for job in jobs)
    print("DRY RUN: no jobs were launched")
    print(f"worker_gpu_slots={selected_gpus}")
    print(f"physical_gpus={sorted(set(selected_gpus))}")
    print("gpu_inventory=")
    for gpu in all_gpus:
        print(
            f"  gpu{gpu.index}: free={gpu.free_mib}MiB used={gpu.used_mib}MiB "
            f"util={gpu.util_gpu}% temp={gpu.temperature}C name={gpu.name}"
        )
    print(f"jobs={len(jobs)} episodes={total_episodes}")
    print(f"summary_json={args.summary_json}")
    print(f"summary_csv={args.summary_csv}")
    for job in jobs[: min(args.preview_jobs, len(jobs))]:
        print(
            f"  {job.job_id}: mode={job.mode} task={job.task_id} "
            f"seeds={job.seed_start}-{job.seed_end} n={job.n_episodes}"
        )
    if len(jobs) > args.preview_jobs:
        print(f"  ... {len(jobs) - args.preview_jobs} more jobs")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", help="Actually launch jobs. Default is dry-run only.")
    parser.add_argument("--python", default=PYTHON)
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--task", default="libero_10")
    parser.add_argument("--task-ids", default="0-9")
    parser.add_argument("--seed-start", type=int, default=1000)
    parser.add_argument("--num-seeds", type=int, default=20)
    parser.add_argument("--modes", default="baseline,full_hmonnx")
    parser.add_argument("--batch-size", type=int, default=1)

    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_ROOT / "eval" / "libero10_20eps")
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--summary-csv", type=Path, default=None)

    parser.add_argument("--visual-hmonnx", type=Path, default=DEFAULT_VISUAL_HMONNX)
    parser.add_argument("--context-hmonnx", type=Path, default=DEFAULT_CONTEXT_HMONNX)
    parser.add_argument("--action-hmonnx", type=Path, default=DEFAULT_ACTION_HMONNX)
    parser.add_argument("--context-dtype", choices=("float16", "float32", "bfloat16"), default="float16")
    parser.add_argument("--trace-hmonnx-progress", action="store_true")

    parser.add_argument("--gpus", default="auto", help="auto or comma/range list, e.g. 0,2-7")
    parser.add_argument("--include-busy-gpus", action="store_true")
    parser.add_argument("--min-free-mib", type=int, default=30_000)
    parser.add_argument("--max-gpu-util", type=int, default=30)
    parser.add_argument("--workers-per-gpu", type=int, default=1)
    parser.add_argument("--baseline-workers", type=int, default=1)
    parser.add_argument("--full-hmonnx-workers", type=int, default=None)
    parser.add_argument("--baseline-chunk-size", type=int, default=5)
    parser.add_argument("--full-chunk-size", type=int, default=2)
    parser.add_argument("--baseline-timeout-s", type=int, default=1800)
    parser.add_argument("--full-timeout-s", type=int, default=7200)
    parser.add_argument("--stop-after-failures", type=int, default=3)
    parser.add_argument(
        "--schedule-order", choices=("full_first", "baseline_first", "interleave"), default="full_first"
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--limit-jobs", type=int, default=None)
    parser.add_argument("--preview-jobs", type=int, default=30)
    parser.add_argument("--min-disk-free-gb", type=float, default=80.0)
    parser.add_argument("--allow-low-disk", action="store_true")
    args = parser.parse_args()

    if args.summary_json is None:
        args.summary_json = args.out_dir / "summary.json"
    if args.summary_csv is None:
        args.summary_csv = args.out_dir / "summary.csv"
    if not args.out_dir.is_absolute():
        args.out_dir = ROOT / args.out_dir
    if not args.log_dir.is_absolute():
        args.log_dir = ROOT / args.log_dir
    if not args.summary_json.is_absolute():
        args.summary_json = ROOT / args.summary_json
    if not args.summary_csv.is_absolute():
        args.summary_csv = ROOT / args.summary_csv

    return args


def main() -> int:
    args = parse_args()
    validate_paths(args)
    check_disk(args)
    selected_gpus, all_gpus = select_gpus(args)
    if not selected_gpus:
        raise RuntimeError("no GPUs selected; lower thresholds or pass --gpus/--include-busy-gpus")
    worker_gpus = expand_gpu_slots(selected_gpus, args.workers_per_gpu)

    if args.full_hmonnx_workers is None:
        args.full_hmonnx_workers = max(1, len(worker_gpus) - args.baseline_workers)
    args.full_hmonnx_workers = min(args.full_hmonnx_workers, len(worker_gpus))
    args.baseline_workers = min(args.baseline_workers, len(worker_gpus))

    jobs = build_jobs(args)
    if args.limit_jobs is not None:
        jobs = jobs[: args.limit_jobs]

    if not args.run:
        print_dry_run(args, jobs, worker_gpus, all_gpus)
        return 0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    scheduler = Scheduler(args, jobs, worker_gpus)
    return scheduler.run()


if __name__ == "__main__":
    sys.exit(main())
