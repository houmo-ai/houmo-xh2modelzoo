#!/usr/bin/env python3
"""Qwen3.6 27B hmonnx benchmark: MTP vs DFlash (quantized models).

Adapted from benchmark_qwen36_27b_compare.py (compiler version) to use
hmonnx runtime (Qwen3_5SpecDecodeONNXModel) instead of raw .hmm demos.

Usage:
    # Both methods on 2 GPUs in parallel:
    python benchmark_qwen36_27b_hmonnx_compare.py --method both --device-id 0 1

    # MTP only:
    python benchmark_qwen36_27b_hmonnx_compare.py --method mtp --device-id 0

    # DFlash only:
    python benchmark_qwen36_27b_hmonnx_compare.py --method dflash --device-id 1
"""

import argparse
import csv
import dataclasses
import json
import math
import os
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from loguru import logger

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from benchmark_qwen36_27b_cases import (
    CATEGORIES,
    LENGTH_BUCKETS,
    BenchmarkCase,
    build_cases,
)

METHODS = ("mtp", "dflash")

MTP_META_DEFAULT = str(
    Path("/data01/home/yujy/work/xh2modelzoo/work_dirs/"
         "qwen3_6_27b_xh2a_8k_w4a8_gptq_spec_mtp_draft4_headw4_20260519_004730/meta.json")
)
MTP_CROPPED_META_DEFAULT = str(
    Path("/data01/home/yujy/work/xh2modelzoo/work_dirs/"
         "qwen3_6_27b_xh2a_8k_w4a8_gptq_spec_mtp_k81920_draft4_headw4_20260519_005031/meta.json")
)
DFLASH_META_DEFAULT = str(
    Path("/data01/home/yujy/work/xh2modelzoo/work_dirs/"
         "qwen3_6_27b_xh2a_8k_w4a8_gptq_spec_dflash_draft9_input10_headw4_20260521_000942/meta.json")
)


def _set_env(device_id: int) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)


def _select_cases(args: argparse.Namespace) -> List[BenchmarkCase]:
    cases = build_cases()
    if args.category:
        selected = set(args.category)
        cases = [c for c in cases if c.category in selected]
    if args.length_bucket:
        selected = set(args.length_bucket)
        cases = [c for c in cases if c.length_bucket in selected]
    if args.case_id:
        selected = set(args.case_id)
        cases = [c for c in cases if c.case_id in selected]
    if args.case_limit > 0:
        cases = cases[: args.case_limit]
    return cases


def _load_rows(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, rows: List[Dict[str, object]]) -> None:
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def _status_by_key(rows: Iterable[Dict[str, object]]) -> Dict[Tuple[str, str], str]:
    return {(str(r["method"]), str(r["case_id"])): str(r.get("status", "")) for r in rows}


def _replace_row(rows: List[Dict[str, object]], row: Dict[str, object]) -> None:
    key = (row["method"], row["case_id"])
    for i, old in enumerate(rows):
        if (old.get("method"), old.get("case_id")) == key:
            rows[i] = row
            return
    rows.append(row)


def _fmt_float(v: Optional[float], digits: int = 2) -> str:
    if v is None or math.isnan(v):
        return "-"
    return f"{v:.{digits}f}"


def _fmt_pct(v: Optional[float], digits: int = 2) -> str:
    if v is None or math.isnan(v):
        return "-"
    return f"{v * 100:.{digits}f}%"


def _ok_rows(rows: Iterable[Dict[str, object]], method: Optional[str] = None, category: Optional[str] = None) -> List[Dict[str, object]]:
    selected = [r for r in rows if r.get("status") == "ok"]
    if method is not None:
        selected = [r for r in selected if r.get("method") == method]
    if category is not None:
        selected = [r for r in selected if r.get("category") == category]
    return selected


def _metric_rows(rows: Iterable[Dict[str, object]], metric: str) -> List[Dict[str, object]]:
    return [r for r in rows if r.get(metric) is not None]


def _summary(rows: List[Dict[str, object]]) -> Dict[str, object]:
    speed_rows = _metric_rows(rows, "decode_tokens_per_s")
    accept_rows = _metric_rows(rows, "acceptance_rate")
    speeds = [float(r["decode_tokens_per_s"]) for r in speed_rows]
    accepts = [float(r["acceptance_rate"]) for r in accept_rows]
    fastest = max(speed_rows, key=lambda r: float(r["decode_tokens_per_s"])) if speed_rows else None
    slowest = min(speed_rows, key=lambda r: float(r["decode_tokens_per_s"])) if speed_rows else None
    best_accept = max(accept_rows, key=lambda r: float(r["acceptance_rate"])) if accept_rows else None
    worst_accept = min(accept_rows, key=lambda r: float(r["acceptance_rate"])) if accept_rows else None
    return {
        "ok": len(rows),
        "median_speed": statistics.median(speeds) if speeds else None,
        "mean_speed": statistics.mean(speeds) if speeds else None,
        "fastest_speed": float(fastest["decode_tokens_per_s"]) if fastest else None,
        "fastest_case_id": fastest.get("case_id") if fastest else None,
        "slowest_speed": float(slowest["decode_tokens_per_s"]) if slowest else None,
        "slowest_case_id": slowest.get("case_id") if slowest else None,
        "median_acceptance": statistics.median(accepts) if accepts else None,
        "mean_acceptance": statistics.mean(accepts) if accepts else None,
        "best_acceptance": float(best_accept["acceptance_rate"]) if best_accept else None,
        "best_acceptance_case_id": best_accept.get("case_id") if best_accept else None,
        "worst_acceptance": float(worst_accept["acceptance_rate"]) if worst_accept else None,
        "worst_acceptance_case_id": worst_accept.get("case_id") if worst_accept else None,
    }


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    fields = [
        "method", "case_id", "category", "length_bucket", "status", "device_id",
        "decode_tokens_per_s", "acceptance_rate", "decode_tokens", "verify_calls",
        "draft_tokens", "accepted_tokens", "prompt_tokens", "question_chars",
        "elapsed_s", "error", "question",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def _write_report(path: Path, rows: List[Dict[str, object]], args: argparse.Namespace, csv_path: Path) -> None:
    total_cases = len(build_cases())
    lines = [
        "# Qwen3.6 27B hmonnx DFlash vs MTP Benchmark (Quantized)",
        "",
        f"- Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Target cases: {total_cases} prompts",
        f"- max_new_tokens: {args.max_new_tokens}",
        f"- MTP meta: {args.mtp_meta}",
        f"- DFlash meta: {args.dflash_meta}",
        f"- Results JSON: {args.json}",
        f"- Results CSV: {csv_path.name}",
        "",
        "## Overall Summary",
        "",
        "| Method | OK | Median speed | Mean speed | Median acceptance | Mean acceptance |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method in METHODS:
        stats = _summary(_ok_rows(rows, method))
        lines.append(
            f"| {method} | {stats['ok']} | {_fmt_float(stats['median_speed'])} | "
            f"{_fmt_float(stats['mean_speed'])} | "
            f"{_fmt_pct(stats['median_acceptance'])} | {_fmt_pct(stats['mean_acceptance'])} |"
        )
    lines.extend([
        "",
        "## Category Summary",
        "",
        "| Category | Method | OK | Median speed | Mean speed | Median acceptance | Mean acceptance |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for category in CATEGORIES:
        for method in METHODS:
            stats = _summary(_ok_rows(rows, method, category))
            lines.append(
                f"| {category} | {method} | {stats['ok']} | {_fmt_float(stats['median_speed'])} | "
                f"{_fmt_float(stats['mean_speed'])} | "
                f"{_fmt_pct(stats['median_acceptance'])} | {_fmt_pct(stats['mean_acceptance'])} |"
            )
    failed = [r for r in rows if r.get("status") == "failed"]
    if failed:
        lines.extend(["", "## Failed Cases", ""])
        for r in failed[:50]:
            error = str(r.get("error", "")).replace("\n", " ")[:200]
            lines.append(f"- {r.get('method')} {r.get('case_id')}: {error}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_artifacts(rows: List[Dict[str, object]], args: argparse.Namespace) -> None:
    json_path = Path(args.json).resolve()
    csv_path = Path(args.csv).resolve()
    report_path = Path(args.report).resolve()
    _write_json(json_path, rows)
    _write_csv(csv_path, rows)
    _write_report(report_path, rows, args, csv_path)


def _run_method(method: str, cases: List[BenchmarkCase], rows: List[Dict[str, object]], args: argparse.Namespace) -> None:
    from qwen3_5_spec_decode_metrics import run_dense_spec_generate_once, reset_dense_spec_runtime
    from qwen3_5_xh2a_spec_decode_test import load_spec_decode_runtime, parse_auto_offload_max_memory, parse_dtype

    meta_path = args.mtp_meta if method == "mtp" else args.dflash_meta
    device = f"cuda:0"
    exec_device = f"cuda:0"

    logger.info(f"Loading {method} runtime from {meta_path} on device {device}")
    runtime, tokenizer, meta_info = load_spec_decode_runtime(
        meta_path=meta_path,
        dtype=parse_dtype(args.dtype),
        device=device,
        exec_device=exec_device,
        auto_offload=True,
        auto_offload_max_memory=None,
        prefill_auto_offload_max_memory=None,
        decode_auto_offload_max_memory=None,
        resource_tight_mode=False,
        num_draft_tokens_override=None,
        enable_cuda_graph=True,
        cuda_graph_modules=None,
        cuda_graph_warmup_runs=3,
        cuda_graph_graph_warmup_runs=6,
    )

    spec_decode = meta_info.get("spec_decode", {})
    block_size = int(getattr(runtime, "block_size", spec_decode.get("block_size", 0) or 0))
    spec_mode = spec_decode.get("mode", method)
    draft_capacity = block_size - 1 if spec_mode == "dflash" else block_size
    logger.info(f"Runtime loaded: mode={spec_mode} block_size={block_size} draft_capacity={draft_capacity}")

    completed = _status_by_key(rows)
    for index, case in enumerate(cases, 1):
        key = (method, case.case_id)
        if args.resume and completed.get(key) == "ok":
            continue
        logger.info(f"BENCH_START method={method} case={index}/{len(cases)} case_id={case.case_id}")
        start = time.time()
        row: Dict[str, object] = {
            "method": method,
            "case_id": case.case_id,
            "category": case.category,
            "length_bucket": case.length_bucket,
            "device_id": args.device_id,
            "question": case.question,
            "question_chars": len(case.question),
            "status": "ok",
        }
        try:
            reset_dense_spec_runtime(runtime)
            result = run_dense_spec_generate_once(
                runtime=runtime,
                tokenizer=tokenizer,
                prompt=case.question,
                max_new_tokens=args.max_new_tokens,
                enable_thinking=False,
                repetition_penalty=1.0,
                presence_penalty=0.0,
            )
            output_tokens = int(result.get("output_tokens", 0))
            target_decoder_calls = int(result.get("target_decoder_calls", 0))
            latency_s = float(result.get("latency_s", 0.0))
            draft_tokens_total = int(result.get("draft_tokens_total", 0))
            accepted_drafts_total = int(result.get("accepted_drafts_total", max(0, output_tokens - target_decoder_calls)))
            acceptance_rate = (accepted_drafts_total / draft_tokens_total) if draft_tokens_total > 0 else 0.0
            decode_tps = (output_tokens / latency_s) if latency_s > 0 else 0.0

            row.update({
                "decode_tokens_per_s": decode_tps,
                "acceptance_rate": acceptance_rate,
                "decode_tokens": output_tokens,
                "verify_calls": target_decoder_calls,
                "draft_tokens": draft_tokens_total,
                "accepted_tokens": accepted_drafts_total,
                "elapsed_s": time.time() - start,
            })
        except Exception as exc:
            row["status"] = "failed"
            row["error"] = f"{type(exc).__name__}: {exc}"
            row["traceback"] = traceback.format_exc()
            row["elapsed_s"] = time.time() - start
            logger.exception(f"BENCH_FAILED method={method} case_id={case.case_id}: {exc}")
        _replace_row(rows, row)
        completed[key] = str(row["status"])
        _write_artifacts(rows, args)
        logger.info(
            f"BENCH_DONE method={method} case_id={case.case_id} status={row['status']} "
            f"speed={row.get('decode_tokens_per_s')} acceptance={row.get('acceptance_rate')}"
        )

    del runtime
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _artifact_path(path: str, method: str) -> Path:
    original = Path(path).resolve()
    return original.with_name(f"{original.stem}.{method}{original.suffix}")


def _strip_replaced_child_args(args_list: Sequence[str]) -> List[str]:
    one_value_options = {"--method", "--json", "--csv", "--report", "--device-id"}
    out: List[str] = []
    i = 0
    while i < len(args_list):
        arg = args_list[i]
        if any(arg.startswith(f"{opt}=") for opt in one_value_options):
            i += 1
            continue
        if arg in one_value_options:
            i += 2
            continue
        out.append(arg)
        i += 1
    return out


def _child_command(method: str, device_id: int, args: argparse.Namespace) -> List[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        *_strip_replaced_child_args(sys.argv[1:]),
        "--method", method,
        "--device-id", str(device_id),
        "--json", str(_artifact_path(args.json, method)),
        "--csv", str(_artifact_path(args.csv, method)),
        "--report", str(_artifact_path(args.report, method)),
    ]


def _run_both(args: argparse.Namespace) -> None:
    main_rows = _load_rows(Path(args.json).resolve())

    if len(args.device_ids) >= 2:
        assignments = {"mtp": args.device_ids[0], "dflash": args.device_ids[1]}
        logger.info(f"PARALLEL mode: {assignments}")
        processes = []
        for method, dev_id in assignments.items():
            cmd = _child_command(method, dev_id, args)
            logger.info(f"CHILD_START method={method} device={dev_id}")
            processes.append((method, subprocess.Popen(cmd)))
        failures = []
        for method, proc in processes:
            rc = proc.wait()
            if rc:
                failures.append((method, rc))
            logger.info(f"CHILD_DONE method={method} rc={rc}")
        # Merge
        merged = []
        for method in METHODS:
            method_json = _artifact_path(args.json, method)
            if method_json.exists():
                for r in _load_rows(method_json):
                    _replace_row(merged, r)
        for r in main_rows:
            if r.get("method") not in METHODS:
                merged.append(r)
        merged.sort(key=lambda r: (str(r.get("case_id", "")), str(r.get("method", ""))))
        _write_artifacts(merged, args)
        if failures:
            raise RuntimeError(f"Child process failures: {failures}")
    else:
        logger.warning("Single device — running sequentially")
        for method in METHODS:
            cmd = _child_command(method, args.device_ids[0], args)
            logger.info(f"CHILD_START method={method}")
            subprocess.run(cmd, check=True)
        merged = []
        for method in METHODS:
            method_json = _artifact_path(args.json, method)
            if method_json.exists():
                for r in _load_rows(method_json):
                    _replace_row(merged, r)
        merged.sort(key=lambda r: (str(r.get("case_id", "")), str(r.get("method", ""))))
        _write_artifacts(merged, args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Qwen3.6 27B hmonnx MTP vs DFlash (quantized)."
    )
    parser.add_argument("--method", choices=("mtp", "dflash", "both", "report"), default="both")
    parser.add_argument("--device-id", type=int, default=0, help="GPU device for single-method run")
    parser.add_argument("--device-ids", type=int, nargs="+", default=None,
                        help="GPU devices for both-mode (first=mtp, second=dflash)")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--dtype", type=str, default="fp16")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", action="store_false", dest="resume")
    parser.add_argument("--case-limit", type=int, default=0)
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--category", action="append", choices=CATEGORIES, default=[])
    parser.add_argument("--length-bucket", action="append", choices=LENGTH_BUCKETS, default=[])
    parser.add_argument("--json", default="qwen36_27b_hmonnx_compare_results.json")
    parser.add_argument("--csv", default="qwen36_27b_hmonnx_compare_results.csv")
    parser.add_argument("--report", default="qwen36_27b_hmonnx_compare_report.md")
    parser.add_argument("--mtp-meta", default=MTP_META_DEFAULT)
    parser.add_argument("--dflash-meta", default=DFLASH_META_DEFAULT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device_ids is None:
        args.device_ids = [args.device_id]

    if args.method == "report":
        rows = _load_rows(Path(args.json).resolve())
        _write_artifacts(rows, args)
        return

    if args.method == "both":
        _run_both(args)
        return

    _set_env(args.device_id)
    rows = _load_rows(Path(args.json).resolve())
    cases = _select_cases(args)
    logger.info(
        f"BENCH_PLAN method={args.method} cases={len(cases)} "
        f"max_new_tokens={args.max_new_tokens} device={args.device_id}"
    )
    _run_method(args.method, cases, rows, args)
    _write_artifacts(rows, args)
    logger.info(f"BENCH_COMPLETE method={args.method} json={Path(args.json).resolve()}")


if __name__ == "__main__":
    main()
