#!/usr/bin/env python3
"""Qwen3.6 27B float benchmark: MTP vs DFlash (HuggingFace models).

Compares speculative decoding acceptance rate and throughput between
MTP and DFlash using float (bf16) HuggingFace models.

Usage:
    # Both methods on 2 GPUs in parallel:
    python benchmark_qwen36_27b_float_compare.py --method both --device-ids 0 1

    # MTP only:
    python benchmark_qwen36_27b_float_compare.py --method mtp --device-id 0

    # DFlash only:
    python benchmark_qwen36_27b_float_compare.py --method dflash --device-id 1
"""

import argparse
import csv
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

MTP_MODEL_DEFAULT = "weights/Qwen3.6-27B"
DFLASH_MODEL_DEFAULT = "weights/Qwen3.6-27B"
DFLASH_DRAFT_MODEL_DEFAULT = "weights/Qwen3.6-27B-DFlash"


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


def _summary(rows: List[Dict[str, object]]) -> Dict[str, object]:
    speed_rows = [r for r in rows if r.get("decode_tokens_per_s") is not None]
    accept_rows = [r for r in rows if r.get("acceptance_rate") is not None]
    speeds = [float(r["decode_tokens_per_s"]) for r in speed_rows]
    accepts = [float(r["acceptance_rate"]) for r in accept_rows]
    return {
        "ok": len(rows),
        "median_speed": statistics.median(speeds) if speeds else None,
        "mean_speed": statistics.mean(speeds) if speeds else None,
        "median_acceptance": statistics.median(accepts) if accepts else None,
        "mean_acceptance": statistics.mean(accepts) if accepts else None,
    }


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    fields = [
        "method", "case_id", "category", "length_bucket", "status", "device_id",
        "decode_tokens_per_s", "acceptance_rate", "decode_tokens", "verify_calls",
        "draft_tokens", "accepted_tokens", "elapsed_s", "error", "question",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def _write_report(path: Path, rows: List[Dict[str, object]], args: argparse.Namespace, csv_path: Path) -> None:
    lines = [
        "# Qwen3.6 27B Float DFlash vs MTP Benchmark",
        "",
        f"- Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- max_new_tokens: {args.max_new_tokens}",
        f"- dtype: {args.dtype}",
        f"- MTP model: {args.mtp_model}",
        f"- DFlash target: {args.dflash_model}",
        f"- DFlash draft: {args.dflash_draft_model}",
        f"- num_draft_tokens (MTP): {args.num_draft_tokens}",
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


def _run_mtp(cases: List[BenchmarkCase], rows: List[Dict[str, object]], args: argparse.Namespace) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from qwen3_5_spec_decode_metrics import (
        FLOAT_DTYPE_MAP,
        build_compatible_mtp_head,
        install_forced_decode_patch,
        ForcedDecodeContext,
        AfterNormCapture,
        generate_drafts,
        rollback_deltanet_states,
        trim_full_attention_kv,
    )

    DTYPE_MAP = FLOAT_DTYPE_MAP
    dtype = args.dtype
    num_draft_tokens = args.num_draft_tokens
    model_path = args.mtp_model

    logger.info(f"Loading MTP model from {model_path} dtype={dtype}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    num_gpus = torch.cuda.device_count()
    load_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": DTYPE_MAP[dtype] if dtype != "auto" else "auto",
        "device_map": "auto",
    }
    logger.info(f"Loading with device_map=auto num_gpus={num_gpus}")
    model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs).eval()
    mtp_head = build_compatible_mtp_head(model, model_path, dtype)
    install_forced_decode_patch(model)
    device = model.device
    eos = tokenizer.eos_token_id
    logger.info("MTP model loaded")

    def _apply_chat_template(prompt: str) -> str:
        msgs = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)

    completed = _status_by_key(rows)
    for index, case in enumerate(cases, 1):
        key = ("mtp", case.case_id)
        if args.resume and completed.get(key) == "ok":
            continue
        logger.info(f"BENCH_START method=mtp case={index}/{len(cases)} case_id={case.case_id}")
        start = time.time()
        row: Dict[str, object] = {
            "method": "mtp",
            "case_id": case.case_id,
            "category": case.category,
            "length_bucket": case.length_bucket,
            "device_id": args.device_id,
            "question": case.question,
            "question_chars": len(case.question),
            "status": "ok",
        }
        try:
            text = _apply_chat_template(case.question)
            inputs = tokenizer([text], return_tensors="pt").to(device)
            prompt_ids = inputs.input_ids
            prompt_len = prompt_ids.shape[1]

            cap = AfterNormCapture(model)
            cap.reset()
            with torch.no_grad():
                outputs = model(**inputs, use_cache=True)
            past_kv = outputs.past_key_values
            next_tok = outputs.logits[:, -1:, :].argmax(dim=-1)
            prefill_final_hidden = cap.hidden_states[0]
            del outputs
            committed_len = prompt_len

            mtp_kv: dict = {}
            if prompt_len >= 2:
                mtp_head.forward_batch(
                    prefill_final_hidden[:, :-1, :],
                    prompt_ids[:, 1:],
                    positions=torch.arange(prompt_len - 1, device=mtp_head._mtp_device),
                    kv_cache=mtp_kv,
                )

            drafts, _ = generate_drafts(
                mtp_head, prefill_final_hidden[:, -1:, :], next_tok, num_draft_tokens,
                position=committed_len - 1, mtp_kv=mtp_kv,
            )

            tokens: List[int] = []
            accepted_drafts_per_round: List[int] = []
            num_main_fwd = 0

            torch.cuda.synchronize()
            t0 = time.perf_counter()

            while len(tokens) < args.max_new_tokens:
                num_main_fwd += 1
                all_toks = torch.cat([next_tok] + [d.to(device) for d in drafts], dim=1)
                kv_len = past_kv.get_seq_length()
                mask = torch.ones(1, kv_len + num_draft_tokens + 1, device=device, dtype=torch.long)

                with torch.no_grad(), ForcedDecodeContext():
                    cap.reset()
                    out = model(input_ids=all_toks, attention_mask=mask, past_key_values=past_kv, use_cache=True)
                past_kv = out.past_key_values
                hidden_all = cap.hidden_states[0]

                accepted_count = 0
                for idx in range(num_draft_tokens):
                    verified = out.logits[:, idx, :].argmax(dim=-1, keepdim=True)
                    if verified.item() != drafts[idx].item():
                        break
                    accepted_count += 1
                accepted_drafts_per_round.append(accepted_count)

                if accepted_count == num_draft_tokens:
                    tokens.append(next_tok.item())
                    for draft in drafts:
                        tokens.append(draft.item())
                    old_committed = committed_len
                    committed_len += num_draft_tokens + 1
                    if any(d.item() == eos for d in drafts) or len(tokens) >= args.max_new_tokens:
                        break
                    for idx in range(num_draft_tokens):
                        mtp_head.forward_step(
                            hidden_all[:, idx:idx+1, :], drafts[idx],
                            position=old_committed + idx, kv_cache=mtp_kv,
                        )
                    bonus_tok = out.logits[:, num_draft_tokens, :].argmax(dim=-1, keepdim=True)
                    next_tok = bonus_tok
                    drafts, _ = generate_drafts(
                        mtp_head, hidden_all[:, num_draft_tokens:num_draft_tokens+1, :],
                        bonus_tok, num_draft_tokens, position=committed_len - 1, mtp_kv=mtp_kv,
                    )
                else:
                    tokens.append(next_tok.item())
                    for idx in range(accepted_count):
                        tokens.append(drafts[idx].item())
                    committed_len += accepted_count + 1
                    if next_tok.item() == eos or len(tokens) >= args.max_new_tokens:
                        break
                    if any(drafts[idx].item() == eos for idx in range(accepted_count)):
                        break
                    rollback_deltanet_states(past_kv, rollback_idx=accepted_count)
                    trim_full_attention_kv(past_kv, n_trim=num_draft_tokens - accepted_count)
                    for idx in range(accepted_count):
                        mtp_head.forward_step(
                            hidden_all[:, idx:idx+1, :], drafts[idx],
                            position=committed_len - accepted_count - 1 + idx, kv_cache=mtp_kv,
                        )
                    replacement = out.logits[:, accepted_count, :].argmax(dim=-1, keepdim=True)
                    next_tok = replacement
                    drafts, _ = generate_drafts(
                        mtp_head, hidden_all[:, accepted_count:accepted_count+1, :],
                        replacement, num_draft_tokens, position=committed_len - 1, mtp_kv=mtp_kv,
                    )

            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            cap.remove()
            tokens = tokens[:args.max_new_tokens]

            total_drafts = len(accepted_drafts_per_round) * num_draft_tokens
            total_accepted = sum(accepted_drafts_per_round)
            acceptance_rate = (total_accepted / total_drafts) if total_drafts > 0 else 0.0
            decode_tps = (len(tokens) / elapsed) if elapsed > 0 else 0.0

            row.update({
                "decode_tokens_per_s": decode_tps,
                "acceptance_rate": acceptance_rate,
                "decode_tokens": len(tokens),
                "verify_calls": num_main_fwd,
                "draft_tokens": total_drafts,
                "accepted_tokens": total_accepted,
                "elapsed_s": elapsed,
            })
        except Exception as exc:
            row["status"] = "failed"
            row["error"] = f"{type(exc).__name__}: {exc}"
            row["traceback"] = traceback.format_exc()
            row["elapsed_s"] = time.time() - start
            logger.exception(f"BENCH_FAILED method=mtp case_id={case.case_id}: {exc}")
        finally:
            torch.cuda.empty_cache()
        _replace_row(rows, row)
        completed[key] = str(row["status"])
        _write_artifacts(rows, args)
        logger.info(
            f"BENCH_DONE method=mtp case_id={case.case_id} status={row['status']} "
            f"speed={row.get('decode_tokens_per_s')} acceptance={row.get('acceptance_rate')}"
        )


def _run_dflash(cases: List[BenchmarkCase], rows: List[Dict[str, object]], args: argparse.Namespace) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from qwen3_5_spec_decode_metrics import (
        FLOAT_DTYPE_MAP,
        resolve_dflash_block_size,
        _is_multi_gpu_dispatched,
    )

    try:
        import dflash.qwen3_5_transformers_benchmark as dflash_benchmark
        from dflash.model import DFlashDraftModel
        from dflash.qwen3_5_transformers_benchmark import (
            generate_dflash_qwen3_5_forced,
        )
    except ImportError as e:
        raise RuntimeError(f"dflash package not importable: {e}. Set PYTHONPATH.") from e

    DTYPE_MAP = FLOAT_DTYPE_MAP
    dtype = args.dtype
    block_size = resolve_dflash_block_size(args.dflash_draft_model, args.dflash_block_size)
    draft_capacity = max(block_size - 1, 0)
    torch_dtype = DTYPE_MAP[dtype] if dtype != "auto" else torch.bfloat16

    logger.info(f"Loading DFlash target from {args.dflash_model} dtype={dtype}")
    target = AutoModelForCausalLM.from_pretrained(
        args.dflash_model,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map="auto",
    ).eval()
    logger.info(f"Loading DFlash draft from {args.dflash_draft_model}")
    draft = DFlashDraftModel.from_pretrained(
        args.dflash_draft_model,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map="auto",
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.dflash_model, trust_remote_code=True)
    logger.info(f"DFlash models loaded, block_size={block_size}")

    completed = _status_by_key(rows)
    for index, case in enumerate(cases, 1):
        key = ("dflash", case.case_id)
        if args.resume and completed.get(key) == "ok":
            continue
        logger.info(f"BENCH_START method=dflash case={index}/{len(cases)} case_id={case.case_id}")
        start = time.time()
        row: Dict[str, object] = {
            "method": "dflash",
            "case_id": case.case_id,
            "category": case.category,
            "length_bucket": case.length_bucket,
            "device_id": args.device_id,
            "question": case.question,
            "question_chars": len(case.question),
            "status": "ok",
        }
        try:
            spec = generate_dflash_qwen3_5_forced(
                draft,
                target,
                tokenizer,
                case.question,
                max_new_tokens=args.max_new_tokens,
                temperature=0.0,
                enable_thinking=False,
                block_size=block_size,
            )
            output_tokens = int(spec.num_output_tokens)
            accepted_drafts_per_round = [max(int(v) - 1, 0) for v in spec.acceptance_lengths]
            total_rounds = len(accepted_drafts_per_round)
            total_accepted = sum(accepted_drafts_per_round)
            total_drafts = total_rounds * draft_capacity
            acceptance_rate = (total_accepted / total_drafts) if total_drafts > 0 else 0.0
            elapsed = time.time() - start
            decode_tps = (output_tokens / elapsed) if elapsed > 0 else 0.0

            row.update({
                "decode_tokens_per_s": decode_tps,
                "acceptance_rate": acceptance_rate,
                "decode_tokens": output_tokens,
                "verify_calls": total_rounds,
                "draft_tokens": total_drafts,
                "accepted_tokens": total_accepted,
                "elapsed_s": elapsed,
            })
        except Exception as exc:
            row["status"] = "failed"
            row["error"] = f"{type(exc).__name__}: {exc}"
            row["traceback"] = traceback.format_exc()
            row["elapsed_s"] = time.time() - start
            logger.exception(f"BENCH_FAILED method=dflash case_id={case.case_id}: {exc}")
        _replace_row(rows, row)
        completed[key] = str(row["status"])
        _write_artifacts(rows, args)
        logger.info(
            f"BENCH_DONE method=dflash case_id={case.case_id} status={row['status']} "
            f"speed={row.get('decode_tokens_per_s')} acceptance={row.get('acceptance_rate')}"
        )


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
        merged = []
        for method in METHODS:
            method_json = _artifact_path(args.json, method)
            if method_json.exists():
                for r in _load_rows(method_json):
                    _replace_row(merged, r)
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
        description="Benchmark Qwen3.6 27B float MTP vs DFlash."
    )
    parser.add_argument("--method", choices=("mtp", "dflash", "both", "report"), default="both")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--device-ids", type=int, nargs="+", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--num-draft-tokens", type=int, default=4)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32", "auto"])
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", action="store_false", dest="resume")
    parser.add_argument("--case-limit", type=int, default=0)
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--category", action="append", choices=CATEGORIES, default=[])
    parser.add_argument("--length-bucket", action="append", choices=LENGTH_BUCKETS, default=[])
    parser.add_argument("--json", default="qwen36_27b_float_compare_results.json")
    parser.add_argument("--csv", default="qwen36_27b_float_compare_results.csv")
    parser.add_argument("--report", default="qwen36_27b_float_compare_report.md")
    parser.add_argument("--mtp-model", default=MTP_MODEL_DEFAULT)
    parser.add_argument("--dflash-model", default=DFLASH_MODEL_DEFAULT)
    parser.add_argument("--dflash-draft-model", default=DFLASH_DRAFT_MODEL_DEFAULT)
    parser.add_argument("--dflash-block-size", type=int, default=None)
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
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    rows = _load_rows(Path(args.json).resolve())
    cases = _select_cases(args)
    logger.info(
        f"BENCH_PLAN method={args.method} cases={len(cases)} "
        f"max_new_tokens={args.max_new_tokens} device={args.device_id}"
    )

    if args.method == "mtp":
        _run_mtp(cases, rows, args)
    elif args.method == "dflash":
        _run_dflash(cases, rows, args)

    _write_artifacts(rows, args)
    logger.info(f"BENCH_COMPLETE method={args.method} json={Path(args.json).resolve()}")


if __name__ == "__main__":
    main()
