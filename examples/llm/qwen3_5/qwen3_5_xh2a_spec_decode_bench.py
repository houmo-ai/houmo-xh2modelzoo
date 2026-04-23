# Copyright (c) XHQuant contributors.
"""
Qwen3.5 xh2a spec-decode large-scale benchmark.

Reads a JSONL dataset of prompts (see `build_spec_decode_dataset.py`) and for
each prompt runs:
  - dense baseline (non-draft, greedy, via `run_dense_target_baseline_from_spec`)
  - spec-decode (via `run_dense_spec_metrics`)

Runs are done for each selected `--think-mode` ∈ {on, off, both}, aggregating
per-category statistics plus one representative example per category.

Outputs a JSON file + a Markdown report; optionally uploads the Markdown to
Feishu Docs via `lark-doc` skill (done outside of this script).

Example
-------
python examples/llm/qwen3_5/qwen3_5_xh2a_spec_decode_bench.py \
    --meta work_dirs/qwen3_5_4b_mtp_k4_8k_export/meta.json \
    --dataset examples/llm/qwen3_5/spec_decode_eval_prompts.jsonl \
    --output-md work_dirs/qwen3_5_4b_mtp_bench.md \
    --output-json work_dirs/qwen3_5_4b_mtp_bench.json \
    --think-mode both --max-new-tokens 8192 --limit 0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from qwen3_5_spec_decode_metrics import (  # noqa: E402
    Qwen3_5SpecDecodeONNXModel,
    load_dense_runtime_from_meta,
    release_dense_runtime,
    reset_dense_spec_runtime,
    run_dense_spec_metrics,
    run_dense_spec_metrics_with_runtime,
    run_dense_target_baseline_from_spec,
)


CATEGORIES = ["humanities", "social", "science_tech", "math", "tool_calls", "coding"]


def parse_auto_offload_max_memory(max_memory_json: Optional[str]):
    if max_memory_json is None or max_memory_json.strip() == "":
        return None
    parsed = json.loads(max_memory_json)
    if not isinstance(parsed, dict):
        raise ValueError("auto_offload_max_memory must be a JSON object")
    fixed = {}
    for key, value in parsed.items():
        try:
            fixed[int(key)] = value
        except Exception:
            fixed[key] = value
    return fixed


def load_dataset(
    path: Path,
    limit: int,
    *,
    shard_index: int = 0,
    num_shards: int = 1,
) -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            cases.append(json.loads(line))
    if limit and limit > 0:
        # keep category balance: round-robin per category
        per_cat: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for c in cases:
            per_cat[c.get("category", "other")].append(c)
        out: List[Dict[str, Any]] = []
        i = 0
        while len(out) < limit and any(per_cat.values()):
            for cat in CATEGORIES:
                bucket = per_cat.get(cat, [])
                if bucket and len(out) < limit:
                    out.append(bucket.pop(0))
            i += 1
        cases = out
    if num_shards > 1:
        cases = [case for idx, case in enumerate(cases) if idx % num_shards == shard_index]
    return cases


def validate_shard_args(shard_index: int, num_shards: int) -> None:
    if num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("--shard-index must satisfy 0 <= shard-index < num-shards")


def summarise(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-category statistics from per-case result rows."""
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        if r.get("error"):
            continue
        buckets[r["category"]].append(r)

    overall_metrics: Dict[str, List[float]] = defaultdict(list)
    per_cat_summary: Dict[str, Dict[str, Any]] = {}
    for cat in CATEGORIES:
        items = buckets.get(cat, [])
        if not items:
            continue
        sums = defaultdict(list)
        for it in items:
            spec = it.get("spec", {})
            base = it.get("baseline", {})
            for k in (
                "target_prefill_calls",
                "target_decoder_calls",
                "mtp_prefill_calls",
                "mtp_decode_calls",
                "dflash_prefill_calls",
                "dflash_decode_calls",
                "output_tokens",
                "overall_acceptance_rate",
                "avg_accepted_per_round",
                "num_rounds",
            ):
                if k in spec and spec[k] is not None:
                    sums[f"spec_{k}"].append(float(spec[k]))
            for k in ("target_prefill_calls", "target_decoder_calls"):
                if k in base and base[k] is not None:
                    sums[f"base_{k}"].append(float(base[k]))
        means = {k: (sum(v) / len(v)) for k, v in sums.items() if v}
        per_cat_summary[cat] = {"n": len(items), **means}
        for k, v in means.items():
            overall_metrics[k].append(v)

    overall = {k: (sum(v) / len(v)) for k, v in overall_metrics.items() if v}
    return {"per_category": per_cat_summary, "overall": overall}


def pick_examples(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """One representative (first successful) case per category."""
    chosen: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        if r.get("error"):
            continue
        cat = r.get("category")
        if cat and cat not in chosen:
            chosen[cat] = r
    return chosen


def dump_progress_json(
    output_json: str | None,
    *,
    meta_path: str,
    dataset: str,
    shard_index: int = 0,
    num_shards: int = 1,
    per_mode_rows: Dict[str, List[Dict[str, Any]]],
    think_modes_done: List[str],
    current_mode: str | None = None,
    current_case_id: str | None = None,
    status: str = "running",
    summary_per_mode: Dict[str, Dict[str, Any]] | None = None,
) -> None:
    if not output_json:
        return
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "meta_path": meta_path,
        "dataset": dataset,
        "shard_index": shard_index,
        "num_shards": num_shards,
        "status": status,
        "current_mode": current_mode,
        "current_case_id": current_case_id,
        "think_modes_done": think_modes_done,
        "rows_per_mode": per_mode_rows,
    }
    if summary_per_mode is not None:
        payload["summary_per_mode"] = summary_per_mode
    with open(output_json, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def load_progress_rows(
    output_json: str | None,
    *,
    meta_path: str,
    dataset: str,
    shard_index: int = 0,
    num_shards: int = 1,
    think_modes: List[str] | None = None,
) -> Dict[str, List[Dict[str, Any]]]:
    def _is_retryable_resource_error(row: Any) -> bool:
        if not isinstance(row, dict):
            return False
        error = str(row.get("error") or "").lower()
        if not error:
            return False
        retry_markers = (
            "cuda out of memory",
            "outofmemoryerror",
            "cublas_status_alloc_failed",
            "hip out of memory",
        )
        return any(marker in error for marker in retry_markers)

    def _norm(path_like: Any) -> str | None:
        if not isinstance(path_like, str) or not path_like:
            return None
        return os.path.normpath(os.path.abspath(path_like))

    if not output_json:
        return {}
    path = Path(output_json)
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except Exception:
        return {}
    if _norm(payload.get("meta_path")) != _norm(meta_path):
        return {}
    if _norm(payload.get("dataset")) != _norm(dataset):
        return {}
    if payload.get("shard_index", 0) != shard_index or payload.get("num_shards", 1) != num_shards:
        return {}
    rows_per_mode = payload.get("rows_per_mode")
    if not isinstance(rows_per_mode, dict):
        return {}
    allowed = set(think_modes or rows_per_mode.keys())
    restored: Dict[str, List[Dict[str, Any]]] = {}
    for mode, rows in rows_per_mode.items():
        if mode not in allowed or not isinstance(rows, list):
            continue
        restored[mode] = [row for row in rows if not _is_retryable_resource_error(row)]
    return restored


def render_markdown(
    *,
    meta_path: str,
    dataset_path: str,
    think_modes: List[str],
    max_new_tokens: int,
    shard_index: int = 0,
    num_shards: int = 1,
    per_mode_summary: Dict[str, Dict[str, Any]],
    per_mode_examples: Dict[str, Dict[str, Dict[str, Any]]],
    per_mode_rows: Dict[str, List[Dict[str, Any]]],
) -> str:
    lines: List[str] = []
    lines.append(f"# Qwen3.5 xh2a spec-decode benchmark")
    lines.append("")
    lines.append(f"- **meta**: `{meta_path}`")
    lines.append(f"- **dataset**: `{dataset_path}`")
    if num_shards > 1:
        lines.append(f"- **shard**: {shard_index}/{num_shards}")
    lines.append(f"- **think modes**: {think_modes}")
    lines.append(f"- **max_new_tokens**: {max_new_tokens}")
    lines.append("")

    for mode in think_modes:
        summary = per_mode_summary.get(mode, {})
        rows = per_mode_rows.get(mode, [])
        ok = [r for r in rows if not r.get("error")]
        fail = [r for r in rows if r.get("error")]
        lines.append(f"## Think mode: `{mode}` (n_ok={len(ok)}, n_fail={len(fail)})")
        lines.append("")
        overall = summary.get("overall", {})
        if overall:
            lines.append("### Overall averages (mean over categories)")
            lines.append("")
            lines.append("| Metric | Mean |")
            lines.append("| --- | --- |")
            for k in sorted(overall):
                lines.append(f"| {k} | {overall[k]:.3f} |")
            lines.append("")

        per_cat = summary.get("per_category", {})
        if per_cat:
            lines.append("### Per-category averages")
            lines.append("")
            header_keys = [
                "n",
                "base_target_decoder_calls",
                "spec_target_decoder_calls",
                "spec_mtp_prefill_calls",
                "spec_mtp_decode_calls",
                "spec_dflash_prefill_calls",
                "spec_dflash_decode_calls",
                "spec_num_rounds",
                "spec_overall_acceptance_rate",
                "spec_avg_accepted_per_round",
                "spec_output_tokens",
            ]
            present = [k for k in header_keys if any(k in v for v in per_cat.values())]
            lines.append("| category | " + " | ".join(present) + " |")
            lines.append("| --- | " + " | ".join(["---"] * len(present)) + " |")
            for cat in CATEGORIES:
                if cat not in per_cat:
                    continue
                v = per_cat[cat]
                row = [cat]
                for k in present:
                    val = v.get(k)
                    if val is None:
                        row.append("-")
                    elif isinstance(val, float):
                        row.append(f"{val:.2f}")
                    else:
                        row.append(str(val))
                lines.append("| " + " | ".join(row) + " |")
            lines.append("")

        examples = per_mode_examples.get(mode, {})
        if examples:
            lines.append("### Representative case per category (baseline vs draft)")
            lines.append("")
            for cat in CATEGORIES:
                ex = examples.get(cat)
                if not ex:
                    continue
                lines.append(f"#### {cat} — `{ex.get('id', '')}` (len={ex.get('char_length')})")
                lines.append("")
                lines.append("**Prompt**")
                lines.append("")
                lines.append("```text")
                lines.append(ex["prompt"])
                lines.append("```")
                lines.append("")
                base = ex.get("baseline_full") or ex.get("baseline", {})
                baseline_counts = ex.get("baseline", {})
                spec = ex.get("spec", {})
                lines.append(
                    f"- baseline same-output counts: decoder_calls={baseline_counts.get('target_decoder_calls')}, "
                    f"prefill_calls={baseline_counts.get('target_prefill_calls')}, "
                    f"output_tokens={baseline_counts.get('output_tokens')}"
                )
                if ex.get("baseline_full"):
                    lines.append(
                        f"- baseline actual run: decoder_calls={base.get('target_decoder_calls')}, "
                        f"prefill_calls={base.get('target_prefill_calls')}, "
                        f"output_tokens={base.get('output_tokens')}"
                    )
                draft_extra = ""
                if "mtp_decode_calls" in spec:
                    draft_extra = (
                        f", mtp_prefill={spec.get('mtp_prefill_calls')}, "
                        f"mtp_decode={spec.get('mtp_decode_calls')}"
                    )
                elif "dflash_decode_calls" in spec:
                    draft_extra = (
                        f", dflash_prefill={spec.get('dflash_prefill_calls')}, "
                        f"dflash_decode={spec.get('dflash_decode_calls')}"
                    )
                lines.append(
                    f"- spec: target_decoder_calls={spec.get('target_decoder_calls')}, "
                    f"output_tokens={spec.get('output_tokens')}, "
                    f"overall_accept_rate={spec.get('overall_acceptance_rate'):.3f}"
                    f"{draft_extra}"
                )
                lines.append("")
                lines.append("<details><summary>baseline output</summary>")
                lines.append("")
                lines.append("```text")
                lines.append(base.get("text", ""))
                lines.append("```")
                lines.append("</details>")
                lines.append("")
                lines.append("<details><summary>spec output</summary>")
                lines.append("")
                lines.append("```text")
                lines.append(spec.get("text", ""))
                lines.append("```")
                lines.append("</details>")
                lines.append("")

        if fail:
            lines.append(f"### Failures ({len(fail)})")
            lines.append("")
            for r in fail[:10]:
                lines.append(f"- `{r.get('id')}` [{r.get('category')}]: {r.get('error')}")
            lines.append("")

    return "\n".join(lines) + "\n"


def run_bench(args: argparse.Namespace) -> None:
    dataset_path = Path(args.dataset)
    validate_shard_args(args.shard_index, args.num_shards)
    auto_offload_max_memory = parse_auto_offload_max_memory(args.auto_offload_max_memory)
    prefill_auto_offload_max_memory = parse_auto_offload_max_memory(
        args.prefill_auto_offload_max_memory
    )
    decode_auto_offload_max_memory = parse_auto_offload_max_memory(
        args.decode_auto_offload_max_memory
    )
    cases = load_dataset(
        dataset_path,
        args.limit,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
    )
    think_modes: List[str]
    if args.think_mode == "both":
        think_modes = ["off", "on"]
    else:
        think_modes = [args.think_mode]

    per_mode_rows: Dict[str, List[Dict[str, Any]]] = {}
    per_mode_summary: Dict[str, Dict[str, Any]] = {}
    per_mode_examples: Dict[str, Dict[str, Dict[str, Any]]] = {}
    existing_rows = load_progress_rows(
        args.output_json,
        meta_path=args.meta,
        dataset=args.dataset,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
        think_modes=think_modes,
    )

    for mode in think_modes:
        et = mode == "on"
        runtime = None
        tokenizer = None
        model_name = args.meta
        rows: List[Dict[str, Any]] = list(existing_rows.get(mode, []))
        seen_case_ids = {
            row.get("id")
            for row in rows
            if isinstance(row, dict) and row.get("id") is not None
        }
        pending_cases = [
            case
            for idx, case in enumerate(cases)
            if case.get("id", f"case_{idx:04d}") not in seen_case_ids
        ]
        if rows:
            print(
                f"[think={mode}] resuming with {len(rows)} saved rows; "
                f"{len(pending_cases)} cases remaining",
                flush=True,
            )
        if args.baseline_meta is None:
            runtime, tokenizer, meta_info = load_dense_runtime_from_meta(
                meta_path=args.meta,
                dtype=args.dtype,
                device=args.device,
                exec_device=args.exec_device,
                auto_offload_max_memory=auto_offload_max_memory,
                prefill_auto_offload_max_memory=prefill_auto_offload_max_memory,
                decode_auto_offload_max_memory=decode_auto_offload_max_memory,
            )
            if not isinstance(runtime, Qwen3_5SpecDecodeONNXModel):
                raise TypeError(f"{args.meta} is not a dense spec-decode runtime.")
            model_name = meta_info.get("model_name", args.meta)
        dump_progress_json(
            args.output_json,
            meta_path=args.meta,
            dataset=args.dataset,
            shard_index=args.shard_index,
            num_shards=args.num_shards,
            per_mode_rows={**per_mode_rows, mode: rows},
            think_modes_done=list(per_mode_rows.keys()),
            current_mode=mode,
            status="starting",
        )
        try:
            for idx, case in enumerate(cases):
                case_id = case.get("id", f"case_{idx:04d}")
                if case_id in seen_case_ids:
                    continue
                prompt = case["prompt"]
                print(
                    f"[think={mode}] start {idx + 1}/{len(cases)} {case_id} cat={case.get('category')}",
                    flush=True,
                )
                dump_progress_json(
                    args.output_json,
                    meta_path=args.meta,
                    dataset=args.dataset,
                    shard_index=args.shard_index,
                    num_shards=args.num_shards,
                    per_mode_rows={**per_mode_rows, mode: rows},
                    think_modes_done=list(per_mode_rows.keys()),
                    current_mode=mode,
                    current_case_id=case_id,
                    status="running",
                )
                t0 = time.time()
                try:
                    if runtime is not None:
                        result = run_dense_spec_metrics_with_runtime(
                            runtime=runtime,
                            tokenizer=tokenizer,
                            model_name=model_name,
                            prompt=prompt,
                            max_new_tokens=args.max_new_tokens,
                            enable_thinking=et,
                            repetition_penalty=args.repetition_penalty,
                            presence_penalty=args.presence_penalty,
                        )
                    else:
                        result = run_dense_spec_metrics(
                            meta_path=args.meta,
                            baseline_meta_path=args.baseline_meta,
                            prompt=prompt,
                            max_new_tokens=args.max_new_tokens,
                            dtype=args.dtype,
                            device=args.device,
                            exec_device=args.exec_device,
                            enable_thinking=et,
                            repetition_penalty=args.repetition_penalty,
                            presence_penalty=args.presence_penalty,
                            auto_offload_max_memory=auto_offload_max_memory,
                            prefill_auto_offload_max_memory=prefill_auto_offload_max_memory,
                            decode_auto_offload_max_memory=decode_auto_offload_max_memory,
                        )
                    row = {
                        "id": case_id,
                        "category": case.get("category"),
                        "char_length": case.get("char_length"),
                        "prompt": prompt,
                        "enable_thinking": et,
                        "baseline": result.get("baseline", {}),
                        "spec": result.get("speculative", {}),
                        "wall_time_s": time.time() - t0,
                    }
                except Exception as exc:  # pragma: no cover
                    row = {
                        "id": case_id,
                        "category": case.get("category"),
                        "char_length": case.get("char_length"),
                        "prompt": prompt,
                        "enable_thinking": et,
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(limit=5),
                        "wall_time_s": time.time() - t0,
                    }
                    print(
                        f"[think={mode}] case failed {case_id}: {row['error']}\n{row['traceback']}",
                        flush=True,
                    )
                rows.append(row)
                seen_case_ids.add(case_id)
                done = idx + 1
                total = len(cases)
                print(
                    f"[think={mode}] {done}/{total} {case_id} cat={case.get('category')} "
                    f"err={'Y' if row.get('error') else 'N'} t={row['wall_time_s']:.1f}s",
                    flush=True,
                )
                dump_progress_json(
                    args.output_json,
                    meta_path=args.meta,
                    dataset=args.dataset,
                    shard_index=args.shard_index,
                    num_shards=args.num_shards,
                    per_mode_rows={**per_mode_rows, mode: rows},
                    think_modes_done=list(per_mode_rows.keys()),
                    current_mode=mode,
                    status="running",
                )
            if runtime is not None:
                for cat, example_row in pick_examples(rows).items():
                    if "baseline_full" in example_row or "baseline_full_error" in example_row:
                        continue
                    try:
                        reset_dense_spec_runtime(runtime)
                        example_row["baseline_full"] = run_dense_target_baseline_from_spec(
                            runtime,
                            tokenizer,
                            example_row["prompt"],
                            args.max_new_tokens,
                            enable_thinking=et,
                            repetition_penalty=args.repetition_penalty,
                            presence_penalty=args.presence_penalty,
                        )
                        print(
                            f"[think={mode}] baseline example ready cat={cat} id={example_row['id']}",
                            flush=True,
                        )
                    except Exception as exc:  # pragma: no cover
                        example_row["baseline_full_error"] = f"{type(exc).__name__}: {exc}"
                        print(
                            f"[think={mode}] baseline example traceback {example_row['id']}: "
                            f"{traceback.format_exc(limit=5)}",
                            flush=True,
                        )
                        print(
                            f"[think={mode}] baseline example failed cat={cat} id={example_row['id']} err={example_row['baseline_full_error']}",
                            flush=True,
                        )
                dump_progress_json(
                    args.output_json,
                    meta_path=args.meta,
                    dataset=args.dataset,
                    shard_index=args.shard_index,
                    num_shards=args.num_shards,
                    per_mode_rows={**per_mode_rows, mode: rows},
                    think_modes_done=list(per_mode_rows.keys()),
                    current_mode=mode,
                    status="running",
                )
        finally:
            if runtime is not None:
                release_dense_runtime(runtime)
        per_mode_rows[mode] = rows
        per_mode_summary[mode] = summarise(rows)
        per_mode_examples[mode] = pick_examples(rows)

    md = render_markdown(
        meta_path=args.meta,
        dataset_path=args.dataset,
        think_modes=think_modes,
        max_new_tokens=args.max_new_tokens,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
        per_mode_summary=per_mode_summary,
        per_mode_examples=per_mode_examples,
        per_mode_rows=per_mode_rows,
    )
    Path(args.output_md).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_md, "w", encoding="utf-8") as fh:
        fh.write(md)
    print(f"wrote {args.output_md}")

    dump_progress_json(
        args.output_json,
        meta_path=args.meta,
        dataset=args.dataset,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
        per_mode_rows=per_mode_rows,
        think_modes_done=think_modes,
        status="completed",
        summary_per_mode=per_mode_summary,
    )
    if args.output_json:
        print(f"wrote {args.output_json}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta", required=True, help="path to spec-decode meta.json")
    parser.add_argument("--baseline-meta", default=None, help="optional dense baseline meta")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--think-mode", choices=["on", "off", "both"], default="both")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--exec-device", default="cuda:0")
    parser.add_argument("--auto-offload-max-memory", dest="auto_offload_max_memory", default=None)
    parser.add_argument(
        "--prefill-auto-offload-max-memory",
        dest="prefill_auto_offload_max_memory",
        default=None,
    )
    parser.add_argument(
        "--decode-auto-offload-max-memory",
        dest="decode_auto_offload_max_memory",
        default=None,
    )
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--presence-penalty", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=0, help="if >0, keep only this many balanced cases")
    parser.add_argument("--shard-index", type=int, default=0, help="0-based shard index")
    parser.add_argument("--num-shards", type=int, default=1, help="split dataset across this many shards")
    args = parser.parse_args()
    run_bench(args)


if __name__ == "__main__":
    main()
