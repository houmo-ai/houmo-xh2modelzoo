from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List

_HERE = Path(__file__).resolve().parent
_DENSE_DIR = _HERE.parent / "qwen3_5"
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
if str(_DENSE_DIR) not in sys.path:
    sys.path.insert(0, str(_DENSE_DIR))

from qwen3_5_moe_spec_decode_metrics import (  # noqa: E402
    _load_spec_runtime,
    _release_runtime,
    run_moe_spec_metrics,
    run_moe_spec_metrics_with_runtime,
    run_moe_target_baseline_from_spec,
)
from qwen3_5_xh2a_spec_decode_bench import (  # noqa: E402
    CATEGORIES,
    dump_progress_json,
    load_dataset,
    load_progress_rows,
    pick_examples,
    render_markdown,
    summarise,
    validate_shard_args,
)


def run_bench(args: argparse.Namespace) -> None:
    dataset_path = Path(args.dataset)
    validate_shard_args(args.shard_index, args.num_shards)
    cases = load_dataset(
        dataset_path,
        args.limit,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
    )
    think_modes = ["off", "on"] if args.think_mode == "both" else [args.think_mode]

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
        runtime = _load_spec_runtime(args.meta, args.device, args.exec_device)
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
                    result = run_moe_spec_metrics_with_runtime(
                        meta_path=args.meta,
                        runtime=runtime,
                        prompt=prompt,
                        max_new_tokens=args.max_new_tokens,
                        enable_thinking=et,
                        repetition_penalty=args.repetition_penalty,
                        presence_penalty=args.presence_penalty,
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
                except Exception as exc:
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
                print(
                    f"[think={mode}] {idx + 1}/{len(cases)} {case_id} "
                    f"cat={case.get('category')} err={'Y' if row.get('error') else 'N'} "
                    f"t={row['wall_time_s']:.1f}s",
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
            for cat, example_row in pick_examples(rows).items():
                if "baseline_full" in example_row or "baseline_full_error" in example_row:
                    continue
                try:
                    example_row["baseline_full"] = run_moe_target_baseline_from_spec(
                        meta_path=args.meta,
                        prompt=example_row["prompt"],
                        max_new_tokens=args.max_new_tokens,
                        device=args.device,
                        exec_device=args.exec_device,
                        enable_thinking=et,
                        repetition_penalty=args.repetition_penalty,
                        presence_penalty=args.presence_penalty,
                    )
                    print(
                        f"[think={mode}] baseline example ready cat={cat} id={example_row['id']}",
                        flush=True,
                    )
                except Exception as exc:
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
            _release_runtime(runtime)
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
    Path(args.output_md).write_text(md, encoding="utf-8")
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
    parser.add_argument("--meta", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--think-mode", choices=["on", "off", "both"], default="both")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--exec-device", default="cuda:0")
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--presence-penalty", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--shard-index", type=int, default=0, help="0-based shard index")
    parser.add_argument("--num-shards", type=int, default=1, help="split dataset across this many shards")
    args = parser.parse_args()
    run_bench(args)


if __name__ == "__main__":
    main()
