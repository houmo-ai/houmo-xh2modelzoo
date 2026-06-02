# Copyright 2025 HOUMO AI
#
# File: build_summary_table.py
# Description:
#   P9 aggregation: compare every quant variant's condensed transcripts against the
#   fp16 golden baseline (baselineA_qwen3_xf_fp16_8k). Per docx, compute char-level
#   and line-level similarity + length ratio, then aggregate per variant into a table.
#   Emits both a JSON and a human-readable markdown table.
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
from difflib import SequenceMatcher
from pathlib import Path

EVAL_ROOT = Path("work_dirs/customer_eval")


def load_outputs(tag):
    """tag -> {docx_stem: text}. Returns {} if dir missing."""
    d = EVAL_ROOT / tag
    if not d.exists():
        return {}
    out = {}
    for f in d.glob("*.txt"):
        out[f.stem] = f.read_text(encoding="utf-8", errors="replace")
    return out


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def speaker_lines(text: str):
    """Count lines that look like speaker turns (contain '：' near start)."""
    n = 0
    for ln in text.splitlines():
        if "：" in ln[:12]:
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-tag", default="baselineA_qwen3_xf_fp16_8k")
    ap.add_argument("--out-json", default="work_dirs/customer_eval/_summary_table.json")
    ap.add_argument("--out-md", default="work_dirs/customer_eval/_summary_table.md")
    args = ap.parse_args()

    base = load_outputs(args.baseline_tag)
    if not base:
        raise SystemExit(f"baseline outputs not found: {args.baseline_tag}")

    # every variant dir except baselines
    variant_tags = sorted(
        d.name for d in EVAL_ROOT.iterdir()
        if d.is_dir() and not d.name.startswith("_") and d.name != args.baseline_tag
        and not d.name.startswith("baseline")
    )

    table = []
    for tag in variant_tags:
        vout = load_outputs(tag)
        if not vout:
            continue
        sims, lenratios, sp_keep = [], [], []
        for stem, btext in base.items():
            vtext = vout.get(stem)
            if vtext is None:
                continue
            sims.append(similarity(btext, vtext))
            lenratios.append(len(vtext) / max(1, len(btext)))
            b_sp = speaker_lines(btext)
            sp_keep.append(speaker_lines(vtext) / max(1, b_sp))
        if not sims:
            continue
        n = len(sims)
        table.append({
            "variant": tag,
            "n_docs": n,
            "avg_similarity": round(sum(sims) / n, 4),
            "min_similarity": round(min(sims), 4),
            "avg_len_ratio": round(sum(lenratios) / n, 3),
            "avg_speaker_keep": round(sum(sp_keep) / n, 3),
        })

    table.sort(key=lambda r: r["avg_similarity"], reverse=True)
    Path(args.out_json).write_text(json.dumps(table, ensure_ascii=False, indent=2))

    # markdown
    lines = [
        f"# Quant Matrix Quality Summary (vs {args.baseline_tag})",
        "",
        "| variant | n | avg_sim | min_sim | len_ratio | speaker_keep |",
        "|---|---|---|---|---|---|",
    ]
    for r in table:
        lines.append(
            f"| {r['variant']} | {r['n_docs']} | {r['avg_similarity']:.4f} | "
            f"{r['min_similarity']:.4f} | {r['avg_len_ratio']:.3f} | {r['avg_speaker_keep']:.3f} |"
        )
    Path(args.out_md).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n[summary] wrote {args.out_json} and {args.out_md}")


if __name__ == "__main__":
    main()
