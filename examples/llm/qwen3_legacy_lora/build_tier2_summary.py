#!/usr/bin/env python3
"""Tier-2 aggregation: per-variant avg over 5 docx, vs fp16 baseline.
Reads any tier2_* dir that has 5 txts and builds a ranked table."""
import argparse
import json
from difflib import SequenceMatcher
from pathlib import Path

EVAL_ROOT = Path("work_dirs/customer_eval")
BASELINE = "baselineA_qwen3_xf_fp16_8k"


def load_outputs(tag):
    d = EVAL_ROOT / tag
    if not d.exists():
        return {}
    return {f.stem: f.read_text(encoding="utf-8", errors="replace")
            for f in d.glob("*.txt")}


def sim(a, b):
    return SequenceMatcher(None, a, b).ratio()


def speaker_count(t):
    return sum(1 for ln in t.splitlines() if "：" in ln[:12])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-md", default="work_dirs/customer_eval/_tier2_summary.md")
    ap.add_argument("--out-json", default="work_dirs/customer_eval/_tier2_summary.json")
    args = ap.parse_args()

    base = load_outputs(BASELINE)
    if not base:
        raise SystemExit("baseline not found")

    # only tier2_* dirs that have >=3 docs (don't require full 5 to show progress)
    variants = []
    for d in sorted(EVAL_ROOT.glob("tier2_*")):
        vout = load_outputs(d.name)
        if not vout:
            continue
        n = len(vout)
        sims, lenratios, sp_keep = [], [], []
        for stem, btext in base.items():
            vtext = vout.get(stem)
            if vtext is None:
                continue
            sims.append(sim(btext, vtext))
            lenratios.append(len(vtext) / max(1, len(btext)))
            b_sp = speaker_count(btext)
            sp_keep.append(speaker_count(vtext) / max(1, b_sp))
        if not sims:
            continue
        variants.append({
            "variant": d.name,
            "n_docs": len(sims),
            "avg_sim": round(sum(sims)/len(sims), 4),
            "min_sim": round(min(sims), 4),
            "avg_len_ratio": round(sum(lenratios)/len(lenratios), 3),
            "avg_speaker_keep": round(sum(sp_keep)/len(sp_keep), 3),
        })

    variants.sort(key=lambda r: r["avg_sim"], reverse=True)
    Path(args.out_json).write_text(json.dumps(variants, ensure_ascii=False, indent=2))

    lines = [
        f"# Tier-2 Confirmation Summary (vs {BASELINE}, top-3 variants × 5 docx, CUDA graph ON)",
        "",
        "| variant | n_docs | avg_sim | min_sim | len_ratio | speaker_keep |",
        "|---|---|---|---|---|---|",
    ]
    for r in variants:
        lines.append(
            f"| {r['variant']} | {r['n_docs']} | {r['avg_sim']:.4f} | {r['min_sim']:.4f} "
            f"| {r['avg_len_ratio']:.3f} | {r['avg_speaker_keep']:.3f} |"
        )
    Path(args.out_md).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n[summary] wrote {args.out_md} and {args.out_json}")


if __name__ == "__main__":
    main()
