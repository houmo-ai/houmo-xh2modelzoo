#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GLM-OCR OCRBench 评测对比：浮点 vs HMONNX

读取两侧的 ocrbench_scores.json，打印对比表格。

用法:
    python examples/llm/glm_ocr/run_ocrbench_compare.py \
        --fp_dir ./work_dirs/glm_ocr_ocrbench_fp \
        --hmonnx_dir ./work_dirs/glm_ocr_ocrbench_hmonnx
"""

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

CATEGORY_ORDER = [
    "Regular Text Recognition",
    "Irregular Text Recognition",
    "Artistic Text Recognition",
    "Handwriting Recognition",
    "Digit String Recognition",
    "Non-Semantic Text Recognition",
    "Scene Text-centric VQA",
    "Doc-oriented VQA",
    "Key Information Extraction",
    "Handwritten Mathematical Expression Recognition",
]


def load_scores(scores_path: str):
    if not os.path.exists(scores_path):
        return None
    with open(scores_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _fmt(v):
    if v is None:
        return "N/A"
    return f"{v:.4f}"


def _fmt_frac(correct, total):
    if correct is None or total is None:
        return "N/A"
    return f"{correct}/{total}"


def print_comparison(fp_scores, hmonnx_scores):
    print(f"\n{'=' * 90}")
    print("GLM-OCR OCRBench: Float vs HMONNX Comparison")
    print(f"{'=' * 90}")
    print(f"{'Category':<50} {'FP':>10} {'HMONNX':>10} {'Delta':>10}")
    print("-" * 90)

    for cat in CATEGORY_ORDER:
        fp_acc = fp_scores.get(cat, {}).get("accuracy")
        hm_acc = hmonnx_scores.get(cat, {}).get("accuracy")
        fp_str = _fmt(fp_acc)
        hm_str = _fmt(hm_acc)
        if fp_acc is not None and hm_acc is not None:
            delta = f"{hm_acc - fp_acc:+.4f}"
        else:
            delta = "N/A"
        print(f"{cat:<50} {fp_str:>10} {hm_str:>10} {delta:>10}")

    print("-" * 90)

    fp_ov = fp_scores.get("__overall__", {})
    hm_ov = hmonnx_scores.get("__overall__", {})
    fp_c, fp_t = fp_ov.get("correct"), fp_ov.get("total")
    hm_c, hm_t = hm_ov.get("correct"), hm_ov.get("total")
    fp_acc = fp_ov.get("accuracy")
    hm_acc = hm_ov.get("accuracy")

    fp_frac = _fmt_frac(fp_c, fp_t)
    hm_frac = _fmt_frac(hm_c, hm_t)

    if fp_acc is not None and hm_acc is not None:
        delta = f"{hm_acc - fp_acc:+.4f}"
    else:
        delta = "N/A"
    print(f"{'OVERALL':<50} {_fmt(fp_acc):>10} {_fmt(hm_acc):>10} {delta:>10}")
    print()
    print(f"  FP Score:     {fp_frac}")
    print(f"  HMONNX Score: {hm_frac}")
    if fp_c is not None and hm_c is not None:
        print(f"  Score Delta:  {hm_c - fp_c:+d}")
    print(f"{'=' * 90}")


def print_per_sample_diff(fp_dir, hmonnx_dir, max_show=20):
    """Show samples where FP and HMONNX disagree."""
    fp_results_path = os.path.join(fp_dir, "ocrbench_results.json")
    hm_results_path = os.path.join(hmonnx_dir, "ocrbench_results.json")
    if not os.path.exists(fp_results_path) or not os.path.exists(hm_results_path):
        return

    with open(fp_results_path, "r") as f:
        fp_results = {r["index"]: r for r in json.load(f)}
    with open(hm_results_path, "r") as f:
        hm_results = {r["index"]: r for r in json.load(f)}

    common_indices = sorted(set(fp_results.keys()) & set(hm_results.keys()))
    diffs = []
    for idx in common_indices:
        fp_r = fp_results[idx]
        hm_r = hm_results[idx]
        if fp_r["correct"] != hm_r["correct"]:
            diffs.append((idx, fp_r, hm_r))

    if not diffs:
        print("\nAll common samples have the same correctness.")
        return

    print(f"\n{'=' * 90}")
    print(f"Samples with different results ({len(diffs)} total, showing up to {max_show})")
    print(f"{'=' * 90}")
    for idx, fp_r, hm_r in diffs[:max_show]:
        fp_mark = "OK" if fp_r["correct"] else "WRONG"
        hm_mark = "OK" if hm_r["correct"] else "WRONG"
        print(f"\n  [{idx}] {fp_r['category']}")
        print(f"    Q: {fp_r['question'][:80]}")
        print(f"    A: {fp_r['answers']}")
        print(f"    FP ({fp_mark}):     {fp_r['prediction'][:80]}")
        print(f"    HMONNX ({hm_mark}): {hm_r['prediction'][:80]}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="GLM-OCR OCRBench 对比: FP vs HMONNX",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--fp_dir", type=str, default="./work_dirs/glm_ocr_ocrbench_fp")
    parser.add_argument("--hmonnx_dir", type=str, default="./work_dirs/glm_ocr_ocrbench_hmonnx")
    parser.add_argument("--show_diffs", type=int, default=20, help="Max per-sample diffs to show")
    return parser.parse_args()


def main():
    args = parse_args()

    fp_scores = load_scores(os.path.join(args.fp_dir, "ocrbench_scores.json"))
    hm_scores = load_scores(os.path.join(args.hmonnx_dir, "ocrbench_scores.json"))

    if fp_scores is None:
        print(f"ERROR: FP scores not found at {args.fp_dir}/ocrbench_scores.json")
        print("  Run glm_ocr_ocrbench_fp_eval.py first.")
        return
    if hm_scores is None:
        print(f"ERROR: HMONNX scores not found at {args.hmonnx_dir}/ocrbench_scores.json")
        print("  Run glm_ocr_ocrbench_hmonnx_eval.py first.")
        return

    print_comparison(fp_scores, hm_scores)
    print_per_sample_diff(args.fp_dir, args.hmonnx_dir, max_show=args.show_diffs)


if __name__ == "__main__":
    main()
