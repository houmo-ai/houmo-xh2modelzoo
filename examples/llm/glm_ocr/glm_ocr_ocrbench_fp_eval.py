#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GLM-OCR OCRBench 浮点模型评测脚本

使用 HF 原生 AutoModelForImageTextToText 对 OCRBench 数据集进行评测。
OCRBench 包含 1000 条样本，涵盖 10 个 OCR 子任务。

数据来源: LMUData/OCRBench.tsv（vlmeval 格式，image 列为 base64 编码）

用法:
    python examples/llm/glm_ocr/glm_ocr_ocrbench_fp_eval.py \
        --model /data02/datasets/GLM-OCR/ \
        --output_dir ./work_dirs/glm_ocr_ocrbench_fp \
        --max_samples 100
"""

import argparse
import ast
import base64
import gc
import io
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "examples" / "llm" / "glm_ocr"))

from common import build_inputs, resolve_torch_dtype


# ────────────────────────────────────────────────────────────────────
#  OCRBench helpers
# ────────────────────────────────────────────────────────────────────

def load_ocrbench(tsv_path: str):
    """Load OCRBench.tsv and return a DataFrame."""
    df = pd.read_csv(tsv_path, sep="\t")
    assert {"index", "image", "question", "answer", "category"}.issubset(df.columns), (
        f"Unexpected columns: {df.columns.tolist()}"
    )
    return df


def decode_base64_image(b64_str: str) -> Image.Image:
    """Decode a base64-encoded JPEG/PNG string into a PIL Image."""
    raw = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def parse_answer_list(answer_str: str) -> list:
    """Parse the answer column, e.g. "['CENTRE']" → ['CENTRE']."""
    try:
        ans = ast.literal_eval(answer_str)
        if isinstance(ans, list):
            return [str(a).strip() for a in ans]
        return [str(ans).strip()]
    except Exception:
        return [str(answer_str).strip()]


def ocrbench_match(prediction: str, answers: list) -> bool:
    """Check if prediction contains any of the ground-truth answers (case-insensitive)."""
    pred_lower = prediction.lower().strip()
    for ans in answers:
        if ans.lower().strip() in pred_lower:
            return True
    return False


def build_messages_from_pil(image: Image.Image, prompt: str):
    """Build chat messages with a PIL Image object instead of a file path."""
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]


# ────────────────────────────────────────────────────────────────────
#  Evaluation & Reporting
# ────────────────────────────────────────────────────────────────────

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


def compute_scores(results: list):
    """Compute per-category and overall accuracy from a list of result dicts."""
    from collections import defaultdict
    cat_correct = defaultdict(int)
    cat_total = defaultdict(int)

    for r in results:
        cat = r["category"]
        cat_total[cat] += 1
        if r["correct"]:
            cat_correct[cat] += 1

    scores = {}
    total_correct = 0
    total_count = 0
    for cat in CATEGORY_ORDER:
        n = cat_total.get(cat, 0)
        c = cat_correct.get(cat, 0)
        if n > 0:
            scores[cat] = {"correct": c, "total": n, "accuracy": c / n}
        total_correct += c
        total_count += n

    # Handle any extra categories not in CATEGORY_ORDER
    for cat in sorted(cat_total.keys()):
        if cat not in scores:
            n = cat_total[cat]
            c = cat_correct[cat]
            scores[cat] = {"correct": c, "total": n, "accuracy": c / n}
            total_correct += c
            total_count += n

    scores["__overall__"] = {
        "correct": total_correct,
        "total": total_count,
        "accuracy": total_correct / total_count if total_count > 0 else 0,
    }
    return scores


def print_scores(scores: dict, label: str = ""):
    """Pretty-print OCRBench scores."""
    title = f"OCRBench Results{f' ({label})' if label else ''}"
    print(f"\n{'=' * 70}")
    print(title)
    print(f"{'=' * 70}")
    print(f"{'Category':<50} {'Correct':>8} {'Total':>6} {'Acc':>8}")
    print("-" * 70)
    for cat in CATEGORY_ORDER:
        if cat in scores:
            s = scores[cat]
            print(f"{cat:<50} {s['correct']:>8} {s['total']:>6} {s['accuracy']:>8.4f}")
    print("-" * 70)
    ov = scores["__overall__"]
    print(f"{'OVERALL':<50} {ov['correct']:>8} {ov['total']:>6} {ov['accuracy']:>8.4f}")
    # OCRBench official score = total correct out of 1000
    print(f"\nOCRBench Score (official): {ov['correct']} / {ov['total']}")
    print("=" * 70)


# ────────────────────────────────────────────────────────────────────
#  main
# ────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="GLM-OCR OCRBench 浮点评测",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", type=str, default="/data02/datasets/GLM-OCR/")
    parser.add_argument("--ocrbench_tsv", type=str, default="/data01/home/henry/LMUData/OCRBench.tsv",
                        help="Path to OCRBench.tsv")
    parser.add_argument("--output_dir", type=str, default="./work_dirs/glm_ocr_ocrbench_fp")
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of samples for debugging")
    parser.add_argument("--dtype", type=str, default="float16", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--resume", action="store_true", help="Skip already-evaluated samples")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # ── load dataset ──
    print(f"Loading OCRBench from {args.ocrbench_tsv} ...")
    df = load_ocrbench(args.ocrbench_tsv)
    if args.max_samples is not None and args.max_samples < len(df):
        df = df.iloc[: args.max_samples]
    print(f"Will evaluate {len(df)} samples.")

    # ── load model ──
    dtype = resolve_torch_dtype(args.dtype)
    device = torch.device(args.device)
    print(f"Loading GLM-OCR model from {args.model} (torch_dtype={dtype}) ...")
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map=str(device),
        trust_remote_code=True,
        attn_implementation="eager",
    ).eval()
    print("Model loaded.")

    # ── load existing results for resume ──
    results_path = os.path.join(output_dir, "ocrbench_results.json")
    existing_results = {}
    if args.resume and os.path.exists(results_path):
        with open(results_path, "r", encoding="utf-8") as f:
            for r in json.load(f):
                existing_results[r["index"]] = r
        print(f"Resumed {len(existing_results)} existing results.")

    # ── inference ──
    results = []
    total_time = 0.0
    for i in tqdm(range(len(df)), desc="OCRBench FP"):
        row = df.iloc[i]
        idx = int(row["index"])

        if idx in existing_results:
            results.append(existing_results[idx])
            continue

        image = decode_base64_image(row["image"])
        question = str(row["question"])
        answers = parse_answer_list(row["answer"])
        category = str(row["category"])

        messages = build_messages_from_pil(image, question)
        inputs = build_inputs(processor, messages, device=device)

        t0 = time.perf_counter()
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
        elapsed = time.perf_counter() - t0
        total_time += elapsed

        input_len = inputs["input_ids"].shape[1]
        prediction = processor.decode(
            generated_ids[0][input_len:], skip_special_tokens=True,
        ).strip()

        del inputs, generated_ids
        gc.collect()
        torch.cuda.empty_cache()

        correct = ocrbench_match(prediction, answers)
        results.append({
            "index": idx,
            "category": category,
            "question": question,
            "answers": answers,
            "prediction": prediction,
            "correct": correct,
            "time": elapsed,
        })

        # Incremental save every 50 samples
        if len(results) % 50 == 0:
            with open(results_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

    # ── save final results ──
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    avg_time = total_time / max(len(df), 1)
    print(f"\nInference done. Total time: {total_time:.1f}s, avg: {avg_time:.2f}s/sample")

    # ── compute and print scores ──
    scores = compute_scores(results)
    print_scores(scores, label="FP")

    scores_path = os.path.join(output_dir, "ocrbench_scores.json")
    with open(scores_path, "w", encoding="utf-8") as f:
        json.dump(scores, f, ensure_ascii=False, indent=2)
    print(f"Scores saved to {scores_path}")


if __name__ == "__main__":
    main()
