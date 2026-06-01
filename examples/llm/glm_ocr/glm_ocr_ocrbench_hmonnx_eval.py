#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GLM-OCR OCRBench HMONNX 量化模型评测脚本

使用已导出的 vision + prefill/decode ONNX/HMONNX 进行 OCRBench 评测。
OCRBench 包含 1000 条样本，涵盖 10 个 OCR 子任务。

数据来源: LMUData/OCRBench.tsv（vlmeval 格式，image 列为 base64 编码）

用法:
    python examples/llm/glm_ocr/glm_ocr_ocrbench_hmonnx_eval.py \
        --model_dir work_dirs/glm_ocr_llm_xh2a_2k_export \
        --vision_export_dir work_dirs/glm_ocr_vision_xh2a_export_hmonnx \
        --output_dir ./work_dirs/glm_ocr_ocrbench_hmonnx \
        --max_samples 100
"""

import argparse
import ast
import base64
import io
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm

import xhquant.utils.suppress_printing
import xhquant.xhonnxruntime.config as xhonnxruntime_config

from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "examples" / "llm" / "glm_ocr"))

from common import xhquant_llm_init, get_root_logger
from xh_model_zoo.xh_llm.models.glm_ocr import GlmOcrONNXModel, GlmOcrProcessor


# ────────────────────────────────────────────────────────────────────
#  OCRBench helpers (shared with fp script)
# ────────────────────────────────────────────────────────────────────

def load_ocrbench(tsv_path: str):
    df = pd.read_csv(tsv_path, sep="\t")
    assert {"index", "image", "question", "answer", "category"}.issubset(df.columns)
    return df


def decode_base64_image(b64_str: str) -> Image.Image:
    raw = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def parse_answer_list(answer_str: str) -> list:
    try:
        ans = ast.literal_eval(answer_str)
        if isinstance(ans, list):
            return [str(a).strip() for a in ans]
        return [str(ans).strip()]
    except Exception:
        return [str(answer_str).strip()]


def ocrbench_match(prediction: str, answers: list) -> bool:
    pred_lower = prediction.lower().strip()
    for ans in answers:
        if ans.lower().strip() in pred_lower:
            return True
    return False


# ────────────────────────────────────────────────────────────────────
#  Path resolution
# ────────────────────────────────────────────────────────────────────

def _resolve_path(path_str, workspace_root):
    if path_str is None:
        return None
    p = Path(path_str).expanduser()
    if p.is_absolute():
        return str(p)
    cwd_p = (Path.cwd() / p).resolve()
    if cwd_p.exists():
        return str(cwd_p)
    return str((workspace_root / p).resolve())


def _resolve_vision_onnx_path(vision_mode, vision_onnx_path_override, default_visual_onnx, vision_export_dir, workspace_root):
    if vision_onnx_path_override is not None:
        return _resolve_path(vision_onnx_path_override, workspace_root), "manual"
    if vision_mode != "unquantized":
        return _resolve_path(default_visual_onnx, workspace_root), "quantized"
    candidates = []
    candidates.append(Path(vision_export_dir) / "onnx" / "visual_1.onnx")
    qp = Path(default_visual_onnx)
    if "vision" in qp.parts:
        idx = qp.parts.index("vision")
        root = Path(*qp.parts[:idx])
        candidates.append(root / "onnx" / "visual_1.onnx")
    for c in candidates:
        r = _resolve_path(str(c), workspace_root)
        if r and Path(r).exists():
            return r, "unquantized"
    raise FileNotFoundError("Cannot find unquantized vision ONNX.")


# ────────────────────────────────────────────────────────────────────
#  Scoring
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
    print(f"\nOCRBench Score (official): {ov['correct']} / {ov['total']}")
    print("=" * 70)


# ────────────────────────────────────────────────────────────────────
#  main
# ────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="GLM-OCR OCRBench HMONNX 评测",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model_dir", type=str, default="work_dirs/glm_ocr_llm_xh2a_2k_export",
                        help="LLM export directory containing hf_config/ and token_embedding.pt")
    parser.add_argument("--vision_export_dir", type=str, default="work_dirs/glm_ocr_vision_xh2a_export_hmonnx",
                        help="Vision export directory")
    parser.add_argument("--visual_onnx", type=str, default=None,
                        help="vision onnx path; default: <vision_export_dir>/vision/<vision_dir_name>.onnx")
    parser.add_argument("--prefill_onnx", type=str, default=None,
                        help="prefill onnx path; default: <model_dir>/prefill_onnx/<model_dir_name>_prefill.onnx")
    parser.add_argument("--decode_onnx", type=str, default=None,
                        help="decode onnx path; default: <model_dir>/decode_onnx/<model_dir_name>_decode.onnx")
    parser.add_argument("--ocrbench_tsv", type=str, default="/data01/home/henry/LMUData/OCRBench.tsv")
    parser.add_argument("--output_dir", type=str, default="./work_dirs/glm_ocr_ocrbench_hmonnx")
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--use_fast", action="store_true", help="Enable fast mode for HMONNX runtime")
    parser.add_argument("--vision_mode", type=str, default="quantized", choices=["quantized", "unquantized"])
    parser.add_argument("--vision_onnx_path", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--num_hidden_layers", type=int, default=16)
    parser.add_argument("--num_kv_heads", type=int, default=8)
    parser.add_argument("--head_dim", type=int, default=128)
    parser.add_argument("--cache_len", type=int, default=2048)
    parser.add_argument("--input_sequence_length", type=int, default=256)
    parser.add_argument("--image_size_w", type=int, default=672)
    parser.add_argument("--image_size_h", type=int, default=672)
    parser.add_argument("--eos_token_id", type=int, nargs="+", default=[151329])
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    workspace_root = PROJECT_ROOT

    # ── load dataset ──
    print(f"Loading OCRBench from {args.ocrbench_tsv} ...")
    df = load_ocrbench(args.ocrbench_tsv)
    if args.max_samples is not None and args.max_samples < len(df):
        df = df.iloc[: args.max_samples]
    print(f"Will evaluate {len(df)} samples.")

    # ── derive paths from model_dir / vision_export_dir ──
    model_dir = Path(args.model_dir)
    vision_export_dir = Path(args.vision_export_dir)
    model_dir_name = model_dir.name
    vision_dir_name = vision_export_dir.name

    default_visual_onnx = args.visual_onnx or str(vision_export_dir / "vision" / f"{vision_dir_name}.onnx")
    prefill_onnx = args.prefill_onnx or str(model_dir / "prefill_onnx" / f"{model_dir_name}_prefill.onnx")
    decode_onnx = args.decode_onnx or str(model_dir / "decode_onnx" / f"{model_dir_name}_decode.onnx")
    hf_model_config_dir = str(model_dir / "hf_config")
    embed_tokens_path = str(model_dir / "token_embedding.pt")

    work_dir = str(Path("./work_dirs") / "glm_ocr_ocrbench_hmonnx")
    Path(work_dir).mkdir(exist_ok=True, parents=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    log_file = Path(output_dir) / "hmonnx_ocrbench_eval.log"
    xhquant_llm_init(log_file, False)
    logger = get_root_logger()

    xhquant.utils.suppress_printing.disable_printing = True
    xhonnxruntime_config.disable_progress = True
    xhonnxruntime_config.verbose_progress = False

    exec_device = torch.device(device)

    # ── load token embedding + processor ──
    hf_model_config_dir = _resolve_path(hf_model_config_dir, workspace_root)
    torch.serialization.add_safe_globals([nn.Embedding])
    token_embedding_path = _resolve_path(embed_tokens_path, workspace_root)
    token_embedding = torch.load(token_embedding_path, weights_only=False, map_location="cpu")
    torch.serialization.clear_safe_globals()

    processor = GlmOcrProcessor.from_pretrained(hf_model_config_dir)

    # ── resolve vision onnx ──
    vision_onnx_path, vision_mode = _resolve_vision_onnx_path(
        args.vision_mode, args.vision_onnx_path, default_visual_onnx,
        str(vision_export_dir), workspace_root,
    )
    logger.info(f"Vision mode: {vision_mode}, onnx={vision_onnx_path}")

    # ── build model directly (matching qwen2_5_vl pattern) ──
    image_feature_cfg = SimpleNamespace(onnx=vision_onnx_path)
    prefill_cfg = SimpleNamespace(
        onnx=_resolve_path(prefill_onnx, workspace_root),
        input_sequence_length=args.input_sequence_length,
    )
    decode_cfg = SimpleNamespace(onnx=_resolve_path(decode_onnx, workspace_root))
    kv_cache_cfg = SimpleNamespace(
        num_hidden_layers=args.num_hidden_layers,
        shape=[1, args.num_kv_heads, args.cache_len, args.head_dim],
    )

    xh_model: GlmOcrONNXModel = GlmOcrONNXModel(
        image_feature=image_feature_cfg,
        prefill=prefill_cfg,
        decode=decode_cfg,
        kv_cache=kv_cache_cfg,
        cache_len=args.cache_len,
        image_size_w=args.image_size_w,
        image_size_h=args.image_size_h,
        eos_token_id=args.eos_token_id,
    )
    xh_model.set_input_embeddings(token_embedding)
    xh_model.set_exec_device(exec_device)
    xh_model.to(exec_device)
    logger.info(f"HMONNX fast mode: {args.use_fast}")

    # ── load existing results for resume ──
    results_path = os.path.join(output_dir, "ocrbench_results.json")
    existing_results = {}
    if args.resume and os.path.exists(results_path):
        with open(results_path, "r", encoding="utf-8") as f:
            for r in json.load(f):
                existing_results[r["index"]] = r
        print(f"Resumed {len(existing_results)} existing results.")

    # ── prepare temp dir for base64 images ──
    tmp_img_dir = os.path.join(output_dir, "tmp_images")
    os.makedirs(tmp_img_dir, exist_ok=True)

    # ── inference ──
    results = []
    total_time = 0.0
    for i in tqdm(range(len(df)), desc="OCRBench HMONNX"):
        row = df.iloc[i]
        idx = int(row["index"])

        if idx in existing_results:
            results.append(existing_results[idx])
            continue

        # Save base64 image to temp file (chat() needs a file path)
        image = decode_base64_image(row["image"])
        tmp_img_path = os.path.join(tmp_img_dir, f"ocrbench_{idx}.png")
        image.save(tmp_img_path)

        question = str(row["question"])
        answers = parse_answer_list(row["answer"])
        category = str(row["category"])

        t0 = time.perf_counter()
        prediction = xh_model.chat(
            question,
            tmp_img_path,
            processor,
            logger,
            use_fast=args.use_fast,
            do_sample=False,
            max_new_tokens=args.max_new_tokens,
            keep_sessions=True,
        )
        elapsed = time.perf_counter() - t0
        total_time += elapsed

        # Clean special tokens
        for tag in ["<|endoftext|>", "<|user|>", "<|assistant|>", "<|observation|>", "<eop>"]:
            prediction = prediction.replace(tag, "")
        prediction = prediction.strip()

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

    xh_model.release_all_sessions()

    # ── save final results ──
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    avg_time = total_time / max(len(df), 1)
    print(f"\nInference done. Total time: {total_time:.1f}s, avg: {avg_time:.2f}s/sample")

    # ── compute and print scores ──
    scores = compute_scores(results)
    print_scores(scores, label="HMONNX")

    scores_path = os.path.join(output_dir, "ocrbench_scores.json")
    with open(scores_path, "w", encoding="utf-8") as f:
        json.dump(scores, f, ensure_ascii=False, indent=2)
    print(f"Scores saved to {scores_path}")


if __name__ == "__main__":
    main()
