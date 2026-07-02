"""Quick smoke test: verify backend creation + single generate call for all target models.

Tests that the model loading and basic inference works for ceval/mmlu_pro-style prompts.
Does NOT run evalscope (too slow). Just validates the hm_eval backend plumbing.

Usage:
    CUDA_VISIBLE_DEVICES=7 python hm_eval/tests/smoke_test_backends.py
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("smoke_test")

CEVAL_SAMPLE_MESSAGES = [
    {"role": "system", "content": "请直接输出最终选项，不要输出解析。最后一行严格写成：答案：A"},
    {"role": "user", "content": "以下是中国关于高中数学的单项选择题，请选出其中的正确答案。\n\n问题：若函数f(x)=x^2+2x+a的值域为[0,+∞)，则a的值为____\n选项：\nA. 1\nB. 2\nC. -1\nD. 0\n\n请只输出一行，格式为：答案：A。不要解释。"},
]

MMLU_PRO_SAMPLE_MESSAGES = [
    {"role": "system", "content": "Return only the final option. The last line must be exactly in the format: Answer: A"},
    {"role": "user", "content": "The following is a multiple choice question about computer science.\n\nQuestion: Which of the following sorting algorithms has the best average-case time complexity?\nOptions:\nA. Bubble Sort\nB. Selection Sort\nC. Merge Sort\nD. Insertion Sort\nE. Quick Sort\nF. Heap Sort\nG. Radix Sort\nH. Counting Sort\nI. Shell Sort\nJ. Bucket Sort\n\nReturn exactly one line in the format: Answer: A. Do not explain."},
]


def test_single_backend(model_id: str, backend_type: str) -> dict:
    from hm_eval.core.model_registry import ModelRegistry
    from hm_eval.core.backends import create_backend
    from hm_eval.core.normalize import normalize_eval_output

    registry = ModelRegistry()
    registry.scan()
    model_config = registry.get_model(model_id)
    if model_config is None:
        return {"model": model_id, "backend": backend_type, "status": "error", "error": f"Model config not found: {model_id}"}

    if backend_type not in model_config.backends:
        return {"model": model_id, "backend": backend_type, "status": "skip", "error": "Backend not configured"}

    if backend_type == "hmonnx":
        meta_path = model_config.backends["hmonnx"].export_meta_info
        if not meta_path or not Path(meta_path).exists():
            return {"model": model_id, "backend": backend_type, "status": "skip", "error": f"HMONNX meta not found: {meta_path}"}

    try:
        logger.info("Creating %s backend for %s...", backend_type, model_id)
        start = time.time()
        backend = create_backend(backend_type, model_config)
        load_time = time.time() - start
        logger.info("Backend loaded in %.1fs", load_time)
    except Exception as e:
        logger.error("Failed to create backend: %s", e)
        return {"model": model_id, "backend": backend_type, "status": "error", "error": f"Backend creation failed: {e}"}

    results = {"model": model_id, "backend": backend_type, "load_time": round(load_time, 1)}

    # Test ceval
    try:
        logger.info("Testing ceval prompt...")
        start = time.time()
        raw_output = backend.generate(CEVAL_SAMPLE_MESSAGES, max_tokens=32)
        gen_time = time.time() - start
        normalized = normalize_eval_output(raw_output, "ceval")
        results["ceval_raw"] = raw_output[:200]
        results["ceval_normalized"] = normalized
        results["ceval_time"] = round(gen_time, 1)
        results["ceval_ok"] = normalized.startswith("答案：") and len(normalized) == 4
        logger.info("  ceval raw: %s", raw_output[:100])
        logger.info("  ceval normalized: %s (ok=%s, time=%.1fs)", normalized, results["ceval_ok"], gen_time)
    except Exception as e:
        results["ceval_ok"] = False
        results["ceval_error"] = str(e)
        logger.error("  ceval FAILED: %s", e)

    # Test mmlu_pro
    try:
        logger.info("Testing mmlu_pro prompt...")
        start = time.time()
        raw_output = backend.generate(MMLU_PRO_SAMPLE_MESSAGES, max_tokens=32)
        gen_time = time.time() - start
        normalized = normalize_eval_output(raw_output, "mmlu_pro")
        results["mmlu_pro_raw"] = raw_output[:200]
        results["mmlu_pro_normalized"] = normalized
        results["mmlu_pro_time"] = round(gen_time, 1)
        results["mmlu_pro_ok"] = normalized.startswith("Answer: ") and len(normalized) == 9
        logger.info("  mmlu_pro raw: %s", raw_output[:100])
        logger.info("  mmlu_pro normalized: %s (ok=%s, time=%.1fs)", normalized, results["mmlu_pro_ok"], gen_time)
    except Exception as e:
        results["mmlu_pro_ok"] = False
        results["mmlu_pro_error"] = str(e)
        logger.error("  mmlu_pro FAILED: %s", e)

    backend.cleanup()

    if results.get("ceval_ok") and results.get("mmlu_pro_ok"):
        results["status"] = "pass"
    elif results.get("ceval_ok") or results.get("mmlu_pro_ok"):
        results["status"] = "partial"
    else:
        results["status"] = "fail"

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="", help="Single model to test (empty=all)")
    parser.add_argument("--backend", default="", help="Single backend to test (empty=both)")
    args = parser.parse_args()

    target_models = [
        "gemma4_e2b_it",
        "gemma4_e4b_it",
        "gemma4_31b_it",
        "gemma4_26b_a4b_it",
        "qwen3_5_35b_a3b",
    ]

    if args.model:
        target_models = [args.model]

    backends = ["float", "hmonnx"]
    if args.backend:
        backends = [args.backend]

    all_results = []
    for model_id in target_models:
        for backend_type in backends:
            result = test_single_backend(model_id, backend_type)
            all_results.append(result)
            logger.info(">>> %s %s: %s", model_id, backend_type, result["status"])
            # Free GPU memory between models
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for r in all_results:
        status_icon = {"pass": "OK", "partial": "PARTIAL", "fail": "FAIL", "skip": "SKIP", "error": "ERROR"}.get(r["status"], "?")
        ceval_status = "OK" if r.get("ceval_ok") else r.get("ceval_error", "FAIL")[:40]
        mmlu_status = "OK" if r.get("mmlu_pro_ok") else r.get("mmlu_pro_error", "FAIL")[:40]
        print(f"  [{status_icon:7s}] {r['model']:25s} {r['backend']:8s} | ceval={ceval_status} | mmlu_pro={mmlu_status}")
    print("=" * 80)

    output_file = Path(__file__).parent / "smoke_test_results.json"
    output_file.write_text(json.dumps(all_results, indent=2, ensure_ascii=False))
    logger.info("Results saved to %s", output_file)


if __name__ == "__main__":
    main()
