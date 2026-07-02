"""Quick validation test for ceval and mmlu_pro on float and hmonnx backends.

Usage:
    CUDA_VISIBLE_DEVICES=7 python -m hm_eval.tests.test_ceval_mmlu_pro [--model MODEL] [--backend BACKEND] [--dataset DATASET]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("test_ceval_mmlu_pro")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gemma4_e2b_it", help="Model config ID")
    parser.add_argument("--backend", default="float", choices=["float", "hmonnx"])
    parser.add_argument("--dataset", default="ceval", choices=["ceval", "mmlu_pro"])
    parser.add_argument("--limit", type=int, default=5, help="Max samples per subset")
    parser.add_argument("--max-tokens", type=int, default=32)
    return parser.parse_args()


def main():
    args = parse_args()
    logger.info("Testing %s on %s with %s backend (limit=%d)", args.model, args.dataset, args.backend, args.limit)

    from hm_eval.core.model_registry import ModelRegistry
    from hm_eval.core.backends import create_backend
    from hm_eval.core.dataset_registry import DatasetRegistry
    from hm_eval.core.eval_runner import run_evaluation

    registry = ModelRegistry()
    registry.scan()
    model_config = registry.get_model(args.model)
    if model_config is None:
        logger.error("Model config not found: %s", args.model)
        sys.exit(1)

    logger.info("Creating %s backend for %s...", args.backend, model_config.display_name)
    start = time.time()
    backend = create_backend(args.backend, model_config)
    logger.info("Backend created in %.1fs", time.time() - start)

    ds_registry = DatasetRegistry()
    work_dir = str(Path(__file__).resolve().parent / "test_outputs" / f"{args.model}_{args.backend}_{args.dataset}_{int(time.time())}")

    logger.info("Running evaluation...")
    results = run_evaluation(
        backend=backend,
        model_display_name=model_config.display_name,
        datasets=[args.dataset],
        work_dir=work_dir,
        dataset_registry=ds_registry,
        limit=args.limit,
        max_tokens=args.max_tokens,
    )

    backend.cleanup()

    ds_result = results.get("datasets", {}).get(args.dataset, {})
    status = ds_result.get("status", "unknown")
    score = ds_result.get("metrics", {}).get("score") or ds_result.get("metrics", {}).get("macro_acc") or ds_result.get("metrics", {}).get("accuracy")

    logger.info("=" * 60)
    logger.info("RESULT: model=%s backend=%s dataset=%s", args.model, args.backend, args.dataset)
    logger.info("  status: %s", status)
    logger.info("  score: %s", score)
    logger.info("  elapsed: %.1fs", ds_result.get("elapsed_seconds", 0))
    logger.info("=" * 60)

    # Check predictions were extracted correctly
    pred_dir = Path(work_dir)
    pred_files = list(pred_dir.rglob("*.jsonl"))
    if pred_files:
        sample_file = pred_files[0]
        lines = sample_file.read_text().strip().splitlines()
        if lines:
            sample = json.loads(lines[0])
            raw_prediction = sample.get("raw_prediction") or sample.get("prediction", "")
            logger.info("  sample prediction: %s", raw_prediction[:100])

    if status == "completed":
        logger.info("PASS")
        return 0
    else:
        logger.error("FAIL: %s", ds_result.get("error", "unknown error"))
        return 1


if __name__ == "__main__":
    sys.exit(main())
