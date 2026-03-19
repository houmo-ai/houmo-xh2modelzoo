"""Quick test: Run vision dialogue validation using existing HMONNX artifacts."""
import os
import sys
import time
from pathlib import Path

# Ensure GPU selection before any torch import
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from _hmonnx_pipeline import (
    _create_hmonnx_session,
    _extract_outputs,
    _ensure_tensor,
    run_dialogue_validation,
    load_json,
)
from xhquant.api import get_root_logger, xhquant_init

WORK_DIR = Path("work_dirs/qwen3omni_fullchain/Qwen3-Omni-30B-A3B-Instruct-XH2a-vision-w8a8h0_sefp")
META_FILE = WORK_DIR / "meta_vision.json"
MODEL_PATH = "/data02/datasets/Qwen3-Omni-30B-A3B-Instruct/"


def main():
    xhquant_init(WORK_DIR / "test_dialogue.log", debug=True)
    logger = get_root_logger()

    meta_info = load_json(META_FILE)
    meta_info["_root_dir"] = str(WORK_DIR)
    meta_info["_meta_path"] = str(META_FILE)

    # Step 1: Quick HMONNX forward test
    onnx_path = WORK_DIR / meta_info["vision_encoder_onnx"]
    logger.info(f"=== Step 1: HMONNX single forward test ===")
    session = _create_hmonnx_session(onnx_path)
    dummy = torch.randn(1, 3, 2, 224, 224, dtype=torch.float16)
    t0 = time.time()
    output = session.forward(dummy)
    t1 = time.time()
    all_out = _extract_outputs(output)
    logger.info(f"HMONNX forward: {len(all_out)} outputs, time={t1-t0:.2f}s")
    for i, o in enumerate(all_out):
        t = _ensure_tensor(o, torch.device("cpu"), torch.float16)
        logger.info(f"  output[{i}]: shape={tuple(t.shape)}")
    del session, output, all_out
    torch.cuda.empty_cache()

    # Step 2: Full dialogue validation
    logger.info(f"=== Step 2: Dialogue validation (vision HMONNX + HF other modules) ===")
    dialogue_artifacts = {"vision": meta_info}

    t0 = time.time()
    report = run_dialogue_validation(
        MODEL_PATH,
        WORK_DIR,
        logger,
        case="vision",
        max_new_tokens=8,  # keep short for fast testing
        device_map="auto",
        artifacts=dialogue_artifacts,
        report_name="vision_dialogue_validation.json",
        output_prefix="vision_dialogue",
    )
    t1 = time.time()

    output_text = report.get("output_text", [])
    applied = report.get("applied_artifacts", [])
    logger.info(f"=== RESULT ===")
    logger.info(f"Time: {t1-t0:.1f}s")
    logger.info(f"Applied artifacts: {applied}")
    logger.info(f"Output text: {output_text}")
    logger.info(f"Report saved: {WORK_DIR / 'vision_dialogue_validation.json'}")

    print(f"\n=== VISION DIALOGUE VALIDATION ===")
    print(f"output_text: {output_text}")
    print(f"applied_artifacts: {applied}")
    print(f"time: {t1-t0:.1f}s")


if __name__ == "__main__":
    main()
