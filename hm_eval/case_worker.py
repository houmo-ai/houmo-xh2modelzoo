"""Subprocess worker for case analysis inference.

Runs in a separate process so CUDA_VISIBLE_DEVICES is set before CUDA init.
Usage: python -m hm_eval.case_worker <case_request.json>
"""

from __future__ import annotations

import json
import sys
import os
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m hm_eval.case_worker <request_json_path>", file=sys.stderr)
        sys.exit(1)

    request_path = Path(sys.argv[1])
    if not request_path.exists():
        print(f"Request file not found: {request_path}", file=sys.stderr)
        sys.exit(1)

    request = json.loads(request_path.read_text(encoding="utf-8"))

    gpu_id = request.get("gpu_id", "0")
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    from hm_eval.core.case_analysis import run_case_inference

    record = run_case_inference(
        backend_type=request["backend_type"],
        model_path=request["model_path"],
        gpu_id=gpu_id,
        prompt_text=request.get("prompt_text", ""),
        image_paths=request.get("image_paths", []),
        audio_paths=request.get("audio_paths", []),
        pdf_paths=request.get("pdf_paths", []),
        max_tokens=request.get("max_tokens", 2048),
        onnx_model_type=request.get("onnx_model_type", ""),
        vision_meta_path=request.get("vision_meta_path", ""),
        max_pdf_pages=request.get("max_pdf_pages", 5),
    )

    result_path = request_path.with_suffix(".result.json")
    result_path.write_text(json.dumps({
        "case_id": record.case_id,
        "model_path": record.model_path,
        "backend_type": record.backend_type,
        "gpu_id": record.gpu_id,
        "prompt_text": record.prompt_text,
        "input_files": record.input_files,
        "response": record.response,
        "elapsed_seconds": record.elapsed_seconds,
        "created_at": record.created_at,
        "label": record.label,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Done: {record.case_id} elapsed={record.elapsed_seconds:.1f}s")


if __name__ == "__main__":
    main()
