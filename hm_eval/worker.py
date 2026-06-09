"""Worker process — executes a single evaluation task in isolation.

This module is invoked as a subprocess by TaskManager to run evaluations
with the correct transformers version and GPU assignment.

For hmonnx backend, this worker expects an existing golden_meta_info.json
or export_meta_info.json path. Model export is intentionally performed offline.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from hm_eval.core.hmonnx_meta import resolve_hmonnx_embedded_vision_meta_path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("hm_eval.worker")

_REPO_ROOT = Path(__file__).resolve().parent.parent

_MULTIMODAL_DATASETS = {"cmmmu", "mmmu", "mmmu_pro", "math_vision", "omnidoc_bench"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="hm_eval worker process")
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--task-file", required=True)
    return parser.parse_args()


def _update_task(task_file: Path, task_data: dict) -> None:
    task_file.write_text(
        json.dumps(task_data, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _fail(task_file: Path, task_data: dict, error: str) -> None:
    logger.error(error)
    task_data["status"] = "failed"
    task_data["error"] = error
    task_data["finished_at"] = datetime.now().isoformat()
    _update_task(task_file, task_data)
    sys.exit(1)


def main() -> None:
    args = parse_args()
    task_file = Path(args.task_file)
    if not task_file.exists():
        logger.error("Task file not found: %s", task_file)
        sys.exit(1)

    task_data = json.loads(task_file.read_text(encoding="utf-8"))
    logger.info("Worker started for task %s", args.task_id)
    logger.info("Task config: %s", json.dumps(task_data, indent=2))

    # Reset stale terminal state when rerunning an existing task file.
    task_data["error"] = ""
    task_data["finished_at"] = ""

    # Step 1: Environment setup — switch transformers version if needed
    target_version = task_data.get("transformers_version", "")
    if target_version:
        from hm_eval.core.env_manager import (
            find_python_with_transformers_version,
            get_current_transformers_version,
            python_can_import_xhquant_hsum,
            switch_transformers_version,
            transformers_version_matches,
        )

        require_xhquant_hsum = task_data.get("backend") == "hmonnx"
        current = get_current_transformers_version()
        current_matches_version = transformers_version_matches(current, target_version)
        current_has_hmonnx_runtime = (
            not require_xhquant_hsum or python_can_import_xhquant_hsum(sys.executable)
        )

        if not current_matches_version or not current_has_hmonnx_runtime:
            matched_python, matched_version = find_python_with_transformers_version(
                target_version,
                require_xhquant_hsum=require_xhquant_hsum,
                require_evalscope=True,
            )
            current_python = str(Path(sys.executable).resolve())
            if matched_python and str(Path(matched_python).resolve()) != current_python:
                logger.info(
                    "Re-executing worker with %s (transformers %s) for target %s%s",
                    matched_python,
                    matched_version,
                    target_version,
                    " and working xhquant runtime" if require_xhquant_hsum else "",
                )
                task_data["status"] = "env_setup"
                _update_task(task_file, task_data)
                exec_env = os.environ.copy()
                exec_args = [
                    matched_python,
                    "-u",
                    "-m",
                    "hm_eval.worker",
                    "--task-id",
                    args.task_id,
                    "--task-file",
                    args.task_file,
                ]
                os.execve(matched_python, exec_args, exec_env)

            if not current_matches_version:
                logger.info("Switching transformers %s -> %s", current, target_version)
                task_data["status"] = "env_setup"
                _update_task(task_file, task_data)

                ok, msg = switch_transformers_version(target_version)
                if not ok:
                    _fail(task_file, task_data, f"transformers switch failed: {msg}")
                logger.info("transformers switched: %s", msg)

            if require_xhquant_hsum and not python_can_import_xhquant_hsum(sys.executable):
                _fail(
                    task_file,
                    task_data,
                    "当前 transformers 环境无法导入可用的 xhquant.lib.hsum；"
                    "请使用已编译 xhquant 扩展的环境运行 hmonnx 评测。",
                )

    # Step 2: GPU selection — user-specified or auto
    cuda_devices = task_data.get("cuda_devices", "")
    if cuda_devices and cuda_devices != "auto":
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices
        logger.info("Using user-specified GPUs: CUDA_VISIBLE_DEVICES=%s", cuda_devices)
    else:
        from hm_eval.core.env_manager import select_free_gpu
        gpu_id = select_free_gpu()
        if gpu_id is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            logger.info("Auto-selected GPU %d", gpu_id)
        else:
            logger.warning("No free GPU found, using default CUDA_VISIBLE_DEVICES")

    # Step 3: For hmonnx backend — require export_meta_info_path (export done offline)
    if task_data["backend"] == "hmonnx" and not task_data.get("export_meta_info_path"):
        _fail(task_file, task_data,
              "hmonnx 后端需要提供 golden_meta_info.json 或 export_meta_info.json 路径。"
              "请先在离线环境完成模型导出（参考 examples_merak/llm/ 目录下的导出脚本），"
              "然后在页面上填写对应 meta 文件的完整绝对路径。")

    if task_data["backend"] == "hmonnx":
        selected_multimodal = sorted(set(task_data.get("datasets", [])) & _MULTIMODAL_DATASETS)
        embedded_vision_meta_path = None
        if selected_multimodal and not task_data.get("vision_export_meta_info_path"):
            embedded_vision_meta_path = resolve_hmonnx_embedded_vision_meta_path(
                Path(task_data.get("export_meta_info_path", ""))
            )
        if selected_multimodal and not task_data.get("vision_export_meta_info_path") and embedded_vision_meta_path is None:
            _fail(task_file, task_data,
                  "多模态 HMONNX 评测需要额外提供 vision export_meta_info.json 路径。"
                  f"当前选择的数据集包含: {', '.join(selected_multimodal)}")
        if embedded_vision_meta_path is not None:
            logger.info(
                "Using embedded vision runtime from LLM HMONNX meta for multimodal datasets %s: %s",
                ", ".join(selected_multimodal),
                embedded_vision_meta_path,
            )

    # Step 4: Load model config and override export_meta_info if provided
    from hm_eval.core.model_registry import ModelRegistry
    registry = ModelRegistry()
    registry.scan()
    model_config = registry.get_model(task_data["model_config_id"])
    if model_config is None:
        _fail(task_file, task_data,
              f"Model config not found: {task_data['model_config_id']}")

    submitted_hf_model_dir = (task_data.get("hf_model_dir") or "").strip()
    if submitted_hf_model_dir:
        model_config.hf_model_dir = submitted_hf_model_dir

    if (
        task_data["backend"] == "float"
        and model_config.processor_class == "AutoProcessor"
        and model_config.model_class.startswith("Qwen3_5")
    ):
        from hm_eval.core.env_manager import ensure_qwen3_5_processor_deps

        ok, msg = ensure_qwen3_5_processor_deps(sys.executable)
        if not ok:
            _fail(task_file, task_data, f"Qwen3.5 processor dependency setup failed: {msg}")
        logger.info(msg)

    # Override export_meta_info in the model config's hmonnx backend
    if task_data.get("export_meta_info_path") and "hmonnx" in model_config.backends:
        model_config.backends["hmonnx"].export_meta_info = task_data["export_meta_info_path"]

    if task_data.get("vision_export_meta_info_path") and "hmonnx" in model_config.backends:
        model_config.backends["hmonnx"].vision_export_meta_info = task_data["vision_export_meta_info_path"]

    # Step 5: Create backend
    from hm_eval.core.backends import create_backend
    try:
        backend = create_backend(task_data["backend"], model_config)
    except Exception as e:
        logger.exception("Failed to create backend")
        _fail(task_file, task_data, f"Backend creation failed: {e}")

    # Step 6: Run evaluation
    task_data["status"] = "running"
    task_data["started_at"] = datetime.now().isoformat()
    _update_task(task_file, task_data)

    from hm_eval.core.dataset_registry import DatasetRegistry
    from hm_eval.core.eval_runner import run_evaluation
    from hm_eval.core.report import generate_report, save_report
    from hm_eval.core.task_manager import EvalTask, TaskStatus, detect_abnormal_result

    ds_registry = DatasetRegistry()
    resume_from_cache = task_data.get("resume_from_cache", "")
    if resume_from_cache:
        logger.info("Resuming task from cache dir: %s", resume_from_cache)

    try:
        eval_results = run_evaluation(
            backend=backend,
            model_display_name=task_data["model_display_name"],
            datasets=task_data["datasets"],
            work_dir=task_data["work_dir"],
            dataset_registry=ds_registry,
            limit=task_data.get("limit", 0),
            max_tokens=task_data.get("max_tokens", 512),
            use_cache_dir=resume_from_cache or None,
        )

        # Generate and save report
        report = generate_report(eval_results, task_data["work_dir"])
        report_path = save_report(report, task_data["work_dir"])
        logger.info("Report saved to %s", report_path)

        task_data["report_path"] = report_path
        task_data["error"] = ""
        task_data["finished_at"] = datetime.now().isoformat()
        task_data["resume_from_cache"] = ""

        task_for_check = EvalTask.from_dict({
            **task_data,
            "status": TaskStatus.COMPLETED.value,
        })
        is_abnormal, abnormal_reason = detect_abnormal_result(task_for_check)
        if is_abnormal:
            task_data["status"] = TaskStatus.ABNORMAL.value
            task_data["error"] = abnormal_reason
        else:
            task_data["status"] = TaskStatus.COMPLETED.value

    except Exception as e:
        logger.exception("Evaluation failed")
        task_data["status"] = "failed"
        task_data["error"] = str(e)
        task_data["finished_at"] = datetime.now().isoformat()

    finally:
        backend.cleanup()
        _update_task(task_file, task_data)

    logger.info("Worker finished with status: %s", task_data["status"])
    sys.exit(0 if task_data["status"] in (TaskStatus.COMPLETED.value, TaskStatus.ABNORMAL.value) else 1)


if __name__ == "__main__":
    main()
