"""Unified evaluation runner — executes evalscope benchmarks with any backend."""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def run_evaluation(
    backend: Any,
    model_display_name: str,
    datasets: List[str],
    work_dir: str,
    dataset_registry: Any,
    limit: int = 0,
    max_tokens: int = 512,
    timeout: int = 1800,
    seed: int = 42,
    use_cache_dir: Optional[str] = None,
    log_callback: Any = None,
) -> Dict[str, Any]:
    """Run evalscope evaluation with the given backend and datasets.

    Args:
        backend: An EvalBackend instance (FloatBackend or HMONNXBackend).
        model_display_name: Display name for reporting.
        datasets: List of dataset names to evaluate.
        work_dir: Output directory for evalscope results.
        dataset_registry: DatasetRegistry instance for dataset metadata.
        limit: Max samples per subset (0 = all).
        max_tokens: Max tokens per generation.
        timeout: Timeout per dataset evaluation in seconds.
        seed: Random seed.
        log_callback: Optional callable(str) for streaming log messages.

    Returns:
        Dict with evaluation results and metadata.
    """
    from evalscope import TaskConfig, run_task
    from evalscope.api.messages import ChatMessage
    from evalscope.api.model import GenerateConfig, ModelAPI, ModelOutput
    from evalscope.api.registry import register_model_api
    try:
        from evalscope.api.tool import ToolChoice, ToolInfo
    except ImportError:
        ToolChoice = Any
        ToolInfo = Any

    from .normalize import (
        get_system_message,
        normalize_eval_output,
        rewrite_messages_for_dataset,
    )

    def _log(msg: str) -> None:
        logger.info(msg)
        if log_callback:
            log_callback(msg)

    # Dynamic ModelAPI class for this evaluation
    class _DynamicEvalModel(ModelAPI):
        def __init__(
            self,
            model_name: str,
            _backend: Any,
            _dataset_name: str,
            config: GenerateConfig,
            **kwargs,
        ) -> None:
            self.model_name = model_name
            self._backend = _backend
            self._dataset_name = _dataset_name
            self.config = config

        def generate(
            self,
            input: List[ChatMessage],
            tools: List[ToolInfo],
            tool_choice: ToolChoice,
            config: GenerateConfig,
        ) -> ModelOutput:
            messages = [{"role": m.role, "content": m.content} for m in input]

            messages = rewrite_messages_for_dataset(messages, self._dataset_name)

            sys_msg = get_system_message(self._dataset_name)
            if sys_msg and not any(m["role"] == "system" for m in messages):
                messages = [{"role": "system", "content": sys_msg}] + messages

            max_tok = getattr(config, "max_tokens", None) or getattr(self.config, "max_tokens", max_tokens)
            try:
                raw_output = self._backend.generate(messages, max_tokens=max_tok)
            except ValueError as exc:
                logger.warning("Sample skipped due to input error: %s", exc)
                return ModelOutput.from_content(model=self.model_name, content="")
            normalized_output = normalize_eval_output(
                raw_output,
                self._dataset_name,
                prompt_text="\n".join(str(message.get("content", "")) for message in messages),
            )
            return ModelOutput.from_content(model=self.model_name, content=normalized_output)

    # Register with a unique name to avoid conflicts
    unique_name = f"HMEval_{model_display_name}_{int(time.time())}"
    register_model_api(name=unique_name)(_DynamicEvalModel)

    import torch
    torch.manual_seed(seed)

    start_time = time.time()
    Path(work_dir).mkdir(parents=True, exist_ok=True)

    all_results: Dict[str, Any] = {
        "model": model_display_name,
        "backend": backend.backend_type,
        "timestamp": datetime.now().isoformat(),
        "datasets": {},
    }

    for dataset_name in datasets:
        evalscope_dataset_name = dataset_name
        if dataset_registry:
            evalscope_dataset_name = dataset_registry.resolve_evalscope_id(dataset_name)

        if evalscope_dataset_name == dataset_name:
            _log(f"===== Evaluating {dataset_name} =====")
        else:
            _log(f"===== Evaluating {dataset_name} (evalscope: {evalscope_dataset_name}) =====")
        ds_start = time.time()

        ds_max_tokens = max_tokens if max_tokens and max_tokens > 0 else 512
        if dataset_registry and ds_max_tokens <= 0:
            ds_max_tokens = dataset_registry.max_tokens_for(dataset_name)

        generate_cfg = GenerateConfig(
            max_tokens=ds_max_tokens,
            temperature=0.0,
            do_sample=False,
            enable_thinking=False,
        )

        eval_model = _DynamicEvalModel(
            model_name=model_display_name,
            _backend=backend,
            _dataset_name=dataset_name,
            config=generate_cfg,
        )

        task_kwargs: Dict[str, Any] = {
            "model": eval_model,
            "datasets": [evalscope_dataset_name],
            "work_dir": work_dir,
            "timeout": timeout,
            "eval_batch_size": 1,
            "seed": seed,
            "generation_config": generate_cfg,
        }

        if use_cache_dir:
            task_kwargs["use_cache"] = use_cache_dir

        if limit and limit > 0:
            task_kwargs["limit"] = limit

        # Build dataset-specific args
        if dataset_registry:
            ds_args = dataset_registry.build_dataset_args(dataset_name)
            if ds_args:
                task_kwargs["dataset_args"] = ds_args

        try:
            task_cfg = TaskConfig(**task_kwargs)
            run_task(task_cfg=task_cfg)
            ds_elapsed = time.time() - ds_start

            # Collect results from work_dir
            ds_result = _collect_dataset_results(work_dir, model_display_name, dataset_name)
            ds_result["elapsed_seconds"] = round(ds_elapsed, 1)
            ds_result["status"] = "completed"
            all_results["datasets"][dataset_name] = ds_result
            _log(f"  {dataset_name} completed in {ds_elapsed:.1f}s")

        except Exception as e:
            logger.exception("Failed to evaluate %s", dataset_name)
            all_results["datasets"][dataset_name] = {
                "status": "failed",
                "error": str(e),
                "elapsed_seconds": round(time.time() - ds_start, 1),
            }
            _log(f"  {dataset_name} FAILED: {e}")

    all_results["total_elapsed_seconds"] = round(time.time() - start_time, 1)
    return all_results


def _collect_dataset_results(
    work_dir: str,
    model_name: str,
    dataset_name: str,
) -> Dict[str, Any]:
    """Collect evalscope results from the output directory.

    Walks the work_dir to find report files (JSON) produced by evalscope.
    """
    import json

    result: Dict[str, Any] = {
        "dataset": dataset_name,
        "metrics": {},
        "subset_scores": {},
        "report_files": [],
    }

    work_path = Path(work_dir)
    # evalscope typically outputs to: work_dir/model_name/timestamp/
    # Look for reports directory
    candidates = sorted(work_path.rglob("reports/*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        candidates = sorted(work_path.rglob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)

    for report_file in candidates[:10]:
        try:
            data = json.loads(report_file.read_text(encoding="utf-8"))
            result["report_files"].append(str(report_file))
            _extract_metrics(data, result)
        except Exception:
            continue

    return result


def _extract_metrics(data: Any, result: Dict[str, Any]) -> None:
    """Extract metrics from an evalscope report JSON."""
    if not isinstance(data, dict):
        return

    # Try common evalscope result structures
    # Structure 1: {"accuracy": 0.xx, ...}
    for key in ["accuracy", "macro_acc", "score", "exact_match", "pass@1"]:
        if key in data:
            result["metrics"][key] = data[key]

    # Structure 2: {"results": {"subset_name": {"accuracy": ...}}}
    if "results" in data and isinstance(data["results"], dict):
        for subset_name, subset_data in data["results"].items():
            if isinstance(subset_data, dict):
                result["subset_scores"][subset_name] = subset_data

    # Structure 3: {"dataset_scores": [...]}
    if "dataset_scores" in data and isinstance(data["dataset_scores"], list):
        for item in data["dataset_scores"]:
            if isinstance(item, dict) and "subset" in item:
                result["subset_scores"][item["subset"]] = item
