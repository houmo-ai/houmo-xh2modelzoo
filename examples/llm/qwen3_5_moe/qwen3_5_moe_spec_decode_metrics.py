from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

_HERE = Path(__file__).resolve().parent
_DENSE_DIR = _HERE.parent / "qwen3_5"
if str(_DENSE_DIR) not in sys.path:
    sys.path.insert(0, str(_DENSE_DIR))

from qwen3_5_spec_decode_metrics import (
    release_dense_runtime,
    reset_dense_spec_runtime,
    run_dense_spec_metrics_with_runtime,
    run_dense_target_baseline_from_spec,
)
from xh_model_zoo.xh_llm.models.qwen3_5_moe import Qwen3_5MoeSpecDecodeInference


def _load_baseline_runtime(
    meta_path: str,
    device: str,
    exec_device: str,
    *,
    enable_cuda_graph: bool = False,
    cuda_graph_modules: Optional[Tuple[str, ...]] = None,
    cuda_graph_warmup_runs: int = 3,
    cuda_graph_graph_warmup_runs: int = 6,
) -> Qwen3_5MoeSpecDecodeInference:
    return _load_spec_runtime(
        meta_path,
        device,
        exec_device,
        enable_cuda_graph=enable_cuda_graph,
        cuda_graph_modules=cuda_graph_modules,
        cuda_graph_warmup_runs=cuda_graph_warmup_runs,
        cuda_graph_graph_warmup_runs=cuda_graph_graph_warmup_runs,
    )


def _load_spec_runtime(
    meta_path: str,
    device: str,
    exec_device: str,
    *,
    enable_cuda_graph: bool = False,
    cuda_graph_modules: Optional[Tuple[str, ...]] = None,
    cuda_graph_warmup_runs: int = 3,
    cuda_graph_graph_warmup_runs: int = 6,
) -> Qwen3_5MoeSpecDecodeInference:
    return Qwen3_5MoeSpecDecodeInference(
        meta_path,
        device=device,
        execution_device=exec_device,
        enable_cuda_graph=enable_cuda_graph,
        cuda_graph_modules=cuda_graph_modules,
        cuda_graph_warmup_runs=cuda_graph_warmup_runs,
        cuda_graph_graph_warmup_runs=cuda_graph_graph_warmup_runs,
    )


def _release_runtime(runtime) -> None:
    release_dense_runtime(runtime)


def reset_moe_spec_runtime(runtime: Qwen3_5MoeSpecDecodeInference) -> None:
    reset_dense_spec_runtime(runtime)


def run_moe_target_baseline_from_spec(
    meta_path: str,
    prompt: str,
    max_new_tokens: int,
    device: str,
    exec_device: str,
    enable_thinking: bool = False,
    repetition_penalty: float = 1.0,
    presence_penalty: float = 0.0,
    runtime: Optional[Qwen3_5MoeSpecDecodeInference] = None,
    enable_cuda_graph: bool = False,
    cuda_graph_modules: Optional[Tuple[str, ...]] = None,
    cuda_graph_warmup_runs: int = 3,
    cuda_graph_graph_warmup_runs: int = 6,
) -> Dict[str, Any]:
    owns_runtime = runtime is None
    if runtime is None:
        runtime = _load_baseline_runtime(
            meta_path,
            device,
            exec_device,
            enable_cuda_graph=enable_cuda_graph,
            cuda_graph_modules=cuda_graph_modules,
            cuda_graph_warmup_runs=cuda_graph_warmup_runs,
            cuda_graph_graph_warmup_runs=cuda_graph_graph_warmup_runs,
        )
    try:
        return run_dense_target_baseline_from_spec(
            runtime,
            runtime.tokenizer,
            prompt,
            max_new_tokens,
            enable_thinking=enable_thinking,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
        )
    finally:
        if owns_runtime:
            _release_runtime(runtime)


def run_moe_spec_metrics_with_runtime(
    *,
    meta_path: str,
    runtime: Qwen3_5MoeSpecDecodeInference,
    prompt: str,
    max_new_tokens: int,
    enable_thinking: bool = False,
    repetition_penalty: float = 1.0,
    presence_penalty: float = 0.0,
) -> Dict[str, Any]:
    del meta_path
    return run_dense_spec_metrics_with_runtime(
        runtime=runtime,
        tokenizer=runtime.tokenizer,
        model_name=runtime.meta_info.get("model_name", "qwen3_5_moe"),
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        enable_thinking=enable_thinking,
        repetition_penalty=repetition_penalty,
        presence_penalty=presence_penalty,
        include_baseline=True,
    )


def run_moe_spec_metrics(
    *,
    meta_path: str,
    prompt: str,
    max_new_tokens: int,
    device: str = "cuda:0",
    exec_device: str = "cuda:0",
    enable_thinking: bool = False,
    repetition_penalty: float = 1.0,
    presence_penalty: float = 0.0,
    enable_cuda_graph: bool = False,
    cuda_graph_modules: Optional[Tuple[str, ...]] = None,
    cuda_graph_warmup_runs: int = 3,
    cuda_graph_graph_warmup_runs: int = 6,
) -> Dict[str, Any]:
    runtime = _load_spec_runtime(
        meta_path,
        device,
        exec_device,
        enable_cuda_graph=enable_cuda_graph,
        cuda_graph_modules=cuda_graph_modules,
        cuda_graph_warmup_runs=cuda_graph_warmup_runs,
        cuda_graph_graph_warmup_runs=cuda_graph_graph_warmup_runs,
    )
    try:
        return run_moe_spec_metrics_with_runtime(
            meta_path=meta_path,
            runtime=runtime,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            enable_thinking=enable_thinking,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
        )
    finally:
        _release_runtime(runtime)
