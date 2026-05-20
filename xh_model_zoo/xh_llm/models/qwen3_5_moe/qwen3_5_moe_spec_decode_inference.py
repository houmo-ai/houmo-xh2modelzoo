from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional

import torch

from ..qwen3_5.qwen3_5_spec_decode_onnx_model import Qwen3_5SpecDecodeONNXModel
from .inference import (
    Qwen3_5MoeInference,
    _build_runtime_linear_attn_mask,
    _load_meta_artifacts,
    _resolve_path,
)


def _build_spec_decode_config(meta_info: dict, model_dir: Path) -> dict:
    spec_decode_mode = meta_info.get("spec_decode_mode")
    if spec_decode_mode not in {"mtp", "dflash"}:
        raise ValueError("meta.json does not contain a supported spec_decode_mode. Expected one of {'mtp', 'dflash'}.")

    hidden_output_name = meta_info.get(
        "spec_decode_hidden_output_name",
        "target_hidden" if spec_decode_mode == "dflash" else "post_norm_hidden",
    )
    block_size = int(meta_info.get("spec_decode_block_size", 4))
    draft_cfg = {
        "prefill": None,
        "context": None,
        "decode": None,
    }
    draft_prefill = meta_info.get("draft_prefill_onnx_file")
    draft_context = meta_info.get("draft_context_onnx_file")
    draft_context_decode = meta_info.get("draft_context_decode_onnx_file")
    draft_decode = meta_info.get("draft_decode_onnx_file")
    if draft_prefill:
        draft_cfg["prefill"] = {"onnx": str(_resolve_path(model_dir, draft_prefill))}
    if draft_context:
        draft_cfg["context"] = {"onnx": str(_resolve_path(model_dir, draft_context))}
    if draft_context_decode:
        draft_cfg["context_decode"] = {"onnx": str(_resolve_path(model_dir, draft_context_decode))}
    if draft_decode is None:
        raise ValueError(f"draft_decode_onnx_file missing from {model_dir / 'meta.json'}")
    draft_cfg["decode"] = {"onnx": str(_resolve_path(model_dir, draft_decode))}
    return {
        "draft": draft_cfg,
        "mode": spec_decode_mode,
        "block_size": block_size,
        "hidden_output_name": hidden_output_name,
    }


class Qwen3_5MoeSpecDecodeInference(Qwen3_5SpecDecodeONNXModel):
    """Thin MoE spec-decode loader built on the dense Qwen3.5 spec runtime."""

    def __init__(
        self,
        model_config_file: str,
        fast_mode: bool = False,
        device: str = "cuda",
        execution_device: str = "cuda",
        auto_offload: bool = False,
        auto_offload_max_memory=None,
        prefill_auto_offload_max_memory=None,
        decode_auto_offload_max_memory=None,
        resource_tight_mode: bool = False,
        enable_cuda_graph: bool = False,
        cuda_graph_modules: Optional[Iterable[str]] = None,
        cuda_graph_warmup_runs: int = 3,
        cuda_graph_graph_warmup_runs: int = 6,
        cuda_graph_clone_outputs: bool = True,
    ):
        artifacts = _load_meta_artifacts(model_config_file)
        self.meta_path = str(artifacts["meta_file"])
        self.meta_info = artifacts["meta_info"]
        self.fast_mode = fast_mode
        self.tokenizer = artifacts["tokenizer"]

        spec_cfg = _build_spec_decode_config(self.meta_info, artifacts["model_dir"])
        super().__init__(
            prefill={"onnx": str(artifacts["prefill_onnx"])},
            decode={"onnx": str(artifacts["decode_onnx"])},
            draft=spec_cfg["draft"],
            spec_decode_mode=spec_cfg["mode"],
            block_size=spec_cfg["block_size"],
            hidden_output_name=spec_cfg["hidden_output_name"],
            max_context_tokens=artifacts["max_context_tokens"],
            auto_offload=auto_offload,
            auto_offload_max_memory=auto_offload_max_memory,
            prefill_auto_offload_max_memory=prefill_auto_offload_max_memory,
            decode_auto_offload_max_memory=decode_auto_offload_max_memory,
            resource_tight_mode=resource_tight_mode,
            pad_token_id=artifacts["pad_token_id"],
            enable_cuda_graph=enable_cuda_graph,
            cuda_graph_modules=cuda_graph_modules,
            cuda_graph_warmup_runs=cuda_graph_warmup_runs,
            cuda_graph_graph_warmup_runs=cuda_graph_graph_warmup_runs,
            cuda_graph_clone_outputs=cuda_graph_clone_outputs,
        )
        self.set_input_embeddings(artifacts["token_embedding"])
        self.to(torch.device(device))
        self.set_exec_device(torch.device(execution_device))

    def _create_hmonnx_session(self, onnx_path: str, session_name: str):
        session = super()._create_hmonnx_session(onnx_path, session_name)
        if self.fast_mode and hasattr(session, "to_fast_mode"):
            session.to_fast_mode()
        return session


def load_moe_inference(meta_json_path: str, **kwargs):
    meta_path = Path(meta_json_path).resolve()
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("spec_decode_mode") in {"mtp", "dflash"}:
        return Qwen3_5MoeSpecDecodeInference(str(meta_path), **kwargs)
    return Qwen3_5MoeInference(str(meta_path), **kwargs)


__all__ = [
    "Qwen3_5MoeSpecDecodeInference",
    "_build_runtime_linear_attn_mask",
    "load_moe_inference",
]
