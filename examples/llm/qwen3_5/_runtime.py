import json
import re
import time
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoTokenizer

from xh_model_zoo.xh_llm.models.qwen3_5 import Qwen3_5ONNXModel, Qwen3_5SpecDecodeONNXModel


DTYPE_NAME_MAP = {
    "fp16": torch.float16,
    "float16": torch.float16,
    "half": torch.float16,
    "fp32": torch.float32,
    "float32": torch.float32,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
}


def parse_dtype(dtype_name: str) -> torch.dtype:
    key = dtype_name.strip().lower()
    if key not in DTYPE_NAME_MAP:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    return DTYPE_NAME_MAP[key]


def parse_auto_offload_max_memory(max_memory_json: Optional[str]):
    if max_memory_json is None or max_memory_json.strip() == "":
        return None
    parsed = json.loads(max_memory_json)
    if not isinstance(parsed, dict):
        raise ValueError("auto_offload_max_memory must be a JSON object")
    fixed = {}
    for key, value in parsed.items():
        try:
            fixed[int(key)] = value
        except Exception:
            fixed[key] = value
    return fixed


def parse_cuda_graph_modules(modules_arg: str):
    if not modules_arg or not modules_arg.strip():
        return None
    modules = tuple(part.strip().lower() for part in modules_arg.split(",") if part.strip())
    return modules or None


def resolve_path(base_dir: Path, path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def load_token_embedding(embed_path: Path) -> nn.Module:
    try:
        obj = torch.load(str(embed_path), map_location="cpu", weights_only=False)
    except TypeError:
        obj = torch.load(str(embed_path), map_location="cpu")

    if isinstance(obj, nn.Module):
        obj.eval()
        return obj

    if isinstance(obj, dict):
        if "weight" not in obj:
            raise ValueError(f"Unsupported token embedding state dict format: {embed_path}")
        emb = nn.Embedding(obj["weight"].shape[0], obj["weight"].shape[1])
        emb.load_state_dict(obj)
        emb.eval()
        return emb

    raise TypeError(f"Unsupported token embedding object type: {type(obj)}")


def postprocess_chat_output(text: str, enable_thinking: bool) -> str:
    output = text.strip()
    if not enable_thinking:
        output = re.sub(r"<think>[\s\S]*?</think>", "", output)
        output = output.replace("<think>", "").replace("</think>", "")
    return output.strip()


def load_runtime_from_meta(
    meta_path: str,
    dtype: torch.dtype,
    device: str,
    exec_device: str,
    auto_offload: bool,
    auto_offload_max_memory=None,
    prefill_auto_offload_max_memory=None,
    decode_auto_offload_max_memory=None,
    resource_tight_mode: bool = False,
    enable_cuda_graph: bool = False,
    cuda_graph_modules=None,
    cuda_graph_warmup_runs: int = 3,
    cuda_graph_graph_warmup_runs: int = 6,
) -> Tuple[Qwen3_5ONNXModel, AutoTokenizer, dict]:
    meta_file = Path(meta_path).resolve()
    model_dir = meta_file.parent
    meta_info = json.load(open(meta_file, "r", encoding="utf-8"))
    wrap_cfg = meta_info.get("wrap_cfg", {})
    if isinstance(wrap_cfg, dict) and int(wrap_cfg.get("batch_size", 1)) > 1:
        # The continue-batch runtime originally landed under qwen3_5_moe, but it
        # is architecture-agnostic: it resolves split ``*_batch_N`` graph inputs,
        # keeps KV/linear caches per batch item, and loads dense meta.json just
        # like MoE meta.json.
        from xh_model_zoo.xh_llm.models.qwen3_5_moe import load_moe_inference

        runtime = load_moe_inference(
            str(meta_file),
            device=device,
            execution_device=exec_device,
            auto_offload=auto_offload,
            auto_offload_max_memory=auto_offload_max_memory,
            prefill_auto_offload_max_memory=prefill_auto_offload_max_memory,
            decode_auto_offload_max_memory=decode_auto_offload_max_memory,
            resource_tight_mode=resource_tight_mode,
            enable_cuda_graph=enable_cuda_graph,
            cuda_graph_modules=cuda_graph_modules,
            cuda_graph_warmup_runs=cuda_graph_warmup_runs,
            cuda_graph_graph_warmup_runs=cuda_graph_graph_warmup_runs,
        )
        runtime.to(dtype)
        return runtime, runtime.tokenizer, runtime.meta_info

    prefill_onnx = resolve_path(model_dir, meta_info.get("prefill_onnx") or meta_info["prefill_onnx_file"])
    decode_onnx = resolve_path(model_dir, meta_info.get("decode_onnx") or meta_info["decode_onnx_file"])
    hf_config_value = meta_info.get("hf_config") or meta_info.get("hf_config_dir") or "hf_config"
    token_embedding_value = (
        meta_info.get("token_embedding_file")
        or meta_info.get("quant_embedding")
        or "quant_embedding.pt"
    )
    hf_model_config_dir = resolve_path(model_dir, hf_config_value)
    token_embedding_file = resolve_path(model_dir, token_embedding_value)

    max_context_tokens = meta_info.get("max_context_tokens")
    if max_context_tokens is None:
        kv_cache = meta_info.get("kv_cache", {})
        shape = kv_cache.get("shape")
        if isinstance(shape, list) and len(shape) >= 3:
            max_context_tokens = int(shape[2])
        else:
            kv_cache_shape = meta_info.get("kv_cache_shape")
            if isinstance(kv_cache_shape, list) and len(kv_cache_shape) >= 3:
                max_context_tokens = int(kv_cache_shape[2])

    tokenizer = AutoTokenizer.from_pretrained(str(hf_model_config_dir))
    token_embedding = load_token_embedding(token_embedding_file).to(dtype=dtype)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad_token_id is None:
        pad_token_id = 0

    # Check for speculative decoding config in meta.  The spec runtime expects
    # a structured draft config so MTP gets prefill/decode sessions and DFlash
    # gets context/context-decode/decode sessions.  Passing only draft_onnx as a
    # flat dict leaves those named sessions unset.
    spec_decode = meta_info.get("spec_decode")
    if spec_decode and spec_decode.get("mode") in ("dflash", "mtp"):
        block_size = spec_decode.get("block_size", 4)
        hidden_output_name = spec_decode.get("hidden_output_name", "post_norm_hidden")

        def _draft_entry(key: str):
            value = spec_decode.get(key)
            if not value:
                return None
            return {"onnx": str(resolve_path(model_dir, value))}

        draft_cfg = {
            "prefill": _draft_entry("draft_prefill_onnx"),
            "context": _draft_entry("draft_context_onnx"),
            "context_decode": _draft_entry("draft_context_decode_onnx"),
            "decode": _draft_entry("draft_decode_onnx") or _draft_entry("draft_onnx"),
        }
        if draft_cfg["decode"] is None:
            raise ValueError(f"draft decode ONNX path missing from {meta_file}")
        runtime = Qwen3_5SpecDecodeONNXModel(
            prefill=dict(onnx=str(prefill_onnx)),
            decode=dict(onnx=str(decode_onnx)),
            draft=draft_cfg,
            spec_decode_mode=spec_decode["mode"],
            block_size=block_size,
            hidden_output_name=hidden_output_name,
            max_context_tokens=max_context_tokens,
            auto_offload=auto_offload,
            auto_offload_max_memory=auto_offload_max_memory,
            prefill_auto_offload_max_memory=prefill_auto_offload_max_memory,
            decode_auto_offload_max_memory=decode_auto_offload_max_memory,
            resource_tight_mode=resource_tight_mode,
            pad_token_id=pad_token_id,
            enable_cuda_graph=enable_cuda_graph,
            cuda_graph_modules=cuda_graph_modules,
            cuda_graph_warmup_runs=cuda_graph_warmup_runs,
            cuda_graph_graph_warmup_runs=cuda_graph_graph_warmup_runs,
        )
    else:
        runtime = Qwen3_5ONNXModel(
            prefill=dict(onnx=str(prefill_onnx)),
            decode=dict(onnx=str(decode_onnx)),
            max_context_tokens=max_context_tokens,
            auto_offload=auto_offload,
            auto_offload_max_memory=auto_offload_max_memory,
            prefill_auto_offload_max_memory=prefill_auto_offload_max_memory,
            decode_auto_offload_max_memory=decode_auto_offload_max_memory,
            resource_tight_mode=resource_tight_mode,
            pad_token_id=pad_token_id,
        )
    runtime.set_input_embeddings(token_embedding)
    runtime.to(torch.device(device))
    runtime.set_exec_device(torch.device(exec_device))
    runtime.to(dtype)

    return runtime, tokenizer, meta_info


def benchmark_chat(
    runtime: Qwen3_5ONNXModel,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int,
    enable_thinking: bool,
    do_sample: bool,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    presence_penalty: float,
    stream_output: bool,
):
    start = time.perf_counter()
    output = runtime.chat(
        prompt=prompt,
        tokenizer=tokenizer,
        history=None,
        system_prompt="",
        max_new_tokens=max_new_tokens,
        enable_thinking=enable_thinking,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        presence_penalty=presence_penalty,
        stream_output=stream_output,
    )
    elapsed = time.perf_counter() - start
    cleaned = postprocess_chat_output(output, enable_thinking=enable_thinking)
    output_token_count = len(tokenizer.encode(cleaned, add_special_tokens=False))
    return cleaned, elapsed, output_token_count
