import json
import re
import time
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoTokenizer

from xh_model_zoo.xh_llm.models.qwen3_next import Qwen3NextONNXModel


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
) -> Tuple[Qwen3NextONNXModel, AutoTokenizer, dict]:
    meta_file = Path(meta_path).resolve()
    model_dir = meta_file.parent
    meta_info = json.load(open(meta_file, "r", encoding="utf-8"))

    prefill_onnx = resolve_path(model_dir, meta_info.get("prefill_onnx") or meta_info["prefill_onnx_file"])
    decode_onnx = resolve_path(model_dir, meta_info.get("decode_onnx") or meta_info["decode_onnx_file"])
    hf_model_config_dir = resolve_path(model_dir, meta_info["hf_config"])
    token_embedding_file = resolve_path(model_dir, meta_info["token_embedding_file"])

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

    runtime = Qwen3NextONNXModel(
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
    runtime: Qwen3NextONNXModel,
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
        system_prompt="You are a helpful assistant.",
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
