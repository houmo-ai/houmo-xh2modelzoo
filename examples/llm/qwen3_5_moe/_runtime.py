from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn

from xh_model_zoo.xh_llm.models.qwen3_5_moe import load_moe_inference


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


def parse_cuda_graph_modules(modules_arg: str) -> Optional[Tuple[str, ...]]:
    if not modules_arg or not modules_arg.strip():
        return None
    modules = tuple(
        part.strip().lower() for part in modules_arg.split(",") if part.strip()
    )
    return modules or None


def load_runtime_from_meta(
    *,
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
    cuda_graph_modules: Optional[Tuple[str, ...]] = None,
    cuda_graph_warmup_runs: int = 3,
    cuda_graph_graph_warmup_runs: int = 6,
):
    runtime = load_moe_inference(
        meta_path,
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


def benchmark_chat(
    runtime,
    tokenizer,
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


def print_cuda_graph_status(runtime) -> None:
    print("cuda_graph_status:")
    for name, status in runtime.get_cuda_graph_status().items():
        reason = status.get("reason") or "-"
        backend = status.get("backend") or "-"
        print(
            f"  {name}: backend={backend} enabled={status.get('enabled')} "
            f"captured={status.get('captured')} reason={reason}"
        )
