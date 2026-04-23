"""Runtime loader and benchmark for speculative decoding with HMONNX on xh2a."""

import argparse
import json
import re
import time
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoTokenizer

from xh_model_zoo.xh_llm.models.qwen3_5 import (
    Qwen3_5ONNXModel,
    Qwen3_5SpecDecodeONNXModel,
)


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
            raise ValueError(
                f"Unsupported token embedding state dict format: {embed_path}"
            )
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


def load_spec_decode_runtime(
    meta_path: str,
    dtype: torch.dtype,
    device: str,
    exec_device: str,
    auto_offload: bool,
    auto_offload_max_memory=None,
    prefill_auto_offload_max_memory=None,
    decode_auto_offload_max_memory=None,
    resource_tight_mode: bool = False,
    num_draft_tokens_override: Optional[int] = None,
) -> Tuple[Qwen3_5SpecDecodeONNXModel, AutoTokenizer, dict]:
    """Load speculative decode runtime from meta.json.

    meta.json must contain a `spec_decode` section with:
    - mode: 'dflash' or 'mtp'
    - draft_prefill_onnx / draft_context_onnx / draft_decode_onnx
    - block_size: number of draft tokens per round
    - hidden_output_name: name of hidden state output
    """
    meta_file = Path(meta_path).resolve()
    model_dir = meta_file.parent
    meta_info = json.load(open(meta_file, "r", encoding="utf-8"))

    prefill_onnx = resolve_path(
        model_dir, meta_info.get("prefill_onnx") or meta_info["prefill_onnx_file"]
    )
    decode_onnx = resolve_path(
        model_dir, meta_info.get("decode_onnx") or meta_info["decode_onnx_file"]
    )
    hf_model_config_dir = resolve_path(model_dir, meta_info["hf_config"])
    token_embedding_file = resolve_path(
        model_dir, meta_info["token_embedding_file"]
    )

    spec_decode = meta_info.get("spec_decode")
    if spec_decode is None:
        raise ValueError(
            "meta.json does not contain spec_decode section. "
            "Export with --spec_decode_mode dflash|mtp first."
        )

    spec_mode = spec_decode["mode"]
    block_size = num_draft_tokens_override or spec_decode.get("block_size", 4)
    hidden_output_name = spec_decode.get(
        "hidden_output_name",
        "target_hidden" if spec_mode == "dflash" else "pre_norm_hidden",
    )
    draft_cfg = {}
    if spec_mode == "mtp":
        draft_cfg["prefill"] = dict(
            onnx=str(
                resolve_path(
                    model_dir,
                    spec_decode.get("draft_prefill_onnx")
                    or spec_decode.get("draft_onnx"),
                )
            )
        )
        draft_cfg["decode"] = dict(
            onnx=str(
                resolve_path(
                    model_dir,
                    spec_decode.get("draft_decode_onnx")
                    or spec_decode.get("draft_onnx"),
                )
            )
        )
    else:
        draft_cfg["context"] = dict(
            onnx=str(resolve_path(model_dir, spec_decode["draft_context_onnx"]))
        )
        draft_cfg["decode"] = dict(
            onnx=str(
                resolve_path(
                    model_dir,
                    spec_decode.get("draft_decode_onnx")
                    or spec_decode.get("draft_onnx"),
                )
            )
        )

    max_context_tokens = meta_info.get("max_context_tokens")
    if max_context_tokens is None:
        kv_cache = meta_info.get("kv_cache", {})
        shape = kv_cache.get("shape")
        if isinstance(shape, list) and len(shape) >= 3:
            max_context_tokens = int(shape[2])

    tokenizer = AutoTokenizer.from_pretrained(str(hf_model_config_dir))
    token_embedding = load_token_embedding(token_embedding_file).to(dtype=dtype)
    pad_token_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )
    if pad_token_id is None:
        pad_token_id = 0

    runtime = Qwen3_5SpecDecodeONNXModel(
        prefill=dict(onnx=str(prefill_onnx)),
        decode=dict(onnx=str(decode_onnx)),
        draft=draft_cfg,
        spec_decode_mode=spec_mode,
        block_size=block_size,
        hidden_output_name=hidden_output_name,
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


def benchmark_spec_decode(
    runtime: Qwen3_5SpecDecodeONNXModel,
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
    messages = [{"role": "user", "content": prompt}]
    # messages = [{"role": "system", "content": "You are a helpful assistant."}]
    # messages.extend(turns)

    kwargs = {"enable_thinking": enable_thinking}
    if hasattr(tokenizer, "apply_chat_template"):
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **kwargs
        )
    else:
        text = "\n".join(
            [f"{m['role']}: {m['content']}" for m in messages] + ["assistant:"]
        )

    input_ids = tokenizer([text], return_tensors="pt").input_ids

    start = time.perf_counter()
    output = runtime.generate(
        input_ids,
        tokenizer,
        max_new_tokens=max_new_tokens,
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


def main(args):
    runtime, tokenizer, meta_info = load_spec_decode_runtime(
        meta_path=args.config,
        dtype=parse_dtype(args.dtype),
        device=args.device,
        exec_device=args.exec_device,
        auto_offload=not args.disable_auto_offload,
        auto_offload_max_memory=parse_auto_offload_max_memory(
            args.auto_offload_max_memory
        ),
        prefill_auto_offload_max_memory=parse_auto_offload_max_memory(
            args.prefill_auto_offload_max_memory
        ),
        decode_auto_offload_max_memory=parse_auto_offload_max_memory(
            args.decode_auto_offload_max_memory
        ),
        resource_tight_mode=args.resource_tight_mode,
        num_draft_tokens_override=(
            args.num_draft_tokens if args.num_draft_tokens > 0 else None
        ),
    )

    spec_decode = meta_info.get("spec_decode", {})
    print(f"spec_decode_mode: {spec_decode.get('mode')}")
    print(f"block_size: {runtime.block_size}")
    print(f"hidden_output_name: {runtime.hidden_output_name}")

    for _ in range(args.warmup_runs):
        benchmark_spec_decode(
            runtime,
            tokenizer,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            enable_thinking=args.enable_thinking,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            presence_penalty=args.presence_penalty,
            stream_output=False,
        )

    timings = []
    output_text = ""
    output_tokens = 0
    for _ in range(args.benchmark_runs):
        output_text, elapsed, output_tokens = benchmark_spec_decode(
            runtime,
            tokenizer,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            enable_thinking=args.enable_thinking,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            presence_penalty=args.presence_penalty,
            stream_output=True,
        )
        timings.append(elapsed)

    avg_latency = sum(timings) / max(len(timings), 1)
    toks_per_sec = output_tokens / avg_latency if avg_latency > 0 else 0.0

    print(f"\nmodel_name: {meta_info.get('model_name')}")
    print(f"quant_scheme: {meta_info.get('quant_scheme')}")
    print(f"prompt: {args.prompt}")
    print(f"output: {output_text}")
    print(f"benchmark_runs: {args.benchmark_runs}")
    print(f"avg_latency_s: {avg_latency:.4f}")
    print(f"output_tokens: {output_tokens}")
    print(f"tokens_per_second: {toks_per_sec:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Speculative decoding inference benchmark for Qwen3.5 on xh2a"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to meta.json from export (must contain spec_decode section)",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--exec_device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="fp16")
    parser.add_argument(
        "--prompt",
        type=str,
        default="Write a short poem about AI.",
    )
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--enable_thinking", action="store_true")
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--presence_penalty", type=float, default=0.0)
    parser.add_argument("--warmup_runs", type=int, default=1)
    parser.add_argument("--benchmark_runs", type=int, default=1)
    parser.add_argument("--stream_output", action="store_true")
    parser.add_argument("--disable_auto_offload", action="store_true")
    parser.add_argument("--auto_offload_max_memory", type=str, default=None)
    parser.add_argument("--prefill_auto_offload_max_memory", type=str, default=None)
    parser.add_argument("--decode_auto_offload_max_memory", type=str, default=None)
    parser.add_argument("--resource_tight_mode", action="store_true")
    parser.add_argument(
        "--num_draft_tokens",
        type=int,
        default=0,
        help="Override number of draft tokens per round (0 = use meta.json value)",
    )
    args = parser.parse_args()
    main(args)
