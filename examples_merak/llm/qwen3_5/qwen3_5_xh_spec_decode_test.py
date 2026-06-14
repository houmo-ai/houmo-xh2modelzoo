"""Canonical Merak Qwen3.5/Qwen3.5-MoE MTP/DFlash HMONNX demo.

This script is intentionally maintained under ``examples_merak`` instead of
wrapping the legacy ``examples/llm`` demo.  It loads Merak-exported
``golden_meta_info.json`` files directly and supports both MTP and DFlash
speculative decoding artifacts.

Example:
    python examples_merak/llm/qwen3_5/qwen3_5_xh_spec_decode_test.py \
        --config work_dirs/qwen3_5_9b_spec_mtp_xh2a_2k/.../golden_meta_info.json \
        --max-new-tokens 40 \
        --min-output-tokens 32
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
from transformers import AutoTokenizer

from xh_model_zoo.xh_llm.models.qwen3_5 import Qwen3_5SpecDecodeONNXModel


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


def parse_auto_offload_max_memory(max_memory_json: Optional[str]) -> Optional[dict[Any, Any]]:
    if max_memory_json is None or max_memory_json.strip() == "":
        return None
    parsed = json.loads(max_memory_json)
    if not isinstance(parsed, dict):
        raise ValueError("auto-offload max memory must be a JSON object")
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


def build_chat_input_ids(tokenizer: AutoTokenizer, prompt: str, enable_thinking: bool) -> torch.Tensor:
    messages = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template"):
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    else:
        text = "\n".join([f"{m['role']}: {m['content']}" for m in messages] + ["assistant:"])
    return tokenizer([text], return_tensors="pt").input_ids


def load_prompt(prompt: str, prompt_file: Optional[str]) -> str:
    if prompt_file:
        return Path(prompt_file).read_text(encoding="utf-8")
    return prompt


def spec_decode_from_meta(meta_info: dict, model_dir: Path) -> tuple[dict[str, dict[str, str]], str, int, str]:
    spec_decode = meta_info.get("spec_decode")
    if spec_decode is None:
        raise ValueError("golden_meta_info.json does not contain a spec_decode section")

    spec_mode = spec_decode["mode"]
    block_size = int(spec_decode.get("block_size", 4))
    hidden_output_name = spec_decode.get(
        "hidden_output_name",
        "target_hidden" if spec_mode == "dflash" else "post_norm_hidden",
    )

    draft: dict[str, dict[str, str]] = {}
    if spec_mode == "mtp":
        draft_prefill = spec_decode.get("draft_prefill_onnx") or spec_decode.get("draft_onnx")
        draft_decode = spec_decode.get("draft_decode_onnx") or spec_decode.get("draft_onnx")
        if not draft_prefill or not draft_decode:
            raise ValueError("MTP spec_decode requires draft_prefill_onnx/draft_decode_onnx or draft_onnx")
        draft["prefill"] = {"onnx": str(resolve_path(model_dir, draft_prefill))}
        draft["decode"] = {"onnx": str(resolve_path(model_dir, draft_decode))}
    elif spec_mode == "dflash":
        draft["context"] = {"onnx": str(resolve_path(model_dir, spec_decode["draft_context_onnx"]))}
        draft_context_decode = spec_decode.get("draft_context_decode_onnx")
        if draft_context_decode:
            draft["context_decode"] = {"onnx": str(resolve_path(model_dir, draft_context_decode))}
        draft_decode = spec_decode.get("draft_decode_onnx") or spec_decode.get("draft_onnx")
        if not draft_decode:
            raise ValueError("DFlash spec_decode requires draft_decode_onnx or draft_onnx")
        draft["decode"] = {"onnx": str(resolve_path(model_dir, draft_decode))}
    else:
        raise ValueError(f"Unsupported spec_decode mode: {spec_mode}")

    return draft, spec_mode, block_size, hidden_output_name


def infer_max_context_tokens(meta_info: dict) -> Optional[int]:
    max_context_tokens = meta_info.get("max_context_tokens")
    if max_context_tokens is not None:
        return int(max_context_tokens)
    kv_cache = meta_info.get("kv_cache", {})
    shape = kv_cache.get("shape")
    if isinstance(shape, list) and len(shape) >= 3:
        return int(shape[2])
    return None


def load_merak_spec_runtime(args) -> tuple[Qwen3_5SpecDecodeONNXModel, AutoTokenizer, dict]:
    meta_file = Path(args.config).resolve()
    model_dir = meta_file.parent
    meta_info = json.loads(meta_file.read_text(encoding="utf-8"))

    prefill_onnx = resolve_path(
        model_dir,
        meta_info.get("prefill_onnx")
        or meta_info.get("prefill_hmonnx")
        or meta_info["prefill_onnx_file"],
    )
    decode_onnx = resolve_path(
        model_dir,
        meta_info.get("decode_onnx")
        or meta_info.get("decode_hmonnx")
        or meta_info["decode_onnx_file"],
    )
    hf_model_config_dir = resolve_path(model_dir, meta_info["hf_config"])
    token_embedding_file = resolve_path(
        model_dir,
        meta_info.get("token_embedding_file")
        or meta_info.get("quant_embedding")
        or meta_info["token_embedding_file"],
    )
    draft, spec_mode, block_size, hidden_output_name = spec_decode_from_meta(meta_info, model_dir)

    tokenizer = AutoTokenizer.from_pretrained(str(hf_model_config_dir))
    token_embedding = load_token_embedding(token_embedding_file).to(dtype=parse_dtype(args.dtype))
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad_token_id is None:
        pad_token_id = 0

    runtime = Qwen3_5SpecDecodeONNXModel(
        prefill={"onnx": str(prefill_onnx)},
        decode={"onnx": str(decode_onnx)},
        draft=draft,
        spec_decode_mode=spec_mode,
        block_size=block_size,
        hidden_output_name=hidden_output_name,
        max_context_tokens=infer_max_context_tokens(meta_info),
        auto_offload=not args.disable_auto_offload,
        auto_offload_max_memory=parse_auto_offload_max_memory(args.auto_offload_max_memory),
        prefill_auto_offload_max_memory=parse_auto_offload_max_memory(args.prefill_auto_offload_max_memory),
        decode_auto_offload_max_memory=parse_auto_offload_max_memory(args.decode_auto_offload_max_memory),
        resource_tight_mode=args.resource_tight_mode,
        pad_token_id=pad_token_id,
        enable_cuda_graph=args.enable_cuda_graph,
        cuda_graph_modules=parse_cuda_graph_modules(args.cuda_graph_modules),
        cuda_graph_warmup_runs=args.cuda_graph_warmup_runs,
        cuda_graph_graph_warmup_runs=args.cuda_graph_graph_warmup_runs,
    )
    dtype = parse_dtype(args.dtype)
    runtime.set_input_embeddings(token_embedding)
    runtime.to(torch.device(args.device))
    runtime.set_exec_device(torch.device(args.exec_device))
    runtime.to(dtype)
    return runtime, tokenizer, meta_info


def parse_cuda_graph_modules(modules_arg: str) -> Optional[tuple[str, ...]]:
    if not modules_arg or not modules_arg.strip():
        return None
    modules = tuple(part.strip().lower() for part in modules_arg.split(",") if part.strip())
    return modules or None


def print_cuda_graph_status(runtime: Qwen3_5SpecDecodeONNXModel) -> None:
    print("cuda_graph_status:")
    for name, status in runtime.get_cuda_graph_status().items():
        reason = status.get("reason") or "-"
        backend = status.get("backend") or "-"
        print(
            f"  {name}: backend={backend} enabled={status.get('enabled')} "
            f"captured={status.get('captured')} reason={reason}"
        )


def ensure_min_output_tokens(output_tokens: int, min_output_tokens: int) -> None:
    if min_output_tokens > 0 and output_tokens < min_output_tokens:
        raise RuntimeError(f"Generated {output_tokens} tokens, expected at least {min_output_tokens}.")


def run_once(runtime, tokenizer, prompt: str, args) -> tuple[str, dict[str, Any], float, int]:
    input_ids = build_chat_input_ids(tokenizer, prompt, args.enable_thinking)
    start = time.perf_counter()
    output, stats = runtime.generate(
        input_ids,
        tokenizer,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        presence_penalty=args.presence_penalty,
        stream_output=args.stream_output,
        return_stats=True,
    )
    elapsed = time.perf_counter() - start
    cleaned = postprocess_chat_output(output, enable_thinking=args.enable_thinking)
    output_tokens = len(tokenizer.encode(cleaned, add_special_tokens=False))
    return cleaned, stats, elapsed, output_tokens


def main(args) -> None:
    prompt = load_prompt(args.prompt, args.prompt_file)
    runtime, tokenizer, meta_info = load_merak_spec_runtime(args)
    spec_decode = meta_info.get("spec_decode", {})

    print(f"spec_decode_mode: {spec_decode.get('mode')}")
    print(f"block_size: {runtime.block_size}")
    print(f"hidden_output_name: {runtime.hidden_output_name}")

    for _ in range(args.warmup_runs):
        run_once(runtime, tokenizer, prompt, args)

    timings = []
    output_text = ""
    output_tokens = 0
    stats: dict[str, Any] = {}
    for _ in range(args.benchmark_runs):
        output_text, stats, elapsed, output_tokens = run_once(runtime, tokenizer, prompt, args)
        timings.append(elapsed)

    ensure_min_output_tokens(output_tokens, args.min_output_tokens)

    avg_latency = sum(timings) / max(len(timings), 1)
    toks_per_sec = output_tokens / avg_latency if avg_latency > 0 else 0.0
    accepted = int(stats.get("accepted_drafts_total", 0) or 0)
    draft_tokens = int(stats.get("draft_tokens_total", 0) or 0)
    accept_rate = accepted / draft_tokens if draft_tokens > 0 else 0.0

    print(f"\nmodel_name: {meta_info.get('model_name')}")
    print(f"prompt: {prompt}")
    print(f"output: {output_text}")
    print(f"benchmark_runs: {args.benchmark_runs}")
    print(f"avg_latency_s: {avg_latency:.4f}")
    print(f"output_tokens: {output_tokens}")
    print(f"tokens_per_second: {toks_per_sec:.4f}")
    print(f"draft_tokens: {draft_tokens}")
    print(f"accepted_drafts: {accepted}")
    print(f"accept_rate: {accept_rate:.4f}")
    print_cuda_graph_status(runtime)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Canonical Merak Qwen3.5 spec decode HMONNX demo")
    parser.add_argument("--config", type=str, required=True, help="Path to Merak golden_meta_info.json")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--exec-device", "--exec_device", dest="exec_device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="fp16")
    parser.add_argument("--prompt", type=str, default="写一首关于 AI的诗")
    parser.add_argument("--prompt-file", "--prompt_file", dest="prompt_file", type=str, default=None)
    parser.add_argument("--max-new-tokens", "--max_new_tokens", dest="max_new_tokens", type=int, default=128)
    parser.add_argument(
        "--min-output-tokens",
        "--min_output_tokens",
        dest="min_output_tokens",
        type=int,
        default=0,
        help="Fail if the generated continuation is shorter than this many tokens",
    )
    parser.add_argument("--enable-thinking", "--enable_thinking", dest="enable_thinking", action="store_true")
    parser.add_argument("--do-sample", "--do_sample", dest="do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", "--top_p", dest="top_p", type=float, default=1.0)
    parser.add_argument("--top-k", "--top_k", dest="top_k", type=int, default=0)
    parser.add_argument(
        "--repetition-penalty",
        "--repetition_penalty",
        dest="repetition_penalty",
        type=float,
        default=1.0,
    )
    parser.add_argument("--presence-penalty", "--presence_penalty", dest="presence_penalty", type=float, default=0.0)
    parser.add_argument("--warmup-runs", "--warmup_runs", dest="warmup_runs", type=int, default=0)
    parser.add_argument("--benchmark-runs", "--benchmark_runs", dest="benchmark_runs", type=int, default=1)
    parser.add_argument("--stream-output", "--stream_output", dest="stream_output", action="store_true")
    parser.add_argument(
        "--disable-auto-offload",
        "--disable_auto_offload",
        dest="disable_auto_offload",
        action="store_true",
    )
    parser.add_argument(
        "--auto-offload-max-memory",
        "--auto_offload_max_memory",
        dest="auto_offload_max_memory",
        type=str,
    )
    parser.add_argument(
        "--prefill-auto-offload-max-memory",
        "--prefill_auto_offload_max_memory",
        dest="prefill_auto_offload_max_memory",
        type=str,
    )
    parser.add_argument(
        "--decode-auto-offload-max-memory",
        "--decode_auto_offload_max_memory",
        dest="decode_auto_offload_max_memory",
        type=str,
    )
    parser.add_argument(
        "--resource-tight-mode",
        "--resource_tight_mode",
        dest="resource_tight_mode",
        action="store_true",
    )
    parser.add_argument("--enable-cuda-graph", "--enable_cuda_graph", dest="enable_cuda_graph", action="store_true")
    parser.add_argument(
        "--cuda-graph-modules",
        "--cuda_graph_modules",
        dest="cuda_graph_modules",
        type=str,
        default="",
        help="Comma-separated session names: prefill,decode,draft_prefill,draft_context,draft_decode",
    )
    parser.add_argument(
        "--cuda-graph-warmup-runs",
        "--cuda_graph_warmup_runs",
        dest="cuda_graph_warmup_runs",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--cuda-graph-graph-warmup-runs",
        "--cuda_graph_graph_warmup_runs",
        dest="cuda_graph_graph_warmup_runs",
        type=int,
        default=6,
    )
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
