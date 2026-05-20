"""Runtime loader and benchmark for speculative decoding with HMONNX on xh2a."""

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoTokenizer

from xh_model_zoo.xh_llm.models.qwen3_5 import (
    Qwen3_5ONNXModel,
    Qwen3_5SpecDecodeONNXModel,
)
from qwen3_5_spec_decode_metrics import (
    run_dense_spec_generate_once,
    run_dense_target_baseline_from_spec,
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
    enable_cuda_graph: bool = False,
    cuda_graph_modules: Optional[Tuple[str, ...]] = None,
    cuda_graph_warmup_runs: int = 3,
    cuda_graph_graph_warmup_runs: int = 6,
) -> Tuple[Qwen3_5SpecDecodeONNXModel, AutoTokenizer, dict]:
    """Load speculative decode runtime from meta.json.

    meta.json must contain a `spec_decode` section with:
    - mode: 'dflash' or 'mtp'
    - draft_prefill_onnx / draft_context_onnx / draft_context_decode_onnx / draft_decode_onnx
    - block_size: number of draft tokens per round
    - hidden_output_name: name of hidden state output
    """
    meta_file = Path(meta_path).resolve()
    model_dir = meta_file.parent
    meta_info = json.load(open(meta_file, "r", encoding="utf-8"))

    prefill_onnx = resolve_path(model_dir, meta_info.get("prefill_onnx") or meta_info["prefill_onnx_file"])
    decode_onnx = resolve_path(model_dir, meta_info.get("decode_onnx") or meta_info["decode_onnx_file"])
    hf_model_config_dir = resolve_path(model_dir, meta_info["hf_config"])
    token_embedding_file = resolve_path(model_dir, meta_info["token_embedding_file"])

    spec_decode = meta_info.get("spec_decode")
    if spec_decode is None:
        raise ValueError(
            "meta.json does not contain spec_decode section. Export with --spec_decode_mode dflash|mtp first."
        )

    spec_mode = spec_decode["mode"]
    block_size = num_draft_tokens_override or spec_decode.get("block_size", 4)
    hidden_output_name = spec_decode.get(
        "hidden_output_name",
        "target_hidden" if spec_mode == "dflash" else "post_norm_hidden",
    )
    draft_cfg = {}
    if spec_mode == "mtp":
        draft_cfg["prefill"] = dict(
            onnx=str(
                resolve_path(
                    model_dir,
                    spec_decode.get("draft_prefill_onnx") or spec_decode.get("draft_onnx"),
                )
            )
        )
        draft_cfg["decode"] = dict(
            onnx=str(
                resolve_path(
                    model_dir,
                    spec_decode.get("draft_decode_onnx") or spec_decode.get("draft_onnx"),
                )
            )
        )
    else:
        draft_cfg["context"] = dict(onnx=str(resolve_path(model_dir, spec_decode["draft_context_onnx"])))
        draft_context_decode = spec_decode.get("draft_context_decode_onnx")
        if draft_context_decode:
            draft_cfg["context_decode"] = dict(onnx=str(resolve_path(model_dir, draft_context_decode)))
        draft_cfg["decode"] = dict(
            onnx=str(
                resolve_path(
                    model_dir,
                    spec_decode.get("draft_decode_onnx") or spec_decode.get("draft_onnx"),
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
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
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
        enable_cuda_graph=enable_cuda_graph,
        cuda_graph_modules=cuda_graph_modules,
        cuda_graph_warmup_runs=cuda_graph_warmup_runs,
        cuda_graph_graph_warmup_runs=cuda_graph_graph_warmup_runs,
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
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kwargs)
    else:
        text = "\n".join([f"{m['role']}: {m['content']}" for m in messages] + ["assistant:"])

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


def print_cuda_graph_status(runtime: Qwen3_5SpecDecodeONNXModel) -> None:
    print("cuda_graph_status:")
    for name, status in runtime.get_cuda_graph_status().items():
        reason = status.get("reason") or "-"
        backend = status.get("backend") or "-"
        print(
            f"  {name}: backend={backend} enabled={status.get('enabled')} "
            f"captured={status.get('captured')} reason={reason}"
        )


def parse_cuda_graph_modules(modules_arg: str) -> Optional[Tuple[str, ...]]:
    if not modules_arg or not modules_arg.strip():
        return None
    modules = tuple(part.strip().lower() for part in modules_arg.split(",") if part.strip())
    return modules or None


def load_prompt(prompt: str, prompt_file: Optional[str]) -> str:
    if prompt_file:
        return Path(prompt_file).read_text(encoding="utf-8")
    return prompt


def compare_texts(reference: str, candidate: str) -> Dict[str, Any]:
    exact_match = reference == candidate
    first_diff = None
    if not exact_match:
        for idx, (lhs, rhs) in enumerate(zip(reference, candidate)):
            if lhs != rhs:
                first_diff = idx
                break
        if first_diff is None:
            first_diff = min(len(reference), len(candidate))
    return {
        "exact_match": exact_match,
        "reference_length": len(reference),
        "candidate_length": len(candidate),
        "first_diff_index": first_diff,
        "reference_snippet": (reference[max(0, first_diff - 80) : first_diff + 160] if first_diff is not None else ""),
        "candidate_snippet": (candidate[max(0, first_diff - 80) : first_diff + 160] if first_diff is not None else ""),
    }


def run_single_case(
    *,
    args,
    prompt: str,
    enable_cuda_graph: bool,
    cuda_graph_modules: Optional[Tuple[str, ...]],
    baseline_only: bool,
) -> Dict[str, Any]:
    runtime, tokenizer, meta_info = load_spec_decode_runtime(
        meta_path=args.config,
        dtype=parse_dtype(args.dtype),
        device=args.device,
        exec_device=args.exec_device,
        auto_offload=not args.disable_auto_offload,
        auto_offload_max_memory=parse_auto_offload_max_memory(args.auto_offload_max_memory),
        prefill_auto_offload_max_memory=parse_auto_offload_max_memory(args.prefill_auto_offload_max_memory),
        decode_auto_offload_max_memory=parse_auto_offload_max_memory(args.decode_auto_offload_max_memory),
        resource_tight_mode=args.resource_tight_mode,
        num_draft_tokens_override=(args.num_draft_tokens if args.num_draft_tokens > 0 else None),
        enable_cuda_graph=enable_cuda_graph,
        cuda_graph_modules=cuda_graph_modules,
        cuda_graph_warmup_runs=args.cuda_graph_warmup_runs,
        cuda_graph_graph_warmup_runs=args.cuda_graph_graph_warmup_runs,
    )
    try:
        spec_decode = meta_info.get("spec_decode", {})
        if baseline_only:
            result = run_dense_target_baseline_from_spec(
                runtime,
                tokenizer,
                prompt,
                args.max_new_tokens,
                enable_thinking=args.enable_thinking,
                repetition_penalty=args.repetition_penalty,
                presence_penalty=args.presence_penalty,
            )
        else:
            result = run_dense_spec_generate_once(
                runtime=runtime,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
                enable_thinking=args.enable_thinking,
                repetition_penalty=args.repetition_penalty,
                presence_penalty=args.presence_penalty,
            )
        return {
            "mode": "baseline" if baseline_only else "spec",
            "enable_cuda_graph": enable_cuda_graph,
            "cuda_graph_modules": list(cuda_graph_modules) if cuda_graph_modules is not None else None,
            "spec_decode_mode": spec_decode.get("mode"),
            "block_size": runtime.block_size,
            "hidden_output_name": runtime.hidden_output_name,
            "cuda_graph_status": runtime.get_cuda_graph_status(),
            **result,
        }
    finally:
        del runtime
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def print_compare_result(label: str, comparison: Dict[str, Any]) -> None:
    print(f"{label}: exact_match={comparison['exact_match']}")
    if comparison["exact_match"]:
        return
    print(f"{label}_first_diff_index: {comparison['first_diff_index']}")
    print(f"{label}_reference_snippet: {comparison['reference_snippet']}")
    print(f"{label}_candidate_snippet: {comparison['candidate_snippet']}")


def run_compare(args) -> None:
    prompt = load_prompt(args.prompt, args.prompt_file)
    cuda_graph_modules = parse_cuda_graph_modules(args.cuda_graph_modules)

    payload: Dict[str, Any] = {
        "prompt": prompt,
        "max_new_tokens": args.max_new_tokens,
        "enable_thinking": args.enable_thinking,
        "baseline_only": args.baseline_only,
    }

    if args.compare_to_no_cuda_graph:
        reference = run_single_case(
            args=args,
            prompt=prompt,
            enable_cuda_graph=False,
            cuda_graph_modules=None,
            baseline_only=args.baseline_only,
        )
        candidate = run_single_case(
            args=args,
            prompt=prompt,
            enable_cuda_graph=args.enable_cuda_graph,
            cuda_graph_modules=cuda_graph_modules,
            baseline_only=args.baseline_only,
        )
        comparison = compare_texts(reference["text"], candidate["text"])
        payload["reference"] = reference
        payload["candidate"] = candidate
        payload["comparison"] = comparison

        print(f"reference_mode: {reference['mode']}")
        print(f"candidate_mode: {candidate['mode']}")
        print(
            f"candidate_cuda_graph_modules: "
            f"{candidate['cuda_graph_modules'] or 'all' if candidate['enable_cuda_graph'] else 'disabled'}"
        )
        print_compare_result("compare_to_no_cuda_graph", comparison)

    if args.compare_baseline:
        baseline = run_single_case(
            args=args,
            prompt=prompt,
            enable_cuda_graph=args.enable_cuda_graph,
            cuda_graph_modules=cuda_graph_modules,
            baseline_only=True,
        )
        spec = run_single_case(
            args=args,
            prompt=prompt,
            enable_cuda_graph=args.enable_cuda_graph,
            cuda_graph_modules=cuda_graph_modules,
            baseline_only=False,
        )
        comparison = compare_texts(baseline["text"], spec["text"])
        payload["baseline"] = baseline
        payload["spec"] = spec
        payload["baseline_vs_spec"] = comparison

        print(
            f"baseline_vs_spec_cuda_graph_modules: "
            f"{spec['cuda_graph_modules'] or 'all' if spec['enable_cuda_graph'] else 'disabled'}"
        )
        print_compare_result("baseline_vs_spec", comparison)

    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run_save_golden(args):
    """Export draft session golden data via HMONNXGrapInference's built-in save_golden mechanism.

    cuda_graph is forced off because HMONNXCUDAGraphInference does not implement save_golden.
    Each draft session call writes its npy bundle to <root>/<session_name>/step_<N>/, since the
    platform's forward() calls shutil.rmtree() if it finds existing files in save_golden_dir.
    """
    prompt = load_prompt(args.prompt, args.prompt_file)
    max_new_tokens = args.max_new_tokens

    runtime, tokenizer, meta_info = load_spec_decode_runtime(
        meta_path=args.config,
        dtype=parse_dtype(args.dtype),
        device=args.device,
        exec_device=args.exec_device,
        auto_offload=not args.disable_auto_offload,
        auto_offload_max_memory=parse_auto_offload_max_memory(args.auto_offload_max_memory),
        prefill_auto_offload_max_memory=parse_auto_offload_max_memory(args.prefill_auto_offload_max_memory),
        decode_auto_offload_max_memory=parse_auto_offload_max_memory(args.decode_auto_offload_max_memory),
        resource_tight_mode=args.resource_tight_mode,
        num_draft_tokens_override=(args.num_draft_tokens if args.num_draft_tokens > 0 else None),
        enable_cuda_graph=False,
        cuda_graph_modules=None,
    )

    golden_root = Path(args.save_golden).resolve()
    golden_root.mkdir(parents=True, exist_ok=True)

    session_to_dir_name: Dict[int, str] = {}
    candidates = [
        ("draft_prefill", runtime.draft_prefill_session),
        ("draft_context", runtime.draft_context_session),
        ("draft_context_decode", runtime.draft_context_decode_session),
        ("draft_decode", runtime.draft_decode_session),
    ]
    for name, session in candidates:
        if session is None or id(session) in session_to_dir_name:
            continue
        session_to_dir_name[id(session)] = name
        session.save_golden = True

    step_counters: Dict[int, int] = {sid: 0 for sid in session_to_dir_name}
    original_run_draft_session = runtime._run_draft_session

    def golden_run_draft_session(session, input_feed):
        sid = id(session)
        if sid in session_to_dir_name:
            step_dir = golden_root / session_to_dir_name[sid] / f"step_{step_counters[sid]:04d}"
            step_dir.mkdir(parents=True, exist_ok=True)
            session.save_golden_dir = str(step_dir)
            step_counters[sid] += 1
        return original_run_draft_session(session, input_feed)

    runtime._run_draft_session = golden_run_draft_session

    output_text, _, output_tokens = benchmark_spec_decode(
        runtime,
        tokenizer,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        enable_thinking=args.enable_thinking,
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
        top_k=1,
        repetition_penalty=args.repetition_penalty,
        presence_penalty=args.presence_penalty,
        stream_output=False,
    )

    spec_decode = meta_info.get("spec_decode", {})
    print(f"\nmodel_name: {meta_info.get('model_name')}")
    print(f"spec_decode_mode: {spec_decode.get('mode')}")
    print(f"max_new_tokens: {max_new_tokens}")
    print(f"output_tokens: {output_tokens}")
    print(f"output: {output_text[:200]}")
    print(f"\ngolden_root: {golden_root}")
    for sid, dir_name in session_to_dir_name.items():
        print(f"  {dir_name}: {step_counters[sid]} steps")


def main(args):
    if args.save_golden:
        run_save_golden(args)
        return

    if args.compare_to_no_cuda_graph or args.compare_baseline:
        run_compare(args)
        return

    prompt = load_prompt(args.prompt, args.prompt_file)
    cuda_graph_modules = parse_cuda_graph_modules(args.cuda_graph_modules)
    runtime, tokenizer, meta_info = load_spec_decode_runtime(
        meta_path=args.config,
        dtype=parse_dtype(args.dtype),
        device=args.device,
        exec_device=args.exec_device,
        auto_offload=not args.disable_auto_offload,
        auto_offload_max_memory=parse_auto_offload_max_memory(args.auto_offload_max_memory),
        prefill_auto_offload_max_memory=parse_auto_offload_max_memory(args.prefill_auto_offload_max_memory),
        decode_auto_offload_max_memory=parse_auto_offload_max_memory(args.decode_auto_offload_max_memory),
        resource_tight_mode=args.resource_tight_mode,
        num_draft_tokens_override=(args.num_draft_tokens if args.num_draft_tokens > 0 else None),
        enable_cuda_graph=args.enable_cuda_graph,
        cuda_graph_modules=cuda_graph_modules,
        cuda_graph_warmup_runs=args.cuda_graph_warmup_runs,
        cuda_graph_graph_warmup_runs=args.cuda_graph_graph_warmup_runs,
    )

    spec_decode = meta_info.get("spec_decode", {})
    print(f"spec_decode_mode: {spec_decode.get('mode')}")
    print(f"block_size: {runtime.block_size}")
    print(f"hidden_output_name: {runtime.hidden_output_name}")

    for _ in range(args.warmup_runs):
        benchmark_spec_decode(
            runtime,
            tokenizer,
            prompt=prompt,
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
            prompt=prompt,
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
        timings.append(elapsed)

    avg_latency = sum(timings) / max(len(timings), 1)
    toks_per_sec = output_tokens / avg_latency if avg_latency > 0 else 0.0

    print(f"\nmodel_name: {meta_info.get('model_name')}")
    print(f"quant_scheme: {meta_info.get('quant_scheme')}")
    print(f"enable_cuda_graph: {args.enable_cuda_graph}")
    print(
        f"cuda_graph_modules: {list(cuda_graph_modules) if cuda_graph_modules is not None else 'all' if args.enable_cuda_graph else 'disabled'}"
    )
    print(f"prompt: {prompt}")
    print(f"output: {output_text}")
    print(f"benchmark_runs: {args.benchmark_runs}")
    print(f"avg_latency_s: {avg_latency:.4f}")
    print(f"output_tokens: {output_tokens}")
    print(f"tokens_per_second: {toks_per_sec:.4f}")
    print_cuda_graph_status(runtime)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Speculative decoding inference benchmark for Qwen3.5 on xh2a")
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
        default="写一首关于 AI的诗",
    )
    parser.add_argument("--prompt_file", type=str, default=None)
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
    parser.add_argument("--enable_cuda_graph", action="store_true")
    parser.add_argument(
        "--cuda_graph_modules",
        type=str,
        default="",
        help="Comma-separated session names: prefill,decode,draft_prefill,draft_context,draft_decode",
    )
    parser.add_argument("--cuda_graph_warmup_runs", type=int, default=3)
    parser.add_argument("--cuda_graph_graph_warmup_runs", type=int, default=6)
    parser.add_argument("--baseline_only", action="store_true")
    parser.add_argument("--compare_to_no_cuda_graph", action="store_true")
    parser.add_argument("--compare_baseline", action="store_true")
    parser.add_argument("--output_json", type=str, default=None)
    parser.add_argument(
        "--save_golden",
        type=str,
        default=None,
        help=(
            "Save draft model golden data (npy) to this directory using "
            "HMONNXGrapInference's built-in save_golden. Forces cuda_graph off."
        ),
    )
    args = parser.parse_args()
    main(args)
