"""Runtime quick-test helpers for Qwen3.5/Qwen3.6 HMONNX exports.

The workflow implementation keeps quant/export concerns separate from runtime inference.
This module provides the small runtime layer that examples and validation
scripts use after export: find the HMONNX meta file, run a quick chat, and
summarize MTP/DFlash speculative-decoding acceptance metrics.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class HMONNXQuickTestResult:
    meta_file: str
    output_text: str
    output_tokens: int
    latency_s: float
    tokens_per_second: float
    spec_decode_mode: str | None = None
    block_size: int | None = None
    num_rounds: int = 0
    draft_tokens_total: int = 0
    accepted_drafts_total: int = 0
    accept_rate: float = 0.0
    avg_accepted_per_round: float = 0.0
    stats: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def find_hmonnx_meta_file(export_result_or_path: Any) -> str:
    """Resolve the golden_meta_info.json used by HMONNX runtime inference."""
    if hasattr(export_result_or_path, "work_dir"):
        path = Path(export_result_or_path.work_dir)
    else:
        path = Path(export_result_or_path)

    if path.is_file():
        return str(path)
    if not path.is_dir():
        raise FileNotFoundError(f"HMONNX export path does not exist: {path}")

    meta_files: list[Path] = []
    for child in path.iterdir():
        if not child.is_dir() or not child.name.startswith("hmquant"):
            continue
        meta_file = child / "golden_meta_info.json"
        if meta_file.is_file():
            meta_files.append(meta_file)

    if not meta_files:
        raise FileNotFoundError(f"No hmquant*/golden_meta_info.json found under {path}")
    if len(meta_files) > 1:
        found = ", ".join(str(item) for item in meta_files)
        raise ValueError(f"Found multiple HMONNX meta files under {path}: {found}")
    return str(meta_files[0])


def quick_test_hmonnx(
    export_result_or_meta_file: Any,
    *,
    prompt: str = "用中文一句话说明模型是否可用。",
    image_path: str | None = None,
    device: str | None = None,
    max_new_tokens: int = 64,
    do_sample: bool = False,
    **kwargs: Any,
) -> HMONNXQuickTestResult:
    """Run the right quick-test path for a full/visual/MTP/DFlash HMONNX export."""
    meta_file = find_hmonnx_meta_file(export_result_or_meta_file)
    meta_info = _load_meta_info(meta_file)
    spec_decode = meta_info.get("spec_decode")
    spec_mode = spec_decode.get("mode") if isinstance(spec_decode, dict) else None
    if spec_mode in {"mtp", "dflash"}:
        return spec_decode_generate(
            meta_file=meta_file,
            prompt=prompt,
            device=device or "cuda:0",
            exec_device=kwargs.pop("exec_device", device or "cuda:0"),
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            **kwargs,
        )
    return hmonnx_generate(
        meta_file=meta_file,
        prompt=prompt,
        image_path=image_path,
        device=device,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        **kwargs,
    )


def hmonnx_generate(
    *,
    meta_file: str | Path,
    prompt: str = "用中文一句话说明模型是否可用。",
    image_path: str | None = None,
    device: str | None = None,
    max_new_tokens: int = 64,
    do_sample: bool = False,
    think: bool = False,
    fast: bool = False,
    debug: bool = False,
    golden: bool = False,
    min_output_tokens: int = 0,
    auto_offload: bool = False,
    cuda_graph: bool = False,
    device_map: None = None,
) -> HMONNXQuickTestResult:
    """Run a lightweight HMONNX generate smoke test for full or visual exports."""
    import torch
    from transformers import TextStreamer

    from xhmodel_merak.xh_llm import AutoLLMHONNXModel, LLMInferenceContextManager
    from xhquant.api import get_xhquant_logger, xhquant_init
    from xhquant.utils import ContextManagers, MemoryTracker, TimeProfiler

    resolved_meta = str(meta_file)
    xhquant_init(None, debug)
    logger = get_xhquant_logger()
    hmonnx_model = AutoLLMHONNXModel.from_pretrained(
        resolved_meta,
        enable_golden=golden,
        enable_cuda_graph=cuda_graph,
        enable_auto_offload=auto_offload,
        device_map=device_map,
    )
    logger.info(f"Resolved HMONNX model type: {type(hmonnx_model).__name__}")

    if auto_offload and hasattr(hmonnx_model, "enable_auto_offload"):
        hmonnx_model.enable_auto_offload = True

    runtime_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    loaded_prompt = _load_prompt(prompt)
    use_multimodal = bool(image_path) and Path(str(image_path)).exists() and hasattr(hmonnx_model, "get_tf_processor")
    if use_multimodal:
        tokenizer, model_inputs = _build_multimodal_inputs(
            hmonnx_model,
            loaded_prompt,
            str(image_path),
            think,
            runtime_device,
        )
    else:
        tokenizer, model_inputs = _build_text_inputs(hmonnx_model, loaded_prompt, think, runtime_device)

    streamer = TextStreamer(tokenizer)
    hmonnx_model.to(runtime_device)
    if fast and hasattr(hmonnx_model, "to_fast"):
        hmonnx_model.to_fast()
    if golden and hasattr(hmonnx_model, "enable_golden"):
        hmonnx_model.enable_golden = True

    contexts = [
        TimeProfiler("hmonnx_generate", logger),
        MemoryTracker(device=runtime_device, name="generate", logger=logger),
        LLMInferenceContextManager(hmonnx_model, devices=[runtime_device]),
    ]
    max_tokens = 2 if golden else max_new_tokens
    start = time.perf_counter()
    with ContextManagers(contexts):
        generated_ids = hmonnx_model.generate(
            **model_inputs,
            max_new_tokens=max_tokens,
            streamer=streamer,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id,
        )
    latency_s = time.perf_counter() - start

    output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()
    output_text = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
    if min_output_tokens > 0 and len(output_ids) < min_output_tokens:
        raise RuntimeError(f"Generated {len(output_ids)} tokens, expected at least {min_output_tokens}.")
    logger.info(f"{'-' * 20} content {'-' * 20}")
    logger.info(output_text)
    return HMONNXQuickTestResult(
        meta_file=resolved_meta,
        output_text=output_text,
        output_tokens=len(output_ids),
        latency_s=latency_s,
        tokens_per_second=_tokens_per_second(len(output_ids), latency_s),
    )


def spec_decode_generate(
    *,
    meta_file: str | Path,
    prompt: str = "写一首关于 AI 的诗",
    prompt_file: str | None = None,
    device: str = "cuda:0",
    exec_device: str = "cuda:0",
    dtype: str = "fp16",
    max_new_tokens: int = 128,
    min_output_tokens: int = 0,
    enable_thinking: bool = False,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 0,
    repetition_penalty: float = 1.0,
    presence_penalty: float = 0.0,
    warmup_runs: int = 0,
    benchmark_runs: int = 1,
    stream_output: bool = False,
    disable_auto_offload: bool = False,
    auto_offload_max_memory: str | None = None,
    prefill_auto_offload_max_memory: str | None = None,
    decode_auto_offload_max_memory: str | None = None,
    resource_tight_mode: bool = False,
    enable_cuda_graph: bool = False,
    cuda_graph_modules: str = "",
    cuda_graph_warmup_runs: int = 3,
    cuda_graph_graph_warmup_runs: int = 6,
    golden: bool = False,
) -> HMONNXQuickTestResult:
    """Run MTP/DFlash speculative decoding and return acceptance metrics."""
    runtime, tokenizer, meta_info = _load_merak_spec_runtime(
        meta_file=meta_file,
        device=device,
        exec_device=exec_device,
        dtype=dtype,
        disable_auto_offload=disable_auto_offload,
        auto_offload_max_memory=auto_offload_max_memory,
        prefill_auto_offload_max_memory=prefill_auto_offload_max_memory,
        decode_auto_offload_max_memory=decode_auto_offload_max_memory,
        resource_tight_mode=resource_tight_mode,
        enable_cuda_graph=enable_cuda_graph,
        cuda_graph_modules=cuda_graph_modules,
        cuda_graph_warmup_runs=cuda_graph_warmup_runs,
        cuda_graph_graph_warmup_runs=cuda_graph_graph_warmup_runs,
    )
    input_prompt = _load_prompt(prompt_file or prompt)
    if prompt_file:
        input_prompt = Path(prompt_file).read_text(encoding="utf-8")

    for _ in range(warmup_runs):
        _run_spec_decode_once(
            runtime,
            tokenizer,
            input_prompt,
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

    if golden:
        set_draft_golden = getattr(runtime, "set_spec_draft_golden", None)
        if not callable(set_draft_golden):
            raise RuntimeError(f"{type(runtime).__name__} does not support spec draft golden dumping.")
        set_draft_golden(True, reset_step=True)

    timings: list[float] = []
    output_text = ""
    stats: dict[str, Any] = {}
    output_tokens = 0
    runs = max(benchmark_runs, 1)
    for _ in range(runs):
        output_text, stats, elapsed, output_tokens = _run_spec_decode_once(
            runtime,
            tokenizer,
            input_prompt,
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
        timings.append(elapsed)

    if min_output_tokens > 0 and output_tokens < min_output_tokens:
        raise RuntimeError(f"Generated {output_tokens} tokens, expected at least {min_output_tokens}.")

    spec_decode = meta_info.get("spec_decode", {})
    result = build_spec_decode_result(
        meta_file=meta_file,
        output_text=output_text,
        output_tokens=output_tokens,
        latency_s=sum(timings) / max(len(timings), 1),
        spec_decode_mode=spec_decode.get("mode"),
        block_size=getattr(runtime, "block_size", spec_decode.get("block_size")),
        stats=stats,
    )
    if hasattr(runtime, "get_cuda_graph_status"):
        result.stats["cuda_graph_status"] = runtime.get_cuda_graph_status()
    return result


def build_spec_decode_result(
    *,
    meta_file: str | Path,
    output_text: str,
    output_tokens: int,
    latency_s: float,
    spec_decode_mode: str | None,
    block_size: int | None,
    stats: dict[str, Any],
) -> HMONNXQuickTestResult:
    draft_tokens = int(stats.get("draft_tokens_total", 0) or 0)
    accepted = int(stats.get("accepted_drafts_total", 0) or 0)
    accept_rate = accepted / draft_tokens if draft_tokens > 0 else 0.0
    return HMONNXQuickTestResult(
        meta_file=str(meta_file),
        output_text=output_text,
        output_tokens=output_tokens,
        latency_s=latency_s,
        tokens_per_second=_tokens_per_second(output_tokens, latency_s),
        spec_decode_mode=spec_decode_mode,
        block_size=int(block_size) if block_size is not None else None,
        num_rounds=int(stats.get("num_rounds", 0) or 0),
        draft_tokens_total=draft_tokens,
        accepted_drafts_total=accepted,
        accept_rate=accept_rate,
        avg_accepted_per_round=float(stats.get("avg_accepted_per_round", 0.0) or 0.0),
        stats=dict(stats),
    )


def print_quick_test_result(result: HMONNXQuickTestResult) -> None:
    print(f"hmonnx_meta_file: {result.meta_file}")
    print(f"output: {result.output_text}")
    print(f"output_tokens: {result.output_tokens}")
    print(f"latency_s: {result.latency_s:.4f}")
    print(f"tokens_per_second: {result.tokens_per_second:.4f}")
    if result.spec_decode_mode:
        print(f"spec_decode_mode: {result.spec_decode_mode}")
        print(f"block_size: {result.block_size}")
        print(f"draft_tokens: {result.draft_tokens_total}")
        print(f"accepted_drafts: {result.accepted_drafts_total}")
        print(f"accept_rate: {result.accept_rate:.4f}")
        print(f"avg_accepted_per_round: {result.avg_accepted_per_round:.4f}")


def _load_meta_info(meta_file: str | Path) -> dict[str, Any]:
    return json.loads(Path(meta_file).read_text(encoding="utf-8"))


def _load_prompt(prompt: str) -> str:
    path = Path(prompt)
    if path.is_file():
        return path.read_text(encoding="utf-8")
    return prompt


def _build_text_inputs(hmonnx_model: Any, prompt: str, think: bool, device: str):
    tokenizer = hmonnx_model.get_tokenizer()
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=think)
    return tokenizer, tokenizer([text], return_tensors="pt", truncation=True).to(device)


def _build_multimodal_inputs(hmonnx_model: Any, prompt: str, image_path: str, think: bool, device: str):
    processor = hmonnx_model.get_tf_processor()
    tokenizer = processor.tokenizer
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt},
            ],
        },
    ]
    return tokenizer, processor.apply_chat_template(messages, enable_thinking=think).to(device)


def _tokens_per_second(output_tokens: int, latency_s: float) -> float:
    return output_tokens / latency_s if latency_s > 0 else 0.0


def _parse_dtype(dtype_name: str):
    import torch

    dtype_map = {
        "fp16": torch.float16,
        "float16": torch.float16,
        "half": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
    }
    key = dtype_name.strip().lower()
    if key not in dtype_map:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    return dtype_map[key]


def _parse_auto_offload_max_memory(max_memory_json: str | None) -> dict[Any, Any] | None:
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


def _resolve_path(base_dir: Path, path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def _load_token_embedding(embed_path: Path):
    import torch
    import torch.nn as nn

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


def _postprocess_chat_output(text: str, enable_thinking: bool) -> str:
    output = text.strip()
    if not enable_thinking:
        output = re.sub(r"<think>[\s\S]*?</think>", "", output)
        output = output.replace("<think>", "").replace("</think>", "")
    return output.strip()


def _build_chat_input_ids(tokenizer: Any, prompt: str, enable_thinking: bool):
    messages = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template"):
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    else:
        text = "\n".join([f"{item['role']}: {item['content']}" for item in messages] + ["assistant:"])
    return tokenizer([text], return_tensors="pt").input_ids


def _spec_decode_from_meta(
    meta_info: dict[str, Any],
    model_dir: Path,
) -> tuple[dict[str, dict[str, str]], str, int, str]:
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
        draft["prefill"] = {"onnx": str(_resolve_path(model_dir, draft_prefill))}
        draft["decode"] = {"onnx": str(_resolve_path(model_dir, draft_decode))}
    elif spec_mode == "dflash":
        draft["context"] = {"onnx": str(_resolve_path(model_dir, spec_decode["draft_context_onnx"]))}
        draft_context_decode = spec_decode.get("draft_context_decode_onnx")
        if draft_context_decode:
            draft["context_decode"] = {"onnx": str(_resolve_path(model_dir, draft_context_decode))}
        draft_decode = spec_decode.get("draft_decode_onnx") or spec_decode.get("draft_onnx")
        if not draft_decode:
            raise ValueError("DFlash spec_decode requires draft_decode_onnx or draft_onnx")
        draft["decode"] = {"onnx": str(_resolve_path(model_dir, draft_decode))}
    else:
        raise ValueError(f"Unsupported spec_decode mode: {spec_mode}")
    return draft, spec_mode, block_size, hidden_output_name


def _infer_max_context_tokens(meta_info: dict[str, Any]) -> int | None:
    max_context_tokens = meta_info.get("max_context_tokens")
    if max_context_tokens is not None:
        return int(max_context_tokens)
    kv_cache = meta_info.get("kv_cache", {})
    shape = kv_cache.get("shape")
    if isinstance(shape, list) and len(shape) >= 3:
        return int(shape[2])
    return None


def _parse_cuda_graph_modules(modules_arg: str) -> tuple[str, ...] | None:
    if not modules_arg or not modules_arg.strip():
        return None
    modules = tuple(part.strip().lower() for part in modules_arg.split(",") if part.strip())
    return modules or None


def _load_merak_spec_runtime(
    *,
    meta_file: str | Path,
    device: str,
    exec_device: str,
    dtype: str,
    disable_auto_offload: bool,
    auto_offload_max_memory: str | None,
    prefill_auto_offload_max_memory: str | None,
    decode_auto_offload_max_memory: str | None,
    resource_tight_mode: bool,
    enable_cuda_graph: bool,
    cuda_graph_modules: str,
    cuda_graph_warmup_runs: int,
    cuda_graph_graph_warmup_runs: int,
):
    import torch
    from transformers import AutoTokenizer

    from xhmodel_merak.xh_llm.models.qwen3_5.qwen3_5_spec_decode_onnx_model import Qwen3_5SpecDecodeONNXModel

    resolved_meta = Path(meta_file).resolve()
    model_dir = resolved_meta.parent
    meta_info = _load_meta_info(resolved_meta)

    prefill_onnx = _resolve_path(
        model_dir,
        meta_info.get("prefill_onnx") or meta_info.get("prefill_hmonnx") or meta_info["prefill_onnx_file"],
    )
    decode_onnx = _resolve_path(
        model_dir,
        meta_info.get("decode_onnx") or meta_info.get("decode_hmonnx") or meta_info["decode_onnx_file"],
    )
    hf_model_config_dir = _resolve_path(model_dir, meta_info["hf_config"])
    token_embedding_file = _resolve_path(
        model_dir,
        meta_info.get("token_embedding_file") or meta_info.get("quant_embedding") or meta_info["token_embedding_file"],
    )
    draft, spec_mode, block_size, hidden_output_name = _spec_decode_from_meta(meta_info, model_dir)

    tokenizer = AutoTokenizer.from_pretrained(str(hf_model_config_dir))
    token_embedding = _load_token_embedding(token_embedding_file).to(dtype=_parse_dtype(dtype))
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
        max_context_tokens=_infer_max_context_tokens(meta_info),
        auto_offload=not disable_auto_offload,
        auto_offload_max_memory=_parse_auto_offload_max_memory(auto_offload_max_memory),
        prefill_auto_offload_max_memory=_parse_auto_offload_max_memory(prefill_auto_offload_max_memory),
        decode_auto_offload_max_memory=_parse_auto_offload_max_memory(decode_auto_offload_max_memory),
        resource_tight_mode=resource_tight_mode,
        pad_token_id=pad_token_id,
        enable_cuda_graph=enable_cuda_graph,
        cuda_graph_modules=_parse_cuda_graph_modules(cuda_graph_modules),
        cuda_graph_warmup_runs=cuda_graph_warmup_runs,
        cuda_graph_graph_warmup_runs=cuda_graph_graph_warmup_runs,
    )
    torch_dtype = _parse_dtype(dtype)
    runtime.set_input_embeddings(token_embedding)
    runtime.to(torch.device(device))
    runtime.set_exec_device(torch.device(exec_device))
    runtime.to(torch_dtype)
    return runtime, tokenizer, meta_info


def _run_spec_decode_once(
    runtime: Any,
    tokenizer: Any,
    prompt: str,
    *,
    max_new_tokens: int,
    enable_thinking: bool,
    do_sample: bool,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    presence_penalty: float,
    stream_output: bool,
) -> tuple[str, dict[str, Any], float, int]:
    input_ids = _build_chat_input_ids(tokenizer, prompt, enable_thinking)
    start = time.perf_counter()
    output, stats = runtime.generate(
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
        return_stats=True,
    )
    elapsed = time.perf_counter() - start
    cleaned = _postprocess_chat_output(output, enable_thinking=enable_thinking)
    output_tokens = len(tokenizer.encode(cleaned, add_special_tokens=False))
    return cleaned, dict(stats), elapsed, output_tokens


__all__ = [
    "HMONNXQuickTestResult",
    "build_spec_decode_result",
    "find_hmonnx_meta_file",
    "hmonnx_generate",
    "print_quick_test_result",
    "quick_test_hmonnx",
    "spec_decode_generate",
]
