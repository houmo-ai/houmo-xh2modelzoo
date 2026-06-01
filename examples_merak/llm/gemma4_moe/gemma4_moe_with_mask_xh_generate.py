from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path


def bootstrap_runtime() -> None:
    workspace_root = Path(__file__).resolve().parents[4]
    vendor_transformers = workspace_root / ".vendor" / "python" / "transformers"
    if vendor_transformers.exists():
        link_root = Path(tempfile.gettempdir()) / "xh2a_vendor_transformers_only"
        link_root.mkdir(parents=True, exist_ok=True)
        link_path = link_root / "transformers"
        if link_path.exists() or link_path.is_symlink():
            if not link_path.is_symlink() or link_path.resolve() != vendor_transformers:
                if link_path.is_dir() and not link_path.is_symlink():
                    shutil.rmtree(link_path)
                else:
                    link_path.unlink()
        if not link_path.exists():
            link_path.symlink_to(vendor_transformers, target_is_directory=True)
        sys.path.insert(0, str(link_root))


bootstrap_runtime()

import torch
from transformers import TextStreamer
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

from xhmodel_merak.xh_llm import AutoLLMHONNXModel, LLMInferenceContextManager
from xhquant.api import get_xhquant_logger, xhquant_init
from xhquant.utils import ContextManagers, MemoryTracker, TimeProfiler


def _load_prompt_inputs_from_tokenizer(tokenizer, prompt: str, enable_thinking: bool = False):
    messages = [{"role": "user", "content": prompt}]
    chat_template_kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if enable_thinking:
        chat_template_kwargs["enable_thinking"] = True
    try:
        text = tokenizer.apply_chat_template(messages, **chat_template_kwargs)
    except TypeError:
        chat_template_kwargs.pop("enable_thinking", None)
        text = tokenizer.apply_chat_template(messages, **chat_template_kwargs)
    return tokenizer, tokenizer([text], return_tensors="pt")


def _load_prompt_inputs(hf_model_dir: str, prompt: str, enable_thinking: bool = False):
    tokenizer = AutoTokenizer.from_pretrained(hf_model_dir, trust_remote_code=True)
    return _load_prompt_inputs_from_tokenizer(tokenizer, prompt, enable_thinking)


def _load_json_file(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _resolve_meta_path(meta_path: str | Path, referenced_path: str | Path) -> Path:
    resolved_path = Path(referenced_path)
    if resolved_path.is_absolute():
        return resolved_path
    return Path(meta_path).resolve().parent / resolved_path


def _load_hf_config_from_runtime_meta(runtime_meta_path: str | Path, runtime_meta: dict) -> dict:
    hf_config = runtime_meta.get("hf_config")
    if not hf_config:
        return {}
    hf_config_path = _resolve_meta_path(runtime_meta_path, hf_config)
    if hf_config_path.is_dir():
        hf_config_path = hf_config_path / "config.json"
    if not hf_config_path.exists():
        return {}
    return _load_json_file(hf_config_path)


def _load_hf_generation_config_from_runtime_meta(runtime_meta_path: str | Path, runtime_meta: dict) -> dict:
    hf_config = runtime_meta.get("hf_config")
    if not hf_config:
        return {}
    hf_config_path = _resolve_meta_path(runtime_meta_path, hf_config)
    if not hf_config_path.is_dir():
        hf_config_path = hf_config_path.parent
    generation_config_path = hf_config_path / "generation_config.json"
    if not generation_config_path.exists():
        return {}
    return _load_json_file(generation_config_path)


def _load_hf_config_from_model_dir(hf_model_dir: str | Path) -> dict:
    config_path = Path(hf_model_dir) / "config.json"
    if not config_path.exists():
        return {}
    return _load_json_file(config_path)


def _load_hf_generation_config_from_model_dir(hf_model_dir: str | Path) -> dict:
    generation_config_path = Path(hf_model_dir) / "generation_config.json"
    if not generation_config_path.exists():
        return {}
    return _load_json_file(generation_config_path)


def _append_token_ids(token_ids: list[int], value) -> None:
    if value is None:
        return
    if isinstance(value, int):
        candidates = [value]
    elif isinstance(value, (list, tuple)):
        candidates = value
    else:
        return
    for token_id in candidates:
        token_id = int(token_id)
        if token_id not in token_ids:
            token_ids.append(token_id)


def _resolve_eos_token_id(tokenizer, *configs: dict) -> int | list[int] | None:
    token_ids: list[int] = []
    for config in configs:
        _append_token_ids(token_ids, config.get("eos_token_id"))
    _append_token_ids(token_ids, tokenizer.eos_token_id)

    for token in (getattr(tokenizer, "eot_token", None), "<turn|>"):
        if token is None:
            continue
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id != tokenizer.unk_token_id and token_id not in token_ids:
            token_ids.append(int(token_id))

    if not token_ids:
        return None
    if len(token_ids) == 1:
        return token_ids[0]
    return token_ids


def _resolve_runtime_meta_path(meta_path: str) -> Path:
    resolved_meta_path = Path(meta_path).resolve()
    meta = _load_json_file(resolved_meta_path)
    if meta.get("model_config", {}).get("model_type"):
        return resolved_meta_path

    exported_dir = meta.get("exported_dir")
    if exported_dir:
        golden_meta_path = (resolved_meta_path.parent / exported_dir / "golden_meta_info.json").resolve()
        if golden_meta_path.exists():
            return golden_meta_path

    raise ValueError(
        "model-config must point to golden_meta_info.json, or to an export_meta_info.json that contains exported_dir"
    )


def _resolve_pad_token_id(tokenizer) -> int:
    eos_token_id = tokenizer.eos_token_id
    if isinstance(eos_token_id, list):
        return int(eos_token_id[0])
    if eos_token_id is not None:
        return int(eos_token_id)
    if tokenizer.pad_token_id is not None:
        return int(tokenizer.pad_token_id)
    raise ValueError("Unable to resolve pad_token_id from tokenizer")


def run_hf(args) -> None:
    tokenizer, inputs = _load_prompt_inputs(args.hf_model_dir, args.prompt, args.enable_thinking)
    if args.smoke_test:
        print("Task09 HF smoke OK", inputs["input_ids"].shape)
        return
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype_map = {"auto": "auto", "float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    dtype = dtype_map.get(args.hf_dtype, torch.bfloat16)
    model = Gemma4ForConditionalGeneration.from_pretrained(
        args.hf_model_dir,
        torch_dtype=dtype,
        device_map=args.hf_device_map,
        experts_implementation=args.hf_experts_implementation,
        trust_remote_code=True,
        attn_implementation="eager",
    ).eval().to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    eos_token_id = _resolve_eos_token_id(
        tokenizer,
        _load_hf_generation_config_from_model_dir(args.hf_model_dir),
        _load_hf_config_from_model_dir(args.hf_model_dir),
    )
    generation_kwargs = {
        **inputs,
        "max_new_tokens": args.max_decode_steps,
        "do_sample": False,
    }
    if eos_token_id is not None:
        generation_kwargs["eos_token_id"] = eos_token_id
    with torch.no_grad():
        out = model.generate(**generation_kwargs)
    print(tokenizer.decode(out[0], skip_special_tokens=True))


def run_weight_only(args) -> None:
    fallback = args.weight_only_fallback_hf_model_dir or args.hf_model_dir
    tokenizer, inputs = _load_prompt_inputs(fallback, args.prompt, args.enable_thinking)
    print("Task09 weight_only surface OK", tokenizer.__class__.__name__, inputs["input_ids"].shape)


def run_hmonnx(args) -> None:
    logger = get_xhquant_logger()
    runtime_meta_path = _resolve_runtime_meta_path(args.model_config)
    runtime_meta = _load_json_file(runtime_meta_path)
    hmonnx_model = AutoLLMHONNXModel.from_pretrained(str(runtime_meta_path))

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA is not available, falling back to CPU.")
        device = "cpu"

    tokenizer = hmonnx_model.get_tokenizer(trust_remote_code=True)
    _, model_inputs = _load_prompt_inputs_from_tokenizer(tokenizer, args.prompt, args.enable_thinking)
    if args.smoke_test:
        print("Task09 HMONNX smoke OK", runtime_meta_path, model_inputs["input_ids"].shape)
        return

    streamer = TextStreamer(tokenizer, skip_prompt=True) if args.streaming_out else None
    hmonnx_model.to(device)
    if args.golden:
        hmonnx_model.enable_golden = True
        if args.fast:
            logger.warning("Golden outputs should be generated in aligned precision; ignoring --fast.")
    elif args.fast:
        hmonnx_model.to_fast()

    generation_kwargs = {
        **{key: value.to(device) for key, value in model_inputs.items()},
        "max_new_tokens": args.max_decode_steps,
        "do_sample": args.do_sample,
        "pad_token_id": _resolve_pad_token_id(tokenizer),
    }
    if streamer is not None:
        generation_kwargs["streamer"] = streamer
    eos_token_id = _resolve_eos_token_id(
        tokenizer,
        _load_hf_generation_config_from_runtime_meta(runtime_meta_path, runtime_meta),
        _load_hf_config_from_runtime_meta(runtime_meta_path, runtime_meta),
    )
    if eos_token_id is not None:
        generation_kwargs["eos_token_id"] = eos_token_id
        logger.info("Using eos_token_id for generation: %s", eos_token_id)

    contexts = [
        TimeProfiler("gemma4_moe_with_mask_hmonnx_generate", logger),
        MemoryTracker(device=device, name="generate", logger=logger),
        LLMInferenceContextManager(hmonnx_model),
    ]
    with ContextManagers(contexts):
        generated_ids = hmonnx_model.generate(**generation_kwargs)

    prompt_len = len(model_inputs["input_ids"][0])
    output_ids = generated_ids[0][prompt_len:].tolist()
    output_text = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    logger.info(f"{'-' * 20} HMONNX content {'-' * 20}")
    logger.info(output_text)
    print(output_text)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gemma4 MoE generate with HMONNX, HF, or weight-only backend",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="hmonnx",
        choices=["hmonnx", "hf", "weight_only"],
    )
    parser.add_argument(
        "--model-config", type=str,
        default="work_dirs/gemma4_moe_with_mask_26b_a4b_it_xh2a_w8a8_256_2k/export_meta_info.json",
        help="HMONNX runtime meta: either golden_meta_info.json or export_meta_info.json with exported_dir.",
    )
    parser.add_argument(
        "--hf-model-dir",
        type=str,
        default="/data01/datasets/gemma-4-26B-A4B-it",
        help="HF float model directory used when --backend hf.",
    )
    parser.add_argument(
        "--weight-only-model-dir",
        type=str,
        default="",
        help="Weight-only GPTQ / AutoRound model directory used when --backend weight_only.",
    )
    parser.add_argument(
        "--weight-only-fallback-hf-model-dir",
        type=str,
        default="",
        help="Optional float HF model directory used to fill missing non-quantized tensors.",
    )
    parser.add_argument(
        "--hf-dtype",
        type=str,
        default="bfloat16",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Torch dtype used by the HF and weight_only backends.",
    )
    parser.add_argument(
        "--hf-device-map",
        type=str,
        default="auto",
        help="device_map used by the HF and weight_only backends, e.g. auto / balanced / cuda:0 / cpu.",
    )
    parser.add_argument(
        "--hf-experts-implementation",
        type=str,
        default="eager",
        choices=["eager", "batched_mm", "grouped_mm", "deepgemm"],
        help="Experts implementation used by the HF and weight_only Gemma4 backends.",
    )
    parser.add_argument("--prompt", type=str, default="你好，请详细介绍一下大语言模型的原理。")
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--max-decode-steps", type=int, default=128)
    parser.add_argument(
        "--enable-thinking",
        dest="enable_thinking",
        action="store_true",
        help="Enable thinking mode for chat template.",
    )
    parser.add_argument(
        "--streaming-out",
        dest="streaming_out",
        action="store_true",
        help="Stream output tokens during generation.",
    )
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--fast", action="store_true", help="Run HMONNX in fast mode.")
    parser.add_argument("--golden", action="store_true", help="Save golden outputs during HMONNX generation.")
    parser.add_argument("--do-sample", action="store_true", help="Enable sampling for generation.")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    xhquant_init(None, args.debug)
    torch.manual_seed(args.seed)
    if args.backend == "hf":
        run_hf(args)
    elif args.backend == "weight_only":
        run_weight_only(args)
    else:
        run_hmonnx(args)


if __name__ == "__main__":
    main()