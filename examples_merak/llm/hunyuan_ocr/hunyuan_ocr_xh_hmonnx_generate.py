#!/usr/bin/env python3
# Copyright 2025 HOUMO AI
# SPDX-License-Identifier: Apache-2.0

"""Generate OCR text from one image with an exported HunyuanOCR HMONNX bundle."""

from __future__ import annotations

import argparse
import copy
import json
import os
import time
from pathlib import Path
from typing import Any


DEFAULT_PROMPT = "Extract the document as markdown."


class _SingleImageAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None) -> None:
        if getattr(namespace, self.dest, None) is not None:
            parser.error(f"{option_string} may only be specified once")
        setattr(namespace, self.dest, values)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hmonnx-config",
        type=Path,
        required=True,
        help="Path to golden_meta_info.json for the exported HunyuanOCR bundle.",
    )
    parser.add_argument(
        "--image",
        type=Path,
        action=_SingleImageAction,
        required=True,
        help="Document image to process. This OCR entrypoint accepts exactly one image.",
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="OCR instruction.")
    parser.add_argument("--device", default="cuda:0", help="Runtime device, for example cuda:0 or cpu.")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--dflash",
        action="store_true",
        help="Enable DFlash speculative decoding. The bundle must be runtime-ready.",
    )
    parser.add_argument(
        "--num-draft-tokens",
        type=int,
        help="Override the DFlash draft token count declared by bundle metadata.",
    )
    parser.add_argument(
        "--no-stream",
        action="store_false",
        dest="stream",
        help="Disable token-by-token terminal output for autoregressive generation.",
    )
    parser.set_defaults(stream=True)
    parser.add_argument(
        "--legacy-runtime",
        action="store_true",
        help="Use the deprecated legacy HMONNX runtime instead of HMONNXInferenceV2.",
    )
    parser.add_argument(
        "--cuda-graph",
        action="store_true",
        help="Enable CUDA Graph capture for compatible HMONNX sessions.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON report path. The decoded OCR text is always printed.",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if not args.hmonnx_config.is_file():
        raise FileNotFoundError(f"HunyuanOCR HMONNX metadata does not exist: {args.hmonnx_config}")
    if not args.image.is_file():
        raise FileNotFoundError(f"HunyuanOCR input image does not exist: {args.image}")
    if not isinstance(args.prompt, str) or not args.prompt.strip():
        raise ValueError("--prompt must be a non-empty string")
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    if args.num_draft_tokens is not None:
        if not args.dflash:
            raise ValueError("--num-draft-tokens requires --dflash")
        if args.num_draft_tokens <= 0:
            raise ValueError("--num-draft-tokens must be positive")
    if args.dflash and args.stream:
        raise ValueError("DFlash generation requires --no-stream")


def _messages(image: Path, prompt: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image)},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def _runtime_device_map(device: str) -> list[str]:
    normalized = device.strip().lower()
    if normalized == "cuda":
        return ["cuda:0"]
    if normalized == "cpu" or (normalized.startswith("cuda:") and normalized.removeprefix("cuda:").isdigit()):
        return [normalized]
    raise ValueError("--device must be cpu, cuda, or cuda:N")


def _configure_runtime_environment(*, legacy_runtime: bool) -> str:
    use_v2 = not legacy_runtime
    os.environ["ENABLE_HMINFERENCE_V2"] = "1" if use_v2 else "0"
    return "HMONNXInferenceV2" if use_v2 else "legacy"


def _generated_sequences(output: Any) -> Any:
    return output.sequences if hasattr(output, "sequences") else output


def main(args: argparse.Namespace) -> dict[str, Any]:
    runtime_name = _configure_runtime_environment(legacy_runtime=args.legacy_runtime)

    import torch

    from xhmodel_merak.xh_llm import AutoLLMHONNXModel, LLMInferenceContextManager

    _validate_args(args)
    if AutoLLMHONNXModel is None:
        raise RuntimeError(
            "HMONNX runtime import failed. Install the xhquanttool build paired with this "
            "xh2modelzoo revision and verify that xhmodel_merak.xh_llm.hmonnx imports successfully."
        )
    device_map = _runtime_device_map(args.device)
    print(
        f"[hunyuan-ocr] loading HMONNX bundle with runtime={runtime_name} "
        f"on {device_map[0]}: {args.hmonnx_config}",
        flush=True,
    )
    runtime = AutoLLMHONNXModel.from_pretrained(
        str(args.hmonnx_config),
        device_map=device_map,
        enable_cuda_graph=args.cuda_graph,
    ).to(args.device)
    processor = runtime.get_tf_processor()
    print(f"[hunyuan-ocr] preprocessing image: {args.image}", flush=True)
    inputs = processor.apply_chat_template(
        _messages(args.image, args.prompt),
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    generation_inputs = inputs.to(args.device)
    prompt_length = int(generation_inputs["input_ids"].shape[1])
    generation_kwargs: dict[str, Any] = {
        "do_sample": False,
        "repetition_penalty": 1.0,
        "max_new_tokens": args.max_new_tokens,
        "pad_token_id": runtime.meta_info.pad_token_id,
        "use_cache": True,
        "dflash_enabled": args.dflash,
    }
    if args.dflash:
        if args.num_draft_tokens is not None:
            generation_kwargs["dflash_num_draft_tokens"] = args.num_draft_tokens
    else:
        generation_kwargs["eos_token_id"] = runtime.meta_info.generation_eos_token_id
    streaming = bool(args.stream)
    if streaming:
        from transformers import TextStreamer

        generation_kwargs["streamer"] = TextStreamer(
            processor.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

    mode = "DFlash" if args.dflash else "autoregressive"
    print(f"[hunyuan-ocr] generating up to {args.max_new_tokens} tokens in {mode} mode", flush=True)
    started = time.perf_counter()
    with torch.no_grad(), LLMInferenceContextManager(runtime):
        output = runtime.generate(**generation_inputs, **generation_kwargs)
    elapsed_seconds = time.perf_counter() - started
    print(f"[hunyuan-ocr] generation finished in {elapsed_seconds:.2f}s", flush=True)

    sequences = _generated_sequences(output)
    generated_tokens = [int(token) for token in sequences[0, prompt_length:].detach().cpu().tolist()]
    text = processor.decode(generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    runtime_summary = copy.deepcopy(runtime.last_request_summary or {})
    report = {
        "hmonnx_config": str(args.hmonnx_config),
        "image": str(args.image),
        "prompt": args.prompt,
        "device": args.device,
        "runtime": runtime_name,
        "cuda_graph": args.cuda_graph,
        "dflash_enabled": args.dflash,
        "num_draft_tokens": args.num_draft_tokens,
        "max_new_tokens": args.max_new_tokens,
        "prompt_length": prompt_length,
        "generated_token_count": len(generated_tokens),
        "generated_tokens": generated_tokens,
        "text": text,
        "elapsed_seconds": elapsed_seconds,
        "runtime_summary": runtime_summary,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            f"{json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)}\n",
            encoding="utf-8",
        )
    if not streaming:
        print(text)
    return report


if __name__ == "__main__":
    main(build_parser().parse_args())
