#!/usr/bin/env python3
"""Run text generation from an exported Ling-3-Flash HMONNX model."""

from __future__ import annotations

import argparse

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meta", required=True, help="golden_meta_info.json")
    parser.add_argument("--prompt", default="请简要介绍一下你自己。")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help=(
            "Enable Ling's reasoning chat template. The default is direct "
            "answer mode so ordinary conversation tests are deterministic."
        ),
    )
    parser.add_argument(
        "--raw-prompt",
        action="store_true",
        help="Tokenize --prompt directly instead of applying the chat template",
    )
    parser.add_argument(
        "--print-token-ids",
        action="store_true",
        help="Print generated token ids before the decoded continuation",
    )
    v2_group = parser.add_mutually_exclusive_group()
    v2_group.add_argument(
        "--use-v2",
        dest="use_v2",
        action="store_true",
        help=(
            "Use HMONNXInferenceV2 (the default), including runtime W4 "
            "packing and prefill/decode initializer sharing"
        ),
    )
    v2_group.add_argument(
        "--no-use-v2",
        dest="use_v2",
        action="store_false",
        help="Use the legacy runtime for diagnostics; large W4 models may not fit one GPU",
    )
    parser.set_defaults(use_v2=True)
    parser.add_argument(
        "--pack-w4",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pack W4 initializers at runtime (default: enabled)",
    )
    parser.add_argument(
        "--cuda-graph",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable CUDA Graph execution (default: enabled)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")

    from xhmodel_merak.xh_llm.utils import configure_hmonnx_validation_runtime

    configure_hmonnx_validation_runtime(
        use_v2=args.use_v2,
        pack_w4=args.pack_w4,
    )

    from xhmodel_merak.xh_llm import AutoLLMHONNXModel, LLMInferenceContextManager

    requested_device = torch.device(args.device)
    model = AutoLLMHONNXModel.from_pretrained(
        args.meta,
        device_map=[requested_device],
        enable_cuda_graph=args.cuda_graph,
    )
    tokenizer = model.get_tokenizer()
    if args.raw_prompt:
        text = args.prompt
    else:
        messages = [{"role": "user", "content": args.prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=args.enable_thinking,
        )
    inputs = tokenizer([text], return_tensors="pt").to(requested_device)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    model.to(requested_device)
    with LLMInferenceContextManager(model):
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            pad_token_id=pad_token_id,
        )
    continuation = output_ids[:, inputs.input_ids.shape[1] :]
    if args.print_token_ids:
        print(f"token_ids={continuation[0].tolist()}")
    print(tokenizer.batch_decode(continuation, skip_special_tokens=True)[0])


if __name__ == "__main__":
    main()
