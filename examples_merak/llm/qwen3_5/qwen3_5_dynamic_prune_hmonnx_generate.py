"""Run a text-generation smoke test for dynamic-prune Qwen3.5 HMONNX."""

from __future__ import annotations

import argparse
import os
from contextlib import nullcontext


try:
    from .qwen3_5_dynamic_prune_workflow import (
        _is_llm_only_meta,
        use_dynamic_prune_text_only_hmonnx_runtime,
    )
    from .qwen3_5_xh_hmonnx_generate import _parse_device_arg
except ImportError:
    from qwen3_5_dynamic_prune_workflow import (
        _is_llm_only_meta,
        use_dynamic_prune_text_only_hmonnx_runtime,
    )
    from qwen3_5_xh_hmonnx_generate import _parse_device_arg

from xhmodel_merak.xh_llm.models.qwen3_5.hmonnx_validation import (
    find_hmonnx_meta_file,
    hmonnx_generate,
    print_quick_test_result,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate text generation with a dynamic-prune Qwen3.5 HMONNX export."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Export directory or golden_meta_info.json path.",
    )
    parser.add_argument(
        "--prompt",
        default="用中文简单介绍 Qwen3.5。",
        help="Text prompt or path to a UTF-8 prompt file.",
    )
    parser.add_argument("--device", default="0", help="GPU id list such as '0' or '0,1', or 'cpu'.")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--min-output-tokens",
        type=int,
        default=8,
        help="Fail unless at least this many continuation tokens are generated.",
    )
    parser.add_argument("--think", action="store_true", help="Enable the thinking chat template.")
    parser.add_argument("--fast", action="store_true", help="Enable fast HMONNX runtime mode.")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--auto-offload", action="store_true")
    parser.add_argument("--cuda-graph", action="store_true")
    parser.add_argument("--use-v2", action="store_true", help="Enable HMONNXInferenceV2.")
    parser.add_argument("--do-sample", action="store_true", help="Enable sampling; greedy decode is the default.")
    return parser


def _runtime_device(devices: list[int | str]) -> str:
    if devices == ["cpu"]:
        return "cpu"
    return f"cuda:{devices[0]}"


def main(args: argparse.Namespace) -> None:
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    if args.min_output_tokens < 0:
        raise ValueError("--min-output-tokens must be non-negative")
    if args.min_output_tokens > args.max_new_tokens:
        raise ValueError("--min-output-tokens cannot exceed --max-new-tokens")

    if args.use_v2:
        os.environ["ENABLE_HMINFERENCE_V2"] = "1"

    meta_file = find_hmonnx_meta_file(args.config)
    devices = _parse_device_arg(args.device)
    runtime = (
        use_dynamic_prune_text_only_hmonnx_runtime()
        if _is_llm_only_meta(meta_file)
        else nullcontext()
    )
    with runtime:
        result = hmonnx_generate(
            meta_file=meta_file,
            prompt=args.prompt,
            device=_runtime_device(devices),
            device_map=devices,
            max_new_tokens=args.max_new_tokens,
            min_output_tokens=args.min_output_tokens,
            do_sample=args.do_sample,
            think=args.think,
            fast=args.fast,
            debug=args.debug,
            auto_offload=args.auto_offload,
            cuda_graph=args.cuda_graph,
        )
    print_quick_test_result(result)


if __name__ == "__main__":
    main(build_parser().parse_args())
