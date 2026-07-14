"""Run Qwen3-Next HMONNX in ordinary/page attention and eager/CUDA Graph modes."""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="golden_meta_info.json")
    parser.add_argument("--prompt", default="用中文一句话说明模型是否可用。")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--device", default="0", help="GPU ids such as 0 or 0,1; use cpu for CPU")
    parser.add_argument(
        "--page-attention",
        action="store_true",
        help="Convert ordinary-attention HMONNX to page attention at load time.",
    )
    parser.add_argument("--cuda-graph", action="store_true")
    parser.add_argument("--golden", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--fast", action="store_true")
    return parser


def _devices(value: str):
    if value.strip().lower() == "cpu":
        return ["cpu"]
    return [int(token) for token in value.split(",") if token.strip()]


def main(args: argparse.Namespace) -> None:
    from xhmodel_merak.xh_llm.models.qwen3_5.hmonnx_validation import (
        hmonnx_generate,
        print_quick_test_result,
    )

    devices = _devices(args.device)
    result = hmonnx_generate(
        meta_file=args.config,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        fast=args.fast,
        debug=args.debug,
        golden=args.golden,
        cuda_graph=args.cuda_graph,
        page_attention=args.page_attention,
        device_map=devices,
    )
    print_quick_test_result(result)


if __name__ == "__main__":
    main(build_parser().parse_args())
