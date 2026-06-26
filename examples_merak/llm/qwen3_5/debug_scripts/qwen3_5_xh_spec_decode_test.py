"""Canonical Merak Qwen3.5/Qwen3.5-MoE MTP/DFlash HMONNX demo.

The implementation lives in ``hmonnx_validation.spec_decode_generate`` so the
same API is used by demos, validation matrix, and this CLI.

Example:
    python examples_merak/llm/qwen3_5/debug_scripts/qwen3_5_xh_spec_decode_test.py \
        --config work_dirs/qwen3_5_9b_spec_mtp_xh2a_2k/.../golden_meta_info.json \
        --max-new-tokens 40 \
        --min-output-tokens 32
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from xhmodel_merak.xh_llm.models.qwen3_5.hmonnx_validation import (  # noqa: E402
    print_quick_test_result,
    spec_decode_generate,
)


def _print_cuda_graph_status(result) -> None:
    status = result.stats.get("cuda_graph_status")
    if not status:
        return
    print("cuda_graph_status:")
    for name, item in status.items():
        reason = item.get("reason") or "-"
        backend = item.get("backend") or "-"
        print(
            f"  {name}: backend={backend} enabled={item.get('enabled')} "
            f"captured={item.get('captured')} reason={reason}"
        )


def main(args: argparse.Namespace) -> None:
    result = spec_decode_generate(
        meta_file=args.config,
        device=args.device,
        exec_device=args.exec_device,
        dtype=args.dtype,
        prompt=args.prompt,
        prompt_file=args.prompt_file,
        max_new_tokens=args.max_new_tokens,
        min_output_tokens=args.min_output_tokens,
        enable_thinking=args.enable_thinking,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        presence_penalty=args.presence_penalty,
        warmup_runs=args.warmup_runs,
        benchmark_runs=args.benchmark_runs,
        stream_output=args.stream_output,
        disable_auto_offload=args.disable_auto_offload,
        auto_offload_max_memory=args.auto_offload_max_memory,
        prefill_auto_offload_max_memory=args.prefill_auto_offload_max_memory,
        decode_auto_offload_max_memory=args.decode_auto_offload_max_memory,
        resource_tight_mode=args.resource_tight_mode,
        enable_cuda_graph=args.enable_cuda_graph,
        cuda_graph_modules=args.cuda_graph_modules,
        cuda_graph_warmup_runs=args.cuda_graph_warmup_runs,
        cuda_graph_graph_warmup_runs=args.cuda_graph_graph_warmup_runs,
    )
    print_quick_test_result(result)
    _print_cuda_graph_status(result)


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
