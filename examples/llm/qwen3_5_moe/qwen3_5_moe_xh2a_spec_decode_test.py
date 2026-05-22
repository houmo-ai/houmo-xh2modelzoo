from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_DENSE_DIR = _HERE.parent / "qwen3_5"
for path in (str(_HERE), str(_DENSE_DIR)):
    while path in sys.path:
        sys.path.remove(path)
sys.path.insert(0, str(_HERE))
sys.path.insert(1, str(_DENSE_DIR))

import qwen3_5_xh2a_spec_decode_test as dense_spec_test

from _runtime import load_runtime_from_meta


def load_spec_decode_runtime(
    meta_path: str,
    dtype,
    device: str,
    exec_device: str,
    auto_offload: bool,
    auto_offload_max_memory=None,
    prefill_auto_offload_max_memory=None,
    decode_auto_offload_max_memory=None,
    resource_tight_mode: bool = False,
    num_draft_tokens_override=None,
    enable_cuda_graph: bool = False,
    cuda_graph_modules=None,
    cuda_graph_warmup_runs: int = 3,
    cuda_graph_graph_warmup_runs: int = 6,
):
    runtime, tokenizer, meta_info = load_runtime_from_meta(
        meta_path=meta_path,
        dtype=dtype,
        device=device,
        exec_device=exec_device,
        auto_offload=auto_offload,
        auto_offload_max_memory=auto_offload_max_memory,
        prefill_auto_offload_max_memory=prefill_auto_offload_max_memory,
        decode_auto_offload_max_memory=decode_auto_offload_max_memory,
        resource_tight_mode=resource_tight_mode,
        enable_cuda_graph=enable_cuda_graph,
        cuda_graph_modules=cuda_graph_modules,
        cuda_graph_warmup_runs=cuda_graph_warmup_runs,
        cuda_graph_graph_warmup_runs=cuda_graph_graph_warmup_runs,
    )
    if not hasattr(runtime, "spec_decode_mode"):
        raise ValueError(f"{meta_path} is not a spec-decode export.")
    if (
        num_draft_tokens_override is not None
        and num_draft_tokens_override > 0
        and num_draft_tokens_override != runtime.block_size
    ):
        raise ValueError(
            f"MoE spec runtime block_size is fixed by export: meta={runtime.block_size}, "
            f"override={num_draft_tokens_override}"
        )
    return runtime, tokenizer, meta_info


dense_spec_test.load_spec_decode_runtime = load_spec_decode_runtime


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Speculative decoding inference benchmark for Qwen3.5-MoE on xh2a"
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
    parser.set_defaults(disable_auto_offload=True)
    parser.add_argument(
        "--enable_auto_offload",
        dest="disable_auto_offload",
        action="store_false",
    )
    parser.add_argument(
        "--disable_auto_offload",
        dest="disable_auto_offload",
        action="store_true",
    )
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
        help="If set, export draft session golden data to this directory",
    )
    return parser


def main() -> None:
    dense_spec_test.main(build_parser().parse_args())


if __name__ == "__main__":
    main()
