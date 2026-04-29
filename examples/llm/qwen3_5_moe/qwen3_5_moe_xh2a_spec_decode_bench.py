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

import qwen3_5_xh2a_spec_decode_bench as dense_bench

from _runtime import load_runtime_from_meta, parse_cuda_graph_modules, parse_dtype
from qwen3_5_moe_spec_decode_metrics import _release_runtime, run_moe_spec_metrics_with_runtime
from xh_model_zoo.xh_llm.models.qwen3_5_moe import Qwen3_5MoeSpecDecodeInference


_CUDA_GRAPH_MODULES = None
_AUTO_OFFLOAD = False


def load_dense_runtime_from_meta(
    *,
    meta_path: str,
    dtype: str,
    device: str,
    exec_device: str,
    auto_offload_max_memory=None,
    prefill_auto_offload_max_memory=None,
    decode_auto_offload_max_memory=None,
    enable_cuda_graph: bool = False,
    cuda_graph_warmup_runs: int = 3,
    cuda_graph_graph_warmup_runs: int = 6,
):
    return load_runtime_from_meta(
        meta_path=meta_path,
        dtype=parse_dtype(dtype),
        device=device,
        exec_device=exec_device,
        auto_offload=_AUTO_OFFLOAD,
        auto_offload_max_memory=auto_offload_max_memory,
        prefill_auto_offload_max_memory=prefill_auto_offload_max_memory,
        decode_auto_offload_max_memory=decode_auto_offload_max_memory,
        enable_cuda_graph=enable_cuda_graph,
        cuda_graph_modules=_CUDA_GRAPH_MODULES,
        cuda_graph_warmup_runs=cuda_graph_warmup_runs,
        cuda_graph_graph_warmup_runs=cuda_graph_graph_warmup_runs,
    )


def run_dense_spec_metrics_with_runtime(
    *,
    runtime,
    tokenizer,
    model_name,
    prompt: str,
    max_new_tokens: int,
    enable_thinking: bool = False,
    repetition_penalty: float = 1.0,
    presence_penalty: float = 0.0,
    include_baseline: bool = False,
):
    del tokenizer, model_name, include_baseline
    return run_moe_spec_metrics_with_runtime(
        meta_path=getattr(runtime, "meta_path", ""),
        runtime=runtime,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        enable_thinking=enable_thinking,
        repetition_penalty=repetition_penalty,
        presence_penalty=presence_penalty,
    )


dense_bench.load_dense_runtime_from_meta = load_dense_runtime_from_meta
dense_bench.release_dense_runtime = _release_runtime
dense_bench.run_dense_spec_metrics_with_runtime = run_dense_spec_metrics_with_runtime
dense_bench.Qwen3_5SpecDecodeONNXModel = Qwen3_5MoeSpecDecodeInference


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta", required=True, help="path to spec-decode meta.json")
    parser.add_argument(
        "--baseline-meta",
        default=None,
        help="deprecated and ignored; bench now reports spec-only generate metrics",
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--think-mode", choices=["on", "off", "both"], default="both")
    parser.add_argument("--dtype", default="fp16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--exec-device", default="cuda:0")
    parser.set_defaults(enable_auto_offload=False)
    parser.add_argument("--enable-auto-offload", dest="enable_auto_offload", action="store_true")
    parser.add_argument("--disable-auto-offload", dest="enable_auto_offload", action="store_false")
    parser.add_argument("--auto-offload-max-memory", dest="auto_offload_max_memory", default=None)
    parser.add_argument(
        "--prefill-auto-offload-max-memory",
        dest="prefill_auto_offload_max_memory",
        default=None,
    )
    parser.add_argument(
        "--decode-auto-offload-max-memory",
        dest="decode_auto_offload_max_memory",
        default=None,
    )
    parser.add_argument("--enable-cuda-graph", action="store_true")
    parser.add_argument(
        "--cuda-graph-modules",
        type=str,
        default="",
        help="Comma-separated session names: prefill,decode,draft_prefill,draft_context,draft_decode",
    )
    parser.add_argument("--cuda-graph-warmup-runs", type=int, default=3)
    parser.add_argument("--cuda-graph-graph-warmup-runs", type=int, default=6)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--presence-penalty", type=float, default=0.0)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="if >0, keep only this many balanced cases",
    )
    parser.add_argument("--shard-index", type=int, default=0, help="0-based shard index")
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="split dataset across this many shards",
    )
    return parser


def main() -> None:
    global _AUTO_OFFLOAD, _CUDA_GRAPH_MODULES

    args = build_parser().parse_args()
    _AUTO_OFFLOAD = bool(args.enable_auto_offload)
    _CUDA_GRAPH_MODULES = parse_cuda_graph_modules(args.cuda_graph_modules)
    dense_bench.run_bench(args)


if __name__ == "__main__":
    main()
