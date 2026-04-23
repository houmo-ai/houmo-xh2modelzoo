import argparse

from _runtime import (
    benchmark_chat,
    load_runtime_from_meta,
    parse_auto_offload_max_memory,
    parse_dtype,
)


def main(args):
    runtime, tokenizer, meta_info = load_runtime_from_meta(
        meta_path=args.config,
        dtype=parse_dtype(args.dtype),
        device=args.device,
        exec_device=args.exec_device,
        auto_offload=not args.disable_auto_offload,
        auto_offload_max_memory=parse_auto_offload_max_memory(args.auto_offload_max_memory),
        prefill_auto_offload_max_memory=parse_auto_offload_max_memory(args.prefill_auto_offload_max_memory),
        decode_auto_offload_max_memory=parse_auto_offload_max_memory(args.decode_auto_offload_max_memory),
        resource_tight_mode=args.resource_tight_mode,
    )

    for _ in range(args.warmup_runs):
        benchmark_chat(
            runtime,
            tokenizer,
            prompt=args.prompt,
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
        output_text, elapsed, output_tokens = benchmark_chat(
            runtime,
            tokenizer,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            enable_thinking=args.enable_thinking,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            presence_penalty=args.presence_penalty,
            stream_output=True,
        )
        timings.append(elapsed)

    avg_latency = sum(timings) / max(len(timings), 1)
    toks_per_sec = output_tokens / avg_latency if avg_latency > 0 else 0.0

    print(f"model_name: {meta_info.get('model_name')}")
    print(f"quant_scheme: {meta_info.get('quant_scheme')}")
    print(f"prompt: {args.prompt}")
    print(f"output: {output_text}")
    print(f"benchmark_runs: {args.benchmark_runs}")
    print(f"avg_latency_s: {avg_latency:.4f}")
    print(f"output_tokens: {output_tokens}")
    print(f"tokens_per_second: {toks_per_sec:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="work_dirs/qwen3_5_27b_bf16_export/meta.json",
        help="Path to meta.json from export",
    )
    parser.add_argument("--prompt", type=str, default="请用中文简要介绍一下混合线性注意力模型。")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--benchmark-runs", type=int, default=1)
    parser.add_argument("--do-sample", dest="do_sample", action="store_true", default=False)
    parser.add_argument("--no-sample", dest="do_sample", action="store_false")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--presence-penalty", type=float, default=0.0)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--dtype", type=str, default="fp16")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--exec-device", type=str, default="cuda")
    parser.add_argument("--disable-auto-offload", action="store_true")
    parser.add_argument("--auto-offload-max-memory", type=str, default=None)
    parser.add_argument("--prefill-auto-offload-max-memory", type=str, default=None)
    parser.add_argument("--decode-auto-offload-max-memory", type=str, default=None)
    parser.add_argument("--resource-tight-mode", action="store_true")
    args = parser.parse_args()
    main(args)
