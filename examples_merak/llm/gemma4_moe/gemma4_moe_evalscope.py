from __future__ import annotations

import argparse
import re
from pathlib import Path


CHOICE_PATTERNS = [
    r"答案\s*[:：]\s*([A-J])",
    r"(?i)answer\s*[:：]\s*([A-J])",
    r"(?i)the answer is\s*[:：]?\s*([A-J])",
    r"^[\s\*\-\(\[]*([A-J])[\)\]\s\.:：]*$",
]


def normalize_choice_answer(text: str) -> str:
    cleaned = text.replace("<|im_end|>", "").strip()
    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>", 1)[1].strip()
    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", cleaned, flags=re.IGNORECASE).strip()
    for pattern in CHOICE_PATTERNS:
        match = re.search(pattern, cleaned)
        if match:
            return f"Answer: {match.group(1).upper()}"
    return cleaned


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gemma4 MoE EvalScope benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--backend", choices=["float", "hmonnx"], default="float")
    parser.add_argument("--model-dir", type=str, default="/data01/datasets/gemma-4-26B-A4B-it")
    parser.add_argument("--model-config", type=str, default="work_dirs/gemma4_moe_with_mask_26b_a4b_it_llm_xh2a_2k_w8a8/export_meta_info.json")
    parser.add_argument(
        "--float-dtype",
        type=str,
        default="bfloat16",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Torch dtype used by the official float backend.",
    )
    parser.add_argument(
        "--float-experts-implementation",
        type=str,
        default="eager",
        choices=["eager", "batched_mm", "grouped_mm", "deepgemm"],
        help="Experts implementation for the official float Gemma4 MoE backend.",
    )
    parser.add_argument(
        "--device-map",
        type=str,
        default="auto",
        help="device_map for the official float backend, e.g. auto / balanced / cuda:0 / cpu",
    )
    parser.add_argument(
        "--hmonnx-device",
        type=str,
        default="cuda" if __import__("torch").cuda.is_available() else "cpu",
        help="Storage device for HMONNX runtime and KV cache.",
    )
    parser.add_argument(
        "--hmonnx-exec-device",
        type=str,
        default="cuda" if __import__("torch").cuda.is_available() else "cpu",
        help="Execution device for HMONNX sessions.",
    )
    parser.add_argument("--datasets", type=str, nargs="+", default=["mmlu_pro", "ceval"])
    parser.add_argument(
        "--mmlu-pro-subsets",
        type=str,
        nargs="+",
        default=None,
        help="Optional MMLU-Pro subset list.",
    )
    parser.add_argument(
        "--mmlu-pro-few-shot-num",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--ceval-subsets",
        type=str,
        nargs="+",
        default=None,
    )
    parser.add_argument(
        "--ceval-few-shot-num",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--limit-mmlu",
        type=int,
        default=0,
        help="Max samples for each MMLU-Pro subset. Use 0 for all samples.",
    )
    parser.add_argument(
        "--limit-ceval",
        type=int,
        default=0,
        help="Max samples for each CEval subset. Use 0 for all samples.",
    )
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--work-dir", type=str, default="./outputs-evalscope/gemma4_moe")
    parser.add_argument(
        "--resume-work-dir",
        type=str,
        default=None,
        help="Reuse an existing EvalScope timestamped work dir.",
    )
    parser.add_argument("--log-dir", type=str, default="./work_dirs/gemma4_moe_evalscope")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--smoke-prompt", type=str, default="请用一句话介绍一下你自己。")
    parser.add_argument("--smoke-max-new-tokens", type=int, default=32)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.smoke_test:
        assert normalize_choice_answer("答案：A") == "Answer: A"
        assert normalize_choice_answer("Answer: b") == "Answer: B"
        if args.backend == "hmonnx" and args.model_config:
            assert Path(args.model_config).exists(), args.model_config
        print("Task11 evalscope smoke OK", args.backend, ",".join(args.datasets))
        return

    try:
        from evalscope import TaskConfig, run_task  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("evalscope is required. Install project dependencies first.") from exc
    raise NotImplementedError("Full EvalScope run is wired by backend-specific ModelAPI registration in deployment.")


if __name__ == "__main__":
    main()