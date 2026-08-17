#!/usr/bin/env python3
"""Create DeepSeek-V4 Flash W8/W4-G64 weights through GPTQModel."""

from __future__ import annotations

import argparse
import json

from xhmodel_merak.xh_llm.models.deepseek_v4.quant_adapter import (
    DeepSeekV4FlashQuantSpec,
    quantization_plan,
    quantize_deepseek_v4_flash,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("gptq", "autoround"), required=True)
    parser.add_argument(
        "--model",
        default="/data01/datasets/DeepSeek-V4-Flash-0731",
    )
    parser.add_argument("--prepared-model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-layers", type=int)
    parser.add_argument("--dataset", default="NeelNanda/pile-10k")
    parser.add_argument("--calibration-jsonl")
    parser.add_argument("--text-key", default="text")
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--base-bits", type=int, default=8, choices=(8,))
    parser.add_argument("--expert-bits", type=int, default=4, choices=(4,))
    parser.add_argument("--group-size", type=int, default=64, choices=(64,))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--preparation-device", default="cuda:0")
    parser.add_argument("--moe-batch-size", type=int, default=128)
    parser.add_argument("--max-quant-layers", type=int)
    parser.add_argument("--auto-round-version", choices=("v1", "v2"), default="v2")
    parser.add_argument("--auto-round-iters", type=int, default=200)
    parser.add_argument("--auto-round-lr", type=float)
    parser.add_argument("--auto-round-minmax-lr", type=float)
    parser.add_argument(
        "--offload-to-disk",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Opt-in only; disabled by default",
    )
    parser.add_argument("--offload-dir")
    parser.add_argument(
        "--auto-forward-data-parallel",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Opt in only; the default layerwise GPTQ/AutoRound path fits one "
            "80-GiB GPU and does not clone forward replay across devices"
        ),
    )
    parser.add_argument(
        "--wait-for-submodule-finalizers",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--hessian-mse",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _build_spec(args: argparse.Namespace) -> DeepSeekV4FlashQuantSpec:
    return DeepSeekV4FlashQuantSpec(
        model_dir=args.model,
        prepared_model_dir=args.prepared_model,
        output_dir=args.output,
        method=args.method,
        num_layers=args.num_layers,
        dataset=args.dataset,
        calibration_jsonl=args.calibration_jsonl,
        text_key=args.text_key,
        nsamples=args.nsamples,
        seqlen=args.seqlen,
        batch_size=args.batch_size,
        base_bits=args.base_bits,
        expert_bits=args.expert_bits,
        group_size=args.group_size,
        device=args.device,
        preparation_device=args.preparation_device,
        offload_to_disk=args.offload_to_disk,
        offload_dir=args.offload_dir,
        wait_for_submodule_finalizers=args.wait_for_submodule_finalizers,
        hessian_mse=args.hessian_mse,
        moe_batch_size=args.moe_batch_size,
        auto_forward_data_parallel=args.auto_forward_data_parallel,
        max_quant_layers=args.max_quant_layers,
        auto_round_version=args.auto_round_version,
        auto_round_iters=args.auto_round_iters,
        auto_round_lr=args.auto_round_lr,
        auto_round_minmax_lr=args.auto_round_minmax_lr,
        seed=args.seed,
        dry_run=args.dry_run,
    )


def main() -> None:
    spec = _build_spec(parse_args())
    print(json.dumps(quantization_plan(spec), ensure_ascii=False, indent=2))
    result = quantize_deepseek_v4_flash(spec)
    print(json.dumps(result.__dict__, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
