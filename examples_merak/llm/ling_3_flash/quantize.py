#!/usr/bin/env python3
"""Quantize Ling-3-Flash to a GPTQModel-compatible HF checkpoint."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from xhmodel_merak.xh_llm.models.ling_3_flash.quant_adapter import (
    Ling3FlashQuantSpec,
    quantization_plan,
    quantize_ling3_flash,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("gptq", "autoround"), required=True)
    parser.add_argument("--model", type=Path, default=Path("weights/Ling-3.0-flash"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", default="NeelNanda/pile-10k")
    parser.add_argument("--calibration-jsonl", type=Path)
    parser.add_argument("--text-key", default="text")
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--moe-batch-size",
        type=int,
        default=None,
        help=(
            "Optional GPTQ expert Linear modules per calibration pass; "
            "the default processes the whole expert subset once"
        ),
    )
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--base-bits", type=int, default=8)
    parser.add_argument("--expert-bits", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--device-map", default="0")
    parser.add_argument(
        "--offload-dir",
        type=Path,
        default=Path("work_dirs/ling_3_flash_offload"),
    )
    parser.add_argument(
        "--offload-to-disk",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Opt-in only for GPTQ; disabled by default because the related "
            "Qwen3.5 hybrid/MoE path has produced invalid checkpoints"
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
    parser.add_argument("--max-quant-layers", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _build_spec(args: argparse.Namespace) -> Ling3FlashQuantSpec:
    return Ling3FlashQuantSpec(
        model_dir=str(args.model),
        output_dir=str(args.output),
        method=args.method,
        dataset=args.dataset,
        calibration_jsonl=(
            None if args.calibration_jsonl is None else str(args.calibration_jsonl)
        ),
        text_key=args.text_key,
        nsamples=args.nsamples,
        seqlen=args.seqlen,
        batch_size=args.batch_size,
        moe_batch_size=args.moe_batch_size,
        iters=args.iters,
        group_size=args.group_size,
        base_bits=args.base_bits,
        expert_bits=args.expert_bits,
        device=args.device,
        device_map=args.device_map,
        offload_to_disk=args.offload_to_disk,
        offload_dir=str(args.offload_dir),
        wait_for_submodule_finalizers=args.wait_for_submodule_finalizers,
        hessian_mse=args.hessian_mse,
        max_quant_layers=args.max_quant_layers,
        seed=args.seed,
        dry_run=args.dry_run,
    )


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    spec = _build_spec(parse_args())
    spec.validate()
    print(json.dumps(quantization_plan(spec), indent=2), flush=True)
    quantize_ling3_flash(spec)


if __name__ == "__main__":
    main()
