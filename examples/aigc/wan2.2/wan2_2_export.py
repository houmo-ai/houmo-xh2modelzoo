# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: E402, I001

# pyright: reportMissingImports=false

import argparse
import sys
from pathlib import Path

from xhquant.api import DeviceType, QuantScheme, xhquant_init

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xh_model_zoo.xh_aigc.models.wan2_2 import Wan2_2ConvertConfig, Wan2_2Converter


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", type=str, default="/data01/home/xuchen/Wan2.2-main/ckpt/Wan_2.2")
    parser.add_argument("--task", type=str, default="i2v-A14B")
    parser.add_argument("--prompt", type=str, default="A calm seaside scene with gentle waves.")
    parser.add_argument(
        "--components",
        nargs="+",
        default=["high_noise_model"],
        choices=["t5", "vae_encode", "vae_decode", "low_noise_model", "high_noise_model"],
    )
    parser.add_argument(
        "--golden-components",
        nargs="+",
        default=[],
        choices=["t5", "vae_encode", "vae_decode", "low_noise_model", "high_noise_model"],
        help="Which components should export golden data; defaults to none.",
    )
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--use-lora-models", action="store_true", help="Whether to use LoRA models for export.")
    parser.add_argument("--quant-type", type=str, default="w8a8_ssfp")
    return parser.parse_args()


def main(args):
    target_device = DeviceType.XH2a
    work_dir = Path("work_dirs") / f"wan2_2_{args.task}"
    work_dir.mkdir(parents=True, exist_ok=True)
    xhquant_init(work_dir / "wan2_2_export.log", debug=False)
    quant_scheme = QuantScheme(target_device=target_device, quant_type=args.quant_type)
    config = Wan2_2ConvertConfig(
        quant_scheme=quant_scheme,
        task=args.task,
        prompt=args.prompt,
        sample_steps=args.steps,
        export_components=tuple(args.components),
        golden_components=tuple(args.golden_components) if args.golden_components is not None else (),
        use_resolved_float_loader=args.use_lora_models,
    )
    Wan2_2Converter.from_pretrained(args.model, config, str(work_dir))


if __name__ == "__main__":
    main(parse_args())
