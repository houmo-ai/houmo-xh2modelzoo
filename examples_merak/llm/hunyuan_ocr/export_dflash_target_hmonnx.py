#!/usr/bin/env python3
# Copyright 2025 HOUMO AI
# SPDX-License-Identifier: Apache-2.0

"""Export HunyuanOCR target prefill/decode graphs with DFlash target hidden outputs."""

from __future__ import annotations

import argparse
from pathlib import Path


def build_model_config(model_path: Path) -> dict:
    return {
        "chip_arch": "XH2a",
        "model_type": "HunYuanVLForConditionalGeneration",
        "hf_model": str(model_path),
        "model_name": "hunyuan_ocr_dflash_target",
        "context_max_length": 131072,
        "prefill_chunk_length": 256,
        "max_pe_length": 131072,
        "use_cache": True,
        "num_logits_to_keep": 1,
        "spec_decode_mode": "dflash",
        "dflash_config": {"hf_model": str(model_path / "dflash")},
        "quant_scheme": {
            "quant_type": "w16a16h0_sefp",
            "nodes": {"lm_head": {"quant_type": "w16a16h0_sefp"}},
            "ops": {},
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("data/models/HunyuanOCR"))
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    from xhmodel_merak.xh_llm import AutoLLMConfig, AutoLLMModel
    from xhquant.api import xhquant_init

    args = _parse_args()
    for required_path in (args.model / "config.json", args.model / "dflash" / "config.json"):
        if not required_path.is_file():
            raise FileNotFoundError(f"HunyuanOCR DFlash target export is missing {required_path}")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite a non-empty export directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    xhquant_init(str(args.output_dir / "export_dflash_target_hmonnx.log"), False)
    config = AutoLLMConfig.from_pretrained(build_model_config(args.model))
    model = AutoLLMModel.from_pretrained(config)
    metadata = model.export_text_hmonnx(str(args.output_dir))
    print(args.output_dir / "text" / "text_meta_info.json")
    print(metadata)


if __name__ == "__main__":
    main()