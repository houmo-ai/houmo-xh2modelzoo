#!/usr/bin/env python3
# Copyright 2025 HOUMO AI
# SPDX-License-Identifier: Apache-2.0

"""Export the fixed HunyuanOCR text prefill/decode HMONNX stage."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path


DEFAULT_WORKFLOW_CONFIG = Path(
    "configs_merak/workflows/xh2a/llm_models/hunyuan_ocr/base/"
    "hunyuan_ocr_base_xh2a_w16a16.yaml"
)


def load_model_config(config_path: Path, model_path: Path) -> dict:
    from xhmodel_merak.xh_llm.workflows import WorkflowConfig

    values = copy.deepcopy(dict(WorkflowConfig.from_file(str(config_path)).export["model"]))
    values["hf_model"] = str(model_path)
    values.pop("resolution_bucket_manifest", None)
    values.pop("visual_config", None)
    return values


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("data/models/HunyuanOCR"))
    parser.add_argument("--config-path", type=Path, default=DEFAULT_WORKFLOW_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> None:
    from xhmodel_merak.xh_llm import AutoLLMConfig, AutoLLMModel
    from xhquant.api import xhquant_init

    args = _parse_args()
    if not (args.model / "config.json").is_file():
        raise FileNotFoundError(f"HunyuanOCR checkpoint is missing config.json: {args.model}")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite a non-empty export directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    xhquant_init(str(args.output_dir / "export_text_hmonnx.log"), args.debug)
    config = AutoLLMConfig.from_pretrained(load_model_config(args.config_path, args.model))
    model = AutoLLMModel.from_pretrained(config)
    metadata = model.export_text_hmonnx(str(args.output_dir))
    print(args.output_dir / "text" / "text_meta_info.json")
    print(metadata)


if __name__ == "__main__":
    main()
