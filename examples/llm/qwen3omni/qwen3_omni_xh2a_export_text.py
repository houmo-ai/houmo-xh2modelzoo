# Copyright 2025 HOUMO AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# SPDX-License-Identifier: Apache-2.0

"""Export Qwen3-Omni thinker text module to HMONNX."""

import argparse
import os.path as osp
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _hmonnx_pipeline import discover_artifacts, run_text_hmonnx_chain_forward

try:
    from _hmonnx_pipeline import release_export_cuda_memory
except ImportError:
    def release_export_cuda_memory(logger=None, label=None):
        return None
from xh_model_zoo.xh_llm import LLMConverter
from xh_model_zoo.xh_llm.models.qwen3_omni import Qwen3OmniMoeConvertConfig

from xhquant.api import DeviceType, QuantScheme, get_root_logger, xhquant_init  # isort:skip
from xh_model_zoo.utils.memory_tracker import MemoryTracker  # isort:skip
from xh_model_zoo.utils.time_profiler import TimeProfiler  # isort:skip


def main(args):
    hf_model_path = osp.normpath(osp.abspath(args.model))
    model_name = Path(hf_model_path).name
    target_device = DeviceType.XH2a
    quant_type = args.quant_type
    quant_scheme = QuantScheme(target_device=target_device, quant_type=quant_type)

    config = Qwen3OmniMoeConvertConfig(
        batch_size=1,
        context_length=args.context_length,
        input_sequence_length=args.input_sequence_length,
        quant_scheme=quant_scheme,
        quant_weight=args.quant_weight,
        num_logits_to_keep=args.num_logits_to_keep,
        export_audio_encoder=False,
        export_vision_encoder=False,
        export_talker_model=False,
        export_talker_prediction=False,
    )

    prefix = f"{model_name}-{target_device}-text-{args.context_length // 1024}k-{quant_type}"
    work_dir = Path(args.work_dir) / prefix
    work_dir.mkdir(exist_ok=True, parents=True)
    log_file = work_dir / "convert.log"
    xhquant_init(log_file, debug=args.debug)
    logger = get_root_logger()

    meta_file = work_dir / "meta.json"
    prefill_file = work_dir / "hmonnx" / "prefill" / f"{model_name}-{target_device}-{args.context_length // 1024}k-{quant_type}_prefill.onnx"
    decode_file = work_dir / "hmonnx" / "decode" / f"{model_name}-{target_device}-{args.context_length // 1024}k-{quant_type}_decoder.onnx"

    if meta_file.exists() and prefill_file.exists() and decode_file.exists():
        logger.info(f"Reusing existing text export artifacts in {work_dir}")
    else:
        logger.info(f"Exporting thinker text module from {hf_model_path}")
        with TimeProfiler("convert_text", logger), MemoryTracker("cuda:0", "convert_text", logger):
            LLMConverter.from_pretrained(
                hf_model_path, "Qwen3OmniMoeForConditionalGeneration", config, str(work_dir)
            )
        logger.info(f"Text module export complete. Artifacts in {work_dir}")
        release_export_cuda_memory(logger, "text export")

    if args.valid:
        artifacts = discover_artifacts(work_dir)
        text_meta = artifacts.get("text")
        if text_meta is None:
            raise RuntimeError(f"No text export meta.json found under {work_dir}")

        chain_case = "multimodal" if "audio" in artifacts and "vision" in artifacts else "text"
        run_text_hmonnx_chain_forward(
            hf_model_path,
            text_meta,
            logger,
            case=chain_case,
            audio_meta=None,
            vision_meta=None,
            report_path=work_dir / "text_hmonnx_chain_report.json",
            max_new_tokens=args.max_new_tokens,
            device_map="auto",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Qwen3-Omni thinker text to HMONNX")
    parser.add_argument("--model", type=str, default="/data02/datasets/Qwen3-Omni-30B-A3B-Instruct/")
    parser.add_argument("--work-dir", type=str, default="work_dirs/qwen3omni")
    parser.add_argument("--quant-type", default="w8a8h0_sefp", help="quantization type")
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--input-sequence-length", type=int, default=32)
    parser.add_argument("--num_logits_to_keep", type=int, default=1)
    parser.add_argument("--quant-weight", type=str, default=None, help="path to quant weight (gptq/quarot)")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--valid", action="store_true", default=True, help="validate exported text artifacts with HMONNX forward")
    parser.add_argument("--no-valid", action="store_false", dest="valid", help="skip validation")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args)
