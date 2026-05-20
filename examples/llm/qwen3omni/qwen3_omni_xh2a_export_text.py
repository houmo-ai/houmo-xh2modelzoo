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
import json
import os.path as osp
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _hmonnx_pipeline import discover_artifacts, run_text_hmonnx_chain_forward
from _thinker_gptq_view import (
    is_qwen3_omni_checkpoint,
    is_qwen3_omni_gptq_checkpoint,
    prepare_qwen3_omni_thinker_text_view,
)

try:
    from _hmonnx_pipeline import release_export_cuda_memory
except ImportError:

    def release_export_cuda_memory(logger=None, label=None):
        return None


from xh_model_zoo.xh_llm import LLMConverter
from xh_model_zoo.xh_llm.models.qwen3_omni import Qwen3OmniMoeConvertConfig

from xhquant.api import DeviceType, QuantScheme, get_root_logger, xhquant_init  # isort:skip
from xhquant.api.quant_type import parse_quant_format  # isort:skip
from xh_model_zoo.utils.memory_tracker import MemoryTracker  # isort:skip
from xh_model_zoo.utils.time_profiler import TimeProfiler  # isort:skip


def _resolve_accept_hidden_layer(model_dir: str):
    config_path = Path(model_dir) / "config.json"
    if not config_path.exists():
        return None

    try:
        config_payload = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    talker_config = config_payload.get("talker_config")
    if not isinstance(talker_config, dict):
        return None

    accept_hidden_layer = talker_config.get("accept_hidden_layer")
    if accept_hidden_layer is None:
        return None
    return int(accept_hidden_layer)


def main(args):
    hf_model_path = osp.normpath(osp.abspath(args.model))
    model_name = Path(hf_model_path).name
    target_device = DeviceType.XH2a
    accept_hidden_layer = _resolve_accept_hidden_layer(hf_model_path)
    use_qwen3omni_text_view = is_qwen3_omni_checkpoint(hf_model_path)
    use_qwen3omni_gptq_view = is_qwen3_omni_gptq_checkpoint(hf_model_path)
    if use_qwen3omni_text_view and accept_hidden_layer is None:
        raise ValueError(f"{hf_model_path} is missing talker_config.accept_hidden_layer")
    quant_type = args.quant_type
    if use_qwen3omni_gptq_view:
        quant_format = parse_quant_format(quant_type)
        if quant_format.weight_bit is not None and quant_format.weight_bit > 8:
            raise ValueError(
                f"GPTQ thinker export does not support quant_type={quant_type}: "
                "GPTQ-backed linear weights are exported through ssfp and currently require weight mantissa bits <= 8. "
                "Use a GPTQ-friendly profile such as w4a8_ssfp or w8a8h1_sefp."
            )

    quant_scheme = QuantScheme(target_device=target_device, quant_type=quant_type)
    if use_qwen3omni_gptq_view and args.quant_weight is not None:
        raise ValueError("GPTQ checkpoint model cannot be combined with external --quant-weight")

    prefix = f"{model_name}-{target_device}-text-{args.context_length // 1024}k-{quant_type}"
    if accept_hidden_layer is not None:
        prefix += f"-ahl{accept_hidden_layer}"
    prefix += "-mmpos-fullhidden"
    if use_qwen3omni_gptq_view:
        prefix += "-gptq"
    work_dir = Path(args.work_dir) / prefix
    golden_root = Path(args.golden_root)
    if not golden_root.is_absolute():
        golden_root = (SCRIPT_DIR.parents[2] / golden_root).resolve()
    golden_dir = golden_root / prefix / "golden"
    work_dir.mkdir(exist_ok=True, parents=True)
    log_file = work_dir / "convert.log"
    xhquant_init(log_file, debug=args.debug)
    logger = get_root_logger()

    meta_file = work_dir / "meta.json"
    prefill_file = (
        work_dir
        / "hmonnx"
        / "prefill"
        / f"{model_name}-{target_device}-{args.context_length // 1024}k-{quant_type}_prefill.onnx"
    )
    decode_file = (
        work_dir
        / "hmonnx"
        / "decode"
        / f"{model_name}-{target_device}-{args.context_length // 1024}k-{quant_type}_decoder.onnx"
    )

    reuse_existing = meta_file.exists() and prefill_file.exists() and decode_file.exists()
    if reuse_existing:
        try:
            existing_meta = json.loads(meta_file.read_text(encoding="utf-8"))
        except Exception:
            existing_meta = {}

        missing_meta_files = []
        for meta_key in ("token_embedding_file", "prefill_onnx", "decode_onnx"):
            meta_path = existing_meta.get(meta_key)
            if not meta_path or not (work_dir / meta_path).exists():
                missing_meta_files.append(meta_key)
        if missing_meta_files:
            logger.info(
                f"Rebuilding stale text export artifacts in {work_dir}: missing {', '.join(missing_meta_files)}"
            )
            reuse_existing = False
        elif use_qwen3omni_gptq_view and not existing_meta.get("gptq_expert_qzeros_normalized", False):
            logger.info(f"Rebuilding stale GPTQ text export artifacts in {work_dir}")
            reuse_existing = False

    if reuse_existing:
        logger.info(f"Reusing existing text export artifacts in {work_dir}")
    else:
        if use_qwen3omni_text_view:
            text_view_dir = prepare_qwen3_omni_thinker_text_view(
                hf_model_path,
                work_dir / "_thinker_text_view" / model_name,
            )
            export_model_path = str(text_view_dir)
            export_architecture = "Qwen3OmniMoeThinkerTextForCausalLM"
            export_config = Qwen3OmniMoeConvertConfig(
                batch_size=1,
                context_length=args.context_length,
                input_sequence_length=args.input_sequence_length,
                quant_scheme=quant_scheme,
                quant_weight=None if use_qwen3omni_gptq_view else args.quant_weight,
                accept_hidden_layer=accept_hidden_layer,
                num_logits_to_keep=args.num_logits_to_keep,
            )
            view_kind = "GPTQ thinker view" if use_qwen3omni_gptq_view else "HF thinker text view"
            logger.info(f"Exporting thinker text module from {view_kind} {export_model_path}")
        else:
            export_model_path = hf_model_path
            export_architecture = "Qwen3OmniMoeForConditionalGeneration"
            export_config = Qwen3OmniMoeConvertConfig(
                batch_size=1,
                context_length=args.context_length,
                input_sequence_length=args.input_sequence_length,
                quant_scheme=quant_scheme,
                quant_weight=args.quant_weight,
                accept_hidden_layer=accept_hidden_layer,
                num_logits_to_keep=args.num_logits_to_keep,
                export_audio_encoder=False,
                export_vision_encoder=False,
                export_talker_model=False,
                export_talker_prediction=False,
            )
            logger.info(f"Exporting thinker text module from {hf_model_path}")

        with TimeProfiler("convert_text", logger), MemoryTracker("cuda:0", "convert_text", logger):
            LLMConverter.from_pretrained(export_model_path, export_architecture, export_config, str(work_dir))
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
            save_golden=args.save_golden,
            golden_dir=golden_dir,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Qwen3-Omni thinker text to HMONNX")
    parser.add_argument("--model", type=str, default="/data01/datasets/Qwen3-Omni-30B-A3B-Instruct")
    parser.add_argument("--work-dir", type=str, default="work_dirs/qwen3omni")
    parser.add_argument("--quant-type", default="w4a8_ssfp", help="quantization type")
    parser.add_argument("--context-length", type=int, default=2048)
    parser.add_argument("--input-sequence-length", type=int, default=256)
    parser.add_argument("--num_logits_to_keep", type=int, default=1)
    parser.add_argument("--quant-weight", type=str, default=None, help="path to external quant weight (gptq/quarot)")
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--golden-root", type=str, default="work_dirs/qwen3omni_no_projection")
    parser.add_argument("--save-golden", action="store_true", default=True, help="save golden outputs after validation")
    parser.add_argument("--no-save-golden", action="store_false", dest="save_golden", help="skip golden output save")
    parser.add_argument(
        "--valid", action="store_true", default=True, help="validate exported text artifacts with HMONNX forward"
    )
    parser.add_argument("--no-valid", action="store_false", dest="valid", help="skip validation")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args)
