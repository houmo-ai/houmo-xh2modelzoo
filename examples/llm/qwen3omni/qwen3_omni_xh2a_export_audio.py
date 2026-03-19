# Copyright 2025 HOUMO AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# SPDX-License-Identifier: Apache-2.0

"""Export Qwen3-Omni audio encoder to HMONNX."""

import argparse
import os.path as osp
import time
from pathlib import Path
import sys

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _hmonnx_pipeline import run_dialogue_validation, save_json
from xh_model_zoo.xh_llm.models.base_converter import BaseConverter
from xh_model_zoo.xh_llm.models.builder import wrap_llm_model

from xhquant.api import (  # isort:skip
    CacheTensor,
    Config,
    ConfigDict,
    DeviceType,
    QuantScheme,
    convert_fx_model_to_hmonnx,
    create_quant_config,
    get_root_logger,
    xhquant_init,
)
from xh_model_zoo.utils.memory_tracker import MemoryTracker  # isort:skip
from xh_model_zoo.utils.time_profiler import TimeProfiler  # isort:skip

def main(args):
    hf_model_path = osp.normpath(osp.abspath(args.model))
    model_name = Path(hf_model_path).name
    target_device = DeviceType.XH2a
    quant_type = args.quant_type

    prefix = f"{model_name}-{target_device}-audio-{quant_type}"
    work_dir = Path(args.work_dir) / prefix
    work_dir.mkdir(exist_ok=True, parents=True)
    log_file = work_dir / "convert.log"
    xhquant_init(log_file, debug=args.debug)
    logger = get_root_logger()

    # ---- 1. Load HF model and extract audio tower ----
    from transformers import Qwen3OmniMoeForConditionalGeneration

    logger.info(f"Loading HF model from {hf_model_path}")
    native_model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        hf_model_path,
        torch_dtype=torch.float16,
        device_map="cpu",
        attn_implementation="eager",
        trust_remote_code=True,
    )
    native_model.eval()

    audio_tower = native_model.thinker.audio_tower
    if audio_tower is None:
        raise RuntimeError("Model does not have audio_tower")

    # Free memory for non-audio components
    del native_model.talker, native_model.code2wav
    import gc

    gc.collect()
    torch.cuda.empty_cache()

    # ---- 2. Register wrap modules and wrap ----
    from xh_model_zoo.xh_llm.models.qwen3_omni._audio_model import (
        register_wrap_modules as audio_register_wrap_modules,
    )

    audio_register_wrap_modules()

    audio_tower = audio_tower.to(torch.float16).cpu()
    wrapped_audio = wrap_llm_model(audio_tower, Config(dict()))

    # ---- 3. Prepare dummy inputs ----
    # After FX trace, only padded_feature and cu_seqlens have graph users
    # (padded_mask_after_cnn is declared in forward but unused)
    batch_size = 1
    mel_bins = int(getattr(audio_tower.config, "num_mel_bins", 128))
    mel_length = 100
    cnn_steps = 13
    dummy_feature = torch.randn(batch_size, mel_bins, mel_length, dtype=torch.float16)
    dummy_cu = torch.tensor([0, cnn_steps], dtype=torch.int32)

    inputs = (dummy_feature, dummy_cu)
    input_names = ["padded_feature", "cu_seqlens"]
    output_names = ["audio_embeds"]

    # ---- 4. Export HMONNX ----
    audio_dir = work_dir / "hmonnx"
    audio_dir.mkdir(exist_ok=True, parents=True)
    audio_onnx_file = audio_dir / f"{model_name}-audio_encoder.onnx"

    logger.info(f"Exporting audio encoder to {audio_onnx_file}")
    with TimeProfiler("export_audio", logger), MemoryTracker("cuda:0", "export_audio", logger):
        compatible_names = BaseConverter.xh1_hmonnx_compatible(input_names)
        convert_fx_model_to_hmonnx(
            wrapped_audio,
            inputs,
            target_device,
            audio_onnx_file,
            input_names=compatible_names,
            output_names=output_names,
        )
    logger.info(f"Audio encoder export successful: {audio_onnx_file}")

    # ---- 5. Save meta ----
    meta_info = {
        "create_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "module": "audio_encoder",
        "model_name": model_name,
        "audio_encoder_onnx": str(audio_onnx_file.relative_to(work_dir)),
        "audio_mel_dim": mel_bins,
        "audio_max_length": mel_length,
        "audio_batch_size": batch_size,
    }
    meta_file = work_dir / "meta_audio.json"
    save_json(meta_file, meta_info)
    logger.info(f"Audio export complete. Meta saved to {meta_file}")

    # ---- 6. Optional validation ----
    if args.valid:
        logger.info("Validating audio HMONNX ...")
        from xhquant.xhonnxruntime.hmonnx_inference import HMONNXInference

        session = HMONNXInference(str(audio_onnx_file))
        output = session(dummy_feature, dummy_cu)
        if isinstance(output, (list, tuple)):
            output = output[0]
        logger.info(f"Audio HMONNX validation passed, output shape: {tuple(output.shape)}")

        dialogue_artifacts = {
            "audio": {**meta_info, "_root_dir": str(work_dir), "_meta_path": str(meta_file)}
        }
        run_dialogue_validation(
            hf_model_path,
            work_dir,
            logger,
            case="audio",
            max_new_tokens=args.max_new_tokens,
            artifacts=dialogue_artifacts,
            report_name="audio_dialogue_validation.json",
            output_prefix="audio_dialogue",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Qwen3-Omni audio encoder to HMONNX")
    parser.add_argument("--model", type=str, default="/data02/datasets/Qwen3-Omni-30B-A3B-Instruct/")
    parser.add_argument("--work-dir", type=str, default="work_dirs/qwen3omni")
    parser.add_argument("--quant-type", default="w8a8h0_sefp")
    parser.add_argument("--valid", default=True, help="validate exported HMONNX")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args)
