# Copyright 2025 HOUMO AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# SPDX-License-Identifier: Apache-2.0

"""Export Qwen3-Omni code2wav vocoder to HMONNX.

Exports code2wav via torch.onnx.export + convert_onnx_to_hmonnx with static code_len,
then monkey-patches forward/chunked_decode and validates with a full generate call.
"""

import argparse
import os.path as osp
import shutil
import time
import sys
from pathlib import Path

import onnx
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _hmonnx_pipeline import run_dialogue_validation, save_json
from xhquant.api import (  # isort:skip
    DeviceType,
    QuantScheme,
    convert_onnx_to_hmonnx,
    create_quant_config,
    get_root_logger,
    xhquant_init,
)

try:
    from qwen_omni_utils import process_mm_info
except ImportError:

    def process_mm_info(conversation, use_audio_in_video=False):
        audios, images, videos = [], [], []
        for turn in conversation:
            for item in turn.get("content", []):
                tp = item.get("type")
                if tp == "audio":
                    audios.append(item.get("audio"))
                elif tp == "image":
                    images.append(item.get("image"))
                elif tp == "video":
                    videos.append(item.get("video"))
        return audios, images, videos


def _dedupe_onnx_node_outputs(onnx_path: Path):
    model = onnx.load(str(onnx_path))
    producers = {}
    for node_idx, node in enumerate(model.graph.node):
        for output_idx, output_name in enumerate(node.output):
            if not output_name:
                continue
            producers.setdefault(output_name, []).append((node_idx, output_idx))

    for tensor_name, occurrences in producers.items():
        if len(occurrences) <= 1:
            continue

        for dup_idx, (producer_idx, output_idx) in enumerate(occurrences[:-1]):
            next_producer_idx = occurrences[dup_idx + 1][0]
            new_name = f"{tensor_name}__dedup_{dup_idx}"
            model.graph.node[producer_idx].output[output_idx] = new_name

            for node in model.graph.node[producer_idx + 1 : next_producer_idx]:
                for input_idx, input_name in enumerate(node.input):
                    if input_name == tensor_name:
                        node.input[input_idx] = new_name

            for value_info in model.graph.value_info:
                if value_info.name == tensor_name:
                    value_info.name = new_name

    onnx.save(model, str(onnx_path))


def _load_native_model_for_export(hf_model_path: str, logger):
    from transformers import Qwen3OmniMoeForConditionalGeneration

    logger.info(f"Loading HF model from {hf_model_path} for code2wav export on CPU")
    native_model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        hf_model_path,
        torch_dtype=torch.float16,
        device_map="cpu",
        attn_implementation="eager",
        trust_remote_code=True,
    )
    native_model.eval()
    return native_model


def _run_code2wav_validation(
    hf_model_path: str,
    work_dir: Path,
    golden_dir: Path,
    logger,
    meta_info,
    meta_file: Path,
    max_new_tokens: int,
    validation_device_map: str,
    save_golden: bool,
):
    dialogue_artifacts = {"code2wav": {**meta_info, "_root_dir": str(work_dir), "_meta_path": str(meta_file)}}
    report = run_dialogue_validation(
        hf_model_path,
        work_dir,
        logger,
        case="multimodal",
        max_new_tokens=max_new_tokens,
        device_map=validation_device_map,
        artifacts=dialogue_artifacts,
        report_name="code2wav_dialogue_validation.json",
        output_prefix="code2wav_dialogue",
        save_golden=save_golden,
        golden_dir=golden_dir,
    )
    logger.info(f"Generate text: {report.get('output_text', [])}")

    audio_file = report.get("audio_file")
    if audio_file:
        generated_wav = work_dir / audio_file
        compat_wav = work_dir / "output_code2wav_valid.wav"
        if generated_wav.exists() and generated_wav != compat_wav:
            shutil.copyfile(generated_wav, compat_wav)
        logger.info(f"Audio saved to {compat_wav}")

    logger.info("code2wav HMONNX validation passed.")
    return report


def main(args):
    hf_model_path = osp.normpath(osp.abspath(args.model))
    model_name = Path(hf_model_path).name
    target_device = DeviceType.XH2a
    quant_type = "w8a8_sefp"
    quant_scheme = QuantScheme(target_device=target_device, quant_type=quant_type)
    quant_config = create_quant_config(quant_scheme)

    prefix = f"{model_name}-{target_device}-code2wav-{quant_type}"
    work_dir = Path(args.work_dir) / prefix
    golden_root = Path(args.golden_root)
    if not golden_root.is_absolute():
        golden_root = (SCRIPT_DIR.parents[2] / golden_root).resolve()
    golden_dir = golden_root / prefix / "golden"
    work_dir.mkdir(exist_ok=True, parents=True)
    log_file = work_dir / "convert.log"
    xhquant_init(log_file, debug=args.debug)
    logger = get_root_logger()

    onnx_dir = work_dir / "onnx"
    onnx_dir.mkdir(exist_ok=True, parents=True)
    onnx_path = onnx_dir / "code2wav.onnx"
    hmonnx_path = onnx_dir / "code2wav_hm.onnx"
    meta_file = work_dir / "meta_code2wav.json"

    # ---- 2. Export code2wav ONNX + HMONNX ----
    static_code_len = args.static_code_len
    native_model = None

    if hmonnx_path.exists() and meta_file.exists():
        logger.info("code2wav HMONNX artifacts already exist, skipping export")
        import json

        with open(meta_file) as f:
            meta_info = json.load(f)
    else:
        native_model = _load_native_model_for_export(hf_model_path, logger)
        dummy_codes = torch.randint(100, (1, 16, static_code_len), dtype=torch.int32)

        logger.info(f"Exporting code2wav with static_code_len={static_code_len} ...")
        torch.onnx.export(
            native_model.code2wav.cpu(),
            dummy_codes,
            onnx_path,
            opset_version=18,
            dynamo=True,
            do_constant_folding=True,
            input_names=["codes"],
            output_names=["embedding"],
        )
        _dedupe_onnx_node_outputs(onnx_path)
        convert_onnx_to_hmonnx(
            onnx_path,
            [dummy_codes],
            target_device,
            hmonnx_path,
            quant_config=quant_config,
            input_names=["codes"],
        )
        logger.info(f"code2wav exported to {hmonnx_path}")

        # ---- 3. Save meta ----
        meta_info = {
            "create_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "module": "code2wav",
            "model_name": model_name,
            "code2wav_hmonnx": str(hmonnx_path.relative_to(work_dir)),
            "static_code_len": static_code_len,
            "quant_type": quant_type,
        }
        save_json(meta_file, meta_info)
        logger.info(f"code2wav export complete. Meta saved to {meta_file}")

    # ---- 4. Optional full-generate validation ----
    if args.valid:
        logger.info("Validating code2wav HMONNX with full generate ...")
        import gc

        if native_model is not None:
            del native_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        _run_code2wav_validation(
            hf_model_path,
            work_dir,
            golden_dir,
            logger,
            meta_info,
            meta_file,
            max_new_tokens=args.max_new_tokens,
            validation_device_map=args.validation_device_map,
            save_golden=args.save_golden,
        )
    else:
        if native_model is not None:
            del native_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Qwen3-Omni code2wav to HMONNX")
    parser.add_argument("--model", type=str, default="/data02/datasets/Qwen3-Omni-30B-A3B-Instruct/")
    parser.add_argument("--work-dir", type=str, default="work_dirs/qwen3omni")
    parser.add_argument("--static-code-len", type=int, default=126, help="static code length for HMONNX export")
    parser.add_argument("--valid", action="store_true", default=True, help="validate with full generate")
    parser.add_argument("--no-valid", action="store_false", dest="valid", help="skip validation")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--validation-device-map", type=str, default="auto")
    parser.add_argument("--golden-root", type=str, default="work_dirs/qwen3omni_no_projection")
    parser.add_argument("--save-golden", action="store_true", default=True, help="save golden outputs after validation")
    parser.add_argument("--no-save-golden", action="store_false", dest="save_golden", help="skip golden output save")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args)
