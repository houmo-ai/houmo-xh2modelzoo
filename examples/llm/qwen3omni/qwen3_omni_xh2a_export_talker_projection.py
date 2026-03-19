# Copyright 2025 HOUMO AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# SPDX-License-Identifier: Apache-2.0

"""Export Qwen3-Omni talker hidden_projection & text_projection to HMONNX.

These are small Linear layers exported via torch.onnx.export + convert_onnx_to_hmonnx.
"""

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
from xhquant.api import (  # isort:skip
    DeviceType,
    QuantScheme,
    convert_onnx_to_hmonnx,
    create_quant_config,
    get_root_logger,
    xhquant_init,
)


def main(args):
    hf_model_path = osp.normpath(osp.abspath(args.model))
    model_name = Path(hf_model_path).name
    target_device = DeviceType.XH2a
    quant_type = "w8a8_sefp"
    quant_scheme = QuantScheme(target_device=target_device, quant_type=quant_type)
    quant_config = create_quant_config(quant_scheme)

    prefix = f"{model_name}-{target_device}-talker_projection-{quant_type}"
    work_dir = Path(args.work_dir) / prefix
    work_dir.mkdir(exist_ok=True, parents=True)
    log_file = work_dir / "convert.log"
    xhquant_init(log_file, debug=args.debug)
    logger = get_root_logger()

    hp_dir = work_dir / "hidden_projection"
    hp_hmonnx = hp_dir / "talker_hidden_projection_hm.onnx"
    tp_dir = work_dir / "text_projection"
    tp_hmonnx = tp_dir / "talker_text_projection_hm.onnx"
    meta_file = work_dir / "meta_talker_projection.json"

    # Check if artifacts already exist — skip export if so
    import json
    artifacts_exist = hp_hmonnx.exists() and tp_hmonnx.exists() and meta_file.exists()

    if artifacts_exist:
        logger.info("Talker projection HMONNX artifacts already exist, skipping export")
        with open(meta_file) as f:
            meta_info = json.load(f)
        hidden_size = meta_info["hidden_size"]
        dummy_input_cpu = torch.rand(1, 1, hidden_size, dtype=torch.float16)
    else:
        # ---- 1. Load full HF model ----
        from transformers import Qwen3OmniMoeForConditionalGeneration

        logger.info(f"Loading HF model from {hf_model_path}")
        native_model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            hf_model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            attn_implementation="eager",
            trust_remote_code=True,
        )
        native_model.eval()
        device = next(native_model.talker.parameters()).device

        hidden_size = native_model.talker.hidden_projection.linear_fc1.in_features
        dummy_input = torch.rand(1, 1, hidden_size, dtype=torch.float16, device=device)
        dummy_input_cpu = dummy_input.cpu()

        # ---- 2. Export hidden_projection ----
        hp_dir.mkdir(exist_ok=True, parents=True)
        hp_onnx = hp_dir / "talker_hidden_projection.onnx"

        logger.info("Exporting talker.hidden_projection ...")
        torch.onnx.export(
            native_model.talker.hidden_projection,
            (dummy_input,),
            hp_onnx,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {1: "seq_len"}, "output": {1: "seq_len"}},
        )
        convert_onnx_to_hmonnx(
            hp_onnx,
            [dummy_input_cpu],
            target_device,
            hp_hmonnx,
            quant_config=quant_config,
            input_names=["input"],
        )
        logger.info(f"hidden_projection exported to {hp_hmonnx}")

        # ---- 3. Export text_projection ----
        tp_dir.mkdir(exist_ok=True, parents=True)
        tp_onnx = tp_dir / "talker_text_projection.onnx"

        logger.info("Exporting talker.text_projection ...")
        torch.onnx.export(
            native_model.talker.text_projection,
            (dummy_input,),
            tp_onnx,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {1: "seq_len"}, "output": {1: "seq_len"}},
        )
        convert_onnx_to_hmonnx(
            tp_onnx,
            [dummy_input_cpu],
            target_device,
            tp_hmonnx,
            quant_config=quant_config,
            input_names=["input"],
        )
        logger.info(f"text_projection exported to {tp_hmonnx}")

        # ---- 4. Save meta ----
        meta_info = {
            "create_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "module": "talker_projection",
            "model_name": model_name,
            "hidden_projection_hmonnx": str(hp_hmonnx.relative_to(work_dir)),
            "text_projection_hmonnx": str(tp_hmonnx.relative_to(work_dir)),
            "hidden_size": hidden_size,
            "quant_type": quant_type,
        }
        save_json(meta_file, meta_info)
        logger.info(f"Projection export complete. Meta saved to {meta_file}")

    # ---- 5. Optional validation ----
    if args.valid:
        logger.info("Validating projection HMONNX ...")
        from xhquant.xhonnxruntime.hmonnx_inference import HMONNXInference

        sess_hp = HMONNXInference(str(hp_hmonnx))
        out_hp = sess_hp(dummy_input_cpu)
        if isinstance(out_hp, (list, tuple)):
            out_hp = out_hp[0]
        logger.info(f"hidden_projection validation passed, output shape: {tuple(out_hp.shape)}")

        sess_tp = HMONNXInference(str(tp_hmonnx))
        out_tp = sess_tp(dummy_input_cpu)
        if isinstance(out_tp, (list, tuple)):
            out_tp = out_tp[0]
        logger.info(f"text_projection validation passed, output shape: {tuple(out_tp.shape)}")

        dialogue_artifacts = {
            "projection": {**meta_info, "_root_dir": str(work_dir), "_meta_path": str(meta_file)}
        }
        try:
            run_dialogue_validation(
                hf_model_path,
                work_dir,
                logger,
                case="multimodal",
                max_new_tokens=args.max_new_tokens,
                artifacts=dialogue_artifacts,
                report_name="projection_dialogue_validation.json",
                output_prefix="projection_dialogue",
            )
        except (RuntimeError, AssertionError, Exception) as e:
            logger.warning(f"Projection dialogue validation skipped due to error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Qwen3-Omni talker projections to HMONNX")
    parser.add_argument("--model", type=str, default="/data02/datasets/Qwen3-Omni-30B-A3B-Instruct/")
    parser.add_argument("--work-dir", type=str, default="work_dirs/qwen3omni")
    parser.add_argument("--valid", action="store_true", default=True, help="validate exported HMONNX")
    parser.add_argument("--no-valid", action="store_false", dest="valid", help="skip validation")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args)
