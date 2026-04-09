# Copyright 2025 HOUMO AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# SPDX-License-Identifier: Apache-2.0

"""Export Qwen3-Omni talker projection bundle to HMONNX.

The talker always consumes ``hidden_projection`` and ``text_projection`` together,
so they are exported as a single logical module with two outputs.
"""

import argparse
import os.path as osp
import time
from pathlib import Path
import sys

import torch
from torch import nn

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _hmonnx_pipeline import run_dialogue_validation, save_json

try:
    from _hmonnx_pipeline import release_export_cuda_memory
except ImportError:
    def release_export_cuda_memory(logger=None, label=None):
        return None
from xhquant.api import (  # isort:skip
    DeviceType,
    QuantScheme,
    convert_onnx_to_hmonnx,
    create_quant_config,
    get_root_logger,
    xhquant_init,
)


class TalkerProjectionBundle(nn.Module):
    def __init__(self, hidden_projection: nn.Module, text_projection: nn.Module):
        super().__init__()
        self.hidden_projection = hidden_projection
        self.text_projection = text_projection

    def forward(self, hidden_states):
        hidden_output = self.hidden_projection(hidden_states)
        text_output = self.text_projection(hidden_states)
        return hidden_output, text_output


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
    native_model = None
    projection_bundle = None
    dummy_input = None

    projection_dir = work_dir / "projection"
    projection_hmonnx = projection_dir / "talker_projection_hm.onnx"
    meta_file = work_dir / "meta_talker_projection.json"

    # Check if artifacts already exist — skip export if so
    import json
    artifacts_exist = projection_hmonnx.exists() and meta_file.exists()

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
            device_map="cpu",
            attn_implementation="eager",
            trust_remote_code=True,
        )
        native_model.eval()

        hidden_size = native_model.talker.hidden_projection.linear_fc1.in_features
        dummy_input = torch.rand(1, 1, hidden_size, dtype=torch.float16)
        dummy_input_cpu = dummy_input

        # ---- 2. Export projection bundle ----
        projection_dir.mkdir(exist_ok=True, parents=True)
        projection_onnx = projection_dir / "talker_projection.onnx"
        projection_bundle = TalkerProjectionBundle(
            native_model.talker.hidden_projection,
            native_model.talker.text_projection,
        ).to(torch.float16).cpu()
        native_model = None
        release_export_cuda_memory(logger, "talker projection preparation")

        logger.info("Exporting talker projection bundle ...")
        torch.onnx.export(
            projection_bundle,
            (dummy_input,),
            projection_onnx,
            input_names=["input"],
            output_names=["hidden_projection_output", "text_projection_output"],
            dynamic_axes={
                "input": {1: "seq_len"},
                "hidden_projection_output": {1: "seq_len"},
                "text_projection_output": {1: "seq_len"},
            },
        )
        convert_onnx_to_hmonnx(
            projection_onnx,
            [dummy_input_cpu],
            target_device,
            projection_hmonnx,
            quant_config=quant_config,
            input_names=["input"],
            output_names=["hidden_projection_output", "text_projection_output"],
        )
        logger.info(f"talker projection bundle exported to {projection_hmonnx}")

        # ---- 3. Save meta ----
        meta_info = {
            "create_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "module": "talker_projection",
            "model_name": model_name,
            "talker_projection_hmonnx": str(projection_hmonnx.relative_to(work_dir)),
            "hidden_size": hidden_size,
            "quant_type": quant_type,
        }
        save_json(meta_file, meta_info)
        logger.info(f"Projection export complete. Meta saved to {meta_file}")

    projection_bundle = None
    dummy_input = None
    native_model = None
    release_export_cuda_memory(logger, "talker projection export")

    # ---- 4. Optional validation ----
    if args.valid:
        logger.info("Validating projection HMONNX ...")
        from xhquant.xhonnxruntime.hmonnx_inference import HMONNXInference

        session = HMONNXInference(str(projection_hmonnx))
        outputs = session(dummy_input_cpu)
        if not isinstance(outputs, (list, tuple)) or len(outputs) != 2:
            raise RuntimeError("Projection bundle validation expected two outputs")
        out_hp, out_tp = outputs
        logger.info(f"hidden_projection validation passed, output shape: {tuple(out_hp.shape)}")
        logger.info(f"text_projection validation passed, output shape: {tuple(out_tp.shape)}")
        session = None
        outputs = None
        out_hp = None
        out_tp = None
        release_export_cuda_memory(logger, "talker projection hmonnx validation")

        dialogue_artifacts = {
            "projection": {**meta_info, "_root_dir": str(work_dir), "_meta_path": str(meta_file)}
        }
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
