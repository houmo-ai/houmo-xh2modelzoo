# Copyright 2025 HOUMO AI
#
# File: deepseek_ocr_export copy.py
# Description:
#   Example script: llm/deepseek_ocr/deepseek_ocr_export copy.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import argparse,os.path as osp,torch
from pathlib import Path
from xhquant.api import DeviceType, xhquant_init, QuantScheme, get_root_logger, Config, release_quanted_model_unused_parameters  # isort:skip
from xh_model_zoo.utils import MemoryTracker, TimeProfiler
from xh_model_zoo.xh_llm.models.deepseek_ocr.deepseekv2_converter import DeepseekOCRConverterConfig, DeepseekOCRConverter

def main(args):
    # 1. Parse Config
    cfg = Config.fromfile(args.extra_config_file) if args.extra_config_file is not None else Config()
    cfg.merge_from_dict(
        dict(
            context_length=args.context_length,
            input_sequence_length=args.input_sequence_length,
            quant_weight=args.quant_weight,
            quant_scheme=dict(
                target_device=args.device, quant_type=args.quant_type, nodes=dict(lm_head="w8a8h1_sefp")
            ),
            export_mode=args.export_mode,
        )
    )
    config = DeepseekOCRConverterConfig.from_dict_or_other(cfg.to_dict())


    # 2. Convert
    hf_model_path = osp.normpath(osp.abspath(args.model))
    model_name = Path(hf_model_path).name
    target_device = args.device
    prefix = f"{model_name}-{target_device}-{args.context_length//1024}k-{args.quant_type}"
    work_dir = Path("work_dirs") / prefix
    work_dir.mkdir(exist_ok=True, parents=True)
    log_file = work_dir / "convert.log"
    xhquant_init(log_file, debug=args.debug)
    logger = get_root_logger()

    with TimeProfiler("convert", logger), MemoryTracker("cuda:0", "convert", logger):
        DeepseekOCRConverter.convert_and_export(hf_model_path, config, str(work_dir))
        C = DeepseekOCRConverter(hf_model_path, config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="debug mode")  # /data02/datasets/qwen3-8B-AWQ
    parser.add_argument(
        "--model",
        type=str,
        default="/data02/datasets/DeepSeek-OCR/",  # /data02/datasets/Qwen2.5-1.5B-Instruct-int4-sym-inc
    )
    parser.add_argument("--export-mode", type=str, default="Tiny", choices=["Tiny", "Small", "Base", "Large", "Gundam"],help="export mode, default is hmonnx")
    parser.add_argument("--context-length", type=int, default=8192, help="max sequence length")
    parser.add_argument("--input-sequence-length", type=int, default=256, help="input sequence length")
    parser.add_argument("--quant-type", default="w8a8h0_ssfp", help="quant type, default is w8a8")
    parser.add_argument(
        "--quant-weight",
        type=str,
        default=None,
        help="quant weight path, for example: gptq or quarot, if empty, use w8a8",
    )
    parser.add_argument("--device",type=str,default="xh2a",help="device, default is xh2a")
    parser.add_argument("--generate-golden", default=True, help="generate golden")
    parser.add_argument("--extra_config_file", type=str, default=None, help="extra config file")
    parser.add_argument("--demo", default=False, action="store_true", help="demo mode")
    args = parser.parse_args()
    main(args)
