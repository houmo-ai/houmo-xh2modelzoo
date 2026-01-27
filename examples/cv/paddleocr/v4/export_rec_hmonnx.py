# Copyright 2025 HOUMO AI
#
# File: export_rec_hmonnx.py
# Description:
#   Example script: cv/paddleocr/v4/export_rec_hmonnx.py
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

import argparse
import os
from pathlib import Path

import torch
from xhquant.api import (
    DeviceType,
    HMONNXGoldenInference,
    QuantScheme,
    convert_onnx_to_hmonnx,
    create_quant_config,
    get_root_logger,
    xhquant_init,
)


def main(args):
    onnx_file = args.onnx
    onnx_name = Path(onnx_file).stem
    work_dirs = Path("work_dirs") / onnx_name
    work_dirs.mkdir(exist_ok=True, parents=True)
    target_device = DeviceType.XH2a
    out_hmonnx_file = work_dirs / "hmonnx" / f"{onnx_name}_{target_device}.onnx"
    out_hmonnx_file.parent.mkdir(exist_ok=True, parents=True)
    out_hmonnx_file = str(out_hmonnx_file)
    if not os.path.exists(out_hmonnx_file):
        xhquant_init(None, debug=args.debug)
        quant_type = args.quant_type
        quant_scheme = QuantScheme(target_device=DeviceType.XH2a, quant_type=quant_type)

        quant_config = create_quant_config(quant_scheme)
        convert_onnx_to_hmonnx(
            onnx_file,
            [
                torch.randn(1, 3, 48, 320, dtype=torch.float32),
            ],
            DeviceType.XH2a,
            out_hmonnx_file,
            quant_config=quant_config,
            input_names=["generated_tensor_968"],
            output_names=["save_infer_model/scale_0.tmp_0"],
        )
        logger = get_root_logger()
        logger.info(f"Convert onnx to hmonnx success, out hmonnx file to: {out_hmonnx_file}")

    hm_model = HMONNXGoldenInference(out_hmonnx_file)
    hm_model.save_golden = True
    hm_model.exec_device = "cuda"

    golden_dir = work_dirs / "golden" / f"{onnx_name}"
    Path(golden_dir).mkdir(exist_ok=True, parents=True)
    hm_model.golden_dir = golden_dir

    import numpy as np

    x = np.load("/data01/home/yujy/work/OnnxOCR/rec_input.npy")
    y = hm_model(torch.from_numpy(x).cuda().half())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=str, default="data/models/paddleocr/v4/paddleocrv4_rec-sim.onnx")
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument("--quant-type", default="w8a8h1_sefp", help="quant type, default is w8a8")
    args = parser.parse_args()
    main(args)
