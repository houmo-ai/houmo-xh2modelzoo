# Copyright 2025 HOUMO AI
#
# File: export_hmonnx.py
# Description:
#   Example script: cv/rtdetr/export_hmonnx.py
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
    xhquant_init(None, debug=args.debug)
    onnx_file = args.onnx
    onnx_name = Path(onnx_file).stem
    onnx_name = f"{onnx_name}"
    work_dirs = Path("work_dirs") / onnx_name
    work_dirs.mkdir(exist_ok=True, parents=True)
    target_device = DeviceType.XH2a
    out_hmonnx_file = work_dirs / "hmonnx" / f"{onnx_name}_{target_device}.onnx"
    out_hmonnx_file.parent.mkdir(exist_ok=True, parents=True)
    out_hmonnx_file: str = str(out_hmonnx_file)

    im_shape = torch.randn(1, 2, dtype=torch.float32)
    image = torch.randn(1, 3, 640, 640, dtype=torch.float32)
    scale_factor = torch.randn(1, 2, dtype=torch.float32)
    input = [
        im_shape,
        image,
        scale_factor,
    ]
    quant_type = args.quant_type
    quant_scheme = QuantScheme(target_device=DeviceType.XH2a, quant_type=quant_type)

    quant_config = create_quant_config(quant_scheme)
    convert_onnx_to_hmonnx(
        onnx_file,
        input,
        DeviceType.XH2a,
        out_hmonnx_file,
        quant_config=quant_config,
        # input_names=["images"],
        # output_names=["cls_score"],
    )
    logger = get_root_logger()
    logger.info(f"Convert onnx to hmonnx success, out hmonnx file to: {out_hmonnx_file}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    session = HMONNXGoldenInference(out_hmonnx_file)
    session.to(device)
    session.save_golden = True
    session.golden_dir = work_dirs / f"hmonnx/golden_{quant_type}"
    session.step = 0

    im_shape = torch.randn(1, 2, dtype=torch.float16)
    image = torch.randn(1, 3, 640, 640, dtype=torch.float16)
    scale_factor = torch.randn(1, 2, dtype=torch.float16)
    input = [
        im_shape,
        image,
        scale_factor,
    ]
    session(
        im_shape,
        image,
        scale_factor,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--onnx",
        type=str,
        default="data/models/rtdetr/rtdetr_hgnetv2_l_6x_coco_d.onnx",
    )
    parser.add_argument("--quant-type", default="w8a8h1_sefp", help="quant type, default is w8a8")
    parser.add_argument("--debug", action="store_true", help="debug mode")
    args = parser.parse_args()
    main(args)
