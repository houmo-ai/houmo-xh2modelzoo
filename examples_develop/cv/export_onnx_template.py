# Copyright 2025 HOUMO AI
#
# File: export_onnx_template.py
# Description:
#   Export utilities for cv in HOUMO AI xh2modelzoo.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
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
from typing import List

import onnx
import torch
from onnx import TensorProto
from torch import Tensor
from xhquant.api import (
    DeviceType,
    QuantScheme,
    convert_onnx_to_hmonnx,
    create_quant_config,
    get_root_logger,
    xhquant_init,
)

TENSOR_TYPE_TO_TORCH_TYPE = {
    int(TensorProto.FLOAT): torch.float32,
    int(TensorProto.UINT8): torch.uint8,
    int(TensorProto.INT8): torch.int8,
    int(TensorProto.INT16): torch.int16,
    int(TensorProto.INT32): torch.int32,
    int(TensorProto.INT64): torch.int64,
    int(TensorProto.BOOL): torch.bool,
    int(TensorProto.FLOAT16): torch.float16,
    int(TensorProto.DOUBLE): torch.float64,
    int(TensorProto.COMPLEX64): torch.complex64,
    int(TensorProto.COMPLEX128): torch.complex128,
}


def main(args):
    xhquant_init()
    logger = get_root_logger()
    onnx_file = args.onnx

    onnx_name = Path(onnx_file).stem
    work_dirs = Path("work_dirs") / onnx_name
    work_dirs.mkdir(exist_ok=True, parents=True)
    target_device = DeviceType.XH2a
    out_hmonnx_file = work_dirs / "hmonnx" / f"{onnx_name}_{target_device}.onnx"
    out_hmonnx_file.parent.mkdir(exist_ok=True, parents=True)
    out_hmonnx_file: str = str(out_hmonnx_file)

    onnx_model = onnx.load(onnx_file)
    inputs: List[Tensor] = []
    input_names: List[str] = []
    for input in onnx_model.graph.input:
        shape = [dim.dim_value if dim.dim_value > 0 else 1 for dim in input.type.tensor_type.shape.dim]
        dtype = TENSOR_TYPE_TO_TORCH_TYPE[input.type.tensor_type.elem_type]
        logger.info(f"inputs: {input.name}, {shape}, {dtype}")
        input_names.append(input.name)
        if dtype == torch.float32:
            inputs.append(torch.randn(shape, dtype=dtype))
        elif dtype in [torch.int32, torch.int64]:
            inputs.append(torch.randint(0, 10, shape, dtype=dtype))
        elif dtype == torch.bool:
            inputs.append(torch.randint(0, 2, shape, dtype=dtype))
        else:
            raise NotImplementedError(f"dtype {dtype} not supported")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    quant_scheme = QuantScheme(target_device=DeviceType.XH2a, quant_type="w8a8_sefp")

    quant_config = create_quant_config(quant_scheme)
    convert_onnx_to_hmonnx(onnx_file, inputs, DeviceType.XH2a, out_hmonnx_file, quant_config=quant_config)
    logger.info(f"Save to {out_hmonnx_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=str, default="resnet50_224x224.onnx")
    args = parser.parse_args()
    main(args)
