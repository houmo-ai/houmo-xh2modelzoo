# Copyright 2025 HOUMO AI
#
# File: dummy_export_golden.py
# Description:
#   Example script: llm/mit/dummy_export_golden.py
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

import tempfile
from copy import deepcopy
from pathlib import Path

import onnx
import torch
import torch.nn as nn
from torch import Tensor

from xhquant.api import (
    DeviceType,
    HMONNXGoldenInference,
    HMONNXInference,
    QuantScheme,
    convert_onnx_to_hmonnx,
    create_quant_config,
    convert_fx_model_to_quanted_model,
    convert_quanted_model_to_hmonnx,
    ConfigDict
)
from xhquant.export.onnx.onnx_opset.schema.xh2a_schema import RESIZE_OP_SCHEMA
from xhquant.api import DeviceType
from xhquant.nn.modules.resizer import OnnxResize 
from xhquant.nn.modules.onnx_style_modules import DynamicImageResize
import cv2
from xhquant.api.ptq_export_hmonnx import normalized_onnx, _convert_model_to_quanted_model, FrontendType

input_shapes = {
    "x": [1, 256, 560],
    "language": [1],
    "text_norm": [1],
    # "x_length": [1],
}


def export_hmonnx_model(
    work_dir: str, save_golden: bool = True
):
    device = "cpu" # "cuda" if torch.cuda.is_available() else 

    onnx_file = "/data01/home/xuchen/xh2/xh2_model_zoo/examples/llm/mit/fp/model_all_del_final.sim_final_sp.onnx"

    model_name = "ASR"
    hm_onnx_file = "work_dirs/MIT" + model_name
    x = torch.randn(input_shapes["x"]).to(device)
    language = torch.randint(0, 1, input_shapes["language"]).to(device)
    text_norm = torch.randint(0, 1, input_shapes["text_norm"]).to(device)
    # x_length = torch.randint(0, 1, input_shapes["text_norm"]).to(device)


    onnx_model = onnx.load(onnx_file)
    convert_onnx_to_hmonnx(onnx_model, [x, language, text_norm], DeviceType.XH2a, hm_onnx_file)

    if save_golden:
        session = HMONNXGoldenInference(hm_onnx_file)
        session.to(device)
        session.save_golden = save_golden
        session.golden_dir = work_dir + f"/hmonnx/golden_{model_name}"
        session.step = 0
    else:
        session = HMONNXInference(hm_onnx_file)
        session.to(device)
    out = session(x.half(), language.to(torch.int32), text_norm.to(torch.int32))

if __name__ == "__main__":
    export_hmonnx_model("work_dir", True)
    # _test_resize_trace(True, "work_dir", 11, DeviceType.XH2a, True)
