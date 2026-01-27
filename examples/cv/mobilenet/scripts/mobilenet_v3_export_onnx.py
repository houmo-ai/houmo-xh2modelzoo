# Copyright 2025 HOUMO AI
#
# File: mobilenet_v3_export_onnx.py
# Description:
#   Example script: cv/mobilenet/scripts/mobilenet_v3_export_onnx.py
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

from pathlib import Path

import onnx
import onnxsim
import torch
import torchvision

if __name__ == "__main__":
    model = torchvision.models.mobilenet_v3_large()
    model.eval()
    input = torch.randn(1, 3, 224, 224)
    out_onnx_file = "data/models/mobilenet/mobilenet_v3_large.onnx"
    Path(out_onnx_file).parent.mkdir(exist_ok=True, parents=True)
    torch.onnx.export(model, input, out_onnx_file, input_names=["images"], output_names=["cls_score"])
    onnx_model = onnx.load(out_onnx_file)
    onnx_model, check = onnxsim.simplify(onnx_model)
    if check:
        onnx.save(onnx_model, out_onnx_file)
    print("Export onnx success, out onnx file to: ", out_onnx_file)
