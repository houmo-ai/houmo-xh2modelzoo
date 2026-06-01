# Copyright 2025 HOUMO AI
#
# File: efficientnet_v2_export_onnx.py
# Description:
#   Example script: cv/efficientnet/scripts/efficientnet_v2_export_onnx.py
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
    weights = torchvision.models.EfficientNet_V2_M_Weights.DEFAULT
    model = torchvision.models.efficientnet_v2_m(weights=weights)
    model.eval()

    input_tensor = torch.randn(1, 3, 480, 480)

    out_onnx_file = "data/models/efficientnet/efficientnet_v2_m.onnx"
    Path(out_onnx_file).parent.mkdir(exist_ok=True, parents=True)

    print(f"Exporting model with pre-trained weights to {out_onnx_file}...")
    torch.onnx.export(
        model, input_tensor, out_onnx_file, input_names=["images"], output_names=["cls_score"], opset_version=11
    )

    print("Simplifying model...")
    onnx_model = onnx.load(out_onnx_file)
    onnx_model, check = onnxsim.simplify(onnx_model)
    if check:
        onnx.save(onnx_model, out_onnx_file)
        print("Export and simplification success, out onnx file to: ", out_onnx_file)
    else:
        print("Simplification failed.")
