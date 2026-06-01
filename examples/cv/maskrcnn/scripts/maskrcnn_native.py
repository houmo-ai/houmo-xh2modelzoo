# Copyright 2025 HOUMO AI
#
# File: maskrcnn_native.py
# Description:
#   Example script: cv/maskrcnn/scripts/maskrcnn_native.py
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

import torch
from torchvision.models.detection import MaskRCNN, MaskRCNN_ResNet50_FPN_V2_Weights, maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

model = maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
model.eval()
input_tensor = torch.randn(1, 3, 256, 256)
onnx_file_path = "maskrcnn.onnx"

# Export the PyTorch model to ONNX format
torch.onnx.export(
    model.cpu(),
    input_tensor.cpu(),
    onnx_file_path,
    export_params=True,
    do_constant_folding=False,
    input_names=["input"],
    output_names=["boxes", "labels", "scores", "masks"],
)
