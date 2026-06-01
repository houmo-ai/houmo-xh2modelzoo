# -*- coding: utf-8 -*-
# Copyright 2025 HOUMO AI
#
# File: speech_tokenizer_v3_export_hmonnx.py
# Description:
#   CosyVoice3 speech-tokenizer-v3 HMONNX export script (HOUMO export pipeline).
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

import os
import os.path as osp
import torch

from xhquant.api import (
    convert_onnx_to_hmonnx,
    QuantScheme,
    create_quant_config,
    DeviceType,
    HMONNXGoldenInference,
)

model_path = "/data01/home/she.gao/xh2modelzoo/examples/audio/Cosyvoice3/onnx/speech_tokenizer_v3_3000_3.onnx"
OUTPUT_DIR = "/data01/home/she.gao/xh2modelzoo/examples/audio/Cosyvoice3/hmquant_xh2_fun_cosyvoice3_0.5B_2512_w8a8_20260320/speech_tokenizer_v3/prefill"
GOLDEN_DIR = "/data01/home/she.gao/xh2modelzoo/examples/audio/Cosyvoice3/hmquant_xh2_fun_cosyvoice3_0.5B_2512_w8a8_20260320/speech_tokenizer_v3/prefill/step_0"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(GOLDEN_DIR, exist_ok=True)

def main():
    # dummy input
    input = torch.randn(1, 128, 3000)
    mask = torch.randn(1, 20, 750, 750)
    mask1 = torch.randn(1, 750, 1280)

    # quant config
    quant_type = "w8a16_sefp"
    quant_scheme = QuantScheme(
        target_device=DeviceType.XH2a,
        quant_type=quant_type
    )
    quant_config = create_quant_config(quant_scheme)

    # convert（加存在判断）
    prefix = f"hmquant_xh2_speech_tokenizer_v3_w8a16_3000_20260320"
    output_file = osp.join(OUTPUT_DIR, f"{prefix}.onnx")

    if not osp.exists(output_file):
        convert_onnx_to_hmonnx(
            model_path,
            (input, mask, mask1),
            out_hmonnx_file=output_file,
            device_type="XH2A",
            quant_config=quant_config
        )

    # golden
    model = HMONNXGoldenInference(output_file)
    model.save_golden = True
    model.exec_device = torch.device("cuda:0")

    input = input.to(torch.float16)
    mask = mask.to(torch.float16)
    mask1 = mask1.to(torch.float16)

    input_args = (input, mask, mask1)
    model.golden_dir = str(GOLDEN_DIR)

    with torch.no_grad():
        model.forward(*input_args)

if __name__ == "__main__":
    main()