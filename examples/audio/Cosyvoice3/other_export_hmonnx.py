# -*- coding: utf-8 -*-
# Copyright 2025 HOUMO AI
#
# File: other_export_hmonnx.py
# Description:
#   CosyVoice3 misc-module HMONNX export script (HOUMO export pipeline).
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

OUTPUT_PATH = "/data01/home/she.gao/xh2modelzoo/examples/audio/Cosyvoice3/hmquant_xh2_fun_cosyvoice3_0.5B_2512_w8a8_20260320"

os.makedirs(OUTPUT_PATH, exist_ok=True)


def build_quant_config():
    quant_type = "w8a16h1_sefp"
    quant_scheme = QuantScheme(
        target_device=DeviceType.XH2a,
        quant_type=quant_type
    )
    return create_quant_config(quant_scheme)

def run_convert(model_path, dummy_inputs, output_name):
    output_file = osp.join(OUTPUT_PATH, output_name)
    os.makedirs(osp.dirname(output_file), exist_ok=True)
    golden_dir = osp.join(osp.dirname(output_file), "step_0")
    os.makedirs(golden_dir, exist_ok=True)

    quant_config = build_quant_config()

    # convert（存在判断）
    if not osp.exists(output_file):
        convert_onnx_to_hmonnx(
            model_path,
            dummy_inputs,
            out_hmonnx_file=output_file,
            device_type="XH2A",
            quant_config=quant_config
        )

    # golden
    model = HMONNXGoldenInference(output_file)
    model.save_golden = True
    model.exec_device = torch.device("cuda:0")
    model.golden_dir = str(golden_dir)

    fp16_inputs = tuple(x.to(torch.float16) for x in dummy_inputs)

    with torch.no_grad():
        model.forward(*fp16_inputs)

def main():
    # llm_decoder
    model_path = "/data01/home/she.gao/xh2modelzoo/examples/audio/Cosyvoice3/onnx/llm_decoder.onnx"
    input = torch.randn(1, 896)
    run_convert(model_path, (input,), "llm_decoder/prefill/hmquant_xh2_llm_decoder_w8a16_896_20260320.onnx")

    # spk_embed_affine_layer
    model_path_spk = "/data01/home/she.gao/xh2modelzoo/examples/audio/Cosyvoice3/onnx/spk_embed_affine_layer.onnx"
    input = torch.randn(1, 192)
    run_convert(model_path_spk, (input,), "spk_embed_affine_layer/prefill/hmquant_xh2_spk_embed_affine_layer_w8a16_192_20260320.onnx")

    # pre_lookahead_layer
    model_path_pre = "/data01/home/she.gao/xh2modelzoo/examples/audio/Cosyvoice3/onnx/pre_lookahead_layer.onnx"
    input = torch.randn(1, 1024, 80)
    run_convert(model_path_pre, (input,), "pre_lookahead_layer/prefill/hmquant_xh2_pre_lookahead_layer_w8a16_1024_20260320.onnx")

if __name__ == "__main__":
    main()