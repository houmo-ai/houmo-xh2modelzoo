# -*- coding: utf-8 -*-
# Copyright 2025 HOUMO AI
#
# File: campplus_export_hmonnx.py
# Description:
#   CosyVoice3 CAM++ HMONNX export script (HOUMO export pipeline).
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
import numpy as np
import torch
import onnx
import onnxruntime as ort
import onnxsim

from xhquant.api import (
   convert_onnx_to_hmonnx,
   QuantScheme,
   create_quant_config,
   DeviceType,
   HMONNXGoldenInference,
)

# ======================
# 路径配置
# ======================
MODEL_PATH = "/data01/home/she.gao/xh2modelzoo/examples/audio/Cosyvoice3/onnx/campplus.onnx"
MODEL_SIMPLIFIED_PATH = "/data01/home/she.gao/xh2modelzoo/examples/audio/Cosyvoice3/onnx/campplus_simplify.onnx"
OUTPUT_DIR = "/data01/home/she.gao/xh2modelzoo/examples/audio/Cosyvoice3/hmquant_xh2_fun_cosyvoice3_0.5B_2512_w8a8_20260320/campplus/prefill"
GOLDEN_DIR = "/data01/home/she.gao/xh2modelzoo/examples/audio/Cosyvoice3/hmquant_xh2_fun_cosyvoice3_0.5B_2512_w8a8_20260320/campplus/prefill/step_0"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(GOLDEN_DIR, exist_ok=True)

# ======================
# 固定 shape
# ======================
FIXED_DIMS = {
   "batch_size": 1,
   "sequence_length": 1000,
}

# ======================
# 打印 ONNX IO 信息
# ======================
def inspect_onnx(model_path):
   session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

   print("模型输入节点:")
   for inp in session.get_inputs():
      print(f"  {inp.name}: {inp.type}, shape={inp.shape}")

   print("模型输出节点:")
   for out in session.get_outputs():
      print(f"  {out.name}: {out.type}, shape={out.shape}")


# ======================
# 固定输入 shape
# ======================
def fix_input_shape(model, fixed_dims):
   for input_node in model.graph.input:
      dims = []
      for d in input_node.type.tensor_type.shape.dim:
         if d.dim_value > 0:
               dims.append(d.dim_value)
         else:
               dims.append(d.dim_param)

      new_dims = []
      for dim in dims:
         if dim in fixed_dims:
               new_dims.append(fixed_dims[dim])
         elif isinstance(dim, int):
               new_dims.append(dim)
         else:
               raise ValueError(f"未处理的动态维度: {dim}")

      input_node.type.tensor_type.shape.dim.clear()
      for val in new_dims:
         input_node.type.tensor_type.shape.dim.add(dim_value=val)

   return model


# ======================
# 主流程
# ======================
def main():
   os.makedirs(OUTPUT_DIR, exist_ok=True)

   # 1. 查看模型 IO
   inspect_onnx(MODEL_PATH)

   # 2. 加载并固定 shape
   model = onnx.load(MODEL_PATH)
   model = fix_input_shape(model, FIXED_DIMS)

   # 3. simplify
   model_simplified, _ = onnxsim.simplify(model)
   onnx.save(model_simplified, MODEL_SIMPLIFIED_PATH)

   # 4. 构造输入
   dummy_input = torch.randn(1, 1000, 80)

   # 5. 量化配置
   quant_type = "w8a16_sefp"
   quant_scheme = QuantScheme(
      target_device=DeviceType.XH2a,
      quant_type=quant_type,
   )
   quant_config = create_quant_config(quant_scheme)

   # 6. 转换 hmonnx
   prefix = f"hmquant_xh2_campplus_w8a16_1000_20260320"
   output_file = osp.join(OUTPUT_DIR, f"{prefix}.onnx")

   if not osp.exists(output_file):
      convert_onnx_to_hmonnx(
         MODEL_SIMPLIFIED_PATH,
         (dummy_input,),
         out_hmonnx_file=output_file,
         device_type="XH2A",
         quant_config=quant_config,
      )
   
   model = HMONNXGoldenInference(output_file)
   model.save_golden = True
   model.exec_device = torch.device("cuda:0")

   dummy_input = dummy_input.to(torch.float16)
   input_args = (dummy_input,)
   model.golden_dir = str(GOLDEN_DIR)
   with torch.no_grad():
      model.forward(*input_args)


if __name__ == "__main__":
   main()