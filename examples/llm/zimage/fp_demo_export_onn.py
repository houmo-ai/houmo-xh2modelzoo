# Copyright 2025 HOUMO AI
#
# File: fp_demo_export_onn.py
# Description:
#   Example script: llm/zimage/fp_demo_export_onn.py
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

import onnx
import torch
from diffusers import ZImagePipeline
from xh_model_zoo.xh_llm.models.builder import wrap_llm_model
from xhquant.api import (
    ConfigDict,
    DeviceType,
    HMONNXGoldenInference,
    HMONNXInference,
    QuantScheme,
    convert_fx_model_to_quanted_model,
    convert_onnx_to_hmonnx,
    convert_quanted_model_to_hmonnx,
    create_quant_config,
)
from safetensors.torch import load_file, save_file
import torch.nn as nn

device = "cuda"


quant_scheme = QuantScheme(target_device=DeviceType.XH2a, quant_type="w4a8h1_ssfp")
quant_config = create_quant_config(quant_scheme)
quant_config = ConfigDict(quant_config)

# x = torch.load("examples/llm/zimage/x.pt").cuda()
# attn_mask = torch.load("examples/llm/zimage/attn_mask.pt").cuda()
# freqs_cis = torch.load("examples/llm/zimage/freqs_cis.pt").cuda()
# scale_msa = torch.load("examples/llm/zimage/scale_msa.pt").cuda()
# gate_msa = torch.load("examples/llm/zimage/gate_msa.pt").cuda()
# scale_mlp= torch.load("examples/llm/zimage/scale_mlp.pt").cuda()
# gate_mlp = torch.load("examples/llm/zimage/gate_mlp.pt").cuda()
# adaln_input = torch.load("examples/llm/zimage/adaln_input.pt").cuda()
# t_frep = torch.load("examples/llm/zimage/t_frep.pt").cuda()

onnx_model = onnx.load("/data02/datasets/zimage_model/shape_onnx/zimage_1_block.sim.onnx")

# import onnxsim
# model_simp, check = onnxsim.simplify(
#     "/data02/datasets/zimage_model/Z-Image-Turbo-transformer-1block_model.onnx",
# )
# assert check, "Simplified ONNX model could not be validated"
# onnx.save(model_simp, "/data02/datasets/zimage_model/shape_onnx/Z-Image-Turbo-transformer-1block_model.onnx")

from onnx import shape_inference
inferred_model = shape_inference.infer_shapes(onnx_model)
onnx.save(inferred_model, "/data02/datasets/zimage_model/shape_onnx/Z-Image-Turbo-transformer-1block_model.onnx",
           save_as_external_data=True, all_tensors_to_one_file=True)
# freqs_cis_expanded = freqs_cis[0].unsqueeze(2)
# f_real = freqs_cis_expanded.real  # 频率的实部，形状匹配x_real
# f_imag = freqs_cis_expanded.imag  # 频率的虚部，形状匹配x_imag

input_data = [
    torch.rand( (1024,64)).float(),
    torch.rand( (256,2560)).float(),
    torch.rand( (1,1024,128)).float(),
    torch.rand( (1,1024,128)).float(),
    torch.rand( (1,256,128)).float(),
    torch.rand( (1,256,128)).float(),
    torch.rand( (1,256)).float(),
    # scale_msa[0].float(),
    # gate_msa[0].float(),
    # scale_mlp[0].float(),
    # gate_mlp[0].float()
]

hm_onnx_file = "/data01/home/xuchen/xh2/xh2_model_zoo/work_dirs/output_zimage/f_block_honnx.onnx"


convert_onnx_to_hmonnx(onnx_model, input_data, DeviceType.XH2a, hm_onnx_file, quant_config)

model_name = "f_block_honnx"

if True:
    session = HMONNXGoldenInference(hm_onnx_file)
    session.to(device)
    session.save_golden = "work_dirs/output_zimage"
    session.golden_dir = "work_dirs" + f"/hmonnx/golden_{model_name}"
    session.step = 0
else:
    session = HMONNXInference(hm_onnx_file)
    session.to(device)

input_data = [ data.half() for data in input_data]

out = session(*input_data)

# [Optional] Attention Backend
# Diffusers uses SDPA by default. Switch to Flash Attention for better efficiency if supported:
# pipe.transformer.set_attention_backend("flash")    # Enable Flash-Attention-2
# pipe.transformer.set_attention_backend("_flash_3") # Enable Flash-Attention-3
# [Optional] Model Compilation
# Compiling the DiT model accelerates inference, but the first run will take longer to compile.
# pipe.transformer.compile()
# [Optional] CPU Offloading
# Enable CPU offloading for memory-constrained devices.
# pipe.enable_model_cpu_offload()


# prompt = "Young Chinese woman in red Hanfu, intricate embroidery. Impeccable makeup, red floral forehead pattern. Elaborate high bun, golden phoenix headdress, red flowers, beads. Holds round folding fan with lady, trees, bird. Neon lightning-bolt lamp (⚡️), bright yellow glow, above extended left palm. Soft-lit outdoor night background, silhouetted tiered pagoda (西安大雁塔), blurred colorful distant lights."
# # 2. Generate Image
# image = pipe(
#     prompt=prompt,
#     height=1024,
#     width=1024,
#     num_inference_steps=9,  # This actually results in 8 DiT forwards
#     guidance_scale=0.0,     # Guidance should be 0 for the Turbo models
#     generator=torch.Generator("cuda").manual_seed(42),
# ).images[0]
# image.save("example.png")