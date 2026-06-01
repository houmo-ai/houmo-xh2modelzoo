# Copyright 2025 HOUMO AI
#
# File: fp_demo.py
# Description:
#   Example script: llm/zimage/fp_demo.py
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
from diffusers import ZImagePipeline
from safetensors.torch import load_file, save_file

# 1. Load the pipeline
# Use bfloat16 for optimal performance on supported GPUs
pipe = ZImagePipeline.from_pretrained(
    "/data02/datasets/zimage",
    low_cpu_mem_usage=False,
)
pipe.to("cuda")

# weights = load_file("/data02/datasets/zimage/diffusion_pytorch_model.safetensors")
# model_dict = pipe.transformer.state_dict()

# load_keys = [k for k in weights.keys() if k in model_dict]
# for k in load_keys:
#     model_dict[k] = weights[k]




# 加载到模型
# pipe.transformer.load_state_dict(model_dict, strict=True)

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
prompt = "一个坐在云朵上唱歌的女生，背后一顿五彩的翅膀"
# 2. Generate Image
image = pipe(
    prompt=prompt,
    height=1024,
    width=1024,
    num_inference_steps=9,  # This actually results in 8 DiT forwards
    guidance_scale=0.0,     # Guidance should be 0 for the Turbo models
    generator=torch.Generator("cuda").manual_seed(47),
).images[0]
image.save("example7.png")