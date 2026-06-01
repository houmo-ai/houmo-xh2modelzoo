# Copyright 2025 HOUMO AI
#
# File: fp_demo.py
# Description:
#   Example script: llm/qwen_image/fp_demo.py
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

from diffusers import DiffusionPipeline
import torch

model_name = "/data02/datasets/qwen-image"

# Load the pipeline
if torch.cuda.is_available():
    torch_dtype = torch.float16
    device = "cuda"
else:
    torch_dtype = torch.float32
    device = "cpu"

pipe = DiffusionPipeline.from_pretrained(model_name, torch_dtype=torch_dtype, device_map="cuda")
# pipe = pipe.to(device)

positive_magic = {
    "en": ", Ultra HD, 4K, cinematic composition.", # for english prompt
    "zh": ", 超清，4K，电影级构图." # for chinese prompt
}

# Generate image
prompt = '''A coffee shop entrance features a chalkboard sign reading "Qwen Coffee 😊 $2 per cup," with a neon light beside it displaying "通义千问". Next to it hangs a poster showing a beautiful Chinese woman, and beneath the poster is written "π≈3.1415926-53589793-23846264-33832795-02384197". Ultra HD, 4K, cinematic composition'''

negative_prompt = " " # using an empty string if you do not have specific concept to remove


# Generate with different aspect ratios
aspect_ratios = {
    "1:1": (1328, 1328),
    "16:9": (1664, 928),
    "9:16": (928, 1664),
    "4:3": (1472, 1140),
    "3:4": (1140, 1472),
    "3:2": (1584, 1056),
    "2:3": (1056, 1584),
}

width, height = aspect_ratios["16:9"]


def auto_clip_hook(clip_value=65504):
    """生成自动裁剪的钩子函数"""
    def hook(module, input, output):
        # 处理单个张量或元组（部分层输出是元组，如RNN）
        if isinstance(output, torch.Tensor):
            clipped_output = torch.clamp(output, -clip_value, clip_value)
            return clipped_output
        elif isinstance(output, (tuple, list)):
            clipped_output = []
            for item in output:
                if item.dtype == torch.complex64:
                    continue
                if isinstance(item, torch.Tensor):
                    clipped_output.append(torch.clamp(item, -clip_value, clip_value))
                else:
                    clipped_output.append(item)
            return tuple(clipped_output)
        return output
    return hook

clip_threshold = 65504
for module in pipe.transformer.modules():
    module.register_forward_hook(auto_clip_hook(clip_threshold))

image = pipe(
    prompt=prompt + positive_magic["en"],
    negative_prompt=negative_prompt,
    width=width,
    height=height,
    num_inference_steps=50,
    true_cfg_scale=4.0,
    generator=torch.Generator(device="cuda").manual_seed(42)
).images[0]

image.save("example_fp16.png")
