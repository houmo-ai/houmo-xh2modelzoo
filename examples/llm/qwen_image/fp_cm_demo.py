# Copyright 2025 HOUMO AI
#
# File: fp_cm_demo.py
# Description:
#   Example script: llm/qwen_image/fp_cm_demo.py
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

from diffusers import DiffusionPipeline, FlowMatchEulerDiscreteScheduler
import torch 
import math


# From https://github.com/ModelTC/Qwen-Image-Lightning/blob/342260e8f5468d2f24d084ce04f55e101007118b/generate_with_diffusers.py#L82C9-L97C10
scheduler_config = {
    "base_image_seq_len": 256,
    "base_shift": math.log(3),  # We use shift=3 in distillation
    "invert_sigmas": False,
    "max_image_seq_len": 8192,
    "max_shift": math.log(3),  # We use shift=3 in distillation
    "num_train_timesteps": 1000,
    "shift": 1.0,
    "shift_terminal": None,  # set shift_terminal to None
    "stochastic_sampling": False,
    "time_shift_type": "exponential",
    "use_beta_sigmas": False,
    "use_dynamic_shifting": True,
    "use_exponential_sigmas": False,
    "use_karras_sigmas": False,
}
scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)
pipe = DiffusionPipeline.from_pretrained(
    "/data02/datasets/qwen-image", scheduler=scheduler, torch_dtype=torch.bfloat16
).to("cuda")

pipe.load_lora_weights(
    "/data02/datasets/qwen_image_fp8/Qwen-Image-fp8-e4m3fn-Lightning-4steps-V1.0-fp32.safetensors"
)

prompt = "a tiny astronaut hatching from an egg on the moon, Ultra HD, 4K, cinematic composition."
negative_prompt = " "

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
                    clipped_output.append(item)
                elif isinstance(item, torch.Tensor):
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
    prompt=prompt,
    negative_prompt=negative_prompt,
    width=1024,
    height=1024,
    num_inference_steps=4,
    true_cfg_scale=1.0,
    generator=torch.manual_seed(0),
).images[0]
image.save("qwen_fewsteps.png")