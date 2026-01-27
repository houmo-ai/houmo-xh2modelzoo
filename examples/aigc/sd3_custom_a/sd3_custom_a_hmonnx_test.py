# Copyright 2025 HOUMO AI
#
# File: sd3_custom_a_hmonnx_test.py
# Description:
#   Example script: aigc/sd3_custom_a/sd3_custom_a_hmonnx_test.py
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

import argparse
from pathlib import Path
from typing import List

import torch
from PIL import ImageDraw
from xhquant.api import get_root_logger, set_random_seed, xhquant_init

from xh_model_zoo.utils.time_profiler import TimeProfiler
from xh_model_zoo.xh_aigc.models.sd3_custom_a import SD3CustomAHFCompatible, SD3CustomAInference


def main(args):
    xhquant_init(None, debug=False)

    logger = get_root_logger()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    guidance_scale = 2.5
    steps = args.steps
    height = 512
    width = 512
    set_random_seed(args.seed, deterministic=False)
    pipe = SD3CustomAHFCompatible.to_hf_compatible(args.hf_model, custom_a_model=args.custom_a_model)
    # pipe.to(torch.float16)  # type: ignore # noqa: F401
    pipe.to(device)  # type: ignore # noqa: F401

    prompts: List[str] = [
        "A beautiful woman on the beach",
        "A beautiful woman on the street",
        "Waterfall on a high mountain, oil painting",
        "A man is runing on the grass",
        "A cat holding a sign that says hello world",
        "a portrait of young girl.",
        "A cat holding a sign that says haha",
        "a panda eating bamboo,Photo - realistic, 8k.",
    ]
    for idx, prompt in enumerate(prompts):
        logger.info(f"prompt {idx}: {prompt}")
        set_random_seed(args.seed, deterministic=False)
        generator = torch.Generator(device="cuda").manual_seed(args.seed)
        images = pipe(
            prompt,
            negative_prompt="",
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            generator=generator,
        ).images  # type: ignore # noqa: F401
        image = images[0]

        image_draw = ImageDraw.Draw(image, "RGB")
        image_draw.text((10, 10), prompt, fill=(255, 0, 0))

        filename = args.output
        Path(filename).parent.mkdir(exist_ok=True, parents=True)
        filename = Path(filename).with_name(f"{Path(filename).stem}_{idx}{Path(filename).suffix}")
        image.save(filename)
        logger.info(f"Image save to {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="work_dirs/sd3_2b_custom_a_XH2a_512x512/meta.json",
    )
    parser.add_argument("--hf-model", type=str, default="data/models/stable-diffusion-3-medium-diffusers")
    parser.add_argument("--custom-a-model", type=str, default="data/models/sd3_2b_custom_a")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--output", type=str, default="work_dirs/sd3_2b_custom_a_XH2a_512x512/tests/sd3-test.png")
    args = parser.parse_args()
    main(args)
