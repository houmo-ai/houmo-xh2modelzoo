# Copyright 2025 HOUMO AI
#
# File: sd3_5_hmonnx_test.py
# Description:
#   Example script: aigc/sd3_5/sd3_5_hmonnx_test.py
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
from xh_model_zoo.xh_aigc.models.sd3_5 import SD3HFCompatible, SD3Inference


def main(args):
    xhquant_init(None, debug=False)
    torch.set_grad_enabled(False)
    logger = get_root_logger()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sd3_inference = SD3Inference(args.config, fast_mode=args.fast)
    sd3_inference.to(device)
    set_random_seed(args.seed, deterministic=False)
    pipe = SD3HFCompatible.to_hf_compatible(args.hf_model)
    # pipe.to(torch.float16)  # type: ignore # noqa: F401
    pipe.to(device)  # type: ignore # noqa: F401
    generator = torch.Generator(device=device).manual_seed(args.seed)
    prompts: List[str] = [
        "A beautiful woman on the beach",
        "A beautiful woman on the street",
        "Waterfall on a high mountain, oil painting",
        "A man is runing on the grass",
        "A cat holding a sign that says hello world",
        "a portrait of young girl.",
        "A cat holding a sign that says haha",
    ]
    prompt = prompts[0]
    images = pipe(
        prompt,
        negative_prompt="",
        num_inference_steps=args.steps,
        guidance_scale=sd3_inference.guidance_scale,
        height=sd3_inference.height,
        width=sd3_inference.width,
        generator=generator,
    ).images  # type: ignore # noqa: F401
    image = images[0]

    image_draw = ImageDraw.Draw(image, "RGB")
    image_draw.text((10, 10), prompt, fill=(255, 0, 0))

    filename = args.output
    Path(filename).parent.mkdir(exist_ok=True, parents=True)
    image.save(filename)
    logger.info(f"Image save to {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="work_dirs/data/models/stable-diffusion-3.5-large-turbo_XH2a_512x512/meta.json",
    )
    parser.add_argument("--hf-model", type=str, default="data/models/data/models/stable-diffusion-3.5-large-turbo")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument(
        "--output",
        type=str,
        default="work_dirs/data/models/stable-diffusion-3.5-large-turbo_XH2a_512x512/tests/sd3-test.png",
    )
    args = parser.parse_args()
    main(args)
