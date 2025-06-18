import argparse
from pathlib import Path
from typing import List

import torch
from PIL import ImageDraw
from xhquant.api import get_root_logger, set_random_seed, xhquant_init

from xh_model_zoo.xh_aigc.models.sd3_lenovo import SD3LenovoHFCompatible


def main(args):
    xhquant_init(None, debug=False)

    logger = get_root_logger()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    guidance_scale = 2.5
    steps = 8
    height = 512
    width = 512
    # sd3_inference = SD3LenovoInference(args.config, fast_mode=args.fast)
    # sd3_inference.to(device)
    set_random_seed(args.seed, deterministic=False)
    pipe = SD3LenovoHFCompatible.to_hf_compatible(args.hf_model, lenovo_model=args.lenovo_model)
    # pipe.to(torch.float16)  # type: ignore # noqa: F401
    pipe.to(device)  # type: ignore # noqa: F401

    prompts = [
        "A beautiful woman on the beach",
        "A beautiful woman on the street",
        "Waterfall on a high mountain, oil painting",
        "A man is runing on the grass",
        "A cat holding a sign that says hello world",
        "a portrait of young girl.",
        "A cat holding a sign that says haha",
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
        default="work_dirs/stable-diffusion-3-medium-diffusers_XH2a_512x512/meta.json",
    )
    parser.add_argument("--hf-model", type=str, default="data/models/stable-diffusion-3-medium-diffusers")
    parser.add_argument("--lenovo-model", type=str, default="data/models/sd3_2b_lenovo")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--output", type=str, default="work_dirs/sd3_2b_lenovo_XH2a_512x512/tests/sd3-test.png")
    args = parser.parse_args()
    main(args)
