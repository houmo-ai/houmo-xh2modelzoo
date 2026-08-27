# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: E402, I001

# pyright: reportMissingImports=false

import argparse
import sys
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", type=str, default="/data01/home/xuchen/Wan2.2-main/ckpt/Wan_2.2")
    parser.add_argument("--wan-repo", type=str, default="/data01/home/xuchen/Wan2.2-main")
    parser.add_argument("--task", type=str, default="i2v-A14B")
    parser.add_argument("--prompt", type=str, default="A calm seaside scene with gentle waves.")
    parser.add_argument("--negative-prompt", type=str, default="")
    parser.add_argument("--steps", type=int, default=None, help="Number of sampling steps used for generation.")
    parser.add_argument(
        "--use-lora-models",
        action="store_true",
        help="Load the resolved LoRA-compatible model files instead of the original Wan pipeline.",
    )
    parser.add_argument("--size", nargs=2, type=int, default=[480, 832], metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--frame-num", type=int, default=81)
    parser.add_argument("--image", type=str, default="", help="Input image required by i2v tasks.")
    parser.add_argument("--output", type=str, default="outputs/wan2_2_fp_demo.mp4")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=16)
    return parser.parse_args()


def main(args):
    wan_repo = Path(args.wan_repo).expanduser().resolve()
    if not wan_repo.is_dir():
        raise FileNotFoundError(f"Wan2.2 source directory does not exist: {wan_repo}")
    if str(wan_repo) not in sys.path:
        sys.path.insert(0, str(wan_repo))

    from xh_model_zoo.xh_aigc.models.wan2_2.common import ensure_wan2_2_repo
    from xh_model_zoo.xh_aigc.models.wan2_2.wan2_2_converter import Wan2_2ConvertConfig, Wan2_2Converter

    ensure_wan2_2_repo(wan_repo)
    from wan.configs import WAN_CONFIGS
    from wan.utils.utils import save_video as wan_save_video

    cfg = WAN_CONFIGS[args.task]
    sample_steps = args.steps if args.steps is not None else cfg.sample_steps
    converter = Wan2_2Converter(
        args.model,
        Wan2_2ConvertConfig(
            task=args.task,
            sample_steps=sample_steps,
            use_resolved_float_loader=args.use_lora_models,
        ),
    )
    pipe = converter.build_float_generation_pipeline()
    negative_prompt = args.negative_prompt if args.negative_prompt else pipe.sample_neg_prompt

    if args.task.startswith("i2v"):
        if not args.image:
            raise ValueError("--image is required for i2v tasks")
        video = pipe.generate(
            args.prompt,
            Image.open(args.image).convert("RGB"),
            max_area=args.size[0] * args.size[1],
            frame_num=args.frame_num,
            shift=cfg.sample_shift,
            sampling_steps=sample_steps,
            guide_scale=cfg.sample_guide_scale,
            n_prompt=negative_prompt,
            seed=args.seed,
            offload_model=True,
        )
    else:
        video = pipe.generate(
            args.prompt,
            size=tuple(args.size),
            frame_num=args.frame_num,
            shift=cfg.sample_shift,
            sampling_steps=sample_steps,
            guide_scale=cfg.sample_guide_scale,
            n_prompt=negative_prompt,
            seed=args.seed,
            offload_model=True,
        )

    if video.ndim == 4:
        video = video.unsqueeze(0)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    wan_save_video(video, save_file=str(output_path), fps=args.fps, nrow=1)

    print(f"Built float Wan2.2 pipeline: {type(pipe).__name__}")
    print(f"Prompt: {args.prompt}")
    print(f"Use LoRA models: {args.use_lora_models}")
    print(f"Sampling steps: {sample_steps}")
    print(f"Saved video to: {output_path}")


if __name__ == "__main__":
    main(parse_args())
