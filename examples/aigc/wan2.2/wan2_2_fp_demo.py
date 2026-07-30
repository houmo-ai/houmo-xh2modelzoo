# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0

# pyright: reportMissingImports=false

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xh_model_zoo.xh_aigc.models.wan2_2.common import ensure_wan2_2_repo
from xh_model_zoo.xh_aigc.models.wan2_2.pipeline_hmonnx import (
    WanI2VHMONNXPipeline,
    WanT2VHMONNXPipeline,
)

ensure_wan2_2_repo()
from wan.configs import WAN_CONFIGS  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", type=str, default="ckpt/wan2.2")
    parser.add_argument("--task", type=str, default="t2v-A14B")
    parser.add_argument("--prompt", type=str, default="A calm seaside scene with gentle waves.")
    return parser.parse_args()


def main(args):
    cfg = WAN_CONFIGS[args.task]
    pipeline_cls = WanI2VHMONNXPipeline if args.task.startswith("i2v") else WanT2VHMONNXPipeline
    _ = pipeline_cls(cfg, args.model)
    print(f"Built float Wan2.2 pipeline: {pipeline_cls.__name__}")
    print(f"Prompt: {args.prompt}")
    print("Use this script as the floating baseline before swapping modules with HMONNX.")


if __name__ == "__main__":
    main(parse_args())
