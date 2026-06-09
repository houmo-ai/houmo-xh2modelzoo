# Copyright 2025 HOUMO AI
#
# File: flux2_klein_edit_export.py
# Description:
#   Example script: export FLUX.2-klein HMONNX components for image-edit mode.
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
import os
from pathlib import Path

from xh_model_zoo.xh_aigc.models.flux2_klein import Flux2KleinConvertConfig, Flux2KleinConverter
from xhquant.api import DeviceType, QuantScheme, xhquant_init
from xhquant.utils import set_random_seed


def main(args):
    model_dir = os.path.normpath(args.model)
    model_name = Path(model_dir).name
    target_device = DeviceType.XH2a
    cfg_name = f"{model_name}_{target_device.name}_{args.width}x{args.height}_edit"
    work_dir = Path(args.work_dir) if args.work_dir else Path("work_dirs") / cfg_name
    work_dir.mkdir(exist_ok=True, parents=True)
    log_file = work_dir / f"{cfg_name}.log"
    xhquant_init(log_file, debug=False)

    quant_scheme = QuantScheme(target_device=target_device, quant_type=args.quant_type)
    convert_config = Flux2KleinConvertConfig(
        quant_scheme=quant_scheme,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.steps,
        width=args.width,
        height=args.height,
        prompt=args.prompt,
        seed=args.seed,
        max_sequence_length=args.max_sequence_length,
        input_sequence_length=args.input_sequence_length,
        export_components=tuple(args.components),
        image_edit=True,
    )

    Flux2KleinConverter.from_pretrained(model_dir, convert_config, str(work_dir))


def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", type=str, default="/data02/datasets/flux-4b")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument("--input-sequence-length", type=int, default=256)
    parser.add_argument("--quant-type", type=str, default="w8a8_sefp")
    parser.add_argument("--prompt", type=str, default="make the image more cinematic and detailed")
    parser.add_argument("--work-dir", type=str, default=None, help="导出目录；默认 work_dirs/{model}_{device}_{WxH}_edit")
    parser.add_argument(
        "--components",
        nargs="+",
        default=["vae_encoder"],
        help="export selected Flux image-edit components",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    set_random_seed(args.seed, deterministic=False)
    main(args)
