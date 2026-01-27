# Copyright 2025 HOUMO AI
#
# File: sd3_5_export.py
# Description:
#   Example script: aigc/sd3_5/sd3_5_export.py
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
import json
import os
from pathlib import Path

from xhquant.api import DeviceType, QuantScheme, xhquant_init
from xhquant.utils import set_random_seed

from xh_model_zoo.xh_aigc.models.sd3_5 import SD3_5_Converter, SD3ConvertConfig


def main(args):
    model = args.model
    model_dir = os.path.normpath(model)
    model_name = Path(model_dir).name
    target_device = DeviceType.XH2a
    height = args.height
    width = args.width
    cfg_name = f"{model_name}_{target_device.name}_{width}x{height}"
    work_dir = Path("work_dirs") / cfg_name
    work_dir.mkdir(exist_ok=True, parents=True)
    log_file = work_dir / f"{cfg_name}.log"
    xhquant_init(log_file, debug=False)

    target_device = DeviceType.XH2a
    quant_type = "w8a8_sefp"
    quant_scheme = QuantScheme(target_device=DeviceType.XH2a, quant_type=quant_type)
    convert_config = SD3ConvertConfig(
        quant_scheme=quant_scheme,
        guidance_scale=args.guidance_scale,
        height=height,
        width=width,
    )

    SD3_5_Converter.from_pretrained(model_dir, convert_config, str(work_dir))


def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--model",
        type=str,
        default="data/models/stable-diffusion-3.5-large-turbo",
    )
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=0.0,
        help="Seed for the random number generator",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arguments()
    set_random_seed(args.seed, deterministic=False)
    main(args)
