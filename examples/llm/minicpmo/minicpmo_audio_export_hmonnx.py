# Copyright 2025 HOUMO AI
#
# File: minicpmo_audio_export_hmonnx.py
# Description:
#   Example script: llm/minicpmo/minicpmo_audio_export_hmonnx.py
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
import os.path as osp
from pathlib import Path

from xh_model_zoo.xh_llm import LLMConverter
from xh_model_zoo.xh_llm.models.minicpmo.minicpmo_audio_convert_config import MinicpmoAudioConvertConfig

from xhquant.api import DeviceType, xhquant_init, QuantScheme, get_root_logger  # isort:skip
from xh_model_zoo.utils.memory_tracker import MemoryTracker  # isort:skip
from xh_model_zoo.utils.time_profiler import TimeProfiler  # isort:skip


def main(args):
    hf_model_path = osp.normpath(osp.abspath(args.model))
    model_name = Path(hf_model_path).name
    target_device = DeviceType.XH2a
    quant_type = args.quant_type
    quant_scheme = QuantScheme(target_device=DeviceType.XH2a, quant_type=quant_type)
    # quant_scheme.nodes["lm_head"] = "w8a8h1_sefp"
    config = MinicpmoAudioConvertConfig(
        quant_scheme=quant_scheme,
        video=args.video,
        audio=args.audio,
        debug=args.debug,
        valid=args.valid,
        image_slice_max_size=[40,40],
    )

    prefix = f"{model_name}-{target_device}-audio-{quant_type}"
    work_dir = Path("work_dirs") / prefix
    work_dir.mkdir(exist_ok=True, parents=True)
    log_file = work_dir / "convert.log"
    xhquant_init(log_file, debug=args.debug)
    logger = get_root_logger()
    with TimeProfiler("convert", logger), MemoryTracker("cuda:0", "convert", logger):
        LLMConverter.from_pretrained(hf_model_path, "MiniCPMWhisperEncoder", config, str(work_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="data/datasets/MiniCPM-o-2_6", type=str, help="HuggingFace model path")
    parser.add_argument("--video", type=str, default="examples/llm/minicpmo/assets/Skiing.mp4")
    parser.add_argument("--audio", type=str, default="examples/llm/minicpmo/assets/demo.wav")
    parser.add_argument("--debug", type=bool,default=False, help="debug mode")
    parser.add_argument("--valid", type=bool,default=False, help="check hmonnx mode")
    parser.add_argument("--quant-type", default="w8a8h0_sefp", help="quant type, default is w8a8")
    args = parser.parse_args()
    main(args)

