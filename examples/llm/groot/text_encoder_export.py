# Copyright 2025 HOUMO AI
#
# File: text_encoder_export.py
# Description:
#   Example script: llm/zimage/text_encoder_export.py
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

from diffusers import ZImagePipeline
import torch
from xh_model_zoo.xh_llm.models.groot.qwen3_convert_config import Qwen3LegacyConvertConfig
# from xh_model_zoo.xh_llm.models.qwen_image.qwen2_5_vl_converter import Qwen2_5_VLConverterXH2a
# from xh_model_zoo.xh_llm.models.qwen_image.pipeline_cus import cus_QwenImagePipeline
from pathlib import Path
from xh_model_zoo.xh_llm.models.groot.qwen3_converter import Qwen3LegacyConverterXH2a
import argparse
from xhquant.api import DeviceType, xhquant_init, QuantScheme, get_root_logger 

import torch
from pathlib import Path
import argparse
from xhquant.api import DeviceType, xhquant_init, QuantScheme, get_root_logger, HMONNXGoldenInference, Config
# from xh_model_zoo.xh_llm.models.groot.cus_groot import cus_GROOT
import numpy as np
from xh_model_zoo.xh_llm.models.groot.gr00t.policy.gr00t_policy import Gr00tPolicy
from xh_model_zoo.xh_llm.models.groot.gr00t.data.embodiment_tags import EmbodimentTag

def main(args):
    device = "cuda"

    target_device = DeviceType.XH2a
    quant_type = args.quant_type

    quant_scheme = QuantScheme(target_device=DeviceType.XH2a, quant_type=quant_type) # , ops=ops

    config = Qwen3LegacyConvertConfig(
        batch_size=1,
        context_length=args.context_length,
        input_sequence_length=args.input_sequence_length,
        quant_scheme=quant_scheme,
        quant_weight=args.quant_weight,
        # mix_search=None,
    )

    policy = Gr00tPolicy(
        model_path='/data02/datasets/GROOT-N1.6-3B',
        embodiment_tag=EmbodimentTag('gr1'),
        device='cuda',
    )

    obs = {
        'video': {
            'ego_view_bg_crop_pad_res256_freq20': np.random.randint(0, 255, (1, 1, 256, 256, 3), dtype=np.uint8),
        },
        'state': {
            'left_arm': np.random.rand(1, 1, 7).astype(np.float32),
            'right_arm': np.random.rand(1, 1, 7).astype(np.float32),
            'left_hand': np.random.rand(1, 1, 6).astype(np.float32),
            'right_hand': np.random.rand(1, 1, 6).astype(np.float32),
            'waist': np.random.rand(1, 1, 3).astype(np.float32),
        },
        'language': {
            'task': [['pick up the red apple']],
        },
    }

    work_dir = Path("work_dirs") / "groot"
    work_dir.mkdir(exist_ok=True, parents=True)

    Qwen3LegacyConverterXH2a(config)._convert(policy.model.backbone.model.language_model.model.half(), work_dir)

    # image = pipe(
    #     prompt=prompt + positive_magic["en"],
    #     negative_prompt=negative_prompt,
    #     width=width,
    #     height=height,
    #     num_inference_steps=50,
    #     true_cfg_scale=4.0,
    #     generator=torch.Generator(device="cuda").manual_seed(42)
    # ).images[0]

    # image.save("example.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument("--model", type=str, default="weights/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--batch-size", type=int, default=1, help="batch size")
    parser.add_argument("--context-length", type=int, default=2048, help="max sequence length")
    parser.add_argument("--input-sequence-length", type=int, default=256, help="input sequence length")
    parser.add_argument("--max_pe_length", type=int, default=32768, help="max pe length")
    parser.add_argument("--quant-type", default="w8a8h1_sefp", help="quant type, default is w8a8")
    parser.add_argument("--image_max_size_h", type=int, default=448, help="image max size height")
    parser.add_argument("--image_max_size_w", type=int, default=448, help="image max size width")
    parser.add_argument("--image_max_size_t", type=int, default=2, help="if image, temporal max size is 2, if video, temporal max size is fps")
    parser.add_argument("--patch_size", type=int, default=14, help="patch size")
    parser.add_argument("--temporal_patch_size", type=int, default=2, help="temporal patch size")
    parser.add_argument("--sample_image_path", type=str, default="data/images/qwen2_vl_demo.jpeg", help="sample image path for generate golden")
    parser.add_argument("--use_gptqmodel", action="store_true", help="use gptqmodel quanted model")
    parser.add_argument(
        "--quant_weight",
        type=str,
        default=None,
        help="quant weight path, for example: gptq or quarot, if empty, use w8a8",
    )
    args = parser.parse_args()
    main(args)
