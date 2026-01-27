# Copyright 2025 HOUMO AI
#
# File: vae_export.py
# Description:
#   Example script: llm/qwen_image/vae_export.py
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

from diffusers import DiffusionPipeline
import torch
from xh_model_zoo.xh_llm.models.qwen_image.vae_converter import VAE_ConverterXH2a
from xh_model_zoo.xh_llm.models.qwen_image.pipeline_cus import cus_QwenImagePipeline
from pathlib import Path
from xh_model_zoo.xh_llm.models.qwen2_5_vl import Qwen2_5_VLConvertConfig, VisualConfig
import argparse
from xhquant.api import DeviceType, xhquant_init, QuantScheme, get_root_logger 

def main(args):
    model_name = "/data02/datasets/qwen-image"

    target_device = DeviceType.XH2a
    quant_type = args.quant_type

    ops=dict(MatMul=dict(
                act_scheme=dict(
                    bits=8,
                    fp_mode="sefp",
                ),
                act_schema_2=dict(
                    bits=16,
                    fp_mode="sefp",
                ),))

    quant_scheme = QuantScheme(target_device=DeviceType.XH2a, quant_type=quant_type, ops=ops)


    config = Qwen2_5_VLConvertConfig(
        batch_size=args.batch_size,
        quant_scheme=quant_scheme,
        quant_weight=args.quant_weight,
        gptqmodel_cfg=args.use_gptqmodel,
    )

    # Load the pipeline
    if torch.cuda.is_available():
        torch_dtype = torch.bfloat16
        device = "cuda"
    else:
        torch_dtype = torch.float32
        device = "cpu"

    pipe = DiffusionPipeline.from_pretrained(model_name, torch_dtype=torch_dtype, device_map="cuda")
    # pipe = pipe.to(device)

    positive_magic = {
        "en": ", Ultra HD, 4K, cinematic composition.", # for english prompt
        "zh": ", 超清，4K，电影级构图." # for chinese prompt
    }

    # Generate image
    prompt = '''A coffee shop entrance features a chalkboard sign reading "Qwen Coffee 😊 $2 per cup," with a neon light beside it displaying "通义千问". Next to it hangs a poster showing a beautiful Chinese woman, and beneath the poster is written "π≈3.1415926-53589793-23846264-33832795-02384197". Ultra HD, 4K, cinematic composition'''

    negative_prompt = " " # using an empty string if you do not have specific concept to remove


    # Generate with different aspect ratios
    aspect_ratios = {
        "1:1": (1328, 1328),
        "16:9": (1664, 928),
        "9:16": (928, 1664),
        "4:3": (1472, 1140),
        "3:4": (1140, 1472),
        "3:2": (1584, 1056),
        "2:3": (1056, 1584),
    }

    width, height = aspect_ratios["16:9"]

    work_dir = Path("work_dirs") / "qwen-image"
    work_dir.mkdir(exist_ok=True, parents=True)

    pipe.text_encoder = None
    pipe.transformer = None
    del pipe.text_encoder
    del pipe.transformer
    torch.cuda.empty_cache()

    VAE_ConverterXH2a(config)._convert(pipe.vae.half(), work_dir, pipe.image_processor.postprocess)

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
