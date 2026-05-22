# Copyright 2025 HOUMO AI
#
# File: vae_export.py
# Description:
#   Example script: llm/zimage/vae_export.py
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

import contextlib

import transformers.modeling_utils as modeling_utils
from diffusers import ZImagePipeline
import torch

if not hasattr(modeling_utils, "no_init_weights"):
    @contextlib.contextmanager
    def no_init_weights(_enable=True):
        yield

    modeling_utils.no_init_weights = no_init_weights

from xh_model_zoo.xh_llm.models.qwen2_legacy import Qwen2LegacyConvertConfig
from pathlib import Path
from xh_model_zoo.xh_llm.models.zimage.vae_converter import VAE_ConverterXH2a
import argparse
from xhquant.api import DeviceType, xhquant_init, QuantScheme, get_root_logger 

def main(args):
    model_name = args.model
    device = "cuda"

    target_device = DeviceType.XH2a
    quant_type = args.quant_type

    quant_scheme = QuantScheme(target_device=DeviceType.XH2a, quant_type=quant_type) # , ops=ops

    config = Qwen2LegacyConvertConfig(
        batch_size=1,
        context_length=args.context_length,
        input_sequence_length=args.input_sequence_length,
        quant_scheme=quant_scheme,
        quant_weight=args.quant_weight,
        # mix_search=None,
    )

    pipe = ZImagePipeline.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=False,
    )    
    pipe = pipe.to(device)


    # Generate image
    prompt = "Young Chinese woman in red Hanfu, intricate embroidery. Impeccable makeup, red floral forehead pattern. Elaborate high bun, golden phoenix headdress, red flowers, beads. Holds round folding fan with lady, trees, bird. Neon lightning-bolt lamp (⚡️), bright yellow glow, above extended left palm. Soft-lit outdoor night background, silhouetted tiered pagoda (西安大雁塔), blurred colorful distant lights."
    negative_prompt = " " # using an empty string if you do not have specific concept to remove

    work_dir = Path(args.work_dir)
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
    parser.add_argument("--model", type=str, default="/data02/datasets/zimage")
    parser.add_argument("--work-dir", type=str, default="work_dirs/zimage")
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
