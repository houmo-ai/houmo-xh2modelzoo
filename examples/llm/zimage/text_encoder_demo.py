# Copyright 2025 HOUMO AI
#
# File: text_encoder_demo.py
# Description:
#   Example script: llm/zimage/text_encoder_demo.py
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
from xh_model_zoo.xh_llm.models.qwen2_legacy import Qwen2LegacyConvertConfig
# from xh_model_zoo.xh_llm.models.qwen_image.qwen2_5_vl_converter import Qwen2_5_VLConverterXH2a
# from xh_model_zoo.xh_llm.models.qwen_image.pipeline_cus import cus_QwenImagePipeline
from pathlib import Path
from xh_model_zoo.xh_llm.models.zimage import Qwen3LegacyConverterXH2a, Qwen3LegacyInference
from xh_model_zoo.xh_llm.models.zimage.pipeline_cus import cus_ZImagePipeline
import argparse
from xhquant.api import DeviceType, xhquant_init, QuantScheme, get_root_logger, HMONNXGoldenInference, Config

def main(args):
    model_name = "/data02/datasets/zimage"
    device = "cuda"
    # target_device = DeviceType.XH2a
    # quant_type = args.quant_type

    # quant_scheme = QuantScheme(target_device=DeviceType.XH2a, quant_type=quant_type) # , ops=ops

    # config = Qwen2LegacyConvertConfig(
    #     batch_size=1,
    #     context_length=args.context_length,
    #     input_sequence_length=args.input_sequence_length,
    #     quant_scheme=quant_scheme,
    #     quant_weight=args.quant_weight,
    #     # mix_search=None,
    # )

    pipe = ZImagePipeline.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=False,
    )    
    pipe = pipe.to(device)

    text_encoder_prefill = "work_dirs/zimage/hmonnx/prefill/zimage_llm-XH2a-2k-w8a8h1_sefp_prefill.onnx"
    text_encoder = HMONNXGoldenInference(text_encoder_prefill)
    text_encoder.exec_device = torch.device("cuda:0")

    hm_vae_path = "work_dirs/zimage/hmonnx/zimage_vae-XH2a-w8a8h1_sefp.onnx"
    hm_vae = HMONNXGoldenInference(hm_vae_path)
    hm_vae.exec_device = torch.device("cuda:0")

    hm_dit_path = "work_dirs/zimage/hmonnx/zimage_dit-XH2a-w8a8h1_sefp.onnx"
    hm_dit = HMONNXGoldenInference(hm_dit_path)
    hm_dit = hm_dit.cuda()
    hm_dit.exec_device = torch.device("cuda:0")

    pipe.text_encoder = None
    # pipe.vae = None
    del pipe.text_encoder
    # del pipe.vae
    torch.cuda.empty_cache()

    # Generate image
    prompt = "一只长得像蝴蝶一样缤纷绚丽的奇异花朵，开在丛林中，散发着柔和的光芒"
    negative_prompt = " " # using an empty string if you do not have specific concept to remove

    work_dir = Path("work_dirs") / "zimage"
    work_dir.mkdir(exist_ok=True, parents=True)

    # Qwen3LegacyConverterXH2a(config)._convert(pipe.text_encoder.half(), work_dir)

    # ========= warp transformer =========
    # from xh_model_zoo.xh_llm.models.zimage._dit_model import register_wrap_cls as llm_register_wrap_cls
    # from xh_model_zoo.xh_llm.models.builder import wrap_llm_model
    # llm_register_wrap_cls(pipe.transformer)
    # wrap_cfg = Config(
    #     dict(
    #         batch_size=1,
    #         token_len=256
    #     )
    # )
    # wraped_transformer = wrap_llm_model(pipe.transformer, wrap_cfg)
    # wraped_transformer.cuda()
    # wraped_transformer.to(torch.float16)

    # wraped_transformer = None
    
    inference_engine = Qwen3LegacyInference("work_dirs/zimage/meta.json", fast_mode=True, tokenizer=pipe.tokenizer)
    xhmodel = cus_ZImagePipeline.to_hf_compatible(
        pipe, text_encoder=inference_engine, vae=hm_vae, transformers=hm_dit,
        meta_info="work_dirs/zimage/meta.json",
    )

    image = xhmodel(
        prompt=prompt,
        height=1024,
        width=1024,
        num_inference_steps=9,  # This actually results in 8 DiT forwards
        guidance_scale=0.0,     # Guidance should be 0 for the Turbo models
        generator=torch.Generator("cuda").manual_seed(42),
        meta_info="work_dirs/zimage/meta.json",
    )
    image[0].save("example.png")


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
