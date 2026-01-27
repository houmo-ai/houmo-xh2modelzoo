# Copyright 2025 HOUMO AI
#
# File: qwen3_vl_moe_common_quant.py
# Description:
#   Example script: llm/qwen3vl_moe/qwen3_vl_moe_common_quant.py
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
import os.path as osp
from pathlib import Path

import torch
import torch.nn as nn
import transformers
from loguru import logger
from qwen_vl_utils import process_vision_info
from safetensors.torch import load_file as load_safetensors_file
from safetensors.torch import save_file as save_safetensors_file
from tqdm import tqdm

from transformers import AutoConfig, AutoProcessor, AutoModelForImageTextToText


def create_template(prompt, image_dir, resized_height=280, resized_width=420):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_dir,
                    "resized_height": resized_height,
                    "resized_width": resized_width,
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]
    return messages


def get_inputs(processor, image_dir, image_size_w=420, image_size_h=280):
    messages = create_template("描述下这张图片.", image_dir, resized_height=image_size_h, resized_width=image_size_w)

    # Preparation for inference
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages, image_patch_size=16)
    inputs = processor(
        text=text,
        images=image_inputs,
        videos=video_inputs,
        do_resize=False,
        return_tensors="pt",
    )
    return inputs


def demo(model, processor, image_dir="data/images/qwen2_vl_demo.jpeg"):
    from xh_model_zoo.xh_llm.quarot import utils

    raw_device = next(model.parameters()).device

    model.to("cuda")
    inputs = get_inputs(processor, image_dir)
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=512)
    generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    print(output_text)
    model.to(raw_device)
    utils.cleanup_memory()


def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", type=str, default="weights/Qwen3-VL-8B-Instruct")
    parser.add_argument("--out_dir", type=str, default="work_dirs/")
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--demo", action="store_true", help="demo")
    parser.add_argument("--image_dir", type=str, default="data/images/qwen2_vl_demo.jpeg")
    parser.add_argument("--calib_dataset", type=str, default="vllm_custom_data")
    parser.add_argument("--calib_samples", type=int, default=8)
    parser.add_argument("--w_bits", type=int, default=4)
    parser.add_argument("--w_head_bits", type=int, default=8)
    parser.add_argument("--data_files", nargs="+", type=str, help="List of dataset files")
    parser.add_argument("--heading_gptq", action="store_true", help="heading_gptq")
    parser.add_argument("--use_hession_mse", action="store_true", help="use_hession_mse")
    parser.add_argument("--self_attn_weight_bits",type=int,default=None)
    return parser.parse_args()


def msg_output_format(title):
    padding_str = "*" * 10
    title = f"{padding_str} {title} {padding_str}"
    return title


def main():
    args = parse_arguments()
    out_dir = Path(args.out_dir)
    hf_model_dir = args.model

    hf_model_path = osp.normpath(osp.abspath(args.model))
    model_name = Path(hf_model_path).name
    cfg_name = model_name
    cfg_name += "_quarot"
    cfg_name += "_gptq"

    cfg_name += f"_transformers-{transformers.__version__}"

    work_dir = Path(out_dir) / cfg_name
    work_dir.mkdir(exist_ok=True, parents=True)
    config = AutoConfig.from_pretrained(hf_model_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16

    native_model: nn.Module = AutoModelForImageTextToText.from_pretrained(
        hf_model_dir,
        torch_dtype=dtype,
        device_map="cpu",
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )

    if native_model.config.tie_word_embeddings:
        old_torchscript = native_model.config.torchscript
        native_model.config.torchscript = True
        native_model.tie_weights()
        native_model.config.tie_word_embeddings = False
        native_model.config.torchscript = old_torchscript

        native_model.config.text_config.tie_word_embeddings = False

    native_model.eval()
    native_model.to(dtype)

    processor = AutoProcessor.from_pretrained(hf_model_dir)

    quant_methods = []

    torch.cuda.reset_peak_memory_stats()
    quant_methods.append("quarot")
    quant_name = "_".join(quant_methods)
    filename = work_dir / f"{quant_name}-state-dict.safetensors"

    if not os.path.exists(filename):
        from xh_model_zoo.xh_llm.quarot.quantizer_utils import quarot

        logger.info(msg_output_format("Start quarot quantization"))
        native_model = quarot(native_model, device=device)
        logger.info(msg_output_format("End quarot quantization"))
        native_model.to(torch.float16)
        # state_dict = native_model.state_dict()
        # logger.info(msg_output_format(f"Saving checkpoint to: {filename}"))
        # # torch.save(state_dict, filename)
        # # save_safetensors_file(state_dict, filename)
        logger.info(f"Save checkpoint to: {filename}")
    else:
        from xh_model_zoo.xh_llm.quarot.quantizer_utils import rotation_utils

        rotation_utils.fuse_layer_norms(native_model)
        state_dict = load_safetensors_file(filename)
        native_model.to(torch.float16)
        native_model.load_state_dict(state_dict)
        logger.info(msg_output_format(f"Load state_dict from {filename}"))

    if args.demo:
        demo(native_model, processor, args.image_dir)

    from xh_model_zoo.xh_llm.quarot.quantizer_utils import gptq

    gptq_config = dict(
        calib_dataset=args.calib_dataset,
        calib_samples=args.calib_samples,
        seqlen=2048,
        w_clip=True,
        w_bits=args.w_bits,
        w_asym=False,
        w_groupsize=64,
        percdamp=0.01,
        act_order=False,
        int8_down_proj=False,
        heading_gptq=args.heading_gptq,
        w_head_bits=args.w_head_bits,
    )

    torch.cuda.reset_peak_memory_stats()
    logger.info(msg_output_format("Start gptq quantization"))
    quant_methods.append("gptq")
    quant_methods.append(f"use_hession_mse_{args.use_hession_mse}")
    quant_methods.append(f"calib_samples_{args.calib_samples}")
    quant_methods.append(f"heading_gptq_{args.heading_gptq}")
    layers_cache_dir = work_dir / "layers_cache"
    layers_cache_dir.mkdir(exist_ok=True, parents=True)

    native_model = gptq(
        native_model,
        args=args,
        model_name=hf_model_dir,
        **gptq_config,
        device=device,
        processor=processor,
        data_files=args.data_files,
        is_qwen3_vl=True,
        is_moe=True,
        use_hession_mse=args.use_hession_mse,
        self_attn_weight=args.self_attn_weight_bits
    )
    logger.info(msg_output_format("End gptq quantization"))

    consumption = torch.cuda.max_memory_allocated()
    unit = "B"
    if consumption > 1024:
        consumption = consumption / 1024
        unit = "k"
        if consumption > 1024:
            consumption = consumption / 1024
            unit = "M"
        consumption = round(consumption, 2)
    logger.info(f"GPU memory cost for export {consumption}{unit}")

    if len(quant_methods) != 0:
        quant_name = "_".join(quant_methods)
        filename = work_dir / f"{quant_name}-state-dict.safetensors"
        state_dict = native_model.state_dict()
        # del native_model
        for k in tqdm(state_dict):
            paths = k.split(".")
            v = state_dict[k]
            if paths[-1] == "quant_weight":
                if v.min().item() >= -pow(2, 7) and v.max().item() <= pow(2, 7) - 1:
                    v = v.to(torch.int8)
                elif v.min().item() >= -pow(2, 15) and v.max().item() <= pow(2, 15) - 1:
                    v = v.to(torch.int16)
                else:
                    v = v.to(torch.float32)
            else:
                v = v.to(torch.float16)

            state_dict[k] = v
        logger.info(msg_output_format(f"Saving checkpoint to: {filename}"))
        save_safetensors_file(state_dict, filename)
        logger.info(f"Save checkpoint to: {filename}")


if __name__ == "__main__":
    main()
