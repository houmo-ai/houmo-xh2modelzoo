# Copyright 2025 HOUMO AI
#
# File: mineru2.5_xh2a_hmonnx_test.py
# Description:
#   Example script: llm/mineru2.5/mineru2.5_xh2a_hmonnx_test.py
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
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, AutoTokenizer
from transformers.models.qwen2_vl.processing_qwen2_vl import Qwen2VLProcessor
from xhquant.api import ConfigDict, HMONNXInference, get_root_logger, HMONNXGoldenInference
from xhquant.core import CacheTensor

from xh_model_zoo.xh_llm.models.qwen2_vl import Qwen2VLDataPreprocess
from xh_model_zoo.xh_llm.utils import decode_next_token


def main(args):
    export_golden = True

    model_dir = Path(args.config).parent
    meta_info = json.load(open(args.config, "r"))
    meta_info = ConfigDict(meta_info)

    # hmonnx files
    prefill_onnx_file = "work_dirs/mineru2.5-XH2a-batch_1-4k-w8a8h1_sefp/hmonnx/mineru2.5-XH2a-w8a8h1_sefp-llm-prefill.onnx"
    decode_onnx_file = "work_dirs/mineru2.5-XH2a-batch_1-4k-w8a8h1_sefp/hmonnx/mineru2.5-XH2a-w8a8h1_sefp-llm-decode.onnx"
    vision_onnx_file = "work_dirs/mineru2.5-XH2a-batch_1-4k-w8a8h1_sefp/hmonnx/mineru2.5-XH2a-w8a8h1_sefp_vision.onnx"

    # tokenizer
    hf_model_config_dir = str(model_dir / meta_info["hf_config"])
    tokenizer = AutoTokenizer.from_pretrained(hf_model_config_dir)

    # token embedding
    token_embedding_state_dict = torch.load(
        model_dir / meta_info["token_embedding_file"], map_location="cpu", weights_only=True
    )
    token_embedding = nn.Embedding(
        token_embedding_state_dict["weight"].shape[0],
        token_embedding_state_dict["weight"].shape[1],
    )
    token_embedding.load_state_dict(token_embedding_state_dict)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    # "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                    "image": "data/images/test_mineru2.5.png",
                },
                {"type": "text", "text": "\nLayout Detection:"},  # Text Recognition:
            ],
        }
    ]
    processor: Qwen2VLProcessor = AutoProcessor.from_pretrained(hf_model_config_dir, use_fast=True)
    # Preparation for inference
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    # resize long side to 1036
    # max_size = meta_info["wrap_cfg"]["visual"].image_max_size
    # patch_size = meta_info["wrap_cfg"]["visual"].patch_size
    # w, h = image_inputs[0].size
    # scale = 1036 / max(w, h)
    # new_w = int(w * scale)
    # new_h = int(h * scale)

    resized_img = image_inputs[0].resize((1036, 1036), Image.Resampling.BICUBIC)

    # pad_img = Image.new("RGB", (max_size, max_size), (122, 116, 104))
    # pad_img.paste(resized_img, (0, 0))
    image_inputs[0] = resized_img
    # processor.image_processor.max_pixels = processor.image_processor.max_pixels
    inputs = processor(
        text=[text],
        images=image_inputs,
        # videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    tokenizer = processor.tokenizer

    device = torch.device(args.device)
    execution_device = torch.device(args.execution_device)

    # 处理图像特征编码
    pixel_values = inputs["pixel_values"].view(-1, 3, 2, 14, 14)[:, :, 0, :, :].contiguous()
    pixel_values = pixel_values.to(torch.float16) # 4956423

    image_feature_session = HMONNXGoldenInference(str(vision_onnx_file))
    image_feature_session.to(device)
    image_feature_session.exec_device = execution_device
    image_feature_session.step = 0
    image_feature_session.save_golden = export_golden
    image_feature_session.golden_dir = "work_dirs/mineru2.5-XH2a-batch_1-4k-w8a8h1_sefp/golden/vision/"
    image_embeds = image_feature_session(pixel_values) # -25568
    image_feature_session = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # kv cache
    kv_cache_shape = meta_info["kv_cache"]["shape"]
    num_decoder_layers = meta_info["kv_cache"]["num_decoder_layers"]
    past_key_caches = []
    past_value_caches = []
    for i in range(num_decoder_layers):
        past_k_cache = CacheTensor(torch.zeros(kv_cache_shape, dtype=torch.float16))
        past_v_cache = CacheTensor(torch.zeros(kv_cache_shape, dtype=torch.float16))
        past_key_caches.append(past_k_cache)
        past_value_caches.append(past_v_cache)

    # prefill
    data_prefill = {
        "input_ids": inputs["input_ids"], 
        "image_embeds": image_embeds,
        "past_seq_length": [0],
        "image_grid_thw": inputs["image_grid_thw"],
    }

    prefill_input_sequence_length = meta_info["wrap_cfg"]["input_sequence_length"]
    input_seq_len = data_prefill["input_ids"].shape[-1]
    steps = (input_seq_len + prefill_input_sequence_length - 1) // prefill_input_sequence_length
    input_sequence_length = prefill_input_sequence_length * steps
    data_preprocess = Qwen2VLDataPreprocess(token_embedding, input_sequence_length)
    data_preprocess.to(execution_device)

    data_inputs = data_preprocess(data_prefill)
    prefill_inputs = list(data_inputs) + [past_key_caches, past_value_caches] # # -25616  0  1394

    prefill_session = HMONNXGoldenInference(str(prefill_onnx_file))
    prefill_session.to(device)
    prefill_session.exec_device = execution_device
    prefill_session.step = 0
    prefill_session.save_golden = export_golden
    prefill_session.golden_dir = "work_dirs/mineru2.5-XH2a-batch_1-4k-w8a8h1_sefp/golden/prefill/"

    inputs_embeds, past_seq_length, _, position_ids, past_key_caches, past_value_caches = prefill_inputs
    print(position_ids.shape)
    for i in range(steps):
        start = i * prefill_input_sequence_length
        end = (i + 1) * prefill_input_sequence_length
        current_input_length = min(end, input_seq_len) - start
        prefill_logits = prefill_session(
            inputs_embeds[:, start:end, :],
            past_seq_length,
            torch.tensor([current_input_length], dtype=torch.int32).to(inputs_embeds.device),
            position_ids[:, :, start:end],
            *past_key_caches,
            *past_value_caches,
        )
        past_seq_length += current_input_length
        break
    prefill_next_token_id, prefill_next_token_text = decode_next_token(tokenizer, prefill_logits)
    logger = get_root_logger()
    logger.info(f"Prefill next token: {prefill_next_token_id} {prefill_next_token_text}")
    del prefill_session
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # decode
    data_decode = {
        "input_ids": prefill_next_token_id,
        "past_seq_length": [inputs["input_ids"].shape[-1]],
    }
    data_preprocess.input_sequence_length = 1
    data_inputs = data_preprocess(data_decode)
    decode_inputs = list(data_inputs) + [past_key_caches, past_value_caches]

    decode_session = HMONNXGoldenInference(str(decode_onnx_file))
    decode_session.to(device)
    decode_session.exec_device = execution_device
    decode_session.step = 0
    decode_session.save_golden = export_golden
    decode_session.golden_dir = "work_dirs/mineru2.5-XH2a-batch_1-4k-w8a8h1_sefp/golden/decode/"

    inputs_embeds, past_seq_length, current_seq_length, position_ids, past_key_caches, past_value_caches = decode_inputs
    decode_logits = decode_session(
        inputs_embeds, past_seq_length, current_seq_length, position_ids, *past_key_caches, *past_value_caches
    )
    decode_next_token_id, decode_next_token_text = decode_next_token(tokenizer, decode_logits)
    logger.info(f"Decode next token: {decode_next_token_id} {decode_next_token_text}")

    del decode_session
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="work_dirs/mineru2.5-XH2a-batch_1-4k-w8a8h1_sefp/meta.json",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--execution_device", type=str, default="cuda:0", help="execution device, default is cuda:0")
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument("--image", type=str, default="data/images/qwen2_vl_demo.jpeg")
    args = parser.parse_args()
    main(args)
