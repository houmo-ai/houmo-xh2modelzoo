# Copyright 2025 HOUMO AI
#
# File: Xiaomi-MiMo-VL-Miloco_xh2a_demo.py
# Description:
#   Example script: llm/Xiaomi-MiMo-VL-Miloco/Xiaomi-MiMo-VL-Miloco_xh2a_demo.py
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

from pathlib import Path
from types import SimpleNamespace
import torch
import json
import os
import torch.nn as nn
import xhquant.utils.suppress_printing

from xh_model_zoo.xh_llm.models.qwen2_5_vl import Qwen2_5_VLONNXModel

from xhquant.api import get_root_logger


def parse_arguments():
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--visual_onnx", type=str, required=True)
    parser.add_argument("--prefill_onnx", type=str, required=True)
    parser.add_argument("--decode_onnx", type=str, required=True)
    parser.add_argument("--blocks_num", type=int, required=True, choices=[28, 36], help="blocks number of the model, 7B is 28, 3B is 36")
    parser.add_argument("--input_sequence_length", type=int, default=256, help="prefill input sequence length")
    parser.add_argument("--cache_len", type=int, default=2048, help="kv cache length")
    parser.add_argument("--image_size_w", type=int, default=1204, help="image width")
    parser.add_argument("--image_size_h", type=int, default=1204, help="image height")
    parser.add_argument("--max_size_t", type=int, default=2, help="max size t")
    parser.add_argument("--temporal_patch_size", type=int, default=2, help="temporal patch size")
    parser.add_argument("--patch_size", type=int, default=14, help="patch size")
    parser.add_argument("--image", type=str, default="data/images/qwen2_vl_demo.jpeg")
    parser.add_argument("--prompt", type=str, default="请用简洁优美的语言描述这张图片。")
    parser.add_argument("--resume", action="store_true", help="resume export golden")
    parser.add_argument("--use_fast", action="store_true", help="fast mode")
    parser.add_argument("--repetition_penalty", type=float, default=1.0, help="repetition penalty")
    parser.add_argument("--not_resize_v1", action="store_false")
    parser.add_argument("--chat_template", type=str, default=None, help="chat template")
    parser.add_argument("--response_emotion_path", type=str, default=None)
    return parser


def main():
    parser = parse_arguments()
    args = parser.parse_args()

    visual_onnx = args.visual_onnx
    prefill_onnx = args.prefill_onnx
    decode_onnx = args.decode_onnx
    blocks_num = args.blocks_num
    cache_len = args.cache_len
    image_size_w = args.image_size_w
    image_size_h = args.image_size_h
    max_size_t = args.max_size_t
    temporal_patch_size = args.temporal_patch_size
    patch_size = args.patch_size
    input_sequence_length = args.input_sequence_length
    image = args.image
    prompt = args.prompt
    resume = args.resume
    use_fast = args.use_fast
    model_dir = args.model_dir

    model_dir = Path(model_dir)

    image_feature_cfg = SimpleNamespace(
        onnx=visual_onnx,
        patch_size=patch_size,
        image_size_w=image_size_w,
        image_size_h=image_size_h,
        max_size_t=max_size_t,
        temporal_patch_size=temporal_patch_size,
    )

    prefill_cfg = SimpleNamespace(
        onnx=prefill_onnx,
        input_sequence_length=input_sequence_length,
    )

    decode_cfg = SimpleNamespace(
        onnx=decode_onnx,
    )

    kv_cache_cfg = SimpleNamespace(
        num_decoder_layers=blocks_num,
        num_hidden_layers=blocks_num,
        shape=[1, 8, cache_len, 128],
    )

    exec_device = "cuda" if torch.cuda.is_available() else "cpu"

    logger = get_root_logger()
    logger.info(f"Config:\n{image_feature_cfg}\n{prefill_cfg}\n{decode_cfg}\n{kv_cache_cfg}")

    xh_model: Qwen2_5_VLONNXModel = Qwen2_5_VLONNXModel(
        image_feature = image_feature_cfg,
        prefill = prefill_cfg,
        decode = decode_cfg,
        kv_cache = kv_cache_cfg,
        image_size_w = image_size_w,
        image_size_h = image_size_h,
        max_size_t = max_size_t,
        resize_v1 = args.not_resize_v1,
        repetition_penalty = args.repetition_penalty,
        chat_template = args.chat_template,
    )

    xhquant.utils.suppress_printing.disable_printing = True  # 屏蔽不必要的打印信息

    # 消除Embedding的安全检查
    torch.serialization.add_safe_globals([nn.Embedding])
    token_embedding = torch.load(model_dir / "token_embedding.pt", weights_only=False, map_location="cpu")
    torch.serialization.clear_safe_globals()

    hf_model_config_dir = model_dir / "hf_config"

    from xh_model_zoo.xh_llm.models.qwen2_5_vl import Qwen2_5_VLProcessor
    processor = Qwen2_5_VLProcessor.from_pretrained(hf_model_config_dir)
    
    xh_model.set_input_embeddings(token_embedding)
    xh_model.set_exec_device(exec_device)

    if args.response_emotion_path is not None:
        output_path = Path("output.txt")
        with output_path.open("w", encoding="utf-8") as output_file:
            for file_name in os.listdir(args.response_emotion_path):
                file_path = os.path.join(args.response_emotion_path, file_name)
                with open(file_path, "r", encoding="utf-8") as input_file:
                    data = json.load(input_file)
                for item in data:
                    output_text = xh_model.chat(item["user_text"], image, processor, logger, use_fast=use_fast)
                    output_file.write(f'{item["user_text"]}, XH2a Model Output: {output_text}\n')

    else:
        xh_model.chat(prompt, image, processor, logger, use_fast=use_fast)


if __name__ == "__main__":
    main()

