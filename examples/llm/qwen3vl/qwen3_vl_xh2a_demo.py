# Copyright 2025 HOUMO AI
#
# File: qwen3_vl_xh2a_demo.py
# Description:
#   Example script: llm/qwen3vl/qwen3_vl_xh2a_demo.py
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

from xh_model_zoo.xh_llm.models.qwen3_vl import Qwen3VLONNXModel

from xhquant.api import get_root_logger

MODEL_TYPE_TO_NUM_BLOCKS = {
    "2B": 28,
    "4B": 36,
    "8B": 36,
}

def parse_arguments():
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--visual_onnx", type=str, required=True)
    parser.add_argument("--prefill_onnx", type=str, required=True)
    parser.add_argument("--decode_onnx", type=str, required=True)
    parser.add_argument("--model_type", type=str, default="2B", choices=["2B", "4B", "8B"])
    parser.add_argument("--input_sequence_length", type=int, default=256, help="prefill input sequence length")
    parser.add_argument("--cache_len", type=int, default=2048, help="kv cache length")
    parser.add_argument("--image_size_w", type=int, default=448, help="image width")
    parser.add_argument("--image_size_h", type=int, default=448, help="image height")
    parser.add_argument("--max_size_t", type=int, default=2, help="max size t")
    parser.add_argument("--temporal_patch_size", type=int, default=2, help="temporal patch size")
    parser.add_argument("--patch_size", type=int, default=16, help="patch size")
    parser.add_argument("--image", type=str, default="data/images/qwen2_vl_demo.jpeg")
    parser.add_argument("--video", type=str, default=None)
    parser.add_argument("--prompt", type=str, default="请用简洁优美的语言描述这张图片。")
    parser.add_argument("--resume", action="store_true", help="resume export golden")
    parser.add_argument("--use_fast", action="store_true", help="fast mode")
    parser.add_argument("--repetition_penalty", type=float, default=1.0, help="repetition penalty")
    parser.add_argument("--not_resize_v1", action="store_false")
    parser.add_argument("--chat_template", type=str, default=None, help="chat template")
    parser.add_argument("--presence_penalty", type=float, default=0.0)
    return parser


def main():
    parser = parse_arguments()
    args = parser.parse_args()

    visual_onnx = args.visual_onnx
    prefill_onnx = args.prefill_onnx
    decode_onnx = args.decode_onnx
    blocks_num = MODEL_TYPE_TO_NUM_BLOCKS[args.model_type]
    cache_len = args.cache_len
    image_size_w = args.image_size_w
    image_size_h = args.image_size_h
    max_size_t = args.max_size_t
    temporal_patch_size = args.temporal_patch_size
    patch_size = args.patch_size
    input_sequence_length = args.input_sequence_length
    image = args.image
    video = args.video
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

    xh_model: Qwen3VLONNXModel = Qwen3VLONNXModel(
        image_feature = image_feature_cfg,
        prefill = prefill_cfg,
        decode = decode_cfg,
        kv_cache = kv_cache_cfg,
        image_size_w = image_size_w,
        image_size_h = image_size_h,
        max_size_t = max_size_t,
        resize_v1 = args.not_resize_v1,
        presence_penalty = args.presence_penalty
        # repetition_penalty = args.repetition_penalty,
        # chat_template = args.chat_template,
    )

    xhquant.utils.suppress_printing.disable_printing = True  # 屏蔽不必要的打印信息

    # 消除Embedding的安全检查
    torch.serialization.add_safe_globals([nn.Embedding])
    token_embedding = torch.load(model_dir / "token_embedding.pt", weights_only=False, map_location="cpu")
    torch.serialization.clear_safe_globals()

    hf_model_config_dir = model_dir / "hf_config"

    from xh_model_zoo.xh_llm.models.qwen3_vl import Qwen3VLProcessor
    processor = Qwen3VLProcessor.from_pretrained(hf_model_config_dir)
    
    xh_model.set_input_embeddings(token_embedding)
    xh_model.set_exec_device(exec_device)

    media_type = "video" if video else "image"
    media_path = video if video else image
    xh_model.chat(prompt, media_path, processor, logger, use_fast=use_fast, media_type=media_type)


if __name__ == "__main__":
    main()

