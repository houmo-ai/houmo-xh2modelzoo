# Copyright 2025 HOUMO AI
#
# File: qwen3_legacy_xh2a_hmonnx_test.py
# Description:
#   Example script: llm/qwen3_legacy_lora/qwen3_legacy_xh2a_hmonnx_test.py
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
from pathlib import Path

import torch
from transformers import TextStreamer
from xhquant.api import get_root_logger, xhquant_init
from xhquant.xhonnxruntime import config as xhonnxruntime_config

from xh_model_zoo.xh_llm.models.qwen3_legacy_lora import Qwen3LegacyLoRAHFCompatible, Qwen3LegacyLoRAInference


def main(args):
    xhquant_init(None, args.debug)
    inference_engine = Qwen3LegacyLoRAInference(args.config, fast_mode=args.fast)
    inference_engine.enable_lora = True
    hf_model_path = inference_engine.meta_info.get("hf_model_path", None)
    if hf_model_path is None:
        hf_model_path = args.hf_model
    assert Path(hf_model_path).exists(), f"HF model path {hf_model_path} does not exist."

    logger = get_root_logger()

    messages_all = [
        [
            {
                "role": "system",
                "content": "你是一个AIPC的意图识别助理，你的任务是按照KYLIN_INTENT_DETECT的要求，结合上下文对话记录和用户输入进行意图识别和槽位信息抽取。",
            },
            {"role": "user", "content": "声音调到80"},
        ],
        [
            {
                "role": "system",
                "content": "你是一个AIPC的意图识别助理，你的任务是按照KYLIN_INTENT_DETECT的要求，结合上下文对话记录和用户输入进行意图识别和槽位信息抽取，若用户输入为没有意义的字符串或已有标签不符合用户意图，则将意图标签识别为其他，语言请保持与用户输入一致。",
            },
            {
                "role": "user",
                "content": "打开设置",
            },
        ],
        [
            {
                "role": "system",
                "content": "你是一个AIPC的意图识别助理，你的任务是按照KYLIN_INTENT_DETECT的要求，结合上下文对话记录和用户输入进行意图识别和槽位信息抽取，若用户输入为没有意义的字符串或已有标签不符合用户意图，则将意图标签识别为其他，语言请保持与用户输入一致。",
            },
            {
                "role": "user",
                "content": "休眠",
            },
        ],
        [
            {
                "role": "system",
                "content": "你是一个AIPC的意图识别助理，你的任务是按照KYLIN_INTENT_DETECT的要求，结合上下文对话记录和用户输入进行意图识别和槽位信息抽取，若用户输入为没有意义的字符串或已有标签不符合用户意图，则将意图标签识别为其他，语言请保持与用户输入一致。",
            },
            {
                "role": "user",
                "content": "打开浏览器",
            },
        ],
    ]
     
    device = inference_engine.device
    tokenizer = inference_engine.tokenizer
    for messages in messages_all:
        texts = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            # enable_thinking=False,
        )

        model_inputs = tokenizer(texts, padding=True, return_tensors="pt").to(device)

        wraped_hf_model = Qwen3LegacyLoRAHFCompatible.to_hf_compatible(hf_model_path, inference_engine)
        wraped_hf_model.eval()  # type: ignore
        wraped_hf_model.to(device)  # type: ignore
        streamer = TextStreamer(tokenizer)
        xhonnxruntime_config.disable_progress = True
        xhonnxruntime_config.verbose_progress = False
        with torch.no_grad():
            generated_ids = wraped_hf_model.generate(  # type: ignore
                **model_inputs,
                max_new_tokens=512,
                streamer=streamer,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()

        # parsing thinking content
        try:
            # rindex finding 151668 (</think>)
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0

        thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
        content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

        logger.info(f"thinking content:{thinking_content}")
        logger.info(f"{texts}")
        logger.info(f"content:{content}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="",
    )
    parser.add_argument("--hf-model", type=str)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--execution_device", type=str, default="cuda:0", help="execution device, default is cuda:0")
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument("--fast", action="store_true", help="run in fast mode")
    args = parser.parse_args()
    main(args)
