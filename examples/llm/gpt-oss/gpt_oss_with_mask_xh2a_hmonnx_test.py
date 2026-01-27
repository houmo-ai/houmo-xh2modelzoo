# Copyright 2025 HOUMO AI
#
# File: gpt_oss_with_mask_xh2a_hmonnx_test.py
# Description:
#   Example script: llm/gpt-oss/gpt_oss_with_mask_xh2a_hmonnx_test.py
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

from xh_model_zoo.xh_llm.models.gpt_oss_with_mask import GptOssWithMask_HFCompatible, GptOssWithMaskInference
from xh_model_zoo.xh_llm.utils import auto_offload


def main(args):
    xhquant_init(None, args.debug)
    inference_engine = GptOssWithMaskInference(args.config, fast_mode=args.fast)
    hf_model_path = inference_engine.meta_info.get("hf_model_path", None)
    if hf_model_path is None:
        hf_model_path = args.hf_model
    assert Path(hf_model_path).exists(), f"HF model path {hf_model_path} does not exist."
    batch_size = inference_engine.batch_size
    logger = get_root_logger()
    prompt = "你是谁？"
    message = [
        {"role": "user", "content": prompt},
    ]
    device = inference_engine.device
    tokenizer = inference_engine.get_tokenizer(hf_model_path)

    model_inputs = tokenizer.apply_chat_template(
        message,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        reasoning_effort="low",  # Defaults to "medium", but also accepts "high" and "low"
    ).to(device)

    streamer = TextStreamer(tokenizer)

    wraped_hf_model = GptOssWithMask_HFCompatible.to_hf_compatible(hf_model_path, inference_engine)
    auto_offload(wraped_hf_model, "XH2aQuantGptOssBlock")
    wraped_hf_model.eval()  # type: ignore
    wraped_hf_model.to(device)  # type: ignore
    streamer = TextStreamer(tokenizer)
    xhonnxruntime_config.disable_progress = True
    xhonnxruntime_config.verbose_progress = False
    with torch.no_grad():
        generated_ids = wraped_hf_model.generate(  # type: ignore
                    **model_inputs,
                    max_new_tokens=2048,
                    temperature=0.0,
                    do_sample=False,
                    streamer=streamer,
        )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()
    content = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")

    logger.info(f"content:{content}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="work_dirs/gpt-oss-20b-XH2a-2k-w8a8h0_sefp/meta.json",
        help="Path to meta.json file",
    )
    parser.add_argument("--hf-model", type=str, default="data/datasets/gpt-oss-20b", help="HuggingFace model path")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--execution-device", type=str, default="cuda", help="execution device, default is cuda:0")
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument("--fast", action="store_true", help="run in fast mode")
    args = parser.parse_args()
    main(args)

