# Copyright 2025 HOUMO AI
#
# File: glm_ocr_fp_demo.py
# Description:
#   GLM-OCR float-point demo: load HF model and run inference on a single image.
#   Example script: examples/llm/glm_ocr/glm_ocr_fp_demo.py
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
import sys
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

# Allow importing common.py from the same directory
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import build_inputs, build_messages, load_quarot_gptq_state_dict, resolve_torch_dtype


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", type=str, default="/data02/datasets/GLM-OCR/")
    parser.add_argument("--image", type=str, default="examples/llm/glm_ocr/data/img3.png")
    parser.add_argument("--prompt", type=str, default="Text Recognition:")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--attn_implementation", type=str, default="eager")
    parser.add_argument("--quarot_gptq_path", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    dtype = resolve_torch_dtype(args.dtype)
    device = torch.device(args.device)

    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        dtype=dtype,
        device_map="cpu",
        trust_remote_code=True,
        attn_implementation=args.attn_implementation,
    )
    model.eval()

    if args.quarot_gptq_path is not None:
        load_quarot_gptq_state_dict(model, args.quarot_gptq_path, strict=False)

    model = model.to(device)
    messages = build_messages(args.image, args.prompt)
    inputs = build_inputs(processor, messages, device=device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)

    output_text = processor.decode(generated_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)
    print(output_text)


if __name__ == "__main__":
    main()
