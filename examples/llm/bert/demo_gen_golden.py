# Copyright 2025 HOUMO AI
#
# File: demo_gen_golden.py
# Description:
#   Example script: llm/bert/demo_gen_golden.py
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

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from xhquant.api import (
    HMONNXGoldenInference,
    get_root_logger,
    xhquant_init,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="BERT Golden Inference Demo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/data02/datasets/bert_chinese",
        help="Path to the pretrained BERT model",
    )
    parser.add_argument(
        "--hmonnx_file",
        type=str,
        default="work_dirs/bert/hmonnx/prefill/bert_ch-XH2a-0k-w8a8h1_sefp_prefill.onnx",
        help="Path to the HMONNX model file",
    )
    parser.add_argument(
        "--work_dirs",
        type=str,
        default="work_dirs/bert",
        help="Base directory for work files",
    )
    parser.add_argument(
        "--context_length",
        type=int,
        default=512,
        help="Maximum context length for tokenization",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run inference on",
    )
    parser.add_argument(
        "--input_text",
        type=str,
        default="你好",
        help="Input text for BERT model",
    )
    parser.add_argument(
        "--save_golden",
        action="store_true",
        help="Whether to save golden outputs",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.work_dirs, exist_ok=True)

    xhquant_init()
    logger = get_root_logger()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    native_model = AutoModelForMaskedLM.from_pretrained(args.model_path)
    native_model = native_model.to(args.device)

    input_ids = tokenizer(
        args.input_text,
        return_tensors="pt",
        padding="max_length",
        max_length=args.context_length,
    ).input_ids.to(args.device)

    with torch.no_grad():
        output = native_model(input_ids)

    token_type_ids = torch.zeros(
        (1, args.context_length), dtype=torch.long, device=args.device
    )
    token_type_embeddings = native_model.bert.embeddings.token_type_embeddings(
        token_type_ids
    )

    position_ids = torch.arange(
        args.context_length, dtype=torch.long, device=args.device
    ).unsqueeze(0)
    position_embeddings = native_model.bert.embeddings.position_embeddings(
        position_ids
    )

    atten_mask = torch.zeros((1, args.context_length), device=args.device)

    token_embedding = native_model.bert.embeddings.word_embeddings
    input_emb = token_embedding(input_ids)

    inputs = [
        input_emb.half().to(args.device),
        token_type_embeddings.half().to(args.device),
        position_embeddings.half().to(args.device),
        atten_mask.half().to(args.device),
    ]

    session = HMONNXGoldenInference(args.hmonnx_file)
    session.to(args.device)
    session.save_golden = args.save_golden
    session.golden_dir = os.path.join(args.work_dirs, "hmonnx/golden")
    session.step = 0

    with torch.no_grad():
        hm_out = session(*inputs)

    similarity = torch.cosine_similarity(output.logits, hm_out)
    logger.info(f"Cosine similarity: {similarity.item():.6f}")

    print(f"Cosine similarity: {similarity.item():.6f}")


if __name__ == "__main__":
    main()
