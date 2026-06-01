# Copyright 2025 HOUMO AI
#
# File: hm_cls_demo.py
# Description:
#   Example script: llm/bert/hm_cls_demo.py
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
import pickle
import re

import torch
from transformers import BertForSequenceClassification, BertTokenizer

from xhquant.api import (
    HMONNXGoldenInference,
    get_root_logger,
    xhquant_init,
)


PAGE_PATTERN = re.compile(r"cur_page:\d+", re.IGNORECASE)


def parse_args():
    parser = argparse.ArgumentParser(
        description="BERT Classification Inference Demo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/data02/users/cc_work/model312/BERT",
        help="Path to the pretrained BERT model",
    )
    parser.add_argument(
        "--label_encoder_path",
        type=str,
        default="/data02/users/cc_work/model312/BERT/label_encoder.pkl",
        help="Path to the label encoder pickle file",
    )
    parser.add_argument(
        "--hm_model",
        type=str,
        default="work_dirs/bert/hmonnx/prefill/bert_ch-XH2a-0k-w8a8h1_sefp_prefill.onnx",
        help="Path to the HMONNX model file",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to run inference on (e.g., cuda:0, cuda:2, cpu)",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum sequence length for tokenization",
    )
    parser.add_argument(
        "--input_text",
        type=str,
        default="受害者是几点死亡的，死亡的具体原因是谋杀",
        help="Input text for classification",
    )
    parser.add_argument(
        "--truncate_length",
        type=int,
        default=100,
        help="Truncate input text to this length after preprocessing",
    )
    return parser.parse_args()


def process_text_content(content: str, truncate_length: int = 100) -> str:
    """Process text content: remove page patterns, whitespace, and truncate.

    Args:
        content: Input text content.
        truncate_length: Maximum length after processing.

    Returns:
        Processed text string.
    """
    content = PAGE_PATTERN.sub("", content)
    content = re.sub(r"\s+", "", content)
    content = content[:truncate_length]
    return content


def main():
    args = parse_args()

    xhquant_init()
    logger = get_root_logger()

    logger.info(f"Loading model from {args.model_path}")
    model = BertForSequenceClassification.from_pretrained(
        args.model_path,
        local_files_only=True,
    )
    token_embedding = model.bert.embeddings.word_embeddings
    model.to(args.device)
    model.eval()

    tokenizer = BertTokenizer.from_pretrained(
        args.model_path,
        local_files_only=True,
    )

    with open(args.label_encoder_path, "rb") as f:
        label_encoder = pickle.load(f)

    text = process_text_content(args.input_text, args.truncate_length)
    logger.info(f"Processed text: {text}")

    encoding = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=args.max_length,
        return_tensors="pt",
    )
    encoding = {k: v.to(args.device) for k, v in encoding.items()}

    with torch.no_grad():
        input_emb = token_embedding(encoding["input_ids"])
        attention_mask = (1 - encoding["attention_mask"]) * -65504
        position_ids = torch.arange(
            args.max_length, dtype=torch.long, device=args.device
        ).unsqueeze(0)
        position_embeddings = model.bert.embeddings.position_embeddings(
            position_ids
        )
        token_type_embeddings = model.bert.embeddings.token_type_embeddings(
            encoding["token_type_ids"]
        )

        inputs = [
            input_emb.half().to(args.device),
            token_type_embeddings.half().to(args.device),
            position_embeddings.half().to(args.device),
            attention_mask.half().to(args.device),
        ]

        session = HMONNXGoldenInference(args.hm_model)
        session.to(args.device)
        outputs = session(*inputs)

        probabilities = torch.nn.functional.softmax(outputs, dim=-1)
        predicted_class = torch.argmax(probabilities, dim=-1)
        confidence = probabilities[0][predicted_class]

    all_probs = probabilities[0].cpu().numpy()
    prob_dict = {}
    for i, prob in enumerate(all_probs):
        label = label_encoder.inverse_transform([i])[0]
        prob_dict[label] = float(prob)
    
    predicted_label = label_encoder.inverse_transform([predicted_class.cpu().numpy()])[0]

    print(predicted_label)

if __name__ == "__main__":
    main()
