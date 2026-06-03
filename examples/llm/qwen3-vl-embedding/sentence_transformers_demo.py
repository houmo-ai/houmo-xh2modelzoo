# -*- coding: utf-8 -*-
# Copyright 2025 HOUMO AI
#
# File: sentence_transformers_demo.py
# Description:
#   SentenceTransformer-style demo for Qwen3-VL-Embedding. Defaults to the
#   locally cached Qwen3-VL-Embedding-2B snapshot. Output embedding dim is
#   2048 for 2B (8B is 4096).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
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

from sentence_transformers import SentenceTransformer


DEFAULT_LOCAL_2B = (
    "/data01/home/she.gao/.cache/huggingface/hub/"
    "models--Qwen--Qwen3-VL-Embedding-2B/snapshots/"
    "9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda"
)


def main(args):
    model_path = args.model_dir or os.environ.get("QWEN3_VL_EMB_DIR", DEFAULT_LOCAL_2B)
    model = SentenceTransformer(model_path, trust_remote_code=True)

    queries = [
        "A woman playing with her dog on a beach at sunset.",
        "Pet owner training dog outdoors near water.",
        "Woman surfing on waves during a sunny day.",
        "City skyline view from a high-rise building at night.",
    ]

    documents = [
        "A woman shares a joyful moment with her golden retriever on a sun-drenched beach at sunset, as the dog offers its paw in a heartwarming display of companionship and trust.",
        "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
        {
            "text": "A woman shares a joyful moment with her golden retriever on a sun-drenched beach at sunset, as the dog offers its paw in a heartwarming display of companionship and trust.",
            "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
        },
    ]

    query_embeddings = model.encode(queries)
    doc_embeddings = model.encode(documents)
    print(query_embeddings.shape, doc_embeddings.shape)
    # 2B -> (4, 2048) (3, 2048); 8B -> (4, 4096) (3, 4096)

    similarities = model.similarity(query_embeddings, doc_embeddings)
    print(similarities)
    # 8B reference (from model card):
    # tensor([[0.7438, 0.6556, 0.6244],
    #         [0.4430, 0.3323, 0.3929],
    #         [0.3685, 0.2310, 0.2874],
    #         [0.0602, -0.0162, 0.0167]])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help=f"Path to Qwen3-VL-Embedding model. Default: env QWEN3_VL_EMB_DIR or {DEFAULT_LOCAL_2B}",
    )
    args = parser.parse_args()
    main(args)
