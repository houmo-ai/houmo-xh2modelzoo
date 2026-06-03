# -*- coding: utf-8 -*-
# Copyright 2025 HOUMO AI
#
# File: native_qwen3_vl_embedding.py
# Description:
#   Native FP16/BF16 inference for Qwen3-VL-Embedding using the model
#   repository's bundled Qwen3VLEmbedder (scripts/qwen3_vl_embedding.py).
#   Backbone = Qwen3VLModel (visual + language). Pooling (last-token) and
#   L2 normalize are done in Python on top of last_hidden_state.
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
import importlib.util
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from loguru import logger


DEFAULT_LOCAL_2B = (
    "/data01/home/she.gao/.cache/huggingface/hub/"
    "models--Qwen--Qwen3-VL-Embedding-2B/snapshots/"
    "9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda"
)


def load_embedder_class(model_dir: str):
    """Load Qwen3VLEmbedder from the model's bundled scripts dir."""
    script_path = Path(model_dir) / "scripts" / "qwen3_vl_embedding.py"
    if not script_path.exists():
        raise FileNotFoundError(
            f"Expected {script_path} (bundled with the HF Qwen3-VL-Embedding model)."
        )
    spec = importlib.util.spec_from_file_location("qwen3_vl_embedding", script_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["qwen3_vl_embedding"] = module
    spec.loader.exec_module(module)
    return module.Qwen3VLEmbedder


def main(args):
    model_dir = args.model_dir
    device = args.device if torch.cuda.is_available() else "cpu"

    Qwen3VLEmbedder = load_embedder_class(model_dir)
    embedder = Qwen3VLEmbedder(
        model_name_or_path=model_dir,
        torch_dtype=torch.bfloat16,
    )
    embedder.model.eval()
    logger.info(f"Loaded Qwen3VLEmbedder from {model_dir} on {embedder.model.device}")

    queries = [
        {"text": "A woman playing with her dog on a beach at sunset."},
        {"text": "Pet owner training dog outdoors near water."},
        {"text": "Woman surfing on waves during a sunny day."},
        {"text": "City skyline view from a high-rise building at night."},
    ]

    documents = [
        {"text": "A woman shares a joyful moment with her golden retriever on a sun-drenched beach at sunset, as the dog offers its paw in a heartwarming display of companionship and trust."},
        {"image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"},
        {
            "text": "A woman shares a joyful moment with her golden retriever on a sun-drenched beach at sunset, as the dog offers its paw in a heartwarming display of companionship and trust.",
            "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
        },
    ]

    with torch.no_grad():
        query_embeddings = embedder.process(queries, normalize=True)
        doc_embeddings = embedder.process(documents, normalize=True)

    query_embeddings = query_embeddings.float().cpu()
    doc_embeddings = doc_embeddings.float().cpu()
    logger.info(f"query_embeddings shape: {tuple(query_embeddings.shape)}")
    logger.info(f"doc_embeddings shape: {tuple(doc_embeddings.shape)}")
    # 2B: (4, 2048) (3, 2048)

    similarities = query_embeddings @ doc_embeddings.T
    logger.info(f"cosine similarities (4x3):\n{similarities}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        type=str,
        default=os.environ.get("QWEN3_VL_EMB_DIR", DEFAULT_LOCAL_2B),
        help="Path to Qwen3-VL-Embedding HF snapshot",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()
    main(args)
