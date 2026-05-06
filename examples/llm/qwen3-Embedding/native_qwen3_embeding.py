# -*- coding: utf-8 -*-
# Copyright 2025 HOUMO AI
#
# File: native_qwen3_embeding.py
# Description:
#   Native Qwen3 Embedding evaluation harness for xh2modelzoo.
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

from typing import cast

import torch
import torch.nn.functional as F
from loguru import logger
from torch import Tensor
from transformers import AutoModel, AutoTokenizer
from transformers.models.qwen3 import Qwen3Model


def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths
        ]


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f"Instruct: {task_description}\nQuery:{query}"


def main(args):
    model_dir = args.model_dir
    # 每个查询都必须附带一个描述任务的一句话指令
    task = "Given a web search query, retrieve relevant passages that answer the query"

    queries = [
        get_detailed_instruct(task, "What is the capital of China?"),
        get_detailed_instruct(task, "Explain gravity"),
    ]
    # 检索文档无需添的指令说明
    documents = [
        "The capital of China is Beijing.",
        "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.",
    ]
    input_texts = queries + documents
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_dir, padding_side="left")
    model = AutoModel.from_pretrained(
        model_dir, attn_implementation="flash_attention_2", torch_dtype=torch.float16
    )
    # model = AutoModel.from_pretrained(model_dir)
    model = cast(Qwen3Model, model)
    model.to(device)
    model.eval()
    print(model)
    with open("model_info.txt", "w") as f:
        f.write(str(model))

    print("模型信息已保存到 model_info.txt")

    # 设置flash_attention_2 以及将`padding_side` 设置为"left"，可以加快模型的加载与运行速度
    # model = AutoModel.from_pretrained('Qwen/Qwen3-Embedding-8B', attn_implementation="flash_attention_2", torch_dtype=torch.float16).cuda()

    max_length = 8192

    # Tokenize the input texts
    batch_dict = tokenizer(
        input_texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    batch_dict.to(model.device)
    with torch.no_grad():
        outputs = model(**batch_dict)
    embeddings = last_token_pool(
        outputs.last_hidden_state, batch_dict["attention_mask"]
    )

    # normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)
    scores = embeddings[:2] @ embeddings[2:].T

    logger.info(f"Model: {model_dir}")
    logger.info(f"{scores.tolist()}")


if __name__ == "__main__":
    import debugpy

    debugpy.listen(("0.0.0.0", 1160))
    print("✅ debugpy listening on 0.0.0.0:5678, waiting for VSCode attach...")
    debugpy.wait_for_client()
    print("✅ VSCode attached, continue running.")
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        type=str,
        default="/data01/home/feilong.kong/llm_models/Qwen/Qwen3-Embedding-4B",
    )
    args = parser.parse_args()
    main(args)
