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

import torch
import torch.nn as nn
import torch.nn.functional as F
from xhquant.api import (
    DeviceType,
    HMONNXGoldenInference,
    HMONNXInference,
    QuantScheme,
    convert_onnx_to_hmonnx,
    create_quant_config,
    get_root_logger,
    xhquant_init,
)
import onnxruntime as ort   
from modelscope import AutoTokenizer, AutoModelForMaskedLM

out_hmonnx_file = "work_dirs/bert/hmonnx/prefill/bert_ch-XH2a-0k-w8a8h1_sefp_prefill.onnx"
device = "cuda"
work_dirs = "work_dirs/bert"
context_length = 256

tokenizer = AutoTokenizer.from_pretrained("/data02/datasets/bert_chinese")
native_model = AutoModelForMaskedLM.from_pretrained("/data02/datasets/bert_chinese")
native_model = native_model.to("cuda")


input_txt = "你好"
input_ids = tokenizer(
    input_txt, return_tensors="pt", padding="max_length", max_length=context_length
).input_ids.cuda()
output = native_model(input_ids)

token_type_ids = torch.zeros((1,context_length), dtype=torch.long, device=device)
token_type_embeddings = native_model.bert.embeddings.token_type_embeddings(token_type_ids)

position_ids = torch.arange(context_length, dtype=torch.long, device=device).unsqueeze(0)
position_embeddings = native_model.bert.embeddings.position_embeddings(position_ids)

atten_mask = torch.zeros((1,context_length), device=device)

token_embedding = native_model.bert.embeddings.word_embeddings
input_emb = token_embedding(input_ids)

inputs = [
    input_emb.half().cuda(), 
    token_type_embeddings.half().cuda(), 
    position_embeddings.half().cuda(), 
    atten_mask.half().cuda()
]

session = HMONNXGoldenInference(out_hmonnx_file)
session.to(device)
session.save_golden = False
session.golden_dir = work_dirs + "/hmonnx/golden"
session.step = 0
hm_out = session(*inputs)

print(torch.cosine_similarity(output.logits, hm_out))