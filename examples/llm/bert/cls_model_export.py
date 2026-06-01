# Copyright 2025 HOUMO AI
#
# File: model_export.py
# Description:
#   Example script: llm/bert/model_export.py
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

from modelscope import AutoTokenizer, AutoModelForMaskedLM
import torch
from xh_model_zoo.xh_llm.models.bert.bert_converter import BertConverterXH2a
from xhquant.api import DeviceType, xhquant_init, QuantScheme, get_root_logger 
import argparse
from xh_model_zoo.xh_llm.models.qwen2_5_vl import Qwen2_5_VLConvertConfig, VisualConfig
from transformers import BertForSequenceClassification

def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = BertForSequenceClassification.from_pretrained(args.model)
    model = model.to("cuda")

    input_txt = "受害者是几点死亡的，死亡的具体原因是谋杀"
    input_ids = tokenizer(
        input_txt, return_tensors="pt", padding="max_length", max_length=args.context_length
    ).input_ids

    output = model(input_ids.cuda())
    target_device = DeviceType.XH2a

    quant_type = args.quant_type
    # ops=dict(MatMul=dict(
    #             act_scheme=dict(
    #                 bits=8,
    #                 fp_mode="sefp",
    #             ),
    #             act_schema_2=dict(
    #                 bits=16,
    #                 fp_mode="sefp",
    #             ),))

    quant_scheme = QuantScheme(target_device=DeviceType.XH2a, quant_type=quant_type) # , ops=ops

    config = Qwen2_5_VLConvertConfig(
        batch_size=args.batch_size,
        context_length=args.context_length,
        quant_scheme=quant_scheme,
        quant_weight=args.quant_weight,
        gptqmodel_cfg=args.use_gptqmodel,
        max_pe_length=args.max_pe_length,
    )

    BertConverterXH2a(config)._convert(
        model,
        "work_dirs/bert",
        tokenizer,
    )
    # current_logits = output.logits[:, -1, :]
    # probs = torch.softmax(current_logits, dim=-1)
    # next_token = torch.argmax(probs, dim=-1).item()
    # reply = tokenizer.decode(next_token)
    # print(reply)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument("--model", type=str, default="/data02/users/cc_work/model312/BERT")
    parser.add_argument("--batch-size", type=int, default=1, help="batch size")
    parser.add_argument("--context-length", type=int, default=512, help="max sequence length")
    parser.add_argument("--max_pe_length", type=int, default=32768, help="max pe length")
    parser.add_argument("--quant-type", default="w8a8h1_sefp", help="quant type, default is w8a8")
    parser.add_argument("--image_max_size_h", type=int, default=448, help="image max size height")
    parser.add_argument("--image_max_size_w", type=int, default=448, help="image max size width")
    parser.add_argument("--image_max_size_t", type=int, default=2, help="if image, temporal max size is 2, if video, temporal max size is fps")
    parser.add_argument("--patch_size", type=int, default=14, help="patch size")
    parser.add_argument("--temporal_patch_size", type=int, default=2, help="temporal patch size")
    parser.add_argument("--sample_image_path", type=str, default="data/images/qwen2_vl_demo.jpeg", help="sample image path for generate golden")
    parser.add_argument("--use_gptqmodel", action="store_true", help="use gptqmodel quanted model")
    parser.add_argument(
        "--quant_weight",
        type=str,
        default=None,
        help="quant weight path, for example: gptq or quarot, if empty, use w8a8",
    )
    args = parser.parse_args()
    main(args)
