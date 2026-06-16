# Copyright 2024-2025 ModelCloud.ai
# Copyright 2024-2025 qubitium@modelcloud.ai
# Contact: qubitium@modelcloud.ai, x.com/qubitium
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
import argparse
import os
import random
from pathlib import Path

import torch
from datasets import load_dataset
from gptqmodel import GPTQModel, QuantizeConfig

# from gptqmodel.utils import Perplexity
from gptqmodel.utils.logger import setup_logger
from tqdm import tqdm
from transformers import AutoTokenizer


logger = setup_logger()


def get_wikitext2(tokenizer, nsamples, seqlen):
    traindata = load_dataset(
        "wikitext",
        "wikitext-2-raw-v1",
        split="train",
        verification_mode="no_checks",
    )
    seed = 0
    trainenc = tokenizer("\n\n".join(traindata["text"]))
    random.seed(seed)
    train_samples = []
    input_ids = trainenc.input_ids
    data_seq_len = len(input_ids)
    nsamples = min(nsamples, data_seq_len // seqlen)

    for i in tqdm(range(nsamples)):
        start = i * seqlen
        end = start + seqlen
        inp = trainenc.input_ids[start:end]
        inp_mask = trainenc["attention_mask"][start:end]
        train_samples.append(
            {
                "input_ids": inp,
                "attention_mask": inp_mask,
            }
        )

    return train_samples


def main(args):
    pretrained_model_id = args.model
    pretrained_model_id = os.path.normpath(pretrained_model_id)
    model_name = Path(pretrained_model_id).name
    bits = args.bits
    group_size = args.group_size
    quantized_model_id = f"{model_name}-{bits}bit-{group_size}g"
    logger.info(f"HF Model: {pretrained_model_id}")
    logger.info(f"Save Model: {quantized_model_id}")
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_id, use_fast=True)

    traindataset = get_wikitext2(tokenizer, nsamples=256, seqlen=2048)

    quantize_config = QuantizeConfig(
        bits=bits,  # quantize model to 4-bit
        group_size=group_size,  # it is recommended to set the value to 128
        sym=True,
        rotation="hadamard",
        device="cuda:0",
        desc_act=False,
        hessian_mse=True,
        mse=2.4,
    )

    # load un-quantized model, the model will always be force loaded into cpu
    model = GPTQModel.load(pretrained_model_id, quantize_config, torch_dtype=torch.float32)

    # calibration_dataset only accepts dict entries for "input_ids" and "attention_mask"
    model.quantize(traindataset)

    # save quantized model using safetensors
    model.save(quantized_model_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model")
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=64)
    args = parser.parse_args()

    main(args)
