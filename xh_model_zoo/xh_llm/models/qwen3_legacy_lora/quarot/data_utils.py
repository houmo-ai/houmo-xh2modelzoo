# -*- coding: utf-8 -*-
# Copyright 2024 The QuaRot Authors. All rights reserved.
# Copyright 2025 HOUMO AI. All rights reserved.
#
# Modifications:
# - Portions of this file have been modified by HOUMO AI.
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
# File: data_utils.py
# Description:
#   Qwen3 legacy LoRA QuaRot calibration / dataset utilities adapted for the xh2 model zoo (xh2modelzoo).

import random

import datasets
import transformers
import json


def get_wikitext2(nsamples, seed, seqlen, model, hf_token, eval_mode=False, cache_dir=None):
    if hf_token is None:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model, token=hf_token, trust_remote_code=True)

    if eval_mode:
        testdata = datasets.load_dataset(
            "wikitext",
            "wikitext-2-raw-v1",
            split="test",
            ignore_verifications=True,
            cache_dir=cache_dir,
        )
        testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")
        return testenc
    else:
        traindata = datasets.load_dataset(
            "wikitext",
            "wikitext-2-raw-v1",
            split="train",
            verification_mode="no_checks",
            cache_dir=cache_dir,
        )
        trainenc = tokenizer("\n\n".join(traindata["text"]), return_tensors="pt")
        random.seed(seed)
        trainloader = []
        for _ in range(nsamples):
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
        return trainloader


def get_c4_new(nsamples, seed, seqlen, model, hf_token=None, eval_mode=False, cache_dir=None):
    if hf_token is None:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model, token=hf_token, trust_remote_code=True)

    if eval_mode:
        valdata = datasets.load_dataset(
            "allenai/c4",
            data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
            split="validation",
            cache_dir=cache_dir,
        )
        valenc = tokenizer(" ".join(valdata[:1100]["text"]), return_tensors="pt")
        valenc = valenc.input_ids[:, : (256 * seqlen)]

        class TokenizerWrapper:
            def __init__(self, input_ids):
                self.input_ids = input_ids

        valenc = TokenizerWrapper(valenc)
        return valenc
    else:
        traindata = datasets.load_dataset(
            "allenai/c4",
            data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
            split="train",
            cache_dir=cache_dir,
        )

        random.seed(seed)
        trainloader = []
        for _ in range(nsamples):
            while True:
                i = random.randint(0, len(traindata) - 1)
                trainenc = tokenizer(traindata[i]["text"], return_tensors="pt")
                if trainenc.input_ids.shape[1] >= seqlen:
                    break
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
        return trainloader


def get_ptb_new(nsamples, seed, seqlen, model, hf_token, eval_mode=False, cache_dir=None):
    if hf_token is None:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model, token=hf_token, trust_remote_code=True)

    if eval_mode:
        testdata = datasets.load_dataset(
            "ptb_text_only",
            "penn_treebank",
            split="test",
            cache_dir=cache_dir,
        )
        testenc = tokenizer(" ".join(testdata["sentence"]), return_tensors="pt")
        return testenc
    else:
        traindata = datasets.load_dataset(
            "ptb_text_only",
            "penn_treebank",
            split="train",
            cache_dir=cache_dir,
        )
        trainenc = tokenizer(" ".join(traindata["sentence"]), return_tensors="pt")
        random.seed(seed)
        trainloader = []
        for _ in range(nsamples):
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
        return trainloader


def get_vllm_custom_data(nsamples, seed, seqlen, model, hf_token, data_files, eval_mode=False, cache_dir=None):
    from xhquant_llm.datasets import VLLMCustomDataset
    import numpy as np
    dataset = VLLMCustomDataset(data_files=data_files)
    rng = np.random.default_rng(seed)
    sample_size = min(nsamples, len(dataset))
    sampled_indices = rng.choice(len(dataset), size=sample_size, replace=False)
    return [dataset.data[idx] for idx in sampled_indices]

def get_loaders(
    name,
    nsamples=128,
    seed=0,
    seqlen=2048,
    model="",
    hf_token=None,
    eval_mode=False,
    cache_dir=None,
    **kwargs
):
    if "wikitext2" in name:
        return get_wikitext2(nsamples, seed, seqlen, model, hf_token, eval_mode, cache_dir=cache_dir)
    if "ptb" in name:
        return get_ptb_new(nsamples, seed, seqlen, model, hf_token, eval_mode, cache_dir=cache_dir)
    if "c4" in name:
        return get_c4_new(nsamples, seed, seqlen, model, hf_token, eval_mode, cache_dir=cache_dir)
    if "vllm_custom_data" in name:
        data_files = kwargs.get("data_files", None)
        if data_files is None:
            raise ValueError("data_files is required for vllm_custom_data")
        return get_vllm_custom_data(nsamples, seed, seqlen, model, hf_token, data_files, eval_mode, cache_dir=cache_dir)
    if "merged_gen_data" in name:
        calib_loader =[]
        data_files = kwargs.get("data_files", None)
        if data_files is None:
            raise ValueError("data_files is required for gen by model")
        lines = open(data_files[0],'r').readlines()
        for line in lines:
            temp_res = json.loads(line)
            temp_res[1]['content'] = temp_res[1]['content'][0]
            calib_loader.append(temp_res)
        return calib_loader
            

        
