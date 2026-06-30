# Copyright 2025 HOUMO AI
#
# File: quantizer_utils.py
# Description:
#   Quantization utilities for QUAROT.
#   This module provides functions for applying rotation-based
#   quantization to models.
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
import os

import torch
import yaml
from attr import has
from xhquant.api import Config, get_root_logger

from . import data_utils, gptq_utils, rotation_utils, utils
import json
from transformers import AutoTokenizer
from tqdm import tqdm

@torch.no_grad()
def quarot(model, rotate_mode="hadamard", device=None, quarot_matrix_size=None, hadamard_block_size=None):
    # if device is None:
    #     device = model.device
    # else:
    #     model = model.to(device)

    utils.cleanup_memory(verbose=True)

    # fuse layernorm
    rotation_utils.fuse_layer_norms(model)

    # rotation
    if hadamard_block_size is not None and hadamard_block_size > 0:
        rotation_utils.rotate_model_blockwise(model, block_size=hadamard_block_size, device=device)
    else:
        rotation_utils.rotate_model(model, rotate_mode, device=device, quarot_matrix_size=quarot_matrix_size)

    utils.cleanup_memory(verbose=True)
    # model.to(device)
    return model


@torch.no_grad()
def gptq(
    model,
    args=None,
    model_name=None,
    calib_dataset="wikitext2",
    calib_samples=128,
    seqlen=2048,
    w_clip=True,
    w_bits=4,
    w_head_bits=8,
    w_asym=False,
    w_groupsize=64,
    percdamp=0.01,
    act_order=False,
    int8_down_proj=False,
    heading_gptq=True,
    device=torch.device("cuda"),
    cache_dir=None,
    layers_cache_dir=None,
    is_qwen2_5_vl=False,
    is_qwen3_vl=False,
    processor=None,
    data_files=None,
    is_moe=False,
    use_hession_mse=False,
    self_attn_weight=None,
):
    yaml_dict = dict()
    if args is not None and hasattr(args, "gptq_config"):
        if os.path.isfile(args.gptq_config):
            with open(args.gptq_config, "r", encoding="utf-8") as fin:
                yaml_dict = yaml.load(fin, Loader=yaml.FullLoader)

        if "calib_dataset" in yaml_dict:
            calib_dataset = yaml_dict["calib_dataset"]
        if "calib_samples" in yaml_dict:
            calib_samples = yaml_dict["calib_samples"]
        if "seqlen" in yaml_dict:
            seqlen = yaml_dict["seqlen"]
        if "w_clip" in yaml_dict:
            w_clip = yaml_dict["w_clip"]
        if "w_bits" in yaml_dict:
            w_bits = yaml_dict["w_bits"]
        if "w_groupsize" in yaml_dict:
            w_groupsize = yaml_dict["w_groupsize"]
        if "w_asym" in yaml_dict:
            w_asym = yaml_dict["w_asym"]
        if "percdamp" in yaml_dict:
            percdamp = yaml_dict["percdamp"]
        if "act_order" in yaml_dict:
            act_order = yaml_dict["act_order"]
        if "int8_down_proj" in yaml_dict:
            int8_down_proj = yaml_dict["int8_down_proj"]
        if "heading_gptq" in yaml_dict:
            heading_gptq = yaml_dict["heading_gptq"]
        if "w_head_bits" in yaml_dict:
            w_head_bits = yaml_dict["w_head_bits"]

    if model_name is None and args is not None and has(args, "model"):
        model_name = args.model

    logger = get_root_logger()
    logger.info("***************** GPTQ *****************")
    logger.info(f"Calibrating with {calib_dataset} dataset")
    gptq_cfg = Config(
        dict(
            w_clip=w_clip,
            w_bits=w_bits,
            w_head_bits=w_head_bits,
            w_asym=w_asym,
            w_groupsize=w_groupsize,
            percdamp=percdamp,
            act_order=act_order,
            int8_down_proj=int8_down_proj,
            heading_gptq=heading_gptq,
            seqlen=seqlen,
            calib_dataset=calib_dataset,
        )
    )
    logger.info(f"gptq config:\n{gptq_cfg.pretty_text}")

    if calib_dataset not in ['wikitext2','c4','ptb','laion/220k-GPT4Vision-captions-from-LIVIS','vllm_custom_data'] and 'rerank' not in calib_dataset.lower():
        print('use gen calib data!')
        dataset = []
        cnt = 0
        with open(calib_dataset,encoding='utf-8') as file:
            for line in file:
                dataset.append(json.loads(line))
                cnt = cnt + 1
                if cnt==calib_samples:
                    break
        trainloader = torch.utils.data.DataLoader(dataset, batch_size=1,shuffle=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        gptq_utils.gptq_fwrd(
            model,
            trainloader,
            nsamples=calib_samples,
            seqlen=seqlen,
            w_clip=w_clip,
            w_bits=w_bits,
            w_head_bits=w_head_bits,
            w_asym=w_asym,
            w_groupsize=w_groupsize,
            percdamp=percdamp,
            act_order=act_order,
            int8_down_proj=int8_down_proj,
            heading_gptq=heading_gptq,
            device=device,
            layers_cache_dir=layers_cache_dir,
            is_qwen2_5_vl=is_qwen2_5_vl,
            processor=processor,
            is_qwen3_vl=is_qwen3_vl,
            is_moe=is_moe,
            use_hession_mse=use_hession_mse,
            tokenizer = tokenizer,
        )
    elif  'rerank' in calib_dataset.lower():
        from datasets import load_dataset
        import random
        dataset = load_dataset('parquet', data_files={"test": calib_dataset})["test"]
        dataset_shuffled = dataset.shuffle(seed=42)
        subset = dataset_shuffled.select(range(calib_samples))
        prompt_template = "查询: {query}\n文档: {text}\n相关性:"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        calib_datasets =  []
        skip_count = 0 
        for sample in tqdm(subset):
            query = sample['query']
            pos_docs = sample['positive'] 
            neg_docs = sample['negative']
            if not pos_docs or (isinstance(pos_docs, list) and len(pos_docs) == 0):
                skip_count += 1
                if skip_count <= 3:
                    print(f"\n[Warning] 跳过一条无正例数据的样本")
                continue 
            true_doc = pos_docs[0] if isinstance(pos_docs, list) else pos_docs
            if not neg_docs:
                neg_docs = []
            candidates = [true_doc] + (neg_docs if isinstance(neg_docs, list) else [neg_docs])
            pairs = [[query, doc] for doc in candidates]
            random_pair = random.choice(pairs)
            selected_pair = prompt_template.format(query=random_pair[0], text=random_pair[1])
            format_text = {'text':selected_pair}
            calib_datasets.append(format_text)
        trainloader = torch.utils.data.DataLoader(calib_datasets, batch_size=1,shuffle=True)
        gptq_utils.gptq_fwrd(
            model,
            trainloader,
            nsamples=calib_samples,
            seqlen=seqlen,
            w_clip=w_clip,
            w_bits=w_bits,
            w_head_bits=w_head_bits,
            w_asym=w_asym,
            w_groupsize=w_groupsize,
            percdamp=percdamp,
            act_order=act_order,
            int8_down_proj=int8_down_proj,
            heading_gptq=heading_gptq,
            device=device,
            layers_cache_dir=layers_cache_dir,
            is_qwen2_5_vl=is_qwen2_5_vl,
            processor=processor,
            is_qwen3_vl=is_qwen3_vl,
            is_moe=is_moe,
            use_hession_mse=use_hession_mse,
            tokenizer = tokenizer,
        )
    else:
        trainloader = data_utils.get_loaders(
            calib_dataset,
            nsamples=calib_samples,
            seed=0,
            model=model_name,
            seqlen=seqlen,
            eval_mode=False,
            cache_dir=cache_dir,
            data_files=data_files,
        )
        gptq_utils.gptq_fwrd(
            model,
            trainloader,
            nsamples=calib_samples,
            seqlen=seqlen,
            w_clip=w_clip,
            w_bits=w_bits,
            w_head_bits=w_head_bits,
            w_asym=w_asym,
            w_groupsize=w_groupsize,
            percdamp=percdamp,
            act_order=act_order,
            int8_down_proj=int8_down_proj,
            heading_gptq=heading_gptq,
            device=device,
            layers_cache_dir=layers_cache_dir,
            is_qwen2_5_vl=is_qwen2_5_vl,
            processor=processor,
            is_qwen3_vl=is_qwen3_vl,
            is_moe=is_moe,
            use_hession_mse=use_hession_mse,
            self_attn_weight = self_attn_weight
        )
    return model
