import os
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import torch


def prepare_inputs(
    model, tokenizer, past_key_caches, past_value_caches, data: Union[dict, tuple, list], out_padding=False
):
    raw_input_ids: List[List[int]] = data["input_ids"]
    assert model.get_input_embeddings() is not None, "Token embedding is not available."

    device = model.device
    input_ids = []
    current_input_length = []
    position_ids = []
    for batch_idx, input_id in enumerate(raw_input_ids):
        input_id = torch.tensor(input_id, dtype=torch.long)
        seq_length = input_id.shape[0]
        past_seq_length = data["past_seq_length"][batch_idx]
        position_id = torch.arange(
            past_seq_length, past_seq_length + seq_length, dtype=torch.long, device=input_id.device
        )
        current_input_length.append(seq_length)
        assert (
            seq_length <= 2048  # model.input_sequence_length
        ), f"Input sequence length is too long. max input sequence length is 2048 but got {seq_length}"
        if 2048 > seq_length and out_padding:
            padding_input_ids = torch.zeros((2048 - seq_length), dtype=torch.long, device=input_id.device)
            padding_input_ids.fill_(tokenizer.pad_token_id)
            input_id = torch.cat([input_id, padding_input_ids], dim=-1)

            padding_position_id = torch.ones((2048 - seq_length), dtype=torch.long, device=input_id.device)
            position_id = torch.cat([position_id, padding_position_id], dim=-1)

        input_id = input_id.unsqueeze(0)
        position_id = position_id.unsqueeze(0)
        position_ids.append(position_id)
        input_ids.append(input_id)

    input_ids = torch.cat(input_ids, dim=0).to(device)
    position_ids = torch.cat(position_ids, dim=0).to(device)

    current_input_length = torch.tensor(current_input_length, dtype=torch.int32).to(device)

    model.get_input_embeddings().to(device)
    inputs_embeds = model.get_input_embeddings()(input_ids)
    past_seq_length = data["past_seq_length"]
    past_seq_length = torch.tensor(past_seq_length, dtype=torch.int32).to(device)
    assert torch.all(past_seq_length >= 0)
    past_key_caches = past_key_caches
    past_value_caches = past_value_caches

    return (
        inputs_embeds.to(device),
        past_seq_length.to(device),
        current_input_length,
        # position_ids.to(device),
        past_key_caches,
        past_value_caches,
    )


def load_calibration_data(tokenizer, device=torch.device("cuda")):
    from datasets import load_dataset

    wiki_testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    full_input_ids = []
    long_tokens = []
    mid_tokens = []
    short_tokens = []

    for text in wiki_testdata["text"]:
        if len(text) == 0:
            continue
        input_ids = tokenizer(text + "\n\n", return_tensors="pt").input_ids
        if input_ids.shape[-1] >= 800:
            long_tokens.append(input_ids)
        elif input_ids.shape[-1] >= 400:
            mid_tokens.append(input_ids)
        else:
            short_tokens.append(input_ids)
        full_input_ids.append(input_ids)
    n_samples = 6  #
    valid_cnt = n_samples // 6
    state = 0
    from random import shuffle

    shuffle(long_tokens)
    shuffle(mid_tokens)
    shuffle(short_tokens)
    lid = 0
    mid = 0
    sid = 0
    dataloader = []
    ki = 0
    while ki < valid_cnt:
        if state == 0:
            if lid < len(long_tokens):
                dataloader.append({"input_ids": long_tokens[lid].to(device), "past_seq_length": [0]})
                ki += 1
                lid += 1
            state = 1
        elif state == 1:
            if mid < len(mid_tokens):
                dataloader.append({"input_ids": mid_tokens[mid].to(device), "past_seq_length": [0]})
                ki += 1
                mid += 1
            state = 2
        else:
            if sid < len(short_tokens):
                dataloader.append({"input_ids": short_tokens[sid].to(device), "past_seq_length": [0]})
                sid += 1
                ki += 1
            state = 0
    full_input_ids = torch.cat(full_input_ids, dim=-1)
    seq_len = 2048
    for _ in range(max(n_samples - len(dataloader), 1)):
        i = torch.randint(0, full_input_ids.shape[-1] - seq_len - 1, (1,)).item()
        j = i + seq_len
        inp = full_input_ids[0:1, i:j]
        dataloader.append({"input_ids": inp.to(device), "past_seq_length": [0]})
    return dataloader


def ms_data_preprocess(model, tokenizer, past_key_caches, past_value_caches):
    dataloader = load_calibration_data(tokenizer)
    calib_dataloader = []
    label_dataloader = []
    for data in dataloader:
        new_args = []
        label_dataloader.append(data["input_ids"].to("cuda"))
        arg = prepare_inputs(model, tokenizer, past_key_caches, past_value_caches, data)
        for e in arg:
            if isinstance(e, (List, Tuple)):
                new_args.extend(e)
            else:
                new_args.append(e)
        calib_dataloader.append(new_args)
    gpu_loader = []
    for data in calib_dataloader:
        if len(label_dataloader) == 0:
            label_dataloader.append(data["input_ids"].to("cuda"))
        item = []
        for e in data:
            if e.device.type == "cpu":
                item.append(e.to("cuda"))
            else:
                item.append(e)
        gpu_loader.append(item)
    return gpu_loader, label_dataloader
