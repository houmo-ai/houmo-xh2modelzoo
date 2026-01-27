# Copyright 2025 HOUMO AI
#
# File: wenet_hmonnx_test.py
# Description:
#   Example script: audio/wenet/fix_length/wenet_hmonnx_test.py
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
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import yaml
from xhquant.api import HMONNXInference, get_root_logger, xhquant_init


class DecodeResult:

    def __init__(
        self,
        tokens: List[int],
        score: float = 0.0,
        confidence: float = 0.0,
        tokens_confidence: List[float] = None,
        times: List[int] = None,
        nbest: List[List[int]] = None,
        nbest_scores: List[float] = None,
        nbest_times: List[List[int]] = None,
    ):
        """
        Args:
            tokens: decode token list
            score: the total decode score of this result
            confidence: the total confidence of this result, it's in 0~1
            tokens_confidence: confidence of each token
            times: timestamp of each token, list of (start, end)
            nbest: nbest result
            nbest_scores: score of each nbest
            nbest_times:
        """
        self.tokens = tokens
        self.score = score
        self.confidence = confidence
        self.tokens_confidence = tokens_confidence
        self.times = times
        self.nbest = nbest
        self.nbest_scores = nbest_scores
        self.nbest_times = nbest_times


def remove_duplicates_and_blank(hyp: List[int], blank_id: int = 0) -> List[int]:
    new_hyp: List[int] = []
    cur = 0
    while cur < len(hyp):
        if hyp[cur] != blank_id:
            new_hyp.append(hyp[cur])
        prev = cur
        while cur < len(hyp) and hyp[cur] == hyp[prev]:
            cur += 1
    return new_hyp


def make_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    """Make mask tensor containing indices of padded part.

    See description of make_non_pad_mask.

    Args:
        lengths (torch.Tensor): Batch of lengths (B,).
    Returns:
        torch.Tensor: Mask tensor containing indices of padded part.

    Examples:
        >>> lengths = [5, 3, 2]
        >>> make_pad_mask(lengths)
        masks = [[0, 0, 0, 0 ,0],
                 [0, 0, 0, 1, 1],
                 [0, 0, 1, 1, 1]]
    """
    batch_size = lengths.size(0)
    max_len = max_len if max_len > 0 else lengths.max().item()
    seq_range = torch.arange(0, max_len, dtype=torch.int64, device=lengths.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_length_expand = lengths.unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand
    return mask


def ctc_greedy_search(ctc_probs: torch.Tensor, ctc_lens: torch.Tensor, blank_id: int = 0):
    batch_size = ctc_probs.shape[0]
    maxlen = ctc_probs.size(1)
    topk_prob, topk_index = ctc_probs.topk(1, dim=2)  # (B, maxlen, 1)
    topk_index = topk_index.view(batch_size, maxlen)  # (B, maxlen)
    mask = make_pad_mask(ctc_lens, maxlen)  # (B, maxlen)
    topk_index = topk_index.masked_fill_(mask, blank_id)  # (B, maxlen)
    hyps = [hyp.tolist() for hyp in topk_index]
    scores = topk_prob.max(1)
    results = []
    for hyp in hyps:
        r = DecodeResult(remove_duplicates_and_blank(hyp, blank_id))
        results.append(r)
    return results


def read_symbol_table(symbol_table_file):
    symbol_table = {}
    with open(symbol_table_file, "r", encoding="utf8") as fin:
        for line in fin:
            arr = line.strip().split()
            assert len(arr) == 2
            symbol_table[arr[0]] = int(arr[1])
    return symbol_table


def main(args):
    xhquant_init(None, debug=args.debug)
    session = HMONNXInference(args.hmonnx)
    exec_device = torch.device("cuda")
    session.to(exec_device)
    logger = get_root_logger()
    logger.info("session is created successfully")
    audio_file = args.audio
    audio_data = torch.load(audio_file)

    input_names = session.get_input_names()

    sample_input = dict()
    for input_name in input_names:
        sample_input[input_name] = audio_data[input_name].to(exec_device).to(torch.float16)
    # inputs = inputs.to(exec_device).to(torch.float16)

    nn_out = session.run(sample_input)

    ctc_output = nn_out[0].log_softmax(dim=-1).cpu()
    ctc_lens = sample_input["mask_cnn"].sum(-1).squeeze(0).int().cpu()
    ck_result = ctc_greedy_search(ctc_output, ctc_lens, 0)
    symbol_table = read_symbol_table("data/models/wenet/units.txt")
    char_dict = {v: k for k, v in symbol_table.items()}
    content = [char_dict[w] for w in ck_result[0].tokens]
    content = "".join(content)
    print(f"asr result: {content}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hmonnx", type=str, default="work_dirs/encoder/hmonnx/encoder_XH2a.onnx")
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument("--audio", type=str, default="data/wenet/encoder/input_0.pth")
    args = parser.parse_args()
    main(args)
