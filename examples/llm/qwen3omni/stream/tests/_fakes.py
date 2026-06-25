# Copyright 2025 HOUMO AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# SPDX-License-Identifier: Apache-2.0

"""Shared test fakes for HMONNX sessions and code2wav."""

from types import SimpleNamespace

import torch


class FakeHMONNXSession:
    """Minimal mock for HMONNXInference sessions."""

    def __init__(self, output_shapes=None, return_logits=True, return_hidden=True):
        self.inputs = [SimpleNamespace(name=f"input_{i}") for i in range(10)]
        self.output_shapes = output_shapes
        self.return_logits = return_logits
        self.return_hidden = return_hidden
        self.call_count = 0

    def forward(self, *args, **kwargs):
        self.call_count += 1
        seq_len = 1
        hidden_size = 64
        for arg in args:
            if isinstance(arg, torch.Tensor) and arg.dim() >= 2:
                seq_len = max(seq_len, int(arg.shape[1]))
                hidden_size = int(arg.shape[-1])
                break

        vocab_size = 128
        outputs = []
        if self.return_logits:
            outputs.append(torch.randn(1, seq_len, vocab_size, dtype=torch.float32))
        if self.return_hidden:
            outputs.append(torch.randn(1, seq_len, hidden_size, dtype=torch.float16))

        if len(outputs) == 1:
            return outputs[0]
        return outputs


class FakeCode2Wav:
    """Mock for code2wav module."""

    def __init__(self, total_upsample=600):
        self.total_upsample = total_upsample
        self.hmonnx = FakeHMONNXSession(return_logits=False, return_hidden=False)
        self.hmonnx_max_code_len = 512

    def forward(self, codes):
        if isinstance(codes, torch.Tensor):
            code_len = int(codes.shape[-1])
        else:
            code_len = 1
        wav_len = code_len * self.total_upsample
        return torch.randn(1, wav_len, dtype=torch.float32)
