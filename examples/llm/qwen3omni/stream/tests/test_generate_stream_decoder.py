# Copyright 2025 HOUMO AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for official-generate streaming Code2Wav decoding."""

import pytest
import torch

from qwen3_omni_hmonnx_generate_stream import StreamingCode2WavDecoder


class CountingCode2Wav:
    def __init__(self, total_upsample=2):
        self.total_upsample = total_upsample
        self.calls = []

    def forward(self, codes):
        self.calls.append(tuple(codes.shape))
        return torch.ones(1, codes.shape[-1] * self.total_upsample)


def test_decoder_rejects_single_residual_group():
    decoder = StreamingCode2WavDecoder(CountingCode2Wav(), chunk_size=2, num_quantizers=16)

    with pytest.raises(ValueError, match=r"Expected residual code shape \[B,16\]"):
        decoder.add_residual_codes(torch.zeros(1, 1, dtype=torch.long))


def test_decoder_finalizes_only_unemitted_tail():
    code2wav = CountingCode2Wav(total_upsample=2)
    decoder = StreamingCode2WavDecoder(code2wav, chunk_size=2, left_context_size=1, num_quantizers=16)

    decoder.add_residual_codes(torch.zeros(1, 16, dtype=torch.long))
    decoder.add_residual_codes(torch.ones(1, 16, dtype=torch.long))
    decoder.add_residual_codes(torch.full((1, 16), 2, dtype=torch.long))
    audio = decoder.finalize()

    assert code2wav.calls == [(1, 16, 2), (1, 16, 2)]
    assert audio.shape == (1, 6)
