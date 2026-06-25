# Copyright 2025 HOUMO AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for HMONNXCode2WavStage."""

import torch

from code2wav_stage import HMONNXCode2WavStage
from events import AudioChunk, Code2WavInput


class TestHMONNXCode2WavStage:
    def _make_stage(self):
        """Create a Code2Wav stage with a fake session."""
        from conftest import FakeCode2Wav

        fake = FakeCode2Wav(total_upsample=600)
        return HMONNXCode2WavStage(
            code2wav_session=fake,
            total_upsample=600,
        )

    def test_single_chunk(self):
        stage = self._make_stage()
        inp = Code2WavInput(
            codes=torch.randint(0, 100, (16, 10)),
            left_context_size=0,
            is_finished=False,
        )
        chunks = list(stage.run(iter([inp])))
        assert len(chunks) == 1
        assert isinstance(chunks[0], AudioChunk)
        assert chunks[0].audio.dim() == 2
        assert chunks[0].chunk_index == 0

    def test_multiple_chunks(self):
        stage = self._make_stage()
        inputs = [
            Code2WavInput(codes=torch.randint(0, 100, (16, 10)), left_context_size=0),
            Code2WavInput(codes=torch.randint(0, 100, (16, 10)), left_context_size=0),
            Code2WavInput(codes=torch.randint(0, 100, (16, 10)), left_context_size=0, is_finished=True),
        ]
        chunks = list(stage.run(iter(inputs)))
        assert len(chunks) == 3
        assert chunks[0].chunk_index == 0
        assert chunks[1].chunk_index == 1
        assert chunks[2].chunk_index == 2
        assert chunks[2].is_finished is True

    def test_left_context_stripped(self):
        stage = self._make_stage()
        inp = Code2WavInput(
            codes=torch.randint(0, 100, (16, 10)),
            left_context_size=5,
        )
        chunks = list(stage.run(iter([inp])))
        assert len(chunks) == 1
        # Audio should be shorter due to left context stripping
        expected_samples = 10 * 600  # code_len * upsample
        actual_samples = chunks[0].audio.shape[-1]
        assert actual_samples < expected_samples

    def test_empty_input(self):
        stage = self._make_stage()
        chunks = list(stage.run(iter([])))
        assert len(chunks) == 0

    def test_reset(self):
        stage = self._make_stage()
        inp = Code2WavInput(codes=torch.randint(0, 100, (16, 5)), is_finished=False)
        list(stage.run(iter([inp])))
        assert stage._chunk_index == 1

        stage.reset()
        assert stage._chunk_index == 0
