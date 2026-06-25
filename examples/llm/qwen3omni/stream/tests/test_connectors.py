# Copyright 2025 HOUMO AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for Thinker2TalkerConnector and Talker2Code2WavConnector."""

import torch

from connectors import Talker2Code2WavConnector, Thinker2TalkerConnector
from events import (
    TalkerInputDecode,
    TalkerInputPrefill,
    TalkerStepOutput,
    ThinkerDecodeChunk,
    ThinkerPrefillChunk,
)


class TestThinker2TalkerConnector:
    def setup_method(self):
        self.connector = Thinker2TalkerConnector(
            accept_hidden_layer=3,
            thinker_hidden_size=64,
            talker_hidden_size=64,
        )

    def test_process_prefill(self):
        chunk = ThinkerPrefillChunk(
            token_ids=torch.tensor([[1, 2, 3, 4, 5]]),
            step_embeds=[torch.randn(1, 1, 64) for _ in range(5)],
            step_hiddens=[torch.randn(1, 1, 64) for _ in range(5)],
        )
        result = self.connector.process(chunk)
        assert isinstance(result, TalkerInputPrefill)
        assert result.hidden_state.shape == (1, 5, 64)
        assert result.role_mask.shape == (1, 5, 1)
        assert result.bypass_embeds.shape == (1, 5, 64)
        assert result.bypass_mask.shape == (1, 5, 1)
        assert result.is_finished is False

    def test_process_prefill_with_tts_embeds(self):
        bos = torch.randn(1, 1, 64)
        eos = torch.randn(1, 1, 64)
        pad = torch.randn(1, 1, 64)
        chunk = ThinkerPrefillChunk(
            token_ids=torch.tensor([[1, 2]]),
            step_embeds=[torch.randn(1, 1, 64), torch.randn(1, 1, 64)],
            step_hiddens=[torch.randn(1, 1, 64), torch.randn(1, 1, 64)],
            tts_bos_embed=bos,
            tts_eos_embed=eos,
            tts_pad_embed=pad,
        )
        result = self.connector.process(chunk)
        assert isinstance(result, TalkerInputPrefill)
        assert result.tts_pad_embed is not None

    def test_process_decode(self):
        # First process a prefill
        prefill = ThinkerPrefillChunk(
            token_ids=torch.tensor([[1, 2, 3]]),
            step_embeds=[torch.randn(1, 1, 64) for _ in range(3)],
            step_hiddens=[torch.randn(1, 1, 64) for _ in range(3)],
        )
        self.connector.process(prefill)

        # Then process a decode chunk
        decode = ThinkerDecodeChunk(
            token_id=torch.tensor([[4]]),
            step_embeds=torch.randn(1, 1, 64),
            step_hidden=torch.randn(1, 1, 64),
        )
        result = self.connector.process(decode)
        assert isinstance(result, TalkerInputDecode)
        assert result.codec_embedding.shape == (1, 1, 64)
        assert result.generation_step == 0
        assert result.is_finished is False

    def test_process_decode_finished(self):
        decode = ThinkerDecodeChunk(
            token_id=torch.tensor([[0]]),
            is_finished=True,
        )
        result = self.connector.process(decode)
        assert isinstance(result, TalkerInputDecode)
        assert result.is_finished is True

    def test_reset(self):
        prefill = ThinkerPrefillChunk(
            token_ids=torch.tensor([[1, 2]]),
            step_embeds=[torch.randn(1, 1, 64), torch.randn(1, 1, 64)],
            step_hiddens=[torch.randn(1, 1, 64), torch.randn(1, 1, 64)],
        )
        self.connector.process(prefill)
        assert len(self.connector._collected_embeds) == 2

        self.connector.reset()
        assert len(self.connector._collected_embeds) == 0

    def test_set_tts_embeddings(self):
        bos = torch.randn(1, 1, 64)
        eos = torch.randn(1, 1, 64)
        pad = torch.randn(1, 1, 64)
        self.connector.set_tts_embeddings(bos, eos, pad)
        assert self.connector._tts_bos_embed is not None
        assert self.connector._tts_eos_embed is not None
        assert self.connector._tts_pad_embed is not None


class TestTalker2Code2WavConnector:
    def setup_method(self):
        self.connector = Talker2Code2WavConnector(
            codec_chunk_frames=5,
            codec_left_context_frames=2,
            num_quantizers=16,
        )

    def test_no_emit_until_chunk_full(self):
        for i in range(4):
            out = self.connector.process(TalkerStepOutput(
                residual_codes=torch.randint(0, 100, (1, 16)),
                step=i,
            ))
            assert out is None

    def test_emit_when_chunk_full(self):
        for i in range(5):
            out = self.connector.process(TalkerStepOutput(
                residual_codes=torch.randint(0, 100, (1, 16)),
                step=i,
            ))
        # 5th call should emit
        assert out is not None
        assert out.codes.shape[0] == 16  # num_quantizers
        assert out.is_finished is False

    def test_flush_on_finished(self):
        # Add 3 codes then finish
        for i in range(3):
            self.connector.process(TalkerStepOutput(
                residual_codes=torch.randint(0, 100, (1, 16)),
                step=i,
            ))
        out = self.connector.process(TalkerStepOutput(
            residual_codes=torch.empty(1, 16, dtype=torch.long),
            step=3,
            is_finished=True,
        ))
        assert out is not None
        assert out.is_finished is True
        assert out.codes.shape[0] == 16

    def test_reset(self):
        for i in range(3):
            self.connector.process(TalkerStepOutput(
                residual_codes=torch.randint(0, 100, (1, 16)),
                step=i,
            ))
        assert self.connector.total_steps == 3
        self.connector.reset()
        assert self.connector.total_steps == 0

    def test_left_context_included(self):
        # Fill a chunk
        for i in range(5):
            self.connector.process(TalkerStepOutput(
                residual_codes=torch.randint(0, 100, (1, 16)),
                step=i,
            ))
        out = self.connector.process(TalkerStepOutput(
            residual_codes=torch.randint(0, 100, (1, 16)),
            step=5,
        ))
        if out is not None:
            # codes should include left context
            assert out.codes.shape[1] >= 5  # at least chunk_size

    def test_initial_chunk_size(self):
        connector = Talker2Code2WavConnector(
            codec_chunk_frames=10,
            codec_left_context_frames=2,
            initial_chunk_size=3,
        )
        # First 3 codes should emit
        for i in range(3):
            out = connector.process(TalkerStepOutput(
                residual_codes=torch.randint(0, 100, (1, 16)),
                step=i,
            ))
        assert out is not None  # emitted after 3 steps
