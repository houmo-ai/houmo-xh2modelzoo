# Copyright 2025 HOUMO AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for event data classes."""

import torch

from events import (
    AudioChunk,
    Code2WavInput,
    StreamEvent,
    TalkerInputDecode,
    TalkerInputPrefill,
    TalkerStepOutput,
    ThinkerChunk,
    ThinkerDecodeChunk,
    ThinkerPrefillChunk,
)


class TestThinkerPrefillChunk:
    def test_create(self):
        chunk = ThinkerPrefillChunk(
            token_ids=torch.tensor([[1, 2, 3]]),
            step_embeds=[torch.randn(1, 1, 64) for _ in range(3)],
            step_hiddens=[torch.randn(1, 1, 64) for _ in range(3)],
        )
        assert chunk.token_ids.shape == (1, 3)
        assert len(chunk.step_embeds) == 3
        assert len(chunk.step_hiddens) == 3
        assert chunk.is_finished is False
        assert chunk.speaker_id == 0

    def test_with_tts_embeddings(self):
        chunk = ThinkerPrefillChunk(
            token_ids=torch.tensor([[1, 2]]),
            step_embeds=[torch.randn(1, 1, 64), torch.randn(1, 1, 64)],
            step_hiddens=[torch.randn(1, 1, 64), torch.randn(1, 1, 64)],
            tts_bos_embed=torch.randn(1, 1, 64),
            tts_eos_embed=torch.randn(1, 1, 64),
            tts_pad_embed=torch.randn(1, 1, 64),
        )
        assert chunk.tts_bos_embed is not None
        assert chunk.tts_eos_embed is not None
        assert chunk.tts_pad_embed is not None


class TestThinkerDecodeChunk:
    def test_create(self):
        chunk = ThinkerDecodeChunk(
            token_id=torch.tensor([[42]]),
            step_embeds=torch.randn(1, 1, 64),
            step_hidden=torch.randn(1, 1, 64),
        )
        assert chunk.token_id.item() == 42
        assert chunk.is_finished is False

    def test_finished(self):
        chunk = ThinkerDecodeChunk(
            token_id=torch.tensor([[0]]),
            is_finished=True,
        )
        assert chunk.is_finished is True


class TestTalkerInputPrefill:
    def test_create(self):
        inp = TalkerInputPrefill(
            hidden_state=torch.randn(1, 10, 64),
            role_mask=torch.ones(1, 10, 1),
            bypass_embeds=torch.randn(1, 10, 64),
            bypass_mask=torch.ones(1, 10, 1),
        )
        assert inp.hidden_state.shape == (1, 10, 64)
        assert inp.is_finished is False


class TestTalkerInputDecode:
    def test_create(self):
        inp = TalkerInputDecode(
            codec_embedding=torch.randn(1, 1, 64),
            generation_step=0,
        )
        assert inp.codec_embedding.shape == (1, 1, 64)
        assert inp.generation_step == 0


class TestTalkerStepOutput:
    def test_create(self):
        out = TalkerStepOutput(
            residual_codes=torch.tensor([[1, 2, 3, 4, 5]]),
            step=0,
        )
        assert out.residual_codes.shape == (1, 5)
        assert out.is_finished is False


class TestCode2WavInput:
    def test_create(self):
        inp = Code2WavInput(
            codes=torch.randint(0, 100, (16, 25)),
            left_context_size=25,
        )
        assert inp.codes.shape == (16, 25)
        assert inp.left_context_size == 25


class TestAudioChunk:
    def test_create(self):
        chunk = AudioChunk(
            audio=torch.randn(1, 24000),
            chunk_index=0,
        )
        assert chunk.audio.shape == (1, 24000)
        assert chunk.sample_rate == 24000


class TestStreamEvent:
    def test_thinker_token(self):
        ev = StreamEvent.thinker_token(token_id=42, step=0)
        assert ev.type == "thinker_token"
        assert ev.data["token_id"] == 42
        assert ev.data["step"] == 0

    def test_audio_chunk(self):
        ev = StreamEvent.audio_chunk(torch.randn(1, 100), chunk_index=0)
        assert ev.type == "audio_chunk"
        assert ev.data["chunk_index"] == 0

    def test_complete(self):
        ev = StreamEvent.complete(
            text_ids=torch.tensor([[1, 2, 3]]),
            audio=torch.randn(1, 100),
        )
        assert ev.type == "complete"
        assert "text_ids" in ev.data
        assert "audio" in ev.data

    def test_error(self):
        exc = ValueError("test error")
        ev = StreamEvent.error(exc)
        assert ev.type == "error"
        assert ev.data["error"] is exc
