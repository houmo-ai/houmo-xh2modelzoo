# Copyright 2025 HOUMO AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for HMONNXThinkerStage."""

import torch

from thinker_stage import HMONNXThinkerStage
from events import ThinkerDecodeChunk, ThinkerPrefillChunk


class TestHMONNXThinkerStage:
    def _make_stage(self, hidden_size=64, vocab_size=128, max_seq=64):
        """Create a ThinkerStage with fake sessions."""
        from conftest import FakeHMONNXSession

        kv_info = {"shape": [1, 4, max_seq, 128], "num_decoder_layers": 4}
        token_emb = torch.nn.Embedding(vocab_size, hidden_size)
        token_emb.eval()

        return HMONNXThinkerStage(
            prefill_session=FakeHMONNXSession(),
            decode_session=FakeHMONNXSession(),
            token_embedding=token_emb,
            kv_cache_info=kv_info,
            input_sequence_length=32,
            accept_hidden_layer=3,
            eos_token_ids={99},
            supports_position_ids=False,
            supports_deepstack=False,
            supports_hidden_states=True,
            device=torch.device("cpu"),
        )

    def test_yields_prefill_then_decode(self):
        stage = self._make_stage()
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        chunks = list(stage.run(input_ids, max_new_tokens=3))

        assert len(chunks) >= 2
        assert isinstance(chunks[0], ThinkerPrefillChunk)
        assert chunks[0].token_ids.shape == (1, 5)
        assert len(chunks[0].step_embeds) == 5
        assert len(chunks[0].step_hiddens) == 5

        # Remaining chunks are decode chunks
        for chunk in chunks[1:]:
            assert isinstance(chunk, ThinkerDecodeChunk)

    def test_prefill_chunk_has_correct_shapes(self):
        stage = self._make_stage(hidden_size=64)
        input_ids = torch.tensor([[10, 20, 30]])
        chunks = list(stage.run(input_ids, max_new_tokens=2))

        prefill = chunks[0]
        assert isinstance(prefill, ThinkerPrefillChunk)
        assert prefill.token_ids.shape == (1, 3)
        assert all(e.shape == (1, 1, 64) for e in prefill.step_embeds)
        assert all(h.shape == (1, 1, 64) for h in prefill.step_hiddens)

    def test_eos_stops_generation(self):
        stage = self._make_stage(vocab_size=200, max_seq=64)
        stage.eos_token_ids = {50}
        input_ids = torch.tensor([[1, 2, 3]])
        # Run with max_new_tokens=10 but EOS should stop early
        chunks = list(stage.run(input_ids, max_new_tokens=10))
        # Should have prefill + at least 1 decode + final finished
        assert len(chunks) >= 2
        # Last chunk should be finished
        assert chunks[-1].is_finished is True

    def test_generate_with_custom_eos(self):
        stage = self._make_stage(vocab_size=200, max_seq=64)
        input_ids = torch.tensor([[1, 2]])
        chunks = list(stage.run(input_ids, max_new_tokens=5, eos_token_id={50}))
        assert len(chunks) >= 2

    def test_reset_between_runs(self):
        stage = self._make_stage()
        input_ids = torch.tensor([[1, 2, 3]])
        # Run twice — should work independently
        chunks1 = list(stage.run(input_ids, max_new_tokens=2))
        chunks2 = list(stage.run(input_ids, max_new_tokens=2))
        assert len(chunks1) >= 2
        assert len(chunks2) >= 2
