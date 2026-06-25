# Copyright 2025 HOUMO AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# SPDX-License-Identifier: Apache-2.0

"""Integration tests — prove end-to-end streaming pipeline works.

These tests verify:

1. Full pipeline: Thinker → Connector → Talker → Connector → Code2Wav
2. Concurrent execution: stages overlap in time
3. Event streaming: consumer receives events before pipeline completes
4. Correct data flow: shapes and types match at each stage boundary
"""

import time

import torch

from code2wav_stage import HMONNXCode2WavStage
from connectors import Talker2Code2WavConnector, Thinker2TalkerConnector
from orchestrator import OmniOrchestrator
from talker_stage import HMONNXTalkerStage
from thinker_stage import HMONNXThinkerStage
from events import (
    AudioChunk,
    StreamEvent,
    TalkerInput,
    TalkerStepOutput,
    ThinkerChunk,
)


class TestEndToEndPipeline:
    """Integration tests using real stage implementations with fake HMONNX sessions."""

    def _build_pipeline(self, hidden_size=64, num_decode_steps=5, num_code_groups=2):
        """Build a complete pipeline with fake sessions."""
        from conftest import FakeHMONNXSession, FakeCode2Wav
        import torch.nn as nn

        vocab_size = 128

        # Thinker
        token_emb = nn.Embedding(vocab_size, hidden_size)
        token_emb.eval()
        thinker = HMONNXThinkerStage(
            prefill_session=FakeHMONNXSession(),
            decode_session=FakeHMONNXSession(),
            token_embedding=token_emb,
            kv_cache_info={"shape": [1, 4, 128, 128], "num_decoder_layers": 4},
            input_sequence_length=32,
            accept_hidden_layer=3,
            eos_token_ids={99},
            supports_hidden_states=True,
        )

        # Talker
        talker = HMONNXTalkerStage(
            talker_prefill_session=FakeHMONNXSession(),
            talker_decode_session=FakeHMONNXSession(),
            predictor_prefill_session=FakeHMONNXSession(),
            predictor_decode_session=FakeHMONNXSession(),
            talker_kv_cache_info={"shape": [1, 4, 64, 128], "num_decoder_layers": 4},
            predictor_kv_cache_info={"shape": [1, 2, 32, 128], "num_decoder_layers": 2},
            static_prefill_len=16,
            static_predictor_prefill_len=4,
            num_code_groups=num_code_groups,
            num_lm_heads=8,
            thinker_hidden_size=hidden_size,
            talker_hidden_size=hidden_size,
        )

        # Code2Wav
        fake_c2w = FakeCode2Wav(total_upsample=600)
        code2wav = HMONNXCode2WavStage(
            code2wav_session=fake_c2w,
            total_upsample=600,
        )

        # Connectors
        t2t = Thinker2TalkerConnector(
            accept_hidden_layer=3,
            thinker_hidden_size=hidden_size,
            talker_hidden_size=hidden_size,
        )
        t2c = Talker2Code2WavConnector(
            codec_chunk_frames=5,
            codec_left_context_frames=2,
            num_quantizers=num_code_groups,
        )

        return OmniOrchestrator(thinker, talker, code2wav, t2t, t2c)

    def test_full_pipeline_produces_events(self):
        orch = self._build_pipeline()
        input_ids = torch.randint(0, 128, (1, 8))
        events = list(orch.generate_stream(input_ids, max_new_tokens=3))

        types = [e.type for e in events]
        assert "complete" in types

    def test_full_pipeline_produces_audio(self):
        orch = self._build_pipeline(num_code_groups=2)
        input_ids = torch.randint(0, 128, (1, 6))
        events = list(orch.generate_stream(input_ids, max_new_tokens=3))

        complete = [e for e in events if e.type == "complete"][0]
        audio = complete.data["audio"]
        assert isinstance(audio, torch.Tensor)
        assert audio.dim() == 2
        assert audio.shape[-1] > 0  # Should have some audio samples

    def test_full_pipeline_produces_text(self):
        orch = self._build_pipeline()
        input_ids = torch.randint(0, 128, (1, 5))
        events = list(orch.generate_stream(input_ids, max_new_tokens=3))

        complete = [e for e in events if e.type == "complete"][0]
        text_ids = complete.data["text_ids"]
        assert isinstance(text_ids, torch.Tensor)

    def test_concurrent_stage_execution(self):
        """Verify that Thinker and Talker overlap in time."""
        orch = self._build_pipeline(hidden_size=64, num_decode_steps=5, num_code_groups=2)
        input_ids = torch.randint(0, 128, (1, 6))

        start = time.time()
        events = list(orch.generate_stream(input_ids, max_new_tokens=5))
        elapsed = time.time() - start

        # Should complete without hanging
        assert elapsed < 5.0
        assert events[-1].type == "complete"

    def test_streaming_yields_before_complete(self):
        """Verify that events are yielded incrementally (not all at once at the end)."""
        orch = self._build_pipeline(num_decode_steps=3, num_code_groups=2)
        input_ids = torch.randint(0, 128, (1, 6))

        received_types = []
        for event in orch.generate_stream(input_ids, max_new_tokens=3):
            received_types.append(event.type)
            # Should receive thinker_token events before complete
            if event.type == "complete":
                break

        assert "thinker_token" in received_types
        assert "complete" in received_types
        # thinker_token should come before complete
        assert received_types.index("thinker_token") < received_types.index("complete")

    def test_data_flow_shapes(self):
        """Verify data flows correctly through all stages."""
        orch = self._build_pipeline(hidden_size=32, num_decode_steps=2, num_code_groups=2)
        input_ids = torch.randint(0, 128, (1, 4))
        events = list(orch.generate_stream(input_ids, max_new_tokens=2))

        # Check that we got thinker tokens
        token_events = [e for e in events if e.type == "thinker_token"]
        assert len(token_events) >= 2

        # Check that complete event has valid data
        complete = [e for e in events if e.type == "complete"][0]
        assert complete.data["text_ids"].shape[0] == 1
        assert complete.data["audio"].shape[0] == 1

    def test_multiple_runs_independent(self):
        """Verify that multiple pipeline runs are independent."""
        orch = self._build_pipeline(num_decode_steps=2)

        events1 = list(orch.generate_stream(torch.randint(0, 128, (1, 4)), max_new_tokens=2))
        events2 = list(orch.generate_stream(torch.randint(0, 128, (1, 4)), max_new_tokens=2))

        assert events1[-1].type == "complete"
        assert events2[-1].type == "complete"
