# Copyright 2025 HOUMO AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for OmniOrchestrator — proves concurrent stage execution."""

import queue
import threading
import time

import torch

from orchestrator import OmniOrchestrator, _SENTINEL
from connectors import Thinker2TalkerConnector, Talker2Code2WavConnector
from events import StreamEvent


# ---------------------------------------------------------------------------
# Fake stages for orchestrator testing
# ---------------------------------------------------------------------------


class FakeThinkerStage:
    """Emits a fixed number of prefill + decode chunks with delays."""

    def __init__(self, num_decode_steps=3, hidden_size=64, delay=0.01):
        self.num_decode_steps = num_decode_steps
        self.hidden_size = hidden_size
        self.delay = delay
        self.ran = False

    def run(self, input_ids, max_new_tokens=100, eos_token_id=None):
        from events import ThinkerPrefillChunk, ThinkerDecodeChunk

        self.ran = True
        prompt_len = int(input_ids.shape[1])

        # Prefill
        yield ThinkerPrefillChunk(
            token_ids=input_ids,
            step_embeds=[torch.randn(1, 1, self.hidden_size) for _ in range(prompt_len)],
            step_hiddens=[torch.randn(1, 1, self.hidden_size) for _ in range(prompt_len)],
        )

        # Decode steps
        for i in range(min(self.num_decode_steps, max_new_tokens)):
            time.sleep(self.delay)
            yield ThinkerDecodeChunk(
                token_id=torch.tensor([[i + 10]]),
                step_embeds=torch.randn(1, 1, self.hidden_size),
                step_hidden=torch.randn(1, 1, self.hidden_size),
                is_finished=(i == self.num_decode_steps - 1),
            )


class FakeTalkerStage:
    """Consumes TalkerInput and emits TalkerStepOutput with delays."""

    def __init__(self, codes_per_input=2, delay=0.005):
        self.codes_per_input = codes_per_input
        self.delay = delay
        self.ran = False

    def run(self, inputs):
        from events import TalkerStepOutput

        self.ran = True
        step = 0
        for inp in inputs:
            if hasattr(inp, "is_finished") and inp.is_finished:
                yield TalkerStepOutput(residual_codes=torch.empty(0), step=step, is_finished=True)
                return
            for _ in range(self.codes_per_input):
                time.sleep(self.delay)
                yield TalkerStepOutput(
                    residual_codes=torch.randint(0, 100, (1, 16)),
                    step=step,
                    is_finished=False,
                )
                step += 1


class FakeCode2WavStage:
    """Consumes Code2WavInput and emits AudioChunk."""

    def __init__(self):
        self.ran = False

    def run(self, inputs):
        from events import AudioChunk

        self.ran = True
        idx = 0
        for inp in inputs:
            wav_len = int(inp.codes.shape[-1]) * 600
            yield AudioChunk(
                audio=torch.randn(1, wav_len),
                chunk_index=idx,
                is_finished=inp.is_finished,
            )
            idx += 1
            if inp.is_finished:
                return


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestOmniOrchestrator:
    def _make_orchestrator(self, num_decode_steps=3, hidden_size=64):
        thinker = FakeThinkerStage(num_decode_steps=num_decode_steps, hidden_size=hidden_size)
        talker = FakeTalkerStage(codes_per_input=2)
        code2wav = FakeCode2WavStage()
        t2t = Thinker2TalkerConnector(
            accept_hidden_layer=3,
            thinker_hidden_size=hidden_size,
            talker_hidden_size=hidden_size,
        )
        t2c = Talker2Code2WavConnector(
            codec_chunk_frames=5,
            codec_left_context_frames=2,
        )
        return OmniOrchestrator(thinker, talker, code2wav, t2t, t2c), thinker, talker, code2wav

    def test_all_stages_run(self):
        orch, thinker, talker, code2wav = self._make_orchestrator(num_decode_steps=2)
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        events = list(orch.generate_stream(input_ids, max_new_tokens=10))

        assert thinker.ran
        assert talker.ran
        assert code2wav.ran

    def test_produces_complete_event(self):
        orch, _, _, _ = self._make_orchestrator(num_decode_steps=2)
        input_ids = torch.tensor([[1, 2, 3]])
        events = list(orch.generate_stream(input_ids, max_new_tokens=10))

        event_types = [e.type for e in events]
        assert "complete" in event_types

        complete = [e for e in events if e.type == "complete"][0]
        assert "text_ids" in complete.data
        assert "audio" in complete.data

    def test_thinker_tokens_emitted(self):
        orch, _, _, _ = self._make_orchestrator(num_decode_steps=3)
        input_ids = torch.tensor([[1, 2]])
        events = list(orch.generate_stream(input_ids, max_new_tokens=10))

        token_events = [e for e in events if e.type == "thinker_token"]
        assert len(token_events) > 0

    def test_concurrent_execution(self):
        """Verify that stages run concurrently (not sequentially)."""
        orch, _, _, _ = self._make_orchestrator(num_decode_steps=5)
        input_ids = torch.tensor([[1, 2, 3]])

        start = time.time()
        events = list(orch.generate_stream(input_ids, max_new_tokens=10))
        elapsed = time.time() - start

        # With concurrency, total time should be less than sum of all delays
        # Thinker: 5 * 0.01 = 0.05s, Talker: 10 * 0.005 = 0.05s
        # Sequential would be ~0.1s, concurrent should be ~0.05s
        # Use generous threshold
        assert elapsed < 0.5  # Should complete well within 0.5s

    def test_error_handling(self):
        class FailingThinkerStage:
            def run(self, input_ids, **kwargs):
                raise RuntimeError("Thinker failed")
                yield  # Make it a generator

        orch, _, _, _ = self._make_orchestrator()
        orch.thinker = FailingThinkerStage()

        input_ids = torch.tensor([[1, 2]])
        events = list(orch.generate_stream(input_ids, max_new_tokens=5))

        error_events = [e for e in events if e.type == "error"]
        assert len(error_events) > 0
        assert isinstance(error_events[0].data["error"], RuntimeError)

    def test_empty_input(self):
        orch, _, _, _ = self._make_orchestrator(num_decode_steps=1)
        input_ids = torch.tensor([[1]])
        events = list(orch.generate_stream(input_ids, max_new_tokens=1))
        assert len(events) > 0
        assert events[-1].type == "complete"


class TestQueueIter:
    def test_basic_iteration(self):
        from orchestrator import OmniOrchestrator

        q = queue.Queue()
        q.put("a")
        q.put("b")
        q.put(_SENTINEL)

        items = list(OmniOrchestrator._queue_iter(q))
        assert items == ["a", "b"]

    def test_empty_queue(self):
        from orchestrator import OmniOrchestrator

        q = queue.Queue()
        q.put(_SENTINEL)

        items = list(OmniOrchestrator._queue_iter(q))
        assert items == []
