# Copyright 2025 HOUMO AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# SPDX-License-Identifier: Apache-2.0

"""Async stage orchestrator — runs Thinker / Talker / Code2Wav concurrently.

The orchestrator mirrors vLLM-Omni's ``AsyncOmni`` + ``Orchestrator`` pattern:
each stage runs in its own thread, communicating via ``queue.Queue``.  Chunks
flow directly from stage to stage through connectors, while the orchestrator
yields unified ``StreamEvent`` objects to the consumer.

Stage execution order::

    Thread 1 (Thinker) ──▶ Connector1 ──▶ Thread 2 (Talker)
                                                      │
                                               Connector2
                                                      │
                                              Thread 3 (Code2Wav)
                                                      │
                                                  Main thread
                                               (yields StreamEvent)
"""

from __future__ import annotations

import logging
import queue
import threading
from typing import Iterator, Optional

import torch

from .code2wav_stage import HMONNXCode2WavStage
from .connectors import Talker2Code2WavConnector, Thinker2TalkerConnector
from .events import AudioChunk, StreamEvent, TalkerInput, ThinkerChunk
from .talker_stage import HMONNXTalkerStage
from .thinker_stage import HMONNXThinkerStage

logger = logging.getLogger(__name__)

_SENTINEL = object()  # poison pill to signal thread completion


class OmniOrchestrator:
    """Run the three-stage pipeline concurrently and yield unified events.

    Parameters
    ----------
    thinker : HMONNXThinkerStage
        Stage 0 – Thinker.
    talker : HMONNXTalkerStage
        Stage 1 – Talker.
    code2wav : HMONNXCode2WavStage
        Stage 2 – Code2Wav.
    thinker2talker : Thinker2TalkerConnector
        Connector 0→1.
    talker2code2wav : Talker2Code2WavConnector
        Connector 1→2.
    """

    def __init__(
        self,
        thinker: HMONNXThinkerStage,
        talker: HMONNXTalkerStage,
        code2wav: HMONNXCode2WavStage,
        thinker2talker: Thinker2TalkerConnector,
        talker2code2wav: Talker2Code2WavConnector,
    ):
        self.thinker = thinker
        self.talker = talker
        self.code2wav = code2wav
        self.thinker2talker = thinker2talker
        self.talker2code2wav = talker2code2wav

    def generate_stream(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 1024,
        eos_token_id: Optional[int | set[int]] = None,
        chunk_size: int = 25,
        left_context_size: int = 25,
    ) -> Iterator[StreamEvent]:
        """Run the full pipeline and yield streaming events.

        This method orchestrates three concurrent threads:

        1. **Thinker thread**: Runs the Thinker stage and feeds chunks to
           the Thinker→Talker connector.
        2. **Talker thread**: Consumes Talker inputs, runs the Talker stage,
           and feeds code chunks to the Talker→Code2Wav connector.
        3. **Code2Wav thread**: Consumes code chunks and emits audio chunks.

        The main thread yields ``StreamEvent`` objects as they arrive.

        Parameters
        ----------
        input_ids : torch.Tensor
            Prompt token IDs, shape ``[1, prompt_len]``.
        max_new_tokens : int
            Maximum Thinker decode steps.
        eos_token_id : int | set[int] | None
            End-of-sequence token IDs.
        chunk_size : int
            Codec frames per audio chunk.
        left_context_size : int
            Overlap frames for smooth audio boundaries.

        Yields
        ------
        StreamEvent
            Streaming events (thinker_token, audio_chunk, complete, error).
        """
        # Reset all stages and connectors
        self.thinker2talker.reset()
        self.talker2code2wav.reset()
        if hasattr(self.talker, "reset"):
            self.talker.reset()
        if hasattr(self.code2wav, "reset"):
            self.code2wav.reset()

        # Queues for inter-stage communication
        thinker_out_q: queue.Queue = queue.Queue()
        talker_in_q: queue.Queue = queue.Queue()
        talker_out_q: queue.Queue = queue.Queue()
        audio_out_q: queue.Queue = queue.Queue()
        error_q: queue.Queue = queue.Queue()

        all_tokens: list[int] = []
        all_audio: list[torch.Tensor] = []
        lock = threading.Lock()

        # ---- Thread 1: Thinker → Connector1 → talker_in_q ----
        def thinker_worker():
            try:
                for chunk in self.thinker.run(
                    input_ids, max_new_tokens=max_new_tokens, eos_token_id=eos_token_id
                ):
                    talker_input = self.thinker2talker.process(chunk)
                    if talker_input is not None:
                        talker_in_q.put(talker_input)
                    # Forward thinker token events to output
                    if hasattr(chunk, "token_id") and hasattr(chunk, "is_finished"):
                        tid = int(chunk.token_id.item()) if isinstance(chunk.token_id, torch.Tensor) else int(chunk.token_id)
                        with lock:
                            all_tokens.append(tid)
                talker_in_q.put(_SENTINEL)
            except BaseException as exc:
                error_q.put(exc)

        # ---- Thread 2: talker_in_q → Talker → Connector2 → audio_out_q ----
        def talker_worker():
            try:
                talker_inputs = self._queue_iter(talker_in_q)
                for code_chunk in self.talker.run(talker_inputs):
                    code2wav_input = self.talker2code2wav.process(code_chunk)
                    if code2wav_input is not None:
                        audio_out_q.put(code2wav_input)
                audio_out_q.put(_SENTINEL)
            except BaseException as exc:
                error_q.put(exc)

        # ---- Thread 3: audio_out_q → Code2Wav → consumer ----
        def code2wav_worker():
            try:
                code_inputs = self._queue_iter(audio_out_q)
                for audio_chunk in self.code2wav.run(code_inputs):
                    with lock:
                        all_audio.append(audio_chunk.audio)
            except BaseException as exc:
                error_q.put(exc)

        # Start all threads
        thinker_thread = threading.Thread(target=thinker_worker, name="omni-thinker", daemon=True)
        talker_thread = threading.Thread(target=talker_worker, name="omni-talker", daemon=True)
        code2wav_thread = threading.Thread(target=code2wav_worker, name="omni-code2wav", daemon=True)

        thinker_thread.start()
        talker_thread.start()
        code2wav_thread.start()

        # ---- Main thread: yield events ----
        # Poll for errors and events from all threads
        try:
            while thinker_thread.is_alive() or talker_thread.is_alive() or code2wav_thread.is_alive():
                # Check for errors
                if not error_q.empty():
                    exc = error_q.get_nowait()
                    yield StreamEvent.error(exc)
                    return

                # Yield thinker token events
                with lock:
                    while all_tokens:
                        tid = all_tokens.pop(0)
                        yield StreamEvent.thinker_token(tid, step=tid)

                # Yield audio chunk events (intermediate)
                # Audio chunks are collected by code2wav_worker; we yield them at completion

                # Small sleep to avoid busy-waiting
                import time
                time.sleep(0.001)
        finally:
            # Wait for all threads to finish
            thinker_thread.join(timeout=5)
            talker_thread.join(timeout=5)
            code2wav_thread.join(timeout=5)

        # Check for late errors
        if not error_q.empty():
            exc = error_q.get_nowait()
            yield StreamEvent.error(exc)
            return

        # Yield completion event
        text_ids = torch.tensor([all_tokens], dtype=torch.long) if all_tokens else torch.empty(1, 0, dtype=torch.long)
        final_audio = torch.cat(all_audio, dim=-1) if all_audio else torch.empty(1, 0, dtype=torch.float32)
        yield StreamEvent.complete(text_ids, final_audio)

    @staticmethod
    def _queue_iter(q: queue.Queue):
        """Convert a queue to an iterator, stopping at sentinel."""
        while True:
            item = q.get()
            if item is _SENTINEL:
                return
            yield item
