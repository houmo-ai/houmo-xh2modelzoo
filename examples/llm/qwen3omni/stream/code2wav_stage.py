# Copyright 2025 HOUMO AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# SPDX-License-Identifier: Apache-2.0

"""Stage 2 – Independent HMONNX Code2Wav stage.

Decodes Talker→Code2Wav connector output (code chunks) into audio waveforms,
mirroring vLLM-Omni's Code2Wav generation stage.
"""

from __future__ import annotations

import logging
from typing import Iterator, Optional

import torch

from .events import AudioChunk, Code2WavInput

logger = logging.getLogger(__name__)


class HMONNXCode2WavStage:
    """Code-to-waveform stage backed by an HMONNX code2wav session.

    Parameters
    ----------
    code2wav_session : HMONNXInference
        HMONNX code2wav session.
    total_upsample : int
        Upsampling factor (e.g. 600 for 24 kHz from 25 fps codec).
    """

    def __init__(self, code2wav_session, total_upsample: int = 600):
        self.code2wav_session = code2wav_session
        self.total_upsample = total_upsample
        self._chunk_index = 0

    def reset(self):
        """Reset internal state for a new generation."""
        self._chunk_index = 0

    def run(self, inputs: Iterator[Code2WavInput]) -> Iterator[AudioChunk]:
        """Consume code chunks and emit audio chunks.

        Parameters
        ----------
        inputs : Iterator[Code2WavInput]
            Code chunks from the Talker→Code2Wav connector.

        Yields
        ------
        AudioChunk
            Decoded audio waveform chunks.
        """
        self.reset()

        for code_input in inputs:
            audio = self._decode_chunk(code_input.codes, code_input.left_context_size)
            chunk = AudioChunk(
                audio=audio,
                chunk_index=self._chunk_index,
                is_finished=code_input.is_finished,
            )
            self._chunk_index += 1
            yield chunk

            if code_input.is_finished:
                return

    def _decode_chunk(self, codes: torch.Tensor, left_context_size: int = 0) -> torch.Tensor:
        """Decode a code chunk to audio via HMONNX code2wav.

        Parameters
        ----------
        codes : torch.Tensor
            Code tensor, shape ``[num_quantizers, chunk_len]`` or ``[1, num_quantizers, chunk_len]``.
        left_context_size : int
            Number of left-context frames to strip from the output.

        Returns
        -------
        torch.Tensor
            Audio waveform, shape ``[1, samples]``.
        """
        if codes.dim() == 3:
            codes = codes[0]  # Remove batch dim: [Q, T]

        codes = codes.detach().to(torch.int32)

        # Pad codes if needed for the HMONNX session
        # The session may expect a fixed length; we handle variable length by padding
        max_code_len = getattr(self.code2wav_session, "_max_code_len", None)

        wav = self.code2wav_session.forward(codes)
        if isinstance(wav, (list, tuple)):
            wav = wav[0]
        if not isinstance(wav, torch.Tensor):
            wav = torch.as_tensor(wav, dtype=torch.float32)

        wav = wav.detach().to(torch.float32)

        # Strip left context from output
        if left_context_size > 0:
            strip_samples = left_context_size * self.total_upsample
            if wav.shape[-1] > strip_samples:
                wav = wav[..., strip_samples:]

        # Ensure batch dimension
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)

        return wav
