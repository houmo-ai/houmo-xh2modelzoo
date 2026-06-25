# Copyright 2025 HOUMO AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# SPDX-License-Identifier: Apache-2.0

"""Event and chunk data classes for the Qwen3-Omni streaming pipeline.

Each stage communicates via typed dataclasses.  The naming convention mirrors
vLLM-Omni's ``OmniPayload`` / ``OmniPayloadStruct`` while remaining specific
to HMONNX inference contracts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union

import torch


# ---------------------------------------------------------------------------
# Stage 0 – Thinker output chunks
# ---------------------------------------------------------------------------


@dataclass
class ThinkerPrefillChunk:
    """First chunk emitted by the Thinker stage after its prefill.

    Contains all prompt-level data needed by the Talker to build its fused
    guidance inputs and run the talker prefill.
    """

    token_ids: torch.Tensor  # [1, prompt_len]
    step_embeds: list[torch.Tensor] = field(default_factory=list)  # per-step [1, 1, D]
    step_hiddens: list[torch.Tensor] = field(default_factory=list)  # per-step [1, 1, D]
    tts_bos_embed: Optional[torch.Tensor] = None  # [1, 1, D]
    tts_eos_embed: Optional[torch.Tensor] = None  # [1, 1, D]
    tts_pad_embed: Optional[torch.Tensor] = None  # [1, 1, D]
    speaker_id: int = 0
    is_finished: bool = False


@dataclass
class ThinkerDecodeChunk:
    """Per-step chunk emitted by the Thinker stage during autoregressive decode."""

    token_id: torch.Tensor  # [1, 1]
    step_embeds: torch.Tensor = field(default_factory=lambda: torch.empty(0))  # [1, 1, D]
    step_hidden: torch.Tensor = field(default_factory=lambda: torch.empty(0))  # [1, 1, D]
    is_finished: bool = False


ThinkerChunk = Union[ThinkerPrefillChunk, ThinkerDecodeChunk]


# ---------------------------------------------------------------------------
# Stage 1 – Talker input chunks (from connector)
# ---------------------------------------------------------------------------


@dataclass
class TalkerInputPrefill:
    """Prefill input for the Talker stage.

    Contains the fused guidance inputs (hidden_state, role_mask, bypass_embeds,
    bypass_mask) that the HMONNX talker prefill session expects.
    """

    hidden_state: torch.Tensor  # [1, seq_len, thinker_hs]
    role_mask: torch.Tensor  # [1, seq_len, 1]
    bypass_embeds: torch.Tensor  # [1, seq_len, D]
    bypass_mask: torch.Tensor  # [1, seq_len, 1]
    trailing_text_hidden: Optional[torch.Tensor] = None
    talker_input_ids: Optional[torch.Tensor] = None
    tts_pad_embed: Optional[torch.Tensor] = None
    is_finished: bool = False


@dataclass
class TalkerInputDecode:
    """Per-step decode input for the Talker stage."""

    codec_embedding: torch.Tensor  # [1, 1, D] – summed codec embedding for this step
    trailing_text_hidden: Optional[torch.Tensor] = None
    tts_pad_embed: Optional[torch.Tensor] = None
    generation_step: int = 0
    is_finished: bool = False


TalkerInput = Union[TalkerInputPrefill, TalkerInputDecode]


# ---------------------------------------------------------------------------
# Stage 1 – Talker output chunks
# ---------------------------------------------------------------------------


@dataclass
class TalkerStepOutput:
    """Per-step output from the Talker stage (residual audio codes)."""

    residual_codes: torch.Tensor  # [1, num_quantizers]
    step: int = 0
    is_finished: bool = False


# ---------------------------------------------------------------------------
# Stage 2 – Code2Wav input / output
# ---------------------------------------------------------------------------


@dataclass
class Code2WavInput:
    """Code chunk consumed by the Code2Wav stage."""

    codes: torch.Tensor  # [num_quantizers, chunk_len]
    left_context_size: int = 0
    is_finished: bool = False


@dataclass
class AudioChunk:
    """Decoded audio chunk emitted by the Code2Wav stage."""

    audio: torch.Tensor  # [1, samples]
    chunk_index: int = 0
    sample_rate: int = 24000
    is_finished: bool = False


# ---------------------------------------------------------------------------
# Unified stream event
# ---------------------------------------------------------------------------


@dataclass
class StreamEvent:
    """Unified event yielded by the orchestrator to the consumer."""

    type: str  # "thinker_token", "audio_chunk", "complete", "error"
    data: dict = field(default_factory=dict)

    @staticmethod
    def thinker_token(token_id: int, step: int, is_finished: bool = False) -> StreamEvent:
        return StreamEvent(
            type="thinker_token",
            data={"token_id": token_id, "step": step, "is_finished": is_finished},
        )

    @staticmethod
    def audio_chunk(audio: torch.Tensor, chunk_index: int, sample_rate: int = 24000) -> StreamEvent:
        return StreamEvent(
            type="audio_chunk",
            data={"audio": audio, "chunk_index": chunk_index, "sample_rate": sample_rate},
        )

    @staticmethod
    def complete(
        text_ids: torch.Tensor, audio: torch.Tensor, sample_rate: int = 24000
    ) -> StreamEvent:
        return StreamEvent(
            type="complete",
            data={"text_ids": text_ids, "audio": audio, "sample_rate": sample_rate},
        )

    @staticmethod
    def error(exc: BaseException) -> StreamEvent:
        return StreamEvent(type="error", data={"error": exc})
