# Copyright 2025 HOUMO AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# SPDX-License-Identifier: Apache-2.0

"""Stage connectors — Thinker→Talker and Talker→Code2Wav.

These connectors mirror vLLM-Omni's ``thinker2talker_async_chunk`` and
``talker2code2wav_async_chunk`` functions, transforming stage output into
the next stage's expected input format.

* **Thinker2TalkerConnector**: Packages Thinker per-step output into
  Talker-compatible fused guidance inputs (hidden_state, role_mask,
  bypass_embeds, bypass_mask).

* **Talker2Code2WavConnector**: Accumulates Talker residual codes and
  emits code chunks with left-context overlap when enough codes are ready.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

from .events import (
    AudioChunk,
    Code2WavInput,
    TalkerInput,
    TalkerInputDecode,
    TalkerInputPrefill,
    TalkerStepOutput,
    ThinkerChunk,
    ThinkerDecodeChunk,
    ThinkerPrefillChunk,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thinker → Talker connector
# ---------------------------------------------------------------------------


class Thinker2TalkerConnector:
    """Transform Thinker output chunks into Talker-compatible input chunks.

    This connector handles the complex mapping from raw Thinker token
    embeddings/hidden states to the fused guidance inputs that the HMONNX
    talker expects.  It mirrors vLLM-Omni's ``thinker2talker_async_chunk``.

    Parameters
    ----------
    accept_hidden_layer : int
        Which Thinker hidden layer to route to ``hidden_state``.
    thinker_hidden_size : int
        Dimensionality of the Thinker hidden states.
    talker_hidden_size : int
        Dimensionality of the Talker's expected input (may differ after projection).
    num_lm_heads : int
        Number of code predictor heads (default 15).
    """

    def __init__(
        self,
        accept_hidden_layer: int = 3,
        thinker_hidden_size: int = 3584,
        talker_hidden_size: int = 3584,
        num_lm_heads: int = 15,
        hidden_projection=None,
    ):
        self.accept_hidden_layer = accept_hidden_layer
        self.thinker_hidden_size = thinker_hidden_size
        self.talker_hidden_size = talker_hidden_size
        self.num_lm_heads = num_lm_heads
        # hidden_projection: tuple (w1, b1, w2, b2) or None, for 2048->1024 projection
        self._hp_weights = hidden_projection
        self._collected_embeds: list[torch.Tensor] = []
        self._collected_hiddens: list[torch.Tensor] = []
        self._collected_token_ids: list[torch.Tensor] = []
        self._tts_bos_embed: Optional[torch.Tensor] = None
        self._tts_eos_embed: Optional[torch.Tensor] = None
        self._tts_pad_embed: Optional[torch.Tensor] = None
        self._prefill_emitted = False
        self._decode_step = 0

    @staticmethod
    def _apply_hidden_projection(x: torch.Tensor, hp_weights) -> torch.Tensor:
        """Apply talker.hidden_projection: 2048 -> 2048 -> SiLU -> 1024."""
        import torch.nn.functional as F  # noqa: F811

        w1, b1, w2, b2 = hp_weights
        x = F.linear(x, w1, b1)
        x = F.silu(x)
        x = F.linear(x, w2, b2)
        return x

    def reset(self):
        """Reset internal state for a new generation."""
        self._collected_embeds.clear()
        self._collected_hiddens.clear()
        self._collected_token_ids.clear()
        self._tts_bos_embed = None
        self._tts_eos_embed = None
        self._tts_pad_embed = None
        self._prefill_emitted = False
        self._decode_step = 0

    def set_tts_embeddings(
        self,
        bos_embed: torch.Tensor,
        eos_embed: torch.Tensor,
        pad_embed: torch.Tensor,
    ):
        """Set TTS special token embeddings (called once before processing)."""
        self._tts_bos_embed = bos_embed
        self._tts_eos_embed = eos_embed
        self._tts_pad_embed = pad_embed

    def process(self, chunk: ThinkerChunk) -> Optional[TalkerInput]:
        """Process a Thinker chunk and return a Talker input (or None)."""
        if isinstance(chunk, ThinkerPrefillChunk):
            return self._process_prefill(chunk)
        elif isinstance(chunk, ThinkerDecodeChunk):
            return self._process_decode(chunk)
        return None

    def _process_prefill(self, chunk: ThinkerPrefillChunk) -> TalkerInputPrefill:
        """Build Talker prefill input from Thinker prefill chunk."""
        self._collected_token_ids.append(chunk.token_ids)
        self._collected_embeds.extend(chunk.step_embeds)
        self._collected_hiddens.extend(chunk.step_hiddens)

        if chunk.tts_bos_embed is not None:
            self._tts_bos_embed = chunk.tts_bos_embed
        if chunk.tts_eos_embed is not None:
            self._tts_eos_embed = chunk.tts_eos_embed
        if chunk.tts_pad_embed is not None:
            self._tts_pad_embed = chunk.tts_pad_embed

        prompt_len = len(chunk.step_embeds)
        # Build fused guidance inputs following the HMONNX contract:
        # hidden_state: from accept_hidden_layer for all positions, dim=talker_hidden_state_size (2048)
        # role_mask: 1 for text positions, 0 for multimodal (simplified: all 1)
        # bypass_embeds: pre-projected talker-space embeddings (1024) that bypass hidden_projection
        #   For user text, bypass is disabled (zeros + mask=0), matching user_parts_hook.
        #   Assistant text gets proper talker-space embeddings from hidden_projection.
        # bypass_mask: 1 where bypass is active.
        all_hiddens = torch.cat(chunk.step_hiddens, dim=1)  # [1, prompt_len, D]
        all_embeds = torch.cat(chunk.step_embeds, dim=1)  # [1, prompt_len, D]

        hidden_state = all_hiddens.detach().to(torch.float16)
        role_mask = torch.ones(1, prompt_len, 1, dtype=torch.float16)
        bypass_embeds = torch.zeros(1, prompt_len, self.talker_hidden_size, dtype=torch.float16)
        bypass_mask = torch.zeros(1, prompt_len, 1, dtype=torch.float16)

        projected_embeds = all_embeds.detach().to(torch.float16)
        if self._hp_weights is not None:
            projected_embeds = self._apply_hidden_projection(projected_embeds, self._hp_weights).detach().to(torch.float16)

        # Build talker_input_ids (all pad tokens for repetition penalty)
        if chunk.token_ids is not None:
            talker_input_ids = chunk.token_ids.detach().to(torch.long)
        else:
            talker_input_ids = torch.zeros(1, prompt_len, dtype=torch.long)

        trailing_text_hidden = projected_embeds.detach().to(torch.float16)

        self._prefill_emitted = True

        return TalkerInputPrefill(
            hidden_state=hidden_state,
            role_mask=role_mask,
            bypass_embeds=bypass_embeds,
            bypass_mask=bypass_mask,
            trailing_text_hidden=trailing_text_hidden,
            talker_input_ids=talker_input_ids,
            tts_pad_embed=self._tts_pad_embed,
            is_finished=chunk.is_finished,
        )

    def _process_decode(self, chunk: ThinkerDecodeChunk) -> Optional[TalkerInput]:
        """Build Talker decode input from Thinker decode chunk."""
        if chunk.is_finished:
            # Send a finished signal
            return TalkerInputDecode(
                codec_embedding=torch.empty(0),
                is_finished=True,
            )

        if chunk.step_embeds.numel() == 0:
            return None

        self._collected_embeds.append(chunk.step_embeds)
        self._collected_hiddens.append(chunk.step_hidden)
        self._collected_token_ids.append(chunk.token_id)

        codec_embedding = chunk.step_embeds.detach().to(torch.float16)
        if self._hp_weights is not None:
            codec_embedding = self._apply_hidden_projection(codec_embedding, self._hp_weights).detach().to(torch.float16)

        # trailing_text_hidden for this step
        trailing_text_hidden = None
        if self._tts_pad_embed is not None:
            trailing_text_hidden = torch.cat(self._collected_hiddens, dim=1).detach().to(torch.float16)

        generation_step = self._decode_step
        self._decode_step += 1

        return TalkerInputDecode(
            codec_embedding=codec_embedding,
            trailing_text_hidden=trailing_text_hidden,
            tts_pad_embed=self._tts_pad_embed,
            generation_step=generation_step,
            is_finished=False,
        )


# ---------------------------------------------------------------------------
# Talker → Code2Wav connector
# ---------------------------------------------------------------------------


class Talker2Code2WavConnector:
    """Accumulate Talker residual codes and emit code chunks for Code2Wav.

    Mirrors vLLM-Omni's ``talker2code2wav_async_chunk`` chunking strategy:
    accumulates codes until a chunk is full, then emits with left-context
    overlap for smooth audio boundaries.

    Parameters
    ----------
    codec_chunk_frames : int
        Number of codec frames per chunk (default 25 ≈ 1 s at 25 fps).
    codec_left_context_frames : int
        Overlap frames for smooth chunk boundaries (default 25).
    num_quantizers : int
        Number of RVQ quantizer levels (default 16).
    initial_chunk_size : int | None
        If set, use a smaller initial chunk for faster first-audio latency.
    """

    def __init__(
        self,
        codec_chunk_frames: int = 25,
        codec_left_context_frames: int = 25,
        num_quantizers: int = 16,
        initial_chunk_size: Optional[int] = None,
    ):
        self.codec_chunk_frames = max(1, codec_chunk_frames)
        self.codec_left_context_frames = max(0, codec_left_context_frames)
        self.num_quantizers = num_quantizers
        self.initial_chunk_size = initial_chunk_size
        self._codes: list[torch.Tensor] = []
        self._total_steps = 0
        self._emitted_steps = 0

    def reset(self):
        """Reset internal state for a new generation."""
        self._codes.clear()
        self._total_steps = 0
        self._emitted_steps = 0

    def process(self, chunk: TalkerStepOutput) -> Optional[Code2WavInput]:
        """Process a Talker step output and return a Code2Wav input (or None).

        Each Talker step must output a complete residual-code frame with one
        code per quantizer level. The resulting history is stored as
        ``[num_quantizers, total_steps]`` for Code2Wav.
        """
        if chunk.is_finished:
            return self._flush(final=True)

        frame = self._normalize_residual_frame(chunk.residual_codes)
        self._codes.append(frame)
        self._total_steps += int(frame.shape[-1])

        effective_chunk_size = self.codec_chunk_frames
        if self.initial_chunk_size is not None and self._total_steps <= self.initial_chunk_size:
            effective_chunk_size = self.initial_chunk_size

        if self._total_steps - self._emitted_steps >= effective_chunk_size:
            return self._emit_chunk(final=False)

        return None

    def _normalize_residual_frame(self, residual_codes: torch.Tensor) -> torch.Tensor:
        codes = residual_codes.detach().to(torch.long)
        if codes.dim() == 1:
            if int(codes.numel()) != self.num_quantizers:
                raise ValueError(f"expected {self.num_quantizers} residual codes, got shape {tuple(codes.shape)}")
            return codes.view(self.num_quantizers, 1)
        if codes.dim() == 2:
            if tuple(codes.shape) == (1, self.num_quantizers):
                return codes[0].view(self.num_quantizers, 1)
            if tuple(codes.shape) == (self.num_quantizers, 1):
                return codes
            raise ValueError(f"expected residual codes [1,{self.num_quantizers}], got {tuple(codes.shape)}")
        if codes.dim() == 3:
            if int(codes.shape[0]) != 1 or int(codes.shape[1]) != self.num_quantizers:
                raise ValueError(f"expected residual codes [1,{self.num_quantizers},T], got {tuple(codes.shape)}")
            return codes[0]
        raise ValueError(f"expected residual codes with 1-3 dims, got {tuple(codes.shape)}")

    def _emit_chunk(self, final: bool) -> Optional[Code2WavInput]:
        """Emit a code chunk with left-context overlap."""
        if not self._codes:
            return None

        codes_2d = torch.cat(self._codes, dim=-1)

        total_steps = int(codes_2d.shape[-1])

        if final:
            if self._emitted_steps >= total_steps:
                return None
            start = self._emitted_steps
            end = total_steps
            left_ctx = min(self.codec_left_context_frames, start)
            codes_chunk = codes_2d[:, start - left_ctx : end]
            self._emitted_steps = end
            return Code2WavInput(codes=codes_chunk, left_context_size=left_ctx, is_finished=True)

        chunk_size = self.codec_chunk_frames
        start = self._emitted_steps
        end = min(start + chunk_size, total_steps)
        left_ctx = min(self.codec_left_context_frames, start)

        codes_chunk = codes_2d[:, start - left_ctx : end]
        self._emitted_steps = end
        return Code2WavInput(
            codes=codes_chunk,
            left_context_size=left_ctx,
            is_finished=False,
        )

    def _flush(self, final: bool = True) -> Optional[Code2WavInput]:
        """Flush any remaining codes."""
        if not self._codes:
            return None
        return self._emit_chunk(final=final)

    @property
    def total_steps(self) -> int:
        return self._total_steps
