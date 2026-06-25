# Copyright 2025 HOUMO AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# SPDX-License-Identifier: Apache-2.0

"""Stage 1 – Independent HMONNX Talker generation loop.

This stage runs the talker + code-predictor HMONNX sessions independently,
consuming TalkerInput chunks from the Thinker→Talker connector and emitting
residual audio codec codes step by step — mirroring vLLM-Omni's async
Talker stage.

The Talker stage handles:

* **Prefill**: Builds fused guidance inputs and runs HMONNX talker prefill.
* **Decode**: For each incoming Thinker decode chunk, runs one HMONNX talker
  decode step, then runs the code-predictor loop to generate ``num_code_groups``
  residual codes.
"""

from __future__ import annotations

import logging
from typing import Iterator, Optional

import torch

from .events import (
    TalkerInput,
    TalkerInputDecode,
    TalkerInputPrefill,
    TalkerStepOutput,
)
from ._utils import (
    extract_logits_and_hidden_states,
    make_kv_cache_state,
    pad_to_length,
)

logger = logging.getLogger(__name__)


class HMONNXTalkerStage:
    """Independent Talker stage backed by HMONNX talker + code-predictor sessions.

    Parameters
    ----------
    talker_prefill_session : HMONNXInference
        Talker prefill HMONNX session.
    talker_decode_session : HMONNXInference
        Talker decode HMONNX session.
    predictor_prefill_session : HMONNXInference | None
        Code-predictor prefill HMONNX session.
    predictor_decode_session : HMONNXInference | None
        Code-predictor decode HMONNX session.
    talker_kv_cache_info : dict
        Talker KV cache metadata.
    predictor_kv_cache_info : dict
        Code-predictor KV cache metadata.
    static_prefill_len : int
        Static prefill length for the talker.
    static_predictor_prefill_len : int
        Static prefill length for the code-predictor.
    num_code_groups : int
        Number of code groups (residual quantizers) per step (default 16).
    num_lm_heads : int
        Number of code-predictor heads (default 15).
    thinker_hidden_size : int
        Thinker hidden size for building fused inputs.
    """

    def __init__(
        self,
        talker_prefill_session,
        talker_decode_session,
        predictor_prefill_session=None,
        predictor_decode_session=None,
        talker_kv_cache_info: Optional[dict] = None,
        predictor_kv_cache_info: Optional[dict] = None,
        static_prefill_len: int = 0,
        static_predictor_prefill_len: int = 0,
        num_code_groups: int = 16,
        num_lm_heads: int = 15,
        thinker_hidden_size: int = 3584,
        talker_hidden_size: int = 3584,
    ):
        self.talker_prefill_session = talker_prefill_session
        self.talker_decode_session = talker_decode_session
        self.predictor_prefill_session = predictor_prefill_session
        self.predictor_decode_session = predictor_decode_session
        self.talker_kv_cache_info = talker_kv_cache_info or {}
        self.predictor_kv_cache_info = predictor_kv_cache_info or {}
        self.static_prefill_len = static_prefill_len
        self.static_predictor_prefill_len = static_predictor_prefill_len
        self.num_code_groups = num_code_groups
        self.num_lm_heads = num_lm_heads
        self.thinker_hidden_size = thinker_hidden_size
        self.talker_hidden_size = talker_hidden_size

        # Mutable state
        self._talker_cache: Optional[dict] = None
        self._predictor_cache: Optional[dict] = None
        self._generation_step = -1
        self._step = 0

    def reset(self):
        """Reset internal state for a new generation."""
        if self.talker_kv_cache_info:
            self._talker_cache = make_kv_cache_state(self.talker_kv_cache_info)
        if self.predictor_kv_cache_info:
            self._predictor_cache = make_kv_cache_state(self.predictor_kv_cache_info)
        self._generation_step = -1
        self._step = 0

    def run(self, inputs: Iterator[TalkerInput]) -> Iterator[TalkerStepOutput]:
        """Consume TalkerInput chunks and emit TalkerStepOutput (residual codes).

        Parameters
        ----------
        inputs : Iterator[TalkerInput]
            Input chunks from the Thinker→Talker connector.

        Yields
        ------
        TalkerStepOutput
            Per-step residual codec codes.
        """
        self.reset()
        prefill_done = False

        for inp in inputs:
            if isinstance(inp, TalkerInputPrefill) and not prefill_done:
                yield from self._run_prefill(inp)
                prefill_done = True
            elif isinstance(inp, TalkerInputDecode):
                if inp.is_finished:
                    yield TalkerStepOutput(
                        residual_codes=torch.empty(0),
                        step=self._step,
                        is_finished=True,
                    )
                    return
                yield from self._run_decode(inp)
            elif isinstance(inp, TalkerInputPrefill) and prefill_done:
                # Second prefill — shouldn't happen in normal flow
                logger.warning("Received duplicate TalkerInputPrefill, skipping")

    def _run_prefill(self, inp: TalkerInputPrefill) -> Iterator[TalkerStepOutput]:
        """Run the HMONNX talker prefill and code-predictor prefill."""
        self._talker_cache = make_kv_cache_state(self.talker_kv_cache_info)
        if self.predictor_kv_cache_info:
            self._predictor_cache = make_kv_cache_state(self.predictor_kv_cache_info)

        # ---- Talker prefill ----
        hidden_state = inp.hidden_state.detach().cpu().to(torch.float16)
        role_mask = inp.role_mask.detach().cpu().to(torch.float16)
        bypass_embeds = inp.bypass_embeds.detach().cpu().to(torch.float16)
        bypass_mask = inp.bypass_mask.detach().cpu().to(torch.float16)
        seq_len = int(hidden_state.shape[1])

        # Pad to static length
        if seq_len < self.static_prefill_len and self.static_prefill_len > 0:
            hidden_state = pad_to_length(hidden_state, self.static_prefill_len, dim=1)
            role_mask = pad_to_length(role_mask, self.static_prefill_len, dim=1)
            bypass_embeds = pad_to_length(bypass_embeds, self.static_prefill_len, dim=1)
            bypass_mask = pad_to_length(bypass_mask, self.static_prefill_len, dim=1)

        past_seq_len = torch.tensor([0], dtype=torch.int32)
        current_len = torch.tensor([seq_len], dtype=torch.int32)

        talker_output = self.talker_prefill_session.forward(
            hidden_state,
            role_mask,
            bypass_embeds,
            bypass_mask,
            past_seq_len,
            current_len,
            *self._talker_cache["past_key_caches"],
            *self._talker_cache["past_value_caches"],
        )
        talker_logits, talker_hidden = extract_logits_and_hidden_states(
            talker_output, torch.device("cpu"), seq_len
        )
        self._talker_cache["past_seq_length"] = seq_len
        self._generation_step = 0

        # ---- Code predictor prefill ----
        # The code predictor takes (past_hidden, last_id_hidden) concatenated
        # and runs with head mask for the prefill step
        if self.predictor_prefill_session is not None and self._predictor_cache is not None:
            # Use the last hidden state from the talker
            last_hidden = talker_hidden[:, -1:, :].detach().cpu().to(torch.float16) if talker_hidden is not None else hidden_state[:, :1, :]
            # For prefill, we use a dummy last_id_hidden (embedding of first code)
            dummy_id_embed = torch.zeros(1, 1, self.thinker_hidden_size, dtype=torch.float16)
            predictor_input = torch.cat([last_hidden, dummy_id_embed], dim=1)

            predictor_seq_len = int(predictor_input.shape[1])
            static_pred_len = self.static_predictor_prefill_len if self.static_predictor_prefill_len > 0 else predictor_seq_len
            if predictor_seq_len < static_pred_len:
                predictor_input = pad_to_length(predictor_input, static_pred_len, dim=1)

            # Head mask for prefill
            step = max(0, min(predictor_seq_len - 2, self.num_lm_heads - 1))
            head_mask = torch.zeros(1, static_pred_len, self.num_lm_heads, 1, dtype=torch.float16)
            head_mask[:, :, step, 0] = 1.0

            pred_past_seq_len = torch.tensor([0], dtype=torch.int32)
            pred_current_len = torch.tensor([predictor_seq_len], dtype=torch.int32)

            pred_output = self.predictor_prefill_session.forward(
                predictor_input,
                head_mask,
                pred_past_seq_len,
                pred_current_len,
                *self._predictor_cache["past_key_caches"],
                *self._predictor_cache["past_value_caches"],
            )
            pred_logits, pred_hidden = extract_logits_and_hidden_states(pred_output, torch.device("cpu"), predictor_seq_len)
            self._predictor_cache["past_seq_length"] = predictor_seq_len

            # Argmax to get first code (head 0)
            first_code = torch.argmax(pred_logits[:, -1, :], dim=-1, keepdim=True)  # [1, 1]
            residual_codes = [first_code]
            next_predictor_input = (
                pred_hidden[:, -1:, :].detach().cpu().to(torch.float16)
                if pred_hidden is not None
                else torch.zeros(1, 1, self.talker_hidden_size, dtype=torch.float16)
            )
            self._generation_step = 1  # heads 1..14 to follow
        else:
            residual_codes = []
            next_predictor_input = torch.zeros(1, 1, self.talker_hidden_size, dtype=torch.float16)

        # Run remaining code predictor decode steps (num_code_groups-1)
        for _ in range(max(0, self.num_code_groups - len(residual_codes))):
            code, next_predictor_input = self._run_one_predictor_decode(next_predictor_input)
            residual_codes.append(code)

        if residual_codes:
            yield TalkerStepOutput(
                residual_codes=torch.cat(residual_codes, dim=-1),
                step=self._step,
                is_finished=False,
            )
            self._step += 1

    def _run_decode(self, inp: TalkerInputDecode) -> Iterator[TalkerStepOutput]:
        """Run one HMONNX talker decode step + code-predictor loop."""
        if self._talker_cache is None:
            self._talker_cache = make_kv_cache_state(self.talker_kv_cache_info)

        # ---- Talker decode ----
        # The talker decode input is the codec embedding for this step
        codec_embed = inp.codec_embedding.detach().cpu().to(torch.float16)
        if codec_embed.dim() == 2:
            codec_embed = codec_embed.unsqueeze(0)

        seq_len = 1  # decode is always seq_len=1
        past_seq_len = torch.tensor([self._talker_cache["past_seq_length"]], dtype=torch.int32)
        current_len = torch.tensor([seq_len], dtype=torch.int32)

        # For the HMONNX talker decode, we use the bypass path:
        # hidden_state=zeros, role_mask=zeros, bypass_embeds=codec_embed, bypass_mask=1
        source = torch.zeros(1, 1, self.thinker_hidden_size, dtype=torch.float16)
        role_mask = torch.zeros(1, 1, 1, dtype=torch.float16)
        bypass_embeds = codec_embed
        bypass_mask = torch.ones(1, 1, 1, dtype=torch.float16)

        talker_output = self.talker_decode_session.forward(
            source,
            role_mask,
            bypass_embeds,
            bypass_mask,
            past_seq_len,
            current_len,
            *self._talker_cache["past_key_caches"],
            *self._talker_cache["past_value_caches"],
        )
        talker_logits, talker_hidden = extract_logits_and_hidden_states(talker_output, torch.device("cpu"), 1)
        self._talker_cache["past_seq_length"] += 1
        self._generation_step += 1

        # ---- Code predictor decode ----
        yield from self._run_predictor_decode_steps(talker_hidden=talker_hidden)

    def _run_one_predictor_decode(self, predictor_input: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one code-predictor decode step and return ``(code, next_input)``."""
        if self.predictor_decode_session is None or self._predictor_cache is None:
            empty_code = torch.zeros(1, 1, dtype=torch.long)
            empty_input = torch.zeros(1, 1, self.talker_hidden_size, dtype=torch.float16)
            return empty_code, empty_input

        step = self._generation_step % self.num_lm_heads
        if predictor_input is None:
            predictor_input = torch.zeros(1, 1, self.talker_hidden_size, dtype=torch.float16)
        last_hidden = predictor_input.detach().cpu().to(torch.float16)

        head_mask = torch.zeros(1, 1, self.num_lm_heads, 1, dtype=torch.float16)
        head_mask[0, 0, step, 0] = 1.0

        past_seq_len = torch.tensor([self._predictor_cache["past_seq_length"]], dtype=torch.int32)
        current_len = torch.tensor([1], dtype=torch.int32)

        pred_output = self.predictor_decode_session.forward(
            last_hidden,
            head_mask,
            past_seq_len,
            current_len,
            *self._predictor_cache["past_key_caches"],
            *self._predictor_cache["past_value_caches"],
        )
        pred_logits, pred_hidden = extract_logits_and_hidden_states(pred_output, torch.device("cpu"), 1)
        self._predictor_cache["past_seq_length"] += 1

        code = torch.argmax(pred_logits[:, -1, :], dim=-1, keepdim=True)  # [1, 1]
        self._generation_step += 1
        next_input = (
            pred_hidden[:, -1:, :].detach().cpu().to(torch.float16)
            if pred_hidden is not None
            else torch.zeros(1, 1, self.talker_hidden_size, dtype=torch.float16)
        )
        return code, next_input

    def _run_predictor_decode_steps(
        self, talker_hidden: Optional[torch.Tensor] = None
    ) -> Iterator[TalkerStepOutput]:
        """Run the code-predictor loop for num_code_groups steps."""
        if self.predictor_decode_session is None or self._predictor_cache is None:
            return

        residual_codes = []
        next_predictor_input = None
        for code_idx in range(self.num_code_groups):
            step = self._generation_step % self.num_lm_heads

            # Build predictor input
            if talker_hidden is not None and code_idx == 0:
                last_hidden = talker_hidden[:, -1:, :].detach().cpu().to(torch.float16)
            elif next_predictor_input is not None:
                last_hidden = next_predictor_input
            else:
                last_hidden = torch.zeros(1, 1, self.talker_hidden_size, dtype=torch.float16)

            head_mask = torch.zeros(1, 1, self.num_lm_heads, 1, dtype=torch.float16)
            head_mask[0, 0, step, 0] = 1.0

            past_seq_len = torch.tensor([self._predictor_cache["past_seq_length"]], dtype=torch.int32)
            current_len = torch.tensor([1], dtype=torch.int32)

            pred_output = self.predictor_decode_session.forward(
                last_hidden,
                head_mask,
                past_seq_len,
                current_len,
                *self._predictor_cache["past_key_caches"],
                *self._predictor_cache["past_value_caches"],
            )
            pred_logits, pred_hidden = extract_logits_and_hidden_states(pred_output, torch.device("cpu"), 1)
            self._predictor_cache["past_seq_length"] += 1

            code = torch.argmax(pred_logits[:, -1, :], dim=-1, keepdim=True)  # [1, 1]
            residual_codes.append(code)
            next_predictor_input = (
                pred_hidden[:, -1:, :].detach().cpu().to(torch.float16)
                if pred_hidden is not None
                else torch.zeros(1, 1, self.talker_hidden_size, dtype=torch.float16)
            )
            self._generation_step += 1

        if residual_codes:
            yield TalkerStepOutput(
                residual_codes=torch.cat(residual_codes, dim=-1),
                step=self._step,
                is_finished=False,
            )
            self._step += 1
