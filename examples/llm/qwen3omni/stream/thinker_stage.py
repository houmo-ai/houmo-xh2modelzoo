# Copyright 2025 HOUMO AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# SPDX-License-Identifier: Apache-2.0

"""Stage 0 – Independent HMONNX Thinker generation loop.

This stage runs the text HMONNX prefill/decode cycle *without* relying on
HuggingFace ``model.generate()``.  It emits a ``ThinkerPrefillChunk`` once
the prefill completes, followed by one ``ThinkerDecodeChunk`` per autoregressive
step — mirroring vLLM-Omni's ``thinker2talker_async_chunk`` contract where
the first chunk carries full prompt data and subsequent chunks carry single
decode outputs.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator, Optional

import torch
import torch.nn as nn

from .events import ThinkerChunk, ThinkerDecodeChunk, ThinkerPrefillChunk
from ._utils import (
    extract_logits_and_hidden_states,
    make_kv_cache_state,
    pad_to_length,
)

logger = logging.getLogger(__name__)


class HMONNXThinkerStage:
    """Independent Thinker stage backed by HMONNX text prefill/decode sessions.

    Parameters
    ----------
    prefill_session : HMONNXInference
        Text prefill HMONNX session.
    decode_session : HMONNXInference
        Text decode HMONNX session.
    token_embedding : nn.Embedding
        Token embedding table (loaded from ``token_embedding_file``).
    kv_cache_info : dict
        KV cache metadata from ``meta.json`` (``shape``, ``num_decoder_layers``).
    input_sequence_length : int
        Static prefill length exported in the HMONNX graph.
    accept_hidden_layer : int
        Which hidden layer to use for talker consumption (e.g. 3).
    eos_token_ids : set[int]
        End-of-sequence token IDs.
    supports_position_ids : bool
        Whether the HMONNX graph accepts multimodal position IDs.
    supports_deepstack : bool
        Whether the HMONNX graph accepts deepstack features.
    supports_hidden_states : bool
        Whether the HMONNX graph outputs ``["logits", "hidden_states"]``.
    rope_deltas : Optional[torch.Tensor]
        Pre-computed rope deltas for multimodal position IDs.
    device : torch.device
        Execution device (typically ``cpu`` for HMONNX).
    """

    def __init__(
        self,
        prefill_session,
        decode_session,
        token_embedding: nn.Embedding,
        kv_cache_info: dict,
        input_sequence_length: int,
        accept_hidden_layer: int = 3,
        eos_token_ids: Optional[set[int]] = None,
        supports_position_ids: bool = False,
        supports_deepstack: bool = False,
        supports_hidden_states: bool = True,
        rope_deltas: Optional[torch.Tensor] = None,
        device: torch.device = torch.device("cpu"),
    ):
        self.prefill_session = prefill_session
        self.decode_session = decode_session
        self.token_embedding = token_embedding
        self.kv_cache_info = kv_cache_info
        self.input_sequence_length = int(input_sequence_length)
        self.accept_hidden_layer = max(int(accept_hidden_layer), 0)
        self.eos_token_ids = eos_token_ids or set()
        self.supports_position_ids = supports_position_ids
        self.supports_deepstack = supports_deepstack
        self.supports_hidden_states = supports_hidden_states
        self.rope_deltas = rope_deltas
        self.device = device

    @classmethod
    def from_meta(
        cls,
        text_meta: dict,
        root_dir: Path,
        logger_instance: Optional[logging.Logger] = None,
    ) -> HMONNXThinkerStage:
        """Construct from a ``meta.json`` dict (as produced by text export)."""
        from ._utils import create_hmonnx_session, resolve_meta_path

        log = logger_instance or logger
        kv_cache_info = text_meta["kv_cache"]
        kv_shape = kv_cache_info["shape"]
        num_layers = kv_cache_info["num_decoder_layers"]
        input_sequence_length = int(text_meta["wrap_cfg"]["input_sequence_length"])

        prefill_session = create_hmonnx_session(resolve_meta_path(text_meta, "prefill_onnx"), log)
        decode_session = create_hmonnx_session(resolve_meta_path(text_meta, "decode_onnx"), log)

        # Load token embedding
        embed_path = root_dir / text_meta["token_embedding_file"]
        state_dict = torch.load(embed_path, map_location="cpu", weights_only=False)
        if not isinstance(state_dict, dict) or "weight" not in state_dict:
            raise RuntimeError(f"Token embedding file {embed_path} does not contain a standard state_dict")
        vocab_size, hidden_size = state_dict["weight"].shape
        token_embedding = nn.Embedding(vocab_size, hidden_size)
        token_embedding.load_state_dict(state_dict)
        token_embedding.eval()

        supports_position_ids = bool(text_meta.get("supports_multimodal_position_ids", False))
        prefill_tensor_input_count = len(prefill_session.inputs) - (2 * num_layers)
        expected_base_inputs = 6 if supports_position_ids else 3
        supports_deepstack = prefill_tensor_input_count == expected_base_inputs + 3
        output_names = text_meta.get("output_names")
        supports_hidden_states = output_names == ["logits", "hidden_states"]
        if not supports_hidden_states:
            prefill_out = getattr(prefill_session, "get_output_names", lambda: [])()
            decode_out = getattr(decode_session, "get_output_names", lambda: [])()
            supports_hidden_states = len(prefill_out) >= 2 and len(decode_out) >= 2

        accept_hidden_layer = getattr(
            getattr(text_meta, "accept_hidden_layer", None)
            or text_meta.get("accept_hidden_layer"),
            None,
            3,
        )
        if accept_hidden_layer is None:
            accept_hidden_layer = 3

        return cls(
            prefill_session=prefill_session,
            decode_session=decode_session,
            token_embedding=token_embedding,
            kv_cache_info=kv_cache_info,
            input_sequence_length=input_sequence_length,
            accept_hidden_layer=int(accept_hidden_layer),
            eos_token_ids=set(),  # caller should set from config
            supports_position_ids=supports_position_ids,
            supports_deepstack=supports_deepstack,
            supports_hidden_states=supports_hidden_states,
            device=torch.device("cpu"),
        )

    def run(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 1024,
        eos_token_id: Optional[int | set[int]] = None,
        step_embeds_pre: Optional[list[torch.Tensor]] = None,
        step_hiddens_pre: Optional[list[torch.Tensor]] = None,
    ) -> Iterator[ThinkerChunk]:
        """Run the Thinker generation loop and yield chunks.

        Parameters
        ----------
        input_ids : torch.Tensor
            Prompt token IDs, shape ``[1, prompt_len]``.
        max_new_tokens : int
            Maximum number of decode steps.
        eos_token_id : int | set[int] | None
            Override EOS token IDs.
        step_embeds_pre, step_hiddens_pre : list[Tensor] | None
            Pre-collected embeds/hiddens for the prompt (used when the caller
            has already processed multimodal inputs and wants to provide the
            embedding-level representation).
        """
        if eos_token_id is not None:
            if isinstance(eos_token_id, int):
                self.eos_token_ids = {eos_token_id}
            elif isinstance(eos_token_id, (set, list, tuple)):
                self.eos_token_ids = set(eos_token_id)

        input_ids_cpu = input_ids.detach().cpu()
        prompt_len = int(input_ids_cpu.shape[1])

        # ---- Build prompt embeddings ----
        if step_embeds_pre is not None and step_hiddens_pre is not None:
            # Caller provided pre-processed embeddings (e.g. after multimodal scattering)
            prompt_embeds = torch.cat(step_embeds_pre, dim=1)  # [1, prompt_len, D]
            prompt_hiddens = torch.cat(step_hiddens_pre, dim=1)
        else:
            prompt_embeds = self.token_embedding(input_ids_cpu)
            prompt_hiddens = prompt_embeds  # hidden == embed when no separate hidden layer

        # ---- Prefill ----
        cache_state = make_kv_cache_state(self.kv_cache_info)
        static_len = self.input_sequence_length

        prefill_embeds = prompt_embeds.detach().cpu().to(torch.float16)
        prefill_embeds = pad_to_length(prefill_embeds, static_len, dim=1)

        prefill_inputs = [prefill_embeds]
        if self.supports_position_ids:
            # Placeholder – caller should provide position_ids via rope_deltas
            time_ids = torch.zeros(static_len, dtype=torch.int32)
            height_ids = torch.zeros(static_len, dtype=torch.int32)
            width_ids = torch.zeros(static_len, dtype=torch.int32)
            prefill_inputs.extend([time_ids, height_ids, width_ids])

        past_seq_len = torch.tensor([0], dtype=torch.int32)
        current_len = torch.tensor([prompt_len], dtype=torch.int32)
        prefill_inputs.extend([past_seq_len, current_len])

        if self.supports_deepstack:
            for _ in range(3):
                prefill_inputs.append(torch.zeros(1, static_len, prompt_embeds.shape[-1], dtype=torch.float16))

        prefill_output = self.prefill_session.forward(*prefill_inputs, *cache_state["past_key_caches"], *cache_state["past_value_caches"])
        prefill_logits, prefill_hidden = extract_logits_and_hidden_states(
            prefill_output, torch.device("cpu"), prompt_len
        )
        next_token = torch.argmax(prefill_logits[:, -1, :], dim=-1, keepdim=True)
        cache_state["past_seq_length"] = prompt_len

        # Build step_embeds and step_hiddens for prompt
        prompt_embeds_fp16 = prompt_embeds[:, :prompt_len, :].detach().cpu().to(torch.float16)
        all_step_embeds = [prompt_embeds_fp16[:, i : i + 1, :] for i in range(prompt_len)]
        all_step_hiddens: list[torch.Tensor] = []
        if prefill_hidden is not None:
            prefill_hidden_fp16 = prefill_hidden[:, :prompt_len, :].detach().cpu().to(torch.float16)
            all_step_hiddens = [prefill_hidden_fp16[:, i : i + 1, :] for i in range(prompt_len)]
        else:
            all_step_hiddens = [prompt_embeds_fp16[:, i : i + 1, :] for i in range(prompt_len)]

        # Yield prefill chunk
        yield ThinkerPrefillChunk(
            token_ids=input_ids_cpu,
            step_embeds=all_step_embeds,
            step_hiddens=all_step_hiddens,
            tts_bos_embed=None,  # caller should inject TTS embeddings
            tts_eos_embed=None,
            tts_pad_embed=None,
            speaker_id=0,
            is_finished=False,
        )

        # ---- Decode loop ----
        one_len = torch.ones(1, dtype=torch.int32)
        decode_step = 0
        generated_tokens: list[torch.Tensor] = [next_token]

        for _ in range(max_new_tokens - 1):
            token_id_val = int(next_token.item())
            if token_id_val in self.eos_token_ids:
                break
            if int(cache_state["past_seq_length"]) + 1 > self.kv_cache_info["shape"][2]:
                logger.warning("KV cache full at step %d, stopping thinker decode", decode_step + 1)
                break

            decode_embeds = self.token_embedding(next_token.cpu()).detach().cpu().to(torch.float16)
            decode_inputs = [decode_embeds]
            if self.supports_position_ids:
                pos = torch.full((3,), int(cache_state["past_seq_length"]), dtype=torch.int32)
                decode_inputs.extend([pos[0], pos[1], pos[2]])
            decode_inputs.extend([torch.tensor([cache_state["past_seq_length"]], dtype=torch.int32), one_len])
            if self.supports_deepstack:
                for _ in range(3):
                    decode_inputs.append(torch.zeros(1, 1, decode_embeds.shape[-1], dtype=torch.float16))

            decode_output = self.decode_session.forward(
                *decode_inputs, *cache_state["past_key_caches"], *cache_state["past_value_caches"]
            )
            decode_logits, decode_hidden = extract_logits_and_hidden_states(decode_output, torch.device("cpu"), 1)
            step_embed = self.token_embedding(next_token.cpu()).detach().cpu()
            next_token = torch.argmax(decode_logits[:, -1, :], dim=-1, keepdim=True)
            generated_tokens.append(next_token)
            cache_state["past_seq_length"] += 1
            decode_step += 1

            step_hidden = step_embed if decode_hidden is None else decode_hidden[:, -1:, :].detach().cpu()
            yield ThinkerDecodeChunk(
                token_id=next_token,
                step_embeds=step_embed.to(torch.float16),
                step_hidden=step_hidden.to(torch.float16),
                is_finished=False,
            )

        # Yield final finished chunk
        output_ids = torch.cat(generated_tokens, dim=-1).to(input_ids_cpu.dtype)
        yield ThinkerDecodeChunk(
            token_id=next_token,
            step_embeds=torch.empty(0),
            step_hidden=torch.empty(0),
            is_finished=True,
        )
