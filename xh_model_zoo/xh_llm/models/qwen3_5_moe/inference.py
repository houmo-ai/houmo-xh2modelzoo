# Copyright 2025 HOUMO AI
#
# File: inference.py
# Description:
#   Qwen3.5-MoE HMONNX Inference implementation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from transformers import AutoTokenizer
from xhquant.api import CacheTensor, GoldenMixin, HMONNXInference

from ....utils import DeviceDtypeMixin
from ....xh_llm.utils import decode_next_token


class Qwen3_5MoeInference(DeviceDtypeMixin):
    """HMONNX inference engine for Qwen3.5-MoE (hybrid full-attention + linear-attention + MoE).

    Loads prefill/decode ONNX files from meta.json and manages:
    - KV cache (full-attention layers)
    - Conv cache + recurrent state (linear-attention layers)
    - Token embedding
    - M-RoPE position IDs
    """

    def __init__(
        self,
        model_config_file: str,
        fast_mode: bool = False,  # fast mode has known bugs for this model (MoE shape, RMSNorm force_fp32)
        device: str = "cuda",
        execution_device: str = "cuda",
    ):
        super().__init__()

        self.fast_mode = fast_mode
        self._device = torch.device(device)
        self._set_exec_device(torch.device(execution_device))

        model_dir = Path(model_config_file).parent
        meta_info = json.load(open(model_config_file, "r"))
        self.meta_info = meta_info

        # HMONNX files
        self.prefill_onnx_file = model_dir / meta_info["prefill_onnx"]
        self.decode_onnx_file = model_dir / meta_info["decode_onnx"]

        # ---- Full-attention KV cache ----
        kv_cache_info = meta_info["kv_cache"]
        kv_cache_shape = kv_cache_info["shape"]
        num_full_attn_layers = kv_cache_info["num_decoder_layers"]

        past_key_caches: List[CacheTensor] = []
        past_value_caches: List[CacheTensor] = []
        for _ in range(num_full_attn_layers):
            past_key_caches.append(CacheTensor(torch.zeros(kv_cache_shape, dtype=torch.float16)))
            past_value_caches.append(CacheTensor(torch.zeros(kv_cache_shape, dtype=torch.float16)))

        self.past_key_caches = past_key_caches
        self.past_value_caches = past_value_caches

        # ---- Linear-attention conv/recurrent cache ----
        linear_cache_info = meta_info.get("linear_cache", {})
        linear_layers_meta = linear_cache_info.get("layers", [])

        past_conv_caches: List[CacheTensor] = []
        past_recurrent_states: List[CacheTensor] = []
        for layer_meta in linear_layers_meta:
            conv_shape = layer_meta["conv_shape"]
            recurrent_shape = layer_meta["recurrent_shape"]
            past_conv_caches.append(CacheTensor(torch.zeros(conv_shape, dtype=torch.float16)))
            past_recurrent_states.append(CacheTensor(torch.zeros(recurrent_shape, dtype=torch.float16)))

        self.past_conv_caches = past_conv_caches
        self.past_recurrent_states = past_recurrent_states
        self.linear_attention_layer_indices = linear_cache_info.get("layer_indices", [])

        # ---- Tokenizer ----
        hf_model_config_dir = str(model_dir / meta_info["hf_config"])
        self.tokenizer = AutoTokenizer.from_pretrained(hf_model_config_dir)

        # ---- Token embedding ----
        token_embedding_state_dict = torch.load(
            model_dir / meta_info["token_embedding_file"],
            map_location="cpu",
            weights_only=True,
        )
        self.token_embedding = nn.Embedding(
            token_embedding_state_dict["weight"].shape[0],
            token_embedding_state_dict["weight"].shape[1],
        ).to(torch.float16)
        self.token_embedding.load_state_dict(token_embedding_state_dict)

        self.batch_size = 1
        self.prefill_input_sequence_length = meta_info["wrap_cfg"]["input_sequence_length"]
        self.input_sequence_length = self.prefill_input_sequence_length
        self.pad_token_id = self.tokenizer.eos_token_id
        self._phase_prefill = True

        self.prefill_session: Optional[HMONNXInference] = None
        self.decode_session: Optional[HMONNXInference] = None

    def set_phase_prefill(self, prefill: bool):
        self._phase_prefill = prefill
        if prefill:
            if self.prefill_session is None:
                self.init_prefill()
            self.input_sequence_length = self.prefill_input_sequence_length
        else:
            self.prefill_session = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if self.decode_session is None:
                self.init_decode()
            self.input_sequence_length = 1

    def init_prefill(self):
        if self.prefill_session is not None:
            return
        self.prefill_session = HMONNXInference(self.prefill_onnx_file)
        if self.fast_mode:
            self.prefill_session.to_fast_mode()
        self.prefill_session.exec_device = self.execution_device
        self.prefill_session.to(self._device)

    def init_decode(self):
        if self.decode_session is not None:
            return
        self.decode_session = HMONNXInference(self.decode_onnx_file)
        if self.fast_mode:
            self.decode_session.to_fast_mode()
        self.decode_session.exec_device = self.execution_device
        self.decode_session.to(self._device)

    def get_input_sequence_length(self) -> int:
        return self.input_sequence_length

    def set_input_sequence_length(self, input_sequence_length: int):
        self.input_sequence_length = input_sequence_length

    def _make_position_ids(self, seq_length: int, past_seq_length: int, device: torch.device) -> Tuple[Tensor, Tensor, Tensor]:
        """Build M-RoPE position IDs (time=height=width for text-only input)."""
        pos = torch.arange(past_seq_length, past_seq_length + seq_length, dtype=torch.long, device=device)
        return pos, pos, pos

    def prepare_inputs(
        self,
        data: dict,
        input_sequence_length: int,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, List[CacheTensor], List[CacheTensor], List[CacheTensor], List[CacheTensor]]:
        input_ids = data["input_ids"]
        assert self.token_embedding is not None, "Token embedding is not available."
        assert input_ids.shape[0] == 1, "Batch size must be 1 in inference mode."
        seq_length = input_ids.shape[1]
        input_ids = input_ids.to(self.execution_device)

        assert seq_length <= input_sequence_length, (
            f"Input sequence length ({seq_length}) exceeds max ({input_sequence_length})"
        )
        if input_sequence_length > seq_length:
            padding = torch.full(
                (1, input_sequence_length - seq_length),
                self.pad_token_id,
                dtype=torch.long,
                device=self.execution_device,
            )
            input_ids = torch.cat([input_ids, padding], dim=-1)

        inputs_embeds = self.token_embedding.to(self.execution_device)(input_ids)

        past_seq_length = data["past_seq_length"]
        assert past_seq_length >= 0

        time_pos, hight_pos, width_pos = self._make_position_ids(
            seq_length, past_seq_length, self.execution_device
        )
        # Pad position IDs to input_sequence_length
        if input_sequence_length > seq_length:
            pad_len = input_sequence_length - seq_length
            last_pos = time_pos[-1:] if seq_length > 0 else torch.zeros(1, dtype=torch.long, device=self.execution_device)
            pos_pad = last_pos.expand(pad_len)
            time_pos = torch.cat([time_pos, pos_pad])
            hight_pos = torch.cat([hight_pos, pos_pad])
            width_pos = torch.cat([width_pos, pos_pad])

        return (
            inputs_embeds.to(self.execution_device),
            time_pos.unsqueeze(0).to(torch.int32),
            hight_pos.unsqueeze(0).to(torch.int32),
            width_pos.unsqueeze(0).to(torch.int32),
            torch.tensor([past_seq_length], dtype=torch.int32, device=self.execution_device),
            torch.tensor([seq_length], dtype=torch.int32, device=self.execution_device),
            self.past_key_caches,
            self.past_value_caches,
            self.past_conv_caches,
            self.past_recurrent_states,
        )

    def _forward(
        self,
        inputs_embeds: Tensor,
        time_position_ids: Tensor,
        hight_position_ids: Tensor,
        width_position_ids: Tensor,
        past_seq_length: Tensor,
        current_input_length: Tensor,
        linear_attn_mask: Tensor,
        past_key_caches: List[CacheTensor],
        past_value_caches: List[CacheTensor],
        past_conv_caches: List[CacheTensor],
        past_recurrent_states: List[CacheTensor],
    ) -> torch.FloatTensor:
        """Core forward pass dispatching to prefill or decode HMONNX session."""
        if self._phase_prefill:
            self.init_prefill()
            assert self.prefill_session is not None
            out = self.prefill_session(
                inputs_embeds.to(self._device),
                time_position_ids.to(self._device),
                hight_position_ids.to(self._device),
                width_position_ids.to(self._device),
                past_seq_length.to(self._device),
                current_input_length.to(self._device),
                linear_attn_mask.to(self._device),
                *past_key_caches,
                *past_value_caches,
                *past_conv_caches,
                *past_recurrent_states,
            )
            if isinstance(self.prefill_session, GoldenMixin):
                self.prefill_session.update_step()
        else:
            self.init_decode()
            assert self.decode_session is not None
            out = self.decode_session(
                inputs_embeds.to(self._device),
                time_position_ids.to(self._device),
                hight_position_ids.to(self._device),
                width_position_ids.to(self._device),
                past_seq_length.to(self._device),
                current_input_length.to(self._device),
                linear_attn_mask.to(self._device),
                *past_key_caches,
                *past_value_caches,
                *past_conv_caches,
                *past_recurrent_states,
            )
            if isinstance(self.decode_session, GoldenMixin):
                self.decode_session.update_step()

        # The ONNX outputs: (logits, conv_cache_out_0..N-1, recurrent_state_out_0..M-1)
        # We must propagate the updated linear-attention states back into the
        # CacheTensor objects so that the next forward call sees correct state.
        if isinstance(out, tuple) and len(out) > 1:
            n_conv = len(past_conv_caches)
            n_rec = len(past_recurrent_states)
            for i in range(n_conv):
                past_conv_caches[i].data = out[1 + i].detach()
            for i in range(n_rec):
                past_recurrent_states[i].data = out[1 + n_conv + i].detach()
            return out[0]
        return out

    def forward(
        self,
        inputs_embeds: Tensor,
        past_seq_length: Tensor,
        current_input_length: Tensor,
        past_key_caches: List[CacheTensor],
        past_value_caches: List[CacheTensor],
    ) -> torch.FloatTensor:
        """Simplified forward without linear cache (backward-compatible interface)."""
        seq_len = inputs_embeds.shape[1]
        past_len = int(past_seq_length.item())
        time_pos, hight_pos, width_pos = self._make_position_ids(seq_len, past_len, inputs_embeds.device)
        lin_mask = torch.ones(1, seq_len, dtype=inputs_embeds.dtype, device=inputs_embeds.device)
        return self._forward(
            inputs_embeds,
            time_pos.unsqueeze(0).to(torch.int32),
            hight_pos.unsqueeze(0).to(torch.int32),
            width_pos.unsqueeze(0).to(torch.int32),
            past_seq_length,
            current_input_length,
            lin_mask,
            past_key_caches,
            past_value_caches,
            self.past_conv_caches,
            self.past_recurrent_states,
        )

    @torch.no_grad()
    def generate(self, messages, enable_thinking: bool = False, max_new_tokens: int = 256):
        """Simple single-turn generation returning (token_ids, text)."""
        assert self.batch_size == 1

        texts = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            enable_thinking=enable_thinking,
            add_generation_prompt=True,
        )

        model_inputs = self.tokenizer([texts] if isinstance(texts, str) else texts, padding=False, return_tensors="pt")
        batch_input_ids = model_inputs.input_ids.cpu().numpy().tolist()
        prefill_len = len(batch_input_ids[0])

        # ----- Prefill -----
        data_prefill = {"input_ids": torch.tensor(batch_input_ids), "past_seq_length": 0}
        (
            inputs_embeds,
            time_pos, hight_pos, width_pos,
            past_seq_length_t,
            seq_length_t,
            past_key_caches, past_value_caches,
            past_conv_caches, past_recurrent_states,
        ) = self.prepare_inputs(data_prefill, self.prefill_input_sequence_length)

        lin_mask = torch.ones(1, self.prefill_input_sequence_length, dtype=inputs_embeds.dtype, device=self.execution_device)
        prefill_logits = self._forward(
            inputs_embeds, time_pos, hight_pos, width_pos,
            past_seq_length_t, seq_length_t,
            lin_mask,
            past_key_caches, past_value_caches,
            past_conv_caches, past_recurrent_states,
        )
        logits_t = prefill_logits[0] if isinstance(prefill_logits, (tuple, list)) else prefill_logits
        next_token_id, _ = decode_next_token(self.tokenizer, logits_t)

        # ----- Decode loop -----
        generated_ids = next_token_id.cpu().tolist()[0]  # list of token ids
        past_len = prefill_len

        self.set_phase_prefill(False)
        try:
            for _ in range(max_new_tokens - 1):
                if generated_ids[-1] == self.tokenizer.eos_token_id:
                    break
                data_decode = {
                    "input_ids": torch.tensor([[generated_ids[-1]]]),
                    "past_seq_length": past_len,
                }
                (
                    inputs_embeds,
                    time_pos, hight_pos, width_pos,
                    past_seq_length_t,
                    seq_length_t,
                    past_key_caches, past_value_caches,
                    past_conv_caches, past_recurrent_states,
                ) = self.prepare_inputs(data_decode, 1)
                lin_mask = torch.ones(1, 1, dtype=inputs_embeds.dtype, device=self.execution_device)
                decode_logits = self._forward(
                    inputs_embeds, time_pos, hight_pos, width_pos,
                    past_seq_length_t, seq_length_t,
                    lin_mask,
                    past_key_caches, past_value_caches,
                    past_conv_caches, past_recurrent_states,
                )
                decode_logits_t = decode_logits[0] if isinstance(decode_logits, (tuple, list)) else decode_logits
                tok_id, _ = decode_next_token(self.tokenizer, decode_logits_t)
                next_id = tok_id.cpu().item() if tok_id.numel() == 1 else tok_id.cpu().tolist()[0][0]
                generated_ids.append(next_id)
                past_len += 1
        finally:
            self.set_phase_prefill(True)

        generate_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return generated_ids, generate_text

    @torch.no_grad()
    def prefill_only(self, input_ids: torch.Tensor) -> Optional[torch.Tensor]:
        """Run prefill over input_ids in fixed-length chunks and return all logits.

        The ONNX prefill session has a fixed input sequence length
        (``self.prefill_input_sequence_length``).  When ``input_ids`` is longer,
        this method splits it into consecutive non-overlapping chunks, runs each
        chunk independently (each with ``past_seq_length=0``), and concatenates
        the resulting logits.  This is stateless — the persistent KV cache is
        never modified.

        Args:
            input_ids: [1, seq_len] integer token ids.

        Returns:
            logits: [1, seq_len, vocab_size] float tensor, or None on error.
        """
        seq_len = input_ids.shape[1]
        if seq_len == 0:
            return None

        isl = self.prefill_input_sequence_length  # fixed ONNX window
        self.set_phase_prefill(True)
        all_logits: list[torch.Tensor] = []

        # Save linear-attention cache state so we can restore it after each
        # stateless chunk (each chunk uses past_seq_length=0 and must start
        # from the same zero-initialised state).
        saved_conv = [c.data.clone() for c in self.past_conv_caches]
        saved_rec = [r.data.clone() for r in self.past_recurrent_states]

        pos = 0
        while pos < seq_len:
            chunk_end = min(pos + isl, seq_len)
            chunk_len = chunk_end - pos
            chunk = input_ids[:, pos:chunk_end]

            # Reset linear caches to the saved (zero-init) state before each chunk.
            for i, c in enumerate(self.past_conv_caches):
                c.data = saved_conv[i].clone()
            for i, r in enumerate(self.past_recurrent_states):
                r.data = saved_rec[i].clone()

            data = {"input_ids": chunk, "past_seq_length": 0}
            (
                inputs_embeds,
                time_pos, hight_pos, width_pos,
                past_seq_length_t, seq_length_t,
                past_key_caches, past_value_caches,
                past_conv_caches, past_recurrent_states,
            ) = self.prepare_inputs(data, isl)
            lin_mask = torch.ones(1, isl, dtype=inputs_embeds.dtype, device=self.execution_device)

            out = self._forward(
                inputs_embeds, time_pos, hight_pos, width_pos,
                past_seq_length_t, seq_length_t,
                lin_mask,
                past_key_caches, past_value_caches,
                past_conv_caches, past_recurrent_states,
            )
            logits = out[0] if isinstance(out, (tuple, list)) else out
            if logits is None:
                # Restore caches before returning
                for i, c in enumerate(self.past_conv_caches):
                    c.data = saved_conv[i]
                for i, r in enumerate(self.past_recurrent_states):
                    r.data = saved_rec[i]
                return None
            all_logits.append(logits[:, :chunk_len, :])
            pos = chunk_end

        # Restore linear cache state to what it was before prefill_only().
        for i, c in enumerate(self.past_conv_caches):
            c.data = saved_conv[i]
        for i, r in enumerate(self.past_recurrent_states):
            r.data = saved_rec[i]

        return torch.cat(all_logits, dim=1) if all_logits else None
