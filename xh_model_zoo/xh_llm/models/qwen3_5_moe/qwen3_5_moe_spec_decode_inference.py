# Copyright 2025 HOUMO AI
#
# File: qwen3_5_moe_spec_decode_inference.py
# Description:
#   Speculative-decoding runtime for Qwen3.5-MoE with MTP and DFlash draft models.
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

"""Speculative-decoding runtime for Qwen3.5-MoE."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from xhquant.api import CacheTensor
from xhquant.xhonnxruntime import HMONNXGrapInference

from .inference import Qwen3_5MoeInference
from ..qwen3_5.qwen3_5_onnx_model import (
    _alloc_cache_inputs,
    _as_cache_value,
    _clone_cache_value,
)


def _pad_hidden_tensor(hidden: torch.Tensor, target_seq_len: int) -> torch.Tensor:
    if hidden.shape[1] >= target_seq_len:
        return hidden[:, :target_seq_len, :]
    pad = torch.zeros(
        hidden.shape[0],
        target_seq_len - hidden.shape[1],
        hidden.shape[2],
        dtype=hidden.dtype,
        device=hidden.device,
    )
    return torch.cat([hidden, pad], dim=1)


def _build_runtime_linear_attn_mask(
    valid_len: int, total_len: int, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    mask = torch.zeros((1, total_len), dtype=dtype, device=device)
    if valid_len > 0:
        mask[:, :valid_len] = 1
    return mask


def load_moe_inference(meta_json_path: str, **kwargs) -> "Qwen3_5MoeInference":
    """Factory: load the right inference engine based on meta.json.

    Returns :class:`Qwen3_5MoeSpecDecodeInference` when ``spec_decode_mode``
    is present in the meta.json, otherwise :class:`Qwen3_5MoeInference`.
    """
    with open(meta_json_path) as f:
        meta = json.load(f)
    if meta.get("spec_decode_mode"):
        return Qwen3_5MoeSpecDecodeInference(meta_json_path, **kwargs)
    return Qwen3_5MoeInference(meta_json_path, **kwargs)


class Qwen3_5MoeSpecDecodeInference(Qwen3_5MoeInference):
    """MoE speculative-decode inference with batched verify."""

    def __init__(self, model_config_file: str, **kwargs):
        super().__init__(model_config_file, **kwargs)

        meta_info = self.meta_info
        spec_decode_mode = meta_info.get("spec_decode_mode")
        if not spec_decode_mode:
            raise ValueError(
                f"meta.json at '{model_config_file}' does not contain 'spec_decode_mode'. "
                "Use Qwen3_5MoeInference for non-spec-decode inference."
            )

        self.spec_decode_mode: str = spec_decode_mode
        self.spec_decode_block_size: int = int(meta_info.get("spec_decode_block_size", 4))
        self.spec_decode_verify_length: int = int(
            meta_info.get("spec_decode_verify_length", self.spec_decode_block_size + 1)
        )
        self.spec_decode_hidden_output_name: str = meta_info.get(
            "spec_decode_hidden_output_name", "pre_norm_hidden"
        )

        model_dir = Path(model_config_file).parent

        draft_prefill_onnx_rel = meta_info.get("draft_prefill_onnx_file")
        draft_context_onnx_rel = meta_info.get("draft_context_onnx_file")
        draft_decode_onnx_rel = meta_info.get("draft_decode_onnx_file")

        if draft_decode_onnx_rel is None:
            raise ValueError(
                "'draft_decode_onnx_file' missing from meta.json. "
                "Re-export the model with --spec_decode_mode mtp."
            )

        self._draft_prefill_onnx: Optional[Path] = (
            model_dir / draft_prefill_onnx_rel if draft_prefill_onnx_rel else None
        )
        self._draft_context_onnx: Optional[Path] = (
            model_dir / draft_context_onnx_rel if draft_context_onnx_rel else None
        )
        self._draft_decode_onnx: Path = model_dir / draft_decode_onnx_rel

        self._draft_prefill_session: Optional[HMONNXGrapInference] = None
        self._draft_context_session: Optional[HMONNXGrapInference] = None
        self._draft_decode_session: Optional[HMONNXGrapInference] = None

        self._mtp_k_cache: Optional[CacheTensor] = None
        self._mtp_v_cache: Optional[CacheTensor] = None
        self._dflash_cache_state: Optional[Dict[str, torch.Tensor]] = None

    # ------------------------------------------------------------------
    # Draft session management
    # ------------------------------------------------------------------

    def _ensure_draft_decode(self) -> HMONNXGrapInference:
        if self._draft_decode_session is None:
            self._draft_decode_session = HMONNXGrapInference(str(self._draft_decode_onnx))
            self._draft_decode_session.exec_device = self.execution_device
            self._draft_decode_session.to(self._device)
            if self.spec_decode_mode == "mtp" and self._mtp_k_cache is None:
                key_info = self._draft_decode_session.get_input("past_key_cache")
                value_info = self._draft_decode_session.get_input("past_value_cache")
                self._mtp_k_cache = CacheTensor(
                    torch.zeros(key_info.shape, dtype=key_info.dtype, device=self._device)
                )
                self._mtp_v_cache = CacheTensor(
                    torch.zeros(value_info.shape, dtype=value_info.dtype, device=self._device)
                )
        return self._draft_decode_session

    def _ensure_draft_prefill(self) -> Optional[HMONNXGrapInference]:
        if self._draft_prefill_session is None and self._draft_prefill_onnx is not None:
            self._draft_prefill_session = HMONNXGrapInference(str(self._draft_prefill_onnx))
            self._draft_prefill_session.exec_device = self.execution_device
            self._draft_prefill_session.to(self._device)
            if self._mtp_k_cache is None:
                key_info = self._draft_prefill_session.get_input("past_key_cache")
                value_info = self._draft_prefill_session.get_input("past_value_cache")
                self._mtp_k_cache = CacheTensor(
                    torch.zeros(key_info.shape, dtype=key_info.dtype, device=self._device)
                )
                self._mtp_v_cache = CacheTensor(
                    torch.zeros(value_info.shape, dtype=value_info.dtype, device=self._device)
                )
        return self._draft_prefill_session

    def _ensure_draft_context(self) -> Optional[HMONNXGrapInference]:
        if self._draft_context_session is None and self._draft_context_onnx is not None:
            self._draft_context_session = HMONNXGrapInference(str(self._draft_context_onnx))
            self._draft_context_session.exec_device = self.execution_device
            self._draft_context_session.to(self._device)
        return self._draft_context_session

    def _reset_mtp_cache(self) -> None:
        self._ensure_draft_prefill() or self._ensure_draft_decode()
        assert self._mtp_k_cache is not None and self._mtp_v_cache is not None
        self._mtp_k_cache.data = torch.zeros_like(self._mtp_k_cache.data)
        self._mtp_v_cache.data = torch.zeros_like(self._mtp_v_cache.data)

    def _ensure_dflash_cache_state(self) -> Dict[str, torch.Tensor]:
        session = self._ensure_draft_context() or self._ensure_draft_decode()
        if session is None:
            raise RuntimeError("DFlash draft sessions are not initialized.")
        if self._dflash_cache_state is None:
            self._dflash_cache_state = _alloc_cache_inputs(session, self._device)
        return self._dflash_cache_state

    @staticmethod
    def _run_draft(session: HMONNXGrapInference, feed: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Run a draft session and return outputs as a named dict."""
        outputs = session.run(feed)
        if not isinstance(outputs, (tuple, list)):
            outputs = (outputs,)
        names = session.get_output_names()
        return dict(zip(names, outputs))

    # ------------------------------------------------------------------
    # Target model forward with hidden output
    # ------------------------------------------------------------------

    def _forward_with_hidden(
        self,
        inputs_embeds: torch.Tensor,
        time_position_ids: torch.Tensor,
        hight_position_ids: torch.Tensor,
        width_position_ids: torch.Tensor,
        past_seq_length: torch.Tensor,
        current_input_length: torch.Tensor,
        linear_attn_mask: torch.Tensor,
        past_key_caches: List[CacheTensor],
        past_value_caches: List[CacheTensor],
        past_conv_caches: List[CacheTensor],
        past_recurrent_states: List[CacheTensor],
        update_linear_cache: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Tuple[torch.Tensor, ...]]:
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

        if not isinstance(out, tuple):
            out = (out,)

        n_conv = len(past_conv_caches)
        n_rec = len(past_recurrent_states)
        if update_linear_cache:
            for i in range(n_conv):
                past_conv_caches[i].data = out[1 + i].detach()
            for i in range(n_rec):
                past_recurrent_states[i].data = out[1 + n_conv + i].detach()

        logits = out[0]
        hidden_idx = 1 + n_conv + n_rec
        pre_norm_hidden = out[hidden_idx] if len(out) > hidden_idx else None
        return logits, pre_norm_hidden, out

    @staticmethod
    def _select_hidden_step(hidden_states: Optional[torch.Tensor], step_idx: int) -> Optional[torch.Tensor]:
        if hidden_states is None:
            return None
        if hidden_states.dim() < 3 or hidden_states.shape[1] == 1:
            return hidden_states
        return hidden_states[:, step_idx : step_idx + 1, :]

    @staticmethod
    def _select_logits_step(logits: torch.Tensor, step_idx: int) -> torch.Tensor:
        if logits.dim() < 3 or logits.shape[1] == 1:
            return logits[:, -1, :]
        return logits[:, step_idx, :]

    @staticmethod
    def _apply_verify_linear_cache_outputs(
        past_conv_caches: List[CacheTensor],
        past_recurrent_states: List[CacheTensor],
        raw_outputs: Tuple[torch.Tensor, ...],
        accepted_steps: int,
    ) -> None:
        n_conv = len(past_conv_caches)
        n_rec = len(past_recurrent_states)
        for idx in range(n_conv):
            conv_out = raw_outputs[1 + idx]
            cache_len = past_conv_caches[idx].data.shape[-1]
            if conv_out.dim() >= 3 and conv_out.shape[-1] != cache_len:
                start = max(accepted_steps - 1, 0)
                end = start + cache_len
                conv_out = conv_out[..., start:end]
            past_conv_caches[idx].data = conv_out.detach()
        for idx in range(n_rec):
            recurrent_out = raw_outputs[1 + n_conv + idx]
            if recurrent_out.dim() >= 5:
                recurrent_out = recurrent_out[accepted_steps - 1]
            past_recurrent_states[idx].data = recurrent_out.detach()

    def _embed_token_ids(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.token_embedding.to(self._device)(token_ids.to(self._device)).to(torch.float16)

    # ------------------------------------------------------------------
    # MTP draft step
    # ------------------------------------------------------------------

    def _mtp_prefill_warmup(
        self,
        next_token_embedding: torch.Tensor,
        pre_norm_hidden: torch.Tensor,
        past_seq_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Warm-up MTP cache over the prefill sequence.

        Args:
            next_token_embedding: [1, prefill_len, hidden] – token embeddings
                starting from position 1 (i.e. target token shifted by 1).
            pre_norm_hidden: [1, prefill_len, hidden] – target pre-norm hidden.
            past_seq_len: 0 for cold start.

        Returns:
            (final_logit [1, 1, vocab], final_pre_norm_out [1, 1, hidden])
        """
        session = self._ensure_draft_prefill()
        if session is None:
            raise RuntimeError("MTP prefill ONNX not available; cannot warm up draft cache.")

        prefill_isl = int(session.get_input("pre_norm_hidden").shape[1])
        # Pad / trim inputs to prefill_isl.
        B, T, H = next_token_embedding.shape
        if T < prefill_isl:
            pad = torch.zeros(B, prefill_isl - T, H, dtype=next_token_embedding.dtype, device=next_token_embedding.device)
            next_token_embedding = torch.cat([next_token_embedding, pad], dim=1)
            pre_norm_hidden = torch.cat([pre_norm_hidden, pad[:, :, :pre_norm_hidden.shape[-1]]], dim=1)
        else:
            next_token_embedding = next_token_embedding[:, :prefill_isl, :]
            pre_norm_hidden = pre_norm_hidden[:, :prefill_isl, :]

        past_seq_t = torch.tensor([past_seq_len], dtype=torch.int32, device=self._device)
        cur_len_t = torch.tensor([T], dtype=torch.int32, device=self._device)

        feed: Dict[str, torch.Tensor] = {}
        for name in session.get_input_names():
            if name == "next_token_embedding":
                feed[name] = next_token_embedding.to(self._device)
            elif name == "pre_norm_hidden":
                feed[name] = pre_norm_hidden.to(self._device)
            elif name in ("past_seq_length", "valid_length"):
                feed[name] = past_seq_t
            elif name in ("current_input_length", "current_length"):
                feed[name] = cur_len_t
            elif name == "past_key_cache":
                feed[name] = self._mtp_k_cache
            elif name == "past_value_cache":
                feed[name] = self._mtp_v_cache
            else:
                info = session.get_input(name)
                feed[name] = torch.zeros(
                    info.shape, dtype=info.dtype, device=self._device
                )
        out = self._run_draft(session, feed)
        self._mtp_k_cache.data = out["present_key_cache"].detach()
        self._mtp_v_cache.data = out["present_value_cache"].detach()
        # Return the last valid position.
        return out["logits"][:, T - 1:T, :], out["pre_norm_out"][:, T - 1:T, :]

    def _mtp_decode_step(
        self,
        next_token_embedding: torch.Tensor,
        pre_norm_hidden: torch.Tensor,
        past_seq_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single MTP draft decode step.

        Args:
            next_token_embedding: [1, 1, hidden]
            pre_norm_hidden: [1, 1, hidden]
            past_seq_len: scalar

        Returns:
            (logits [1, 1, vocab], pre_norm_out [1, 1, hidden])
        """
        session = self._ensure_draft_decode()

        past_seq_t = torch.tensor([past_seq_len], dtype=torch.int32, device=self._device)
        cur_len_t = torch.ones(1, dtype=torch.int32, device=self._device)

        feed: Dict[str, torch.Tensor] = {}
        for name in session.get_input_names():
            if name == "next_token_embedding":
                feed[name] = next_token_embedding.to(self._device)
            elif name == "pre_norm_hidden":
                feed[name] = pre_norm_hidden.to(self._device)
            elif name in ("past_seq_length", "valid_length"):
                feed[name] = past_seq_t
            elif name in ("current_input_length", "current_length"):
                feed[name] = cur_len_t
            elif name == "past_key_cache":
                feed[name] = self._mtp_k_cache
            elif name == "past_value_cache":
                feed[name] = self._mtp_v_cache
            else:
                info = session.get_input(name)
                feed[name] = torch.zeros(
                    info.shape, dtype=info.dtype, device=self._device
                )
        out = self._run_draft(session, feed)
        self._mtp_k_cache.data = out["present_key_cache"].detach()
        self._mtp_v_cache.data = out["present_value_cache"].detach()
        return out["logits"], out["pre_norm_out"]

    def _append_dflash_context(
        self,
        target_hidden: Optional[torch.Tensor],
        past_seq_len: int,
    ) -> None:
        if target_hidden is None or target_hidden.shape[1] == 0:
            return
        session = self._ensure_draft_context()
        if session is None:
            raise RuntimeError("DFlash context session is not available.")
        cache_state = self._ensure_dflash_cache_state()
        actual_input_length = int(target_hidden.shape[1])
        context_seq_len = int(session.get_input("target_hidden").shape[1])
        target_hidden = _pad_hidden_tensor(
            target_hidden.to(device=self._device, dtype=torch.float16), context_seq_len
        )
        feed: Dict[str, torch.Tensor] = {}
        for name in session.get_input_names():
            if name == "target_hidden":
                feed[name] = target_hidden
            elif name in ("past_seq_length", "valid_length"):
                feed[name] = torch.tensor([past_seq_len], dtype=torch.int32, device=self._device)
            elif name in ("current_input_length", "current_length"):
                feed[name] = torch.tensor([actual_input_length], dtype=torch.int32, device=self._device)
            elif name in cache_state:
                feed[name] = cache_state[name]
            else:
                info = session.get_input(name)
                feed[name] = torch.zeros(info.shape, dtype=info.dtype, device=self._device)
        out = self._run_draft(session, feed)
        for name, tensor in list(cache_state.items()):
            present_name = name.replace("past_", "present_", 1)
            if present_name in out:
                cache_state[name] = _as_cache_value(tensor, out[present_name])

    def _build_dflash_decode_feed(
        self,
        noise_embedding: torch.Tensor,
        past_seq_len: int,
        cache_state: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        session = self._ensure_draft_decode()
        cache_length = next(iter(cache_state.values())).shape[2]
        attn_mask = torch.full(
            (noise_embedding.shape[0], cache_length),
            -10000.0,
            dtype=noise_embedding.dtype,
            device=self._device,
        )
        draft_visible_length = min(
            cache_length, int(past_seq_len) + int(noise_embedding.shape[1])
        )
        attn_mask[:, :draft_visible_length] = 0
        feed: Dict[str, torch.Tensor] = {}
        for name in session.get_input_names():
            if name == "noise_embedding":
                feed[name] = noise_embedding
            elif name in ("past_seq_length", "valid_length"):
                feed[name] = torch.tensor([past_seq_len], dtype=torch.int32, device=self._device)
            elif name in ("current_input_length", "current_length"):
                feed[name] = torch.tensor([noise_embedding.shape[1]], dtype=torch.int32, device=self._device)
            elif name == "attn_mask":
                feed[name] = attn_mask
            elif name in cache_state:
                feed[name] = cache_state[name]
            else:
                info = session.get_input(name)
                feed[name] = torch.zeros(info.shape, dtype=info.dtype, device=self._device)
        return feed

    def _run_draft_dflash(self, current_token: int, past_seq_len: int) -> List[int]:
        cache_state = self._ensure_dflash_cache_state()
        block_len = int(self._ensure_draft_decode().get_input("noise_embedding").shape[1])
        block_ids = torch.full(
            (1, block_len),
            self.pad_token_id,
            dtype=torch.long,
            device=self._device,
        )
        block_ids[0, 0] = current_token
        noise_embedding = self._embed_token_ids(block_ids)
        out = self._run_draft(
            self._ensure_draft_decode(),
            self._build_dflash_decode_feed(noise_embedding, past_seq_len, cache_state),
        )
        logits = out["logits"]
        draft_tokens: List[int] = []
        max_positions = min(self.spec_decode_block_size + 1, logits.shape[1])
        for idx in range(1, max_positions):
            draft_tokens.append(int(torch.argmax(logits[:, idx : idx + 1, :], dim=-1)[0, 0].item()))
        return draft_tokens

    def _prefill_mtp_chunk(
        self,
        next_token_embedding: torch.Tensor,
        pre_norm_hidden: torch.Tensor,
        past_seq_len: int,
    ) -> None:
        if next_token_embedding.shape[1] == 0:
            return
        self._mtp_prefill_warmup(next_token_embedding, pre_norm_hidden, past_seq_len)

    def generate(self, messages, enable_thinking: bool = False, max_new_tokens: int = 256):
        return self.spec_generate(
            messages,
            enable_thinking=enable_thinking,
            max_new_tokens=max_new_tokens,
        )

    # ------------------------------------------------------------------
    # Speculative generate
    # ------------------------------------------------------------------

    @torch.no_grad()
    def spec_generate(
        self,
        messages,
        enable_thinking: bool = False,
        max_new_tokens: int = 256,
    ) -> Tuple[List[int], str]:
        assert self.batch_size == 1, "spec_generate requires batch_size=1."
        verify_len = self.spec_decode_verify_length
        num_drafts = (
            self.spec_decode_block_size - 1
            if self.spec_decode_mode == "dflash"
            else self.spec_decode_block_size
        )

        texts = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            enable_thinking=enable_thinking,
            add_generation_prompt=True,
        )
        if isinstance(texts, str):
            texts = [texts]
        model_inputs = self.tokenizer(texts, padding=False, return_tensors="pt")
        batch_input_ids = model_inputs.input_ids.cpu()
        prefill_len = batch_input_ids.shape[1]
        dev = self._device

        # ── Prefill ──────────────────────────────────────────────────────────
        data_prefill = {"input_ids": batch_input_ids, "past_seq_length": 0}
        isl = self.prefill_input_sequence_length
        (
            inputs_embeds, time_pos, hight_pos, width_pos,
            past_seq_t, seq_len_t,
            past_key_caches, past_value_caches,
            past_conv_caches, past_recurrent_states,
        ) = self.prepare_inputs(data_prefill, isl)
        lin_mask = _build_runtime_linear_attn_mask(
            int(seq_len_t.item()), isl, inputs_embeds.dtype, dev
        )

        self.set_phase_prefill(True)
        prefill_logits, prefill_hidden_all, _ = self._forward_with_hidden(
            inputs_embeds, time_pos, hight_pos, width_pos,
            past_seq_t, seq_len_t, lin_mask,
            past_key_caches, past_value_caches,
            past_conv_caches, past_recurrent_states,
        )
        self.set_phase_prefill(False)

        # First token from greedy prefill decode.
        last_valid = prefill_len - 1
        current_token = int(
            self._select_logits_step(prefill_logits, last_valid)[0].argmax(dim=-1).item()
        )
        generated_ids: List[int] = [current_token]
        past_len: int = prefill_len

        mtp_past_seq_len = 0
        current_mtp_hidden: Optional[torch.Tensor] = None
        if self.spec_decode_mode == "mtp":
            self._reset_mtp_cache()
            if prefill_hidden_all is not None:
                prefill_hidden_all = prefill_hidden_all[:, :prefill_len, :]
                if prefill_len > 1:
                    next_token_embedding = self._embed_token_ids(batch_input_ids[:, 1:prefill_len].to(dev))
                    self._prefill_mtp_chunk(
                        next_token_embedding,
                        prefill_hidden_all[:, : prefill_len - 1, :],
                        0,
                    )
                    mtp_past_seq_len = prefill_len - 1
                current_mtp_hidden = prefill_hidden_all[:, last_valid : last_valid + 1, :]
        elif self.spec_decode_mode == "dflash":
            self._append_dflash_context(
                prefill_hidden_all[:, :prefill_len, :] if prefill_hidden_all is not None else None,
                0,
            )

        # ── Decode loop ───────────────────────────────────────────────────────
        try:
            step = 0
            while step < max_new_tokens:
                if generated_ids[-1] == self.tokenizer.eos_token_id:
                    break

                draft_tokens: List[int] = []
                mtp_snapshot = None
                if self.spec_decode_mode == "dflash":
                    draft_tokens = self._run_draft_dflash(current_token, past_len)
                else:
                    if current_mtp_hidden is not None:
                        assert self._mtp_k_cache is not None and self._mtp_v_cache is not None
                        mtp_snapshot = (
                            _clone_cache_value(self._mtp_k_cache),
                            _clone_cache_value(self._mtp_v_cache),
                        )
                    mtp_hidden = current_mtp_hidden
                    for k in range(num_drafts):
                        if mtp_hidden is None:
                            break
                        tok_id = current_token if k == 0 else draft_tokens[-1]
                        tok_emb = self._embed_token_ids(
                            torch.tensor([[tok_id]], dtype=torch.long, device=dev)
                        )
                        d_logits, mtp_hidden = self._mtp_decode_step(
                            tok_emb,
                            mtp_hidden.to(dev),
                            mtp_past_seq_len + k,
                        )
                        draft_tokens.append(int(d_logits[0, 0, :].argmax(dim=-1).item()))

                if not draft_tokens:
                    tok_emb = self._embed_token_ids(
                        torch.tensor([[current_token]], dtype=torch.long, device=dev)
                    )
                    vpos = torch.arange(past_len, past_len + 1, dtype=torch.int32, device=dev).unsqueeze(0)
                    v_past = torch.tensor([past_len], dtype=torch.int32, device=dev)
                    v_cur = torch.ones(1, dtype=torch.int32, device=dev)
                    lin_v = _build_runtime_linear_attn_mask(
                        int(v_cur.item()), 1, tok_emb.dtype, dev
                    )
                    v_logits, v_hidden, _ = self._forward_with_hidden(
                        tok_emb, vpos, vpos, vpos, v_past, v_cur, lin_v,
                        past_key_caches, past_value_caches,
                        past_conv_caches, past_recurrent_states,
                    )
                    next_tok = int(v_logits[0, 0, :].argmax(dim=-1).item())
                    generated_ids.append(next_tok)
                    past_len += 1
                    step += 1
                    current_token = next_tok
                    current_mtp_hidden = self._select_hidden_step(v_hidden, 0)
                    if self.spec_decode_mode == "dflash":
                        self._append_dflash_context(v_hidden[:, :1, :] if v_hidden is not None else None, past_len - 1)
                    else:
                        mtp_past_seq_len += 1
                    continue

                # ── Verify: single target-model decode pass ─────────────────
                verify_ids = [current_token] + draft_tokens
                actual_vlen = len(verify_ids)  # <= verify_len
                verify_embeds_t = self._embed_token_ids(
                    torch.tensor([verify_ids], dtype=torch.long, device=dev)
                )

                if actual_vlen < verify_len:
                    pad = torch.zeros(
                        1, verify_len - actual_vlen, verify_embeds_t.shape[2],
                        dtype=verify_embeds_t.dtype, device=dev,
                    )
                    verify_embeds_t = torch.cat([verify_embeds_t, pad], dim=1)

                vpos = torch.arange(past_len, past_len + verify_len, dtype=torch.int32, device=dev).unsqueeze(0)
                v_past = torch.tensor([past_len], dtype=torch.int32, device=dev)
                v_cur = torch.tensor([actual_vlen], dtype=torch.int32, device=dev)
                lin_mask_v = _build_runtime_linear_attn_mask(
                    actual_vlen, verify_len, verify_embeds_t.dtype, dev
                )

                initial_seq_len = past_len
                verify_logits, verify_hidden, verify_outputs = self._forward_with_hidden(
                    verify_embeds_t, vpos, vpos, vpos, v_past, v_cur, lin_mask_v,
                    past_key_caches, past_value_caches,
                    past_conv_caches, past_recurrent_states,
                    update_linear_cache=False,
                )

                # ── Accept / Reject ─────────────────────────────────────────
                accepted = 0
                new_tok = -1
                for k, d_tok in enumerate(draft_tokens):
                    t_logit = verify_logits[0, k, :]  # [vocab]
                    t_tok = int(t_logit.argmax(dim=-1).item())
                    if t_tok == d_tok:
                        accepted += 1
                    else:
                        new_tok = t_tok
                        break
                else:
                    bonus_logit = verify_logits[0, len(draft_tokens), :]
                    new_tok = int(bonus_logit.argmax(dim=-1).item())

                accepted_steps = accepted + 1
                self._apply_verify_linear_cache_outputs(
                    past_conv_caches,
                    past_recurrent_states,
                    verify_outputs,
                    accepted_steps=accepted_steps,
                )

                generated_ids.extend(draft_tokens[:accepted])
                generated_ids.append(new_tok)
                n_added = accepted_steps
                past_len += n_added
                step += n_added
                current_token = new_tok

                current_mtp_hidden = self._select_hidden_step(verify_hidden, accepted)
                accepted_hidden = (
                    verify_hidden[:, :accepted_steps, :]
                    if verify_hidden is not None
                    else None
                )
                if self.spec_decode_mode == "dflash":
                    self._append_dflash_context(accepted_hidden, initial_seq_len)
                else:
                    if mtp_snapshot is not None:
                        assert self._mtp_k_cache is not None and self._mtp_v_cache is not None
                        self._mtp_k_cache.data = mtp_snapshot[0].data.clone()
                        self._mtp_v_cache.data = mtp_snapshot[1].data.clone()
                    if accepted_hidden is not None:
                        mtp_initial_seq_len = mtp_past_seq_len
                        for step_idx in range(accepted_steps):
                            next_token_for_cache = (
                                verify_ids[step_idx + 1]
                                if step_idx < accepted_steps - 1
                                else new_tok
                            )
                            tok_emb = self._embed_token_ids(
                                torch.tensor([[next_token_for_cache]], dtype=torch.long, device=dev)
                            )
                            self._mtp_decode_step(
                                tok_emb,
                                accepted_hidden[:, step_idx : step_idx + 1, :],
                                mtp_initial_seq_len + step_idx,
                            )
                        mtp_past_seq_len = mtp_initial_seq_len + accepted_steps

                if generated_ids[-1] == self.tokenizer.eos_token_id:
                    break
        finally:
            self.set_phase_prefill(True)

        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return generated_ids, text
