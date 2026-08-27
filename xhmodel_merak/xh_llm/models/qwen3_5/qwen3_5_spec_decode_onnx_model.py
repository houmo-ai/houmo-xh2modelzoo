# Copyright 2025 HOUMO AI
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

"""Speculative decoding runtime for Qwen3.5 with DFlash or MTP draft models on HMONNX."""

from typing import Dict, Iterable, List, Optional, Tuple, Union

import torch
from transformers import TextStreamer

from ._spec_decode_shared import SpecDecodeVerifyResult, run_spec_decode_loop
from .qwen3_5_onnx_model import (
    HMONNXSession,
    Qwen3_5ONNXModel,
    _alloc_cache_inputs,
    _apply_presence_penalty,
    _apply_repetition_penalty,
    _as_cache_value,
    _clone_cache_value,
    _ensure_logits_shape,
    _is_kv_cache_name,
    _parse_conv_cache_name,
    _sample_next_token,
    _select_last_valid_logits,
)


_DFLASH_ATTN_MASK_FILL_VALUE = -65504.0


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


class Qwen3_5SpecDecodeONNXModel(Qwen3_5ONNXModel):  # noqa: N801
    """Speculative decoding runtime extending Qwen3_5ONNXModel with a draft model.

    Supports two speculative decoding modes:
    - 'dflash': DFlash draft model with cross-attention to target hidden states
    - 'mtp': MTP (Multi-Token Prediction) head auto-regressive drafting

    Three-phase loop per round:
    1. Draft: generate K candidate tokens using the draft model
    2. Verify: process [current_token] + K draft tokens in one target decode call
    3. Accept/Reject: advance hidden / KV / linear states to the accepted point
    """

    def __init__(
        self,
        prefill,
        decode,
        draft,
        spec_decode_mode: str = "mtp",
        block_size: int = 4,
        hidden_output_name: str = "post_norm_hidden",
        dflash_noise_token_id: Optional[int] = None,
        dflash_sliding_window: Optional[int] = None,
        max_context_tokens: Optional[int] = None,
        auto_offload: bool = True,
        auto_offload_max_memory: Optional[Dict[Union[int, str], Union[int, str]]] = None,
        prefill_auto_offload_max_memory: Optional[Dict[Union[int, str], Union[int, str]]] = None,
        decode_auto_offload_max_memory: Optional[Dict[Union[int, str], Union[int, str]]] = None,
        resource_tight_mode: bool = False,
        pad_token_id: int = 0,
        enable_cuda_graph: bool = False,
        cuda_graph_modules: Optional[Iterable[str]] = None,
        cuda_graph_warmup_runs: int = 3,
        cuda_graph_graph_warmup_runs: int = 6,
        cuda_graph_clone_outputs: bool = True,
    ):
        super().__init__(
            prefill=prefill,
            decode=decode,
            max_context_tokens=max_context_tokens,
            auto_offload=auto_offload,
            auto_offload_max_memory=auto_offload_max_memory,
            prefill_auto_offload_max_memory=prefill_auto_offload_max_memory,
            decode_auto_offload_max_memory=decode_auto_offload_max_memory,
            resource_tight_mode=resource_tight_mode,
            pad_token_id=pad_token_id,
            enable_cuda_graph=enable_cuda_graph,
            cuda_graph_modules=cuda_graph_modules,
            cuda_graph_warmup_runs=cuda_graph_warmup_runs,
            cuda_graph_graph_warmup_runs=cuda_graph_graph_warmup_runs,
            cuda_graph_clone_outputs=cuda_graph_clone_outputs,
        )
        if isinstance(draft, dict):
            self.draft_prefill_config = draft.get("prefill")
            self.draft_context_config = draft.get("context")
            self.draft_context_decode_config = draft.get("context_decode") or self.draft_context_config
            self.draft_decode_config = draft.get("decode")
        else:
            self.draft_prefill_config = None
            self.draft_context_config = None
            self.draft_context_decode_config = None
            self.draft_decode_config = draft
        self.spec_decode_mode = spec_decode_mode
        self.block_size = block_size
        self.hidden_output_name = hidden_output_name
        self._dflash_sliding_window = (
            int(dflash_sliding_window)
            if dflash_sliding_window is not None
            else None
        )
        self._dflash_noise_mask_token_id: Optional[int] = None
        if self.spec_decode_mode == "dflash":
            if dflash_noise_token_id is None or isinstance(
                dflash_noise_token_id, bool
            ):
                raise ValueError(
                    "Qwen3.5 DFlash runtime requires a non-negative "
                    "dflash_noise_token_id from the exported assistant contract"
                )
            try:
                parsed_noise_token_id = int(dflash_noise_token_id)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "Qwen3.5 DFlash runtime requires a non-negative "
                    "dflash_noise_token_id from the exported assistant contract"
                ) from exc
            if parsed_noise_token_id < 0:
                raise ValueError(
                    "Qwen3.5 DFlash runtime requires a non-negative "
                    "dflash_noise_token_id from the exported assistant contract"
                )
            self._dflash_noise_mask_token_id = parsed_noise_token_id
        self.draft_prefill_session: Optional[HMONNXSession] = None
        self.draft_context_session: Optional[HMONNXSession] = None
        self.draft_context_decode_session: Optional[HMONNXSession] = None
        self.draft_decode_session: Optional[HMONNXSession] = None
        self._create_draft_sessions()
        self._is_dflash2 = bool(
            self.draft_decode_session is not None
            and "candidate_ids"
            in set(self.draft_decode_session.get_output_names())
        )
        if self._is_dflash2 and (
            self._dflash_sliding_window is None
            or self._dflash_sliding_window <= 0
        ):
            raise ValueError(
                "Qwen3.5 DFlash2 runtime requires a positive "
                "dflash_sliding_window from the exported assistant contract"
            )
        self._mtp_cache_state: Optional[Dict[str, torch.Tensor]] = None
        self._dflash_cache_state: Optional[Dict[str, torch.Tensor]] = None

    def _set_exec_device(self, device):
        super()._set_exec_device(device)
        for session in (
            self.draft_prefill_session,
            self.draft_context_session,
            self.draft_context_decode_session,
            self.draft_decode_session,
        ):
            if session is not None:
                session.exec_device = device

    def _set_device(self, device):
        super()._set_device(device)
        if not self.auto_offload:
            for session in (
                self.draft_prefill_session,
                self.draft_context_session,
                self.draft_context_decode_session,
                self.draft_decode_session,
            ):
                if session is not None:
                    session.to(device)
        return self

    def _create_single_draft_session(self, cfg, session_name: str):
        if cfg is None:
            return None
        onnx_path = cfg["onnx"] if isinstance(cfg, dict) else cfg.onnx
        session = self._create_hmonnx_session(onnx_path, session_name)
        self._apply_auto_offload(session, self.auto_offload_max_memory)
        return session

    def _create_draft_sessions(self):
        self.draft_prefill_session = self._create_single_draft_session(self.draft_prefill_config, "draft_prefill")
        self.draft_context_session = self._create_single_draft_session(self.draft_context_config, "draft_context")
        if self.draft_context_decode_config is self.draft_context_config:
            self.draft_context_decode_session = self.draft_context_session
        else:
            self.draft_context_decode_session = self._create_single_draft_session(
                self.draft_context_decode_config, "draft_context_decode"
            )
        self.draft_decode_session = self._create_single_draft_session(self.draft_decode_config, "draft_decode")

    def _iter_unique_draft_sessions(self):
        seen: set[int] = set()
        for session in (
            self.draft_prefill_session,
            self.draft_context_session,
            self.draft_context_decode_session,
            self.draft_decode_session,
        ):
            if session is None or id(session) in seen:
                continue
            seen.add(id(session))
            yield session

    def set_spec_draft_golden(self, enable: bool = True, *, reset_step: bool = True) -> None:
        for session in self._iter_unique_draft_sessions():
            self._set_hmonnx_session_golden(session, enable, reset_step=reset_step)

    def _init_mtp_rope(self, rope_theta: float, rotary_dim: int, partial_rotary_factor: float):
        """Initialize MTP RoPE inverse frequencies for cos/sin computation."""
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim))
        self._mtp_inv_freq = inv_freq
        self._mtp_rotary_dim = rotary_dim

    def _compute_mtp_rope(self, position: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute RoPE cos/sin for a single position."""
        if not hasattr(self, "_mtp_inv_freq"):
            # Auto-init from draft session input shapes
            if self.draft_decode_session is None:
                raise RuntimeError("MTP draft decode session is not available.")
            info = self.draft_decode_session.get_input("rope_cos")
            rotary_dim = info.shape[-1]
            self._init_mtp_rope(10_000_000.0, rotary_dim, 0.25)

        inv_freq = self._mtp_inv_freq.to(device=self.device)
        pos = torch.tensor([position], dtype=torch.float32, device=self.device).unsqueeze(1)  # [1, 1]
        freqs = pos * inv_freq.unsqueeze(0)  # [1, rotary_dim//2]
        emb = torch.cat([freqs, freqs], dim=-1)  # [1, rotary_dim]
        cos = emb.cos().unsqueeze(0).unsqueeze(0)  # [1, 1, 1, rotary_dim]
        sin = emb.sin().unsqueeze(0).unsqueeze(0)
        return cos.to(dtype=self._dtype, device=self.device), sin.to(dtype=self._dtype, device=self.device)

    def _init_dflash_rope(self, rope_theta: float, head_dim: int):
        """Initialize DFlash RoPE inverse frequencies."""
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self._dflash_inv_freq = inv_freq

    def _compute_dflash_rope(self, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute RoPE cos/sin for DFlash from position_ids [1, 2*block_size]."""
        if not hasattr(self, "_dflash_inv_freq"):
            if self.draft_decode_session is None:
                raise RuntimeError("DFlash draft decode session is not available.")
            info = self.draft_decode_session.get_input("rope_cos")
            head_dim = info.shape[-1]
            self._init_dflash_rope(10_000_000.0, head_dim)

        inv_freq = self._dflash_inv_freq.to(device=position_ids.device)
        pos = position_ids.reshape(-1, 1).float()  # [2*BS, 1]
        freqs = pos * inv_freq.unsqueeze(0)  # [2*BS, head_dim//2]
        emb = torch.cat([freqs, freqs], dim=-1)  # [2*BS, head_dim]
        cos = emb.cos().unsqueeze(0).unsqueeze(0)  # [1, 1, 2*BS, head_dim]
        sin = emb.sin().unsqueeze(0).unsqueeze(0)
        return cos.to(dtype=self._dtype, device=self.device), sin.to(dtype=self._dtype, device=self.device)

    def _extract_hidden(self, output_map: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        """Extract spec_decode hidden state from target model output."""
        if self.hidden_output_name in output_map:
            return output_map[self.hidden_output_name]
        return None

    @staticmethod
    def _select_hidden_step(hidden_states: Optional[torch.Tensor], step_idx: int) -> Optional[torch.Tensor]:
        if hidden_states is None:
            return None
        if hidden_states.dim() < 3 or hidden_states.shape[1] == 1:
            return hidden_states
        return hidden_states[:, step_idx : step_idx + 1, :]

    @staticmethod
    def _apply_verify_linear_cache_outputs(
        cache_state: Dict[str, torch.Tensor],
        output_map: Dict[str, torch.Tensor],
        accepted_steps: int,
    ) -> None:
        for name, cache_tensor in list(cache_state.items()):
            if name.startswith("past_conv_cache_"):
                branch, idx = _parse_conv_cache_name(name)
                step = max(accepted_steps - 1, 0)
                if branch is None:
                    per_step_name = f"conv_cache_out_{idx}_{step}"
                    out_name = f"conv_cache_out_{idx}"
                else:
                    per_step_name = f"conv_cache_out_{branch}_{idx}_{step}"
                    out_name = f"conv_cache_out_{branch}_{idx}"
                if per_step_name in output_map:
                    cache_state[name] = _as_cache_value(cache_tensor, output_map[per_step_name])
                    continue
                if out_name not in output_map:
                    continue
                conv_out = output_map[out_name]
                if conv_out.dim() >= 3 and conv_out.shape[-1] != cache_tensor.shape[-1]:
                    start = max(accepted_steps - 1, 0)
                    end = start + cache_tensor.shape[-1]
                    conv_out = conv_out[..., start:end]
                cache_state[name] = _as_cache_value(cache_tensor, conv_out)
            elif name.startswith("past_recurrent_state_"):
                idx = name.rsplit("_", 1)[-1]
                # New per-step naming: recurrent_state_out_{layer}_{t}. Pick the
                # (accepted_steps-1)-th snapshot by name (NPU-friendly, no slice).
                step = max(accepted_steps - 1, 0)
                per_step_name = f"recurrent_state_out_{idx}_{step}"
                if per_step_name in output_map:
                    cache_state[name] = _as_cache_value(cache_tensor, output_map[per_step_name])
                    continue
                # Fallback to legacy stacked/flat output for backward compat with
                # older ONNX exports that still emit a single recurrent_state_out_{idx}.
                out_name = f"recurrent_state_out_{idx}"
                if out_name not in output_map:
                    continue
                recurrent_out = output_map[out_name]
                if recurrent_out.dim() >= 5:
                    recurrent_out = recurrent_out[step]
                cache_state[name] = _as_cache_value(cache_tensor, recurrent_out)

    def _ensure_mtp_cache_state(self) -> Dict[str, torch.Tensor]:
        if self.draft_prefill_session is None and self.draft_decode_session is None:
            raise RuntimeError("MTP draft sessions are not initialized.")
        if self._mtp_cache_state is None:
            cache_session = self.draft_prefill_session or self.draft_decode_session
            self._mtp_cache_state = _alloc_cache_inputs(cache_session, self.device)
        return self._mtp_cache_state

    def _ensure_dflash_cache_state(self) -> Dict[str, torch.Tensor]:
        if self.draft_context_session is None and self.draft_decode_session is None:
            raise RuntimeError("DFlash draft sessions are not initialized.")
        if self._dflash_cache_state is None:
            cache_session = self.draft_context_session or self.draft_decode_session
            self._dflash_cache_state = _alloc_cache_inputs(cache_session, self.device)
        return self._dflash_cache_state

    def _run_draft_session(
        self,
        session: HMONNXSession,
        input_feed: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        _, output_map = self._run_hmonnx(session, input_feed)
        return output_map

    def get_cuda_graph_status(self) -> Dict[str, Dict[str, Union[bool, Optional[str]]]]:
        status = super().get_cuda_graph_status()
        status.update(
            {
                "draft_prefill": self._get_session_cuda_graph_status(self.draft_prefill_session),
                "draft_context": self._get_session_cuda_graph_status(self.draft_context_session),
                "draft_context_decode": self._get_session_cuda_graph_status(self.draft_context_decode_session),
                "draft_decode": self._get_session_cuda_graph_status(self.draft_decode_session),
            }
        )
        return status

    def _embed_token_ids(self, token_ids: torch.Tensor) -> torch.Tensor:
        if self.token_embedding is None:
            raise ValueError("token_embedding is not set, call set_input_embeddings first.")
        embed_device = self.token_embedding.weight.device
        token_ids = token_ids.to(device=embed_device, dtype=torch.long)
        return self.token_embedding(token_ids).to(device=self.device, dtype=self._dtype)

    def _build_mtp_prefill_feed(
        self,
        hidden_states: torch.Tensor,
        next_token_ids: torch.Tensor,
        past_seq_len: int,
        cache_state: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        if self.token_embedding is None:
            raise ValueError("token_embedding is not set, call set_input_embeddings first.")
        if self.draft_prefill_session is None:
            raise RuntimeError("MTP prefill session is not available.")
        actual_input_length = int(hidden_states.shape[1])
        prefill_seq_len = int(self.draft_prefill_session.get_input("post_norm_hidden").shape[1])
        next_token_embedding = _pad_hidden_tensor(self._embed_token_ids(next_token_ids), prefill_seq_len)
        hidden_states = _pad_hidden_tensor(hidden_states.to(device=self.device, dtype=self._dtype), prefill_seq_len)
        batch_size = hidden_states.shape[0]
        current_input_length = torch.full(
            (batch_size,),
            actual_input_length,
            dtype=torch.int32,
            device=self.device,
        )
        past_seq_length = torch.full(
            (batch_size,),
            int(past_seq_len),
            dtype=torch.int32,
            device=self.device,
        )
        feed: Dict[str, torch.Tensor] = {}
        for name in self.draft_prefill_session.get_input_names():
            if name == "next_token_embedding":
                feed[name] = next_token_embedding
            elif name == "post_norm_hidden":
                feed[name] = hidden_states
            elif name in ["past_seq_length", "valid_length"]:
                feed[name] = past_seq_length
            elif name in ["current_input_length", "current_length"]:
                feed[name] = current_input_length
            elif name in cache_state:
                feed[name] = cache_state[name]
            else:
                info = self.draft_prefill_session.get_input(name)
                feed[name] = torch.zeros(info.shape, dtype=info.dtype, device=self.device)
        return feed

    def _build_mtp_decode_feed(
        self,
        next_token_id: torch.Tensor,
        hidden_state: torch.Tensor,
        past_seq_len: int,
        cache_state: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        if self.token_embedding is None:
            raise ValueError("token_embedding is not set, call set_input_embeddings first.")
        if self.draft_decode_session is None:
            raise RuntimeError("MTP decode session is not available.")
        next_token_embedding = self._embed_token_ids(next_token_id)
        batch_size = hidden_state.shape[0]
        current_input_length = torch.ones(batch_size, dtype=torch.int32, device=self.device)
        past_seq_length = torch.full(
            (batch_size,),
            int(past_seq_len),
            dtype=torch.int32,
            device=self.device,
        )
        feed: Dict[str, torch.Tensor] = {}
        for name in self.draft_decode_session.get_input_names():
            if name == "next_token_embedding":
                feed[name] = next_token_embedding
            elif name == "post_norm_hidden":
                feed[name] = hidden_state.to(device=self.device, dtype=self._dtype)
            elif name in ["past_seq_length", "valid_length"]:
                feed[name] = past_seq_length
            elif name in ["current_input_length", "current_length"]:
                feed[name] = current_input_length
            elif name in cache_state:
                feed[name] = cache_state[name]
            else:
                info = self.draft_decode_session.get_input(name)
                feed[name] = torch.zeros(info.shape, dtype=info.dtype, device=self.device)
        return feed

    def _prefill_mtp_chunk(
        self,
        hidden_states: torch.Tensor,
        next_token_ids: torch.Tensor,
        past_seq_len: int,
    ) -> None:
        if hidden_states.shape[1] == 0:
            return
        cache_state = self._ensure_mtp_cache_state()
        self._run_draft_session(
            self.draft_prefill_session,
            self._build_mtp_prefill_feed(hidden_states, next_token_ids, past_seq_len, cache_state),
        )

    def _complete_mtp_full_accept_tail(
        self,
        verify_result: SpecDecodeVerifyResult,
        *,
        accepted_steps: int,
        mtp_past_seq_len: int,
    ) -> None:
        """Materialize the one shifted MTP pair missing on full acceptance.

        Before a round, the private MTP cache is one token behind the target.
        K draft invocations consume ``current, draft[0], ..., draft[K-2]`` and
        therefore cache K shifted pairs, ending with
        ``hidden(draft[K-2]) -> embedding(draft[K-1])``.  Full acceptance also
        advances the target through ``draft[K-1]`` and emits the bonus token
        ``draft[K]``.  Materialize exactly that final shifted pair so the next
        round starts one token behind again:
        ``hidden(draft[K-1]) -> embedding(draft[K])``.
        """

        verify_length = len(verify_result.verify_token_ids)
        if accepted_steps != verify_length or verify_length <= 1:
            return
        if verify_result.verify_hidden is None:
            raise RuntimeError(
                "Qwen3.5 MTP full acceptance requires target verify hidden states"
            )
        tail_index = accepted_steps - 1
        self._prefill_mtp_chunk(
            verify_result.verify_hidden[
                :,
                tail_index - 1 : tail_index,
                :,
            ],
            torch.tensor(
                [[verify_result.verify_token_ids[tail_index]]],
                dtype=torch.long,
                device=verify_result.verify_hidden.device,
            ),
            mtp_past_seq_len + accepted_steps - 1,
        )

    def _append_dflash_context(
        self,
        target_hidden: torch.Tensor,
        past_seq_len: int,
        *,
        decode_step: bool = False,
    ) -> None:
        if target_hidden is None or target_hidden.shape[1] == 0:
            return
        session = (
            self.draft_context_decode_session
            if decode_step and self.draft_context_decode_session is not None
            else self.draft_context_session
        )
        if session is None:
            raise RuntimeError("DFlash context session is not available.")
        cache_state = self._ensure_dflash_cache_state()
        actual_input_length = int(target_hidden.shape[1])
        target_hidden_info = session.get_input("target_hidden")
        context_seq_len = int(target_hidden_info.shape[1])
        expected_hidden_dim = target_hidden_info.shape[-1] if len(target_hidden_info.shape) >= 3 else None
        if expected_hidden_dim not in (None, -1):
            expected_hidden_dim = int(expected_hidden_dim)
            actual_hidden_dim = int(target_hidden.shape[-1])
            if actual_hidden_dim != expected_hidden_dim:
                raise RuntimeError(
                    "DFlash target_hidden shape mismatch before running draft context: "
                    f"expected hidden dim {expected_hidden_dim}, got {actual_hidden_dim}. "
                    "If this artifact was exported with --num_blocks/max_layers, ensure it covers all "
                    "dflash_config.target_layer_ids required by the DFlash checkpoint."
                )
        target_hidden = _pad_hidden_tensor(target_hidden.to(device=self.device, dtype=self._dtype), context_seq_len)
        batch_size = target_hidden.shape[0]
        current_input_length = torch.full(
            (batch_size,),
            actual_input_length,
            dtype=torch.int32,
            device=self.device,
        )
        past_seq_length = torch.full(
            (batch_size,),
            int(past_seq_len),
            dtype=torch.int32,
            device=self.device,
        )
        feed: Dict[str, torch.Tensor] = {}
        for name in session.get_input_names():
            if name == "target_hidden":
                feed[name] = target_hidden
            elif name in ("past_seq_length", "valid_length"):
                feed[name] = past_seq_length
            elif name in ("current_input_length", "current_length"):
                feed[name] = current_input_length
            elif name in cache_state:
                feed[name] = cache_state[name]
            else:
                raise RuntimeError(f"Unexpected input '{name}' in DFlash draft context session.")
        self._run_draft_session(session, feed)

    def _build_dflash_decode_feed(
        self,
        noise_embedding: torch.Tensor,
        past_seq_len: int,
        cache_state: Dict[str, torch.Tensor],
        anchor_token_id: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if self.draft_decode_session is None:
            raise RuntimeError("DFlash decode session is not available.")
        batch_size = noise_embedding.shape[0]
        current_input_length = torch.full(
            (batch_size,),
            int(noise_embedding.shape[1]),
            dtype=torch.int32,
            device=self.device,
        )
        past_seq_length = torch.full(
            (batch_size,),
            int(past_seq_len),
            dtype=torch.int32,
            device=self.device,
        )
        feed: Dict[str, torch.Tensor] = {}
        for name in self.draft_decode_session.get_input_names():
            if name == "noise_embedding":
                feed[name] = noise_embedding
            elif name in ("past_seq_length", "valid_length"):
                feed[name] = past_seq_length
            elif name in ("current_input_length", "current_length"):
                feed[name] = current_input_length
            elif name == "anchor_token_id":
                input_info = self.draft_decode_session.get_input(name)
                feed[name] = anchor_token_id.reshape(batch_size).to(
                    device=self.device,
                    dtype=input_info.dtype,
                )
            elif name == "attn_mask":
                input_info = self.draft_decode_session.get_input(name)
                mask_shape = tuple(int(dim) for dim in input_info.shape)
                attn_mask = torch.full(
                    mask_shape,
                    _DFLASH_ATTN_MASK_FILL_VALUE,
                    dtype=input_info.dtype,
                    device=self.device,
                )
                valid_length = min(
                    int(mask_shape[-1]),
                    int(past_seq_len) + int(noise_embedding.shape[1]),
                )
                if len(mask_shape) == 2:
                    attn_mask[:, :valid_length] = 0
                elif len(mask_shape) == 3:
                    window = int(self._dflash_sliding_window or 0)
                    key_positions = torch.arange(
                        mask_shape[-1],
                        device=self.device,
                    ).reshape(1, 1, -1)
                    query_positions = (
                        int(past_seq_len)
                        + torch.arange(
                            mask_shape[1],
                            device=self.device,
                        ).reshape(1, -1, 1)
                    )
                    visible = (
                        (key_positions < valid_length)
                        & (query_positions - key_positions < window)
                        & (key_positions - query_positions < window)
                    )
                    attn_mask.masked_fill_(visible, 0)
                else:
                    raise RuntimeError(
                        "DFlash attention mask input must be rank 2 or 3, "
                        f"got shape={mask_shape}"
                    )
                feed[name] = attn_mask
            elif name == "query_kv_range_abs":
                input_info = self.draft_decode_session.get_input(name)
                range_shape = tuple(int(dim) for dim in input_info.shape)
                if len(range_shape) != 3 or range_shape[-1] != 2:
                    raise RuntimeError(
                        "DFlash2 query_kv_range_abs must have shape [B, Q, 2], "
                        f"got {range_shape}"
                    )
                window = int(self._dflash_sliding_window or 0)
                valid_length = min(
                    next(iter(cache_state.values())).shape[2]
                    if cache_state
                    else int(self.max_context_tokens),
                    int(past_seq_len) + int(noise_embedding.shape[1]),
                )
                query_positions = (
                    int(past_seq_len)
                    + torch.arange(
                        range_shape[1],
                        dtype=input_info.dtype,
                        device=self.device,
                    )
                )
                starts = torch.clamp(query_positions - window + 1, min=0)
                ends = torch.clamp(
                    query_positions + window,
                    max=int(valid_length),
                )
                feed[name] = torch.stack((starts, ends), dim=-1).unsqueeze(0)
            elif name in cache_state:
                feed[name] = cache_state[name]
            else:
                raise RuntimeError(f"Unexpected input '{name}' in DFlash draft decode session.")
        return feed

    # ---------------------------------------------------------------
    # DFlash draft
    # ---------------------------------------------------------------
    def _run_draft_dflash(
        self,
        next_tok: torch.Tensor,
        past_seq_len: int,
    ) -> List[torch.Tensor]:
        """Generate block_size draft tokens using the DFlash decode graph."""
        if self.token_embedding is None:
            raise ValueError("token_embedding is not set, call set_input_embeddings first.")
        cache_state = self._ensure_dflash_cache_state()
        block_len = int(self.draft_decode_session.get_input("noise_embedding").shape[1])
        block_ids = torch.full(
            (1, block_len),
            self._dflash_noise_mask_token_id,
            dtype=torch.long,
            device=self.device,
        )
        block_ids[0, 0] = next_tok[0, 0]
        noise_embedding = self._embed_token_ids(block_ids)
        draft_output_map = self._run_draft_session(
            self.draft_decode_session,
            self._build_dflash_decode_feed(
                noise_embedding,
                past_seq_len,
                cache_state,
                next_tok,
            ),
        )
        candidate_ids = draft_output_map.get("candidate_ids")
        if candidate_ids is not None:
            first_scores = draft_output_map.get("selector_first_scores")
            transition_scores = draft_output_map.get(
                "selector_transition_scores"
            )
            if first_scores is None or transition_scores is None:
                raise RuntimeError(
                    "DFlash2 decode graph must return candidate_ids, "
                    "selector_first_scores, and selector_transition_scores"
                )
            selected_index = torch.argmax(first_scores.float(), dim=-1)
            draft_tokens = [
                candidate_ids[:, 0].gather(
                    -1,
                    selected_index.unsqueeze(-1),
                )
            ]
            for step in range(transition_scores.shape[1]):
                row = transition_scores[:, step].gather(
                    1,
                    selected_index.reshape(-1, 1, 1).expand(
                        -1,
                        1,
                        transition_scores.shape[-1],
                    ),
                )[:, 0]
                selected_index = torch.argmax(row.float(), dim=-1)
                draft_tokens.append(
                    candidate_ids[:, step + 1].gather(
                        -1,
                        selected_index.unsqueeze(-1),
                    )
                )
            return draft_tokens[: self.block_size - 1]
        draft_logits = draft_output_map["logits"]
        draft_tokens = []
        max_positions = min(self.block_size + 1, draft_logits.shape[1])
        for j in range(1, max_positions):
            tok = torch.argmax(draft_logits[:, j : j + 1, :], dim=-1)  # [1, 1]
            draft_tokens.append(tok)
        return draft_tokens

    # ---------------------------------------------------------------
    # MTP draft
    # ---------------------------------------------------------------
    def _run_draft_mtp(
        self,
        next_tok: torch.Tensor,
        post_norm_hidden: torch.Tensor,
        past_seq_len: int,
        num_drafts: int,
    ) -> List[torch.Tensor]:
        """Generate num_drafts draft tokens using the MTP decode graph."""
        cache_state = self._ensure_mtp_cache_state()
        drafts = []
        current_hidden = post_norm_hidden  # [1, 1, H]

        for j in range(num_drafts):
            draft_output_map = self._run_draft_session(
                self.draft_decode_session,
                self._build_mtp_decode_feed(next_tok, current_hidden, past_seq_len + j, cache_state),
            )
            logits = draft_output_map["logits"]  # [1, 1, V]
            post_norm_out = draft_output_map.get("post_norm_out")  # [1, 1, H]
            if post_norm_out is None:
                raise RuntimeError("MTP draft graph must export post_norm_out for autoregressive chaining.")

            tok = torch.argmax(logits[:, -1:, :], dim=-1)  # [1, 1]
            drafts.append(tok)

            # Feed back for next step
            current_hidden = post_norm_out
            next_tok = tok

        return drafts

    # ---------------------------------------------------------------
    # Speculative decoding generate
    # ---------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        tokenizer,
        max_new_tokens: int = 256,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        stream_output: bool = False,
        return_stats: bool = False,
    ) -> Union[str, Tuple[str, Dict[str, Union[int, float, List[int]]]]]:
        if max_new_tokens <= 0:
            return ""
        if input_ids.dim() != 2 or input_ids.shape[0] != 1:
            raise ValueError(f"input_ids must be [1, seq], got {tuple(input_ids.shape)}")
        if self.token_embedding is None:
            raise ValueError("token_embedding is not set, call set_input_embeddings first.")

        if self.max_context_tokens is not None and input_ids.shape[1] > self.max_context_tokens:
            input_ids = input_ids[:, -self.max_context_tokens :]

        total_prompt_len = int(input_ids.shape[1])
        if total_prompt_len <= 0:
            raise ValueError("Empty prompt is not supported.")

        # --- Phase 1: Prefill ---
        self._ensure_prefill_session()
        prefill_cache_state = _alloc_cache_inputs(self.prefill_session, self.device)
        prefill_chunk_len = int(self._prefill_inputs_info.shape[1])

        last_prefill_logits = None
        last_hidden = None
        past_seq_len = 0
        mtp_prefill_seq_len = 0
        mtp_pending_hidden: Optional[torch.Tensor] = None
        for start in range(0, total_prompt_len, prefill_chunk_len):
            end = min(start + prefill_chunk_len, total_prompt_len)
            chunk_ids = input_ids[:, start:end]
            valid_len = int(chunk_ids.shape[1])

            prefill_feed = self._build_prefill_feed(chunk_ids, valid_len, past_seq_len, prefill_cache_state)
            _, prefill_output_map = self._run_hmonnx(self.prefill_session, prefill_feed)
            prefill_logits = self._extract_logits(prefill_output_map)
            last_prefill_logits = _select_last_valid_logits(prefill_logits, valid_len)
            prefill_hidden_all = self._extract_hidden(prefill_output_map)
            if prefill_hidden_all is not None:
                prefill_hidden_all = prefill_hidden_all[:, :valid_len, :]
                last_hidden = self._select_hidden_step(prefill_hidden_all, valid_len - 1)
                if self.spec_decode_mode == "dflash":
                    self._append_dflash_context(prefill_hidden_all, past_seq_len)
                elif self.spec_decode_mode == "mtp":
                    hidden_parts = []
                    token_parts = []
                    if mtp_pending_hidden is not None:
                        hidden_parts.append(mtp_pending_hidden)
                        token_parts.append(chunk_ids[:, :1])
                    if valid_len > 1:
                        hidden_parts.append(prefill_hidden_all[:, : valid_len - 1, :])
                        token_parts.append(chunk_ids[:, 1:valid_len])
                    if hidden_parts:
                        mtp_hidden = torch.cat(hidden_parts, dim=1)
                        mtp_tokens = torch.cat(token_parts, dim=1)
                        self._prefill_mtp_chunk(mtp_hidden, mtp_tokens, mtp_prefill_seq_len)
                        mtp_prefill_seq_len += int(mtp_hidden.shape[1])
                    mtp_pending_hidden = prefill_hidden_all[:, valid_len - 1 : valid_len, :]

            self._update_linear_cache(prefill_cache_state, prefill_output_map)
            past_seq_len += valid_len

        if last_prefill_logits is None:
            return ""

        # Sample first token from prefill
        history_token_ids = input_ids[0].tolist()
        first_logits = _apply_repetition_penalty(last_prefill_logits, history_token_ids, repetition_penalty)
        first_logits = _apply_presence_penalty(first_logits, history_token_ids, presence_penalty)
        next_token_id = _sample_next_token(
            first_logits,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )

        if self.resource_tight_mode:
            self._release_prefill_session()

        # Transfer caches to decode session
        self._ensure_decode_session()
        decode_cache_state = _alloc_cache_inputs(self.decode_session, self.device)
        for name in decode_cache_state:
            if name in prefill_cache_state and (
                _is_kv_cache_name(name) or name.startswith(("past_conv_cache_", "past_recurrent_state_"))
            ):
                decode_cache_state[name] = prefill_cache_state[name]

        eos_token_id = tokenizer.eos_token_id
        streamer: Optional[TextStreamer] = None
        if stream_output:
            streamer = TextStreamer(tokenizer, skip_prompt=False, skip_special_tokens=True)

        token_val = int(next_token_id[0][0].item())
        if eos_token_id is not None and token_val == eos_token_id:
            if streamer is not None:
                streamer.end()
            return ""
        mtp_past_seq_len = mtp_prefill_seq_len
        num_drafts = self.block_size - 1 if self.spec_decode_mode == "dflash" else self.block_size

        def _on_token(token_id: int) -> None:
            history_token_ids.append(token_id)
            if streamer is not None:
                streamer.put(torch.tensor([[token_id]], dtype=torch.long))

        def _run_round_draft(
            current_token_id: int,
            round_last_hidden: Optional[torch.Tensor],
            round_past_seq_len: int,
            round_mtp_past_seq_len: int,
            round_num_drafts: int,
        ) -> List[int]:
            current_token = torch.tensor([[current_token_id]], dtype=torch.long, device=self.device)
            if self.spec_decode_mode == "dflash":
                return [int(tok[0, 0].item()) for tok in self._run_draft_dflash(current_token, round_past_seq_len)]
            if round_last_hidden is None:
                return []
            return [
                int(tok[0, 0].item())
                for tok in self._run_draft_mtp(
                    current_token,
                    round_last_hidden,
                    round_mtp_past_seq_len,
                    round_num_drafts,
                )
            ]

        def _verify_round(
            current_token_id: int,
            draft_token_ids: List[int],
            round_past_seq_len: int,
        ) -> SpecDecodeVerifyResult:
            verify_ids = [current_token_id] + draft_token_ids
            verify_input_ids = torch.tensor([verify_ids], dtype=torch.long, device=self.device)
            decode_feed = self._build_decode_feed(
                verify_input_ids,
                round_past_seq_len,
                decode_cache_state,
                current_input_length=verify_input_ids.shape[1],
            )
            _, decode_output_map = self._run_hmonnx(self.decode_session, decode_feed)
            verify_logits = _ensure_logits_shape(self._extract_logits(decode_output_map))
            verify_hidden_all = self._extract_hidden(decode_output_map)
            predicted_token_ids = [
                int(torch.argmax(verify_logits[:, j : j + 1, :], dim=-1)[0, 0].item()) for j in range(len(verify_ids))
            ]
            return SpecDecodeVerifyResult(
                initial_seq_len=round_past_seq_len,
                verify_token_ids=verify_ids,
                predicted_token_ids=predicted_token_ids,
                verify_hidden=verify_hidden_all,
                raw_result=decode_output_map,
            )

        def _apply_round_result(
            verify_result: SpecDecodeVerifyResult,
            accepted_steps: int,
        ) -> None:
            self._apply_verify_linear_cache_outputs(
                decode_cache_state,
                verify_result.raw_result,
                accepted_steps=accepted_steps,
            )

        def _post_round(
            verify_result: SpecDecodeVerifyResult,
            accepted_count: int,
            accepted_steps: int,
            _next_token_id: int,
            round_mtp_past_seq_len: int,
        ) -> Tuple[Optional[torch.Tensor], int]:
            next_hidden = self._select_hidden_step(verify_result.verify_hidden, accepted_count)
            accepted_hidden = (
                verify_result.verify_hidden[:, :accepted_steps, :] if verify_result.verify_hidden is not None else None
            )
            if self.spec_decode_mode == "dflash":
                assert accepted_hidden is not None, "DFlash mode requires hidden states from the verify step."
                self._append_dflash_context(accepted_hidden, verify_result.initial_seq_len, decode_step=True)
                return next_hidden, round_mtp_past_seq_len
            self._complete_mtp_full_accept_tail(
                verify_result,
                accepted_steps=accepted_steps,
                mtp_past_seq_len=round_mtp_past_seq_len,
            )
            return next_hidden, round_mtp_past_seq_len + accepted_steps

        generated_ids, stats = run_spec_decode_loop(
            max_new_tokens=max_new_tokens,
            initial_token_id=token_val,
            initial_past_seq_len=past_seq_len,
            initial_last_hidden=last_hidden,
            initial_mtp_past_seq_len=mtp_past_seq_len,
            eos_token_id=eos_token_id,
            num_drafts=num_drafts,
            run_draft=_run_round_draft,
            verify_round=_verify_round,
            apply_verify_result=_apply_round_result,
            post_verify=_post_round,
            on_token=_on_token,
        )

        out_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        if streamer is not None:
            streamer.end()
        if stats["num_rounds"] > 0:
            print(
                f"[SpecDecode] mode={self.spec_decode_mode} "
                f"rounds={stats['num_rounds']} "
                f"total_tokens={len(generated_ids)} "
                f"draft_tokens={stats['draft_tokens_total']} "
                f"accepted={stats['accepted_drafts_total']} "
                f"avg_accepted_per_round={stats['avg_accepted_per_round']:.2f}"
            )

        if self.resource_tight_mode:
            self._release_decode_session()
        if return_stats:
            return out_text, stats
        return out_text

    # ---------------------------------------------------------------
    # Conv/Recurrent state snapshot & restore for DeltaNet rollback
    # ---------------------------------------------------------------
    @staticmethod
    def _snapshot_deltanet_states(
        cache_state: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Save a copy of all conv_cache and recurrent_state tensors.

        KV cache does not need explicit snapshotting — it uses fixed-size
        buffers indexed by past_seq_len, so rolling back past_seq_len is
        sufficient (stale entries beyond the index are never read and will
        be overwritten on the next write).
        """
        snapshot = {}
        for name, tensor in cache_state.items():
            if name.startswith("past_conv_cache_") or name.startswith("past_recurrent_state_"):
                snapshot[name] = _clone_cache_value(tensor)
        return snapshot

    @staticmethod
    def _restore_deltanet_states(
        cache_state: Dict[str, torch.Tensor],
        snapshot: Dict[str, torch.Tensor],
    ) -> None:
        """Restore conv_cache and recurrent_state from a snapshot."""
        for name, tensor in snapshot.items():
            cache_state[name] = _as_cache_value(cache_state[name], tensor)
