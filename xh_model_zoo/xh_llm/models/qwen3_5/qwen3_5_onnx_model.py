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

from typing import Dict, Iterable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import TextStreamer

from xhquant.xhonnxruntime import (
    AutoOffloadGraphModel,
    HMONNXCUDAGraphInference,
    HMONNXGrapInference,
)
from xhquant.core import CacheTensor

from ..builder import MODELS
from ..device_dtype_mixin import DeviceDtypeMixin


def _build_linear_attn_mask(
    valid_len: int, mask_info, device: torch.device
) -> torch.Tensor:
    mask = torch.zeros(mask_info.shape, dtype=mask_info.dtype, device=device)
    if valid_len > 0:
        slices = [slice(None)] * mask.dim()
        slices[-1] = slice(0, valid_len)
        mask[tuple(slices)] = 1
    return mask


def _build_inputs_embeds(
    token_embedding: nn.Module,
    input_ids: torch.Tensor,
    target_seq_len: int,
    _pad_token_id: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    input_ids = input_ids.to(device=device, dtype=torch.long)
    if input_ids.shape[1] > target_seq_len:
        input_ids = input_ids[:, :target_seq_len]
    inputs_embeds = token_embedding(input_ids).to(dtype=dtype)
    if inputs_embeds.shape[1] < target_seq_len:
        pad_shape = (
            inputs_embeds.shape[0],
            target_seq_len - inputs_embeds.shape[1],
            inputs_embeds.shape[2],
        )
        pad_embeds = torch.zeros(pad_shape, dtype=dtype, device=device)
        inputs_embeds = torch.cat([inputs_embeds, pad_embeds], dim=1)
    return inputs_embeds


HMONNXSession = Union[HMONNXGrapInference, HMONNXCUDAGraphInference]


def _resolve_input_name(
    session: HMONNXSession, candidates: Tuple[str, ...], fallback=None
) -> str:
    input_names = session.get_input_names()
    for name in candidates:
        if name in input_names:
            return name
    if fallback is not None:
        return fallback(session)
    raise ValueError(f"None of {candidates} found in inputs: {input_names}")


def _infer_inputs_embeds_name(session: HMONNXSession) -> str:
    for name in session.get_input_names():
        info = session.get_input(name)
        if (
            info.dtype in (torch.float16, torch.float32, torch.bfloat16)
            and len(info.shape) == 3
        ):
            return name
    return session.get_input_names()[0]


def _alloc_cache_inputs(
    session: HMONNXSession, device: torch.device
) -> Dict[str, torch.Tensor]:
    cache_inputs: Dict[str, torch.Tensor] = {}
    for name in session.get_input_names():
        if (
            name.startswith(
                (
                    "past_key_cache",
                    "past_value_cache",
                    "past_key_cache_",
                    "past_value_cache_",
                    "past_conv_cache_",
                    "past_recurrent_state_",
                )
            )
            or "kcache_input" in name
            or "vcache_input" in name
        ):
            info = session.get_input(name)
            cache_inputs[name] = CacheTensor(torch.zeros(info.shape, dtype=info.dtype, device=device))
    return cache_inputs


def _as_cache_value(reference: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    if isinstance(reference, CacheTensor) and not isinstance(value, CacheTensor):
        return CacheTensor(value)
    return value


def _clone_cache_value(value: torch.Tensor) -> torch.Tensor:
    cloned = value.clone()
    if isinstance(value, CacheTensor):
        return CacheTensor(cloned)
    return cloned


def _ensure_logits_shape(logits: torch.Tensor) -> torch.Tensor:
    if logits.dim() == 3:
        return logits
    if logits.dim() == 2:
        return logits.unsqueeze(1)
    if logits.dim() == 1:
        return logits.view(1, 1, -1)
    raise ValueError(f"Unexpected logits shape: {tuple(logits.shape)}")


def _select_last_valid_logits(logits: torch.Tensor, valid_len: int) -> torch.Tensor:
    logits = _ensure_logits_shape(logits)
    if logits.shape[1] == 1:
        return logits
    if valid_len <= 0:
        raise ValueError(f"valid_len must be > 0, got {valid_len}")
    return logits[:, valid_len - 1 : valid_len, :]


def _is_kv_cache_name(name: str) -> bool:
    return (
        name.startswith(
            ("past_key_cache", "past_value_cache", "past_key_cache_", "past_value_cache_")
        )
        or "kcache_input" in name
        or "vcache_input" in name
    )


def _parse_conv_cache_name(name: str) -> Tuple[Optional[str], str]:
    """Parse past_conv_cache_X or past_conv_cache_{branch}_{idx}.

    Returns (branch, idx) where branch is None for old-style names.
    """
    suffix = name[len("past_conv_cache_"):]
    parts = suffix.rsplit("_", 1)
    if len(parts) == 2 and parts[0] in ("q", "k", "v"):
        return parts[0], parts[1]
    # Old-style: past_conv_cache_{idx}
    return None, suffix


def _apply_repetition_penalty(
    logits: torch.Tensor, token_ids: List[int], repetition_penalty: float
) -> torch.Tensor:
    if repetition_penalty == 1.0 or len(token_ids) == 0:
        return logits
    unique_token_ids = torch.unique(
        torch.tensor(token_ids, dtype=torch.long, device=logits.device)
    )
    updated = logits.clone()
    values = updated[..., unique_token_ids]
    values = torch.where(
        values < 0, values * repetition_penalty, values / repetition_penalty
    )
    updated[..., unique_token_ids] = values
    return updated


def _apply_presence_penalty(
    logits: torch.Tensor, token_ids: List[int], presence_penalty: float
) -> torch.Tensor:
    if presence_penalty == 0.0 or len(token_ids) == 0:
        return logits
    unique_token_ids = torch.unique(
        torch.tensor(token_ids, dtype=torch.long, device=logits.device)
    )
    updated = logits.clone()
    updated[..., unique_token_ids] = updated[..., unique_token_ids] - presence_penalty
    return updated


def _sample_next_token(
    logits: torch.Tensor,
    do_sample: bool,
    temperature: float,
    top_p: float,
    top_k: int,
) -> torch.Tensor:
    logits = _ensure_logits_shape(logits).to(torch.float32)
    next_token_logits = logits[:, -1, :]

    if not do_sample or temperature <= 0:
        return torch.argmax(next_token_logits, dim=-1, keepdim=True)

    filtered_logits = next_token_logits.clone()
    vocab_size = int(filtered_logits.shape[-1])
    if top_k > 0 and top_k < vocab_size:
        kth_values = torch.topk(filtered_logits, top_k, dim=-1).values[..., -1, None]
        filtered_logits = filtered_logits.masked_fill(
            filtered_logits < kth_values, float("-inf")
        )

    if 0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(
            filtered_logits, descending=True, dim=-1
        )
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_mask = cumulative_probs > top_p
        sorted_mask[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(sorted_mask, float("-inf"))
        filtered_logits = torch.full_like(filtered_logits, float("-inf"))
        filtered_logits.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)

    scaled_logits = filtered_logits / max(temperature, 1e-5)
    probs = torch.softmax(scaled_logits, dim=-1)
    probs_sum = probs.sum(dim=-1, keepdim=True)
    invalid_probs = (~torch.isfinite(probs)).any() or (probs_sum <= 0).any()
    if invalid_probs:
        return torch.argmax(next_token_logits, dim=-1, keepdim=True)
    return torch.multinomial(probs, num_samples=1)


@MODELS.register_module()
class Qwen3_5ONNXModel(DeviceDtypeMixin):
    def __init__(
        self,
        prefill,
        decode,
        max_context_tokens: Optional[int] = None,
        auto_offload: bool = True,
        auto_offload_max_memory: Optional[
            Dict[Union[int, str], Union[int, str]]
        ] = None,
        prefill_auto_offload_max_memory: Optional[
            Dict[Union[int, str], Union[int, str]]
        ] = None,
        decode_auto_offload_max_memory: Optional[
            Dict[Union[int, str], Union[int, str]]
        ] = None,
        resource_tight_mode: bool = False,
        pad_token_id: int = 0,
        enable_cuda_graph: bool = False,
        cuda_graph_modules: Optional[Iterable[str]] = None,
        cuda_graph_warmup_runs: int = 3,
        cuda_graph_graph_warmup_runs: int = 6,
        cuda_graph_clone_outputs: bool = True,
    ):
        super().__init__()
        self._device = torch.device("cpu")
        self._dtype = torch.float16
        self._exec_device = torch.device("cpu")

        self.prefill_config = prefill
        self.decode_config = decode
        self.max_context_tokens = max_context_tokens
        self.auto_offload = auto_offload
        self.auto_offload_max_memory = auto_offload_max_memory
        self.prefill_auto_offload_max_memory = prefill_auto_offload_max_memory
        self.decode_auto_offload_max_memory = decode_auto_offload_max_memory
        self.resource_tight_mode = resource_tight_mode
        self.pad_token_id = pad_token_id
        self.enable_cuda_graph = enable_cuda_graph
        self.cuda_graph_modules = (
            {name.strip().lower() for name in cuda_graph_modules if str(name).strip()}
            if cuda_graph_modules is not None
            else None
        )
        self.cuda_graph_warmup_runs = max(cuda_graph_warmup_runs, 0)
        self.cuda_graph_graph_warmup_runs = max(cuda_graph_graph_warmup_runs, 0)
        self.cuda_graph_clone_outputs = cuda_graph_clone_outputs

        self.token_embedding: Optional[nn.Module] = None
        self.prefill_session: Optional[HMONNXSession] = None
        self.decode_session: Optional[HMONNXSession] = None

        self._prefill_inputs_name = None
        self._prefill_past_seq_name = None
        self._prefill_current_seq_name = None
        self._prefill_mask_name = None

        self._decode_inputs_name = None
        self._decode_past_seq_name = None
        self._decode_current_seq_name = None
        self._decode_mask_name = None

        self._prefill_inputs_info = None
        self._prefill_mask_info = None
        self._prefill_past_seq_info = None
        self._prefill_current_seq_info = None

        self._decode_inputs_info = None
        self._decode_mask_info = None
        self._decode_past_seq_info = None
        self._decode_current_seq_info = None

        self._init_sessions()

    def _set_exec_device(self, device):
        super()._set_exec_device(device)
        if self.prefill_session is not None:
            self.prefill_session.exec_device = device
        if self.decode_session is not None:
            self.decode_session.exec_device = device

    def _set_device(self, device):
        super()._set_device(device)
        if not self.auto_offload:
            if self.prefill_session is not None:
                self.prefill_session.to(device)
            if self.decode_session is not None:
                self.decode_session.to(device)
        if self.token_embedding is not None:
            self.token_embedding.to(device)
        return self

    def _set_dtype(self, dtype):
        super()._set_dtype(dtype)
        if self.token_embedding is not None:
            self.token_embedding.to(dtype)
        return self

    def set_input_embeddings(self, value: nn.Module):
        self.token_embedding = value
        self.token_embedding.to(self.device)
        self.token_embedding.to(self.dtype)
        self.token_embedding.eval()

    def set_pad_token_id(self, pad_token_id: int):
        self.pad_token_id = int(pad_token_id)

    def _apply_auto_offload(
        self,
        session: HMONNXSession,
        max_memory: Optional[Dict[Union[int, str], Union[int, str]]] = None,
    ) -> None:
        if isinstance(session, HMONNXCUDAGraphInference):
            session.configure_auto_offload(max_memory=max_memory)
            session.enable_auto_offload = self.auto_offload and torch.cuda.is_available()
            return
        if not self.auto_offload:
            return
        if not torch.cuda.is_available():
            return
        if session.graph_module is None:
            return
        if AutoOffloadGraphModel.is_auto_offload_model(session.graph_module):
            return
        try:
            session.graph_module = AutoOffloadGraphModel.from_graph_model(
                session.graph_module,
                max_memory=max_memory,
            )
        except AssertionError as e:
            raise RuntimeError(
                "AutoOffloadGraphModel mapping failed. "
                "The provided max_memory is too tight and triggered CPU fallback. "
                "Please increase GPU budgets or provide more GPU devices in max_memory."
            ) from e

    def _should_enable_cuda_graph(self, session_name: str) -> bool:
        if not self.enable_cuda_graph:
            return False
        if self.cuda_graph_modules is None:
            return True
        return session_name.strip().lower() in self.cuda_graph_modules

    def _create_hmonnx_session(self, onnx_path: str, session_name: str) -> HMONNXSession:
        if self._should_enable_cuda_graph(session_name):
            session: HMONNXSession = HMONNXCUDAGraphInference(
                onnx_path,
                enable_cuda_graph=True,
                warmup_runs=self.cuda_graph_warmup_runs,
                graph_warmup_runs=self.cuda_graph_graph_warmup_runs,
                clone_outputs=self.cuda_graph_clone_outputs,
            )
        else:
            session = HMONNXGrapInference(onnx_path)
        if not self.auto_offload:
            session.to(self.device)
        session.exec_device = self.exec_device
        return session

    def _get_session_cuda_graph_status(
        self, session: Optional[HMONNXSession]
    ) -> Dict[str, Union[bool, Optional[str]]]:
        if session is None:
            return {
                "enabled": False,
                "captured": False,
                "reason": "session not initialized",
                "backend": None,
            }
        backend = type(session).__name__
        if isinstance(session, HMONNXCUDAGraphInference):
            reason = None if session.has_captured_graph else session.capture_unavailable_reason
            return {
                "enabled": session.enable_cuda_graph,
                "captured": session.has_captured_graph,
                "reason": reason,
                "backend": backend,
            }
        return {
            "enabled": False,
            "captured": False,
            "reason": "session backend does not use cuda graph",
            "backend": backend,
        }

    def get_cuda_graph_status(self) -> Dict[str, Dict[str, Union[bool, Optional[str]]]]:
        return {
            "prefill": self._get_session_cuda_graph_status(self.prefill_session),
            "decode": self._get_session_cuda_graph_status(self.decode_session),
        }

    def _create_prefill_session(self):
        onnx_path = self.prefill_config["onnx"] if isinstance(self.prefill_config, dict) else self.prefill_config.onnx
        self.prefill_session = self._create_hmonnx_session(onnx_path, "prefill")
        prefill_max_memory = (
            self.prefill_auto_offload_max_memory
            if self.prefill_auto_offload_max_memory is not None
            else self.auto_offload_max_memory
        )
        self._apply_auto_offload(self.prefill_session, prefill_max_memory)

        self._prefill_inputs_name = _resolve_input_name(
            self.prefill_session,
            ("inputs_embeds", "input_1"),
            fallback=_infer_inputs_embeds_name,
        )
        self._prefill_past_seq_name = _resolve_input_name(
            self.prefill_session, ("past_seq_length", "valid_length")
        )
        self._prefill_current_seq_name = _resolve_input_name(
            self.prefill_session,
            ("current_input_length", "current_length"),
        )
        self._prefill_mask_name = _resolve_input_name(
            self.prefill_session,
            ("linear_attn_mask", "attention_mask", "attn_mask"),
        )

        self._prefill_inputs_info = self.prefill_session.get_input(
            self._prefill_inputs_name
        )
        self._prefill_mask_info = self.prefill_session.get_input(
            self._prefill_mask_name
        )
        self._prefill_past_seq_info = self.prefill_session.get_input(
            self._prefill_past_seq_name
        )
        self._prefill_current_seq_info = self.prefill_session.get_input(
            self._prefill_current_seq_name
        )

    def _create_decode_session(self):
        onnx_path = self.decode_config["onnx"] if isinstance(self.decode_config, dict) else self.decode_config.onnx
        self.decode_session = self._create_hmonnx_session(onnx_path, "decode")
        decode_max_memory = (
            self.decode_auto_offload_max_memory
            if self.decode_auto_offload_max_memory is not None
            else self.auto_offload_max_memory
        )
        self._apply_auto_offload(self.decode_session, decode_max_memory)

        self._decode_inputs_name = _resolve_input_name(
            self.decode_session,
            ("inputs_embeds", "input_1"),
            fallback=_infer_inputs_embeds_name,
        )
        self._decode_past_seq_name = _resolve_input_name(
            self.decode_session, ("past_seq_length", "valid_length")
        )
        self._decode_current_seq_name = _resolve_input_name(
            self.decode_session,
            ("current_input_length", "current_length"),
        )
        self._decode_mask_name = _resolve_input_name(
            self.decode_session,
            ("linear_attn_mask", "attention_mask", "attn_mask"),
        )

        self._decode_inputs_info = self.decode_session.get_input(
            self._decode_inputs_name
        )
        self._decode_mask_info = self.decode_session.get_input(self._decode_mask_name)
        self._decode_past_seq_info = self.decode_session.get_input(
            self._decode_past_seq_name
        )
        self._decode_current_seq_info = self.decode_session.get_input(
            self._decode_current_seq_name
        )

    def _release_prefill_session(self):
        self.prefill_session = None
        self._prefill_inputs_name = None
        self._prefill_past_seq_name = None
        self._prefill_current_seq_name = None
        self._prefill_mask_name = None
        self._prefill_inputs_info = None
        self._prefill_mask_info = None
        self._prefill_past_seq_info = None
        self._prefill_current_seq_info = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _release_decode_session(self):
        self.decode_session = None
        self._decode_inputs_name = None
        self._decode_past_seq_name = None
        self._decode_current_seq_name = None
        self._decode_mask_name = None
        self._decode_inputs_info = None
        self._decode_mask_info = None
        self._decode_past_seq_info = None
        self._decode_current_seq_info = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _ensure_prefill_session(self):
        if self.prefill_session is None:
            self._create_prefill_session()

    def _ensure_decode_session(self):
        if self.decode_session is None:
            self._create_decode_session()

    def _init_sessions(self):
        if self.resource_tight_mode:
            return
        self._create_prefill_session()
        self._create_decode_session()

    def _run_hmonnx(
        self, session: HMONNXSession, input_feed: Dict[str, torch.Tensor]
    ):
        outputs = session.run(input_feed)
        if not isinstance(outputs, (tuple, list)):
            outputs = (outputs,)
        output_names = session.get_output_names()
        output_map = {name: out for name, out in zip(output_names, outputs)}
        return tuple(outputs), output_map

    def _extract_logits(self, output_map: Dict[str, torch.Tensor]) -> torch.Tensor:
        if "logits" in output_map:
            return output_map["logits"]
        return next(iter(output_map.values()))

    def _update_linear_cache(
        self, cache_state: Dict[str, torch.Tensor], output_map: Dict[str, torch.Tensor]
    ) -> None:
        for name in list(cache_state.keys()):
            if name in output_map:
                cache_state[name] = _as_cache_value(cache_state[name], output_map[name])
                continue
            if name.startswith("past_conv_cache_"):
                branch, idx = _parse_conv_cache_name(name)
                if branch is None:
                    per_step = f"conv_cache_out_{idx}_0"
                    out_name = f"conv_cache_out_{idx}"
                else:
                    per_step = f"conv_cache_out_{branch}_{idx}_0"
                    out_name = f"conv_cache_out_{branch}_{idx}"
                if per_step in output_map:
                    cache_state[name] = _as_cache_value(
                        cache_state[name], output_map[per_step]
                    )
                    continue
                if out_name in output_map:
                    conv_out = output_map[out_name]
                    if (
                        conv_out.dim() >= 3
                        and conv_out.shape[-1] != cache_state[name].shape[-1]
                    ):
                        conv_out = conv_out[..., : cache_state[name].shape[-1]]
                    cache_state[name] = _as_cache_value(cache_state[name], conv_out)
            elif name.startswith("past_recurrent_state_"):
                idx = name.rsplit("_", 1)[-1]
                out_name = f"recurrent_state_out_{idx}"
                if out_name in output_map:
                    cache_state[name] = _as_cache_value(
                        cache_state[name], output_map[out_name]
                    )
                else:
                    per_step = f"recurrent_state_out_{idx}_0"
                    if per_step in output_map:
                        cache_state[name] = _as_cache_value(
                            cache_state[name], output_map[per_step]
                        )

    def _build_prefill_feed(
        self,
        chunk_input_ids: torch.Tensor,
        valid_len: int,
        past_seq_len: int,
        cache_state: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        assert self.token_embedding is not None
        assert self.prefill_session is not None
        inputs_embeds = _build_inputs_embeds(
            self.token_embedding,
            chunk_input_ids,
            self._prefill_inputs_info.shape[1],
            self.pad_token_id,
            self.device,
            self._prefill_inputs_info.dtype,
        )
        linear_attn_mask = _build_linear_attn_mask(
            valid_len, self._prefill_mask_info, self.device
        )

        batch_size = self._prefill_inputs_info.shape[0]
        past_seq_length = torch.full(
            (batch_size,),
            int(past_seq_len),
            dtype=self._prefill_past_seq_info.dtype,
            device=self.device,
        )
        current_input_length = torch.full(
            (batch_size,),
            int(valid_len),
            dtype=self._prefill_current_seq_info.dtype,
            device=self.device,
        )
        prefill_position_ids = torch.arange(
            past_seq_len,
            past_seq_len + self._prefill_inputs_info.shape[1],
            device=self.device,
            dtype=torch.int32,
        )

        feed: Dict[str, torch.Tensor] = {}
        for name in self.prefill_session.get_input_names():
            if name == self._prefill_inputs_name:
                feed[name] = inputs_embeds
            elif name == self._prefill_past_seq_name:
                feed[name] = past_seq_length
            elif name == self._prefill_current_seq_name:
                feed[name] = current_input_length
            elif name == self._prefill_mask_name:
                feed[name] = linear_attn_mask
            elif name in (
                "time_position_ids",
                "hight_position_ids",
                "width_position_ids",
            ):
                info = self.prefill_session.get_input(name)
                feed[name] = prefill_position_ids.to(dtype=info.dtype).reshape(
                    info.shape
                )
            elif name in cache_state:
                feed[name] = cache_state[name]
            else:
                info = self.prefill_session.get_input(name)
                feed[name] = torch.zeros(
                    info.shape, dtype=info.dtype, device=self.device
                )
        return feed

    def _build_decode_feed(
        self,
        token_ids: torch.Tensor,
        past_seq_len: int,
        cache_state: Dict[str, torch.Tensor],
        current_input_length: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        assert self.token_embedding is not None
        assert self.decode_session is not None
        if token_ids.dim() != 2:
            raise ValueError(
                f"token_ids must be [B, seq], got {tuple(token_ids.shape)}"
            )
        if current_input_length is None:
            current_input_length = int(token_ids.shape[1])
        if current_input_length <= 0:
            raise ValueError(
                f"current_input_length must be > 0, got {current_input_length}"
            )
        inputs_embeds = _build_inputs_embeds(
            self.token_embedding,
            token_ids,
            self._decode_inputs_info.shape[1],
            self.pad_token_id,
            self.device,
            self._decode_inputs_info.dtype,
        )
        linear_attn_mask = _build_linear_attn_mask(
            current_input_length, self._decode_mask_info, self.device
        )

        batch_size = self._decode_inputs_info.shape[0]
        past_seq_length = torch.full(
            (batch_size,),
            int(past_seq_len),
            dtype=self._decode_past_seq_info.dtype,
            device=self.device,
        )
        current_input_length = torch.full(
            (batch_size,),
            int(current_input_length),
            dtype=self._decode_current_seq_info.dtype,
            device=self.device,
        )
        decode_position_ids = torch.arange(
            past_seq_len,
            past_seq_len + self._decode_inputs_info.shape[1],
            device=self.device,
            dtype=torch.int32,
        )

        feed: Dict[str, torch.Tensor] = {}
        for name in self.decode_session.get_input_names():
            if name == self._decode_inputs_name:
                feed[name] = inputs_embeds
            elif name == self._decode_past_seq_name:
                feed[name] = past_seq_length
            elif name == self._decode_current_seq_name:
                feed[name] = current_input_length
            elif name == self._decode_mask_name:
                feed[name] = linear_attn_mask
            elif name in (
                "time_position_ids",
                "hight_position_ids",
                "width_position_ids",
            ):
                info = self.decode_session.get_input(name)
                feed[name] = decode_position_ids.to(dtype=info.dtype).reshape(
                    info.shape
                )
            elif name in cache_state:
                feed[name] = cache_state[name]
            else:
                info = self.decode_session.get_input(name)
                feed[name] = torch.zeros(
                    info.shape, dtype=info.dtype, device=self.device
                )
        return feed

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
    ) -> str:
        if max_new_tokens <= 0:
            return ""
        if input_ids.dim() != 2 or input_ids.shape[0] != 1:
            raise ValueError(
                f"input_ids must be [1, seq], got {tuple(input_ids.shape)}"
            )
        if self.token_embedding is None:
            raise ValueError(
                "token_embedding is not set, call set_input_embeddings first."
            )

        if (
            self.max_context_tokens is not None
            and input_ids.shape[1] > self.max_context_tokens
        ):
            input_ids = input_ids[:, -self.max_context_tokens :]

        total_prompt_len = int(input_ids.shape[1])
        if total_prompt_len <= 0:
            raise ValueError("Empty prompt is not supported.")

        self._ensure_prefill_session()
        prefill_cache_state = _alloc_cache_inputs(self.prefill_session, self.device)
        prefill_chunk_len = int(self._prefill_inputs_info.shape[1])

        last_prefill_logits = None
        past_seq_len = 0
        for start in range(0, total_prompt_len, prefill_chunk_len):
            end = min(start + prefill_chunk_len, total_prompt_len)
            chunk_ids = input_ids[:, start:end]
            valid_len = int(chunk_ids.shape[1])

            prefill_feed = self._build_prefill_feed(
                chunk_ids, valid_len, past_seq_len, prefill_cache_state
            )
            _, prefill_output_map = self._run_hmonnx(self.prefill_session, prefill_feed)
            prefill_logits = self._extract_logits(prefill_output_map)
            last_prefill_logits = _select_last_valid_logits(prefill_logits, valid_len)

            self._update_linear_cache(prefill_cache_state, prefill_output_map)
            past_seq_len += valid_len

        if last_prefill_logits is None:
            return ""

        history_token_ids = input_ids[0].tolist()
        first_step_logits = _apply_repetition_penalty(
            last_prefill_logits, history_token_ids, repetition_penalty
        )
        first_step_logits = _apply_presence_penalty(
            first_step_logits, history_token_ids, presence_penalty
        )
        next_token_id = _sample_next_token(
            first_step_logits,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )

        if self.resource_tight_mode:
            self._release_prefill_session()

        self._ensure_decode_session()
        decode_cache_state = _alloc_cache_inputs(self.decode_session, self.device)
        for name in decode_cache_state:
            if name in prefill_cache_state and (
                _is_kv_cache_name(name)
                or name.startswith(("past_conv_cache_", "past_recurrent_state_"))
            ):
                decode_cache_state[name] = prefill_cache_state[name]

        eos_token_id = tokenizer.eos_token_id
        generated_ids: List[int] = []
        streamer: Optional[TextStreamer] = None
        if stream_output:
            streamer = TextStreamer(
                tokenizer,
                skip_prompt=False,
                skip_special_tokens=True,
            )

        token_val = int(next_token_id[0][0].item())
        if eos_token_id is not None and token_val == eos_token_id:
            if streamer is not None:
                streamer.end()
            return ""
        generated_ids.append(token_val)
        history_token_ids.append(token_val)
        if streamer is not None:
            streamer.put(next_token_id.detach().cpu())

        current_token = next_token_id.to(self.device)
        for _ in range(max_new_tokens - 1):
            decode_feed = self._build_decode_feed(
                current_token, past_seq_len, decode_cache_state, current_input_length=1
            )
            _, decode_output_map = self._run_hmonnx(self.decode_session, decode_feed)
            decode_logits = self._extract_logits(decode_output_map)
            decode_logits = _select_last_valid_logits(decode_logits, 1)

            decode_logits = _apply_repetition_penalty(
                decode_logits, history_token_ids, repetition_penalty
            )
            decode_logits = _apply_presence_penalty(
                decode_logits, history_token_ids, presence_penalty
            )
            next_token_id = _sample_next_token(
                decode_logits,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
            self._update_linear_cache(decode_cache_state, decode_output_map)

            token_val = int(next_token_id[0][0].item())
            if eos_token_id is not None and token_val == eos_token_id:
                break
            generated_ids.append(token_val)
            history_token_ids.append(token_val)
            if streamer is not None:
                streamer.put(next_token_id.detach().cpu())
            current_token = next_token_id.to(self.device)
            past_seq_len += 1

        out_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        if streamer is not None:
            streamer.end()
        if self.resource_tight_mode:
            self._release_decode_session()
        return out_text

    def chat(
        self,
        prompt: str,
        tokenizer,
        history: Optional[List[Dict[str, str]]] = None,
        system_prompt: str = "You are a helpful assistant.",
        max_new_tokens: int = 256,
        enable_thinking: bool = False,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        stream_output: bool = False,
    ) -> str:
        turns = [] if history is None else list(history)
        turns.append({"role": "user", "content": prompt})

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(turns)

        kwargs = {"enable_thinking": enable_thinking}

        if hasattr(tokenizer, "apply_chat_template"):
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, **kwargs
            )
        else:
            text = "\n".join(
                [f"{m['role']}: {m['content']}" for m in messages] + ["assistant:"]
            )

        input_ids = tokenizer([text], return_tensors="pt").input_ids
        if (
            self.max_context_tokens is not None
            and input_ids.shape[1] > self.max_context_tokens
        ):
            input_ids = input_ids[:, -self.max_context_tokens :]
        return self.generate(
            input_ids,
            tokenizer,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            stream_output=stream_output,
        )
