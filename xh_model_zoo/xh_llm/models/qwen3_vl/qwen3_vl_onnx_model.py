# Copyright 2025 HOUMO AI
#
# File: qwen3_vl_onnx_model.py
# Description:
#   Qwen3 VL ONNX/HMONNX runtime implementation built on graph inference.
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

from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import TextStreamer
from transformers.video_utils import VideoMetadata

from xhquant.core import CacheTensor
from xhquant.xhonnxruntime import (
    AutoOffloadGraphModel,
    HMONNXCUDAGraphInference,
    HMONNXGrapInference,
)

from .device_dtype_mixin import DeviceDtypeMixin
from .postprocess import VLLMPresencePenaltyLogitsProcessor


HMONNXSession = Union[HMONNXGrapInference, HMONNXCUDAGraphInference]


def _cfg_get(config, key: str, default=None):
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def _resolve_optional_input_name(session: Optional[HMONNXSession], candidates: Tuple[str, ...]) -> Optional[str]:
    if session is None:
        return None
    input_names = session.get_input_names()
    for name in candidates:
        if name in input_names:
            return name
    return None


def _normalize_visual_embeds(
    visual_embeds: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    if visual_embeds is None:
        return None
    if visual_embeds.dim() == 3 and visual_embeds.shape[0] == 1:
        return visual_embeds.squeeze(0)
    return visual_embeds


def _normalize_deepstack_embeds(
    deepstack_embeds: Optional[Sequence[torch.Tensor]],
) -> Optional[List[torch.Tensor]]:
    if deepstack_embeds is None:
        return None
    normalized: List[torch.Tensor] = []
    for deepstack_embed in deepstack_embeds:
        if deepstack_embed.dim() == 3 and deepstack_embed.shape[0] == 1:
            normalized.append(deepstack_embed.squeeze(0))
        else:
            normalized.append(deepstack_embed)
    return normalized


def _build_stop_token_ids(tokenizer, extra_token_ids: Optional[Sequence[int]] = None) -> set[int]:
    stop_token_ids: set[int] = set()
    tokenizer_eos = getattr(tokenizer, "eos_token_id", None)
    if isinstance(tokenizer_eos, (list, tuple)):
        stop_token_ids.update(int(token_id) for token_id in tokenizer_eos)
    elif tokenizer_eos is not None:
        stop_token_ids.add(int(tokenizer_eos))
    if extra_token_ids is not None:
        stop_token_ids.update(int(token_id) for token_id in extra_token_ids)
    return stop_token_ids


def _infer_inputs_embeds_name(session: HMONNXSession) -> str:
    for name in session.get_input_names():
        info = session.get_input(name)
        if info.dtype in (torch.float16, torch.float32, torch.bfloat16) and len(info.shape) == 3:
            return name
    return session.get_input_names()[0]


def _resolve_input_name(
    session: HMONNXSession,
    candidates: Tuple[str, ...],
    fallback=None,
) -> str:
    input_names = session.get_input_names()
    for name in candidates:
        if name in input_names:
            return name
    if fallback is not None:
        return fallback(session)
    raise ValueError(f"None of {candidates} found in inputs: {input_names}")


def _alloc_cache_inputs(session: HMONNXSession, device: torch.device) -> Dict[str, torch.Tensor]:
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


def _apply_repetition_penalty(
    logits: torch.Tensor,
    token_ids: List[int],
    repetition_penalty: float,
) -> torch.Tensor:
    if repetition_penalty == 1.0 or len(token_ids) == 0:
        return logits
    unique_token_ids = torch.unique(torch.tensor(token_ids, dtype=torch.long, device=logits.device))
    updated = logits.clone()
    values = updated[..., unique_token_ids]
    values = torch.where(
        values < 0,
        values * repetition_penalty,
        values / repetition_penalty,
    )
    updated[..., unique_token_ids] = values
    return updated


def _apply_presence_penalty(
    logits: torch.Tensor,
    token_ids: List[int],
    presence_penalty: float,
) -> torch.Tensor:
    if presence_penalty == 0.0 or len(token_ids) == 0:
        return logits
    unique_token_ids = torch.unique(torch.tensor(token_ids, dtype=torch.long, device=logits.device))
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
            filtered_logits < kth_values,
            float("-inf"),
        )

    if 0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(filtered_logits, descending=True, dim=-1)
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


def decode_next_token(
    tokenizer,
    logits: torch.Tensor,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 0,
):
    next_tokens = _sample_next_token(
        logits,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )
    next_token_str = tokenizer.batch_decode(next_tokens, skip_special_tokens=True)
    return next_tokens, next_token_str


class GraphHMONNXModelBase(DeviceDtypeMixin):
    def __init__(
        self,
        prefill,
        decode,
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
        self.pad_token_id = int(pad_token_id)
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
        self.rope_deltas: Optional[torch.Tensor] = None

        self._prefill_inputs_name = None
        self._prefill_past_seq_name = None
        self._prefill_current_seq_name = None
        self._prefill_mask_name = None
        self._prefill_inputs_info = None
        self._prefill_mask_info = None
        self._prefill_past_seq_info = None
        self._prefill_current_seq_info = None

        self._decode_inputs_name = None
        self._decode_past_seq_name = None
        self._decode_current_seq_name = None
        self._decode_mask_name = None
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
        if not self.auto_offload or not torch.cuda.is_available():
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
                "Please increase GPU budgets or provide more GPU devices in max_memory."
            ) from e

    def _should_enable_cuda_graph(self, session_name: str) -> bool:
        if not self.enable_cuda_graph:
            return False
        if self.cuda_graph_modules is None:
            return True
        return session_name.strip().lower() in self.cuda_graph_modules

    def _create_hmonnx_session(
        self,
        onnx_path: str,
        session_name: str,
    ) -> HMONNXSession:
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
        session.exec_device = self._exec_device
        return session

    def _get_session_cuda_graph_status(
        self,
        session: Optional[HMONNXSession],
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
            self.prefill_session,
            ("past_seq_length", "valid_length"),
        )
        self._prefill_current_seq_name = _resolve_input_name(
            self.prefill_session,
            ("current_input_length", "current_length"),
        )
        self._prefill_mask_name = _resolve_optional_input_name(
            self.prefill_session,
            ("linear_attn_mask", "attention_mask", "attn_mask"),
        )
        self._prefill_inputs_info = self.prefill_session.get_input(self._prefill_inputs_name)
        self._prefill_mask_info = (
            self.prefill_session.get_input(self._prefill_mask_name) if self._prefill_mask_name is not None else None
        )
        self._prefill_past_seq_info = self.prefill_session.get_input(self._prefill_past_seq_name)
        self._prefill_current_seq_info = self.prefill_session.get_input(self._prefill_current_seq_name)

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
            self.decode_session,
            ("past_seq_length", "valid_length"),
        )
        self._decode_current_seq_name = _resolve_input_name(
            self.decode_session,
            ("current_input_length", "current_length"),
        )
        self._decode_mask_name = _resolve_optional_input_name(
            self.decode_session,
            ("linear_attn_mask", "attention_mask", "attn_mask"),
        )
        self._decode_inputs_info = self.decode_session.get_input(self._decode_inputs_name)
        self._decode_mask_info = (
            self.decode_session.get_input(self._decode_mask_name) if self._decode_mask_name is not None else None
        )
        self._decode_past_seq_info = self.decode_session.get_input(self._decode_past_seq_name)
        self._decode_current_seq_info = self.decode_session.get_input(self._decode_current_seq_name)

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
        self,
        session: HMONNXSession,
        input_feed: Dict[str, torch.Tensor],
    ):
        outputs = session.run(input_feed)
        if not isinstance(outputs, (tuple, list)):
            outputs = (outputs,)
        output_names = session.get_output_names()
        output_map = {name: out for name, out in zip(output_names, outputs, strict=False)}
        return tuple(outputs), output_map

    def _extract_logits(self, output_map: Dict[str, torch.Tensor]) -> torch.Tensor:
        if "logits" in output_map:
            return output_map["logits"]
        return next(iter(output_map.values()))

    def _update_linear_cache(
        self,
        cache_state: Dict[str, torch.Tensor],
        output_map: Dict[str, torch.Tensor],
    ) -> None:
        for name in list(cache_state.keys()):
            if name in output_map:
                cache_state[name] = _as_cache_value(cache_state[name], output_map[name])
                continue
            if name.startswith("past_conv_cache_"):
                idx = name.rsplit("_", 1)[-1]
                per_step = f"conv_cache_out_{idx}_0"
                if per_step in output_map:
                    cache_state[name] = _as_cache_value(cache_state[name], output_map[per_step])
                    continue
                out_name = f"conv_cache_out_{idx}"
                if out_name in output_map:
                    conv_out = output_map[out_name]
                    if conv_out.dim() >= 3 and conv_out.shape[-1] != cache_state[name].shape[-1]:
                        conv_out = conv_out[..., : cache_state[name].shape[-1]]
                    cache_state[name] = _as_cache_value(cache_state[name], conv_out)
            elif name.startswith("past_recurrent_state_"):
                idx = name.rsplit("_", 1)[-1]
                out_name = f"recurrent_state_out_{idx}"
                if out_name in output_map:
                    cache_state[name] = _as_cache_value(cache_state[name], output_map[out_name])
                else:
                    per_step = f"recurrent_state_out_{idx}_0"
                    if per_step in output_map:
                        cache_state[name] = _as_cache_value(cache_state[name], output_map[per_step])


class Qwen3VLONNXModel(GraphHMONNXModelBase):
    def __init__(
        self,
        image_feature,
        prefill,
        decode,
        kv_cache=None,
        cache_len: int = 2048,
        image_size_w: int = 448,
        image_size_h: int = 448,
        max_size_t: int = 2,
        resize_v1: bool = True,
        presence_penalty: float = 0.0,
        max_context_tokens: Optional[int] = None,
        auto_offload: bool = True,
        auto_offload_max_memory: Optional[Dict[Union[int, str], Union[int, str]]] = None,
        prefill_auto_offload_max_memory: Optional[Dict[Union[int, str], Union[int, str]]] = None,
        decode_auto_offload_max_memory: Optional[Dict[Union[int, str], Union[int, str]]] = None,
        vision_auto_offload_max_memory: Optional[Dict[Union[int, str], Union[int, str]]] = None,
        resource_tight_mode: bool = False,
        pad_token_id: int = 0,
        enable_cuda_graph: bool = False,
        cuda_graph_modules: Optional[Iterable[str]] = None,
        cuda_graph_warmup_runs: int = 3,
        cuda_graph_graph_warmup_runs: int = 6,
        cuda_graph_clone_outputs: bool = True,
    ):
        resolved_max_context_tokens = int(max_context_tokens or cache_len)
        super().__init__(
            prefill=prefill,
            decode=decode,
            max_context_tokens=resolved_max_context_tokens,
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
        del kv_cache

        self.image_feature_config = image_feature
        self.vision_auto_offload_max_memory = vision_auto_offload_max_memory

        self.cache_len = resolved_max_context_tokens
        self.image_size_w = int(image_size_w)
        self.image_size_h = int(image_size_h)
        self.max_size_t = int(max_size_t)
        self.resize_v1 = bool(resize_v1)
        self.presence_penalty = float(presence_penalty)
        self.logits_processor = VLLMPresencePenaltyLogitsProcessor(self.presence_penalty, 0)

        self.image_token_id = 151655
        self.video_token_id = 151656
        self.vision_start_token_id = 151652
        self.vision_end_token_id = 151653
        self.vision_token_id = 151654
        self.eos_token_id = [151645, 151643]
        self.spatial_merge_size = 2
        self.temporal_patch_size = int(_cfg_get(image_feature, "temporal_patch_size", 2))
        self.patch_size = int(_cfg_get(image_feature, "patch_size", 16))
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size

        self.image_feature_session: Optional[HMONNXSession] = None
        self._prefill_cache_state: Optional[Dict[str, torch.Tensor]] = None
        self._decode_cache_state: Optional[Dict[str, torch.Tensor]] = None

        self.prefill_input_sequence_length = int(_cfg_get(prefill, "input_sequence_length", 0) or 0)

    def _create_prefill_session(self):
        super()._create_prefill_session()
        self._prefill_time_pos_name = _resolve_input_name(self.prefill_session, ("time_position_ids",))
        self._prefill_height_pos_name = _resolve_input_name(
            self.prefill_session,
            ("hight_position_ids", "height_position_ids"),
        )
        self._prefill_width_pos_name = _resolve_input_name(self.prefill_session, ("width_position_ids",))
        self._prefill_deepstack_names = [
            _resolve_optional_input_name(
                self.prefill_session,
                (
                    f"deepstack_image_embed_{idx}",
                    f"deepstack_embed_{idx}",
                    f"deepstack_hidden_state_{idx}",
                ),
            )
            for idx in range(3)
        ]
        if getattr(self, "prefill_input_sequence_length", 0) <= 0:
            self.prefill_input_sequence_length = int(self._prefill_inputs_info.shape[1])

    def _create_decode_session(self):
        super()._create_decode_session()
        self._decode_time_pos_name = _resolve_input_name(self.decode_session, ("time_position_ids",))
        self._decode_height_pos_name = _resolve_input_name(
            self.decode_session,
            ("hight_position_ids", "height_position_ids"),
        )
        self._decode_width_pos_name = _resolve_input_name(self.decode_session, ("width_position_ids",))
        self._decode_deepstack_names = [
            _resolve_optional_input_name(
                self.decode_session,
                (
                    f"deepstack_image_embed_{idx}",
                    f"deepstack_embed_{idx}",
                    f"deepstack_hidden_state_{idx}",
                ),
            )
            for idx in range(3)
        ]

    def _release_prefill_session(self):
        super()._release_prefill_session()
        self._prefill_time_pos_name = None
        self._prefill_height_pos_name = None
        self._prefill_width_pos_name = None
        self._prefill_deepstack_names = [None, None, None]

    def _release_decode_session(self):
        super()._release_decode_session()
        self._decode_time_pos_name = None
        self._decode_height_pos_name = None
        self._decode_width_pos_name = None
        self._decode_deepstack_names = [None, None, None]

    def get_cuda_graph_status(self) -> Dict[str, Dict[str, Union[bool, Optional[str]]]]:
        status = super().get_cuda_graph_status()
        status["vision"] = self._get_session_cuda_graph_status(self.image_feature_session)
        return status

    def _init_image_feature_session(self):
        if self.image_feature_session is not None:
            return
        onnx_path = _cfg_get(self.image_feature_config, "onnx")
        if onnx_path is None:
            raise ValueError("image_feature.onnx is required")
        self.image_feature_session = self._create_hmonnx_session(str(onnx_path), "vision")
        vision_max_memory = (
            self.vision_auto_offload_max_memory
            if self.vision_auto_offload_max_memory is not None
            else self.auto_offload_max_memory
        )
        self._apply_auto_offload(self.image_feature_session, vision_max_memory)

    def init_image_feature(self):
        self._init_image_feature_session()

    def save_image_feature_golden(self, output_dir):
        self._init_image_feature_session()
        if hasattr(self.image_feature_session, "step"):
            self.image_feature_session.step = 0
        self.image_feature_session.save_golden = True
        self.image_feature_session.golden_dir = output_dir

    def release_image_feature(self):
        self.image_feature_session = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def init_prefill(self):
        self._ensure_prefill_session()
        self._prefill_cache_state = None

    def init_decode(self):
        self._ensure_decode_session()
        self._decode_cache_state = _alloc_cache_inputs(self.decode_session, self.device)
        if self._prefill_cache_state is None:
            return
        for name in self._decode_cache_state:
            if name in self._prefill_cache_state:
                self._decode_cache_state[name] = self._prefill_cache_state[name]

    def release_prefill_session(self):
        self._release_prefill_session()

    def release_decode_session(self):
        self._release_decode_session()
        self._decode_cache_state = None

    def _move_visual_output(self, value):
        if not isinstance(value, torch.Tensor):
            return value
        if value.is_floating_point():
            return value.to(device=self.device, dtype=self.dtype)
        return value.to(device=self.device)

    def preprocess_visual(self, inputs):
        hidden_states = inputs["hm_pixel_values"][0].to(self.device)
        return (hidden_states.to(dtype=torch.float16),)

    def extract_image_features(self, visual_inputs: Sequence[Tensor]):
        self._init_image_feature_session()
        outputs = self.image_feature_session(*visual_inputs)
        if isinstance(outputs, (tuple, list)):
            return tuple(self._move_visual_output(output) for output in outputs)
        return self._move_visual_output(outputs)

    def extract_all_image_features(self, hm_pixel_values):
        if hm_pixel_values is None:
            raise ValueError("hm_pixel_values is required")

        if isinstance(hm_pixel_values, torch.Tensor):
            outputs = self.extract_image_features((hm_pixel_values.to(device=self.device, dtype=torch.float16),))
            if isinstance(outputs, (tuple, list)):
                normalized = []
                for output in outputs:
                    if output.dim() == 3 and output.shape[0] == 1:
                        normalized.append(output.squeeze(0))
                    else:
                        normalized.append(output)
                return tuple(normalized)
            if outputs.dim() == 3 and outputs.shape[0] == 1:
                return outputs.squeeze(0)
            return outputs

        if not isinstance(hm_pixel_values, Sequence) or len(hm_pixel_values) == 0:
            raise ValueError("hm_pixel_values must be a non-empty tensor or sequence of tensors")

        stacked_outputs: Optional[List[List[torch.Tensor]]] = None
        for pixel_values in hm_pixel_values:
            current_outputs = self.extract_image_features((pixel_values.to(device=self.device, dtype=torch.float16),))
            if not isinstance(current_outputs, (tuple, list)):
                current_outputs = (current_outputs,)
            normalized_outputs = []
            for output in current_outputs:
                if output.dim() == 3 and output.shape[0] == 1:
                    normalized_outputs.append(output.squeeze(0))
                else:
                    normalized_outputs.append(output)
            if stacked_outputs is None:
                stacked_outputs = [[] for _ in range(len(normalized_outputs))]
            for idx, output in enumerate(normalized_outputs):
                stacked_outputs[idx].append(output)

        assert stacked_outputs is not None
        merged_outputs = [torch.cat(parts, dim=0) for parts in stacked_outputs]
        return tuple(merged_outputs)

    def _build_video_raw_clip(self, video_tensor: torch.Tensor) -> torch.Tensor:
        if video_tensor.dim() != 4:
            raise ValueError(
                f"Expected sampled video tensor with shape [T, C, H, W], but got {tuple(video_tensor.shape)}"
            )

        video_tensor = video_tensor.float()
        target_t = self.max_size_t
        if video_tensor.shape[0] != target_t:
            if video_tensor.shape[0] > target_t:
                indices = torch.linspace(0, video_tensor.shape[0] - 1, target_t).round().long()
                video_tensor = video_tensor.index_select(0, indices)
            else:
                pad_count = target_t - video_tensor.shape[0]
                pad_frames = video_tensor[-1:].repeat(pad_count, 1, 1, 1)
                video_tensor = torch.cat([video_tensor, pad_frames], dim=0)

        if video_tensor.shape[-2:] != (self.image_size_h, self.image_size_w):
            video_tensor = F.interpolate(
                video_tensor,
                size=(self.image_size_h, self.image_size_w),
                mode="bilinear",
                align_corners=False,
            )

        return video_tensor.permute(1, 0, 2, 3).unsqueeze(0).contiguous()

    def _build_sampled_video_metadata(self, video_tensor: torch.Tensor, sample_fps: float) -> list[VideoMetadata]:
        num_frames = int(video_tensor.shape[0])
        duration = None if sample_fps <= 0 else num_frames / sample_fps
        return [
            VideoMetadata(
                total_num_frames=num_frames,
                fps=sample_fps,
                width=int(video_tensor.shape[-1]),
                height=int(video_tensor.shape[-2]),
                duration=duration,
                video_backend="sampled_clip",
                frames_indices=list(range(num_frames)),
            )
        ]

    def get_rope_index(
        self,
        input_ids: torch.LongTensor,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if video_grid_thw is not None:
            video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
            video_grid_thw[:, 0] = 1

        spatial_merge_size = self.spatial_merge_size
        image_token_id = self.image_token_id
        video_token_id = self.video_token_id
        vision_start_token_id = self.vision_start_token_id
        mrope_position_deltas = []
        if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
            total_input_ids = input_ids
            if attention_mask is None:
                attention_mask = torch.ones_like(total_input_ids)
            position_ids = torch.ones(
                3,
                input_ids.shape[0],
                input_ids.shape[1],
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            image_index, video_index = 0, 0
            for batch_idx, batch_input_ids in enumerate(total_input_ids):
                batch_input_ids = batch_input_ids[attention_mask[batch_idx] == 1]
                vision_start_indices = torch.argwhere(batch_input_ids == vision_start_token_id).squeeze(1)
                vision_tokens = batch_input_ids[vision_start_indices + 1]
                image_nums = (vision_tokens == image_token_id).sum()
                video_nums = (vision_tokens == video_token_id).sum()
                input_tokens = batch_input_ids.tolist()
                llm_pos_ids_list: list[torch.Tensor] = []
                start = 0
                remain_images, remain_videos = image_nums, video_nums
                for _ in range(image_nums + video_nums):
                    if image_token_id in input_tokens and remain_images > 0:
                        end_image = input_tokens.index(image_token_id, start)
                    else:
                        end_image = len(input_tokens) + 1
                    if video_token_id in input_tokens and remain_videos > 0:
                        end_video = input_tokens.index(video_token_id, start)
                    else:
                        end_video = len(input_tokens) + 1
                    if end_image < end_video:
                        t, h, w = (
                            image_grid_thw[image_index][0],
                            image_grid_thw[image_index][1],
                            image_grid_thw[image_index][2],
                        )
                        image_index += 1
                        remain_images -= 1
                        end = end_image
                    else:
                        t, h, w = (
                            video_grid_thw[video_index][0],
                            video_grid_thw[video_index][1],
                            video_grid_thw[video_index][2],
                        )
                        video_index += 1
                        remain_videos -= 1
                        end = end_video
                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        t.item(),
                        h.item() // spatial_merge_size,
                        w.item() // spatial_merge_size,
                    )
                    text_len = end - start
                    start_index = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
                    llm_pos_ids_list.append(
                        torch.arange(text_len, device=input_ids.device).view(1, -1).expand(3, -1) + start_index
                    )

                    t_index = (
                        torch.arange(llm_grid_t, device=input_ids.device)
                        .view(-1, 1)
                        .expand(-1, llm_grid_h * llm_grid_w)
                        .flatten()
                    )
                    h_index = (
                        torch.arange(llm_grid_h, device=input_ids.device)
                        .view(1, -1, 1)
                        .expand(llm_grid_t, -1, llm_grid_w)
                        .flatten()
                    )
                    w_index = (
                        torch.arange(llm_grid_w, device=input_ids.device)
                        .view(1, 1, -1)
                        .expand(llm_grid_t, llm_grid_h, -1)
                        .flatten()
                    )
                    llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + start_index)
                    start = end + llm_grid_t * llm_grid_h * llm_grid_w

                if start < len(input_tokens):
                    start_index = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
                    text_len = len(input_tokens) - start
                    llm_pos_ids_list.append(
                        torch.arange(text_len, device=input_ids.device).view(1, -1).expand(3, -1) + start_index
                    )

                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                position_ids[..., batch_idx, attention_mask[batch_idx] == 1] = llm_positions.to(position_ids.device)
                mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[batch_idx]))
            deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
            return position_ids, deltas

        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(input_ids.device)
            max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
            deltas = max_position_ids + 1 - attention_mask.shape[-1]
            return position_ids, deltas

        position_ids = (
            torch.arange(input_ids.shape[1], device=input_ids.device).view(1, 1, -1).expand(3, input_ids.shape[0], -1)
        )
        deltas = torch.zeros(
            [input_ids.shape[0]],
            device=input_ids.device,
            dtype=input_ids.dtype,
        ).unsqueeze(1)
        return position_ids, deltas

    def _build_deepstack_tensors(
        self,
        inputs_embeds: torch.Tensor,
        image_mask: Optional[torch.Tensor],
        deepstack_image_embeds: Optional[Sequence[torch.Tensor]],
        video_mask: Optional[torch.Tensor],
        deepstack_video_embeds: Optional[Sequence[torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        deepstack_outputs = []
        for layer_index in range(3):
            layer_embed = torch.zeros_like(inputs_embeds)
            if (
                image_mask is not None
                and deepstack_image_embeds is not None
                and layer_index < len(deepstack_image_embeds)
            ):
                layer_embed = layer_embed.masked_scatter(
                    image_mask,
                    deepstack_image_embeds[layer_index].to(
                        device=layer_embed.device,
                        dtype=layer_embed.dtype,
                    ),
                )
            if (
                video_mask is not None
                and deepstack_video_embeds is not None
                and layer_index < len(deepstack_video_embeds)
            ):
                layer_embed = layer_embed.masked_scatter(
                    video_mask,
                    deepstack_video_embeds[layer_index].to(
                        device=layer_embed.device,
                        dtype=layer_embed.dtype,
                    ),
                )
            deepstack_outputs.append(layer_embed)
        return tuple(deepstack_outputs)  # type: ignore[return-value]

    def _prepare_prefill_tensors(self, data: dict):
        if self.token_embedding is None:
            raise ValueError("token_embedding is not set, call set_input_embeddings first.")
        if self.prefill_session is None:
            raise RuntimeError("prefill session is not initialized")

        input_ids = data["input_ids"]
        if input_ids.dim() != 2 or input_ids.shape[0] != 1:
            raise ValueError(f"input_ids must be [1, seq], got {tuple(input_ids.shape)}")

        prompt_len = int(input_ids.shape[1])
        if prompt_len <= 0:
            raise ValueError("Empty prompt is not supported.")
        if self.max_context_tokens is not None and prompt_len > self.max_context_tokens:
            raise ValueError(f"Prompt too long: prompt_len={prompt_len}, max_context_tokens={self.max_context_tokens}")

        prefill_chunk_len = int(self._prefill_inputs_info.shape[1])
        target_seq_len = ((prompt_len + prefill_chunk_len - 1) // prefill_chunk_len) * prefill_chunk_len
        pad_len = target_seq_len - prompt_len

        input_ids = input_ids.to(device=self.device, dtype=torch.long)
        if pad_len > 0:
            padding_ids = torch.full(
                (1, pad_len),
                self.pad_token_id,
                dtype=torch.long,
                device=self.device,
            )
            input_ids = torch.cat([input_ids, padding_ids], dim=-1)

        attention_mask = data.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones((1, prompt_len), dtype=torch.long, device=self.device)
        else:
            attention_mask = attention_mask.to(device=self.device, dtype=torch.long)
        if pad_len > 0:
            padding_mask = torch.zeros((1, pad_len), dtype=attention_mask.dtype, device=self.device)
            attention_mask = torch.cat([attention_mask, padding_mask], dim=-1)

        inputs_embeds = self.token_embedding(input_ids).to(dtype=self._prefill_inputs_info.dtype)

        image_mask = None
        video_mask = None

        image_embeds = _normalize_visual_embeds(data.get("image_embeds"))
        image_token_count = int((input_ids == self.image_token_id).sum().item())
        if image_token_count > 0:
            if image_embeds is None:
                raise ValueError("image_embeds is required when image tokens are present")
            if image_embeds.shape[0] != image_token_count:
                raise ValueError(
                    "Image features and image tokens do not match: "
                    f"tokens={image_token_count}, features={image_embeds.shape[0]}"
                )
            image_mask = (input_ids == self.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(
                image_mask,
                image_embeds.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype),
            )

        video_embeds = _normalize_visual_embeds(data.get("video_embeds"))
        video_token_count = int((input_ids == self.video_token_id).sum().item())
        if video_token_count > 0:
            if video_embeds is None:
                raise ValueError("video_embeds is required when video tokens are present")
            if video_embeds.shape[0] != video_token_count:
                raise ValueError(
                    "Video features and video tokens do not match: "
                    f"tokens={video_token_count}, features={video_embeds.shape[0]}"
                )
            video_mask = (input_ids == self.video_token_id).unsqueeze(-1).expand_as(inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(
                video_mask,
                video_embeds.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype),
            )

        deepstack_image_embeds = _normalize_deepstack_embeds(data.get("deepstack_image_embeds"))
        deepstack_video_embeds = _normalize_deepstack_embeds(data.get("deepstack_video_embeds"))
        deepstack_tensors = self._build_deepstack_tensors(
            inputs_embeds,
            image_mask,
            deepstack_image_embeds,
            video_mask,
            deepstack_video_embeds,
        )

        image_grid_thw = data.get("image_grid_thw")
        if image_grid_thw is not None:
            image_grid_thw = image_grid_thw.to(device=self.device, dtype=torch.long)
        video_grid_thw = data.get("video_grid_thw")
        if video_grid_thw is not None:
            video_grid_thw = video_grid_thw.to(device=self.device, dtype=torch.long)

        position_ids, rope_deltas = self.get_rope_index(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            attention_mask=attention_mask,
        )
        if pad_len > 0:
            rope_deltas = rope_deltas + pad_len
        self.rope_deltas = rope_deltas

        return (
            inputs_embeds,
            deepstack_tensors,
            position_ids[0, 0],
            position_ids[1, 0],
            position_ids[2, 0],
            prompt_len,
            target_seq_len,
        )

    def _build_mm_prefill_feed(
        self,
        chunk_embeds: torch.Tensor,
        deepstack_chunks: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        time_position_ids: torch.Tensor,
        height_position_ids: torch.Tensor,
        width_position_ids: torch.Tensor,
        valid_len: int,
        past_seq_len: int,
        cache_state: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
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
        linear_attn_mask = None
        if self._prefill_mask_name is not None:
            linear_attn_mask = torch.zeros(
                self._prefill_mask_info.shape,
                dtype=self._prefill_mask_info.dtype,
                device=self.device,
            )
            if valid_len > 0:
                linear_attn_mask[..., :valid_len] = 1

        deepstack_feed = {}
        for idx, name in enumerate(self._prefill_deepstack_names):
            if name is None:
                continue
            info = self.prefill_session.get_input(name)
            deepstack_feed[name] = deepstack_chunks[idx].to(dtype=info.dtype, device=self.device)

        feed = {}
        for name in self.prefill_session.get_input_names():
            if name == self._prefill_inputs_name:
                feed[name] = chunk_embeds
            elif name == self._prefill_time_pos_name:
                info = self.prefill_session.get_input(name)
                feed[name] = time_position_ids.to(dtype=info.dtype, device=self.device)
            elif name == self._prefill_height_pos_name:
                info = self.prefill_session.get_input(name)
                feed[name] = height_position_ids.to(dtype=info.dtype, device=self.device)
            elif name == self._prefill_width_pos_name:
                info = self.prefill_session.get_input(name)
                feed[name] = width_position_ids.to(dtype=info.dtype, device=self.device)
            elif name == self._prefill_past_seq_name:
                feed[name] = past_seq_length
            elif name == self._prefill_current_seq_name:
                feed[name] = current_input_length
            elif name == self._prefill_mask_name and linear_attn_mask is not None:
                feed[name] = linear_attn_mask
            elif name in deepstack_feed:
                feed[name] = deepstack_feed[name]
            elif name in cache_state:
                feed[name] = cache_state[name]
            else:
                info = self.prefill_session.get_input(name)
                feed[name] = torch.zeros(info.shape, dtype=info.dtype, device=self.device)
        return feed

    def _build_mm_decode_feed(
        self,
        token_ids: torch.Tensor,
        past_seq_len: int,
        cache_state: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        if self.token_embedding is None:
            raise ValueError("token_embedding is not set, call set_input_embeddings first.")
        if self.rope_deltas is None:
            raise RuntimeError("rope_deltas is not initialized")

        current_input_length_int = int(token_ids.shape[1])
        inputs_embeds = self.token_embedding(token_ids.to(device=self.device, dtype=torch.long)).to(
            dtype=self._decode_inputs_info.dtype
        )

        linear_attn_mask = None
        if self._decode_mask_name is not None:
            linear_attn_mask = torch.zeros(
                self._decode_mask_info.shape,
                dtype=self._decode_mask_info.dtype,
                device=self.device,
            )
            linear_attn_mask[..., :current_input_length_int] = 1
        delta = (past_seq_len + self.rope_deltas).view(-1)

        batch_size = self._decode_inputs_info.shape[0]
        past_seq_length = torch.full(
            (batch_size,),
            int(past_seq_len),
            dtype=self._decode_past_seq_info.dtype,
            device=self.device,
        )
        current_input_length = torch.full(
            (batch_size,),
            current_input_length_int,
            dtype=self._decode_current_seq_info.dtype,
            device=self.device,
        )

        feed = {}
        for name in self.decode_session.get_input_names():
            if name == self._decode_inputs_name:
                feed[name] = inputs_embeds
            elif name == self._decode_time_pos_name:
                info = self.decode_session.get_input(name)
                feed[name] = delta.to(dtype=info.dtype, device=self.device)
            elif name == self._decode_height_pos_name:
                info = self.decode_session.get_input(name)
                feed[name] = delta.to(dtype=info.dtype, device=self.device)
            elif name == self._decode_width_pos_name:
                info = self.decode_session.get_input(name)
                feed[name] = delta.to(dtype=info.dtype, device=self.device)
            elif name == self._decode_past_seq_name:
                feed[name] = past_seq_length
            elif name == self._decode_current_seq_name:
                feed[name] = current_input_length
            elif name == self._decode_mask_name and linear_attn_mask is not None:
                feed[name] = linear_attn_mask
            elif name in self._decode_deepstack_names:
                info = self.decode_session.get_input(name)
                feed[name] = torch.zeros(info.shape, dtype=info.dtype, device=self.device)
            elif name in cache_state:
                feed[name] = cache_state[name]
            else:
                info = self.decode_session.get_input(name)
                feed[name] = torch.zeros(info.shape, dtype=info.dtype, device=self.device)
        return feed

    @torch.no_grad()
    def prefill(self, data, save_golden: bool = False):
        del save_golden
        self._ensure_prefill_session()
        (
            inputs_embeds,
            deepstack_tensors,
            time_position_ids,
            height_position_ids,
            width_position_ids,
            prompt_len,
            target_seq_len,
        ) = self._prepare_prefill_tensors(data)

        self._prefill_cache_state = _alloc_cache_inputs(self.prefill_session, self.device)
        self._decode_cache_state = None

        prefill_chunk_len = int(self._prefill_inputs_info.shape[1])
        past_seq_len = 0
        last_prefill_logits = None
        for start in range(0, target_seq_len, prefill_chunk_len):
            end = start + prefill_chunk_len
            valid_len = min(prefill_chunk_len, max(0, prompt_len - start))
            if valid_len <= 0:
                break
            deepstack_chunks = tuple(tensor[:, start:end, :] for tensor in deepstack_tensors)
            prefill_feed = self._build_mm_prefill_feed(
                chunk_embeds=inputs_embeds[:, start:end, :],
                deepstack_chunks=deepstack_chunks,  # type: ignore[arg-type]
                time_position_ids=time_position_ids[start:end],
                height_position_ids=height_position_ids[start:end],
                width_position_ids=width_position_ids[start:end],
                valid_len=valid_len,
                past_seq_len=past_seq_len,
                cache_state=self._prefill_cache_state,
            )
            _, prefill_output_map = self._run_hmonnx(self.prefill_session, prefill_feed)
            prefill_logits = self._extract_logits(prefill_output_map)
            last_prefill_logits = _select_last_valid_logits(prefill_logits, valid_len)
            self._update_linear_cache(self._prefill_cache_state, prefill_output_map)
            past_seq_len += valid_len

        if last_prefill_logits is None:
            return torch.empty((1, 0, 0), device=self.device, dtype=self.dtype)
        return last_prefill_logits

    @torch.no_grad()
    def decode(self, data: Union[dict, tuple, list]):
        if not isinstance(data, dict):
            raise TypeError("decode expects a dict input")
        token_ids = data["input_ids"]
        if token_ids.dim() != 2 or token_ids.shape[0] != 1:
            raise ValueError(f"decode input_ids must be [1, seq], got {tuple(token_ids.shape)}")

        self._ensure_decode_session()
        if self._decode_cache_state is None:
            self.init_decode()
        assert self._decode_cache_state is not None

        past_seq_len = int(data["past_seq_length"])
        decode_feed = self._build_mm_decode_feed(
            token_ids=token_ids,
            past_seq_len=past_seq_len,
            cache_state=self._decode_cache_state,
        )
        _, decode_output_map = self._run_hmonnx(self.decode_session, decode_feed)
        decode_logits = self._extract_logits(decode_output_map)
        self._update_linear_cache(self._decode_cache_state, decode_output_map)
        return decode_logits

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
        presence_penalty: Optional[float] = None,
        stream_output: bool = False,
        streamer: Optional[TextStreamer] = None,
    ) -> str:
        return self.generate_multimodal(
            input_ids=input_ids,
            attention_mask=None,
            image_embeds=None,
            video_embeds=None,
            deepstack_image_embeds=None,
            deepstack_video_embeds=None,
            image_grid_thw=None,
            video_grid_thw=None,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            stream_output=stream_output,
            streamer=streamer,
        )

    @torch.no_grad()
    def generate_multimodal(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        image_embeds: Optional[torch.Tensor],
        image_grid_thw: Optional[torch.Tensor],
        tokenizer,
        max_new_tokens: int,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        repetition_penalty: float = 1.0,
        presence_penalty: Optional[float] = None,
        stream_output: bool = False,
        streamer: Optional[TextStreamer] = None,
        video_embeds: Optional[torch.Tensor] = None,
        video_grid_thw: Optional[torch.Tensor] = None,
        deepstack_image_embeds: Optional[Sequence[torch.Tensor]] = None,
        deepstack_video_embeds: Optional[Sequence[torch.Tensor]] = None,
    ) -> str:
        if max_new_tokens <= 0:
            return ""
        if input_ids.dim() != 2 or input_ids.shape[0] != 1:
            raise ValueError(f"input_ids must be [1, seq], got {tuple(input_ids.shape)}")
        if tokenizer is None:
            raise ValueError("tokenizer is required for multimodal generation")

        presence_penalty = self.presence_penalty if presence_penalty is None else float(presence_penalty)
        prefill_logits = self.prefill(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "image_embeds": image_embeds,
                "video_embeds": video_embeds,
                "deepstack_image_embeds": deepstack_image_embeds,
                "deepstack_video_embeds": deepstack_video_embeds,
                "image_grid_thw": image_grid_thw,
                "video_grid_thw": video_grid_thw,
            },
            save_golden=False,
        )
        if prefill_logits.numel() == 0:
            return ""

        history_token_ids = input_ids[0].tolist()
        first_step_logits = _apply_repetition_penalty(prefill_logits, history_token_ids, repetition_penalty)
        first_step_logits = _apply_presence_penalty(first_step_logits, history_token_ids, presence_penalty)
        next_token_id = _sample_next_token(
            first_step_logits,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )

        if self.resource_tight_mode:
            self.release_prefill_session()

        self.init_decode()
        stop_token_ids = _build_stop_token_ids(tokenizer, self.eos_token_id)
        generated_ids: List[int] = []

        owns_streamer = False
        if stream_output and streamer is None:
            streamer = TextStreamer(
                tokenizer,
                skip_prompt=False,
                skip_special_tokens=True,
            )
            owns_streamer = True

        token_val = int(next_token_id[0][0].item())
        if token_val in stop_token_ids:
            if streamer is not None and owns_streamer:
                streamer.end()
            elif streamer is not None:
                streamer.end()
            return ""
        generated_ids.append(token_val)
        history_token_ids.append(token_val)
        if streamer is not None:
            streamer.put(next_token_id.detach().cpu())

        current_token = next_token_id.to(self.device)
        past_seq_len = int(input_ids.shape[1])
        for _ in range(max_new_tokens - 1):
            decode_logits = self.decode(
                {
                    "input_ids": current_token,
                    "past_seq_length": past_seq_len,
                }
            )
            decode_logits = _select_last_valid_logits(decode_logits, int(current_token.shape[1]))
            decode_logits = _apply_repetition_penalty(decode_logits, history_token_ids, repetition_penalty)
            decode_logits = _apply_presence_penalty(decode_logits, history_token_ids, presence_penalty)
            next_token_id = _sample_next_token(
                decode_logits,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
            token_val = int(next_token_id[0][0].item())
            if token_val in stop_token_ids:
                break
            generated_ids.append(token_val)
            history_token_ids.append(token_val)
            if streamer is not None:
                streamer.put(next_token_id.detach().cpu())
            current_token = next_token_id.to(self.device)
            past_seq_len += int(current_token.shape[1])

        if streamer is not None:
            streamer.end()
        if self.resource_tight_mode:
            self.release_decode_session()
        return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    def create_template(self, prompt, media_path=None, media_type="image"):
        if media_path is None:
            messages = [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}],
                }
            ]
        else:
            media_key = "image" if media_type == "image" else "video"
            media_content = {
                "type": media_type,
                media_key: media_path,
            }
            if media_type == "video":
                media_content["nframes"] = self.max_size_t
                media_content["resized_height"] = self.image_size_h
                media_content["resized_width"] = self.image_size_w
            messages = [
                {
                    "role": "user",
                    "content": [media_content, {"type": "text", "text": prompt}],
                }
            ]
        return messages

    def preprocess(
        self,
        prompt,
        media_path,
        processor,
        cus_temp: bool = False,
        media_type: str = "image",
    ):
        from qwen_vl_utils import process_vision_info

        if not cus_temp:
            messages = self.create_template(prompt, media_path, media_type=media_type)
        else:
            messages = prompt

        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        if media_path is not None and media_type == "video":
            image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
            sampled_video = video_inputs[0]
            sampled_metadata = self._build_sampled_video_metadata(sampled_video, float(video_kwargs["fps"][0]))
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
                videos_kwargs={
                    "video_metadata": sampled_metadata,
                    "return_metadata": True,
                },
            )
            inputs["hm_pixel_values"] = [self._build_video_raw_clip(sampled_video)]
        elif media_path is not None:
            image_inputs, video_inputs = process_vision_info(messages, image_patch_size=self.patch_size)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
        else:
            inputs = processor(
                text=[text],
                images=None,
                videos=None,
                padding=True,
                return_tensors="pt",
            )
        return inputs

    def load_and_process_image(self, image_path):
        from PIL import Image, ImageOps

        target_w, target_h = self.image_size_w, self.image_size_h
        image = Image.open(image_path).convert("RGB")
        orig_w, orig_h = image.size
        if (orig_w, orig_h) != (target_w, target_h):
            scale = min(target_w / orig_w, target_h / orig_h)
            new_w = int(orig_w * scale)
            new_h = int(orig_h * scale)
            image = image.resize((new_w, new_h), Image.BICUBIC)
            pad_w = target_w - new_w
            pad_h = target_h - new_h
            image = ImageOps.expand(
                image,
                border=(0, 0, pad_w, pad_h),
                fill=(114, 114, 114),
            )
        return image

    def load_and_process_image_v2(self, image_path):
        from PIL import Image

        target_w, target_h = self.image_size_w, self.image_size_h
        image = Image.open(image_path).convert("RGB")
        orig_w, orig_h = image.size
        if (orig_w, orig_h) != (target_w, target_h):
            image = image.resize((target_w, target_h), Image.BICUBIC)
        return image

    @torch.no_grad()
    def chat(
        self,
        prompt,
        media_path,
        processor,
        logger=None,
        use_fast: bool = False,
        do_sample: bool = False,
        cus_temp: bool = False,
        media_type: str = "image",
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        repetition_penalty: float = 1.0,
        presence_penalty: Optional[float] = None,
        stream_output: bool = False,
    ):
        del use_fast

        if media_path is not None and media_type == "image":
            media_input = (
                self.load_and_process_image(media_path)
                if self.resize_v1
                else self.load_and_process_image_v2(media_path)
            )
        elif media_path is not None:
            media_input = media_path
        else:
            media_input = None

        inputs = self.preprocess(
            prompt,
            media_input,
            processor,
            cus_temp=cus_temp,
            media_type=media_type,
        )
        inputs = inputs.to(self.device)

        image_embeds = None
        video_embeds = None
        deepstack_image_embeds = None
        deepstack_video_embeds = None
        image_grid_thw = inputs.get("image_grid_thw")
        video_grid_thw = inputs.get("video_grid_thw")

        if media_input is not None:
            extracted_features = self.extract_all_image_features(inputs["hm_pixel_values"])
            self.release_image_feature()
            if not isinstance(extracted_features, tuple):
                extracted_features = (extracted_features,)
            if media_type == "video":
                video_embeds = extracted_features[0]
                if len(extracted_features) > 1:
                    deepstack_video_embeds = tuple(extracted_features[1:4])
            else:
                image_embeds = extracted_features[0]
                if len(extracted_features) > 1:
                    deepstack_image_embeds = tuple(extracted_features[1:4])

        output = self.generate_multimodal(
            input_ids=inputs["input_ids"].to(self.device),
            attention_mask=inputs.get("attention_mask"),
            image_embeds=image_embeds,
            video_embeds=video_embeds,
            deepstack_image_embeds=deepstack_image_embeds,
            deepstack_video_embeds=deepstack_video_embeds,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            tokenizer=processor.tokenizer,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            stream_output=stream_output,
        )
        if logger is not None:
            logger.info("Output: %s", output)
        return output
