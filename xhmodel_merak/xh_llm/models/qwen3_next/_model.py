# -*- coding: utf-8 -*-
# Copyright 2025 The Qwen Team, Alibaba Group and The HuggingFace Inc. team. All rights reserved.
# Copyright 2025 HOUMO AI. All rights reserved.
#
# Modifications:
# - Portions of this file have been modified by HOUMO AI.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# File: _model.py
# Description:
#   Qwen3-Next model implementation adapted for the xh2 model zoo (xh2modelzoo).

import types
from typing import Dict, List, Optional

import torch
from torch import Tensor
from transformers.models.qwen3_next.modeling_qwen3_next import (
    Qwen3NextAttention,
    Qwen3NextDecoderLayer,
    Qwen3NextForCausalLM,
    Qwen3NextGatedDeltaNet,
    Qwen3NextModel,
    Qwen3NextRMSNorm,
    Qwen3NextRMSNormGated,
    Qwen3NextRotaryEmbedding,
    Qwen3NextSparseMoeBlock,
)

from xhquant import nn as xhnn
from xhquant.utils.registry import DynamicModule

from ...register import XHLLM_TRACEABLE_MODULES
from ..qwen3_5._hybrid_gated_delta_net import (
    HybridDecoderLayerMixin,
    HybridGatedAttentionMixin,
    HybridGatedDeltaNetMixin,
    HybridRMSNormGatedMixin,
    HybridRMSNormMixin,
)
from ..qwen3_5._hybrid_moe import HybridSparseMoeMixin
from ..qwen3_5._hybrid_registry import ensure_hybrid_fused_rms_norm_registered
from ..qwen3_5._hybrid_text_model import HybridTextModelMixin


ensure_hybrid_fused_rms_norm_registered()


@XHLLM_TRACEABLE_MODULES.register_module(
    {
        Qwen3NextRotaryEmbedding: "Qwen3NextRotaryEmbedding",
    }
)
class _Qwen3NextRotaryEmbedding(DynamicModule):
    def _setup(self, cfg: Optional[Dict] = None):
        assert "dynamic" not in self.rope_type, f"{self.rope_type} is not supported in dynamic mode"
        self._setup_cos_sin_cache(seq_len=self.max_seq_len_cached, dtype=self.inv_freq.dtype)
        if hasattr(self, "setup_after_callback"):
            self.setup_after_callback()

    def _setup_cos_sin_cache(self, seq_len, dtype):
        position_ids = torch.arange(0, seq_len, dtype=torch.long, device=self.inv_freq.device).unsqueeze(0)
        inv_freq = self.inv_freq.to(torch.float32)
        device = self.inv_freq.device
        if torch.cuda.is_available() and inv_freq.device.type != "cuda":
            inv_freq = self.inv_freq.cuda()
        cos, sin = self.forward(inv_freq, position_ids.to(inv_freq.device))
        cos = cos.to(device).squeeze(0)
        sin = sin.to(device).squeeze(0)
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
        self.register_buffer("sin_cached", sin.to(dtype=dtype), persistent=False)
        self.register_buffer("cos_cached", cos.to(dtype=dtype), persistent=False)

    def _set_dtype(self, dtype: torch.dtype) -> None:
        self.inv_freq = self.inv_freq.to(dtype)
        self._setup_cos_sin_cache(seq_len=self.max_seq_len_cached, dtype=self.inv_freq.dtype)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


@XHLLM_TRACEABLE_MODULES.register_module(
    {
        Qwen3NextRMSNorm: "Qwen3NextRMSNorm",
    }
)
class _Qwen3NextRMSNorm(HybridRMSNormMixin, DynamicModule):  # noqa: N801
    """Registered adapter for the shared hybrid RMSNorm implementation."""

    pass


_rms_norm_gated_registry = {Qwen3NextRMSNormGated: "Qwen3NextRMSNormGated"}


@XHLLM_TRACEABLE_MODULES.register_module(_rms_norm_gated_registry)
class _Qwen3NextRMSNormGated(HybridRMSNormGatedMixin, DynamicModule):  # noqa: N801
    """Registered adapter for the shared hybrid RMSNorm implementation."""

    pass


@XHLLM_TRACEABLE_MODULES.register_module(
    {
        Qwen3NextAttention: "Qwen3NextAttention",
    }
)
class _Qwen3NextAttention(HybridGatedAttentionMixin, DynamicModule):  # noqa: N801
    """Registered adapter for the shared hybrid gated-attention path."""

    pass


@XHLLM_TRACEABLE_MODULES.register_module(
    {
        Qwen3NextGatedDeltaNet: "Qwen3NextGatedDeltaNet",
    }
)
class _Qwen3NextGatedDeltaNet(HybridGatedDeltaNetMixin, DynamicModule):
    """Qwen3-Next projection adapter for the canonical hybrid GDN path.

    The packed projection modules are deliberately kept intact so GPTQ
    QuantLinear boundaries and the checkpoint schema remain unchanged.
    """

    _uses_packed_input_projections = True

    def fix_query_key_value_ordering(self, mixed_qkvz, mixed_ba):
        bsz, seq_len, _ = mixed_qkvz.size()
        qkvz_last_dim = 2 * self.head_k_dim + 2 * self.head_v_dim * self.num_v_heads // self.num_k_heads
        ba_last_dim = 2 * self.num_v_heads // self.num_k_heads
        mixed_qkvz = mixed_qkvz.view(bsz, seq_len, self.num_k_heads, qkvz_last_dim)
        mixed_ba = mixed_ba.view(bsz, seq_len, self.num_k_heads, ba_last_dim)
        query, key, value, z = torch.split(
            mixed_qkvz,
            [
                self.head_k_dim,
                self.head_k_dim,
                self.num_v_heads // self.num_k_heads * self.head_v_dim,
                self.num_v_heads // self.num_k_heads * self.head_v_dim,
            ],
            dim=3,
        )
        b, a = torch.split(
            mixed_ba,
            [
                self.num_v_heads // self.num_k_heads,
                self.num_v_heads // self.num_k_heads,
            ],
            dim=3,
        )
        value = value.reshape(bsz, seq_len, -1, self.head_v_dim)
        z = z.reshape(bsz, seq_len, -1, self.head_v_dim)
        b = b.reshape(bsz, seq_len, self.num_v_heads)
        a = a.reshape(bsz, seq_len, self.num_v_heads)
        return query, key, value, z, b, a

    def _project_qkvzba(self, hidden_states):
        # Do not split the packed modules: they can be GPTQ QuantLinear.
        return self.fix_query_key_value_ordering(self.in_proj_qkvz(hidden_states), self.in_proj_ba(hidden_states))


@XHLLM_TRACEABLE_MODULES.register_module(
    {
        Qwen3NextSparseMoeBlock: "Qwen3NextSparseMoeBlock",
    }
)
class _Qwen3NextSparseMoeBlock(HybridSparseMoeMixin, DynamicModule):  # noqa: N801
    """Registered adapter for the shared sparse-MoE implementation."""

    pass


@XHLLM_TRACEABLE_MODULES.register_module(
    {
        Qwen3NextDecoderLayer: "Qwen3NextDecoderLayer",
    }
)
class _Qwen3NextDecoderLayer(HybridDecoderLayerMixin, DynamicModule):  # noqa: N801
    """Registered adapter for the shared hybrid decoder implementation."""

    pass


@XHLLM_TRACEABLE_MODULES.register_module(
    {
        Qwen3NextModel: "Qwen3NextModel",
    }
)
class _Qwen3NextModel(HybridTextModelMixin, DynamicModule):
    """Qwen3-Next 1D-RoPE adapter for the shared text/cache loop."""

    def _build_position_embeddings(
        self,
        past_seq_length,
        time_position_ids,
        hight_position_ids,
        width_position_ids,
    ):
        del time_position_ids, hight_position_ids, width_position_ids
        cos = self.cos_slice(self.rotary_emb.cos_cached, past_seq_length)
        sin = self.sin_slice(self.rotary_emb.sin_cached, past_seq_length)
        return cos, sin

    def _setup_position_embeddings(self, cfg):
        input_seq_len = cfg.input_sequence_length
        self.sin_slice = xhnn.DynamicSlice([input_seq_len], [2], [1])
        self.cos_slice = xhnn.DynamicSlice([input_seq_len], [2], [1])

        def _sin_cos_slice_update_cfg(slice_module, cfg_inner=None):
            slice_module.valid_length = [cfg_inner.input_sequence_length]

        update = _sin_cos_slice_update_cfg
        self.sin_slice._update_cfg = types.MethodType(update, self.sin_slice)
        self.cos_slice._update_cfg = types.MethodType(update, self.cos_slice)
        if not hasattr(self.rotary_emb, "cos_cached"):
            self.rotary_emb.setup_after_callback = self._setup_cos_sin_embeding
        else:
            self._setup_cos_sin_embeding()

    def _setup_cos_sin_embeding(self):
        _ = self.rotary_emb.cos_cached, self.rotary_emb.sin_cached


@XHLLM_TRACEABLE_MODULES.register_module(
    {
        Qwen3NextForCausalLM: "Qwen3NextForCausalLM",
    }
)
class _Qwen3NextForCausalLM(DynamicModule):
    def forward(
        self,
        inputs_embeds: Optional[Tensor] = None,
        past_seq_length: Optional[Tensor] = None,
        current_input_length: Optional[Tensor] = None,
        linear_attn_mask: Optional[Tensor] = None,
        past_key_cache: Optional[List[Tensor]] = None,
        past_value_cache: Optional[List[Tensor]] = None,
        past_conv_cache: Optional[List[Tensor]] = None,
        past_recurrent_state: Optional[List[Tensor]] = None,
    ):
        """
        Returns:
            logits: output logits
            conv_cache_out_list: list of updated conv caches (only for linear_attention layers)
            recurrent_state_out_list: list of updated recurrent states (only for linear_attention layers)
        """
        model_outputs = self.model(
            inputs_embeds=inputs_embeds,
            past_seq_length=past_seq_length,
            current_input_length=current_input_length,
            linear_attn_mask=linear_attn_mask,
            past_key_cache=past_key_cache,
            past_value_cache=past_value_cache,
            past_conv_cache=past_conv_cache,
            past_recurrent_state=past_recurrent_state,
        )
        hidden_states, conv_cache_out_list, recurrent_state_out_list = model_outputs[:3]
        logits = self.lm_head(hidden_states)
        if len(model_outputs) > 3:
            return logits, conv_cache_out_list, recurrent_state_out_list, model_outputs[3]
        return logits, conv_cache_out_list, recurrent_state_out_list

    def _setup(self, cfg: Optional[Dict] = None):
        return self


def register_wrap_modules():
    pass
