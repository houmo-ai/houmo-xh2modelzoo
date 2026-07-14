# Copyright 2025 HOUMO AI
#
# File: _model.py
# Description:
#   Qwen3.5-MoE model implementation for xh2modelzoo (adapted from xhquant_llm).
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

"""
Qwen3.5-MoE model implementation for xhquant framework.

Combines:
- Qwen3.5's M-RoPE position encoding (separate T/H/W cos/sin with interleaved masks)
- Qwen3.5's GatedDeltaNet linear attention (chunk/recurrent, separate projections)
- Qwen3.5's gated full attention (q_proj outputs query + gate, attn * sigmoid(gate))
- Qwen3.5-MoE's SparseMoeBlock (packed Experts + shared expert + shared_expert_gate)
- RMSNorm with (1 + weight) style
"""

from typing import List, Optional

import torch
from torch import Tensor
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeAttention,
    Qwen3_5MoeDecoderLayer,
    Qwen3_5MoeForCausalLM,
    Qwen3_5MoeForConditionalGeneration,
    Qwen3_5MoeGatedDeltaNet,
    Qwen3_5MoeRMSNorm,
    Qwen3_5MoeRMSNormGated,
    Qwen3_5MoeSparseMoeBlock,
    Qwen3_5MoeTextModel,
    Qwen3_5MoeTextRotaryEmbedding,
)

from xhquant.utils.registry import DynamicModule

from ...register import XHLLM_TRACEABLE_MODULES
from ..qwen3_5 import split_conv_cache_utils as _split_conv_cache_utils
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


_regroup_split_conv_cache_inputs = _split_conv_cache_utils._regroup_flat_split_conv_cache
_flatten_split_conv_cache_outputs = _split_conv_cache_utils._flatten_split_conv_cache_outputs

ensure_hybrid_fused_rms_norm_registered()


# ============================================================================
# RMSNorm wrappers
# ============================================================================


@XHLLM_TRACEABLE_MODULES.register_module({Qwen3_5MoeRMSNorm: "Qwen3_5MoeRMSNorm"})
class _Qwen3_5MoeRMSNorm(HybridRMSNormMixin, DynamicModule):  # noqa: N801
    """Registered adapter for the shared hybrid RMSNorm implementation."""

    pass


_rms_norm_gated_registry = {Qwen3_5MoeRMSNormGated: "Qwen3_5MoeRMSNormGated"}


@XHLLM_TRACEABLE_MODULES.register_module(_rms_norm_gated_registry)
class _Qwen3_5MoeRMSNormGated(HybridRMSNormGatedMixin, DynamicModule):  # noqa: N801
    """Registered adapter for the shared hybrid RMSNorm implementation."""

    pass


# ============================================================================
# Rotary Embedding
# ============================================================================


@XHLLM_TRACEABLE_MODULES.register_module({Qwen3_5MoeTextRotaryEmbedding: "Qwen3_5MoeTextRotaryEmbedding"})
class _Qwen3_5MoeTextRotaryEmbedding(DynamicModule):  # noqa: N801
    """Pre-compute cos/sin cache for M-RoPE.

    Unlike the HF model which applies interleaved mrope at forward time,
    here we pre-compute per-position cos/sin (without interleaving).
    The interleaving is handled by T/H/W masks in ``_Qwen3_5MoeTextModel``.
    """

    def _setup(self, cfg):
        support_long_context = (
            cfg.get("support_long_context_over_fp16_limit", True)
            if hasattr(cfg, "get")
            else getattr(cfg, "support_long_context_over_fp16_limit", True)
        )
        self.support_long_context_over_fp16_limit = support_long_context
        # Use max_pe_length if explicitly set, otherwise fall back to max_sequence_length.
        if hasattr(cfg, "max_pe_length") and cfg.max_pe_length:
            max_pe_length = cfg.max_pe_length
        elif hasattr(cfg, "max_sequence_length") and cfg.max_sequence_length:
            max_pe_length = cfg.max_sequence_length
        else:
            max_pe_length = 4096
        self._setup_cos_sin_cache(seq_len=max_pe_length)

        # ---- M-RoPE interleaved masks ----
        partial_rotary_factor = getattr(self.config, "partial_rotary_factor", 0.25)
        head_dim = self.config.head_dim
        rotary_dim = int(head_dim * partial_rotary_factor)
        half_dim = rotary_dim // 2

        rope_parameters = self.config.rope_parameters
        if isinstance(rope_parameters, dict):
            mrope_section = rope_parameters.get("mrope_section", [11, 11, 10])
        else:
            mrope_section = getattr(rope_parameters, "mrope_section", [11, 11, 10])

        h_ids = torch.arange(1, mrope_section[1] * 3, 3)
        w_ids = torch.arange(2, mrope_section[2] * 3, 3)

        time_mask = torch.ones(half_dim)
        time_mask[h_ids] = 0
        time_mask[w_ids] = 0
        time_mask = torch.cat([time_mask, time_mask], 0)
        time_mask.unsqueeze_(0).unsqueeze_(0)
        self.register_buffer("time_mask", time_mask.half(), persistent=False)

        hight_mask = torch.zeros(half_dim)
        hight_mask[h_ids] = 1
        hight_mask = torch.cat([hight_mask, hight_mask], 0)
        hight_mask.unsqueeze_(0).unsqueeze_(0)
        self.register_buffer("hight_mask", hight_mask.half(), persistent=False)

        width_mask = torch.zeros(half_dim)
        width_mask[w_ids] = 1
        width_mask = torch.cat([width_mask, width_mask], 0)
        width_mask.unsqueeze_(0).unsqueeze_(0)
        self.register_buffer("width_mask", width_mask.half(), persistent=False)

        if hasattr(self, "setup_after_callback"):
            self.setup_after_callback()

    def _setup_cos_sin_cache(self, seq_len):
        cos, sin = self._compute_cos_sin(max_seq_len=seq_len)
        self.register_buffer("cos_cached", cos[0], persistent=True)
        self.register_buffer("sin_cached", sin[0], persistent=True)

    def _compute_cos_sin(self, max_seq_len=2048):
        """Compute cos/sin embeddings per position without interleaving.

        Returns:
            cos: (1, max_seq_len, 1, rotary_dim)
            sin: (1, max_seq_len, 1, rotary_dim)
        """
        inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(1, max_seq_len, -1, 1)
        positions = torch.arange(max_seq_len, device=self.inv_freq.device).float()[None, :, None, None]

        device_type = (
            self.inv_freq.device.type
            if isinstance(self.inv_freq.device.type, str) and self.inv_freq.device.type != "mps"
            else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ positions).transpose(2, 3)
            # freqs shape: (1, max_seq_len, 1, inv_freq_len)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=self.inv_freq.dtype), sin.to(dtype=self.inv_freq.dtype)

    def forward(self, time_position_ids: Tensor, hight_position_ids: Tensor, width_position_ids: Tensor):
        cos = self.cos_cached
        sin = self.sin_cached

        time_cos = cos[time_position_ids] * self.time_mask
        time_sin = sin[time_position_ids] * self.time_mask
        hight_cos = cos[hight_position_ids] * self.hight_mask
        hight_sin = sin[hight_position_ids] * self.hight_mask
        width_cos = cos[width_position_ids] * self.width_mask
        width_sin = sin[width_position_ids] * self.width_mask

        combined_cos = time_cos + hight_cos + width_cos
        combined_sin = time_sin + hight_sin + width_sin

        rotary_dim = combined_cos.shape[-1]
        combined_cos = combined_cos.reshape(-1, rotary_dim).unsqueeze(0).unsqueeze(0)
        combined_sin = combined_sin.reshape(-1, rotary_dim).unsqueeze(0).unsqueeze(0)
        return combined_cos, combined_sin


# ============================================================================
# Full Attention (with gating + partial rotary)
# ============================================================================


@XHLLM_TRACEABLE_MODULES.register_module({Qwen3_5MoeAttention: "Qwen3_5MoeAttention"})
class _Qwen3_5MoeAttention(HybridGatedAttentionMixin, DynamicModule):  # noqa: N801
    """Registered adapter for the shared hybrid gated-attention path."""

    pass


# ============================================================================
# GatedDeltaNet (linear attention)
# ============================================================================


@XHLLM_TRACEABLE_MODULES.register_module({Qwen3_5MoeGatedDeltaNet: "Qwen3_5MoeGatedDeltaNet"})
class _Qwen3_5MoeGatedDeltaNet(HybridGatedDeltaNetMixin, DynamicModule):  # noqa: N801
    """Registered Qwen3.5-MoE adapter for the shared hybrid GDN path."""

    pass


# ============================================================================
# SparseMoeBlock (packed Experts + shared expert)
# ============================================================================


@XHLLM_TRACEABLE_MODULES.register_module({Qwen3_5MoeSparseMoeBlock: "Qwen3_5MoeSparseMoeBlock"})
class _Qwen3_5MoeSparseMoeBlock(HybridSparseMoeMixin, DynamicModule):  # noqa: N801
    """Registered adapter for the shared sparse-MoE implementation."""

    pass


# ============================================================================
# Decoder Layer
# ============================================================================


@XHLLM_TRACEABLE_MODULES.register_module({Qwen3_5MoeDecoderLayer: "Qwen3_5MoeDecoderLayer"})
class _Qwen3_5MoeDecoderLayer(HybridDecoderLayerMixin, DynamicModule):  # noqa: N801
    """Registered adapter for the shared hybrid decoder implementation."""

    pass


# ============================================================================
# TextModel (with M-RoPE masks)
# ============================================================================


@XHLLM_TRACEABLE_MODULES.register_module({Qwen3_5MoeTextModel: "Qwen3_5MoeTextModel"})
class _Qwen3_5MoeTextModel(HybridTextModelMixin, DynamicModule):  # noqa: N801
    """Qwen3.5-MoE M-RoPE adapter for the shared text/cache loop."""

    pass


# ============================================================================
# ForCausalLM (top-level wrapper)
# ============================================================================


@XHLLM_TRACEABLE_MODULES.register_module({Qwen3_5MoeForCausalLM: "Qwen3_5MoeForCausalLM"})
class _Qwen3_5MoeForCausalLM(DynamicModule):  # noqa: N801
    """Top-level wrapper for Qwen3.5-MoE CausalLM.

    Qwen3_5MoeForCausalLM.model is Qwen3_5MoeTextModel directly.
    """

    def _setup(self, cfg):
        self.cfg = cfg
        self._has_extra_hidden_output = cfg.get("output_hidden_state_indices") is not None or cfg.get(
            "output_post_norm_hidden", False
        )

    def forward(
        self,
        inputs_embeds: Optional[Tensor] = None,
        time_position_ids: Optional[Tensor] = None,
        hight_position_ids: Optional[Tensor] = None,
        width_position_ids: Optional[Tensor] = None,
        past_seq_length: Optional[Tensor] = None,
        current_input_length: Optional[Tensor] = None,
        linear_attn_mask: Optional[Tensor] = None,
        past_key_cache: Optional[List[Tensor]] = None,
        past_value_cache: Optional[List[Tensor]] = None,
        past_conv_cache: Optional[List[Tensor]] = None,
        past_recurrent_state: Optional[List[Tensor]] = None,
    ):
        if self._has_extra_hidden_output:
            hidden_states, conv_cache_out_list, recurrent_state_out_list, extra_hidden = self.model(
                inputs_embeds=inputs_embeds,
                time_position_ids=time_position_ids,
                hight_position_ids=hight_position_ids,
                width_position_ids=width_position_ids,
                past_seq_length=past_seq_length,
                current_input_length=current_input_length,
                linear_attn_mask=linear_attn_mask,
                past_key_cache=past_key_cache,
                past_value_cache=past_value_cache,
                past_conv_cache=past_conv_cache,
                past_recurrent_state=past_recurrent_state,
            )
            logits = self.lm_head(hidden_states)
            return logits, conv_cache_out_list, recurrent_state_out_list, extra_hidden
        else:
            hidden_states, conv_cache_out_list, recurrent_state_out_list = self.model(
                inputs_embeds=inputs_embeds,
                time_position_ids=time_position_ids,
                hight_position_ids=hight_position_ids,
                width_position_ids=width_position_ids,
                past_seq_length=past_seq_length,
                current_input_length=current_input_length,
                linear_attn_mask=linear_attn_mask,
                past_key_cache=past_key_cache,
                past_value_cache=past_value_cache,
                past_conv_cache=past_conv_cache,
                past_recurrent_state=past_recurrent_state,
            )
            logits = self.lm_head(hidden_states)
            return logits, conv_cache_out_list, recurrent_state_out_list


# ============================================================================
# ForConditionalGeneration (VL wrapper)
# ============================================================================


@XHLLM_TRACEABLE_MODULES.register_module({Qwen3_5MoeForConditionalGeneration: "Qwen3_5MoeForConditionalGeneration"})
class _Qwen3_5MoeForConditionalGeneration(DynamicModule):  # noqa: N801
    """Top-level wrapper for Qwen3.5-MoE VL (ConditionalGeneration)."""

    def _setup(self, cfg):
        self.cfg = cfg
        self._has_extra_hidden_output = cfg.get("output_hidden_state_indices") is not None or cfg.get(
            "output_post_norm_hidden", False
        )
        # if hasattr(self, "visual"):
        #     del self.visual

    def forward(
        self,
        inputs_embeds: Optional[Tensor] = None,
        time_position_ids: Optional[Tensor] = None,
        hight_position_ids: Optional[Tensor] = None,
        width_position_ids: Optional[Tensor] = None,
        past_seq_length: Optional[Tensor] = None,
        current_input_length: Optional[Tensor] = None,
        linear_attn_mask: Optional[Tensor] = None,
        past_key_cache: Optional[List[Tensor]] = None,
        past_value_cache: Optional[List[Tensor]] = None,
        past_conv_cache: Optional[List[Tensor]] = None,
        past_recurrent_state: Optional[List[Tensor]] = None,
    ):
        language_model = getattr(self.model, "language_model", self.model)
        if self._has_extra_hidden_output:
            hidden_states, conv_cache_out_list, recurrent_state_out_list, extra_hidden = language_model(
                inputs_embeds=inputs_embeds,
                time_position_ids=time_position_ids,
                hight_position_ids=hight_position_ids,
                width_position_ids=width_position_ids,
                past_seq_length=past_seq_length,
                current_input_length=current_input_length,
                linear_attn_mask=linear_attn_mask,
                past_key_cache=past_key_cache,
                past_value_cache=past_value_cache,
                past_conv_cache=past_conv_cache,
                past_recurrent_state=past_recurrent_state,
            )
            logits = self.lm_head(hidden_states)
            return logits, conv_cache_out_list, recurrent_state_out_list, extra_hidden
        else:
            hidden_states, conv_cache_out_list, recurrent_state_out_list = language_model(
                inputs_embeds=inputs_embeds,
                time_position_ids=time_position_ids,
                hight_position_ids=hight_position_ids,
                width_position_ids=width_position_ids,
                past_seq_length=past_seq_length,
                current_input_length=current_input_length,
                linear_attn_mask=linear_attn_mask,
                past_key_cache=past_key_cache,
                past_value_cache=past_value_cache,
                past_conv_cache=past_conv_cache,
                past_recurrent_state=past_recurrent_state,
            )
            logits = self.lm_head(hidden_states)
            return logits, conv_cache_out_list, recurrent_state_out_list


def register_wrap_modules():
    """Called to ensure all wrapper classes are registered."""
    pass
