# -*- coding: utf-8 -*-
# ruff: noqa: N801
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
# File: _llm_model_impl.py
# Description:
#   Qwen3.5 LLM model adapted for the xh2 model zoo (xh2modelzoo).
"""
Qwen3.5 LLM model implementation for xhquant framework.
Combines:
- Qwen3VL's M-RoPE position encoding (separate T/H/W cos/sin with interleaved masks)
- Qwen3Next's GatedDeltaNet linear attention (chunk/recurrent)
- Qwen3.5's gated full attention (q_proj outputs query + gate, attn * sigmoid(gate))
"""

import importlib
from typing import TYPE_CHECKING, List, Optional

import torch
from torch import Tensor

from xhquant.nn.modules import Cos, Sin
from xhquant.utils.registry import DynamicModule

from ...register import XHLLM_TRACEABLE_MODULES
from ._hybrid_gated_delta_net import (
    HybridDecoderLayerMixin,
    HybridGatedAttentionMixin,
    HybridGatedDeltaNetMixin,
    HybridRMSNormGatedMixin,
    HybridRMSNormMixin,
)
from ._hybrid_registry import ensure_hybrid_fused_rms_norm_registered
from ._hybrid_text_model import HybridTextModelMixin
from .modeling_qwen3_5 import (
    Qwen3_5Attention,
    Qwen3_5DecoderLayer,
    Qwen3_5ForCausalLM,
    Qwen3_5ForConditionalGeneration,
    Qwen3_5GatedDeltaNet,
    Qwen3_5Model,
    Qwen3_5RMSNorm,
    Qwen3_5RMSNormGated,
    Qwen3_5TextModel,
    Qwen3_5TextRotaryEmbedding,
)


_HF_QWEN35_MODELING = "transformers.models.qwen3_5.modeling_qwen3_5"


def _resolve_hf_qwen35_class(class_name: str):
    try:
        module = importlib.import_module(_HF_QWEN35_MODELING)
    except Exception:
        return None
    return getattr(module, class_name, None)


def _with_hf_alias(local_cls, registry_name: str):
    mapping = {local_cls: registry_name}
    hf_cls = _resolve_hf_qwen35_class(local_cls.__name__)
    if hf_cls is not None and hf_cls is not local_cls:
        mapping[hf_cls] = registry_name
    return mapping


def _get_safe_autocast_device_type(device: torch.device) -> str:
    device_type = device.type if isinstance(device.type, str) else "cpu"
    return device_type if device_type != "mps" else "cpu"


def _compute_qwen3_5_rotary_cache(
    inv_freq: Tensor,
    attention_scaling: float,
    max_seq_len: int,
) -> tuple[Tensor, Tensor]:
    inv_freq_expanded = inv_freq[None, None, :, None].float().expand(1, max_seq_len, -1, 1)
    positions = torch.arange(max_seq_len, device=inv_freq.device, dtype=torch.float32)[None, :, None, None]
    device_type = _get_safe_autocast_device_type(inv_freq.device)
    with torch.autocast(device_type=device_type, enabled=False):
        freqs = (inv_freq_expanded @ positions).transpose(2, 3)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * attention_scaling
        sin = emb.sin() * attention_scaling
    return cos[0].to(dtype=inv_freq.dtype), sin[0].to(dtype=inv_freq.dtype)


# noqa: E402
ensure_hybrid_fused_rms_norm_registered()
# ============================================================================
# RMSNorm wrappers
# ============================================================================
if TYPE_CHECKING:

    class _Qwen3_5TextRMSNormBase(DynamicModule, Qwen3_5RMSNorm):  # type: ignore[misc]
        ...
else:
    _Qwen3_5TextRMSNormBase = DynamicModule


@XHLLM_TRACEABLE_MODULES.register_module(_with_hf_alias(Qwen3_5RMSNorm, "Qwen3_5RMSNorm"))
class _Qwen3_5TextRMSNorm(HybridRMSNormMixin, _Qwen3_5TextRMSNormBase):  # noqa: N801
    """Registered adapter for the shared hybrid RMSNorm implementation."""

    pass


_rms_norm_gated_registry = _with_hf_alias(Qwen3_5RMSNormGated, "Qwen3_5RMSNormGated")
if TYPE_CHECKING:

    class _Qwen3_5RMSNormGatedBase(DynamicModule, Qwen3_5RMSNormGated):  # type: ignore[misc]
        ...
else:
    _Qwen3_5RMSNormGatedBase = DynamicModule


@XHLLM_TRACEABLE_MODULES.register_module(_rms_norm_gated_registry)
class _Qwen3_5RMSNormGated(HybridRMSNormGatedMixin, _Qwen3_5RMSNormGatedBase):  # noqa: N801
    """Registered adapter for the shared hybrid RMSNorm implementation."""

    pass


# ============================================================================
# Rotary Embedding
# ============================================================================
if TYPE_CHECKING:

    class _Qwen3_5TextRotaryEmbeddingBase(DynamicModule, Qwen3_5TextRotaryEmbedding):  # type: ignore[misc]
        ...
else:
    _Qwen3_5TextRotaryEmbeddingBase = DynamicModule


@XHLLM_TRACEABLE_MODULES.register_module(_with_hf_alias(Qwen3_5TextRotaryEmbedding, "Qwen3_5TextRotaryEmbedding"))
class _Qwen3_5TextRotaryEmbedding(_Qwen3_5TextRotaryEmbeddingBase):  # noqa: N801
    """Pre-compute cos/sin cache for M-RoPE.
    Unlike the HF model which applies interleaved mrope at forward time,
    here we pre-compute per-position cos/sin (without interleaving).
    The interleaving is handled by T/H/W masks in ``_Qwen3_5TextModel``.
    """

    def _setup(self, cfg):
        support_long_context = (
            cfg.get("support_long_context_over_fp16_limit", True)
            if hasattr(cfg, "get")
            else getattr(cfg, "support_long_context_over_fp16_limit", True)
        )
        self.support_long_context_over_fp16_limit = support_long_context
        if support_long_context:
            max_pe_length = cfg.max_pe_length
            self._setup_cos_sin_cache(seq_len=max_pe_length)
        if hasattr(self, "setup_after_callback"):
            self.setup_after_callback()

    def _setup_cos_sin_cache(self, seq_len):
        cos, sin = _compute_qwen3_5_rotary_cache(
            self.inv_freq,
            self.attention_scaling,
            seq_len,
        )
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

    def _compute_cos_sin(self, max_seq_len=2048):
        """Compute cos/sin embeddings per position without interleaving.
        Returns:
            cos: (1, max_seq_len, 1, rotary_dim)
            sin: (1, max_seq_len, 1, rotary_dim)
        """
        cos, sin = _compute_qwen3_5_rotary_cache(
            self.inv_freq,
            self.attention_scaling,
            max_seq_len,
        )
        return cos.unsqueeze(0), sin.unsqueeze(0)


# ============================================================================
# Full Attention (with gating + partial rotary)
# ============================================================================
if TYPE_CHECKING:

    class _Qwen3_5TextAttentionBase(DynamicModule, Qwen3_5Attention):  # type: ignore[misc]
        ...
else:
    _Qwen3_5TextAttentionBase = DynamicModule


@XHLLM_TRACEABLE_MODULES.register_module(_with_hf_alias(Qwen3_5Attention, "Qwen3_5Attention"))
class _Qwen3_5TextAttention(HybridGatedAttentionMixin, _Qwen3_5TextAttentionBase):  # noqa: N801
    """Registered adapter for the shared hybrid gated-attention path."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_seq_length: Optional[Tensor] = None,
        current_input_length: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        past_k_cache: Optional[Tensor] = None,
        past_v_cache: Optional[Tensor] = None,
    ) -> torch.Tensor:
        return super().forward(
            hidden_states=hidden_states,
            past_seq_length=past_seq_length,
            current_input_length=current_input_length,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            past_k_cache=past_k_cache,
            past_v_cache=past_v_cache,
        )


# ============================================================================
# GatedDeltaNet (linear attention)
# ============================================================================
if TYPE_CHECKING:

    class _Qwen3_5GatedDeltaNetBase(DynamicModule, Qwen3_5GatedDeltaNet):  # type: ignore[misc]
        ...
else:
    _Qwen3_5GatedDeltaNetBase = DynamicModule


@XHLLM_TRACEABLE_MODULES.register_module(_with_hf_alias(Qwen3_5GatedDeltaNet, "Qwen3_5GatedDeltaNet"))
class _Qwen3_5GatedDeltaNet(HybridGatedDeltaNetMixin, _Qwen3_5GatedDeltaNetBase):  # noqa: N801
    """Registered Qwen3.5 adapter for the shared hybrid GDN implementation."""

    pass


# ============================================================================
# Decoder Layer
# ============================================================================
if TYPE_CHECKING:

    class _Qwen3_5DecoderLayerBase(DynamicModule, Qwen3_5DecoderLayer):  # type: ignore[misc]
        ...
else:
    _Qwen3_5DecoderLayerBase = DynamicModule


@XHLLM_TRACEABLE_MODULES.register_module(_with_hf_alias(Qwen3_5DecoderLayer, "Qwen3_5DecoderLayer"))
class _Qwen3_5DecoderLayer(HybridDecoderLayerMixin, _Qwen3_5DecoderLayerBase):  # noqa: N801
    """Registered adapter for the shared hybrid decoder implementation."""

    pass


# ============================================================================
# TextModel (with M-RoPE masks)
# ============================================================================
if TYPE_CHECKING:

    class _Qwen3_5TextModelBase(DynamicModule, Qwen3_5TextModel):  # type: ignore[misc]
        ...
else:
    _Qwen3_5TextModelBase = DynamicModule


@XHLLM_TRACEABLE_MODULES.register_module(_with_hf_alias(Qwen3_5TextModel, "Qwen3_5TextModel"))
class _Qwen3_5TextModel(HybridTextModelMixin, _Qwen3_5TextModelBase):  # noqa: N801
    """Qwen3.5 dense M-RoPE adapter for the shared text/cache loop."""

    def _setup_position_embeddings(self, cfg):
        self.support_long_context_over_fp16_limit = cfg.get("support_long_context_over_fp16_limit", True)
        self.cos = Cos()
        self.sin = Sin()
        rope_parameters = self.config.rope_parameters
        if isinstance(rope_parameters, dict):
            mrope_section = rope_parameters.get("mrope_section", [11, 11, 10])
        else:
            mrope_section = getattr(rope_parameters, "mrope_section", [11, 11, 10])
        partial_rotary_factor = getattr(self.config, "partial_rotary_factor", 0.25)
        rotary_dim = int(self.config.head_dim * partial_rotary_factor)
        half_dim = rotary_dim // 2
        mask_device = self.rotary_emb.inv_freq.device
        h_ids = torch.arange(1, mrope_section[1] * 3, 3, device=mask_device)
        w_ids = torch.arange(2, mrope_section[2] * 3, 3, device=mask_device)
        time_mask = torch.ones(half_dim, device=mask_device)
        time_mask[h_ids] = 0
        time_mask[w_ids] = 0
        time_mask = torch.cat([time_mask, time_mask], 0).unsqueeze(0).unsqueeze(0)
        self.rotary_emb.register_buffer("time_mask", time_mask.half(), persistent=False)
        hight_mask = torch.zeros(half_dim, device=mask_device)
        hight_mask[h_ids] = 1
        hight_mask = torch.cat([hight_mask, hight_mask], 0).unsqueeze(0).unsqueeze(0)
        self.rotary_emb.register_buffer("hight_mask", hight_mask.half(), persistent=False)
        width_mask = torch.zeros(half_dim, device=mask_device)
        width_mask[w_ids] = 1
        width_mask = torch.cat([width_mask, width_mask], 0).unsqueeze(0).unsqueeze(0)
        self.rotary_emb.register_buffer("width_mask", width_mask.half(), persistent=False)
        if self.support_long_context_over_fp16_limit:
            if not hasattr(self.rotary_emb, "cos_cached"):
                self.rotary_emb.setup_after_callback = self._setup_cos_sin_embeding
            else:
                self._setup_cos_sin_embeding()

    def _build_position_embeddings(
        self,
        past_seq_length,
        time_position_ids,
        hight_position_ids,
        width_position_ids,
    ):
        del past_seq_length
        return self._build_qwen3_5_mrope_position_embeddings(
            time_position_ids=time_position_ids,
            hight_position_ids=hight_position_ids,
            width_position_ids=width_position_ids,
            inv_freq=self.rotary_emb.inv_freq,
            attention_scaling=self.rotary_emb.attention_scaling,
            time_mask=self.rotary_emb.time_mask,
            hight_mask=self.rotary_emb.hight_mask,
            width_mask=self.rotary_emb.width_mask,
            use_precomputed_cache=self.support_long_context_over_fp16_limit,
            cos_cached=getattr(self.rotary_emb, "cos_cached", None),
            sin_cached=getattr(self.rotary_emb, "sin_cached", None),
        )

    def _setup_cos_sin_embeding(self):
        # Keep rotary cache buffers on rotary_emb (same pattern as qwen3next).
        # Multi-GPU placement is controlled by device_map assignment in script.
        if hasattr(self.rotary_emb, "cos_cached"):
            _ = self.rotary_emb.cos_cached
        if hasattr(self.rotary_emb, "sin_cached"):
            _ = self.rotary_emb.sin_cached

    def _compute_qwen3_5_rotary_from_position_ids(
        self,
        position_ids: Tensor,
        inv_freq: Tensor,
        attention_scaling: float,
    ) -> tuple[Tensor, Tensor]:
        freqs = position_ids.reshape(-1, 1) * inv_freq.reshape(1, -1)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = self.cos(emb) * attention_scaling
        sin = self.sin(emb) * attention_scaling
        return cos.unsqueeze(1), sin.unsqueeze(1)

    def _build_qwen3_5_mrope_position_embeddings(
        self,
        *,
        time_position_ids: Tensor,
        hight_position_ids: Tensor,
        width_position_ids: Tensor,
        inv_freq: Tensor,
        attention_scaling: float,
        time_mask: Tensor,
        hight_mask: Tensor,
        width_mask: Tensor,
        use_precomputed_cache: bool,
        cos_cached: Optional[Tensor] = None,
        sin_cached: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        if use_precomputed_cache:
            if cos_cached is None or sin_cached is None:
                raise ValueError("cos_cached and sin_cached are required when use_precomputed_cache=True")
            time_cos = cos_cached[time_position_ids]
            time_sin = sin_cached[time_position_ids]
            hight_cos = cos_cached[hight_position_ids]
            hight_sin = sin_cached[hight_position_ids]
            width_cos = cos_cached[width_position_ids]
            width_sin = sin_cached[width_position_ids]
        else:
            time_cos, time_sin = self._compute_qwen3_5_rotary_from_position_ids(
                time_position_ids, inv_freq, attention_scaling
            )
            hight_cos, hight_sin = self._compute_qwen3_5_rotary_from_position_ids(
                hight_position_ids, inv_freq, attention_scaling
            )
            width_cos, width_sin = self._compute_qwen3_5_rotary_from_position_ids(
                width_position_ids, inv_freq, attention_scaling
            )
        time_cos = time_cos * time_mask
        time_sin = time_sin * time_mask
        hight_cos = hight_cos * hight_mask
        hight_sin = hight_sin * hight_mask
        width_cos = width_cos * width_mask
        width_sin = width_sin * width_mask
        combined_cos = time_cos + hight_cos + width_cos
        combined_sin = time_sin + hight_sin + width_sin
        combined_cos = combined_cos.squeeze(1).unsqueeze(0).unsqueeze(0)
        combined_sin = combined_sin.squeeze(1).unsqueeze(0).unsqueeze(0)
        return combined_cos, combined_sin


# ============================================================================
# Model wrapper (Qwen3_5Model has .visual + .language_model)
# ============================================================================
if TYPE_CHECKING:

    class _Qwen3_5ModelBase(DynamicModule, Qwen3_5Model):  # type: ignore[misc]
        ...
else:
    _Qwen3_5ModelBase = DynamicModule


@XHLLM_TRACEABLE_MODULES.register_module(_with_hf_alias(Qwen3_5Model, "Qwen3_5Model"))
class _Qwen3_5Model(_Qwen3_5ModelBase):  # noqa: N801
    """Wrapper for Qwen3_5Model. Deletes visual module for LLM-only export."""

    def _setup(self, cfg):
        self.cfg = cfg
        if hasattr(self, "visual"):
            del self.visual

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
        return self.language_model(
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


# ============================================================================
# ForConditionalGeneration (top-level wrapper)
# ============================================================================
if TYPE_CHECKING:

    class _Qwen3_5ForConditionalGenerationBase(DynamicModule, Qwen3_5ForConditionalGeneration):  # type: ignore[misc]
        ...
else:
    _Qwen3_5ForConditionalGenerationBase = DynamicModule


@XHLLM_TRACEABLE_MODULES.register_module(
    _with_hf_alias(Qwen3_5ForConditionalGeneration, "Qwen3_5ForConditionalGeneration")
)
class _Qwen3_5ForConditionalGeneration(_Qwen3_5ForConditionalGenerationBase):  # noqa: N801
    """Top-level wrapper. Calls language_model.forward() directly, adds lm_head."""

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
        result = self.model.language_model.forward(
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
        hidden_states = result[0]
        conv_cache_out_list = result[1]
        recurrent_state_out_list = result[2]
        logits = self.lm_head(hidden_states)
        if self._has_extra_hidden_output:
            return logits, conv_cache_out_list, recurrent_state_out_list, result[3]
        return logits, conv_cache_out_list, recurrent_state_out_list


if TYPE_CHECKING:

    class _Qwen3_5ForCausalLMBase(DynamicModule, Qwen3_5ForCausalLM):  # type: ignore[misc]
        ...
else:
    _Qwen3_5ForCausalLMBase = DynamicModule


@XHLLM_TRACEABLE_MODULES.register_module(_with_hf_alias(Qwen3_5ForCausalLM, "Qwen3_5ForCausalLM"))
class _Qwen3_5ForCausalLM(_Qwen3_5ForCausalLMBase):  # noqa: N801
    """Top-level wrapper for text-only Qwen3.5 models."""

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
        result = self.model.forward(
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
        hidden_states = result[0]
        conv_cache_out_list = result[1]
        recurrent_state_out_list = result[2]
        logits = self.lm_head(hidden_states)
        if self._has_extra_hidden_output:
            return logits, conv_cache_out_list, recurrent_state_out_list, result[3]
        return logits, conv_cache_out_list, recurrent_state_out_list


def register_wrap_modules():
    """Called to ensure all wrapper classes are registered."""
    pass
