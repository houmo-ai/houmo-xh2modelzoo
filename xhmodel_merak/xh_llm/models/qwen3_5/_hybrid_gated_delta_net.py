# -*- coding: utf-8 -*-
"""Registry-free canonical hybrid GatedDeltaNet implementation.
Both Qwen3.5-MoE and Qwen3-Next wrappers inherit this mixin.  Model-family
classes provide only the packed/separate input-projection hook; convolution,
cache, delta-rule, fused-op, and dynamic-config behavior lives here.
"""

import math
import sys
import warnings
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from xhquant import nn as xhnn
from xhquant.nn import FlashAttention, LLMCacheV2, MaskedSoftmax, MatMul, RMSNorm

from ._delta_rule import torch_chunk_gated_delta_rule, torch_recurrent_gated_delta_rule
from ._gdr_ops import GDRBlockTriInverse, GDRChunkScan, GDRRecurrentScan


class HybridRMSNormMixin:
    """Shared 0-centered Qwen RMSNorm adapter."""

    def forward(self, hidden_states):
        return self.norm(hidden_states)

    def _setup(self, cfg: Optional[Dict] = None):
        del cfg
        hidden_size = self.weight.shape[0]
        self.norm = RMSNorm(hidden_size, self.eps).to(dtype=self.weight.dtype, device=self.weight.device)
        with torch.no_grad():
            self.norm.weight.copy_(self.weight + 1.0)
        return self


class HybridRMSNormGatedMixin:
    """Shared gated RMSNorm adapter used by the hybrid GDN output."""

    def forward(self, hidden_states, gate=None):
        hidden_states = self.norm(hidden_states)
        return hidden_states * F.silu(gate)

    def _setup(self, cfg: Optional[Dict] = None):
        del cfg
        hidden_size = self.weight.shape[0]
        eps = getattr(self, "variance_epsilon", getattr(self, "eps", 1e-6))
        self.norm = RMSNorm(hidden_size, eps).to(dtype=self.weight.dtype, device=self.weight.device)
        with torch.no_grad():
            self.norm.weight.copy_(self.weight)
        return self


class HybridGatedAttentionMixin:
    def rotate_half(self, x: Tensor):
        x1 = self.rotate_half_slice_1(x)
        x2 = self.rotate_half_slice_2(x)
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(self, q: Tensor, k: Tensor, cos: Tensor, sin: Tensor, unsqueeze_dim: int = 1):
        rotary_dim = cos.shape[-1]
        q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
        k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
        if self.enable_rope:
            q_embed = self.rope(q_rot, cos, sin)
            k_embed = self.rope(k_rot, cos, sin)
        else:
            q_embed = (q_rot * cos) + (self.rotate_half(q_rot) * sin)
            k_embed = (k_rot * cos) + (self.rotate_half(k_rot) * sin)
        q_embed = torch.cat([q_embed, q_pass], dim=-1)
        k_embed = torch.cat([k_embed, k_pass], dim=-1)
        return q_embed, k_embed

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_seq_length: Optional[Tensor] = None,
        current_input_length: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        past_k_cache: Optional[Tensor] = None,
        past_v_cache: Optional[Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()
        q_proj = self.q_proj(hidden_states)
        q_proj = q_proj.view(bsz, q_len, self.num_heads, self.head_dim * 2)
        query_states, gate = torch.split(q_proj, self.head_dim, dim=-1)
        gate = gate.reshape(bsz, q_len, -1)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        query_states = self.q_norm(query_states.view(bsz, q_len, self.num_heads, self.head_dim)).transpose(1, 2)
        key_states = self.k_norm(key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        if position_embeddings is not None:
            cos, sin = position_embeddings
            query_states, key_states = self.apply_rotary_pos_emb(query_states, key_states, cos, sin, unsqueeze_dim=1)
        if self.use_cache:
            key_states = self.k_cache(key_states, past_seq_length, current_input_length, past_k_cache)
            value_states = self.v_cache(value_states, past_seq_length, current_input_length, past_v_cache)
        if self.use_flash_attention:
            attn_output = self.flash_attn(
                query_states,
                key_states,
                value_states,
                past_seq_length=past_seq_length,
                current_input_length=current_input_length,
            )
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(bsz, q_len, self.attn_hidden_dim)
        else:
            query_states = query_states * self.kv_scale
            key_states = key_states.transpose(2, 3)
            key_states = torch.repeat_interleave(key_states, self.num_key_value_groups, dim=1)
            attn_weights = torch.matmul(query_states, key_states)
            attn_weights = self.masked_softmax(attn_weights, past_seq_length)
            value_states = torch.repeat_interleave(value_states, self.num_key_value_groups, dim=1)
            attn_output = torch.matmul(attn_weights, value_states)
            attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, self.attn_hidden_dim)
        attn_output = attn_output * torch.sigmoid(gate)
        attn_output = self.o_proj(attn_output)
        return attn_output

    def _setup(self, cfg: Optional[Dict] = None):
        if not hasattr(self, "num_key_value_heads"):
            self.num_key_value_heads = self.config.num_key_value_heads
        if not hasattr(self, "num_heads"):
            self.num_heads = self.config.num_attention_heads
        self.attn_hidden_dim = self.head_dim * self.num_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        rotary_dim = int(self.config.partial_rotary_factor * self.head_dim)
        self.rotate_half_slice_1 = xhnn.Slice([0], [rotary_dim // 2], [3], [1])
        self.rotate_half_slice_2 = xhnn.Slice([rotary_dim // 2], [sys.maxsize], [3], [1])
        self.masked_softmax = MaskedSoftmax(dim=-1)
        self.enable_rope = cfg.get("enable_rope", True)
        if self.enable_rope:
            self.rope = xhnn.Rope()
        flash_attention_cfg = cfg.get("flash_attention", None)
        if flash_attention_cfg is None and cfg.get("bfp_flash_attention", None) is not None:
            warnings.warn(
                "Qwen3-Next `bfp_flash_attention` is deprecated and no longer enables "
                "FlashAttention. Use `flash_attention.enable=True` instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        self.use_flash_attention = False
        if flash_attention_cfg is not None:
            cfg_get = flash_attention_cfg.get if hasattr(flash_attention_cfg, "get") else None
            if cfg_get is not None:
                self.use_flash_attention = bool(cfg_get("enable", False))
                legacy_manbit = cfg_get("sefp_manbit", 8)
                legacy_manbit = legacy_manbit + 1 if legacy_manbit <= 7 else legacy_manbit
                self.flash_q_bits = cfg_get("q_bits", cfg_get("q_manbit", legacy_manbit))
                self.flash_k_bits = cfg_get("k_bits", cfg_get("k_manbit", 8))
                self.flash_v_bits = cfg_get("v_bits", cfg_get("v_manbit", 8))
                self.flash_s_bits = cfg_get("s_bits", cfg_get("p_manbit", self.flash_q_bits))
                self.flash_p_bits = cfg_get("p_bits", cfg_get("p_manbit", self.flash_s_bits))
                flash_bits = {
                    "q_bits": self.flash_q_bits,
                    "k_bits": self.flash_k_bits,
                    "v_bits": self.flash_v_bits,
                    "s_bits": self.flash_s_bits,
                    "p_bits": self.flash_p_bits,
                }
                invalid_bits = {name: value for name, value in flash_bits.items() if value not in (8, 16)}
                if invalid_bits:
                    invalid = ", ".join(f"{name}={value}" for name, value in invalid_bits.items())
                    raise ValueError(
                        f"flash_attention q_bits/k_bits/v_bits/s_bits/p_bits must be 8 or 16, got {invalid}"
                    )
        if self.use_flash_attention:
            self.flash_attn = FlashAttention(
                num_heads=self.num_heads,
                scale=1 / math.sqrt(self.head_dim),
                num_kv_heads=self.num_key_value_heads,
                is_causal=True,
                q_bits=self.flash_q_bits,
                k_bits=self.flash_k_bits,
                v_bits=self.flash_v_bits,
                s_bits=self.flash_s_bits,
                p_bits=self.flash_p_bits,
            )
        self.use_cache = cfg.use_cache
        if self.use_cache:
            cache_axis = cfg.kv_cache.cache_axis
            self.k_cache = LLMCacheV2(axis=cache_axis)
            self.v_cache = LLMCacheV2(axis=cache_axis)
        else:
            self.k_cache = None
            self.v_cache = None
        kv_scale = 1 / math.sqrt(self.head_dim)
        kv_scale_dtype = self.q_proj.weight.dtype
        self.register_buffer(
            "kv_scale",
            torch.tensor(kv_scale, dtype=kv_scale_dtype, device=self.q_proj.weight.device),
            persistent=False,
        )
        return self


class HybridDecoderLayerMixin:
    """Shared residual/cache dispatch for hybrid attention decoder layers."""

    def forward(
        self,
        hidden_states: Tensor,
        past_seq_length: Optional[Tensor] = None,
        current_input_length: Optional[Tensor] = None,
        position_embeddings: Optional[Tuple[Tensor, Tensor]] = None,
        linear_attn_mask: Optional[Tensor] = None,
        past_k_cache: Optional[Tensor] = None,
        past_v_cache: Optional[Tensor] = None,
        past_conv_cache=None,
        past_recurrent_state: Optional[Tensor] = None,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        if self.layer_type == "linear_attention":
            hidden_states, conv_cache_out, recurrent_state_out = self.linear_attn(
                hidden_states=hidden_states,
                conv_cache=past_conv_cache,
                recurrent_state=past_recurrent_state,
                linear_attn_mask=linear_attn_mask,
                current_input_length=current_input_length,
            )
        else:
            # for fx trace
            cos, sin = position_embeddings
            position_embeddings = (cos, sin)
            hidden_states = self.self_attn(
                hidden_states=hidden_states,
                past_seq_length=past_seq_length,
                current_input_length=current_input_length,
                attention_mask=None,
                position_embeddings=position_embeddings,
                past_k_cache=past_k_cache,
                past_v_cache=past_v_cache,
            )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]
        hidden_states = residual + hidden_states
        if self.layer_type == "linear_attention":
            return hidden_states, conv_cache_out, recurrent_state_out
        return hidden_states

    def _setup(self, cfg: Optional[Dict] = None):
        del cfg
        # Transformers 5.13 renamed this decoder discriminator from
        # ``layer_type`` to ``block_type`` in both Qwen3.5 and Qwen3-Next.
        # Keep one canonical attribute for the shared exported forward.
        layer_type = getattr(self, "layer_type", None)
        if layer_type is None:
            layer_type = getattr(self, "block_type", None)
        if layer_type is None:
            if hasattr(self, "linear_attn") and not hasattr(self, "self_attn"):
                layer_type = "linear_attention"
            elif hasattr(self, "self_attn") and not hasattr(self, "linear_attn"):
                layer_type = "full_attention"
            else:
                raise AttributeError(
                    "Cannot determine hybrid decoder layer type from 'layer_type', 'block_type', or token-mixer modules"
                )
        self.layer_type = layer_type
        return self


def _resolve_python_int_length(value) -> Optional[int]:
    """Best-effort convert ``current_input_length`` to python int.
    During TorchFX tracing this can be a Proxy and must not be materialized.
    """
    if value is None:
        return None
    proxy_type_name = value.__class__.__name__
    if proxy_type_name in {"Proxy", "Attribute"}:
        return None
    if isinstance(value, Tensor):
        if value.numel() == 0:
            return None
        flat = value.reshape(-1)
        elem = flat[0]
        if elem.__class__.__name__ in {"Proxy", "Attribute"}:
            return None
        try:
            return int(elem.item())
        except Exception:
            return None
    try:
        return int(value)
    except Exception:
        return None


def _normalize_linear_conv_cache_rank(conv_cache: Tensor) -> Tensor:
    if conv_cache.__class__.__name__ in {"Proxy", "Attribute"}:
        return conv_cache
    if conv_cache.dim() == 3:
        return conv_cache
    if conv_cache.dim() == 4:
        if conv_cache.shape[2] == 1:
            return conv_cache.squeeze(2)
        if conv_cache.shape[3] == 1:
            return conv_cache.squeeze(3)
    raise RuntimeError(f"Unsupported conv_cache shape for linear attention: {tuple(conv_cache.shape)}")


def _is_linear_recurrent_state_shape(
    tensor: Tensor,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
) -> bool:
    return (
        tensor.dim() == 4
        and tensor.shape[1] == num_v_heads
        and tensor.shape[2] == head_k_dim
        and tensor.shape[3] == head_v_dim
    )


def _align_linear_cache_args(
    conv_cache: Optional[Tensor],
    recurrent_state: Optional[Tensor],
    conv_dim: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
) -> tuple[Optional[Tensor], Optional[Tensor]]:
    if conv_cache is None or recurrent_state is None:
        return conv_cache, recurrent_state
    if conv_cache.__class__.__name__ in {"Proxy", "Attribute"}:
        return conv_cache, recurrent_state
    if recurrent_state.__class__.__name__ in {"Proxy", "Attribute"}:
        return conv_cache, recurrent_state
    if (
        _is_linear_recurrent_state_shape(conv_cache, num_v_heads, head_k_dim, head_v_dim)
        and recurrent_state.dim() == 3
        and recurrent_state.shape[1] == conv_dim
    ):
        return recurrent_state, conv_cache
    return conv_cache, recurrent_state


def _manual_depthwise_conv1d_tail(
    hidden_states_new: Tensor,
    weight: Tensor,
    bias: Optional[Tensor],
    output_length: int,
) -> Tensor:
    """Compute the kept tail of a depthwise Conv1d directly with slices.
    Equivalent to:
    ``F.conv1d(hidden_states_new, weight.unsqueeze(1), bias, groups=conv_dim)[:, :, -output_length:]``
    when ``weight`` is shaped ``[conv_dim, kernel_size]``.
    """
    if weight.dim() == 3:
        if weight.shape[1] != 1:
            raise ValueError(f"Expected depthwise weight with shape [C, 1, K], got {tuple(weight.shape)}")
        weight = weight.squeeze(1)
    elif weight.dim() != 2:
        raise ValueError(f"Expected weight rank 2 or 3, got {weight.dim()}")
    kernel_size = weight.shape[-1]
    output = hidden_states_new[:, :, 1 : 1 + output_length] * weight[:, 0].view(1, -1, 1)
    for kernel_idx in range(1, kernel_size):
        output = output + hidden_states_new[:, :, 1 + kernel_idx : 1 + kernel_idx + output_length] * weight[
            :, kernel_idx
        ].view(1, -1, 1)
    if bias is not None:
        output = output + bias.view(1, -1, 1)
    return output


def _split_linear_qkv_tensor(
    tensor: Tensor,
    key_dim: int,
    value_dim: int,
    dim: int,
) -> tuple[Tensor, Tensor, Tensor]:
    return torch.split(tensor, [key_dim, key_dim, value_dim], dim=dim)


class HybridGatedDeltaNetMixin:
    """Canonical convolution/cache/GDR implementation for hybrid Qwen MoE.
    Qwen3.5 owns the default separate projection implementation below.
    Qwen3-Next overrides :meth:`_project_qkvzba` because its checkpoint keeps
    Q/K/V/Z and B/A interleaved in two packed (potentially GPTQ QuantLinear)
    modules.  Nothing in this mixin splits those packed modules.
    """

    _uses_packed_input_projections = False

    def _project_qkvzba(self, hidden_states: Tensor):
        """Project Q/K/V/Z/B/A using Qwen3.5's separate checkpoint layout."""
        batch_size, seq_len, _ = hidden_states.shape
        if self.split_conv_cache:
            query = self.in_proj_q(hidden_states)
            key = self.in_proj_k(hidden_states)
            value = self.in_proj_v(hidden_states)
        else:
            mixed_qkv = self.in_proj_qkv(hidden_states)
            query, key, value = _split_linear_qkv_tensor(mixed_qkv, self.key_dim, self.value_dim, dim=-1)
        query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)
        z = self.in_proj_z(hidden_states).reshape(batch_size, seq_len, -1, self.head_v_dim)
        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)
        return query, key, value, z, b, a

    def _input_projection_dtype(self) -> torch.dtype:
        if self._uses_packed_input_projections:
            return self.in_proj_qkvz.weight.dtype
        if self.split_conv_cache:
            return self.in_proj_q.weight.dtype
        return self.in_proj_qkv.weight.dtype

    def forward(
        self,
        hidden_states: torch.Tensor,
        conv_cache: Optional[Tensor] = None,
        recurrent_state: Optional[Tensor] = None,
        linear_attn_mask: Optional[Tensor] = None,
        current_input_length: Optional[Tensor] = None,
    ):
        mask = linear_attn_mask.to(hidden_states.dtype)
        mask = mask.unsqueeze(-1)
        batch_size, seq_len, _ = hidden_states.shape
        query, key, value, z, b, a = self._project_qkvzba(hidden_states)
        use_recurrent = self.linear_attention_mode == "recurrent"
        if self.linear_attention_mode == "auto" and current_input_length is not None:
            resolved_len = _resolve_python_int_length(current_input_length)
            use_recurrent = (resolved_len == 1) if resolved_len is not None else False
        _verify_intermediates = getattr(self, "_verify_output_intermediates", False)
        resolved_seq_len = _resolve_python_int_length(seq_len)
        # Speculative/MTP verification can feed one hidden-state seed while
        # tracing a fixed N-token recurrent transition.  Qwen3.5-MoE already
        # relied on this contract; keeping it in the canonical path makes the
        # behavior available to Qwen3-Next without a family-specific fork.
        _expand_verify_tokens = (
            _verify_intermediates and self.input_sequence_length > 1 and use_recurrent and resolved_seq_len == 1
        )
        attn_seq_len = self.input_sequence_length if _expand_verify_tokens else seq_len
        if self.split_conv_cache:
            # === Split conv_cache path (3 separate q/k/v tensors) ===
            query_states = query.reshape(batch_size, seq_len, self.key_dim)
            key_states = key.reshape(batch_size, seq_len, self.key_dim)
            value_states = value.reshape(batch_size, seq_len, self.value_dim)
            query_states = query_states.transpose(1, 2)
            key_states = key_states.transpose(1, 2)
            value_states = value_states.transpose(1, 2)
            if isinstance(conv_cache, (list, tuple)):
                if len(conv_cache) != 3:
                    raise RuntimeError(f"Expected 3 conv caches for linear attention, got {len(conv_cache)}")
                conv_cache_q = _normalize_linear_conv_cache_rank(conv_cache[0])
                conv_cache_k = _normalize_linear_conv_cache_rank(conv_cache[1])
                conv_cache_v = _normalize_linear_conv_cache_rank(conv_cache[2])
            else:
                conv_cache, recurrent_state = _align_linear_cache_args(
                    conv_cache,
                    recurrent_state,
                    self.conv_dim,
                    self.num_v_heads,
                    self.head_k_dim,
                    self.head_v_dim,
                )
                assert conv_cache is not None, "conv_cache is required"
                conv_cache = _normalize_linear_conv_cache_rank(conv_cache)
                conv_cache_q, conv_cache_k, conv_cache_v = _split_linear_qkv_tensor(
                    conv_cache,
                    self.key_dim,
                    self.value_dim,
                    dim=1,
                )
            query_states_new = torch.cat([conv_cache_q, query_states], dim=-1).to(self.conv1d_q.weight.dtype)
            key_states_new = torch.cat([conv_cache_k, key_states], dim=-1).to(self.conv1d_k.weight.dtype)
            value_states_new = torch.cat([conv_cache_v, value_states], dim=-1).to(self.conv1d_v.weight.dtype)
            if _verify_intermediates and self.input_sequence_length > 1 and use_recurrent:
                _kernel = int(self.conv_kernel_size)
                conv_cache_out = (
                    tuple(query_states_new[..., 1 + t : 1 + t + _kernel] for t in range(self.input_sequence_length))
                    + tuple(key_states_new[..., 1 + t : 1 + t + _kernel] for t in range(self.input_sequence_length))
                    + tuple(value_states_new[..., 1 + t : 1 + t + _kernel] for t in range(self.input_sequence_length))
                )
            else:
                conv_cache_out = (
                    self.conv_cache_slice(query_states_new, current_input_length),
                    self.conv_cache_slice(key_states_new, current_input_length),
                    self.conv_cache_slice(value_states_new, current_input_length),
                )
            if self.use_manual_depthwise_conv1d:
                query_states = _manual_depthwise_conv1d_tail(
                    query_states_new,
                    self.conv1d_q_manual_weight,
                    getattr(self, "conv1d_q_manual_bias", None),
                    self.input_sequence_length,
                )
                key_states = _manual_depthwise_conv1d_tail(
                    key_states_new,
                    self.conv1d_k_manual_weight,
                    getattr(self, "conv1d_k_manual_bias", None),
                    self.input_sequence_length,
                )
                value_states = _manual_depthwise_conv1d_tail(
                    value_states_new,
                    self.conv1d_v_manual_weight,
                    getattr(self, "conv1d_v_manual_bias", None),
                    self.input_sequence_length,
                )
            else:
                _k = self.conv_kernel_size
                _l = self.input_sequence_length
                query_states = self.conv1d_q(query_states_new)[:, :, _k : _k + _l]
                key_states = self.conv1d_k(key_states_new)[:, :, _k : _k + _l]
                value_states = self.conv1d_v(value_states_new)[:, :, _k : _k + _l]
            query_states = F.silu(query_states).to(query_states.dtype)
            key_states = F.silu(key_states).to(key_states.dtype)
            value_states = F.silu(value_states).to(value_states.dtype)
            mask_qkv = linear_attn_mask.unsqueeze(1)
            query_states = query_states * mask_qkv
            key_states = key_states * mask_qkv
            value_states = value_states * mask_qkv
            query = query_states.transpose(1, 2).reshape(batch_size, attn_seq_len, -1, self.head_k_dim)
            key = key_states.transpose(1, 2).reshape(batch_size, attn_seq_len, -1, self.head_k_dim)
            value = value_states.transpose(1, 2).reshape(batch_size, attn_seq_len, -1, self.head_v_dim)
        else:
            # === Merged conv_cache path (single tensor, default) ===
            mixed_qkv = torch.cat(
                (
                    query.reshape(batch_size, seq_len, self.key_dim),
                    key.reshape(batch_size, seq_len, self.key_dim),
                    value.reshape(batch_size, seq_len, self.value_dim),
                ),
                dim=-1,
            ).transpose(1, 2)
            conv_cache, recurrent_state = _align_linear_cache_args(
                conv_cache,
                recurrent_state,
                self.conv_dim,
                self.num_v_heads,
                self.head_k_dim,
                self.head_v_dim,
            )
            assert conv_cache is not None, "conv_cache is required"
            conv_cache = _normalize_linear_conv_cache_rank(conv_cache)
            hidden_states_new = torch.cat([conv_cache, mixed_qkv], dim=-1).to(self.conv1d.weight.dtype)
            if _verify_intermediates and self.input_sequence_length > 1 and use_recurrent:
                _kernel = int(self.conv_kernel_size)
                conv_cache_out = tuple(
                    hidden_states_new[..., 1 + t : 1 + t + _kernel] for t in range(self.input_sequence_length)
                )
            else:
                conv_cache_out = self.conv_cache_slice(hidden_states_new, current_input_length)
            if self.use_manual_depthwise_conv1d:
                conv_out = _manual_depthwise_conv1d_tail(
                    hidden_states_new,
                    self.conv1d_manual_weight,
                    getattr(self, "conv1d_manual_bias", None),
                    self.input_sequence_length,
                )
            else:
                _k = self.conv_kernel_size
                _l = self.input_sequence_length
                conv_out = self.conv1d(hidden_states_new)[:, :, _k : _k + _l]
            mixed_qkv = F.silu(conv_out).to(mixed_qkv.dtype)
            mask_qkv = linear_attn_mask.unsqueeze(1)
            mixed_qkv = mixed_qkv * mask_qkv
            mixed_qkv = mixed_qkv.transpose(1, 2)
            query, key, value = torch.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
            query = query.reshape(batch_size, attn_seq_len, -1, self.head_k_dim)
            key = key.reshape(batch_size, attn_seq_len, -1, self.head_k_dim)
            value = value.reshape(batch_size, attn_seq_len, -1, self.head_v_dim)
        beta = b.sigmoid()
        g = self.A_log_exp * F.softplus(a + self.dt_bias)
        mask_qkv = linear_attn_mask.unsqueeze(-1).unsqueeze(-1)
        value = value * mask_qkv
        beta = beta * mask
        g = g * mask
        if _expand_verify_tokens:
            beta = beta.repeat_interleave(self.input_sequence_length, dim=1)
            g = g.repeat_interleave(self.input_sequence_length, dim=1)
            mask_qkv = mask_qkv.repeat_interleave(self.input_sequence_length, dim=1)
        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
        if use_recurrent:
            if _verify_intermediates and self.input_sequence_length > 1:
                _recurrent_snapshots = []
                _current_rs = recurrent_state
                _core_parts = []
                for _t in range(self.input_sequence_length):
                    _out_t, _current_rs = torch_recurrent_gated_delta_rule(
                        query[:, _t : _t + 1],
                        key[:, _t : _t + 1],
                        value[:, _t : _t + 1],
                        g=g[:, _t : _t + 1],
                        beta=beta[:, _t : _t + 1],
                        mask_qkv=mask_qkv[:, _t : _t + 1],
                        initial_state=_current_rs,
                        output_final_state=True,
                        use_qk_l2norm_in_kernel=True,
                        num_heads=self.chunk_num_heads,
                        k_head_dim=self.chunk_k_head_dim,
                        v_head_dim=self.chunk_v_head_dim,
                        batch_size=self.batch_size,
                        scale=self.chunk_scale,
                        sequence_length=1,
                        recurrent_scan_op=self.recurrent_scan_op,
                    )
                    _core_parts.append(_out_t)
                    _recurrent_snapshots.append(_current_rs)
                core_attn_out = torch.cat(_core_parts, dim=1)
                last_recurrent_state = _recurrent_snapshots[-1]
            else:
                core_attn_out, last_recurrent_state = torch_recurrent_gated_delta_rule(
                    query,
                    key,
                    value,
                    g=g,
                    beta=beta,
                    mask_qkv=mask_qkv,
                    initial_state=recurrent_state,
                    output_final_state=self.use_cache,
                    use_qk_l2norm_in_kernel=True,
                    num_heads=self.chunk_num_heads,
                    k_head_dim=self.chunk_k_head_dim,
                    v_head_dim=self.chunk_v_head_dim,
                    batch_size=self.batch_size,
                    scale=self.chunk_scale,
                    # Decode is normally one token, while MTP decode traces a
                    # fixed multi-token input.  The recurrent helper invokes
                    # GDRRecurrentScan once per token, preserving the same
                    # state transition contract in both cases.
                    sequence_length=self.input_sequence_length,
                    recurrent_scan_op=self.recurrent_scan_op,
                )
        else:
            core_attn_out, last_recurrent_state = torch_chunk_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                mask_qkv=mask_qkv,
                initial_state=recurrent_state,
                output_final_state=self.use_cache,
                use_qk_l2norm_in_kernel=True,
                chunk_size=self.linear_chunk_size,
                chunk_mask_incl=self.chunk_mask_incl,
                chunk_mask_strict=self.chunk_mask_strict,
                chunk_eye_matrix=self.chunk_eye_matrix,
                chunk_eye_8_batched=self.chunk_eye_8_batched,
                cumsum_matrix=self.chunk_cumsum_matrix,
                input_sequence_length=self.input_sequence_length,
                num_heads=self.chunk_num_heads,
                k_head_dim=self.chunk_k_head_dim,
                v_head_dim=self.chunk_v_head_dim,
                batch_size=self.batch_size,
                scale=self.chunk_scale,
                chunk_row_masks=self.chunk_row_masks,
                cumsum_matmul=self.cumsum_matmul,
                block_tri_inverse_op=self.block_tri_inverse_op,
                chunk_scan_op=self.chunk_scan_op,
            )
        # GDRChunkScan variants whose ``state_is_cache`` metadata is true
        # update the CacheTensor input in place.  Exporting the same state as
        # an additional graph output would duplicate the cache contract and
        # can make HMONNX allocate a second recurrent-state buffer.  Derive
        # suppression from the concrete op class rather than a model-family
        # switch; the explicit cfg flag remains available for legacy ops.
        chunk_state_is_cache = bool(getattr(getattr(self, "chunk_scan_op", None), "state_is_cache", False))
        suppress_recurrent_state_outputs = (
            getattr(self, "suppress_recurrent_state_outputs", False) or chunk_state_is_cache
        ) and not use_recurrent
        if suppress_recurrent_state_outputs:
            recurrent_state_out = None
        elif _verify_intermediates and self.input_sequence_length > 1 and use_recurrent:
            recurrent_state_out = tuple(_recurrent_snapshots)
        else:
            recurrent_state_out = last_recurrent_state if last_recurrent_state is not None else recurrent_state
        # b_sz, s, n, h = z.shape
        # core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        # z = z.reshape(-1, z.shape[-1])
        if _expand_verify_tokens:
            z = z.repeat_interleave(self.input_sequence_length, dim=1)
        core_attn_out = self.norm(core_attn_out, z)
        # core_attn_out = core_attn_out.reshape(b_sz, s, n, h)
        core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1)
        core_attn_out = core_attn_out.to(self.out_proj.weight.dtype)
        output = self.out_proj(core_attn_out)
        output = output * mask
        return output, conv_cache_out, recurrent_state_out

    def _setup(self, cfg: Optional[Dict] = None):
        self.use_cache = cfg.use_cache
        self.linear_attention_mode = cfg.get("linear_attention_mode", "auto")
        self.linear_chunk_size = cfg.get("linear_chunk_size", 64)
        self.return_cache = cfg.get("return_cache", False)
        self.input_sequence_length = cfg.get("input_sequence_length", 256)
        self.batch_size = cfg.get("batch_size", 1)
        self._verify_output_intermediates = cfg.get("verify_output_intermediates", False)
        self.split_conv_cache = cfg.get("split_conv_cache", True)
        self.suppress_recurrent_state_outputs = cfg.get("suppress_recurrent_state_outputs", False)
        self.fuse_gdr_ops = cfg.get("fuse_gdr_ops", False)
        self.fuse_gdr_block_recurrent_ops = cfg.get("fuse_gdr_block_recurrent_ops", False)
        # QTL-341: route depthwise conv1d tail through self.conv1d so hmonnx
        # export emits a clean Conv op. xhquant 2d86b60+ routes any-kernel
        # depthwise conv2d (groups==in==out) into the VP triton impl, so the
        # slice/mul/add manual unroll is no longer needed. Set True to fall
        # back to the legacy _manual_depthwise_conv1d_tail.
        self.use_manual_depthwise_conv1d = cfg.get("use_manual_depthwise_conv1d", False)
        # Convert nn.Parameter to buffer for FX graph compatibility
        if "dt_bias" in self._parameters:
            _dt_bias_data = self.dt_bias.data.clone()
            del self._parameters["dt_bias"]
            self.register_buffer("dt_bias", _dt_bias_data, persistent=False)
        if "A_log" in self._parameters:
            _a_log_data = self.A_log.data.clone()
            del self._parameters["A_log"]
            self.register_buffer("A_log", _a_log_data, persistent=False)
        # Split in_proj_qkv into separate q/k/v projections if requested
        if self.split_conv_cache and not self._uses_packed_input_projections and not hasattr(self, "in_proj_q"):
            proj_has_bias = self.in_proj_qkv.bias is not None
            proj_device = self.in_proj_qkv.weight.device
            proj_dtype = self.in_proj_qkv.weight.dtype
            q_weight, k_weight, v_weight = _split_linear_qkv_tensor(
                self.in_proj_qkv.weight.detach().clone(),
                self.key_dim,
                self.value_dim,
                dim=0,
            )
            self.in_proj_q = nn.Linear(
                self.hidden_size,
                self.key_dim,
                bias=proj_has_bias,
                device=proj_device,
                dtype=proj_dtype,
            )
            self.in_proj_k = nn.Linear(
                self.hidden_size,
                self.key_dim,
                bias=proj_has_bias,
                device=proj_device,
                dtype=proj_dtype,
            )
            self.in_proj_v = nn.Linear(
                self.hidden_size,
                self.value_dim,
                bias=proj_has_bias,
                device=proj_device,
                dtype=proj_dtype,
            )
            self.in_proj_q.weight.data.copy_(q_weight)
            self.in_proj_k.weight.data.copy_(k_weight)
            self.in_proj_v.weight.data.copy_(v_weight)
            if proj_has_bias:
                q_bias, k_bias, v_bias = _split_linear_qkv_tensor(
                    self.in_proj_qkv.bias.detach().clone(),
                    self.key_dim,
                    self.value_dim,
                    dim=0,
                )
                self.in_proj_q.bias.data.copy_(q_bias)
                self.in_proj_k.bias.data.copy_(k_bias)
                self.in_proj_v.bias.data.copy_(v_bias)
            del self.in_proj_qkv
        # Split conv1d into separate q/k/v conv layers if requested
        if self.split_conv_cache and not hasattr(self, "conv1d_q"):
            conv_has_bias = self.conv1d.bias is not None
            conv_device = self.conv1d.weight.device
            conv_dtype = self.conv1d.weight.dtype
            q_weight, k_weight, v_weight = _split_linear_qkv_tensor(
                self.conv1d.weight.detach().clone(),
                self.key_dim,
                self.value_dim,
                dim=0,
            )
            self.conv1d_q = nn.Conv1d(
                self.key_dim,
                self.key_dim,
                bias=conv_has_bias,
                kernel_size=self.conv_kernel_size,
                groups=self.key_dim,
                padding=self.conv_kernel_size - 1,
                device=conv_device,
                dtype=conv_dtype,
            )
            self.conv1d_k = nn.Conv1d(
                self.key_dim,
                self.key_dim,
                bias=conv_has_bias,
                kernel_size=self.conv_kernel_size,
                groups=self.key_dim,
                padding=self.conv_kernel_size - 1,
                device=conv_device,
                dtype=conv_dtype,
            )
            self.conv1d_v = nn.Conv1d(
                self.value_dim,
                self.value_dim,
                bias=conv_has_bias,
                kernel_size=self.conv_kernel_size,
                groups=self.value_dim,
                padding=self.conv_kernel_size - 1,
                device=conv_device,
                dtype=conv_dtype,
            )
            self.conv1d_q.weight.data.copy_(q_weight)
            self.conv1d_k.weight.data.copy_(k_weight)
            self.conv1d_v.weight.data.copy_(v_weight)
            if conv_has_bias:
                q_bias, k_bias, v_bias = _split_linear_qkv_tensor(
                    self.conv1d.bias.detach().clone(),
                    self.key_dim,
                    self.value_dim,
                    dim=0,
                )
                self.conv1d_q.bias.data.copy_(q_bias)
                self.conv1d_k.bias.data.copy_(k_bias)
                self.conv1d_v.bias.data.copy_(v_bias)
            del self.conv1d
        # Pre-compute head dimensions for TorchFX tracing compatibility
        self.chunk_num_heads = self.num_v_heads
        self.chunk_k_head_dim = self.head_k_dim
        self.chunk_v_head_dim = self.head_v_dim
        target_dtype = self._input_projection_dtype()
        self.register_buffer(
            "chunk_scale",
            torch.tensor(
                1.0 / (self.head_k_dim**0.5),
                dtype=target_dtype,
                device=self.A_log.device,
            ),
            persistent=False,
        )
        a_log_exp = (-self.A_log.exp()).to(device=self.A_log.device, dtype=self.dt_bias.dtype)
        self.register_buffer("A_log_exp", a_log_exp, persistent=False)
        if self.split_conv_cache:
            self.register_buffer(
                "conv1d_q_manual_weight",
                self.conv1d_q.weight.detach().clone().squeeze(1),
                persistent=False,
            )
            self.register_buffer(
                "conv1d_k_manual_weight",
                self.conv1d_k.weight.detach().clone().squeeze(1),
                persistent=False,
            )
            self.register_buffer(
                "conv1d_v_manual_weight",
                self.conv1d_v.weight.detach().clone().squeeze(1),
                persistent=False,
            )
            if self.conv1d_q.bias is not None:
                self.register_buffer(
                    "conv1d_q_manual_bias",
                    self.conv1d_q.bias.detach().clone(),
                    persistent=False,
                )
                self.register_buffer(
                    "conv1d_k_manual_bias",
                    self.conv1d_k.bias.detach().clone(),
                    persistent=False,
                )
                self.register_buffer(
                    "conv1d_v_manual_bias",
                    self.conv1d_v.bias.detach().clone(),
                    persistent=False,
                )
        else:
            self.register_buffer(
                "conv1d_manual_weight",
                self.conv1d.weight.detach().clone().squeeze(1),
                persistent=False,
            )
            if self.conv1d.bias is not None:
                self.register_buffer(
                    "conv1d_manual_bias",
                    self.conv1d.bias.detach().clone(),
                    persistent=False,
                )
        # Conv cache slice: extract last conv_kernel_size elements along time dim
        self.conv_cache_slice = xhnn.DynamicSlice([self.conv_kernel_size], [2], [1])
        # Pre-create triangular masks for TorchFX tracing
        chunk_size = self.linear_chunk_size
        idx = torch.arange(chunk_size, dtype=torch.long, device=self.A_log.device)
        mask_incl = (idx[:, None] >= idx[None, :]).to(target_dtype)
        mask_strict = (idx[:, None] > idx[None, :]).to(target_dtype)
        num_chunks = (self.input_sequence_length + chunk_size - 1) // chunk_size
        flat_batch_size = self.batch_size * self.num_v_heads * num_chunks
        eye_matrix = (
            torch.eye(chunk_size, dtype=target_dtype, device=self.A_log.device)
            .unsqueeze(0)
            .expand(flat_batch_size, chunk_size, chunk_size)
            .contiguous()
        )
        cumsum_matrix = torch.triu(torch.ones(chunk_size, chunk_size, dtype=target_dtype, device=self.A_log.device))
        self.register_buffer("chunk_mask_incl", mask_incl, persistent=False)
        self.register_buffer("chunk_mask_strict", mask_strict, persistent=False)
        self.register_buffer("chunk_eye_matrix", eye_matrix, persistent=False)
        self.register_buffer("chunk_cumsum_matrix", cumsum_matrix, persistent=False)
        self.cumsum_matmul = MatMul()
        # Pre-compute row masks for Method 4 forward substitution
        chunk_row_masks = (
            torch.eye(chunk_size, dtype=target_dtype, device=self.A_log.device)
            .unsqueeze(-1)
            .expand(chunk_size, chunk_size, chunk_size)
            .contiguous()
        )
        self.register_buffer("chunk_row_masks", chunk_row_masks, persistent=False)
        # Pre-computed 8x8 identity for block Neumann inverse (BHN*nb, 8, 8)
        block_size = 8
        nb = chunk_size // block_size
        eye_8_batched = (
            torch.eye(block_size, dtype=target_dtype, device=self.A_log.device)
            .unsqueeze(0)
            .expand(flat_batch_size * nb, block_size, block_size)
            .contiguous()
        )
        self.register_buffer("chunk_eye_8_batched", eye_8_batched, persistent=False)
        # GDR fused ops. ``fuse_gdr_ops`` only enables GDRChunkScan because
        # that op changes the prefill recurrent-state I/O contract.
        # ``fuse_gdr_block_recurrent_ops`` enables the contract-preserving
        # GDRBlockTriInverse and GDRRecurrentScan ops independently.
        if self.fuse_gdr_block_recurrent_ops:
            self.block_tri_inverse_op = GDRBlockTriInverse(chunk_size=chunk_size, block_size=block_size)
            self.recurrent_scan_op = GDRRecurrentScan(sequence_length=1, output_all_states=False)
        else:
            self.block_tri_inverse_op = None
            self.recurrent_scan_op = None
        if self.fuse_gdr_ops:
            self.chunk_scan_op = GDRChunkScan(
                num_chunks=num_chunks,
                num_heads=self.num_v_heads,
                k_head_dim=self.head_k_dim,
                v_head_dim=self.head_v_dim,
                chunk_size=chunk_size,
            )
        else:
            self.chunk_scan_op = None
        return self

    def _update_cfg(self, cfg: Optional[Dict] = None):
        if cfg is None:
            return
        # These options determine module topology, custom-op presence, or
        # precomputed mask shapes. DynamicModule's graph-specific update phase
        # may resize batch/sequence buffers, but cannot safely replace modules
        # after wrapping. Reject topology drift instead of silently exposing a
        # cfg value that the graph does not implement.
        immutable_fields = {
            "linear_chunk_size": self.linear_chunk_size,
            "split_conv_cache": self.split_conv_cache,
            "fuse_gdr_ops": self.fuse_gdr_ops,
            "fuse_gdr_block_recurrent_ops": self.fuse_gdr_block_recurrent_ops,
            "use_manual_depthwise_conv1d": self.use_manual_depthwise_conv1d,
        }
        for field, current_value in immutable_fields.items():
            requested_value = cfg.get(field, current_value)
            if requested_value != current_value:
                raise ValueError(
                    f"{field} is setup-time immutable for HybridGatedDeltaNet: "
                    f"wrapped value={current_value!r}, requested value={requested_value!r}. "
                    "Re-wrap the model with the requested structural config."
                )
        self.use_cache = cfg.use_cache
        self.linear_attention_mode = cfg.get("linear_attention_mode", self.linear_attention_mode)
        self.return_cache = cfg.get("return_cache", self.return_cache)
        self.input_sequence_length = cfg.get("input_sequence_length", self.input_sequence_length)
        self.batch_size = cfg.get("batch_size", self.batch_size)
        self._verify_output_intermediates = cfg.get("verify_output_intermediates", self._verify_output_intermediates)
        self.suppress_recurrent_state_outputs = cfg.get(
            "suppress_recurrent_state_outputs",
            getattr(self, "suppress_recurrent_state_outputs", False),
        )
        # Update eye_matrix for new batch/seq config
        chunk_size = self.linear_chunk_size
        num_chunks = (self.input_sequence_length + chunk_size - 1) // chunk_size
        flat_batch_size = self.batch_size * self.num_v_heads * num_chunks
        if hasattr(self, "chunk_scale"):
            target_dtype = self.chunk_scale.dtype
        else:
            target_dtype = self._input_projection_dtype()
        eye_matrix = (
            torch.eye(chunk_size, dtype=target_dtype, device=self.A_log.device)
            .unsqueeze(0)
            .expand(flat_batch_size, chunk_size, chunk_size)
            .contiguous()
        )
        self.register_buffer("chunk_eye_matrix", eye_matrix, persistent=False)
        # Update 8x8 identity for block Neumann inverse
        block_size = 8
        nb = chunk_size // block_size
        eye_8_batched = (
            torch.eye(block_size, dtype=target_dtype, device=self.A_log.device)
            .unsqueeze(0)
            .expand(flat_batch_size * nb, block_size, block_size)
            .contiguous()
        )
        self.register_buffer("chunk_eye_8_batched", eye_8_batched, persistent=False)
        # Update GDR fused op buffers
        if getattr(self, "chunk_scan_op", None) is not None:
            self.chunk_scan_op.num_chunks = num_chunks
        self.conv_cache_slice = xhnn.DynamicSlice([self.conv_kernel_size], [2], [1])
