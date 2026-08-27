# -*- coding: utf-8 -*-
# Copyright 2025 HOUMO AI. All rights reserved.
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
# File: _dflash_model_impl.py
# Description:
#   DFlash (cross-attention draft) model for Qwen3.5 speculative decoding.
#   Two modes: context (builds KV cache from target hidden) and decode (generates draft tokens).

"""
Qwen3.5 DFlash draft model implementation.

DFlash uses cross-attention between draft tokens and target model hidden states.
It operates in two modes:
- context: processes target hidden states to build KV cache
- decode: generates draft tokens using cross-attention to target KV cache
"""

import json
import math
from pathlib import Path
from typing import List, Mapping, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from xhquant import nn as xhnn
from xhquant.core import CacheTensor
from xhquant.nn import LLMCacheV2, MaskedAdd, RMSNorm


def _build_rope_cache(
    *,
    head_dim: int,
    rope_theta: float,
    max_pe_length: int,
) -> Tuple[Tensor, Tensor]:
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    positions = torch.arange(max_pe_length, dtype=torch.float32).reshape(-1, 1)
    freqs = positions * inv_freq.reshape(1, -1)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos().unsqueeze(0).unsqueeze(0), emb.sin().unsqueeze(0).unsqueeze(0)


def _ensure_cache_tensor(cache: Tensor) -> CacheTensor:
    if isinstance(cache, CacheTensor):
        return cache
    if isinstance(cache, torch.Tensor):
        return CacheTensor(cache)
    return cache


class DFlashCrossAttention(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        rms_norm_eps: float,
        input_sequence_length: int,
        max_pe_length: int,
        rope_theta: float,
        use_cache: bool,
        flash_attention: Mapping | None = None,
        query_dependent_mask: bool = False,
    ):
        super().__init__()
        self.num_heads = num_attention_heads
        self.num_kv_heads = num_key_value_heads
        self.head_dim = head_dim
        self.num_kv_groups = num_attention_heads // num_key_value_heads
        self.use_cache = use_cache
        self.query_dependent_mask = query_dependent_mask
        flash_attention = flash_attention or {}
        self.use_flash_attention = bool(flash_attention.get("enable", False))
        if self.use_flash_attention:
            flash_bits = {
                name: int(flash_attention.get(name, 8)) for name in ("q_bits", "k_bits", "v_bits", "s_bits", "p_bits")
            }
            invalid_bits = {name: value for name, value in flash_bits.items() if value not in (8, 16)}
            if invalid_bits:
                invalid = ", ".join(f"{name}={value}" for name, value in invalid_bits.items())
                raise ValueError(
                    f"DFlash flash_attention q_bits/k_bits/v_bits/s_bits/p_bits must be 8 or 16, got {invalid}"
                )
            # DFlash predicts its complete masked-token block in parallel.
            # Every query sees the same target-prefix + noise-token K/V range,
            # so this is deliberately non-causal. kv_valid_length below
            # excludes the unused full-capacity cache tail.
            self.flash_attn = xhnn.FlashAttention(
                num_heads=num_attention_heads,
                num_kv_heads=num_key_value_heads,
                scale=head_dim**-0.5,
                is_causal=False,
                **flash_bits,
            )

        self.q_proj = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_attention_heads * head_dim, hidden_size, bias=False)
        self.q_norm = RMSNorm(head_dim, rms_norm_eps)
        self.k_norm = RMSNorm(head_dim, rms_norm_eps)
        self.q_rope = xhnn.Rope()
        self.k_rope = xhnn.Rope()
        self.cos_slice = xhnn.DynamicSlice([input_sequence_length], [2], [1])
        self.sin_slice = xhnn.DynamicSlice([input_sequence_length], [2], [1])
        self.k_cache = LLMCacheV2(axis=2) if use_cache else None
        self.v_cache = LLMCacheV2(axis=2) if use_cache else None
        self.masked_add = MaskedAdd()

        cos_cached, sin_cached = _build_rope_cache(
            head_dim=head_dim,
            rope_theta=rope_theta,
            max_pe_length=max_pe_length,
        )
        self.register_buffer("cos_cached", cos_cached, persistent=False)
        self.register_buffer("sin_cached", sin_cached, persistent=False)
        self.register_buffer(
            "scale",
            torch.tensor(head_dim**-0.5, dtype=torch.float32),
            persistent=False,
        )

    def build_target_kv(
        self,
        target_hidden: Tensor,
        past_seq_length: Tensor,
        current_input_length: Tensor,
        past_key_cache: Tensor | None = None,
        past_value_cache: Tensor | None = None,
    ) -> Tuple[Tensor, Tensor]:
        bsz, seq_len, _ = target_hidden.shape
        key_states = self.k_norm(
            self.k_proj(target_hidden).view(bsz, seq_len, self.num_kv_heads, self.head_dim)
        ).transpose(1, 2)
        value_states = self.v_proj(target_hidden).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        cos = self.cos_slice(self.cos_cached, past_seq_length)
        sin = self.sin_slice(self.sin_cached, past_seq_length)
        key_states = self.k_rope(key_states, cos, sin)
        if not self.use_cache:
            return key_states, value_states
        return (
            self.k_cache(key_states, past_seq_length, current_input_length, past_key_cache),
            self.v_cache(value_states, past_seq_length, current_input_length, past_value_cache),
        )

    def forward_decode(
        self,
        hidden_states: Tensor,
        past_seq_length: Tensor,
        current_input_length: Tensor,
        target_key_cache: Tensor,
        target_value_cache: Tensor,
        attn_mask: Tensor | None,
        query_kv_range_abs: Tensor | None = None,
    ) -> Tensor:
        bsz, q_len, _ = hidden_states.shape
        query_states = self.q_norm(
            self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim)
        ).transpose(1, 2)
        noise_key_states = self.k_norm(
            self.k_proj(hidden_states).view(bsz, q_len, self.num_kv_heads, self.head_dim)
        ).transpose(1, 2)
        noise_value_states = (
            self.v_proj(hidden_states).view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        )
        q_cos = self.cos_slice(self.cos_cached, past_seq_length)
        q_sin = self.sin_slice(self.sin_cached, past_seq_length)
        query_states = self.q_rope(query_states, q_cos, q_sin)
        noise_key_states = self.k_rope(noise_key_states, q_cos, q_sin)
        target_key_cache = _ensure_cache_tensor(target_key_cache)
        target_value_cache = _ensure_cache_tensor(target_value_cache)

        combined_key_states = self.k_cache(
            noise_key_states,
            past_seq_length,
            current_input_length,
            target_key_cache,
        )
        combined_value_states = self.v_cache(
            noise_value_states,
            past_seq_length,
            current_input_length,
            target_value_cache,
        )
        if self.use_flash_attention:
            attn_output = self.flash_attn(
                query_states,
                combined_key_states,
                combined_value_states,
                past_seq_length=past_seq_length,
                current_input_length=current_input_length,
                kv_valid_length=past_seq_length + current_input_length,
                query_kv_range_abs=query_kv_range_abs,
            )
        else:
            if attn_mask is None:
                raise ValueError("Non-FlashAttention DFlash decode requires attn_mask.")
            query_states = query_states * self.scale.to(query_states.dtype)
            key_states = torch.repeat_interleave(combined_key_states.transpose(2, 3), self.num_kv_groups, dim=1)
            value_states = torch.repeat_interleave(combined_value_states, self.num_kv_groups, dim=1)
            attn_weights = torch.matmul(query_states, key_states)
            expanded_attn_mask = (
                attn_mask.unsqueeze(1)
                if self.query_dependent_mask
                else attn_mask.unsqueeze(1).unsqueeze(1)
            )
            attn_weights = self.masked_add(
                attn_weights,
                expanded_attn_mask,
            )
            attn_weights = F.softmax(
                attn_weights,
                dim=-1,
                dtype=torch.float32,
            ).to(query_states.dtype)
            attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, -1)
        return self.o_proj(attn_output)


def _validate_activation_residual_scale(value: float) -> float:
    """Validate the exact power-of-two scale used by the FP16 graph rewrite."""

    value = float(value)
    if (
        not math.isfinite(value)
        or value < 1.0
        or math.frexp(value)[0] != 0.5
    ):
        raise ValueError(
            "DFlash2 activation_residual_scale must be a finite power of two "
            f"greater than or equal to 1, got {value}"
        )
    return value


class DFlashMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        *,
        branch_output_scale: float = 1.0,
    ):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.branch_output_scale_inverse = 1.0 / _validate_activation_residual_scale(
            branch_output_scale
        )

    def forward(self, x: Tensor) -> Tensor:
        activation = F.silu(self.gate_proj(x)) * self.up_proj(x)
        # DFlash2's first MLP down projection can exceed FP16's finite range.
        # Scaling before the linear (and its following homogeneous convolution)
        # is algebraically equivalent to scaling the completed branch, but
        # prevents the intermediate accumulation from overflowing.
        if self.branch_output_scale_inverse != 1.0:
            activation = activation * self.branch_output_scale_inverse
        return self.down_proj(activation)


def _grouped_dynamic_convolve(
    hidden_states: Tensor,
    dynamic_kernel: Tensor,
    base_kernel: Tensor | nn.ParameterList,
    group_size: int,
    num_groups: int,
    kernel_size: int,
) -> Tensor:
    """Apply DFlash2's causal grouped depthwise convolution to a full block.

    Position zero is the anchor row that is already present in
    ``hidden_states``. Its missing predecessor is zero padded; position one
    therefore consumes the anchor through tap one. This is the exact reference
    boundary rule for ``[anchor, mask, ..., mask]`` and deliberately computes
    all rows, including the anchor row.
    """

    batch_size = hidden_states.shape[0]
    block_size = hidden_states.shape[1]
    hidden_size = num_groups * group_size
    grouped_hidden = hidden_states.reshape(
        batch_size,
        block_size,
        num_groups,
        group_size,
    )
    dynamic_kernel = dynamic_kernel.reshape(
        batch_size,
        block_size,
        kernel_size,
        num_groups,
        1,
    )
    output = torch.zeros_like(grouped_hidden)
    for tap in range(kernel_size):
        if tap == 0:
            shifted = grouped_hidden
        else:
            shifted = F.pad(
                grouped_hidden[:, :-tap],
                (0, 0, 0, 0, tap, 0),
            )
        base = base_kernel[tap]
        if isinstance(base_kernel, Tensor):
            base = base.reshape(
                1,
                1,
                num_groups,
                group_size,
            )
        output = output + (base + dynamic_kernel[:, :, tap]) * shifted
    return output.reshape(batch_size, block_size, hidden_size)


class GroupedDynamicCausalConv(nn.Module):
    """Two-sided DFlash2 dynamic convolution around one transformer branch."""

    def __init__(
        self,
        hidden_size: int,
        kernel_size: int,
        group_size: int,
    ) -> None:
        super().__init__()
        if hidden_size % group_size:
            raise ValueError(
                f"DFlash2 conv_group_size={group_size} must divide hidden_size={hidden_size}"
            )
        self.kernel_size = kernel_size
        self.group_size = group_size
        self.hidden_size = hidden_size
        self.num_groups = hidden_size // group_size
        self.base_kernel = nn.Parameter(
            torch.empty(2, kernel_size, hidden_size)
        )
        self.kernel_projection = nn.Linear(
            hidden_size,
            2 * kernel_size * self.num_groups,
            bias=False,
        )

    def prepare_for_export(self) -> None:
        """Store each fixed branch/tap kernel in its exported shape.

        Checkpoints use one ``[2, K, H]`` parameter.  Keeping that layout
        through checkpoint loading preserves the upstream state-dict ABI;
        splitting it once while building the export wrapper makes every
        fixed branch/tap a direct initializer and avoids exporting constant
        Gather/Reshape chains.
        """

        if isinstance(self.base_kernel, nn.ModuleList):
            return

        source_kernel = self.base_kernel
        prepared_kernel = nn.ModuleList(
            [
                nn.ParameterList(
                    [
                        nn.Parameter(
                            source_kernel[branch, tap]
                            .reshape(
                                1,
                                1,
                                self.num_groups,
                                self.group_size,
                            )
                            .detach()
                            .clone(),
                            requires_grad=source_kernel.requires_grad,
                        )
                        for tap in range(self.kernel_size)
                    ]
                )
                for branch in range(2)
            ]
        )
        del self.base_kernel
        self.base_kernel = prepared_kernel

    def prepare(self, hidden_states: Tensor) -> Tuple[Tensor, Tensor]:
        dynamic_kernel = self.kernel_projection(hidden_states).reshape(
            hidden_states.shape[0],
            hidden_states.shape[1],
            2,
            self.kernel_size,
            self.num_groups,
        )
        pre_branch = _grouped_dynamic_convolve(
            hidden_states,
            dynamic_kernel[..., 0, :, :],
            self.base_kernel[0],
            self.group_size,
            self.num_groups,
            self.kernel_size,
        )
        return pre_branch, dynamic_kernel[..., 1, :, :]

    def finish(self, hidden_states: Tensor, dynamic_kernel: Tensor) -> Tensor:
        return _grouped_dynamic_convolve(
            hidden_states,
            dynamic_kernel,
            self.base_kernel[1],
            self.group_size,
            self.num_groups,
            self.kernel_size,
        )


class DFlash2CandidateSelector(nn.Module):
    """Precompute Scheme-B selector scores for a fixed-length path walk."""

    def __init__(
        self,
        *,
        hidden_size: int,
        vocab_size: int,
        rank: int,
        top_k: int,
    ) -> None:
        super().__init__()
        self.top_k = top_k
        self.predecessor_codebook = nn.Embedding(vocab_size, rank)
        self.successor_codebook = nn.Embedding(vocab_size, rank)
        self.hidden_projection = nn.Linear(hidden_size, rank, bias=False)

    def forward(
        self,
        hidden_states: Tensor,
        logits: Tensor,
        anchor_token_id: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        unary_logits, candidate_ids = torch.topk(
            logits,
            self.top_k,
            dim=-1,
            # XH2A's HMONNX TopK requires sorted=1. Sorting only applies the
            # same per-position permutation to ids, unary scores, and both
            # selector axes; Scheme B's token path and proposal distribution
            # are therefore invariant to it.
            sorted=True,
        )
        projected_hidden = self.hidden_projection(hidden_states)
        predecessor = self.predecessor_codebook(candidate_ids)
        successor = self.successor_codebook(candidate_ids)

        anchor = self.predecessor_codebook(anchor_token_id)
        first_pair_score = torch.sum(
            (anchor * projected_hidden[:, 0]).unsqueeze(1) * successor[:, 0],
            dim=-1,
        )
        first_score = unary_logits[:, 0] + first_pair_score

        gated_predecessor = (
            predecessor[:, :-1]
            * projected_hidden[:, 1:].unsqueeze(2)
        )
        transition_score = torch.matmul(
            gated_predecessor,
            successor[:, 1:].transpose(-1, -2),
        )
        transition_score = transition_score + unary_logits[:, 1:].unsqueeze(2)
        return candidate_ids, first_score, transition_score


class DFlashDecoderLayer(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        intermediate_size: int,
        rms_norm_eps: float,
        input_sequence_length: int,
        max_pe_length: int,
        rope_theta: float,
        use_cache: bool,
        flash_attention: Mapping | None = None,
        conv_kernel_size: int | None = None,
        conv_group_size: int | None = None,
        input_residual_scale: float = 1.0,
        output_residual_scale: float = 1.0,
    ):
        super().__init__()
        input_residual_scale = _validate_activation_residual_scale(
            input_residual_scale
        )
        output_residual_scale = _validate_activation_residual_scale(
            output_residual_scale
        )
        # The tensor carried between layers represents h / residual_scale.
        # RMSNorm is scale invariant apart from its negligible epsilon term.
        self.attention_branch_scale = 1.0 / input_residual_scale
        self.residual_rescale = input_residual_scale / output_residual_scale
        self.self_attn = DFlashCrossAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            rms_norm_eps=rms_norm_eps,
            input_sequence_length=input_sequence_length,
            max_pe_length=max_pe_length,
            rope_theta=rope_theta,
            use_cache=use_cache,
            flash_attention=flash_attention,
            query_dependent_mask=conv_kernel_size is not None,
        )
        self.input_layernorm = RMSNorm(hidden_size, rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, rms_norm_eps)
        self.mlp = DFlashMLP(
            hidden_size,
            intermediate_size,
            branch_output_scale=output_residual_scale,
        )
        if conv_kernel_size is None:
            self.attention_conv = None
            self.mlp_conv = None
        else:
            if conv_group_size is None:
                raise ValueError("DFlash2 conv_group_size is required with conv_kernel_size")
            self.attention_conv = GroupedDynamicCausalConv(
                hidden_size,
                conv_kernel_size,
                conv_group_size,
            )
            self.mlp_conv = GroupedDynamicCausalConv(
                hidden_size,
                conv_kernel_size,
                conv_group_size,
            )

    def forward_decode(
        self,
        hidden_states: Tensor,
        past_seq_length: Tensor,
        current_input_length: Tensor,
        target_key_cache: Tensor,
        target_value_cache: Tensor,
        attn_mask: Tensor | None,
        query_kv_range_abs: Tensor | None = None,
    ) -> Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attention_dynamic_kernel = None
        if self.attention_conv is not None:
            hidden_states, attention_dynamic_kernel = self.attention_conv.prepare(
                hidden_states
            )
        hidden_states = self.self_attn.forward_decode(
            hidden_states,
            past_seq_length=past_seq_length,
            current_input_length=current_input_length,
            target_key_cache=target_key_cache,
            target_value_cache=target_value_cache,
            attn_mask=attn_mask,
            query_kv_range_abs=query_kv_range_abs,
        )
        if self.attention_conv is not None:
            hidden_states = self.attention_conv.finish(
                hidden_states,
                attention_dynamic_kernel,
            )
        if self.attention_branch_scale != 1.0:
            hidden_states = hidden_states * self.attention_branch_scale
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        mlp_dynamic_kernel = None
        if self.mlp_conv is not None:
            hidden_states, mlp_dynamic_kernel = self.mlp_conv.prepare(hidden_states)
        hidden_states = self.mlp(hidden_states)
        if self.mlp_conv is not None:
            hidden_states = self.mlp_conv.finish(
                hidden_states,
                mlp_dynamic_kernel,
            )
        if self.residual_rescale != 1.0:
            residual = residual * self.residual_rescale
        return residual + hidden_states


class DFlashModel(nn.Module):
    """DFlash draft model with context (KV build) and decode (token generation) modes."""

    def __init__(
        self,
        *,
        mode: str,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        intermediate_size: int,
        num_hidden_layers: int,
        rms_norm_eps: float,
        num_target_layers: int,
        target_layer_ids: List[int],
        vocab_size: int,
        input_sequence_length: int,
        max_pe_length: int,
        max_sequence_length: int,
        rope_theta: float = 10_000_000.0,
        flash_attention: Mapping | None = None,
        architecture: str = "DFlashDraftModel",
        block_size: int = 16,
        conv_kernel_size: int | None = None,
        conv_group_size: int | None = None,
        selector_rank: int | None = None,
        selector_top_k: int | None = None,
        sliding_window: int | None = None,
        input_embedding_scale: float = 1.0,
        output_multiplier: float = 1.0,
        final_logit_softcapping: float | None = None,
        activation_residual_scale: float | None = None,
    ):
        super().__init__()
        if mode not in {"context", "decode"}:
            raise ValueError(f"Unsupported DFlash mode: {mode}")
        self.mode = mode
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_target_layers = num_target_layers
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.vocab_size = vocab_size
        self.input_sequence_length = input_sequence_length
        self.max_sequence_length = max_sequence_length
        self.target_layer_ids = target_layer_ids
        self.architecture = architecture
        self.is_dflash2 = architecture == "DFlash2DraftModel"
        if activation_residual_scale is None:
            activation_residual_scale = 128.0 if self.is_dflash2 else 1.0
        self.activation_residual_scale = _validate_activation_residual_scale(
            activation_residual_scale
        )
        if not self.is_dflash2 and self.activation_residual_scale != 1.0:
            raise ValueError(
                "Legacy DFlash does not support activation_residual_scale != 1"
            )
        self.block_size = block_size
        self.sliding_window = sliding_window
        self.input_embedding_scale = input_embedding_scale
        self.output_multiplier = output_multiplier
        self.final_logit_softcapping = final_logit_softcapping

        if self.is_dflash2:
            if input_sequence_length != block_size and mode == "decode":
                raise ValueError(
                    "DFlash2 decode must retain the checkpoint's complete "
                    f"block: input_sequence_length={input_sequence_length}, block_size={block_size}"
                )
            missing_dflash2 = {
                "conv_kernel_size": conv_kernel_size,
                "conv_group_size": conv_group_size,
                "selector_rank": selector_rank,
                "selector_top_k": selector_top_k,
                "sliding_window": sliding_window,
            }
            missing_names = [
                name for name, value in missing_dflash2.items() if value is None
            ]
            if missing_names:
                raise ValueError(
                    "DFlash2 checkpoint is missing required fields: "
                    + ", ".join(missing_names)
                )

        self.fc = nn.Linear(len(target_layer_ids) * hidden_size, hidden_size, bias=False)
        self.hidden_norm = RMSNorm(hidden_size, rms_norm_eps)
        self.layers = nn.ModuleList(
            [
                DFlashDecoderLayer(
                    hidden_size=hidden_size,
                    num_attention_heads=num_attention_heads,
                    num_key_value_heads=num_key_value_heads,
                    head_dim=head_dim,
                    intermediate_size=intermediate_size,
                    rms_norm_eps=rms_norm_eps,
                    input_sequence_length=input_sequence_length,
                    max_pe_length=max_pe_length,
                    rope_theta=rope_theta,
                    # All three DFlash graphs bind the same persistent cache.
                    # Context/context_decode append verified target hidden
                    # states; draft_decode appends the transient anchor/mask
                    # query K/V after the valid target prefix.
                    use_cache=True,
                    flash_attention=flash_attention,
                    conv_kernel_size=(conv_kernel_size if self.is_dflash2 else None),
                    conv_group_size=(conv_group_size if self.is_dflash2 else None),
                    input_residual_scale=(
                        1.0
                        if not self.is_dflash2 or layer_idx == 0
                        else self.activation_residual_scale
                    ),
                    output_residual_scale=(
                        self.activation_residual_scale
                        if self.is_dflash2
                        else 1.0
                    ),
                )
                for layer_idx in range(num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(hidden_size, rms_norm_eps)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.candidate_selector = (
            DFlash2CandidateSelector(
                hidden_size=hidden_size,
                vocab_size=vocab_size,
                rank=int(selector_rank),
                top_k=int(selector_top_k),
            )
            if self.is_dflash2
            else None
        )

    def _split_cache_tensors(self, cache_tensors: Tuple[Tensor, ...]) -> Tuple[List[Tensor], List[Tensor]]:
        if len(cache_tensors) != self.num_hidden_layers * 2:
            raise ValueError(f"Expected {self.num_hidden_layers * 2} cache tensors, got {len(cache_tensors)}")
        key_caches = list(cache_tensors[: self.num_hidden_layers])
        value_caches = list(cache_tensors[self.num_hidden_layers :])
        return key_caches, value_caches

    def prepare_for_export(self) -> None:
        """Materialize DFlash2-only static layouts before graph capture."""

        if not self.is_dflash2:
            return
        for layer in self.layers:
            if layer.attention_conv is None or layer.mlp_conv is None:
                raise RuntimeError(
                    "DFlash2 export requires both grouped convolution branches"
                )
            layer.attention_conv.prepare_for_export()
            layer.mlp_conv.prepare_for_export()

    def forward_context(
        self,
        target_hidden: Tensor,
        past_seq_length: Tensor,
        current_input_length: Tensor,
        *cache_tensors: Tensor,
    ) -> Tuple[Tensor, ...]:
        target_proj = self.hidden_norm(self.fc(target_hidden))
        past_key_caches, past_value_caches = self._split_cache_tensors(cache_tensors)
        present_key_caches: List[Tensor] = []
        present_value_caches: List[Tensor] = []
        for idx, layer in enumerate(self.layers):
            present_k, present_v = layer.self_attn.build_target_kv(
                target_proj,
                past_seq_length=past_seq_length,
                current_input_length=current_input_length,
                past_key_cache=past_key_caches[idx],
                past_value_cache=past_value_caches[idx],
            )
            present_key_caches.append(present_k)
            present_value_caches.append(present_v)
        return tuple(present_key_caches + present_value_caches)

    def forward_decode(
        self,
        noise_embedding: Tensor,
        past_seq_length: Tensor,
        current_input_length: Tensor,
        attn_mask: Tensor | None,
        anchor_token_id: Tensor | None,
        query_kv_range_abs: Tensor | None,
        *cache_tensors: Tensor,
    ) -> Tensor | Tuple[Tensor, Tensor, Tensor]:
        past_key_caches, past_value_caches = self._split_cache_tensors(cache_tensors)
        # DFlash checkpoints reuse the target token embedding table. Some
        # checkpoints train the draft backbone with an explicit embedding
        # scale, so bake that scale into the exported graph rather than making
        # every runtime reproduce an otherwise-unadvertised preprocessing step.
        hidden_states = noise_embedding
        if self.input_embedding_scale != 1.0:
            hidden_states = hidden_states * self.input_embedding_scale
        for idx, layer in enumerate(self.layers):
            hidden_states = layer.forward_decode(
                hidden_states,
                past_seq_length=past_seq_length,
                current_input_length=current_input_length,
                target_key_cache=past_key_caches[idx],
                target_value_cache=past_value_caches[idx],
                attn_mask=attn_mask,
                query_kv_range_abs=query_kv_range_abs,
            )
        normalized_hidden = self.norm(hidden_states)
        if not self.is_dflash2:
            return self.lm_head(normalized_hidden)
        if anchor_token_id is None:
            raise ValueError("DFlash2 decode requires anchor_token_id")
        draft_hidden = normalized_hidden[:, 1:]
        logits = self.lm_head(draft_hidden)
        if self.output_multiplier != 1.0:
            logits = logits * self.output_multiplier
        if (
            self.final_logit_softcapping is not None
            and self.final_logit_softcapping > 0
        ):
            softcap = self.final_logit_softcapping
            logits = torch.tanh(logits / softcap) * softcap
        return self.candidate_selector(
            draft_hidden,
            logits,
            anchor_token_id,
        )

    def forward(
        self,
        input0: Tensor,
        input1: Tensor,
        input2: Tensor,
        input3: Tensor | None = None,
        input4: Tensor | None = None,
        input5: Tensor | None = None,
        input6: Tensor | None = None,
        input7: Tensor | None = None,
        input8: Tensor | None = None,
        input9: Tensor | None = None,
        input10: Tensor | None = None,
        input11: Tensor | None = None,
        input12: Tensor | None = None,
        input13: Tensor | None = None,
        input14: Tensor | None = None,
        input15: Tensor | None = None,
        input16: Tensor | None = None,
        input17: Tensor | None = None,
        input18: Tensor | None = None,
        input19: Tensor | None = None,
    ):
        if self.mode == "context":
            cache_inputs = (
                input3,
                input4,
                input5,
                input6,
                input7,
                input8,
                input9,
                input10,
                input11,
                input12,
                input13,
                input14,
                input15,
                input16,
                input17,
                input18,
                input19,
            )[: self.num_hidden_layers * 2]
            return self.forward_context(input0, input1, input2, *cache_inputs)
        if self.is_dflash2:
            raise RuntimeError(
                "DFlash2 decode must be invoked through its named export adapter"
            )
        if input3 is None:
            raise ValueError("DFlash decode requires attn_mask input.")
        cache_inputs = (
            input4,
            input5,
            input6,
            input7,
            input8,
            input9,
            input10,
            input11,
            input12,
            input13,
            input14,
            input15,
            input16,
            input17,
            input18,
            input19,
        )[: self.num_hidden_layers * 2]
        return self.forward_decode(
            input0,
            input1,
            input2,
            input3,
            None,
            None,
            *cache_inputs,
        )

    @staticmethod
    def from_pretrained(
        dflash_model_dir: str,
        target_model_dir: str,
        *,
        mode: str,
        dtype: torch.dtype = torch.float16,
        input_sequence_length: int,
        max_pe_length: int,
        max_sequence_length: int,
        flash_attention: Mapping | None = None,
        activation_residual_scale: float | None = None,
    ) -> "DFlashModel":
        from safetensors import safe_open

        with open(Path(dflash_model_dir) / "config.json", encoding="utf-8") as f:
            cfg = json.load(f)

        draft_cfg = cfg["dflash_config"]
        target_layer_ids = draft_cfg["target_layer_ids"]
        hidden_size = cfg["hidden_size"]
        head_dim = cfg.get("head_dim", hidden_size // cfg["num_attention_heads"])
        architectures = cfg.get("architectures") or ["DFlashDraftModel"]
        architecture = str(architectures[0])
        is_dflash2 = architecture == "DFlash2DraftModel"
        rope_parameters = cfg.get("rope_parameters") or {}
        model = DFlashModel(
            mode=mode,
            hidden_size=hidden_size,
            num_attention_heads=cfg["num_attention_heads"],
            num_key_value_heads=cfg["num_key_value_heads"],
            head_dim=head_dim,
            intermediate_size=cfg["intermediate_size"],
            num_hidden_layers=cfg["num_hidden_layers"],
            rms_norm_eps=cfg.get("rms_norm_eps", 1e-6),
            num_target_layers=len(target_layer_ids),
            target_layer_ids=target_layer_ids,
            vocab_size=cfg["vocab_size"],
            input_sequence_length=input_sequence_length,
            max_pe_length=max_pe_length,
            max_sequence_length=max_sequence_length,
            rope_theta=rope_parameters.get(
                "rope_theta",
                cfg.get("rope_theta", 10_000_000.0),
            ),
            flash_attention=flash_attention,
            architecture=architecture,
            block_size=int(draft_cfg.get("block_size", cfg.get("block_size", 16))),
            conv_kernel_size=(int(draft_cfg["conv_kernel_size"]) if is_dflash2 else None),
            conv_group_size=(int(draft_cfg["conv_group_size"]) if is_dflash2 else None),
            selector_rank=(int(draft_cfg["selector_rank"]) if is_dflash2 else None),
            selector_top_k=(int(draft_cfg["selector_top_k"]) if is_dflash2 else None),
            sliding_window=(int(cfg["sliding_window"]) if is_dflash2 else None),
            input_embedding_scale=float(
                draft_cfg.get(
                    "input_embedding_scale",
                    cfg.get("input_embedding_scale", 1.0),
                )
            ),
            output_multiplier=float(
                draft_cfg.get(
                    "output_multiplier",
                    cfg.get("output_multiplier", 1.0),
                )
            ),
            final_logit_softcapping=draft_cfg.get(
                "final_logit_softcapping",
                cfg.get("final_logit_softcapping"),
            ),
            activation_residual_scale=activation_residual_scale,
        )

        with safe_open(str(Path(dflash_model_dir) / "model.safetensors"), framework="pt") as f:
            dflash_sd = {k: f.get_tensor(k).to(dtype) for k in f.keys()}
        if is_dflash2:
            for name in ("predecessor_codebook", "successor_codebook"):
                checkpoint_name = f"candidate_selector.{name}"
                if checkpoint_name in dflash_sd:
                    dflash_sd[f"{checkpoint_name}.weight"] = dflash_sd.pop(
                        checkpoint_name
                    )
        missing, unexpected = model.load_state_dict(dflash_sd, strict=False)
        unexpected = [name for name in unexpected if not name.endswith(("cos_cached", "sin_cached"))]
        missing = [
            name for name in missing if not name.endswith(("cos_cached", "sin_cached")) and name != "lm_head.weight"
        ]
        if unexpected:
            raise RuntimeError(f"Unexpected DFlash keys: {unexpected}")
        if missing:
            raise RuntimeError(f"Missing DFlash keys: {missing}")

        lm_head_loaded = False
        for sf_path in sorted(Path(target_model_dir).glob("*.safetensors")):
            with safe_open(str(sf_path), framework="pt") as f:
                if "lm_head.weight" in f.keys():
                    model.lm_head.weight.data.copy_(f.get_tensor("lm_head.weight").to(dtype))
                    lm_head_loaded = True
                    break
        if not lm_head_loaded:
            for sf_path in sorted(Path(target_model_dir).glob("*.safetensors")):
                with safe_open(str(sf_path), framework="pt") as f:
                    for key in f.keys():
                        if "embed_tokens.weight" in key:
                            model.lm_head.weight.data.copy_(f.get_tensor(key).to(dtype))
                            lm_head_loaded = True
                            break
                if lm_head_loaded:
                    break
        if not lm_head_loaded:
            raise RuntimeError("Could not load lm_head.weight from target model")

        model.to(dtype)
        return model


def register_wrap_cls(hf_model):
    """No-op: DFlash model is built from xhquant ops directly, no HF wrapping needed."""
    pass
