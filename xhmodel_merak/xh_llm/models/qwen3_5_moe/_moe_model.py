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

import math
import types
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
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

from xhquant import nn as xhnn
from xhquant.nn import BfpFlashAttention, LLMCacheV2, MaskedSoftmax, MatMul, RMSNorm
from xhquant.nn.modules.moeblock import MoeBlock
from xhquant.utils.registry import DynamicModule

from ...register import XHLLM_TRACEABLE_MODULES

# Re-use the chunk/recurrent gated delta rule implementations from Qwen3Next
from ..qwen3_5._delta_rule import (
    torch_chunk_gated_delta_rule,
    torch_recurrent_gated_delta_rule,
)
from ..qwen3_5._gdr_ops import GDRBlockTriInverse, GDRChunkScan, GDRRecurrentScan
from ..qwen3_5.split_conv_cache_utils import (
    _flatten_merged_conv_cache_outputs,
    _flatten_split_conv_cache_outputs,
    _get_linear_layer_conv_cache,
    _is_nested_split_conv_cache,
    _layers_use_split_conv_cache,
    _looks_like_flat_split_conv_cache,
    _regroup_flat_split_conv_cache,
    _select_linear_attn_conv_cache,
)


# FusedRMSNormGated from fla package (used when fla is installed)
try:
    from fla.modules import FusedRMSNormGated
except ImportError:
    FusedRMSNormGated = None


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


def _split_linear_qkv_tensor(
    tensor: Tensor,
    key_dim: int,
    value_dim: int,
    dim: int,
) -> tuple[Tensor, Tensor, Tensor]:
    return torch.split(tensor, [key_dim, key_dim, value_dim], dim=dim)


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
    # expected_input_length = output_length + kernel_size
    # if hidden_states_new.shape[-1] < expected_input_length:
    #     raise ValueError(
    #         "hidden_states_new is shorter than the required tail-conv window: "
    #         f"need at least {expected_input_length}, got {hidden_states_new.shape[-1]}"
    #     )

    output = hidden_states_new[:, :, 1 : 1 + output_length] * weight[:, 0].view(1, -1, 1)
    for kernel_idx in range(1, kernel_size):
        output = output + hidden_states_new[:, :, 1 + kernel_idx : 1 + kernel_idx + output_length] * weight[
            :, kernel_idx
        ].view(1, -1, 1)
    if bias is not None:
        output = output + bias.view(1, -1, 1)
    return output


def _get_activation_name(act_fn) -> str:
    if hasattr(act_fn, "_get_name"):
        return act_fn._get_name().lower()
    return act_fn.__class__.__name__.lower()


def _move_parameter_to_meta(module: nn.Module, parameter_name: str) -> None:
    parameter = getattr(module, parameter_name)
    meta_parameter = nn.Parameter(
        torch.empty_like(parameter.data, device="meta"),
        requires_grad=parameter.requires_grad,
    )
    setattr(module, parameter_name, meta_parameter)


def _move_buffer_to_meta(module: nn.Module, buffer_name: str) -> None:
    buffer = getattr(module, buffer_name)
    module._buffers[buffer_name] = torch.empty_like(buffer, device="meta")


def _init_single_moe_linear_storage(moeblock: MoeBlock, experts, linear, linear_name: str, device: torch.device):
    weight = linear.weight

    setattr(
        moeblock,
        f"expert_{linear_name}_weight",
        nn.Parameter(
            torch.empty(
                len(experts),
                weight.shape[0],
                weight.shape[1],
                device=device,
                dtype=weight.dtype,
            )
        ),
    )

    quant_weight = getattr(linear, "quant_weight", None)
    if quant_weight is not None:
        setattr(
            moeblock,
            f"expert_{linear_name}_quant_weight",
            torch.empty(
                len(experts),
                quant_weight.shape[0],
                quant_weight.shape[1],
                device=device,
                dtype=quant_weight.dtype,
            ),
        )

    bias = linear.bias
    if bias is not None:
        setattr(
            moeblock,
            f"expert_{linear_name}_bias",
            nn.Parameter(
                torch.empty(
                    len(experts),
                    bias.shape[0],
                    device=device,
                    dtype=bias.dtype,
                )
            ),
        )
    else:
        setattr(moeblock, f"expert_{linear_name}_bias", None)


def _pack_defused_expert_linear_to_moeblock(moeblock: MoeBlock, experts, linear_name: str, device: torch.device):
    _init_single_moe_linear_storage(moeblock, experts, getattr(experts[0], linear_name), linear_name, device)

    with torch.no_grad():
        for expert_idx, expert in enumerate(experts):
            linear = getattr(expert, linear_name)
            weight_data = linear.weight.data
            if weight_data.device != device:
                weight_data = weight_data.to(device)
            getattr(moeblock, f"expert_{linear_name}_weight")[expert_idx].copy_(weight_data)

            quant_weight = getattr(linear, "quant_weight", None)
            if quant_weight is not None and hasattr(moeblock, f"expert_{linear_name}_quant_weight"):
                quant_weight_data = quant_weight.data
                if quant_weight_data.device != device:
                    quant_weight_data = quant_weight_data.to(device)
                getattr(moeblock, f"expert_{linear_name}_quant_weight")[expert_idx].copy_(quant_weight_data)

            bias = linear.bias
            if bias is not None and getattr(moeblock, f"expert_{linear_name}_bias") is not None:
                bias_data = bias.data
                if bias_data.device != device:
                    bias_data = bias_data.to(device)
                getattr(moeblock, f"expert_{linear_name}_bias")[expert_idx].copy_(bias_data)

            _move_parameter_to_meta(linear, "weight")
            if bias is not None:
                _move_parameter_to_meta(linear, "bias")
            if quant_weight is not None:
                _move_buffer_to_meta(linear, "quant_weight")


# ============================================================================
# RMSNorm wrappers
# ============================================================================


@XHLLM_TRACEABLE_MODULES.register_module({Qwen3_5MoeRMSNorm: "Qwen3_5MoeRMSNorm"})
class _Qwen3_5MoeRMSNorm(DynamicModule):  # noqa: N801
    """Wrap Qwen3_5MoeRMSNorm.

    Qwen3.5-MoE uses ``(1 + weight)`` style normalization (weight init = 0).
    Convert to standard RMSNorm by copying ``weight + 1``.
    """

    def forward(self, hidden_states):
        return self.norm(hidden_states)

    def _setup(self, cfg: Optional[Dict] = None):
        hidden_size = self.weight.shape[0]
        self.norm = RMSNorm(hidden_size, self.eps).to(self.weight.dtype).to(self.weight.device)
        with torch.no_grad():
            self.norm.weight.copy_(self.weight + 1.0)
        return self


_rms_norm_gated_registry = {Qwen3_5MoeRMSNormGated: "Qwen3_5MoeRMSNormGated"}
if FusedRMSNormGated is not None:
    _rms_norm_gated_registry[FusedRMSNormGated] = "FusedRMSNormGated_Qwen35Moe"


@XHLLM_TRACEABLE_MODULES.register_module(_rms_norm_gated_registry)
class _Qwen3_5MoeRMSNormGated(DynamicModule):  # noqa: N801
    """Wrap Qwen3_5MoeRMSNormGated / FusedRMSNormGated used inside GatedDeltaNet.

    Original forward: ``silu(gate) * x -> norm -> scale(weight+1)``.
    Wrapper: ``norm(x) * scale -> silu(gate)``  (decomposed for hardware).
    """

    def forward(self, hidden_states, gate=None):
        hidden_states = self.norm(hidden_states)
        return hidden_states * F.silu(gate)

    def _setup(self, cfg: Optional[Dict] = None):
        hidden_size = self.weight.shape[0]
        eps = getattr(self, "variance_epsilon", getattr(self, "eps", 1e-6))
        self.norm = RMSNorm(hidden_size, eps).to(self.weight.dtype).to(self.weight.device)
        with torch.no_grad():
            # NOTE: Qwen3_5RMSNormGated / FusedRMSNormGated use 1-centered
            # weights (init ones), NOT 0-centered like Qwen3_5RMSNorm.
            # Their forward is ``weight * norm(x)`` (no +1), so copy directly.
            self.norm.weight.copy_(self.weight)
        return self


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
            cfg.get("support_long_context_over_fp16_limit", False)
            if hasattr(cfg, "get")
            else getattr(cfg, "support_long_context_over_fp16_limit", False)
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
class _Qwen3_5MoeAttention(DynamicModule):  # noqa: N801
    """Qwen3.5-MoE full attention with gating mechanism.

    q_proj outputs ``query + gate`` (2x heads), attention result is
    multiplied by ``sigmoid(gate)``.  Uses partial rotary (only first
    ``rotary_dim`` of ``head_dim`` get RoPE).
    """

    def apply_rotary_pos_emb(
        self,
        q: Tensor,
        k: Tensor,
        cos: Tensor,
        sin: Tensor,
        unsqueeze_dim: int = 1,
    ):
        rotary_dim = cos.shape[-1]
        q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
        k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

        if self.enable_rope:
            q_rot = self.rope(q_rot, cos, sin)
            k_rot = self.rope(k_rot, cos, sin)
        else:
            q_rot = (q_rot * cos) + (self.rotate_half(q_rot) * sin)
            k_rot = (k_rot * cos) + (self.rotate_half(k_rot) * sin)

        q_embed = torch.cat([q_rot, q_pass], dim=-1)
        k_embed = torch.cat([k_rot, k_pass], dim=-1)
        return q_embed, k_embed

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_seq_length: Optional[Tensor] = None,
        current_input_length: Optional[Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        past_k_cache: Optional[Tensor] = None,
        past_v_cache: Optional[Tensor] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tensor], Optional[Tensor]]:
        bsz, q_len, _ = hidden_states.size()

        # Gated Q projection: q_proj outputs (num_heads, head_dim*2) per position.
        qg = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim * 2)
        query_states, gate = torch.split(qg, self.head_dim, dim=-1)
        gate = gate.reshape(bsz, q_len, -1)

        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = self.q_norm(query_states).transpose(1, 2)
        key_states = self.k_norm(key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is not None:
            cos, sin = position_embeddings
            query_states, key_states = self.apply_rotary_pos_emb(query_states, key_states, cos, sin, unsqueeze_dim=1)

        if self.use_cache:
            key_states = self.k_cache(key_states, past_seq_length, current_input_length, past_k_cache)
            value_states = self.v_cache(value_states, past_seq_length, current_input_length, past_v_cache)

        if self.use_bfp_flash_attention:
            attn_output = self.bfp_attn(query_states, key_states, value_states)
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
        return attn_output, None, None

    def _setup(self, cfg: Optional[Dict] = None):
        if not hasattr(self, "num_key_value_heads"):
            self.num_key_value_heads = self.config.num_key_value_heads
        if not hasattr(self, "num_heads"):
            self.num_heads = self.config.num_attention_heads

        self.attn_hidden_dim = self.head_dim * self.num_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        self.masked_softmax = MaskedSoftmax(dim=-1)
        self.enable_rope = cfg.get("enable_rope", True)
        if self.enable_rope:
            self.rope = xhnn.Rope()

        bfp_flash_attention_cfg = cfg.get("bfp_flash_attention", None)
        self.use_bfp_flash_attention = False
        if bfp_flash_attention_cfg is not None:
            self.use_bfp_flash_attention = bfp_flash_attention_cfg.enable
            self.sefp_manbit = bfp_flash_attention_cfg.sefp_manbit
            self.out_fp_manbit = bfp_flash_attention_cfg.out_fp_manbit
            self.out_fp_expbit = bfp_flash_attention_cfg.out_fp_expbit
        if self.use_bfp_flash_attention:
            self.bfp_attn = BfpFlashAttention(
                self.attn_hidden_dim,
                self.num_heads,
                True,
                self.sefp_manbit,
                self.out_fp_expbit,
                self.out_fp_manbit,
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
            persistent=True,
        )
        return self


# ============================================================================
# GatedDeltaNet (linear attention)
# ============================================================================


@XHLLM_TRACEABLE_MODULES.register_module({Qwen3_5MoeGatedDeltaNet: "Qwen3_5MoeGatedDeltaNet"})
class _Qwen3_5MoeGatedDeltaNet(DynamicModule):  # noqa: N801
    """Qwen3.5-MoE GatedDeltaNet linear attention wrapper.

    Uses **separate** projections (in_proj_qkv, in_proj_z, in_proj_b, in_proj_a).
    """

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

        z = self.in_proj_z(hidden_states)  # [bs, seq, value_dim]
        z = z.reshape(batch_size, seq_len, -1, self.head_v_dim)
        b = self.in_proj_b(hidden_states)  # [bs, seq, num_v_heads]
        a = self.in_proj_a(hidden_states)  # [bs, seq, num_v_heads]

        use_recurrent = self.linear_attention_mode == "recurrent"
        _verify_intermediates = getattr(self, "_verify_output_intermediates", False)
        if self.linear_attention_mode == "auto" and current_input_length is not None:
            resolved_len = _resolve_python_int_length(current_input_length)
            # FX trace path: unresolved symbolic length falls back to chunk mode.
            use_recurrent = (resolved_len == 1) if resolved_len is not None else False

        resolved_seq_len = _resolve_python_int_length(seq_len)
        _expand_verify_tokens = (
            _verify_intermediates and self.input_sequence_length > 1 and use_recurrent and resolved_seq_len == 1
        )
        attn_seq_len = self.input_sequence_length if _expand_verify_tokens else seq_len

        if getattr(self, "split_conv_cache", False) or hasattr(self, "in_proj_q"):
            query_states = self.in_proj_q(hidden_states)
            key_states = self.in_proj_k(hidden_states)
            value_states = self.in_proj_v(hidden_states)

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
            mixed_qkv = self.in_proj_qkv(hidden_states)  # [bs, seq, key_dim*2+value_dim]
            mixed_qkv = mixed_qkv.transpose(1, 2)  # [bs, conv_dim, seq]

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
            if self.input_sequence_length > 1:
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
                    sequence_length=1,
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

        if _verify_intermediates and self.input_sequence_length > 1 and use_recurrent:
            recurrent_state_out = tuple(_recurrent_snapshots)
        else:
            recurrent_state_out = last_recurrent_state if last_recurrent_state is not None else recurrent_state

        if _expand_verify_tokens:
            z = z.repeat_interleave(self.input_sequence_length, dim=1)

        b_sz, s, n, h = z.shape
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(b_sz, s, n, h)
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
        self._verify_output_intermediates = cfg.get("verify_output_intermediates", False)
        self.input_sequence_length = cfg.get("input_sequence_length", 256)
        self.batch_size = cfg.get("batch_size", 1)
        self.split_conv_cache = cfg.get("split_conv_cache", True) or hasattr(self, "in_proj_q")
        self.fuse_gdr_ops = cfg.get("fuse_gdr_ops", False)
        # QTL-341: route depthwise conv1d tail through self.conv1d module
        # (default) so hmonnx export emits a clean Conv op. Set True to fall
        # back to the legacy _manual_depthwise_conv1d_tail manual unroll.
        self.use_manual_depthwise_conv1d = cfg.get("use_manual_depthwise_conv1d", False)

        # Convert nn.Parameter to buffer for FX graph compatibility
        if "dt_bias" in self._parameters:
            _dt_bias_data = self.dt_bias.data.clone()
            del self._parameters["dt_bias"]
            self.register_buffer("dt_bias", _dt_bias_data, persistent=True)
        if "A_log" in self._parameters:
            _a_log_data = self.A_log.data.clone()
            del self._parameters["A_log"]
            self.register_buffer("A_log", _a_log_data, persistent=True)

        # Keep MoE trace/export semantics aligned with dense Qwen3.5:
        # split_conv_cache=True means external flat q/k/v cache inputs are
        # regrouped by TextModel and consumed by separate q/k/v conv paths here.
        if self.split_conv_cache and not hasattr(self, "in_proj_q"):
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
        target_dtype = self.in_proj_q.weight.dtype if self.split_conv_cache else self.in_proj_qkv.weight.dtype
        self.register_buffer(
            "chunk_scale",
            torch.tensor(
                1.0 / (self.head_k_dim**0.5),
                dtype=target_dtype,
                device=self.A_log.device,
            ),
            persistent=True,
        )

        a_log_exp = (-self.A_log.exp()).to(device=self.A_log.device, dtype=self.dt_bias.dtype)
        self.register_buffer("A_log_exp", a_log_exp, persistent=True)

        if self.split_conv_cache:
            self.register_buffer(
                "conv1d_q_manual_weight",
                self.conv1d_q.weight.detach().clone().squeeze(1),
                persistent=True,
            )
            self.register_buffer(
                "conv1d_k_manual_weight",
                self.conv1d_k.weight.detach().clone().squeeze(1),
                persistent=True,
            )
            self.register_buffer(
                "conv1d_v_manual_weight",
                self.conv1d_v.weight.detach().clone().squeeze(1),
                persistent=True,
            )
            if self.conv1d_q.bias is not None:
                self.register_buffer(
                    "conv1d_q_manual_bias",
                    self.conv1d_q.bias.detach().clone(),
                    persistent=True,
                )
                self.register_buffer(
                    "conv1d_k_manual_bias",
                    self.conv1d_k.bias.detach().clone(),
                    persistent=True,
                )
                self.register_buffer(
                    "conv1d_v_manual_bias",
                    self.conv1d_v.bias.detach().clone(),
                    persistent=True,
                )
        else:
            self.register_buffer(
                "conv1d_manual_weight",
                self.conv1d.weight.detach().clone().squeeze(1),
                persistent=True,
            )
            if self.conv1d.bias is not None:
                self.register_buffer(
                    "conv1d_manual_bias",
                    self.conv1d.bias.detach().clone(),
                    persistent=True,
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
        self.register_buffer("chunk_mask_incl", mask_incl, persistent=True)
        self.register_buffer("chunk_mask_strict", mask_strict, persistent=True)
        self.register_buffer("chunk_eye_matrix", eye_matrix, persistent=True)
        self.register_buffer("chunk_cumsum_matrix", cumsum_matrix, persistent=True)

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

        # GDR fused ops (conditional on fuse_gdr_ops flag).  MoE linear-attention
        # uses the same GatedDeltaNet recurrence/chunk math as dense Qwen3.5, so
        # the fused op contract is identical: Dense and MoE both pass the op
        # modules into the shared _delta_rule kernels when enabled.
        if self.fuse_gdr_ops:
            self.block_tri_inverse_op = GDRBlockTriInverse(chunk_size=chunk_size, block_size=block_size)
            self.chunk_scan_op = GDRChunkScan(
                num_chunks=num_chunks,
                num_heads=self.num_v_heads,
                k_head_dim=self.head_k_dim,
                v_head_dim=self.head_v_dim,
                chunk_size=chunk_size,
            )
            self.recurrent_scan_op = GDRRecurrentScan(sequence_length=1, output_all_states=False)
        else:
            self.block_tri_inverse_op = None
            self.chunk_scan_op = None
            self.recurrent_scan_op = None

        return self

    def _update_cfg(self, cfg: Optional[Dict] = None):
        if cfg is None:
            return
        self.use_cache = cfg.use_cache
        self.linear_attention_mode = cfg.get("linear_attention_mode", self.linear_attention_mode)
        self.linear_chunk_size = cfg.get("linear_chunk_size", self.linear_chunk_size)
        self.return_cache = cfg.get("return_cache", self.return_cache)
        self._verify_output_intermediates = cfg.get(
            "verify_output_intermediates",
            getattr(self, "_verify_output_intermediates", False),
        )
        self.input_sequence_length = cfg.get("input_sequence_length", self.input_sequence_length)
        self.batch_size = cfg.get("batch_size", self.batch_size)
        self.split_conv_cache = cfg.get("split_conv_cache", self.split_conv_cache) or hasattr(self, "in_proj_q")

        # Update eye_matrix for new batch/seq config
        chunk_size = self.linear_chunk_size
        num_chunks = (self.input_sequence_length + chunk_size - 1) // chunk_size
        flat_batch_size = self.batch_size * self.num_v_heads * num_chunks
        if hasattr(self, "chunk_scale"):
            target_dtype = self.chunk_scale.dtype
        elif self.split_conv_cache:
            target_dtype = self.in_proj_q.weight.dtype
        else:
            target_dtype = self.in_proj_qkv.weight.dtype
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

        # Keep fused chunk-scan metadata aligned with prefill/decode sequence
        # changes, matching the dense Qwen3.5 path.
        if self.fuse_gdr_ops and self.block_tri_inverse_op is not None:
            self.chunk_scan_op.num_chunks = num_chunks

        self.conv_cache_slice = xhnn.DynamicSlice([self.conv_kernel_size], [2], [1])


# ============================================================================
# SparseMoeBlock (packed Experts + shared expert)
# ============================================================================


@XHLLM_TRACEABLE_MODULES.register_module({Qwen3_5MoeSparseMoeBlock: "Qwen3_5MoeSparseMoeBlock"})
class _Qwen3_5MoeSparseMoeBlock(DynamicModule):  # noqa: N801
    """Wrap Qwen3.5-MoE SparseMoeBlock.

    The HF Qwen3_5MoeExperts uses packed weight tensors:
      - gate_up_proj: (num_experts, 2*intermediate_dim, hidden_dim)
      - down_proj: (num_experts, hidden_dim, intermediate_dim)

    We split gate_up into gate + up and pack into standard MoeBlock format.
    Also handles shared_expert + shared_expert_gate.
    """

    # def forward(self, hidden_states):
    #     batch_size, sequence_length, hidden_dim = hidden_states.shape
    #     hidden_states_reshaped = hidden_states.view(-1, hidden_dim)

    #     # Router: softmax + topk done inside MoeBlock
    #     router_logits = self.gate(hidden_states_reshaped)
    #     routing_weights = F.softmax(router_logits, dim=-1)
    #     # MoeBlock expects 3D: [batch, seq, num_experts]
    #     routing_weights = routing_weights.view(batch_size, sequence_length, -1)

    #     # MoE block forward
    #     moe_out = self.moeblock(hidden_states, routing_weights)

    #     # Shared expert
    #     shared_out = self.shared_expert(hidden_states_reshaped)
    #     shared_out = torch.sigmoid(self.shared_expert_gate(hidden_states_reshaped)) * shared_out
    #     shared_out = shared_out.reshape(batch_size, sequence_length, hidden_dim)

    #     return moe_out + shared_out

    def forward(self, hidden_states):
        router_logits = self.gate(hidden_states)
        routing_weights = F.softmax(router_logits, dim=-1)
        moe_out = self.moeblock(hidden_states, routing_weights)
        shared_out = self.shared_expert(hidden_states)
        shared_out = torch.sigmoid(self.shared_expert_gate(hidden_states)) * shared_out
        return moe_out + shared_out, router_logits

    def _setup(self, cfg: Optional[Dict] = None):
        self.input_seq_len = cfg.input_sequence_length
        self.batch_size = cfg.batch_size
        self.device = self.gate.weight.device

        # Save gate weight as buffer and keep TopKRouter info
        gate_weight_data = self.gate.weight.data.clone()
        top_k = self.gate.top_k
        num_experts_dim, hidden_dim = gate_weight_data.shape

        # Replace TopKRouter with standard nn.Linear so it traces as call_module
        self.gate = nn.Linear(
            hidden_dim,
            num_experts_dim,
            bias=False,
            device=self.device,
            dtype=gate_weight_data.dtype,
        )
        self.gate.weight.data.copy_(gate_weight_data)

        # Determine expert dimensions from the packed Experts
        experts = self.experts
        if hasattr(experts, "act_fn"):
            act_fn_name = _get_activation_name(experts.act_fn)
        else:
            act_fn_name = _get_activation_name(experts[0].act_fn)

        # Qwen3_5Moe normalizes routing weights inside gate, so norm_topk_prob=True
        self.moeblock = MoeBlock(act_fn_name, top_k, True)

        if hasattr(experts, "gate_up_proj"):
            # Split packed gate_up_proj -> gate_proj + up_proj
            # gate_up_proj shape: (num_experts, 2*intermediate_dim, hidden_dim)
            intermediate_dim = experts.intermediate_dim
            gate_up = experts.gate_up_proj.data
            num_experts = gate_up.shape[0]
            target_dtype = gate_up.dtype

            self.moeblock.expert_gate_proj_weight = nn.Parameter(
                torch.empty(num_experts, intermediate_dim, hidden_dim, device=self.device, dtype=target_dtype)
            )
            self.moeblock.expert_gate_proj_bias = None
            self.moeblock.expert_gate_proj_weight.data.copy_(gate_up[:, :intermediate_dim, :].to(self.device))

            self.moeblock.expert_up_proj_weight = nn.Parameter(
                torch.empty(num_experts, intermediate_dim, hidden_dim, device=self.device, dtype=target_dtype)
            )
            self.moeblock.expert_up_proj_bias = None
            self.moeblock.expert_up_proj_weight.data.copy_(gate_up[:, intermediate_dim:, :].to(self.device))

            # Release the packed gate_up tensor before wiring the remaining
            # expert weights so the wrap peak does not hold both layouts longer
            # than necessary.
            _move_parameter_to_meta(experts, "gate_up_proj")
            del gate_up

            down_proj = experts.down_proj
            if down_proj.device == self.device and down_proj.is_contiguous():
                self.moeblock.expert_down_proj_weight = down_proj
            else:
                self.moeblock.expert_down_proj_weight = nn.Parameter(down_proj.data.to(self.device).contiguous())
            _move_parameter_to_meta(experts, "down_proj")

            self.moeblock.expert_down_proj_bias = None
            del down_proj
            # Release original packed experts to free memory
            del self.experts
        elif (
            len(experts) > 0
            and hasattr(experts[0], "gate_proj")
            and hasattr(experts[0], "up_proj")
            and hasattr(experts[0], "down_proj")
        ):
            for linear_name in ("gate_proj", "up_proj", "down_proj"):
                _pack_defused_expert_linear_to_moeblock(self.moeblock, experts, linear_name, self.device)

            # Release expert modules after packing to reduce memory.
            self.experts = nn.ModuleList()
        else:
            raise RuntimeError(f"Unsupported Qwen3.5-MoE experts structure: {type(experts)}")
        torch.cuda.empty_cache()
        return self


# ============================================================================
# Decoder Layer
# ============================================================================


@XHLLM_TRACEABLE_MODULES.register_module({Qwen3_5MoeDecoderLayer: "Qwen3_5MoeDecoderLayer"})
class _Qwen3_5MoeDecoderLayer(DynamicModule):  # noqa: N801
    """Dispatches between linear_attention and full_attention layers."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_seq_length: Optional[Tensor] = None,
        current_input_length: Optional[Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        linear_attn_mask: Optional[Tensor] = None,
        past_k_cache: Optional[Tensor] = None,
        past_v_cache: Optional[Tensor] = None,
        past_conv_cache: Optional[Tensor] = None,
        past_recurrent_state: Optional[Tensor] = None,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        if self.layer_type == "linear_attention":
            attn_output = self.linear_attn(
                hidden_states=hidden_states,
                conv_cache=past_conv_cache,
                recurrent_state=past_recurrent_state,
                linear_attn_mask=linear_attn_mask,
                current_input_length=current_input_length,
            )
            hidden_states, conv_cache_out, recurrent_state_out = attn_output
        else:
            hidden_states, _, _ = self.self_attn(
                hidden_states=hidden_states,
                past_seq_length=past_seq_length,
                current_input_length=current_input_length,
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
        else:
            return hidden_states

    def _setup(self, cfg: Optional[Dict] = None):
        return self


# ============================================================================
# TextModel (with M-RoPE masks)
# ============================================================================


@XHLLM_TRACEABLE_MODULES.register_module({Qwen3_5MoeTextModel: "Qwen3_5MoeTextModel"})
class _Qwen3_5MoeTextModel(DynamicModule):  # noqa: N801
    """Qwen3.5-MoE TextModel with M-RoPE interleaved position encoding."""

    def _setup(self, cfg):
        self.batch_size = cfg.get("batch_size", 1)
        self.batch_size = cfg.get("batch_size", 1)
        self.only_first_block = cfg.get("only_first_block", False)
        self.max_layers = -1
        if self.only_first_block:
            self.max_layers = 1
        else:
            if "max_layers" in cfg and cfg.max_layers is not None:
                self.max_layers = cfg.max_layers
        self.num_logits_to_keep = cfg.num_logits_to_keep
        assert self.num_logits_to_keep in [0, 1]
        self.output_hidden_state_indices = cfg.get("output_hidden_state_indices", None)
        if self.output_hidden_state_indices is not None:
            self._output_hidden_set = set(self.output_hidden_state_indices)
        self.output_post_norm_hidden = cfg.get("output_post_norm_hidden", False)

        input_seq_len = cfg.input_sequence_length
        self.slice = xhnn.Slice([0], [input_seq_len], [1], [1])

        self.llm_gather = xhnn.BatchGather(1)
        self.llm_gather.update_offset_indices(self.batch_size, input_seq_len)

        def _llm_gather_update_cfg(self_g: xhnn.BatchGather, cfg_inner: Optional[Dict] = None):
            input_seq_len_inner = cfg_inner.input_sequence_length
            batch_size_inner = cfg_inner.get("batch_size", 1)
            self_g.update_offset_indices(batch_size_inner, input_seq_len_inner)

        self.llm_gather._update_cfg = types.MethodType(_llm_gather_update_cfg, self.llm_gather)

        def _slice_update_cfg(self_s, cfg_inner: Optional[Dict] = None):
            input_seq_len_inner = cfg_inner.input_sequence_length
            self_s.ends = [input_seq_len_inner]

        self.slice._update_cfg = types.MethodType(_slice_update_cfg, self.slice)
        self.use_cache = cfg.use_cache
        self.split_conv_cache = cfg.get("split_conv_cache", True)

        # Layer type tracking
        self.layer_types = self.config.layer_types
        self.num_full_attention_layers = sum(1 for t in self.layer_types if t == "full_attention")
        self.num_linear_attention_layers = sum(1 for t in self.layer_types if t == "linear_attention")

        # MoE conversion can enter TextModel tracing before child linear-attn
        # modules have independently consumed the full wrap cfg. Make the split
        # cache contract explicit at the TextModel boundary so a flat external
        # q/k/v signature is never threaded into a merged qkv GatedDeltaNet.
        if self.split_conv_cache:
            for idx_layer, decoder_layer in enumerate(self.layers):
                if self.layer_types[idx_layer] != "linear_attention":
                    continue
                linear_attn = getattr(decoder_layer, "linear_attn", None)
                if linear_attn is None:
                    continue
                linear_attn.split_conv_cache = True
                if hasattr(linear_attn, "_setup") and not hasattr(linear_attn, "in_proj_q"):
                    linear_attn._setup(cfg)

        # Mark specific linear_attention layers for alpha scaling
        alpha_scaling_layers = cfg.get("alpha_scaling_layers", [8, 20])
        chunk_inverse_alpha = cfg.get("chunk_inverse_alpha", 0.5)
        for idx_layer, decoder_layer in enumerate(self.layers):
            if self.layer_types[idx_layer] == "linear_attention" and idx_layer in alpha_scaling_layers:
                gdn = decoder_layer.linear_attn
                gdn._alpha_scaling_config = {"alpha": chunk_inverse_alpha}

        # Set up cos/sin cache
        if not hasattr(self.rotary_emb, "cos_cached"):
            self.rotary_emb.setup_after_callback = self._setup_cos_sin_embeding
        else:
            self._setup_cos_sin_embeding()

    def _setup_cos_sin_embeding(self):
        if hasattr(self.rotary_emb, "cos_cached"):
            _ = self.rotary_emb.cos_cached
        if hasattr(self.rotary_emb, "sin_cached"):
            _ = self.rotary_emb.sin_cached

    def forward(
        self,
        input_embeds: Optional[Tensor] = None,
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
        position_embeddings = self.rotary_emb(time_position_ids, hight_position_ids, width_position_ids)
        hidden_states = input_embeds

        conv_cache_out_list = []
        recurrent_state_out_list = []
        collected_hidden_states = []
        full_attn_cache_idx = 0
        linear_attn_cache_idx = 0
        split_conv_cache = self.split_conv_cache
        if not split_conv_cache and (
            _looks_like_flat_split_conv_cache(past_conv_cache) or _layers_use_split_conv_cache(self.layers)
        ):
            split_conv_cache = True
        if split_conv_cache and _is_nested_split_conv_cache(past_conv_cache):
            past_conv_cache = _regroup_flat_split_conv_cache(past_conv_cache)

        for idx_layer, decoder_layer in enumerate(self.layers):
            layer_type = self.layer_types[idx_layer]

            if self.use_cache:
                if layer_type == "full_attention":
                    _past_k_cache = past_key_cache[full_attn_cache_idx] if past_key_cache is not None else None
                    _past_v_cache = past_value_cache[full_attn_cache_idx] if past_value_cache is not None else None
                    _past_conv_cache = None
                    _past_recurrent_state = None
                    full_attn_cache_idx += 1
                else:
                    _past_k_cache = None
                    _past_v_cache = None
                    _past_conv_cache = _select_linear_attn_conv_cache(
                        past_conv_cache,
                        linear_attn_cache_idx,
                        split_conv_cache,
                    )
                    _past_recurrent_state = (
                        past_recurrent_state[linear_attn_cache_idx] if past_recurrent_state is not None else None
                    )
                    linear_attn_cache_idx += 1
            else:
                _past_k_cache = None
                _past_v_cache = None
                # Linear attention layers need conv_cache/recurrent_state even
                # when use_cache=False (as initial state for the computation)
                if layer_type == "linear_attention":
                    _past_conv_cache = _select_linear_attn_conv_cache(
                        past_conv_cache,
                        linear_attn_cache_idx,
                        split_conv_cache,
                    )
                    _past_recurrent_state = (
                        past_recurrent_state[linear_attn_cache_idx] if past_recurrent_state is not None else None
                    )
                    linear_attn_cache_idx += 1
                else:
                    _past_conv_cache = None
                    _past_recurrent_state = None

            if layer_type == "linear_attention":
                hidden_states, conv_cache_out, recurrent_state_out = decoder_layer(
                    hidden_states,
                    past_seq_length=past_seq_length,
                    current_input_length=current_input_length,
                    position_embeddings=position_embeddings,
                    linear_attn_mask=linear_attn_mask,
                    past_k_cache=_past_k_cache,
                    past_v_cache=_past_v_cache,
                    past_conv_cache=_past_conv_cache,
                    past_recurrent_state=_past_recurrent_state,
                )
                if isinstance(conv_cache_out, (list, tuple)):
                    conv_cache_out_list.extend(conv_cache_out)
                else:
                    conv_cache_out_list.append(conv_cache_out)
                if isinstance(recurrent_state_out, (list, tuple)):
                    recurrent_state_out_list.extend(recurrent_state_out)
                else:
                    recurrent_state_out_list.append(recurrent_state_out)
            else:
                hidden_states = decoder_layer(
                    hidden_states,
                    past_seq_length=past_seq_length,
                    current_input_length=current_input_length,
                    position_embeddings=position_embeddings,
                    linear_attn_mask=linear_attn_mask,
                    past_k_cache=_past_k_cache,
                    past_v_cache=_past_v_cache,
                    past_conv_cache=_past_conv_cache,
                    past_recurrent_state=_past_recurrent_state,
                )

            if self.output_hidden_state_indices is not None and idx_layer in self._output_hidden_set:
                collected_hidden_states.append(hidden_states)

            if self.max_layers > 0 and idx_layer + 1 >= self.max_layers:
                break

            # if True:
            #     break
            # if idx_layer == 3:
            #     break

        if self.num_logits_to_keep == 0:
            pass
        else:
            hidden_states = self.llm_gather(hidden_states, current_input_length - 1)
        if self.output_hidden_state_indices is not None:
            target_hidden = torch.cat(collected_hidden_states, dim=-1)
            if self.num_logits_to_keep != 0:
                target_hidden = self.llm_gather(target_hidden, current_input_length - 1)
        hidden_states = self.norm(hidden_states)
        post_norm_out = hidden_states

        if split_conv_cache:
            conv_cache_out_list = _flatten_split_conv_cache_outputs(conv_cache_out_list)
        else:
            conv_cache_out_list = _flatten_merged_conv_cache_outputs(conv_cache_out_list)
        if self.output_hidden_state_indices is not None:
            return hidden_states, conv_cache_out_list, recurrent_state_out_list, target_hidden
        if self.output_post_norm_hidden:
            return hidden_states, conv_cache_out_list, recurrent_state_out_list, post_norm_out
        return hidden_states, conv_cache_out_list, recurrent_state_out_list


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
                input_embeds=inputs_embeds,
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
                input_embeds=inputs_embeds,
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
                input_embeds=inputs_embeds,
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
                input_embeds=inputs_embeds,
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
