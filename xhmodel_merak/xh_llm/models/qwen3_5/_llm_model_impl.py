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
import math
import types
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from xhquant import nn as xhnn
from xhquant.nn import BfpFlashAttention, LLMCacheV2, MaskedSoftmax, MatMul, RMSNorm
from xhquant.nn.modules import Cos, Sin
from xhquant.utils.registry import DynamicModule

from ...register import XHLLM_TRACEABLE_MODULES

# Re-use the chunk/recurrent gated delta rule implementations from Qwen3Next
# from ..qwen3next._next_model import (
#     torch_chunk_gated_delta_rule,
#     torch_recurrent_gated_delta_rule,
# )
from ._delta_rule import torch_chunk_gated_delta_rule, torch_recurrent_gated_delta_rule
from ._gdr_ops import GDRBlockTriInverse, GDRChunkScan, GDRRecurrentScan
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
from .split_conv_cache_utils import (
    _flatten_merged_conv_cache_outputs,
    _flatten_split_conv_cache_outputs,
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


def _neumann_8x8(block: torch.Tensor, eye_8: torch.Tensor) -> torch.Tensor:
    p = block
    r = eye_8 + p
    p = (p.unsqueeze(-1) * p.unsqueeze(-3)).sum(-2)
    r = (r.unsqueeze(-1) * (eye_8 + p).unsqueeze(-3)).sum(-2)
    p = (p.unsqueeze(-1) * p.unsqueeze(-3)).sum(-2)
    r = (r.unsqueeze(-1) * (eye_8 + p).unsqueeze(-3)).sum(-2)
    return r


_HF_QWEN35_MODELING = "transformers.models.qwen3_5.modeling_qwen3_5"


def parallel_chunk_inverse_block(
    attn: torch.Tensor,
    eye_matrix: torch.Tensor,
    eye_8_batched: torch.Tensor,
    chunk_size: int = 64,
    block_size: int = 8,
) -> torch.Tensor:
    """Compute (I - A)^{-1} using block forward substitution.

    Numerically stable (like _ref) AND uses matmul (GPU-friendly).

    Algorithm:
        1. Split 64x64 into 8 blocks of 8x8
        2. Diagonal blocks: (I - A_ii)^{-1} via Neumann series (safe at 8x8)
        3. Off-diagonal: R[I,:I] = R[I,I] @ A[I,:I] @ R[:I,:I] (matmul)

    Only 8 serial steps (vs 64 in _ref), each using proper matmul.
    No in-place operations — result built via torch.cat.

    Args:
        attn: [B, heads, num_chunks, chunk_size, chunk_size] strictly lower triangular
        eye_matrix: Pre-computed identity (BHN, chunk_size, chunk_size)
        eye_8_batched: Pre-computed 8x8 identity (BHN*nb, block_size, block_size)
        chunk_size: 64
        block_size: 8 (divides chunk_size evenly)
    """
    B, heads, num_chunks, cs, _ = attn.shape  # noqa: N806
    flat = attn.flatten(0, -3)  # (BHN, 64, 64)
    b = block_size  # 8
    nb = chunk_size // b  # 8

    # --- Step 1: Batch compute all 8 diagonal block inverses ---
    # Stack diagonal blocks: (BHN, 8, 8, 8) → (BHN*8, 8, 8)
    diag_blocks = torch.stack(
        [flat[:, block_idx * b : (block_idx + 1) * b, block_idx * b : (block_idx + 1) * b] for block_idx in range(nb)],
        dim=1,
    )  # (BHN, 8, 8, 8)
    batched = diag_blocks.reshape(-1, b, b)  # (BHN*8, 8, 8)
    diag_invs_batched = _neumann_8x8(batched, eye_8_batched)  # (BHN*8, 8, 8)
    diag_invs_all = diag_invs_batched.reshape(-1, nb, b, b)  # (BHN, 8, 8, 8)

    # --- Step 2: Block forward substitution (8 serial steps, no in-place) ---
    # Build R incrementally as a growing upper-left block.
    # R_upper starts as the first diagonal block and grows each step.
    R_upper = diag_invs_all[:, 0]  # (BHN, 8, 8) — R[0:8, 0:8]  # noqa: N806

    for block_idx in range(1, nb):
        s = block_idx * b  # start index of current block row

        # A[s:s+8, 0:s] @ R_upper → off-diagonal contribution
        # shapes: (BHN, 8, s) @ (BHN, s, s) → (BHN, 8, s)
        T = torch.matmul(flat[:, s : s + b, :s], R_upper)  # noqa: N806

        # off_diag = diag_inv[block_idx] @ T
        # shapes: (BHN, 8, 8) @ (BHN, 8, s) → (BHN, 8, s)
        diag_inv_i = diag_invs_all[:, block_idx]  # (BHN, 8, 8)
        off_diag = torch.matmul(diag_inv_i, T)  # (BHN, 8, s)

        # Extend R_upper from (BHN, s, s) to (BHN, s+8, s+8)
        # Old rows get 8 zero-columns appended on the right via Pad
        old_rows = F.pad(R_upper, (0, b))  # (BHN, s, s+8)

        # New rows: [off_diag | diag_inv_i]
        new_rows = torch.cat([off_diag, diag_inv_i], dim=-1)  # (BHN, 8, s+8)

        # Stack vertically
        R_upper = torch.cat([old_rows, new_rows], dim=-2)  # (BHN, s+8, s+8)  # noqa: N806

    # R_upper is now (BHN, 64, 64) — the full R
    return R_upper.reshape(B, heads, num_chunks, cs, cs)


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
# ============================================================================
# RMSNorm wrappers
# ============================================================================


if TYPE_CHECKING:

    class _Qwen3_5TextRMSNormBase(DynamicModule, Qwen3_5RMSNorm):  # type: ignore[misc]
        ...

else:
    _Qwen3_5TextRMSNormBase = DynamicModule


@XHLLM_TRACEABLE_MODULES.register_module(_with_hf_alias(Qwen3_5RMSNorm, "Qwen3_5RMSNorm"))
class _Qwen3_5TextRMSNorm(_Qwen3_5TextRMSNormBase):  # noqa: N801
    """Wrap Qwen3_5RMSNorm.

    Qwen3.5 uses ``(1 + weight)`` style normalization (weight init = 0).
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


_rms_norm_gated_registry = _with_hf_alias(Qwen3_5RMSNormGated, "Qwen3_5RMSNormGated")
if FusedRMSNormGated is not None:
    _rms_norm_gated_registry[FusedRMSNormGated] = "FusedRMSNormGated"


if TYPE_CHECKING:

    class _Qwen3_5RMSNormGatedBase(DynamicModule, Qwen3_5RMSNormGated):  # type: ignore[misc]
        ...

else:
    _Qwen3_5RMSNormGatedBase = DynamicModule


@XHLLM_TRACEABLE_MODULES.register_module(_rms_norm_gated_registry)
class _Qwen3_5RMSNormGated(_Qwen3_5RMSNormGatedBase):  # noqa: N801
    """Wrap Qwen3_5RMSNormGated / FusedRMSNormGated used inside GatedDeltaNet.

    Original forward: ``silu(gate) * x → norm → scale(weight+1)``.
    Wrapper: ``norm(x) * scale → silu(gate)``  (decomposed for hardware).
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
class _Qwen3_5TextAttention(_Qwen3_5TextAttentionBase):  # noqa: N801
    """Qwen3.5 full attention with gating mechanism.

    q_proj outputs ``query + gate`` (2× heads), attention result is
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
        # Must reshape FIRST, then split per head to separate query & gate correctly.
        # Use split instead of chunk for xhquant frontend graph compatibility.
        qg = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim * 2)
        query_states, gate = torch.split(qg, self.head_dim, dim=-1)
        # gate: (bsz, q_len, num_heads, head_dim) → flatten heads
        gate = gate.reshape(bsz, q_len, -1)

        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # query_states is already (bsz, q_len, num_heads, head_dim) from chunk
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
            persistent=False,
        )
        return self


# ============================================================================
# GatedDeltaNet (linear attention)
# ============================================================================


if TYPE_CHECKING:

    class _Qwen3_5GatedDeltaNetBase(DynamicModule, Qwen3_5GatedDeltaNet):  # type: ignore[misc]
        ...

else:
    _Qwen3_5GatedDeltaNetBase = DynamicModule


@XHLLM_TRACEABLE_MODULES.register_module(_with_hf_alias(Qwen3_5GatedDeltaNet, "Qwen3_5GatedDeltaNet"))
class _Qwen3_5GatedDeltaNet(_Qwen3_5GatedDeltaNetBase):  # noqa: N801
    """Qwen3.5 GatedDeltaNet linear attention wrapper.

    Key difference from Qwen3Next: Qwen3.5 uses **separate** projections
    (in_proj_qkv, in_proj_z, in_proj_b, in_proj_a) instead of Qwen3Next's
    merged projections (in_proj_qkvz, in_proj_ba).
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

        # Separate projections (Qwen3.5 style)
        z = self.in_proj_z(hidden_states)  # [bs, seq, value_dim]
        z = z.reshape(batch_size, seq_len, -1, self.head_v_dim)
        b = self.in_proj_b(hidden_states)  # [bs, seq, num_v_heads]
        a = self.in_proj_a(hidden_states)  # [bs, seq, num_v_heads]

        use_recurrent = self.linear_attention_mode == "recurrent"
        if self.linear_attention_mode == "auto" and current_input_length is not None:
            resolved_len = _resolve_python_int_length(current_input_length)
            use_recurrent = (resolved_len == 1) if resolved_len is not None else False

        _verify_intermediates = getattr(self, "_verify_output_intermediates", False)

        if getattr(self, "split_conv_cache", False) or hasattr(self, "in_proj_q"):
            # === Split conv_cache path (3 separate q/k/v tensors) ===
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

            query = query_states.transpose(1, 2).reshape(batch_size, seq_len, -1, self.head_k_dim)
            key = key_states.transpose(1, 2).reshape(batch_size, seq_len, -1, self.head_k_dim)
            value = value_states.transpose(1, 2).reshape(batch_size, seq_len, -1, self.head_v_dim)
        else:
            # === Merged conv_cache path (single tensor, default) ===
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
            query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
            key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
            value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

        beta = b.sigmoid()
        g = self.A_log_exp * F.softplus(a + self.dt_bias)

        mask_qkv = linear_attn_mask.unsqueeze(-1).unsqueeze(-1)
        value = value * mask_qkv
        beta = beta * mask
        g = g * mask

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

        suppress_recurrent_state_outputs = (
            getattr(self, "suppress_recurrent_state_outputs", False) and not use_recurrent
        )
        if suppress_recurrent_state_outputs:
            recurrent_state_out = None
        elif _verify_intermediates and self.input_sequence_length > 1 and use_recurrent:
            recurrent_state_out = tuple(_recurrent_snapshots)
        else:
            recurrent_state_out = last_recurrent_state if last_recurrent_state is not None else recurrent_state

        # b_sz, s, n, h = z.shape
        # core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        # z = z.reshape(-1, z.shape[-1])
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
        if self.split_conv_cache:
            target_dtype = self.in_proj_q.weight.dtype
        else:
            target_dtype = self.in_proj_qkv.weight.dtype
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
        self.use_cache = cfg.use_cache
        self.linear_attention_mode = cfg.get("linear_attention_mode", self.linear_attention_mode)
        self.linear_chunk_size = cfg.get("linear_chunk_size", self.linear_chunk_size)
        self.return_cache = cfg.get("return_cache", self.return_cache)
        self.input_sequence_length = cfg.get("input_sequence_length", self.input_sequence_length)
        self.batch_size = cfg.get("batch_size", self.batch_size)
        self._verify_output_intermediates = cfg.get("verify_output_intermediates", self._verify_output_intermediates)
        self.split_conv_cache = cfg.get("split_conv_cache", self.split_conv_cache) or hasattr(self, "in_proj_q")
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

        # Update GDR fused op buffers
        if getattr(self, "chunk_scan_op", None) is not None:
            self.chunk_scan_op.num_chunks = num_chunks

        self.conv_cache_slice = xhnn.DynamicSlice([self.conv_kernel_size], [2], [1])


# ============================================================================
# Decoder Layer
# ============================================================================


if TYPE_CHECKING:

    class _Qwen3_5DecoderLayerBase(DynamicModule, Qwen3_5DecoderLayer):  # type: ignore[misc]
        ...

else:
    _Qwen3_5DecoderLayerBase = DynamicModule


@XHLLM_TRACEABLE_MODULES.register_module(_with_hf_alias(Qwen3_5DecoderLayer, "Qwen3_5DecoderLayer"))
class _Qwen3_5DecoderLayer(_Qwen3_5DecoderLayerBase):  # noqa: N801
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


if TYPE_CHECKING:

    class _Qwen3_5TextModelBase(DynamicModule, Qwen3_5TextModel):  # type: ignore[misc]
        ...

else:
    _Qwen3_5TextModelBase = DynamicModule


@XHLLM_TRACEABLE_MODULES.register_module(_with_hf_alias(Qwen3_5TextModel, "Qwen3_5TextModel"))
class _Qwen3_5TextModel(_Qwen3_5TextModelBase):  # noqa: N801
    """Qwen3.5 TextModel with M-RoPE interleaved position encoding.

    Position encoding approach (from Qwen3VL adaptation):
    - Pre-compute cos/sin cache per position (no interleaving)
    - At forward time, look up cos/sin for T/H/W positions separately
    - Apply interleaved mrope masks to combine T/H/W embeddings
    """

    def _setup(self, cfg):
        self.batch_size = cfg.get("batch_size", 1)
        self.only_first_block = cfg.get("only_first_block", False)
        self.max_layers = -1
        if self.only_first_block:
            self.max_layers = 1
        else:
            if "max_layers" in cfg and cfg.max_layers is not None:
                self.max_layers = cfg.max_layers
        self.num_logits_to_keep = cfg.num_logits_to_keep
        self.support_long_context_over_fp16_limit = cfg.get("support_long_context_over_fp16_limit", True)
        assert self.num_logits_to_keep in [0, 1]

        # Speculative decoding: optional hidden state outputs
        self.output_hidden_state_indices = cfg.get("output_hidden_state_indices", None)
        if self.output_hidden_state_indices is not None:
            self._output_hidden_set = set(self.output_hidden_state_indices)
        self.output_post_norm_hidden = cfg.get("output_post_norm_hidden", False)

        # Detect split_conv_cache from cfg (propagated to linear attention layers)
        self.split_conv_cache = cfg.get("split_conv_cache", True)

        input_seq_len = cfg.input_sequence_length
        self.slice = xhnn.Slice([0], [input_seq_len], [1], [1])

        self.llm_gather = xhnn.BatchGather(1)
        self.llm_gather.update_offset_indices(self.batch_size, input_seq_len)

        self.cos = Cos()
        self.sin = Sin()

        def _llm_gather_update_cfg(self: xhnn.BatchGather, cfg: Optional[Dict] = None):
            input_seq_len = cfg.input_sequence_length
            batch_size = cfg.get("batch_size", 1)
            self.update_offset_indices(batch_size, input_seq_len)

        self.llm_gather._update_cfg = types.MethodType(_llm_gather_update_cfg, self.llm_gather)

        def _slice_update_cfg(self, cfg: Optional[Dict] = None):
            input_seq_len = cfg.input_sequence_length
            self.ends = [input_seq_len]

        self.slice._update_cfg = types.MethodType(_slice_update_cfg, self.slice)
        self.use_cache = cfg.use_cache

        # Layer type tracking
        self.layer_types = self.config.layer_types
        self.num_full_attention_layers = sum(1 for t in self.layer_types if t == "full_attention")
        self.num_linear_attention_layers = sum(1 for t in self.layer_types if t == "linear_attention")

        # Dense conversion can enter TextModel tracing before child linear-attn
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

        # ---- M-RoPE interleaved masks ----
        # Get mrope_section from config
        rope_parameters = self.config.rope_parameters
        if isinstance(rope_parameters, dict):
            mrope_section = rope_parameters.get("mrope_section", [11, 11, 10])
        else:
            mrope_section = getattr(rope_parameters, "mrope_section", [11, 11, 10])

        # Compute rotary dim
        partial_rotary_factor = getattr(self.config, "partial_rotary_factor", 0.25)
        head_dim = self.config.head_dim
        rotary_dim = int(head_dim * partial_rotary_factor)
        half_dim = rotary_dim // 2  # inv_freq length

        # Build interleaved mask indices from mrope_section
        # H positions: offset=1, stride=3, count=mrope_section[1]
        h_ids = torch.arange(1, mrope_section[1] * 3, 3)
        # W positions: offset=2, stride=3, count=mrope_section[2]
        w_ids = torch.arange(2, mrope_section[2] * 3, 3)

        # T mask: all positions not occupied by H or W
        time_mask = torch.ones(half_dim)
        time_mask[h_ids] = 0
        time_mask[w_ids] = 0
        time_mask = torch.cat([time_mask, time_mask], 0)
        time_mask.unsqueeze_(0).unsqueeze_(0)
        self.rotary_emb.register_buffer("time_mask", time_mask.half(), persistent=False)

        # H mask
        hight_mask = torch.zeros(half_dim)
        hight_mask[h_ids] = 1
        hight_mask = torch.cat([hight_mask, hight_mask], 0)
        hight_mask.unsqueeze_(0).unsqueeze_(0)
        self.rotary_emb.register_buffer("hight_mask", hight_mask.half(), persistent=False)

        # W mask
        width_mask = torch.zeros(half_dim)
        width_mask[w_ids] = 1
        width_mask = torch.cat([width_mask, width_mask], 0)
        width_mask.unsqueeze_(0).unsqueeze_(0)
        self.rotary_emb.register_buffer("width_mask", width_mask.half(), persistent=False)

        # Set up cos/sin cache only for long-context mode.
        if self.support_long_context_over_fp16_limit:
            if not hasattr(self.rotary_emb, "cos_cached"):
                self.rotary_emb.setup_after_callback = self._setup_cos_sin_embeding
            else:
                self._setup_cos_sin_embeding()

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
        """
        Returns:
            hidden_states, conv_cache_out_list, recurrent_state_out_list
        """
        hidden_states = input_embeds
        position_embeddings = self._build_qwen3_5_mrope_position_embeddings(
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

        # Track cache indices separately for full/linear attention layers
        conv_cache_out_list = []
        recurrent_state_out_list = []
        full_attn_cache_idx = 0
        linear_attn_cache_idx = 0
        collected_hidden_states = []
        split_conv_cache = self.split_conv_cache
        if not split_conv_cache and (
            _looks_like_flat_split_conv_cache(past_conv_cache) or _layers_use_split_conv_cache(self.layers)
        ):
            split_conv_cache = True
        if split_conv_cache and _is_nested_split_conv_cache(past_conv_cache):
            past_conv_cache = _regroup_flat_split_conv_cache(past_conv_cache)

        for idx, decoder_layer in enumerate(self.layers):
            layer_type = self.layer_types[idx]

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
                # Full-attention KV cache is disabled, but linear attention still
                # requires zero/init state tensors for the current forward/trace.
                _past_k_cache = None
                _past_v_cache = None
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
                conv_cache_out_list.append(conv_cache_out)
                if recurrent_state_out is not None:
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

            # Collect hidden states at target layer indices (for DFlash)
            if self.output_hidden_state_indices is not None and idx in self._output_hidden_set:
                collected_hidden_states.append(hidden_states)

            if self.max_layers > 0 and idx + 1 >= self.max_layers:
                break

        # Build target_hidden for DFlash: cat collected hidden states along last dim
        if self.output_hidden_state_indices is not None:
            target_hidden = torch.cat(collected_hidden_states, dim=-1)
            if self.num_logits_to_keep != 0:
                target_hidden = self.llm_gather(target_hidden, current_input_length - 1)

        if self.num_logits_to_keep == 0:
            pass
        else:
            hidden_states = self.llm_gather(hidden_states, current_input_length - 1)
        hidden_states = self.norm(hidden_states)

        if self.output_post_norm_hidden:
            post_norm_hidden = hidden_states

        if split_conv_cache:
            conv_cache_out_list = _flatten_split_conv_cache_outputs(conv_cache_out_list)
        else:
            conv_cache_out_list = _flatten_merged_conv_cache_outputs(conv_cache_out_list)
        if self.output_hidden_state_indices is not None:
            return hidden_states, conv_cache_out_list, recurrent_state_out_list, target_hidden
        elif self.output_post_norm_hidden:
            return hidden_states, conv_cache_out_list, recurrent_state_out_list, post_norm_hidden
        return hidden_states, conv_cache_out_list, recurrent_state_out_list


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
