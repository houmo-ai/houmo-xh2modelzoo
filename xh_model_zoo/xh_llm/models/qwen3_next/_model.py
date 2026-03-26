import math
import sys
import types
from copy import deepcopy
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.models.qwen3_next.modeling_qwen3_next import (
    Qwen3NextAttention,
    Qwen3NextDecoderLayer,
    Qwen3NextForCausalLM,
    Qwen3NextGatedDeltaNet,
    Qwen3NextMLP,
    Qwen3NextModel,
    Qwen3NextRMSNorm,
    Qwen3NextRMSNormGated,
    Qwen3NextRotaryEmbedding,
    Qwen3NextSparseMoeBlock,
)

from xhquant import nn as xhnn
from xhquant.nn import BfpFlashAttention, LLMCache, LLMCacheV2, MaskedSoftmax, MatMul, RMSNorm
from xhquant.nn.modules.moeblock import MoeBlock
from xhquant.utils.registry import DynamicModule

from ..builder import XHLLM_TRACEABLE_MODULES

try:
    from fla.modules import FusedRMSNormGated
except ImportError:
    FusedRMSNormGated = None


def _lower_triangular_mask(
    size: int, ref_tensor: torch.Tensor, dtype: Optional[torch.dtype] = None, strict: bool = False
) -> torch.Tensor:
    """Create lower triangular mask using a reference tensor for device/dtype compatibility.
    
    Args:
        size: Size of the mask
        ref_tensor: Reference tensor to get device from
        dtype: Output dtype (defaults to ref_tensor's dtype)
        strict: If True, use strict lower triangular (diagonal is False)
    """
    # Create indices on CPU first, then move to device using tensor operations
    # This avoids TorchFX tracing issues with torch.arange(device=...)
    idx = torch.arange(size, dtype=torch.long)
    # Create mask on CPU
    if strict:
        mask = idx[:, None] > idx[None, :]
    else:
        mask = idx[:, None] >= idx[None, :]
    target_dtype = dtype if dtype is not None else ref_tensor.dtype
    # Move to same device as ref_tensor using type_as for tracing compatibility
    return mask.to(target_dtype).to(ref_tensor.device)


def _manual_depthwise_conv1d_tail(
    hidden_states_new: Tensor,
    weight: Tensor,
    bias: Optional[Tensor],
    output_length: int,
) -> Tensor:
    """Compute kept tail of depthwise Conv1d via slice + mul + add.

    Equivalent to:
    ``F.conv1d(hidden_states_new, weight.unsqueeze(1), bias, groups=conv_dim)[:, :, -output_length:]``
    when ``weight`` is shaped ``[conv_dim, kernel_size]``.
    """
    if weight.dim() == 3:
        if weight.shape[1] != 1:
            raise ValueError(
                f"Expected depthwise weight with shape [C, 1, K], got {tuple(weight.shape)}"
            )
        weight = weight.squeeze(1)
    elif weight.dim() != 2:
        raise ValueError(f"Expected weight rank 2 or 3, got {weight.dim()}")

    kernel_size = weight.shape[-1]
    expected_input_length = output_length + kernel_size

    output = hidden_states_new[:, :, 1 : 1 + output_length] * weight[:, 0].view(1, -1, 1)
    for kernel_idx in range(1, kernel_size):
        output = output + hidden_states_new[
            :, :, 1 + kernel_idx : 1 + kernel_idx + output_length
        ] * weight[:, kernel_idx].view(1, -1, 1)
    if bias is not None:
        output = output + bias.view(1, -1, 1)
    return output


def l2norm(x: torch.FloatTensor, dim: int = -1, eps: float = 1e-6):
    # inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    # return x * inv_norm
    y = F.normalize(x, p=2, dim=dim, eps=eps)
    return y


def parallel_chunk_inverse_simple(attn: torch.Tensor, eye_matrix: torch.Tensor, chunk_size: int = 64) -> torch.Tensor:
    """Simplified (I - attn)^(-1) for alpha=1.0 (no scaling, cleaner compute graph).

    Args:
        attn: Input attention tensor with shape [B, heads, num_chunks, chunk_size, chunk_size]
        eye_matrix: Pre-computed identity matrix for TorchFX compatibility
        chunk_size: The chunk size (used to compute steps for TorchFX compatibility)
    """
    size = chunk_size
    B, heads, num_chunks, chunk_size, chunk_size = attn.shape
    flat = attn.flatten(0, -3)
    eye = eye_matrix
    r = eye + flat
    p = flat
    steps = (size - 1).bit_length()
    for i in range(steps - 1):
        p = p @ p
        r = r @ (eye + p)
    return r.reshape(B, heads, num_chunks, chunk_size, chunk_size)

def parallel_chunk_inverse_simple_inner_fp32(attn: torch.Tensor, eye_matrix: torch.Tensor, chunk_size: int = 64) -> torch.Tensor:
    """Simplified (I - attn)^(-1) for alpha=1.0 (no scaling, cleaner compute graph).

    Args:
        attn: Input attention tensor with shape [B, heads, num_chunks, chunk_size, chunk_size]
        eye_matrix: Pre-computed identity matrix for TorchFX compatibility
        chunk_size: The chunk size (used to compute steps for TorchFX compatibility)
    """
    ori_dtype = attn.dtype
    size = chunk_size
    B, heads, num_chunks, chunk_size, chunk_size = attn.shape
    flat = attn.flatten(0, -3)
    eye = eye_matrix
    r = eye + flat
    p = flat
    r,p = r.float(),p.float()
    steps = (size - 1).bit_length()
    for i in range(steps - 1):
        p = p @ p
        r = r @ (eye + p)
    return r.to(ori_dtype).reshape(B, heads, num_chunks, chunk_size, chunk_size)

def _ref(attn,chunk_size=64,dtype=torch.float32):
    ori_dtype = attn.dtype
    attn = attn.clone().to(dtype)
    for i in range(1, chunk_size):
        # # Method1 native method
        # row = attn[..., i, :i].clone() # 取 A 第 i 行、前 i 列 → (B, nvh, NC, i)
        # sub = attn[..., :i, :i].clone() # 取左上 i×i 子矩阵（已更新部分 R[:i,:i]）→ (B, nvh, NC, i, i)
        # attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2) # 等价于: R[i,:i] = A[i,:i] + A[i,:i] @ R[:i,:i]（前向替代）

        # # Method2 partial matmul
        # row = attn[..., i, :i].clone() # 取 A 第 i 行、前 i 列 → (B, nvh, NC, i)
        # sub = attn[..., :i, :i].clone() # 取左上 i×i 子矩阵（已更新部分 R[:i,:i]）→ (B, nvh, NC, i, i)
        # attn[..., i, :i] = row + (row.unsqueeze(-2) @ sub).squeeze(-2)
        # """
        # # row.unsqueeze(-2): (B, nvh, NC, 1, i) 行向量
        # # @ sub:             (B, nvh, NC, i, i) 矩阵
        # # → (B, nvh, NC, 1, i) → squeeze → (B, nvh, NC, i)
        # """
        # Method 3.  全量方法
        row = attn[...,i,:].clone()
        sub = attn[...,:,:].clone()
        if False:
            from xhquant.quantization.xh2a.quantizer._sefp_utils import sefp
            row = sefp(row,dim=-1,manbit=7,assemble=True)[0]
            sub = sefp(sub,dim=-2,manbit=7,assemble=True)[0]
        attn[...,i,:] = row + (row.unsqueeze(-2) @ sub).squeeze(-2)

        # Method 4.  全量方法
        row = attn[..., :,:].clone()
        sub = attn[...,:,:].clone()
        mask = torch.zeros_like(attn, dtype=torch.bool)
        mask[..., i, :] = 1
        attn = row @ sub * mask + row 
        # attn[...,i,:] = row + (row.unsqueeze(-2) @ sub).squeeze(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    return attn.to(ori_dtype)


def parallel_chunk_inverse_method4(
    attn: torch.Tensor,
    eye_matrix: torch.Tensor,
    chunk_row_masks: torch.Tensor,
    chunk_size: int = 64,
) -> torch.Tensor:
    """Compute (I - attn)^(-1) via forward substitution (Method 4).

    Uses row-by-row full matmul + mask to avoid the exponential intermediate
    growth of the repeated-squaring (Neumann) approach.  Each iteration only
    updates a single row, so values stay bounded.

    Args:
        attn: [B, heads, num_chunks, chunk_size, chunk_size]  strictly lower triangular
        eye_matrix: Pre-computed identity matrix (flat_batch, cs, cs)
        chunk_row_masks: Pre-computed row masks (cs, cs, cs) where
                         chunk_row_masks[i] is 1 at row i, 0 elsewhere.
        chunk_size: The chunk size
    """
    B, heads, num_chunks, cs, _ = attn.shape
    flat = attn.flatten(0, -3)

    for i in range(1, chunk_size):
        flat = flat @ flat * chunk_row_masks[i] + flat

    flat = flat + eye_matrix
    return flat.reshape(B, heads, num_chunks, cs, cs)


def parallel_chunk_inverse_method5(
    attn: torch.Tensor,
    eye_matrix: torch.Tensor,
    chunk_row_masks: torch.Tensor,
    chunk_size: int = 64,
) -> torch.Tensor:
    """Compute (I - attn)^(-1) via forward substitution (Method 4).

    Uses row-by-row full matmul + mask to avoid the exponential intermediate
    growth of the repeated-squaring (Neumann) approach.  Each iteration only
    updates a single row, so values stay bounded.

    Args:
        attn: [B, heads, num_chunks, chunk_size, chunk_size]  strictly lower triangular
        eye_matrix: Pre-computed identity matrix (flat_batch, cs, cs)
        chunk_row_masks: Pre-computed row masks (cs, cs, cs) where
                         chunk_row_masks[i] is 1 at row i, 0 elsewhere.
        chunk_size: The chunk size
    """
    B, heads, num_chunks, cs, _ = attn.shape
    flat = attn.flatten(0, -3)

    for i in range(1, chunk_size):
        flat = flat[..., i:i+1, :] @ flat * chunk_row_masks[i] + flat

    flat = flat + eye_matrix
    return flat.reshape(B, heads, num_chunks, cs, cs)


def _vp_matmul_8x8(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """VP (vector-product) matmul for 8x8 matrices.

    Equivalent to a @ b, implemented as broadcast mul + reduce-sum.
    Maps to ONNX Unsqueeze + Mul + ReduceSum (no MatMul op).

    Args:
        a: (..., 8, 8)
        b: (..., 8, 8)
    Returns:
        (..., 8, 8)
    """
    # a[..., i, k, 1] * b[..., 1, k, j] → [..., i, k, j] → sum over k (dim=-2)
    return (a.unsqueeze(-1) * b.unsqueeze(-3)).sum(-2)


def _neumann_8x8(block: torch.Tensor, eye_8: torch.Tensor) -> torch.Tensor:
    """Compute (I - block)^{-1} for 8x8 strictly lower triangular blocks.

    Uses repeated squaring (Neumann series) with VP matmul.
    Safe because:
    - A^8 = 0 (nilpotent for 8x8 strict lower tri)
    - Intermediate max ~4 for typical inputs (well within fp16 range)
    - Only 2 squaring steps needed: (I+A)(I+A²)(I+A⁴) = I+A+...+A⁷

    Args:
        block: (N, 8, 8) strictly lower triangular matrices
        eye_8: (N, 8, 8) identity matrices
    """
    p = block
    r = eye_8 + p
    p = _vp_matmul_8x8(p, p)          # A²
    r = _vp_matmul_8x8(r, eye_8 + p)  # (I+A)(I+A²) = I+A+A²+A³
    p = _vp_matmul_8x8(p, p)          # A⁴
    r = _vp_matmul_8x8(r, eye_8 + p)  # (I+A+A²+A³)(I+A⁴) = I+A+...+A⁷
    return r


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
    B, heads, num_chunks, cs, _ = attn.shape
    flat = attn.flatten(0, -3)  # (BHN, 64, 64)
    b = block_size  # 8
    nb = chunk_size // b    # 8

    # --- Step 1: Batch compute all 8 diagonal block inverses ---
    # Stack diagonal blocks: (BHN, 8, 8, 8) → (BHN*8, 8, 8)
    diag_blocks = torch.stack(
        [flat[:, I * b:(I + 1) * b, I * b:(I + 1) * b] for I in range(nb)],
        dim=1,
    )  # (BHN, 8, 8, 8)
    batched = diag_blocks.reshape(-1, b, b)  # (BHN*8, 8, 8)
    diag_invs_batched = _neumann_8x8(batched, eye_8_batched)  # (BHN*8, 8, 8)
    diag_invs_all = diag_invs_batched.reshape(-1, nb, b, b)  # (BHN, 8, 8, 8)

    # --- Step 2: Block forward substitution (8 serial steps, no in-place) ---
    # Build R incrementally as a growing upper-left block.
    # R_upper starts as the first diagonal block and grows each step.
    R_upper = diag_invs_all[:, 0]  # (BHN, 8, 8) — R[0:8, 0:8]

    for I in range(1, nb):
        s = I * b  # start index of current block row

        # A[s:s+8, 0:s] @ R_upper → off-diagonal contribution
        # shapes: (BHN, 8, s) @ (BHN, s, s) → (BHN, 8, s)
        T = torch.matmul(flat[:, s:s + b, :s], R_upper)

        # off_diag = diag_inv[I] @ T
        # shapes: (BHN, 8, 8) @ (BHN, 8, s) → (BHN, 8, s)
        diag_inv_I = diag_invs_all[:, I]  # (BHN, 8, 8)
        off_diag = torch.matmul(diag_inv_I, T)  # (BHN, 8, s)

        # Extend R_upper from (BHN, s, s) to (BHN, s+8, s+8)
        # Old rows get 8 zero-columns appended on the right via Pad
        old_rows = F.pad(R_upper, (0, b))  # (BHN, s, s+8)

        # New rows: [off_diag | diag_inv_I]
        new_rows = torch.cat([off_diag, diag_inv_I], dim=-1)  # (BHN, 8, s+8)

        # Stack vertically
        R_upper = torch.cat([old_rows, new_rows], dim=-2)  # (BHN, s+8, s+8)

    # R_upper is now (BHN, 64, 64) — the full R
    return R_upper.reshape(B, heads, num_chunks, cs, cs)


def torch_chunk_gated_delta_rule(
    query,
    key,
    value,
    g,
    beta,
    mask_qkv,
    chunk_size=64,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
    chunk_mask_incl=None,
    chunk_mask_strict=None,
    chunk_eye_matrix=None,
    chunk_eye_8_batched=None,
    cumsum_matrix=None,
    input_sequence_length=None,
    num_heads=None,
    k_head_dim=None,
    v_head_dim=None,
    batch_size=1,
    scale=None,
    chunk_row_masks=None,
    cumsum_matmul=None,
):
    """
    Chunked gated delta rule attention.
    
    Args:
        input_sequence_length: The concrete sequence length from _update_cfg.
        num_heads: Number of attention heads (from _setup).
        k_head_dim: Key head dimension (from _setup).
        v_head_dim: Value head dimension (from _setup).
        batch_size: Batch size (from _setup).
        scale: Pre-computed scale factor 1/sqrt(k_head_dim) (from _setup).
        cumsum_matrix: Pre-computed lower triangular ones matrix for cumsum via matmul (from _setup).
        cumsum_matmul: MatMul sub-module for cumsum via matmul (from _setup). When provided,
            uses cumsum_matmul(g, cumsum_matrix) instead of g.cumsum(dim=-1) for hardware compatibility.
    """
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query = query * mask_qkv
    key = key * mask_qkv
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous() for x in (query, key, value, beta, g)
    ]

    # Use concrete values from _setup/_update_cfg for TorchFX tracing compatibility
    sequence_length = input_sequence_length
    
    # Optimization: skip padding if sequence_length is already a multiple of chunk_size
    # (common case for ONNX export where input_sequence_length is typically 64, 128, 256, etc.)
    needs_pad = (sequence_length % chunk_size) != 0
    if needs_pad:
        pad_size = chunk_size - (sequence_length % chunk_size)
        total_sequence_length = sequence_length + pad_size
        query = F.pad(query, (0, 0, 0, pad_size))
        key = F.pad(key, (0, 0, 0, pad_size))
        value = F.pad(value, (0, 0, 0, pad_size))
        beta = F.pad(beta, (0, pad_size))
        g = F.pad(g, (0, pad_size))
    else:
        total_sequence_length = sequence_length
    
    num_chunks = total_sequence_length // chunk_size
    
    # Use pre-computed scale from _setup
    query = query * scale

    beta = beta.unsqueeze(-1)
    v_beta = value * beta
    k_beta = key * beta
    # reshape to chunks
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], num_chunks, chunk_size, x.shape[-1]) for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], num_chunks, chunk_size)
    
    # Use pre-computed masks if provided, otherwise create them
    if chunk_mask_incl is not None:
        mask_incl = chunk_mask_incl.to(query.dtype)
    else:
        mask_incl = _lower_triangular_mask(chunk_size, query, dtype=query.dtype, strict=False)
    if chunk_mask_strict is not None:
        mask_strict = chunk_mask_strict.to(query.dtype)
    else:
        mask_strict = _lower_triangular_mask(chunk_size, query, dtype=query.dtype, strict=True)

    # chunk decay - use matmul instead of cumsum for better hardware compatibility
    # cumsum via upper triangular matrix multiplication: g @ cumsum_matrix
    if cumsum_matmul is not None and cumsum_matrix is not None:
        g = cumsum_matmul(g, cumsum_matrix.to(g.dtype))
    else:
        g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)) * mask_incl).exp() * mask_incl
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask) * mask_strict
    chunk_eye_matrix = chunk_eye_matrix.to(attn.dtype)
    chunk_row_masks = chunk_row_masks.to(attn.dtype)
    # attn = parallel_chunk_inverse_method5(
    #     attn,
    #     eye_matrix=chunk_eye_matrix,
    #     chunk_row_masks=chunk_row_masks,
    #     chunk_size=chunk_size,
    # )
    attn = parallel_chunk_inverse_block(
        attn,
        eye_matrix=chunk_eye_matrix,
        eye_8_batched=chunk_eye_8_batched.to(attn.dtype),
        chunk_size=chunk_size,
        block_size=8,
    )
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
        if initial_state is None
        else initial_state.to(value)
    )

    core_attn_chunks = []
    # Use concrete num_chunks for TorchFX tracing compatibility
    for i in range(num_chunks):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]) * mask_incl
        v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_chunks.append(attn_inter + attn @ v_new)
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = torch.concat(core_attn_chunks, dim=2)
    core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1])
    
    # Optimization: skip slicing if no padding was applied
    if needs_pad:
        core_attn_out = core_attn_out[:, :, :sequence_length]
    
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


def torch_recurrent_gated_delta_rule(
    query,
    key,
    value,
    g,
    beta,
    mask_qkv,
    initial_state,
    output_final_state,
    use_qk_l2norm_in_kernel=False,
    num_heads=None,
    k_head_dim=None,
    v_head_dim=None,
    batch_size=1,
    scale=None,
    sequence_length=1,
):
    """
    Recurrent gated delta rule attention.
    
    Args:
        num_heads: Number of attention heads (from _setup).
        k_head_dim: Key head dimension (from _setup).
        v_head_dim: Value head dimension (from _setup).
        batch_size: Batch size (from _setup).
        scale: Pre-computed scale factor 1/sqrt(k_head_dim) (from _setup).
        sequence_length: Sequence length (typically 1 for decode phase).
    """
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query = query * mask_qkv
    key = key * mask_qkv
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous() for x in (query, key, value, beta, g)
    ]

    # Use pre-computed values from _setup for TorchFX compatibility
    query = query * scale

    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
        if initial_state is None
        else initial_state.to(value)
    )
    core_attn_steps = []
    for i in range(sequence_length):
        q_t = query[:, :, i]
        k_t = key[:, :, i]
        v_t = value[:, :, i]
        g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, :, i].unsqueeze(-1)

        last_recurrent_state = last_recurrent_state * g_t
        kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        core_attn_steps.append((last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2))

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = torch.stack(core_attn_steps, dim=2)
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


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
class _Qwen3NextRMSNorm(DynamicModule):
    def forward(self, hidden_states):
        return self.norm(hidden_states)

    def _setup(self, cfg: Optional[Dict] = None):
        hidden_size = self.weight.shape[0]
        self.norm = RMSNorm(hidden_size, self.eps)
        with torch.no_grad():
            self.norm.weight.copy_(self.weight + 1.0)
        return self


_rms_norm_gated_registry = {Qwen3NextRMSNormGated: "Qwen3NextRMSNormGated"}
if FusedRMSNormGated is not None:
    _rms_norm_gated_registry[FusedRMSNormGated] = "FusedRMSNormGated_Qwen3Next"


@XHLLM_TRACEABLE_MODULES.register_module(_rms_norm_gated_registry)
class _Qwen3NextRMSNormGated(DynamicModule):
    def forward(self, hidden_states, gate=None):
        hidden_states = self.norm(hidden_states)
        return hidden_states * F.silu(gate)

    def _setup(self, cfg: Optional[Dict] = None):
        hidden_size = self.weight.shape[0]
        eps = getattr(self, "variance_epsilon", getattr(self, "eps", 1e-6))
        self.norm = RMSNorm(hidden_size, eps)
        with torch.no_grad():
            self.norm.weight.copy_(self.weight)
        return self


@XHLLM_TRACEABLE_MODULES.register_module(
    {
        Qwen3NextAttention: "Qwen3NextAttention",
    }
)
class _Qwen3NextAttention(DynamicModule):
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

        rotary_dim = int(self.config.partial_rotary_factor * self.head_dim)
        self.rotate_half_slice_1 = xhnn.Slice([0], [rotary_dim // 2], [3], [1])
        self.rotate_half_slice_2 = xhnn.Slice([rotary_dim // 2], [sys.maxsize], [3], [1])
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


def _get_activation_name(act_fn) -> str:
    if hasattr(act_fn, "_get_name"):
        return act_fn._get_name().lower()
    return act_fn.__class__.__name__.lower()


def _init_moe_linear_storage(moeblock: MoeBlock, experts, linear_name: str, device: torch.device):
    linear = getattr(experts[0], linear_name)
    weight = linear.weight

    setattr(
        moeblock,
        f"expert_{linear_name}_weight",
        nn.Parameter(
            torch.zeros(
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
            torch.zeros(
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
                torch.zeros(
                    len(experts),
                    bias.shape[0],
                    device=device,
                    dtype=bias.dtype,
                )
            ),
        )
    else:
        setattr(moeblock, f"expert_{linear_name}_bias", None)


def _copy_defused_expert_weights_to_moeblock(moeblock: MoeBlock, experts, device: torch.device):
    with torch.no_grad():
        for expert_idx, expert in enumerate(experts):
            for linear_name in ("gate_proj", "up_proj", "down_proj"):
                linear = getattr(expert, linear_name)
                # For GPTQ quantized layers (e.g. TorchQuantLinear), .weight
                # returns a metadata shim without .data; use dequantize_weight().
                # dequantize_weight() returns (in_features, out_features) while
                # MoeBlock/F.linear expects (out_features, in_features), so transpose.
                if hasattr(linear, "dequantize_weight"):
                    weight_data = linear.dequantize_weight().t().to(device)
                else:
                    weight_data = linear.weight.data.to(device)
                getattr(moeblock, f"expert_{linear_name}_weight")[expert_idx].copy_(
                    weight_data
                )

                quant_weight = getattr(linear, "quant_weight", None)
                if (
                    quant_weight is not None
                    and hasattr(moeblock, f"expert_{linear_name}_quant_weight")
                ):
                    getattr(moeblock, f"expert_{linear_name}_quant_weight")[expert_idx].copy_(
                        quant_weight.data.to(device)
                    )

                bias = linear.bias
                if bias is not None and getattr(moeblock, f"expert_{linear_name}_bias") is not None:
                    getattr(moeblock, f"expert_{linear_name}_bias")[expert_idx].copy_(
                        bias.data.to(device)
                    )



@XHLLM_TRACEABLE_MODULES.register_module(
    {
        Qwen3NextGatedDeltaNet: "Qwen3NextGatedDeltaNet",
    }
)
class _Qwen3NextGatedDeltaNet(DynamicModule):
    def fix_query_key_value_ordering(self, mixed_qkvz, mixed_ba):
        bsz, seq_len, _ = mixed_qkvz.size()
        qkvz_last_dim = 2 * self.head_k_dim + 2 * self.head_v_dim * self.num_v_heads // self.num_k_heads
        ba_last_dim = 2 * self.num_v_heads // self.num_k_heads
        mixed_qkvz = mixed_qkvz.view(bsz, seq_len, self.num_k_heads, qkvz_last_dim)
        mixed_ba = mixed_ba.view(bsz, seq_len, self.num_k_heads, ba_last_dim)
        split_arg_list_qkvz = [
            self.head_k_dim,
            self.head_k_dim,
            (self.num_v_heads // self.num_k_heads * self.head_v_dim),
            (self.num_v_heads // self.num_k_heads * self.head_v_dim),
        ]
        split_arg_list_ba = [self.num_v_heads // self.num_k_heads, self.num_v_heads // self.num_k_heads]
        query, key, value, z = torch.split(mixed_qkvz, split_arg_list_qkvz, dim=3)
        b, a = torch.split(mixed_ba, split_arg_list_ba, dim=3)
        # [b, sq, ng, np/ng * hn] -> [b, sq, np, hn]
        value = value.reshape(value.size(0), value.size(1), -1, self.head_v_dim)
        z = z.reshape(z.size(0), z.size(1), -1, self.head_v_dim)
        b = b.reshape(b.size(0), b.size(1), self.num_v_heads)
        a = a.reshape(a.size(0), a.size(1), self.num_v_heads)
        return query, key, value, z, b, a

    def _build_conv_state(self, mixed_qkv: Tensor):
        seq_len = mixed_qkv.shape[-1]
        if seq_len >= self.conv_kernel_size:
            return mixed_qkv[:, :, -self.conv_kernel_size :]
        pad = self.conv_kernel_size - seq_len
        return F.pad(mixed_qkv, (pad, 0))

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

        projected_states_qkvz = self.in_proj_qkvz(hidden_states)
        projected_states_ba = self.in_proj_ba(hidden_states)
        query, key, value, z, b, a = self.fix_query_key_value_ordering(projected_states_qkvz, projected_states_ba)
        query, key, value = (x.reshape(x.shape[0], x.shape[1], -1) for x in (query, key, value))

        mixed_qkv = torch.cat((query, key, value), dim=-1).transpose(1, 2)

        use_recurrent = self.linear_attention_mode == "recurrent"
        if self.linear_attention_mode == "auto" and current_input_length is not None:
            if isinstance(current_input_length, Tensor):
                if current_input_length.numel() == 1:
                    use_recurrent = int(current_input_length.item()) == 1
                else:
                    use_recurrent = int(current_input_length[0].item()) == 1
            else:
                use_recurrent = int(current_input_length) == 1

        assert conv_cache is not None, "conv_cache is required"
        hidden_states_new = torch.cat([conv_cache, mixed_qkv], dim=-1).to(self.conv1d.weight.dtype)
        conv_cache_out = self.conv_cache_slice(hidden_states_new, current_input_length)
        conv_out = _manual_depthwise_conv1d_tail(
            hidden_states_new,
            self.conv1d_manual_weight,
            getattr(self, "conv1d_manual_bias", None),
            self.input_sequence_length,
        )
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
        # query = query * mask_qkv
        # key = key * mask_qkv
        value = value * mask_qkv
        beta = beta * mask
        g = g * mask

        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

        if use_recurrent:
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
                sequence_length=1,  # Recurrent mode typically for decode (seq_len=1)
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
            )

        # Prepare recurrent_state output
        recurrent_state_out = last_recurrent_state if last_recurrent_state is not None else recurrent_state

        b, s, n, h = z.shape
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(b, s, n, h)
        core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1)

        # Always convert to the expected dtype for TorchFX tracing compatibility
        # (avoid using core_attn_out.dtype in control flow)
        core_attn_out = core_attn_out.to(self.out_proj.weight.dtype)
        output = self.out_proj(core_attn_out)
        output = output * mask
        # Always return cache as output (conv_cache_out, recurrent_state_out)
        return output, conv_cache_out, recurrent_state_out

    def _setup(self, cfg: Optional[Dict] = None):
        self.use_cache = cfg.use_cache
        self.linear_attention_mode = cfg.get("linear_attention_mode", "auto")
        self.linear_chunk_size = cfg.get("linear_chunk_size", 64)
        self.return_cache = cfg.get("return_cache", False)
        self.input_sequence_length = cfg.get("input_sequence_length", 256)
        self.batch_size = cfg.get("batch_size", 1)
        
        # Convert nn.Parameter to buffer for FX graph compatibility.
        # In HF model, dt_bias and A_log are nn.Parameter, but during inference
        # they are used as constants. AutoOffloadGraphModel._get_node_memory_footprint
        # calls get_buffer() on FX get_attr nodes and will raise AttributeError
        # if the attribute is still an nn.Parameter.
        if "dt_bias" in self._parameters:
            _dt_bias_data = self.dt_bias.data.clone()
            del self._parameters["dt_bias"]
            self.register_buffer("dt_bias", _dt_bias_data, persistent=False)
        if "A_log" in self._parameters:
            _a_log_data = self.A_log.data.clone()
            del self._parameters["A_log"]
            self.register_buffer("A_log", _a_log_data, persistent=False)
        
        # Pre-compute head dimensions for TorchFX tracing compatibility
        self.chunk_num_heads = self.num_v_heads
        self.chunk_k_head_dim = self.head_k_dim
        self.chunk_v_head_dim = self.head_v_dim
        target_dtype = self.in_proj_qkvz.weight.dtype
        self.register_buffer(
            "chunk_scale",
            torch.tensor(
                1.0 / (self.head_k_dim ** 0.5),
                dtype=target_dtype,
                device=self.A_log.device,
            ),
            persistent=False,
        )
        
        a_log_exp = (-self.A_log.exp()).to(device=self.A_log.device, dtype=self.dt_bias.dtype)
        self.register_buffer("A_log_exp", a_log_exp, persistent=False)
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

        # Conv cache slice must be ready right after wrapping.
        # WRAPED/PyTorch eval path may run without calling _update_cfg first.
        self.conv_cache_slice = xhnn.DynamicSlice([self.conv_kernel_size], [2], [1])
        
        # Pre-create triangular masks for TorchFX tracing compatibility
        chunk_size = self.linear_chunk_size
        idx = torch.arange(chunk_size, dtype=torch.long, device=self.A_log.device)
        mask_incl = (idx[:, None] >= idx[None, :]).to(target_dtype)
        mask_strict = (idx[:, None] > idx[None, :]).to(target_dtype)
        # Create eye_matrix with final shape (batch*heads*num_chunks, chunk_size, chunk_size)
        # to avoid any operations in forward (unsqueeze, expand, to)
        # Use ceiling division to handle cases where input_sequence_length is not divisible by chunk_size
        num_chunks = (self.input_sequence_length + chunk_size - 1) // chunk_size
        flat_batch_size = self.batch_size * self.num_v_heads * num_chunks
        eye_matrix = (
            torch.eye(chunk_size, dtype=target_dtype, device=self.A_log.device)
            .unsqueeze(0)
            .expand(flat_batch_size, chunk_size, chunk_size)
            .contiguous()
        )
        # Create upper triangular ones matrix for cumsum via matmul (hardware-friendly)
        # For g @ cumsum_matrix to work correctly, we need upper triangular (transpose of lower)
        cumsum_matrix = torch.triu(
            torch.ones(chunk_size, chunk_size, dtype=target_dtype, device=self.A_log.device)
        )
        self.register_buffer("chunk_mask_incl", mask_incl, persistent=False)
        self.register_buffer("chunk_mask_strict", mask_strict, persistent=False)
        self.register_buffer("chunk_eye_matrix", eye_matrix, persistent=False)
        self.register_buffer("chunk_cumsum_matrix", cumsum_matrix, persistent=False)

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

        # Create MatMul sub-module for cumsum operation.
        # This becomes a call_module node in the FX graph, which can be individually
        # configured for quantization (e.g., left input sefp16) via node.meta["quant_config"].
        # During quantization, MatMul → QMatMul, and the quant_config is applied.
        self.cumsum_matmul = MatMul()

        # Pre-compute row masks for Method 4 forward substitution
        # chunk_row_masks[i] is a (cs, cs) matrix with 1.0 at row i, 0 elsewhere
        chunk_row_masks = torch.eye(
            chunk_size, dtype=target_dtype, device=self.A_log.device
        ).unsqueeze(-1).expand(chunk_size, chunk_size, chunk_size).contiguous()
        self.register_buffer("chunk_row_masks", chunk_row_masks, persistent=False)
        
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
        
        # Update eye_matrix when input_sequence_length or batch_size changes
        chunk_size = self.linear_chunk_size
        # Use ceiling division to handle cases where input_sequence_length is not divisible by chunk_size
        num_chunks = (self.input_sequence_length + chunk_size - 1) // chunk_size
        flat_batch_size = self.batch_size * self.num_v_heads * num_chunks
        target_dtype = self.chunk_scale.dtype if hasattr(self, "chunk_scale") else self.in_proj_qkvz.weight.dtype
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

        self.conv_cache_slice = xhnn.DynamicSlice([self.conv_kernel_size], [2], [1])


@XHLLM_TRACEABLE_MODULES.register_module(
    {
        Qwen3NextSparseMoeBlock: "Qwen3NextSparseMoeBlock",
    }
)
class _Qwen3NextSparseMoeBlock(DynamicModule):
    def forward(self, hidden_states):
        # Unconditionally cast to expert weight dtype to prevent Triton kernel
        # dtype mismatch (fp16 vs bf16) during accelerate auto_offload.
        # .to() is a no-op when dtypes already match, and avoids TorchFX trace errors.
        if hasattr(self, 'moeblock') and hasattr(self.moeblock, 'expert_gate_proj_weight'):
            hidden_states = hidden_states.to(self.moeblock.expert_gate_proj_weight.dtype)
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
            gate_up = experts.gate_up_proj.data.to(self.device)
            gate_proj_weight = gate_up[:, :intermediate_dim, :].contiguous()
            up_proj_weight = gate_up[:, intermediate_dim:, :].contiguous()
            down_proj_weight = experts.down_proj.data.to(self.device).contiguous()

            self.moeblock.expert_gate_proj_weight = nn.Parameter(gate_proj_weight)
            self.moeblock.expert_gate_proj_bias = None
            self.moeblock.expert_up_proj_weight = nn.Parameter(up_proj_weight)
            self.moeblock.expert_up_proj_bias = None
            self.moeblock.expert_down_proj_weight = nn.Parameter(down_proj_weight)
            self.moeblock.expert_down_proj_bias = None

            # Release original packed experts to free memory
            del self.experts
        elif (
            len(experts) > 0
            and hasattr(experts[0], "gate_proj")
            and hasattr(experts[0], "up_proj")
            and hasattr(experts[0], "down_proj")
        ):
            for linear_name in ("gate_proj", "up_proj", "down_proj"):
                _init_moe_linear_storage(self.moeblock, experts, linear_name, self.device)

            _copy_defused_expert_weights_to_moeblock(self.moeblock, experts, self.device)

            # Release expert modules after packing to reduce memory.
            self.experts = nn.ModuleList()
        else:
            raise RuntimeError(
                f"Unsupported Qwen3.5-MoE experts structure: {type(experts)}"
            )
        torch.cuda.empty_cache()

        return self


@XHLLM_TRACEABLE_MODULES.register_module(
    {
        Qwen3NextDecoderLayer: "Qwen3NextDecoderLayer",
    }
)
class _Qwen3NextDecoderLayer(DynamicModule):
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
        """
        Returns:
            For linear_attention: (hidden_states, conv_cache_out, recurrent_state_out)
            For full_attention: hidden_states only
        """
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
            # linear_attn returns (output, conv_cache_out, recurrent_state_out)
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
            hidden_states, _ = hidden_states
        hidden_states = residual + hidden_states

        # Return different outputs based on layer type
        if self.layer_type == "linear_attention":
            return hidden_states, conv_cache_out, recurrent_state_out
        else:
            return hidden_states

    def _setup(self, cfg: Optional[Dict] = None):
        return self


@XHLLM_TRACEABLE_MODULES.register_module(
    {
        Qwen3NextModel: "Qwen3NextModel",
    }
)
class _Qwen3NextModel(DynamicModule):
    def forward(
        self,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_seq_length: Optional[Tensor] = None,
        current_input_length: Optional[Tensor] = None,
        linear_attn_mask: Optional[Tensor] = None,
        past_key_cache: Optional[List[Tensor]] = None,
        past_value_cache: Optional[List[Tensor]] = None,
        past_conv_cache: Optional[List[Tensor]] = None,
        past_recurrent_state: Optional[List[Tensor]] = None,
    ) -> Tuple[Tensor, Optional[List[Tensor]], Optional[List[Tensor]]]:
        """
        Args:
            past_key_cache: List of KV caches for full_attention layers only
            past_value_cache: List of KV caches for full_attention layers only
            past_conv_cache: List of conv caches for linear_attention layers only
            past_recurrent_state: List of recurrent states for linear_attention layers only
            
        Returns:
            hidden_states: output hidden states
            conv_cache_out_list: list of updated conv caches (only for linear_attention layers)
            recurrent_state_out_list: list of updated recurrent states (only for linear_attention layers)
        """
        hidden_states = inputs_embeds

        cos = self.cos_slice(self.rotary_emb.cos_cached, past_seq_length)
        sin = self.sin_slice(self.rotary_emb.sin_cached, past_seq_length)
        position_embeddings = (cos, sin)

        # Collect output caches (only for linear_attention layers)
        conv_cache_out_list = []
        recurrent_state_out_list = []
        
        # Track indices for each cache type
        full_attn_cache_idx = 0
        linear_attn_cache_idx = 0

        for idx, decoder_layer in enumerate(self.layers):
            # print("Processing layer index:", idx)
            layer_type = self.layer_types[idx]
            
            if self.use_cache:
                if layer_type == "full_attention":
                    # Get cache from full_attention cache list
                    _past_k_cache = past_key_cache[full_attn_cache_idx] if past_key_cache is not None else None
                    _past_v_cache = past_value_cache[full_attn_cache_idx] if past_value_cache is not None else None
                    _past_conv_cache = None
                    _past_recurrent_state = None
                    full_attn_cache_idx += 1
                else:  # linear_attention
                    # Get cache from linear_attention cache list
                    _past_k_cache = None
                    _past_v_cache = None
                    _past_conv_cache = past_conv_cache[linear_attn_cache_idx] if past_conv_cache is not None else None
                    _past_recurrent_state = past_recurrent_state[linear_attn_cache_idx] if past_recurrent_state is not None else None
                    linear_attn_cache_idx += 1
            else:
                _past_k_cache = None
                _past_v_cache = None
                if layer_type == "linear_attention":
                    _past_conv_cache = (
                        past_conv_cache[linear_attn_cache_idx]
                        if past_conv_cache is not None
                        else None
                    )
                    _past_recurrent_state = (
                        past_recurrent_state[linear_attn_cache_idx]
                        if past_recurrent_state is not None
                        else None
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
                # Collect caches from linear_attention layers
                conv_cache_out_list.append(conv_cache_out)
                recurrent_state_out_list.append(recurrent_state_out)
            else:  # full_attention
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
                
            if self.only_first_block:
                break

        if self.num_logits_to_keep == 0:
            pass
        else:
            hidden_states = self.llm_gather(hidden_states, current_input_length - 1)
        hidden_states = self.norm(hidden_states)

        return hidden_states, conv_cache_out_list, recurrent_state_out_list

    def _setup_cos_sin_embeding(self):
        self.rotary_emb.cos_cached
        self.rotary_emb.sin_cached

    def _setup(self, cfg: Optional[Dict] = None):
        self.batch_size = cfg.get("batch_size", 1)
        self.only_first_block = cfg.get("only_first_block", False)
        self.num_logits_to_keep = cfg.num_logits_to_keep
        assert self.num_logits_to_keep in [0, 1]
        
        # Get layer types from config to distinguish full_attention and linear_attention layers
        self.layer_types = self.config.layer_types
        self.num_full_attention_layers = sum(1 for t in self.layer_types if t == "full_attention")
        self.num_linear_attention_layers = sum(1 for t in self.layer_types if t == "linear_attention")

        input_seq_len = cfg.input_sequence_length
        self.slice = xhnn.Slice([0], [input_seq_len], [1], [1])

        self.llm_gather = xhnn.BatchGather(1)
        self.llm_gather.update_offset_indices(self.batch_size, input_seq_len)

        def _llm_gather_update_cfg(self: xhnn.BatchGather, cfg: Optional[Dict] = None):
            self.num_logits_to_keep = cfg.num_logits_to_keep
            input_seq_len = cfg.input_sequence_length
            batch_size = cfg.get("batch_size", 1)
            self.update_offset_indices(batch_size, input_seq_len)

        self.llm_gather._update_cfg = types.MethodType(_llm_gather_update_cfg, self.llm_gather)

        def _slice_update_cfg(self, cfg: Optional[Dict] = None):
            input_seq_len = cfg.input_sequence_length
            self.ends = [input_seq_len]

        self.slice._update_cfg = types.MethodType(_slice_update_cfg, self.slice)
        self.use_cache = cfg.use_cache

        self.sin_slice = xhnn.DynamicSlice([input_seq_len], [2], [1])
        self.cos_slice = xhnn.DynamicSlice([input_seq_len], [2], [1])

        def _sin_cos_slice_update_cfg(self, cfg: Optional[Dict] = None):
            input_seq_len = cfg.input_sequence_length
            self.valid_length = [input_seq_len]

        self.sin_slice._update_cfg = types.MethodType(_sin_cos_slice_update_cfg, self.sin_slice)
        self.cos_slice._update_cfg = types.MethodType(_sin_cos_slice_update_cfg, self.cos_slice)

        if not hasattr(self.rotary_emb, "cos_cached"):
            self.rotary_emb.setup_after_callback = self._setup_cos_sin_embeding
        else:
            self._setup_cos_sin_embeding()

        # Mark specific linear_attention layers for alpha scaling.
        # This sets a lightweight config dict on the raw GDN module BEFORE it is converted;
        # _Qwen3NextGatedDeltaNet._setup will read it and create the actual buffers.
        alpha_scaling_layers = cfg.get("alpha_scaling_layers", [8, 20])
        chunk_inverse_alpha = cfg.get("chunk_inverse_alpha", 0.5)
        for idx, decoder_layer in enumerate(self.layers):
            if self.layer_types[idx] == "linear_attention" and idx in alpha_scaling_layers:
                gdn = decoder_layer.linear_attn
                gdn._alpha_scaling_config = {'alpha': chunk_inverse_alpha}

        return self


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
        hidden_states, conv_cache_out_list, recurrent_state_out_list = self.model(
            inputs_embeds=inputs_embeds,
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

    def _setup(self, cfg: Optional[Dict] = None):
        return self


def register_wrap_modules():
    pass
