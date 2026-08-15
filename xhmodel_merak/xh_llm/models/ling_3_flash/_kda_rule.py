"""Kimi Delta Attention reference and fused-boundary helpers.

Ling-3 uses a per-key-dimension decay gate.  This is the important semantic
difference from Qwen3.5's Gated Delta Rule (GDR), whose decay gate is scalar
per head.  The block triangular inverse is shared by both algorithms, while
the chunk scan and recurrent state decay need KDA-aware handling.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.fx import Proxy

from xhquant.core.cache_tensor import CacheTensor
from xhquant.nn.modules import GDRRecurrentScan


def kda_log_decay(
    raw_gate: Tensor,
    a_log: Tensor,
    dt_bias: Tensor,
    *,
    safe_gate: bool,
    lower_bound: float | None,
    num_heads: int | None = None,
    key_dim: int | None = None,
    decay_rate: Tensor | None = None,
    gate_bias: Tensor | None = None,
) -> Tensor:
    """Convert Ling's raw KDA gate to a log-space per-dimension decay.

    Args:
        raw_gate: ``[B, T, H, K]``.
        a_log: ``[H]``.
        dt_bias: ``[H * K]`` or ``[H, K]``.

    The safe-gate expression follows FLA's KDA contract.  It bounds the log
    decay to ``[lower_bound, 0)`` and is the path used by Ling-3-Flash.
    """

    if num_heads is None or key_dim is None:
        if isinstance(raw_gate, Proxy):
            raise ValueError("TorchFX KDA tracing requires static num_heads/key_dim")
        num_heads, key_dim = raw_gate.shape[-2:]
    bias = (
        dt_bias.reshape(1, 1, num_heads, key_dim).to(raw_gate.dtype)
        if gate_bias is None
        else gate_bias
    )
    rate = (
        a_log.exp().to(raw_gate.dtype).reshape(1, 1, num_heads, 1)
        if decay_rate is None
        else decay_rate
    )
    gate_input = raw_gate + bias
    if safe_gate:
        if lower_bound is None or lower_bound >= 0:
            raise ValueError("KDA safe_gate requires a negative lower_bound")
        return float(lower_bound) * torch.sigmoid(rate * gate_input)
    return -rate * F.softplus(gate_input)


def _l2_normalize(x: Tensor) -> Tensor:
    return F.normalize(x, p=2, dim=-1, eps=1e-6)


def recurrent_kda_reference(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    log_decay: Tensor,
    beta: Tensor,
    initial_state: Tensor | None,
    *,
    scale: float | Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Straightforward KDA recurrence used as the numerical oracle.

    Inputs use the HF/FLA layout: Q/K/V/g are ``[B, T, H, D]`` and beta is
    ``[B, T, H]``.  Q/K are L2-normalized like Ling's FLA invocation.
    """

    query = _l2_normalize(query)
    key = _l2_normalize(key)
    if scale is None:
        scale = query.shape[-1] ** -0.5
    query = query * scale
    batch, _, heads, key_dim = query.shape
    value_dim = value.shape[-1]
    state = (
        torch.zeros(batch, heads, key_dim, value_dim, dtype=value.dtype, device=value.device)
        if initial_state is None
        else initial_state.to(value)
    )
    outputs = []
    for token_idx in range(query.shape[1]):
        q_t = query[:, token_idx]
        k_t = key[:, token_idx]
        v_t = value[:, token_idx]
        beta_t = beta[:, token_idx].unsqueeze(-1)
        state = state * log_decay[:, token_idx].exp().unsqueeze(-1)
        memory = (state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - memory) * beta_t
        state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        outputs.append((state * q_t.unsqueeze(-1)).sum(dim=-2))
    return torch.stack(outputs, dim=1), state


def recurrent_kda_with_gdr_scan(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    log_decay: Tensor,
    beta: Tensor,
    initial_state: Tensor | None,
    *,
    scale: float | Tensor | None = None,
    recurrent_scan_op: Optional[nn.Module] = None,
    batch_size: int | None = None,
    sequence_length: int | None = None,
    num_heads: int | None = None,
    key_dim: int | None = None,
    value_dim: int | None = None,
) -> tuple[Tensor, Tensor]:
    """Execute exact KDA decode while reusing ``GDRRecurrentScan``.

    KDA first scales every state row by its own decay.  Once that operation is
    applied outside the fused op, the remaining delta update is exactly GDR
    with a zero scalar log gate.  This transformation is exact, not an
    approximation, and retains the existing GDR recurrent custom-op ABI.
    """

    query = _l2_normalize(query)
    key = _l2_normalize(key)
    if scale is None:
        scale = query.shape[-1] ** -0.5
    query = query * scale
    query_h = query.transpose(1, 2).contiguous()
    key_h = key.transpose(1, 2).contiguous()
    value_h = value.transpose(1, 2).contiguous()
    decay_h = log_decay.transpose(1, 2).contiguous()
    beta_h = beta.transpose(1, 2).contiguous()
    if None in (batch_size, sequence_length, num_heads, key_dim, value_dim):
        if isinstance(query, Proxy):
            raise ValueError("TorchFX KDA tracing requires static recurrent dimensions")
        batch_size, sequence_length, num_heads, key_dim = query.shape
        value_dim = value.shape[-1]
    state = (
        torch.zeros(batch_size, num_heads, key_dim, value_dim, dtype=value.dtype, device=value.device)
        if initial_state is None
        else initial_state.to(value)
    )
    scan = recurrent_scan_op or GDRRecurrentScan(sequence_length=1, output_all_states=False)
    outputs = []
    for token_idx in range(sequence_length):
        # Per-row KDA decay.  The fused GDR op receives a zero scalar gate and
        # therefore performs only the shared delta-rule update.
        state = state * decay_h[:, :, token_idx].exp().unsqueeze(-1)
        q_t = query_h[:, :, token_idx : token_idx + 1]
        state = scan(
            q_t,
            key_h[:, :, token_idx : token_idx + 1],
            value_h[:, :, token_idx : token_idx + 1],
            torch.zeros_like(beta_h[:, :, token_idx : token_idx + 1]),
            beta_h[:, :, token_idx : token_idx + 1],
            state,
        )
        outputs.append((state * q_t[:, :, 0].unsqueeze(-1)).sum(dim=-2))
    return torch.stack(outputs, dim=2).transpose(1, 2).contiguous(), state


def kda_chunk_scan_reference(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    k_cumdecay: Tensor,
    aqk: Tensor,
    mask_incl: Tensor,
    cumulative_decay: Tensor,
    initial_state: Tensor,
) -> tuple[Tensor, Tensor]:
    """KDA chunk-state scan over precomputed per-chunk intermediates.

    All tensors use ``[B, H, N, C, ...]`` layout.  Unlike GDRChunkScan,
    ``cumulative_decay`` is ``[B,H,N,C,K]`` and ``aqk`` already contains the
    per-dimension Q/K decay product.
    """

    state = initial_state
    outputs = []
    for chunk_idx in range(query.shape[2]):
        q_i = query[:, :, chunk_idx]
        k_i = key[:, :, chunk_idx]
        v_i = value[:, :, chunk_idx]
        g_i = cumulative_decay[:, :, chunk_idx]
        v_new = v_i - k_cumdecay[:, :, chunk_idx] @ state
        within_chunk = (aqk[:, :, chunk_idx] * mask_incl) @ v_new
        from_state = (q_i * g_i.exp()) @ state
        outputs.append(from_state + within_chunk)
        last_decay = g_i[:, :, -1]
        state = state * last_decay.exp().unsqueeze(-1)
        decayed_key = k_i * (last_decay.unsqueeze(-2) - g_i).exp()
        state = state + decayed_key.transpose(-1, -2) @ v_new
    return torch.stack(outputs, dim=2), state


def chunk_kda(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    log_decay: Tensor,
    beta: Tensor,
    initial_state: Tensor | None,
    *,
    chunk_size: int,
    block_tri_inverse_op: Optional[nn.Module],
    chunk_scan_op: Optional[nn.Module],
    eye_block_batched: Tensor,
    scale: float | Tensor | None = None,
    batch_size: int | None = None,
    sequence_length: int | None = None,
    num_heads: int | None = None,
    key_dim: int | None = None,
    value_dim: int | None = None,
    causal_pair_mask: Tensor | None = None,
    strict_pair_mask: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Vectorized KDA prefill using the shared GDR triangular inverse.

    This is the FLA ``naive_chunk_kda`` factorization expressed with tensor
    operations suitable for TorchFX. ``GDRBlockTriInverse`` is mathematically
    generic and reused unchanged. ``GDRChunkScan`` now accepts the KDA gate's
    final key dimension directly; scalar GDR is its broadcast ``K=1`` case.
    """

    if not isinstance(query, Proxy) and query.shape[1] % chunk_size:
        raise ValueError(
            f"KDA prefill length {query.shape[1]} must be divisible by chunk_size={chunk_size}"
        )
    input_dtype = value.dtype
    query = _l2_normalize(query)
    key = _l2_normalize(key)
    if scale is None:
        scale = query.shape[-1] ** -0.5
    query = query * scale
    if None in (batch_size, sequence_length, num_heads, key_dim, value_dim):
        if isinstance(query, Proxy):
            raise ValueError("TorchFX KDA tracing requires static chunk dimensions")
        batch_size, sequence_length, num_heads, key_dim = query.shape
        value_dim = value.shape[-1]
    num_chunks = sequence_length // chunk_size

    def as_chunks(x: Tensor, feature_dim: int) -> Tensor:
        return x.transpose(1, 2).reshape(
            batch_size,
            num_heads,
            num_chunks,
            chunk_size,
            feature_dim,
        )

    state_is_cache = bool(getattr(chunk_scan_op, "state_is_cache", False))
    # A cache-mutating scan must receive the original CacheTensor object.
    # Casting it would create a detached temporary and the final prefill state
    # would never reach decode.  The fused path therefore computes in the
    # wrapped model/cache dtype; the explicit reference path keeps float32
    # accumulation for its role as a numerical fallback.
    if state_is_cache:
        query_c = as_chunks(query, key_dim)
        key_c = as_chunks(key, key_dim)
        value_c = as_chunks(value, value_dim)
        decay_c = as_chunks(log_decay, key_dim).cumsum(dim=-2)
        beta_c = beta.transpose(1, 2).reshape(batch_size, num_heads, num_chunks, chunk_size)
    else:
        query_c = as_chunks(query.float(), key_dim)
        key_c = as_chunks(key.float(), key_dim)
        value_c = as_chunks(value.float(), value_dim)
        decay_c = as_chunks(log_decay.float(), key_dim).cumsum(dim=-2)
        beta_c = beta.transpose(1, 2).reshape(batch_size, num_heads, num_chunks, chunk_size).float()

    row_decay = decay_c.unsqueeze(-2)
    column_decay = decay_c.unsqueeze(-3)
    if causal_pair_mask is None:
        if isinstance(decay_c, Proxy):
            raise ValueError("TorchFX KDA tracing requires a precomputed causal_pair_mask")
        causal_pair_mask = torch.tril(
            torch.ones(chunk_size, chunk_size, dtype=decay_c.dtype, device=decay_c.device)
        )
    # Only causal (row >= column) decay is consumed.  Mask the log-domain
    # difference before exp as well as after it: upper-triangle differences
    # are positive and can overflow fp16 before a later ``* 0`` turns them
    # into NaNs.
    pair_decay = (
        (row_decay - column_decay) * causal_pair_mask.unsqueeze(-1)
    ).exp() * causal_pair_mask.unsqueeze(-1)
    # [B,H,N,row,col,K] -> [B,H,N,row,col]
    pair_kk = (
        key_c.unsqueeze(-2)
        * key_c.unsqueeze(-3)
        * pair_decay
    ).sum(dim=-1)
    if strict_pair_mask is None:
        if isinstance(pair_kk, Proxy):
            raise ValueError("TorchFX KDA tracing requires a precomputed strict_pair_mask")
        strict_pair_mask = torch.tril(
            torch.ones(chunk_size, chunk_size, dtype=pair_kk.dtype, device=pair_kk.device),
            diagonal=-1,
        )
    inclusive = causal_pair_mask
    triangular = -(pair_kk * beta_c.unsqueeze(-1)) * strict_pair_mask
    if block_tri_inverse_op is None:
        # Small/test fallback.  Production enables GDRBlockTriInverse by
        # default and avoids materializing a generic inverse op in HMONNX.
        eye = torch.eye(chunk_size, dtype=triangular.dtype, device=triangular.device)
        inverse = torch.linalg.inv((eye - triangular).float()).to(triangular.dtype)
    else:
        eye = eye_block_batched if isinstance(triangular, Proxy) else eye_block_batched.to(triangular.dtype)
        inverse = block_tri_inverse_op(triangular, eye)

    v_beta = value_c * beta_c.unsqueeze(-1)
    k_beta = key_c * beta_c.unsqueeze(-1)
    u = inverse @ v_beta
    w = inverse @ (k_beta * decay_c.exp())

    state = (
        torch.zeros(
            batch_size,
            num_heads,
            key_dim,
            value_dim,
            device=value.device,
            dtype=value_c.dtype if state_is_cache else torch.float32,
        )
        if initial_state is None
        else initial_state if state_is_cache else initial_state.float()
    )
    if chunk_scan_op is None:
        pair_qk = (
            query_c.unsqueeze(-2)
            * key_c.unsqueeze(-3)
            * pair_decay
        ).sum(dim=-1) * inclusive
        output, state = kda_chunk_scan_reference(
            query_c,
            key_c,
            u,
            w,
            pair_qk,
            inclusive,
            decay_c,
            state,
        )
    else:
        if state_is_cache and not isinstance(state, (CacheTensor, Proxy)):
            state = CacheTensor(state)
        result = chunk_scan_op(
            query_c,
            key_c,
            u,
            w,
            pair_decay,
            inclusive,
            decay_c,
            state,
        )
        if state_is_cache:
            output = result
        else:
            output, state = result
    output = output.permute(0, 2, 3, 1, 4).reshape(
        batch_size,
        sequence_length,
        num_heads,
        value_dim,
    )
    return output.to(input_dtype), state


__all__ = [
    "chunk_kda",
    "kda_chunk_scan_reference",
    "kda_log_decay",
    "recurrent_kda_reference",
    "recurrent_kda_with_gdr_scan",
]
