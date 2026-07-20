from typing import Optional

import torch
import torch.nn.functional as F
from torch.fx import Proxy

from xhquant.core import CacheTensor
from xhquant.ops.xh.xh_pragram_op import xh_pragma_fx


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
    p = _vp_matmul_8x8(p, p)  # A²
    r = _vp_matmul_8x8(r, eye_8 + p)  # (I+A)(I+A²) = I+A+A²+A³
    p = _vp_matmul_8x8(p, p)  # A⁴
    r = _vp_matmul_8x8(r, eye_8 + p)  # (I+A+A²+A³)(I+A⁴) = I+A+...+A⁷
    return r


def l2norm(x: torch.FloatTensor, dim: int = -1, eps: float = 1e-6):
    # inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    # return x * inv_norm
    y = F.normalize(x, p=2, dim=dim, eps=eps)
    return y


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
    block_tri_inverse_op=None,
    chunk_scan_op=None,
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
    query, key, value, beta, g = [x.transpose(1, 2).contiguous() for x in (query, key, value, beta, g)]

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
        x.reshape(x.shape[0], x.shape[1], num_chunks, chunk_size, x.shape[-1])
        for x in (query, key, value, k_beta, v_beta)
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
        xh_pragma_fx(
            g,
            {
                "type": "quanted",
                "action": "start",
                "qconfig": {
                    "act_schema": {
                        "bits": 16,
                        "fp_mode": "sefp",
                    },
                },
            },
        )
        g = cumsum_matmul(g, cumsum_matrix.to(g.dtype))
        xh_pragma_fx(
            g,
            {
                "type": "quanted",
                "action": "end",
            },
        )
    else:
        g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)) * mask_incl).exp() * mask_incl
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask) * mask_strict
    chunk_eye_matrix = chunk_eye_matrix.to(attn.dtype)
    chunk_row_masks = chunk_row_masks.to(attn.dtype)
    if block_tri_inverse_op is not None:
        # QTL-339: externalize eye buffer to op input (matches xhquant T2 signature).
        attn = block_tri_inverse_op(attn, chunk_eye_8_batched.to(attn.dtype))
    else:
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

    if chunk_scan_op is not None:
        chunk_scan_state_is_cache = bool(getattr(chunk_scan_op, "state_is_cache", False))
        if chunk_scan_state_is_cache and not isinstance(last_recurrent_state, (CacheTensor, Proxy)):
            last_recurrent_state = CacheTensor(last_recurrent_state)
        chunk_scan_result = chunk_scan_op(
            query,
            key,
            value,
            k_cumdecay,
            decay_mask,
            mask_incl,
            g,
            last_recurrent_state,
        )
        if chunk_scan_state_is_cache:
            core_attn_out = chunk_scan_result
        else:
            core_attn_out, last_recurrent_state = chunk_scan_result
    else:
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
        core_attn_out = torch.stack(core_attn_chunks, dim=2)

    if not output_final_state:
        last_recurrent_state = None
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
    recurrent_scan_op=None,
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
    query, key, value, beta, g = [x.transpose(1, 2).contiguous() for x in (query, key, value, beta, g)]

    # Use pre-computed values from _setup for TorchFX compatibility
    query = query * scale

    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
        if initial_state is None
        else initial_state.to(value)
    )

    if recurrent_scan_op is not None:
        recurrent_scan_returns_state = bool(getattr(recurrent_scan_op, "returns_state", False))
        if recurrent_scan_returns_state:
            core_attn_steps = []
            for i in range(sequence_length):
                q_t = query[:, :, i]
                last_recurrent_state = recurrent_scan_op(
                    query[:, :, i : i + 1],
                    key[:, :, i : i + 1],
                    value[:, :, i : i + 1],
                    g[:, :, i : i + 1],
                    beta[:, :, i : i + 1],
                    last_recurrent_state,
                )
                core_attn_steps.append((last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2))
            core_attn_out = torch.stack(core_attn_steps, dim=2)
        else:
            result = recurrent_scan_op(query, key, value, g, beta, last_recurrent_state)
            core_attn_out = result[0]
            last_recurrent_state = result[1]
    else:
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
        core_attn_out = torch.stack(core_attn_steps, dim=2)

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state
