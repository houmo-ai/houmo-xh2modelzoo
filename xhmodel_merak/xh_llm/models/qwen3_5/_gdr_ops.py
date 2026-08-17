# -*- coding: utf-8 -*-
# Copyright 2025 HOUMO AI. All rights reserved.
#
# GDR (Gated Delta Rule) custom operator modules for xhquant FX tracing.
# These modules are registered as FX leaf nodes so the tracer treats them
# as opaque ops, producing a single node in the graph.

import os as _os
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F


# QTL-332: route GDRBlockTriInverse through the xhquant first-class custom op
# (xh::GDRBlockTriInverse). The legacy in-file implementation is preserved
# below as ``LegacyGDRBlockTriInverse`` for regression/diff testing; set
# ``XHQUANT_GDR_USE_LEGACY=1`` in the environment to force it.
try:
    from xhquant.nn.modules import GDRBlockTriInverse as _XHQuantGDRBlockTriInverse

    _HAS_XHQUANT_GDR = True
except ImportError:  # pragma: no cover - xhquant always installed in CI
    _HAS_XHQUANT_GDR = False

# QTL-333: route GDRChunkScan through the xhquant first-class custom op
# (xh::GDRChunkScan). The legacy in-file implementation is preserved below as
# ``LegacyGDRChunkScan``; set ``XHQUANT_GDR_USE_LEGACY=1`` to force it.
try:
    from xhquant.nn.modules import GDRChunkScan as _XHQuantGDRChunkScan

    _HAS_XHQUANT_GDR_CHUNK_SCAN = True
except ImportError:  # pragma: no cover - xhquant always installed in CI
    _HAS_XHQUANT_GDR_CHUNK_SCAN = False

# QTL-334: route GDRRecurrentScan through the xhquant first-class custom op
# (xh::GDRRecurrentScan). The legacy in-file implementation is preserved below
# as ``LegacyGDRRecurrentScan``; set ``XHQUANT_GDR_USE_LEGACY=1`` to force it
# (also enables the variable-length ``output_all_states=True`` path used by
# speculative decoding verification).
try:
    from xhquant.nn.modules import GDRRecurrentScan as _XHQuantGDRRecurrentScan

    _HAS_XHQUANT_GDR_RECURRENT_SCAN = True
except ImportError:  # pragma: no cover - xhquant always installed in CI
    _HAS_XHQUANT_GDR_RECURRENT_SCAN = False


def _vp_matmul_8x8(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a.unsqueeze(-1) * b.unsqueeze(-3)).sum(-2)


def _neumann_8x8(block: torch.Tensor, eye_8: torch.Tensor) -> torch.Tensor:
    p = block
    r = eye_8 + p
    p = _vp_matmul_8x8(p, p)
    r = _vp_matmul_8x8(r, eye_8 + p)
    p = _vp_matmul_8x8(p, p)
    r = _vp_matmul_8x8(r, eye_8 + p)
    return r


class LegacyGDRBlockTriInverse(nn.Module):
    """DEPRECATED legacy block inverse implementation.

    Kept only for ``XHQUANT_GDR_USE_LEGACY=1`` regression diff. Bit-identical
    evidence (QTL-343 / T7) shows the xhquant first-class fused op is a
    complete drop-in replacement.

    Block lower-triangular matrix inverse: (I - A)^{-1}.

    Uses block forward substitution with Neumann series for diagonal blocks.
    Registered as FX leaf so the entire computation appears as one graph node.

    Args:
        chunk_size: Size of the square matrix (default 64).
        block_size: Size of diagonal blocks for Neumann inversion (default 8).
    """

    def __init__(self, chunk_size: int = 64, block_size: int = 8):
        super().__init__()
        self.chunk_size = chunk_size
        self.block_size = block_size
        self._nb = chunk_size // block_size
        warnings.warn(
            "LegacyGDRBlockTriInverse is deprecated and only kept behind "
            "XHQUANT_GDR_USE_LEGACY=1; the xhquant first-class fused op is "
            "bit-identical to this legacy chain (QTL-343).",
            DeprecationWarning,
            stacklevel=2,
        )

    def forward(self, attn: torch.Tensor, eye_b_batched: torch.Tensor) -> torch.Tensor:
        # QTL-339: eye_b_batched is externally provided (was self.eye_8_batched
        # registered via setup()). Keeps signature aligned with the xhquant
        # first-class op (xh::GDRBlockTriInverse) post QTL-338.
        B, heads, num_chunks, cs, _ = attn.shape
        flat = attn.flatten(0, -3)  # (BHN, cs, cs)
        b = self.block_size
        nb = self._nb

        # Step 1: Batch compute all diagonal block inverses via Neumann series
        diag_blocks = torch.stack(
            [
                flat[:, block_idx * b : (block_idx + 1) * b, block_idx * b : (block_idx + 1) * b]
                for block_idx in range(nb)
            ],
            dim=1,
        )  # (BHN, nb, b, b)
        batched = diag_blocks.reshape(-1, b, b)  # (BHN*nb, b, b)
        diag_invs_batched = _neumann_8x8(batched, eye_b_batched.to(attn.dtype))
        diag_invs_all = diag_invs_batched.reshape(-1, nb, b, b)  # (BHN, nb, b, b)

        # Step 2: Block forward substitution (nb serial steps)
        R_upper = diag_invs_all[:, 0]  # (BHN, b, b)

        for block_idx in range(1, nb):
            s = block_idx * b
            T = torch.matmul(flat[:, s : s + b, :s], R_upper)
            diag_inv_i = diag_invs_all[:, block_idx]
            off_diag = torch.matmul(diag_inv_i, T)
            old_rows = F.pad(R_upper, (0, b))
            new_rows = torch.cat([off_diag, diag_inv_i], dim=-1)
            R_upper = torch.cat([old_rows, new_rows], dim=-2)

        return R_upper.reshape(B, heads, num_chunks, cs, cs)


class LegacyGDRChunkScan(nn.Module):
    """DEPRECATED legacy chunk scan implementation.

    Kept only for ``XHQUANT_GDR_USE_LEGACY=1`` regression diff. Bit-identical
    evidence (QTL-343 / T7) shows the xhquant first-class fused op is a
    complete drop-in replacement.

    Chunk-level state scan for Gated Delta Rule.

    Iterates over chunks, computing intra-chunk attention and cross-chunk
    state contributions, updating the recurrent state each step.

    Args:
        num_chunks: Number of chunks (unrolled loop count).
        num_heads: Number of attention heads.
        k_head_dim: Key head dimension.
        v_head_dim: Value head dimension.
        chunk_size: Chunk size (default 64).
    """

    state_is_cache = False
    returns_state = True

    def __init__(
        self,
        num_chunks: int,
        num_heads: int,
        k_head_dim: int,
        v_head_dim: int,
        chunk_size: int = 64,
    ):
        super().__init__()
        self.num_chunks = num_chunks
        self.num_heads = num_heads
        self.k_head_dim = k_head_dim
        self.v_head_dim = v_head_dim
        self.chunk_size = chunk_size
        warnings.warn(
            "LegacyGDRChunkScan is deprecated and only kept behind "
            "XHQUANT_GDR_USE_LEGACY=1; the xhquant first-class fused op is "
            "bit-identical to this legacy chain (QTL-343).",
            DeprecationWarning,
            stacklevel=2,
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        k_cumdecay: torch.Tensor,
        decay_mask: torch.Tensor,
        mask_incl: torch.Tensor,
        g: torch.Tensor,
        initial_state: torch.Tensor,
    ) -> tuple:
        """
        Args:
            query: (B, H, num_chunks, CS, k_dim)
            key: (B, H, num_chunks, CS, k_dim)
            value: (B, H, num_chunks, CS, v_dim) — post block-tri-inverse
            k_cumdecay: (B, H, num_chunks, CS, k_dim)
            decay_mask: (B, H, num_chunks, CS, CS)
            mask_incl: (CS, CS)
            g: (B, H, num_chunks, CS)
            initial_state: (B, H, k_dim, v_dim)

        Returns:
            core_attn_out: (B, H, num_chunks, CS, v_dim)
            last_recurrent_state: (B, H, k_dim, v_dim)
        """
        last_recurrent_state = initial_state
        core_attn_chunks = []

        for i in range(self.num_chunks):
            q_i = query[:, :, i]
            k_i = key[:, :, i]
            v_i = value[:, :, i]

            attn = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]) * mask_incl
            v_prime = k_cumdecay[:, :, i] @ last_recurrent_state
            v_new = v_i - v_prime
            attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
            core_attn_chunks.append(attn_inter + attn @ v_new)

            last_recurrent_state = (
                last_recurrent_state * g[:, :, i, -1, None, None].exp()
                + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
            )

        core_attn_out = torch.stack(core_attn_chunks, dim=2)
        return core_attn_out, last_recurrent_state


class LegacyGDRRecurrentScan(nn.Module):
    """DEPRECATED legacy recurrent scan implementation.

    Kept only for ``XHQUANT_GDR_USE_LEGACY=1`` regression diff. Bit-identical
    evidence (QTL-343 / T7) shows the xhquant first-class fused op is a
    complete drop-in replacement.

    Token-level recurrent state scan for Gated Delta Rule.

    Iterates over T tokens, updating the recurrent state each step and
    emitting per-step attention output. Optionally returns all intermediate
    states (for speculative decoding verification, T <= 10).

    Args:
        sequence_length: Number of tokens to scan (unrolled loop count).
        output_all_states: If True, return per-step state snapshots.
    """

    returns_state = False

    def __init__(self, sequence_length: int = 1, output_all_states: bool = False):
        super().__init__()
        self.sequence_length = sequence_length
        self.output_all_states = output_all_states
        warnings.warn(
            "LegacyGDRRecurrentScan is deprecated and only kept behind "
            "XHQUANT_GDR_USE_LEGACY=1; the xhquant first-class fused op is "
            "bit-identical to this legacy chain (QTL-343).",
            DeprecationWarning,
            stacklevel=2,
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor,
    ) -> tuple:
        """
        Args:
            query: (B, H, T, k_dim) — pre-processed (l2norm + scaled)
            key: (B, H, T, k_dim) — pre-processed (l2norm)
            value: (B, H, T, v_dim)
            g: (B, H, T) — gate values (pre-transposed)
            beta: (B, H, T) — beta values (pre-transposed)
            initial_state: (B, H, k_dim, v_dim)

        Returns:
            core_attn_out: (B, H, T, v_dim)
            last_recurrent_state: (B, H, k_dim, v_dim)
            *state_snapshots: T tensors of (B, H, k_dim, v_dim) if output_all_states
        """
        last_recurrent_state = initial_state
        core_attn_steps = []
        state_snapshots = []

        for i in range(self.sequence_length):
            q_t = query[:, :, i]
            k_t = key[:, :, i]
            v_t = value[:, :, i]
            g_t = g[:, :, i].exp()
            if g_t.ndim == 2:
                g_t = g_t.unsqueeze(-1).unsqueeze(-1)
            elif g_t.ndim == 3:
                if g_t.shape[-1] != last_recurrent_state.shape[-2]:
                    raise ValueError(
                        "Per-channel GDRRecurrentScan gate must match the recurrent state's key dimension"
                    )
                g_t = g_t.unsqueeze(-1)
            else:
                raise ValueError(
                    "GDRRecurrentScan gate must have shape [B,H,T] or [B,H,T,K]"
                )
            beta_t = beta[:, :, i].unsqueeze(-1)

            last_recurrent_state = last_recurrent_state * g_t
            kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
            delta = (v_t - kv_mem) * beta_t
            last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
            core_attn_steps.append((last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2))

            if self.output_all_states:
                state_snapshots.append(last_recurrent_state)

        core_attn_out = torch.stack(core_attn_steps, dim=2)

        if self.output_all_states:
            return (core_attn_out, last_recurrent_state, *state_snapshots)
        return core_attn_out, last_recurrent_state


# --- QTL-332 dispatch -----------------------------------------------------
def _select_gdr_block_tri_inverse_impl():
    if _HAS_XHQUANT_GDR and _os.environ.get("XHQUANT_GDR_USE_LEGACY", "0") != "1":
        return _XHQuantGDRBlockTriInverse
    return LegacyGDRBlockTriInverse


GDRBlockTriInverse = _select_gdr_block_tri_inverse_impl()


# --- QTL-333 dispatch -----------------------------------------------------
def _select_gdr_chunk_scan_impl():
    if _HAS_XHQUANT_GDR_CHUNK_SCAN and _os.environ.get("XHQUANT_GDR_USE_LEGACY", "0") != "1":
        return _XHQuantGDRChunkScan
    return LegacyGDRChunkScan


GDRChunkScan = _select_gdr_chunk_scan_impl()


# --- QTL-334 dispatch -----------------------------------------------------
def _select_gdr_recurrent_scan_impl():
    if _HAS_XHQUANT_GDR_RECURRENT_SCAN and _os.environ.get("XHQUANT_GDR_USE_LEGACY", "0") != "1":
        return _XHQuantGDRRecurrentScan
    return LegacyGDRRecurrentScan


GDRRecurrentScan = _select_gdr_recurrent_scan_impl()
