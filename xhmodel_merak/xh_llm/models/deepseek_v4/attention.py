"""Static latent-attention kernels for DeepSeek-V4 Flash.

DeepSeek-V4 has one normalized 512-wide latent row that is read as both K and
V. The XH2 graph nevertheless receives distinct K and V tensors because
K-SEFP and V-SEFP use different scale-reduction axes. These modules keep that
physical distinction all the way through attention.

The model does not contain the kv_b_proj factor used by classic DeepSeek MLA
weight absorption. Its efficient form is therefore a compact shared latent
cache plus fused Q/RoPE and inverse-RoPE/grouped-output kernels, not an offline
matrix product that expands K/V per head.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from xhquant import nn as xhnn

from ._trace import is_fx_proxy


def _rotate_interleaved_pairs(x: Tensor, width: int) -> Tensor:
    pairs = x.reshape(
        x.shape[0],
        x.shape[1],
        x.shape[2],
        int(width) // 2,
        2,
    )
    first = pairs[..., 0]
    second = pairs[..., 1]
    return torch.stack((-second, first), dim=-1).flatten(-2)


def apply_partial_rope(
    x: Tensor,
    cos: Tensor,
    sin: Tensor,
    *,
    rope_dim: int = 64,
    inverse: bool = False,
) -> Tensor:
    """Apply V4 interleaved RoPE to the trailing rope_dim channels.

    x is [B,P,H,D] and cos/sin are [B,P,rope_dim/2]. Passing inverse=True
    applies the conjugate rotation used before the grouped output projection.
    """

    rope_dim = int(rope_dim)
    if not is_fx_proxy(x):
        if x.ndim != 4:
            raise ValueError("x must be [B,P,H,D]")
        if rope_dim <= 0 or rope_dim > x.shape[-1] or rope_dim % 2:
            raise ValueError("rope_dim must be positive, even, and no larger than D")
        expected = (x.shape[0], x.shape[1], rope_dim // 2)
        if tuple(cos.shape) != expected or tuple(sin.shape) != expected:
            raise ValueError(f"cos/sin must have shape {expected}")

    cos_full = cos.repeat_interleave(2, dim=-1).unsqueeze(2)
    sin_full = sin.repeat_interleave(2, dim=-1).unsqueeze(2)
    if inverse:
        sin_full = -sin_full
    nope, rope = x[..., :-rope_dim], x[..., -rope_dim:]
    rotated = rope * cos_full + _rotate_interleaved_pairs(rope, rope_dim) * sin_full
    return torch.cat((nope, rotated), dim=-1)


class InterleavedPartialRope(nn.Module):
    """Lower V4's interleaved partial RoPE through the XH Rope operator.

    DeepSeek-V4 stores each complex pair as ``[real, imag]`` while XH2 Rope
    consumes the half-split layout ``[all_real, all_imag]``. The surrounding
    reshapes/transposes are fixed permutations; the actual rotation remains
    one ``XH2a::Rope`` node.
    """

    def __init__(self, rope_dim: int = 64) -> None:
        super().__init__()
        self.rope_dim = int(rope_dim)
        if self.rope_dim <= 0 or self.rope_dim % 2:
            raise ValueError("rope_dim must be positive and even")
        self.rope = xhnn.Rope()

    def forward(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        *,
        inverse: bool = False,
    ) -> Tensor:
        rope_dim = self.rope_dim
        if not is_fx_proxy(x):
            if x.ndim != 4:
                raise ValueError("x must be [B,P,H,D]")
            if rope_dim > x.shape[-1]:
                raise ValueError("rope_dim must not exceed D")
            expected = (x.shape[0], x.shape[1], rope_dim // 2)
            if tuple(cos.shape) != expected or tuple(sin.shape) != expected:
                raise ValueError(f"cos/sin must have shape {expected}")

        nope, interleaved = x[..., :-rope_dim], x[..., -rope_dim:]
        half_split = (
            interleaved.reshape(
                interleaved.shape[0],
                interleaved.shape[1],
                interleaved.shape[2],
                rope_dim // 2,
                2,
            )
            .transpose(-1, -2)
            .reshape(
                interleaved.shape[0],
                interleaved.shape[1],
                interleaved.shape[2],
                rope_dim,
            )
            .transpose(1, 2)
        )
        cos_half = torch.cat((cos, cos), dim=-1).unsqueeze(1)
        sin_half = torch.cat((sin, sin), dim=-1).unsqueeze(1)
        if inverse:
            sin_half = -sin_half
        rotated = self.rope(half_split, cos_half, sin_half)
        rotated = (
            rotated.transpose(1, 2)
            .reshape(
                interleaved.shape[0],
                interleaved.shape[1],
                interleaved.shape[2],
                2,
                rope_dim // 2,
            )
            .transpose(-1, -2)
            .reshape(
                interleaved.shape[0],
                interleaved.shape[1],
                interleaved.shape[2],
                rope_dim,
            )
        )
        # XH Rope's ABI preserves the input dtype. Keeping the operator result
        # directly also avoids an unsupported graph-side ``type_as`` node.
        return torch.cat((nope, rotated), dim=-1)


class _LatentAttentionBase(nn.Module):
    def __init__(
        self,
        *,
        head_dim: int = 512,
        num_heads: int = 64,
        invalid_score: float = -65504.0,
    ) -> None:
        super().__init__()
        self.head_dim = int(head_dim)
        self.num_heads = int(num_heads)
        self.invalid_score = float(invalid_score)
        if self.head_dim <= 0 or self.num_heads <= 0:
            raise ValueError("attention dimensions must be positive")
        self.scale = self.head_dim**-0.5
        self.softmax = xhnn.SinksSoftmax(dim=-1)

    def _validate_query(self, query: Tensor, sinks: Tensor) -> None:
        if not is_fx_proxy(query):
            if query.ndim != 4 or query.shape[-2:] != (self.num_heads, self.head_dim):
                raise ValueError(f"query must be [B,P,{self.num_heads},{self.head_dim}]")
            if tuple(sinks.shape) != (self.num_heads,):
                raise ValueError(f"sinks must have shape ({self.num_heads},)")

    def _probabilities(self, scores: Tensor, attention_mask: Tensor, sinks: Tensor) -> Tensor:
        expected = (scores.shape[0], 1, scores.shape[2], scores.shape[3])
        if not is_fx_proxy(scores) and tuple(attention_mask.shape) != expected:
            raise ValueError(f"additive attention mask must have shape {expected}")
        scores = scores * self.scale + attention_mask
        sink_logits = sinks.reshape(1, self.num_heads, 1, 1)
        return self.softmax(scores, sinks=sink_logits)


class SWALatentAttention(_LatentAttentionBase):
    """Attention over one shared physical SWA context and a per-query mask."""

    def forward(
        self,
        query: Tensor,
        swa_k: Tensor,
        swa_v: Tensor,
        attention_mask: Tensor,
        sinks: Tensor,
    ) -> Tensor:
        self._validate_query(query, sinks)
        if not is_fx_proxy(query):
            expected = (query.shape[0], swa_k.shape[1], self.head_dim)
            if tuple(swa_k.shape) != expected or tuple(swa_v.shape) != expected:
                raise ValueError("SWA K/V must be [B,T,D]")
        scores = torch.matmul(
            query.transpose(1, 2),
            swa_k.transpose(-1, -2).unsqueeze(1),
        )
        probs = self._probabilities(scores, attention_mask, sinks)
        return torch.matmul(
            probs,
            swa_v.unsqueeze(1),
        ).transpose(1, 2)


class CSALatentAttention(_LatentAttentionBase):
    """Attention over TopK compressed rows plus the SWA window."""

    def forward(
        self,
        query: Tensor,
        compressed_kv: Tensor,
        swa_k: Tensor,
        swa_v: Tensor,
        attention_mask: Tensor,
        sinks: Tensor,
    ) -> Tensor:
        self._validate_query(query, sinks)
        if not is_fx_proxy(query):
            if compressed_kv.ndim != 4:
                raise ValueError("compressed latent cache must be [B,P,K,D]")
            if swa_k.ndim != 3 or tuple(swa_v.shape) != tuple(swa_k.shape):
                raise ValueError("SWA K/V must be [B,T,D]")
            if compressed_kv.shape[:2] != query.shape[:2] or swa_k.shape[0] != query.shape[0]:
                raise ValueError("K/V batch and query dimensions must match query")
            if compressed_kv.shape[-1] != self.head_dim or swa_k.shape[-1] != self.head_dim:
                raise ValueError("K/V latent width must match head_dim")

        compressed_scores = torch.matmul(
            query,
            compressed_kv.transpose(-1, -2),
        ).transpose(1, 2)
        swa_scores = torch.matmul(
            query.transpose(1, 2),
            swa_k.transpose(-1, -2).unsqueeze(1),
        )
        scores = torch.cat((compressed_scores, swa_scores), dim=-1)
        probs = self._probabilities(scores, attention_mask, sinks)
        compressed_count = compressed_kv.shape[2]
        compressed_output = torch.matmul(
            probs[..., :compressed_count].transpose(1, 2),
            compressed_kv,
        )
        swa_output = torch.matmul(
            probs[..., compressed_count:],
            swa_v.unsqueeze(1),
        ).transpose(1, 2)
        return compressed_output + swa_output


class HCALatentAttention(_LatentAttentionBase):
    """Dense HCA attention without expanding compressed rows per query."""

    def forward(
        self,
        query: Tensor,
        compressed_kv: Tensor,
        swa_k: Tensor,
        swa_v: Tensor,
        attention_mask: Tensor,
        sinks: Tensor,
    ) -> Tensor:
        self._validate_query(query, sinks)
        if not is_fx_proxy(query):
            if compressed_kv.ndim != 3:
                raise ValueError("HCA latent cache must be [B,C,D]")
            if compressed_kv.shape[0] != query.shape[0] or compressed_kv.shape[-1] != self.head_dim:
                raise ValueError("HCA K/V dimensions do not match query")
            if swa_k.ndim != 3 or tuple(swa_v.shape) != tuple(swa_k.shape):
                raise ValueError("SWA K/V must be [B,T,D]")

        long_scores = torch.matmul(
            query.transpose(1, 2),
            compressed_kv.transpose(-1, -2).unsqueeze(1),
        )
        swa_scores = torch.matmul(
            query.transpose(1, 2),
            swa_k.transpose(-1, -2).unsqueeze(1),
        )
        scores = torch.cat((long_scores, swa_scores), dim=-1)
        probs = self._probabilities(scores, attention_mask, sinks)
        long_count = compressed_kv.shape[1]
        long_probs = probs[..., :long_count]
        swa_probs = probs[..., long_count:]
        long_output = torch.matmul(
            long_probs,
            compressed_kv.unsqueeze(1),
        ).transpose(1, 2)
        swa_output = torch.matmul(
            swa_probs,
            swa_v.unsqueeze(1),
        ).transpose(1, 2)
        return long_output + swa_output


class GroupedLatentOutputProjection(nn.Module):
    """Inverse partial RoPE followed by V4 grouped o_a and o_b."""

    def __init__(
        self,
        *,
        num_heads: int = 64,
        head_dim: int = 512,
        num_groups: int = 8,
        o_lora_rank: int = 1024,
        hidden_size: int = 4096,
        rope_dim: int = 64,
        dtype: torch.dtype = torch.float16,
    ) -> None:
        super().__init__()
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.num_groups = int(num_groups)
        self.o_lora_rank = int(o_lora_rank)
        self.hidden_size = int(hidden_size)
        self.rope_dim = int(rope_dim)
        self.partial_rope = InterleavedPartialRope(self.rope_dim)
        if self.num_heads % self.num_groups:
            raise ValueError("num_heads must be divisible by num_groups")
        group_width = (self.num_heads // self.num_groups) * self.head_dim
        # ``from_hf``/``load_hf_weights`` replaces these with checkpoint
        # values.  The plain constructor is also used by graph and operator
        # tests, so an uninitialised ``empty`` tensor can otherwise leak NaNs
        # into an unrelated static-attention test depending on allocator
        # history.  Zero is the deterministic neutral placeholder.
        self.o_a_weight = nn.Parameter(
            torch.zeros(
                self.num_groups,
                self.o_lora_rank,
                group_width,
                dtype=dtype,
            )
        )
        self.o_b_weight = nn.Parameter(
            torch.zeros(
                self.hidden_size,
                self.num_groups * self.o_lora_rank,
                dtype=dtype,
            )
        )

    @torch.no_grad()
    def load_hf_weights(self, o_a_weight: Tensor, o_b_weight: Tensor) -> None:
        expected_a = (
            self.num_groups * self.o_lora_rank,
            (self.num_heads // self.num_groups) * self.head_dim,
        )
        expected_b = (self.hidden_size, self.num_groups * self.o_lora_rank)
        if tuple(o_a_weight.shape) != expected_a or tuple(o_b_weight.shape) != expected_b:
            raise ValueError(f"HF output weights must be {expected_a} and {expected_b}")
        self.o_a_weight.copy_(o_a_weight.reshape_as(self.o_a_weight))
        self.o_b_weight.copy_(o_b_weight)

    def forward(self, attention_output: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        expected = (
            attention_output.shape[0],
            attention_output.shape[1],
            self.num_heads,
            self.head_dim,
        )
        if tuple(attention_output.shape) != expected:
            raise ValueError(f"attention_output must have shape {expected}")
        output = self.partial_rope(
            attention_output,
            cos,
            sin,
            inverse=True,
        )
        grouped = output.reshape(
            output.shape[0],
            output.shape[1],
            self.num_groups,
            -1,
        )
        low_rank = torch.matmul(
            grouped.unsqueeze(-2),
            self.o_a_weight.transpose(-1, -2),
        ).squeeze(-2)
        return F.linear(
            low_rank.flatten(2),
            self.o_b_weight,
        )


__all__ = [
    "CSALatentAttention",
    "GroupedLatentOutputProjection",
    "HCALatentAttention",
    "InterleavedPartialRope",
    "SWALatentAttention",
    "apply_partial_rope",
]
