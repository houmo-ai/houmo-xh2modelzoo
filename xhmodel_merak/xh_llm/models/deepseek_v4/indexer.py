"""Static Lightning Indexer for DeepSeek-V4 CSA layers."""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor, nn

from ._trace import is_fx_proxy
from .attention import InterleavedPartialRope
from .compressor import LearnedOverlappingCompressor
from .norm import rms_norm_from_hf
from .static_cache import CSAIndexerTopK, FixedCapacityCacheWriter


def _normalized_hadamard_matrix(width: int) -> Tensor:
    """Build the Sylvester matrix matching the previous butterfly ordering."""

    width = int(width)
    if width <= 0 or width & (width - 1):
        raise ValueError("Hadamard width must be a positive power of two")
    matrix = torch.ones((1, 1), dtype=torch.float16)
    while matrix.shape[0] < width:
        matrix = torch.cat(
            (
                torch.cat((matrix, matrix), dim=1),
                torch.cat((matrix, -matrix), dim=1),
            ),
            dim=0,
        )
    return matrix * (width**-0.5)


def normalized_hadamard(
    x: Tensor,
    width: int | None = None,
    *,
    rank: int | None = None,
) -> Tensor:
    """Apply an orthonormal power-of-two Walsh-Hadamard transform."""

    if width is None:
        if is_fx_proxy(x):
            raise ValueError("symbolic Hadamard requires an explicit width")
        width = int(x.shape[-1])
    width = int(width)
    if width <= 0 or width & (width - 1):
        raise ValueError("Hadamard width must be a positive power of two")
    if rank is None:
        if is_fx_proxy(x):
            raise ValueError("symbolic Hadamard requires an explicit rank")
        rank = int(x.ndim)
    if rank not in {3, 4}:
        raise ValueError("Hadamard input rank must be 3 or 4")
    # Public eager helper. StaticCSAIndexer owns the same matrix as a buffer so
    # its exported graph carries one shared initializer instead of rebuilding
    # the constant per call.
    matrix = _normalized_hadamard_matrix(width).to(device=x.device)
    return torch.matmul(x, matrix)


class NormalizedHadamard(nn.Module):
    """One frozen FP16 Linear implementing orthonormal H{width}/sqrt(width)."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.width = int(width)
        self.matmul = nn.Linear(self.width, self.width, bias=False, dtype=torch.float16)
        with torch.no_grad():
            # Linear computes x @ weight.T, while the reference helper computes
            # x @ matrix.  Keep the transpose explicit even though a Sylvester
            # Hadamard matrix is symmetric.
            self.matmul.weight.copy_(_normalized_hadamard_matrix(self.width).T)
        self.matmul.weight.requires_grad_(False)

    def forward(self, x: Tensor) -> Tensor:
        return self.matmul(x)


class StaticCSAIndexerOutput(NamedTuple):
    topk_values: Tensor
    topk_indices: Tensor
    updated_key_cache: Tensor
    next_kv_state: Tensor
    next_score_state: Tensor
    new_count: Tensor


class StaticCSAIndexer(nn.Module):
    """Compress Indexer keys, update its fixed cache, and select global TopK."""

    def __init__(
        self,
        *,
        hidden_size: int,
        q_lora_rank: int,
        input_sequence_length: int,
        cache_capacity: int,
        num_heads: int = 64,
        head_dim: int = 128,
        ratio: int = 4,
        topk: int = 512,
        rms_norm_eps: float = 1e-6,
        rope_dim: int = 64,
        rotate: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.q_lora_rank = int(q_lora_rank)
        self.input_sequence_length = int(input_sequence_length)
        self.cache_capacity = int(cache_capacity)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.ratio = int(ratio)
        self.rope_dim = int(rope_dim)
        self.rotate = bool(rotate)
        self.hadamard = NormalizedHadamard(self.head_dim)
        self.partial_rope = InterleavedPartialRope(self.rope_dim)
        if self.cache_capacity < int(topk):
            raise ValueError("cache_capacity must be at least topk")

        self.compressor = LearnedOverlappingCompressor(
            hidden_size=self.hidden_size,
            head_dim=self.head_dim,
            input_sequence_length=self.input_sequence_length,
            ratio=self.ratio,
            rms_norm_eps=rms_norm_eps,
            rope_dim=self.rope_dim,
        )
        self.q_b_proj = nn.Linear(
            self.q_lora_rank,
            self.num_heads * self.head_dim,
            bias=False,
            dtype=torch.float16,
        )
        self.weights_proj = nn.Linear(self.hidden_size, self.num_heads, bias=False, dtype=torch.float16)
        self.topk = CSAIndexerTopK(
            topk=topk,
            cache_capacity=self.cache_capacity,
            score_scale=(self.num_heads * self.head_dim) ** -0.5,
        )
        self.key_cache_writer = FixedCapacityCacheWriter()

    @classmethod
    def from_hf(
        cls,
        indexer: nn.Module,
        *,
        input_sequence_length: int,
        cache_capacity: int,
    ) -> "StaticCSAIndexer":
        config = indexer.rotary_emb.config
        module = cls(
            hidden_size=config.hidden_size,
            q_lora_rank=config.q_lora_rank,
            input_sequence_length=input_sequence_length,
            cache_capacity=cache_capacity,
            num_heads=indexer.num_heads,
            head_dim=indexer.head_dim,
            ratio=indexer.compress_rate,
            topk=indexer.index_topk,
            rms_norm_eps=config.rms_norm_eps,
            rope_dim=config.qk_rope_head_dim,
        )
        module.compressor.kv_proj = indexer.kv_proj
        module.compressor.gate_proj = indexer.gate_proj
        module.compressor.position_bias = indexer.position_bias
        module.compressor.kv_norm = rms_norm_from_hf(indexer.kv_norm)
        module.q_b_proj = indexer.q_b_proj
        module.weights_proj = indexer.scorer.weights_proj
        return module

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float16,
    ) -> tuple[Tensor, Tensor]:
        return self.compressor.initial_state(batch_size, device=device, dtype=dtype)

    @torch.no_grad()
    def load_hf_weights(self, indexer: nn.Module) -> None:
        self.compressor.load_hf_weights(indexer)
        self.q_b_proj.load_state_dict(indexer.q_b_proj.state_dict())
        self.weights_proj.load_state_dict(indexer.scorer.weights_proj.state_dict())

    def forward(
        self,
        hidden_states: Tensor,
        q_residual: Tensor,
        key_cache: Tensor,
        write_start: Tensor,
        current_length: Tensor,
        compressor_validity: Tensor,
        compressor_new_count: Tensor,
        compressor_offset: Tensor,
        compressor_phase_indices: Tensor,
        index_validity: Tensor,
        kv_state: Tensor,
        score_state: Tensor,
        query_cos: Tensor,
        query_sin: Tensor,
        compressed_cos: Tensor,
        compressed_sin: Tensor,
    ) -> StaticCSAIndexerOutput:
        if not is_fx_proxy(hidden_states):
            expected_hidden = (
                hidden_states.shape[0],
                self.input_sequence_length,
                self.hidden_size,
            )
            expected_q = (
                hidden_states.shape[0],
                self.input_sequence_length,
                self.q_lora_rank,
            )
            expected_cache = (
                hidden_states.shape[0],
                1,
                self.cache_capacity,
                self.head_dim,
            )
            if tuple(hidden_states.shape) != expected_hidden:
                raise ValueError(f"hidden_states must have shape {expected_hidden}")
            if tuple(q_residual.shape) != expected_q:
                raise ValueError(f"q_residual must have shape {expected_q}")
            if tuple(key_cache.shape) != expected_cache:
                raise ValueError(f"key_cache must have shape {expected_cache}")
            expected_validity = (
                hidden_states.shape[0],
                self.input_sequence_length,
                self.cache_capacity,
            )
            if tuple(index_validity.shape) != expected_validity:
                raise ValueError(f"index_validity must have shape {expected_validity}")

        compressed = self.compressor(
            hidden_states,
            kv_state,
            score_state,
            compressor_offset,
            compressor_phase_indices,
            current_length,
            compressor_validity,
            compressor_new_count,
            compressed_cos,
            compressed_sin,
        )
        compressed_key = compressed.compressed
        if self.rotate:
            compressed_key = self.hadamard(compressed_key)
        updated_key_cache = self.key_cache_writer(
            key_cache,
            compressed_key,
            write_start,
            compressed.new_count,
        )

        query = self.q_b_proj(q_residual).reshape(
            1,
            self.input_sequence_length,
            self.num_heads,
            self.head_dim,
        )
        query = self.partial_rope(
            query,
            query_cos,
            query_sin,
        )
        if self.rotate:
            query = self.hadamard(query)
        index_weights = self.weights_proj(hidden_states)

        topk_values, topk_indices = self.topk(
            query,
            index_weights,
            updated_key_cache,
            index_validity,
        )
        return StaticCSAIndexerOutput(
            topk_values,
            topk_indices,
            updated_key_cache,
            compressed.next_kv_state,
            compressed.next_score_state,
            compressed.new_count,
        )


__all__ = [
    "StaticCSAIndexer",
    "StaticCSAIndexerOutput",
    "NormalizedHadamard",
    "normalized_hadamard",
]
