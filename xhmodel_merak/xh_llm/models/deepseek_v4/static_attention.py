"""Complete static SWA/CSA/HCA attention paths for DeepSeek-V4 Flash."""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor, nn

from .attention import (
    CSALatentAttention,
    GroupedLatentOutputProjection,
    HCALatentAttention,
    SWALatentAttention,
)
from .compressor import (
    LearnedNonOverlappingCompressor,
    LearnedOverlappingCompressor,
)
from .indexer import StaticCSAIndexer
from .projection import BoundGroupedOutputProjection, DeepSeekV4QKVProjection
from .static_cache import FixedCapacityCacheWriter, SeparateKVGather
from .swa import StaticSWAOutput, StaticSWAUpdate


class StaticSWAAttentionOutput(NamedTuple):
    output: Tensor
    swa_k_context: Tensor
    swa_v_context: Tensor


class StaticCSAAttentionOutput(NamedTuple):
    output: Tensor
    swa_k_context: Tensor
    swa_v_context: Tensor
    main_k_cache: Tensor
    main_v_cache: Tensor
    index_k_cache: Tensor
    main_next_kv_state: Tensor
    main_next_score_state: Tensor
    index_next_kv_state: Tensor
    index_next_score_state: Tensor
    topk_indices: Tensor


class StaticHCAAttentionOutput(NamedTuple):
    output: Tensor
    swa_k_context: Tensor
    swa_v_context: Tensor
    main_k_cache: Tensor
    main_v_cache: Tensor
    main_next_kv_state: Tensor
    main_next_score_state: Tensor


class _StaticAttentionBase(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        q_lora_rank: int,
        num_heads: int,
        head_dim: int,
        num_groups: int,
        o_lora_rank: int,
        rope_dim: int,
        input_sequence_length: int,
        window_size: int,
        rms_norm_eps: float,
        swa_backing_length: int | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.q_lora_rank = int(q_lora_rank)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.rope_dim = int(rope_dim)
        self.input_sequence_length = int(input_sequence_length)
        self.window_size = int(window_size)
        self.projection = DeepSeekV4QKVProjection(
            hidden_size=self.hidden_size,
            q_lora_rank=self.q_lora_rank,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            rope_dim=self.rope_dim,
            rms_norm_eps=rms_norm_eps,
        )
        self.swa_update = StaticSWAUpdate(
            input_sequence_length=self.input_sequence_length,
            window_size=self.window_size,
            backing_length=swa_backing_length,
        )
        self.output_projection: nn.Module = GroupedLatentOutputProjection(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            num_groups=int(num_groups),
            o_lora_rank=int(o_lora_rank),
            hidden_size=self.hidden_size,
            rope_dim=self.rope_dim,
        )
        self.sinks = nn.Parameter(torch.zeros(self.num_heads, dtype=torch.float16))

    def _bind_hf_attention(self, attention: nn.Module) -> None:
        self.projection = DeepSeekV4QKVProjection.from_hf(attention)
        self.output_projection = BoundGroupedOutputProjection.from_hf(attention)
        self.sinks = attention.sinks

    def _project_and_update_swa(
        self,
        hidden_states: Tensor,
        query_cos: Tensor,
        query_sin: Tensor,
        past_length: Tensor,
        current_length: Tensor,
        past_swa_k_cache: Tensor,
        past_swa_v_cache: Tensor,
    ) -> tuple[Tensor, Tensor, StaticSWAOutput]:
        projected = self.projection(hidden_states, query_cos, query_sin)
        swa = self.swa_update(
            projected.latent_kv,
            past_length,
            current_length,
            past_swa_k_cache,
            past_swa_v_cache,
        )
        return projected.q_residual, projected.query, swa


class StaticSWAAttention(_StaticAttentionBase):
    """Sliding-only layer with a shared 384-row prefill KV context."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.attention = SWALatentAttention(
            head_dim=self.head_dim,
            num_heads=self.num_heads,
        )

    @classmethod
    def from_hf(
        cls,
        attention: nn.Module,
        *,
        input_sequence_length: int,
        swa_backing_length: int | None = None,
    ) -> "StaticSWAAttention":
        config = attention.config
        module = cls(
            hidden_size=config.hidden_size,
            q_lora_rank=config.q_lora_rank,
            num_heads=config.num_attention_heads,
            head_dim=config.head_dim,
            num_groups=config.o_groups,
            o_lora_rank=config.o_lora_rank,
            rope_dim=config.qk_rope_head_dim,
            input_sequence_length=input_sequence_length,
            window_size=config.sliding_window,
            swa_backing_length=swa_backing_length,
            rms_norm_eps=config.rms_norm_eps,
        )
        module._bind_hf_attention(attention)
        return module

    def forward(
        self,
        hidden_states: Tensor,
        past_length: Tensor,
        current_length: Tensor,
        swa_attention_mask: Tensor,
        past_swa_k_cache: Tensor,
        past_swa_v_cache: Tensor,
        query_cos: Tensor,
        query_sin: Tensor,
    ) -> StaticSWAAttentionOutput:
        _, query, swa = self._project_and_update_swa(
            hidden_states,
            query_cos,
            query_sin,
            past_length,
            current_length,
            past_swa_k_cache,
            past_swa_v_cache,
        )
        latent_output = self.attention(
            query,
            swa.physical_k,
            swa.physical_v,
            swa_attention_mask,
            self.sinks,
        )
        output = self.output_projection(latent_output, query_cos, query_sin)
        return StaticSWAAttentionOutput(
            output,
            swa.physical_k,
            swa.physical_v,
        )


class StaticCSAAttention(_StaticAttentionBase):
    """C4 compressor + Lightning Indexer TopK + SWA attention."""

    def __init__(
        self,
        *,
        cache_capacity: int,
        compressor_ratio: int = 4,
        index_heads: int = 64,
        index_head_dim: int = 128,
        index_topk: int = 512,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.cache_capacity = int(cache_capacity)
        self.compressor_ratio = int(compressor_ratio)
        self.main_compressor = LearnedOverlappingCompressor(
            hidden_size=self.hidden_size,
            head_dim=self.head_dim,
            input_sequence_length=self.input_sequence_length,
            ratio=self.compressor_ratio,
            rms_norm_eps=kwargs["rms_norm_eps"],
            rope_dim=self.rope_dim,
        )
        self.indexer = StaticCSAIndexer(
            hidden_size=self.hidden_size,
            q_lora_rank=self.q_lora_rank,
            input_sequence_length=self.input_sequence_length,
            cache_capacity=self.cache_capacity,
            num_heads=index_heads,
            head_dim=index_head_dim,
            ratio=self.compressor_ratio,
            topk=index_topk,
            rms_norm_eps=kwargs["rms_norm_eps"],
            rope_dim=self.rope_dim,
        )
        self.main_k_writer = FixedCapacityCacheWriter()
        self.main_v_writer = FixedCapacityCacheWriter()
        self.kv_gather = SeparateKVGather(feature_dim=self.head_dim)
        self.attention = CSALatentAttention(
            head_dim=self.head_dim,
            num_heads=self.num_heads,
        )

    @classmethod
    def from_hf(
        cls,
        attention: nn.Module,
        *,
        input_sequence_length: int,
        cache_capacity: int,
        swa_backing_length: int | None = None,
    ) -> "StaticCSAAttention":
        config = attention.config
        compressor = attention.compressor
        module = cls(
            hidden_size=config.hidden_size,
            q_lora_rank=config.q_lora_rank,
            num_heads=config.num_attention_heads,
            head_dim=config.head_dim,
            num_groups=config.o_groups,
            o_lora_rank=config.o_lora_rank,
            rope_dim=config.qk_rope_head_dim,
            input_sequence_length=input_sequence_length,
            window_size=config.sliding_window,
            swa_backing_length=swa_backing_length,
            rms_norm_eps=config.rms_norm_eps,
            cache_capacity=cache_capacity,
            compressor_ratio=compressor.compress_rate,
            index_heads=compressor.indexer.num_heads,
            index_head_dim=compressor.indexer.head_dim,
            index_topk=compressor.indexer.index_topk,
        )
        module._bind_hf_attention(attention)
        module.main_compressor = LearnedOverlappingCompressor.from_hf(
            compressor,
            input_sequence_length=input_sequence_length,
        )
        module.indexer = StaticCSAIndexer.from_hf(
            compressor.indexer,
            input_sequence_length=input_sequence_length,
            cache_capacity=cache_capacity,
        )
        return module

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float16,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        main = self.main_compressor.initial_state(
            batch_size,
            device=device,
            dtype=dtype,
        )
        index = self.indexer.initial_state(
            batch_size,
            device=device,
            dtype=dtype,
        )
        return (*main, *index)

    def forward(
        self,
        hidden_states: Tensor,
        past_length: Tensor,
        current_length: Tensor,
        csa_index_validity: Tensor,
        csa_attention_mask: Tensor,
        past_swa_k_cache: Tensor,
        past_swa_v_cache: Tensor,
        past_main_k_cache: Tensor,
        past_main_v_cache: Tensor,
        past_index_k_cache: Tensor,
        compressed_write_start: Tensor,
        compressor_validity: Tensor,
        compressor_new_count: Tensor,
        compressor_offset: Tensor,
        compressor_phase_indices: Tensor,
        main_kv_state: Tensor,
        main_score_state: Tensor,
        index_kv_state: Tensor,
        index_score_state: Tensor,
        query_cos: Tensor,
        query_sin: Tensor,
        compressed_cos: Tensor,
        compressed_sin: Tensor,
    ) -> StaticCSAAttentionOutput:
        q_residual, query, swa = self._project_and_update_swa(
            hidden_states,
            query_cos,
            query_sin,
            past_length,
            current_length,
            past_swa_k_cache,
            past_swa_v_cache,
        )
        compressed = self.main_compressor(
            hidden_states,
            main_kv_state,
            main_score_state,
            compressor_offset,
            compressor_phase_indices,
            current_length,
            compressor_validity,
            compressor_new_count,
            compressed_cos,
            compressed_sin,
        )
        main_k_cache = self.main_k_writer(
            past_main_k_cache,
            compressed.compressed,
            compressed_write_start,
            compressed.new_count,
        )
        main_v_cache = self.main_v_writer(
            past_main_v_cache,
            compressed.compressed,
            compressed_write_start,
            compressed.new_count,
        )
        index = self.indexer(
            hidden_states,
            q_residual,
            past_index_k_cache,
            compressed_write_start,
            current_length,
            compressor_validity,
            compressor_new_count,
            compressor_offset,
            compressor_phase_indices,
            csa_index_validity,
            index_kv_state,
            index_score_state,
            query_cos,
            query_sin,
            compressed_cos,
            compressed_sin,
        )
        selected_k, selected_v = self.kv_gather(
            main_k_cache,
            main_v_cache,
            index.topk_indices,
        )
        latent_output = self.attention(
            query,
            selected_k,
            selected_v,
            swa.physical_k,
            swa.physical_v,
            csa_attention_mask,
            self.sinks,
        )
        output = self.output_projection(latent_output, query_cos, query_sin)
        return StaticCSAAttentionOutput(
            output,
            swa.physical_k,
            swa.physical_v,
            main_k_cache,
            main_v_cache,
            index.updated_key_cache,
            compressed.next_kv_state,
            compressed.next_score_state,
            index.next_kv_state,
            index.next_score_state,
            index.topk_indices,
        )


class StaticHCAAttention(_StaticAttentionBase):
    """C128 dense compressed history + SWA attention."""

    def __init__(
        self,
        *,
        cache_capacity: int,
        compressor_ratio: int = 128,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.cache_capacity = int(cache_capacity)
        self.compressor_ratio = int(compressor_ratio)
        self.main_compressor = LearnedNonOverlappingCompressor(
            hidden_size=self.hidden_size,
            head_dim=self.head_dim,
            input_sequence_length=self.input_sequence_length,
            ratio=self.compressor_ratio,
            rms_norm_eps=kwargs["rms_norm_eps"],
            rope_dim=self.rope_dim,
        )
        self.main_k_writer = FixedCapacityCacheWriter()
        self.main_v_writer = FixedCapacityCacheWriter()
        self.attention = HCALatentAttention(
            head_dim=self.head_dim,
            num_heads=self.num_heads,
        )

    @classmethod
    def from_hf(
        cls,
        attention: nn.Module,
        *,
        input_sequence_length: int,
        cache_capacity: int,
        swa_backing_length: int | None = None,
    ) -> "StaticHCAAttention":
        config = attention.config
        compressor = attention.compressor
        module = cls(
            hidden_size=config.hidden_size,
            q_lora_rank=config.q_lora_rank,
            num_heads=config.num_attention_heads,
            head_dim=config.head_dim,
            num_groups=config.o_groups,
            o_lora_rank=config.o_lora_rank,
            rope_dim=config.qk_rope_head_dim,
            input_sequence_length=input_sequence_length,
            window_size=config.sliding_window,
            swa_backing_length=swa_backing_length,
            rms_norm_eps=config.rms_norm_eps,
            cache_capacity=cache_capacity,
            compressor_ratio=compressor.compress_rate,
        )
        module._bind_hf_attention(attention)
        module.main_compressor = LearnedNonOverlappingCompressor.from_hf(
            compressor,
            input_sequence_length=input_sequence_length,
        )
        return module

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float16,
    ) -> tuple[Tensor, Tensor]:
        return self.main_compressor.initial_state(
            batch_size,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        hidden_states: Tensor,
        past_length: Tensor,
        current_length: Tensor,
        hca_attention_mask: Tensor,
        past_swa_k_cache: Tensor,
        past_swa_v_cache: Tensor,
        past_main_k_cache: Tensor,
        past_main_v_cache: Tensor,
        compressed_write_start: Tensor,
        compressor_validity: Tensor,
        compressor_new_count: Tensor,
        compressor_offset: Tensor,
        compressor_phase_indices: Tensor,
        main_kv_state: Tensor,
        main_score_state: Tensor,
        query_cos: Tensor,
        query_sin: Tensor,
        compressed_cos: Tensor,
        compressed_sin: Tensor,
    ) -> StaticHCAAttentionOutput:
        _, query, swa = self._project_and_update_swa(
            hidden_states,
            query_cos,
            query_sin,
            past_length,
            current_length,
            past_swa_k_cache,
            past_swa_v_cache,
        )
        compressed = self.main_compressor(
            hidden_states,
            main_kv_state,
            main_score_state,
            compressor_offset,
            compressor_phase_indices,
            current_length,
            compressor_validity,
            compressor_new_count,
            compressed_cos,
            compressed_sin,
        )
        main_k_cache = self.main_k_writer(
            past_main_k_cache,
            compressed.compressed,
            compressed_write_start,
            compressed.new_count,
        )
        main_v_cache = self.main_v_writer(
            past_main_v_cache,
            compressed.compressed,
            compressed_write_start,
            compressed.new_count,
        )

        latent_output = self.attention(
            query,
            main_k_cache,
            main_v_cache,
            swa.physical_k,
            swa.physical_v,
            hca_attention_mask,
            self.sinks,
        )
        output = self.output_projection(latent_output, query_cos, query_sin)
        return StaticHCAAttentionOutput(
            output,
            swa.physical_k,
            swa.physical_v,
            main_k_cache,
            main_v_cache,
            compressed.next_kv_state,
            compressed.next_score_state,
        )


__all__ = [
    "StaticCSAAttention",
    "StaticCSAAttentionOutput",
    "StaticHCAAttention",
    "StaticHCAAttentionOutput",
    "StaticSWAAttention",
    "StaticSWAAttentionOutput",
]
