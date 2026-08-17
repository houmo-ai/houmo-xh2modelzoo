"""Static decoder layers combining mHC, attention, norms, and MoE."""

from __future__ import annotations

from typing import NamedTuple

from torch import Tensor, nn

from .mhc import BoundMHC
from .moe import DeepSeekV4MoE, DeepSeekV4MoEPlaceHolder
from .norm import rms_norm_from_hf
from .static_attention import (
    StaticCSAAttention,
    StaticCSAAttentionOutput,
    StaticHCAAttention,
    StaticHCAAttentionOutput,
    StaticSWAAttention,
    StaticSWAAttentionOutput,
)


class StaticSWADecoderOutput(NamedTuple):
    hidden_states: Tensor
    attention: StaticSWAAttentionOutput


class StaticCSADecoderOutput(NamedTuple):
    hidden_states: Tensor
    attention: StaticCSAAttentionOutput


class StaticHCADecoderOutput(NamedTuple):
    hidden_states: Tensor
    attention: StaticHCAAttentionOutput


class _StaticDecoderBase(nn.Module):
    def __init__(
        self,
        *,
        input_layernorm: nn.Module,
        post_attention_layernorm: nn.Module,
        attn_hc: BoundMHC,
        ffn_hc: BoundMHC,
        mlp: DeepSeekV4MoE,
    ) -> None:
        super().__init__()
        self.input_layernorm = input_layernorm
        self.post_attention_layernorm = post_attention_layernorm
        self.attn_hc = attn_hc
        self.ffn_hc = ffn_hc
        self.mlp = mlp

    @classmethod
    def _components_from_hf(
        cls,
        layer: nn.Module,
        *,
        moe_fast_mode: bool,
    ) -> dict[str, nn.Module]:
        if getattr(layer.mlp, "_xh_v4_streaming_placeholder", False):
            mlp = DeepSeekV4MoEPlaceHolder()
        else:
            mlp = DeepSeekV4MoE.from_hf(
                layer.mlp,
                layer.self_attn.config,
                fast_mode=moe_fast_mode,
            )
        return {
            "input_layernorm": rms_norm_from_hf(layer.input_layernorm),
            "post_attention_layernorm": rms_norm_from_hf(layer.post_attention_layernorm),
            "attn_hc": BoundMHC.from_hf(layer.attn_hc),
            "ffn_hc": BoundMHC.from_hf(layer.ffn_hc),
            "mlp": mlp,
        }

    def _attention_input(
        self,
        hidden_states: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        collapsed, post, combination = self.attn_hc.collapse(hidden_states)
        return self.input_layernorm(collapsed), post, combination

    def _finish_attention(
        self,
        attention_output: Tensor,
        residual: Tensor,
        post: Tensor,
        combination: Tensor,
    ) -> Tensor:
        return self.attn_hc.expand(
            attention_output,
            residual,
            post,
            combination,
        )

    def _finish_mlp(self, hidden_states: Tensor, input_ids: Tensor) -> Tensor:
        collapsed, post, combination = self.ffn_hc.collapse(hidden_states)
        mlp_input = self.post_attention_layernorm(collapsed)
        mlp_output = self.mlp(mlp_input, input_ids)
        return self.ffn_hc.expand(
            mlp_output,
            hidden_states,
            post,
            combination,
        )


class StaticSWADecoderLayer(_StaticDecoderBase):
    def __init__(self, *, attention: StaticSWAAttention, **kwargs) -> None:
        super().__init__(**kwargs)
        self.attention = attention

    @classmethod
    def from_hf(
        cls,
        layer: nn.Module,
        *,
        input_sequence_length: int,
        swa_backing_length: int | None = None,
        moe_fast_mode: bool = True,
    ) -> "StaticSWADecoderLayer":
        return cls(
            attention=StaticSWAAttention.from_hf(
                layer.self_attn,
                input_sequence_length=input_sequence_length,
                swa_backing_length=swa_backing_length,
            ),
            **cls._components_from_hf(
                layer,
                moe_fast_mode=moe_fast_mode,
            ),
        )

    def forward(
        self,
        hidden_states: Tensor,
        input_ids: Tensor,
        past_length: Tensor,
        current_length: Tensor,
        swa_attention_mask: Tensor,
        past_swa_k_cache: Tensor,
        past_swa_v_cache: Tensor,
        query_cos: Tensor,
        query_sin: Tensor,
    ) -> StaticSWADecoderOutput:
        residual = hidden_states
        attention_input, post, combination = self._attention_input(hidden_states)
        attention = self.attention(
            attention_input,
            past_length,
            current_length,
            swa_attention_mask,
            past_swa_k_cache,
            past_swa_v_cache,
            query_cos,
            query_sin,
        )
        hidden_states = self._finish_attention(
            attention.output,
            residual,
            post,
            combination,
        )
        hidden_states = self._finish_mlp(hidden_states, input_ids)
        return StaticSWADecoderOutput(hidden_states, attention)


class StaticCSADecoderLayer(_StaticDecoderBase):
    def __init__(self, *, attention: StaticCSAAttention, **kwargs) -> None:
        super().__init__(**kwargs)
        self.attention = attention

    @classmethod
    def from_hf(
        cls,
        layer: nn.Module,
        *,
        input_sequence_length: int,
        cache_capacity: int,
        swa_backing_length: int | None = None,
        moe_fast_mode: bool = True,
    ) -> "StaticCSADecoderLayer":
        return cls(
            attention=StaticCSAAttention.from_hf(
                layer.self_attn,
                input_sequence_length=input_sequence_length,
                cache_capacity=cache_capacity,
                swa_backing_length=swa_backing_length,
            ),
            **cls._components_from_hf(
                layer,
                moe_fast_mode=moe_fast_mode,
            ),
        )

    def forward(
        self,
        hidden_states: Tensor,
        input_ids: Tensor,
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
    ) -> StaticCSADecoderOutput:
        residual = hidden_states
        attention_input, post, combination = self._attention_input(hidden_states)
        attention = self.attention(
            attention_input,
            past_length,
            current_length,
            csa_index_validity,
            csa_attention_mask,
            past_swa_k_cache,
            past_swa_v_cache,
            past_main_k_cache,
            past_main_v_cache,
            past_index_k_cache,
            compressed_write_start,
            compressor_validity,
            compressor_new_count,
            compressor_offset,
            compressor_phase_indices,
            main_kv_state,
            main_score_state,
            index_kv_state,
            index_score_state,
            query_cos,
            query_sin,
            compressed_cos,
            compressed_sin,
        )
        hidden_states = self._finish_attention(
            attention.output,
            residual,
            post,
            combination,
        )
        hidden_states = self._finish_mlp(hidden_states, input_ids)
        return StaticCSADecoderOutput(hidden_states, attention)


class StaticHCADecoderLayer(_StaticDecoderBase):
    def __init__(self, *, attention: StaticHCAAttention, **kwargs) -> None:
        super().__init__(**kwargs)
        self.attention = attention

    @classmethod
    def from_hf(
        cls,
        layer: nn.Module,
        *,
        input_sequence_length: int,
        cache_capacity: int,
        swa_backing_length: int | None = None,
        moe_fast_mode: bool = True,
    ) -> "StaticHCADecoderLayer":
        return cls(
            attention=StaticHCAAttention.from_hf(
                layer.self_attn,
                input_sequence_length=input_sequence_length,
                cache_capacity=cache_capacity,
                swa_backing_length=swa_backing_length,
            ),
            **cls._components_from_hf(
                layer,
                moe_fast_mode=moe_fast_mode,
            ),
        )

    def forward(
        self,
        hidden_states: Tensor,
        input_ids: Tensor,
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
    ) -> StaticHCADecoderOutput:
        residual = hidden_states
        attention_input, post, combination = self._attention_input(hidden_states)
        attention = self.attention(
            attention_input,
            past_length,
            current_length,
            hca_attention_mask,
            past_swa_k_cache,
            past_swa_v_cache,
            past_main_k_cache,
            past_main_v_cache,
            compressed_write_start,
            compressor_validity,
            compressor_new_count,
            compressor_offset,
            compressor_phase_indices,
            main_kv_state,
            main_score_state,
            query_cos,
            query_sin,
            compressed_cos,
            compressed_sin,
        )
        hidden_states = self._finish_attention(
            attention.output,
            residual,
            post,
            combination,
        )
        hidden_states = self._finish_mlp(hidden_states, input_ids)
        return StaticHCADecoderOutput(hidden_states, attention)


__all__ = [
    "StaticCSADecoderLayer",
    "StaticCSADecoderOutput",
    "StaticHCADecoderLayer",
    "StaticHCADecoderOutput",
    "StaticSWADecoderLayer",
    "StaticSWADecoderOutput",
]
