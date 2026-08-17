from __future__ import annotations

import torch
from transformers.models.deepseek_v4.configuration_deepseek_v4 import (
    DeepseekV4Config,
)
from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4DecoderLayer,
    DeepseekV4RotaryEmbedding,
)

from xhmodel_merak.xh_llm.models.deepseek_v4.decoder import (
    StaticCSADecoderLayer,
    StaticHCADecoderLayer,
    StaticSWADecoderLayer,
)
from xhmodel_merak.xh_llm.models.deepseek_v4.host_masks import (
    build_deepseek_v4_attention_masks,
)
from xhmodel_merak.xh_llm.models.deepseek_v4.static_cache import (
    DeepSeekV4StaticCacheSpec,
)


def _host_masks():
    return build_deepseek_v4_attention_masks(
        DeepSeekV4StaticCacheSpec(
            max_context_length=16,
            prefill_chunk_length=8,
            sliding_window=4,
            latent_head_dim=8,
            index_head_dim=8,
            csa_ratio=4,
            hca_ratio=4,
            index_topk=4,
        ),
        input_sequence_length=8,
        past_length=0,
        current_length=8,
    )


def _tiny_config(layer_type: str = "sliding_attention") -> DeepseekV4Config:
    return DeepseekV4Config(
        vocab_size=11,
        hidden_size=6,
        moe_intermediate_size=5,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=8,
        q_lora_rank=5,
        num_experts_per_tok=2,
        n_routed_experts=3,
        n_shared_experts=1,
        max_position_embeddings=32,
        layer_types=[layer_type],
        compress_rates={
            "compressed_sparse_attention": 4,
            "heavily_compressed_attention": 4,
        },
        mlp_layer_types=["moe"],
        hc_mult=2,
        hc_sinkhorn_iters=4,
        sliding_window=4,
        o_groups=2,
        o_lora_rank=3,
        index_n_heads=4,
        index_head_dim=8,
        index_topk=4,
        num_nextn_predict_layers=0,
        partial_rotary_factor=0.5,
        use_cache=False,
    )


def _sliding_mask(length: int, window: int) -> torch.Tensor:
    mask = torch.full((1, 1, length, length), -65504.0, dtype=torch.float16)
    for query in range(length):
        mask[:, :, query, max(0, query - window + 1) : query + 1] = 0
    return mask


def _initialize_layer(
    layer: DeepseekV4DecoderLayer,
    generator: torch.Generator,
) -> None:
    # Constructing a decoder layer directly intentionally leaves checkpoint
    # tensors empty; PreTrainedModel.post_init normally initializes them.
    with torch.no_grad():
        for parameter in layer.parameters():
            if parameter.is_floating_point():
                parameter.uniform_(-0.05, 0.05, generator=generator)


def test_static_swa_decoder_matches_transformers_tiny_reference() -> None:
    generator = torch.Generator().manual_seed(101)
    config = _tiny_config()
    reference = DeepseekV4DecoderLayer(config, 0).eval()
    _initialize_layer(reference, generator)
    reference.half()
    rotary = DeepseekV4RotaryEmbedding(config).half()
    hidden = torch.randn(1, 8, 2, 6, generator=generator, dtype=torch.float16)
    input_ids = torch.randint(0, config.vocab_size, (1, 8), generator=generator)
    positions = torch.arange(8).reshape(1, -1)
    cos, sin = rotary(
        hidden[:, :, 0],
        position_ids=positions,
        layer_type="main",
    )

    expected = reference(
        hidden,
        input_ids=input_ids,
        position_embeddings={"main": (cos, sin)},
        position_ids=positions,
        attention_mask=_sliding_mask(8, 4),
        past_key_values=None,
    )
    module = StaticSWADecoderLayer.from_hf(
        reference,
        input_sequence_length=8,
        moe_fast_mode=False,
    ).eval()
    masks = _host_masks()
    actual = module(
        hidden,
        input_ids,
        torch.tensor([0]),
        torch.tensor([8]),
        masks.swa_attention_mask,
        torch.zeros(1, 1, module.attention.swa_update.backing_length, 8, dtype=torch.float16),
        torch.zeros(1, 1, module.attention.swa_update.backing_length, 8, dtype=torch.float16),
        cos,
        sin,
    ).hidden_states

    torch.testing.assert_close(actual, expected, rtol=3e-2, atol=3e-3)


def _compressed_reference_inputs(
    layer_type: str,
) -> tuple[
    DeepseekV4Config,
    DeepseekV4DecoderLayer,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    generator = torch.Generator().manual_seed(103)
    config = _tiny_config(layer_type)
    layer = DeepseekV4DecoderLayer(config, 0).eval()
    _initialize_layer(layer, generator)
    layer.half()
    rotary = DeepseekV4RotaryEmbedding(config).half()
    hidden = torch.randn(1, 8, 2, 6, generator=generator, dtype=torch.float16)
    input_ids = torch.randint(0, config.vocab_size, (1, 8), generator=generator)
    positions = torch.arange(8).reshape(1, -1)
    query_cos, query_sin = rotary(
        hidden[:, :, 0],
        position_ids=positions,
        layer_type="compress",
    )
    compressed_positions = torch.tensor([[0, 4]])
    compressed_cos, compressed_sin = rotary(
        hidden[:, :2, 0],
        position_ids=compressed_positions,
        layer_type="compress",
    )
    return (
        config,
        layer,
        hidden,
        input_ids,
        query_cos,
        query_sin,
        compressed_cos,
        compressed_sin,
    )


def test_static_csa_decoder_matches_transformers_tiny_reference() -> None:
    (
        config,
        reference,
        hidden,
        input_ids,
        query_cos,
        query_sin,
        compressed_cos,
        compressed_sin,
    ) = _compressed_reference_inputs("compressed_sparse_attention")
    positions = torch.arange(8).reshape(1, -1)
    expected = reference(
        hidden,
        input_ids=input_ids,
        position_embeddings={"compress": (query_cos, query_sin)},
        position_ids=positions,
        attention_mask=_sliding_mask(8, 4),
        past_key_values=None,
    )
    module = StaticCSADecoderLayer.from_hf(
        reference,
        input_sequence_length=8,
        cache_capacity=4,
        moe_fast_mode=False,
    ).eval()
    main_kv, main_score, index_kv, index_score = module.attention.initial_state(1)
    masks = _host_masks()
    actual = module(
        hidden,
        input_ids,
        torch.tensor([0]),
        torch.tensor([8]),
        masks.csa_index_validity,
        masks.csa_attention_mask,
        torch.zeros(1, 1, module.attention.swa_update.backing_length, 8, dtype=torch.float16),
        torch.zeros(1, 1, module.attention.swa_update.backing_length, 8, dtype=torch.float16),
        torch.zeros(1, 1, 4, 8, dtype=torch.float16),
        torch.zeros(1, 1, 4, 8, dtype=torch.float16),
        torch.zeros(1, 1, 4, 8, dtype=torch.float16),
        torch.tensor([0]),
        masks.csa_compressor_validity,
        masks.csa_compressor_new_count,
        masks.csa_compressor_offset,
        masks.csa_compressor_phase_indices,
        main_kv,
        main_score,
        index_kv,
        index_score,
        query_cos,
        query_sin,
        compressed_cos,
        compressed_sin,
    ).hidden_states

    torch.testing.assert_close(actual, expected, rtol=3e-2, atol=3e-3)


def test_static_hca_decoder_matches_transformers_tiny_reference() -> None:
    (
        config,
        reference,
        hidden,
        input_ids,
        query_cos,
        query_sin,
        compressed_cos,
        compressed_sin,
    ) = _compressed_reference_inputs("heavily_compressed_attention")
    positions = torch.arange(8).reshape(1, -1)
    expected = reference(
        hidden,
        input_ids=input_ids,
        position_embeddings={"compress": (query_cos, query_sin)},
        position_ids=positions,
        attention_mask=_sliding_mask(8, 4),
        past_key_values=None,
    )
    module = StaticHCADecoderLayer.from_hf(
        reference,
        input_sequence_length=8,
        cache_capacity=4,
        moe_fast_mode=False,
    ).eval()
    kv_state, score_state = module.attention.initial_state(1)
    masks = _host_masks()
    actual = module(
        hidden,
        input_ids,
        torch.tensor([0]),
        torch.tensor([8]),
        masks.hca_attention_mask,
        torch.zeros(1, 1, module.attention.swa_update.backing_length, 8, dtype=torch.float16),
        torch.zeros(1, 1, module.attention.swa_update.backing_length, 8, dtype=torch.float16),
        torch.zeros(1, 1, 4, 8, dtype=torch.float16),
        torch.zeros(1, 1, 4, 8, dtype=torch.float16),
        torch.tensor([0]),
        masks.hca_compressor_validity,
        masks.hca_compressor_new_count,
        masks.hca_compressor_offset,
        masks.hca_compressor_phase_indices,
        kv_state,
        score_state,
        query_cos,
        query_sin,
        compressed_cos,
        compressed_sin,
    ).hidden_states

    torch.testing.assert_close(actual, expected, rtol=3e-2, atol=3e-3)
