from __future__ import annotations

import torch

from xhmodel_merak.xh_llm.models.deepseek_v4.host_masks import (
    build_deepseek_v4_attention_masks,
)
from xhmodel_merak.xh_llm.models.deepseek_v4.static_attention import (
    StaticCSAAttention,
    StaticHCAAttention,
    StaticSWAAttention,
)
from xhmodel_merak.xh_llm.models.deepseek_v4.static_cache import (
    DeepSeekV4StaticCacheSpec,
)


def _common_kwargs() -> dict[str, int | float]:
    return {
        "hidden_size": 6,
        "q_lora_rank": 5,
        "num_heads": 4,
        "head_dim": 8,
        "num_groups": 2,
        "o_lora_rank": 3,
        "rope_dim": 4,
        "input_sequence_length": 8,
        "window_size": 4,
        "rms_norm_eps": 1e-6,
    }


def _masks(*, max_context: int = 64):
    return build_deepseek_v4_attention_masks(
        DeepSeekV4StaticCacheSpec(
            max_context_length=max_context,
            prefill_chunk_length=8,
            sliding_window=4,
            csa_ratio=4,
            hca_ratio=4,
            index_topk=4,
        ),
        input_sequence_length=8,
        past_length=0,
        current_length=7,
    )


def _swa_inputs(module: StaticSWAAttention) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(79)
    angles = torch.randn(1, 8, 2, generator=generator, dtype=torch.float16)
    masks = _masks()
    return (
        torch.randn(1, 8, 6, generator=generator, dtype=torch.float16),
        torch.tensor([0]),
        torch.tensor([7]),
        masks.swa_attention_mask,
        torch.zeros(1, 1, module.swa_update.backing_length, 8, dtype=torch.float16),
        torch.zeros(1, 1, module.swa_update.backing_length, 8, dtype=torch.float16),
        angles.cos(),
        angles.sin(),
    )


def test_complete_swa_attention_keeps_shared_context_and_exports() -> None:
    module = StaticSWAAttention(**_common_kwargs()).eval()
    inputs = _swa_inputs(module)

    output = module(*inputs)
    graph = torch.export.export(module, inputs)

    assert output.output.shape == (1, 8, 6)
    assert output.swa_k_context.shape == (1, 16, 8)
    valid = _masks().swa_attention_mask == 0
    assert valid[0, 0].sum(dim=-1).tolist() == [1, 2, 3, 4, 4, 4, 4, 0]
    torch.testing.assert_close(output.output[:, 7], torch.zeros_like(output.output[:, 7]))
    assert sum(node.target == torch.ops.xh.LLMCache.default for node in graph.graph.nodes) == 2
    assert any(node.target == torch.ops.xh.SinksSoftmax.default for node in graph.graph.nodes)


def test_complete_csa_attention_updates_three_long_caches_and_exports() -> None:
    module = StaticCSAAttention(
        **_common_kwargs(),
        cache_capacity=16,
        compressor_ratio=4,
        index_heads=4,
        index_head_dim=8,
        index_topk=4,
    ).eval()
    main_kv, main_score, index_kv, index_score = module.initial_state(1)
    query_angles = torch.randn(1, 8, 2, dtype=torch.float16)
    compressed_angles = torch.randn(1, 2, 2, dtype=torch.float16)
    masks = _masks()
    inputs = (
        torch.randn(1, 8, 6, dtype=torch.float16),
        torch.tensor([0]),
        torch.tensor([7]),
        masks.csa_index_validity,
        masks.csa_attention_mask,
        torch.zeros(1, 1, module.swa_update.backing_length, 8, dtype=torch.float16),
        torch.zeros(1, 1, module.swa_update.backing_length, 8, dtype=torch.float16),
        torch.zeros(1, 1, 16, 8, dtype=torch.float16),
        torch.zeros(1, 1, 16, 8, dtype=torch.float16),
        torch.zeros(1, 1, 16, 8, dtype=torch.float16),
        torch.tensor([0]),
        masks.csa_compressor_validity,
        masks.csa_compressor_new_count,
        masks.csa_compressor_offset,
        masks.csa_compressor_phase_indices,
        main_kv,
        main_score,
        index_kv,
        index_score,
        query_angles.cos(),
        query_angles.sin(),
        compressed_angles.cos(),
        compressed_angles.sin(),
    )

    output = module(*inputs)
    graph = torch.export.export(module, inputs)

    assert output.output.shape == (1, 8, 6)
    assert output.topk_indices.shape == (1, 8, 4)
    assert masks.csa_index_validity.sum(dim=-1)[0].tolist() == [0, 0, 0, 1, 1, 1, 1, 0]
    torch.testing.assert_close(output.output[:, 7], torch.zeros_like(output.output[:, 7]))
    assert torch.count_nonzero(output.main_k_cache[:, :1]) > 0
    assert torch.count_nonzero(output.main_v_cache[:, :1]) > 0
    assert torch.count_nonzero(output.index_k_cache[:, :1]) > 0
    assert sum(node.target == torch.ops.xh.LLMCache.default for node in graph.graph.nodes) == 5
    # The enclosing attention owns additional projection/attention MatMuls;
    # the indexer-specific unit test proves its exact three-MatMul inventory.
    assert sum(node.target == torch.ops.aten.matmul.default for node in graph.graph.nodes) >= 3
    assert sum(node.target == torch.ops.aten.topk.default for node in graph.graph.nodes) == 1


def test_complete_hca_attention_updates_dense_compressed_history_and_exports() -> None:
    module = StaticHCAAttention(
        **_common_kwargs(),
        cache_capacity=8,
        compressor_ratio=4,
    ).eval()
    kv_state, score_state = module.initial_state(1)
    query_angles = torch.randn(1, 8, 2, dtype=torch.float16)
    compressed_angles = torch.randn(1, 2, 2, dtype=torch.float16)
    masks = _masks(max_context=32)
    inputs = (
        torch.randn(1, 8, 6, dtype=torch.float16),
        torch.tensor([0]),
        torch.tensor([7]),
        masks.hca_attention_mask,
        torch.zeros(1, 1, module.swa_update.backing_length, 8, dtype=torch.float16),
        torch.zeros(1, 1, module.swa_update.backing_length, 8, dtype=torch.float16),
        torch.zeros(1, 1, 8, 8, dtype=torch.float16),
        torch.zeros(1, 1, 8, 8, dtype=torch.float16),
        torch.tensor([0]),
        masks.hca_compressor_validity,
        masks.hca_compressor_new_count,
        masks.hca_compressor_offset,
        masks.hca_compressor_phase_indices,
        kv_state,
        score_state,
        query_angles.cos(),
        query_angles.sin(),
        compressed_angles.cos(),
        compressed_angles.sin(),
    )

    output = module(*inputs)
    graph = torch.export.export(module, inputs)

    assert output.output.shape == (1, 8, 6)
    compressed_valid = masks.hca_attention_mask[:, :, :, :8] == 0
    assert compressed_valid[0, 0].sum(dim=-1).tolist() == [0, 0, 0, 1, 1, 1, 1, 0]
    torch.testing.assert_close(output.output[:, 7], torch.zeros_like(output.output[:, 7]))
    assert torch.count_nonzero(output.main_k_cache[:, :1]) > 0
    assert torch.count_nonzero(output.main_v_cache[:, :1]) > 0
    assert sum(node.target == torch.ops.xh.LLMCache.default for node in graph.graph.nodes) == 4
    assert any(node.target == torch.ops.xh.SinksSoftmax.default for node in graph.graph.nodes)
