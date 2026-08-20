from __future__ import annotations

import torch

from xhmodel_merak.xh_llm.models.deepseek_v4.attention import apply_partial_rope
from xhmodel_merak.xh_llm.models.deepseek_v4.compressor import (
    LearnedNonOverlappingCompressor,
    LearnedOverlappingCompressor,
)
from xhmodel_merak.xh_llm.models.deepseek_v4.host_masks import (
    build_compressor_host_transition,
    build_deepseek_v4_attention_masks,
)
from xhmodel_merak.xh_llm.models.deepseek_v4.indexer import (
    NormalizedHadamard,
    StaticCSAIndexer,
    normalized_hadamard,
)
from xhmodel_merak.xh_llm.models.deepseek_v4.static_cache import (
    DeepSeekV4StaticCacheSpec,
)


def _index_validity(current_length: int) -> torch.Tensor:
    masks = build_deepseek_v4_attention_masks(
        DeepSeekV4StaticCacheSpec(
            max_context_length=64,
            prefill_chunk_length=8,
            sliding_window=4,
            csa_ratio=4,
            hca_ratio=4,
            index_topk=4,
        ),
        input_sequence_length=8,
        past_length=0,
        current_length=current_length,
    )
    return masks.csa_index_validity


def _transition(current_length: int):
    return build_compressor_host_transition(
        input_sequence_length=8,
        past_length=0,
        current_length=current_length,
        ratio=4,
        dtype=torch.float16,
    )


def test_plain_compressor_uses_neutral_position_bias_until_checkpoint_binding() -> None:
    module = LearnedOverlappingCompressor(
        hidden_size=6,
        head_dim=8,
        input_sequence_length=8,
        rope_dim=4,
    )

    assert torch.equal(module.position_bias, torch.zeros_like(module.position_bias))


def test_learned_overlap_compressor_adds_norm_and_rope_to_fixed_step() -> None:
    generator = torch.Generator().manual_seed(53)
    module = LearnedOverlappingCompressor(
        hidden_size=6,
        head_dim=8,
        input_sequence_length=8,
        rope_dim=4,
    )
    hidden_states = torch.randn(1, 8, 6, generator=generator, dtype=torch.float16)
    kv_state, score_state = module.initial_state(1)
    validity, new_count, offset, phase_indices = _transition(8)
    angles = torch.randn(1, 2, 2, generator=generator, dtype=torch.float16)
    cos, sin = angles.cos(), angles.sin()

    actual = module(
        hidden_states,
        kv_state,
        score_state,
        offset,
        phase_indices,
        torch.tensor([8]),
        validity,
        new_count,
        cos,
        sin,
    )
    fixed = module.step(
        module.kv_proj(hidden_states),
        module.gate_proj(hidden_states),
        kv_state,
        score_state,
        offset,
        phase_indices,
        torch.tensor([8]),
        validity,
        new_count,
        module.position_bias,
    )
    expected = module.kv_norm(fixed.pooled.to(hidden_states.dtype))
    expected = apply_partial_rope(
        expected.unsqueeze(2),
        cos,
        sin,
        rope_dim=4,
    ).squeeze(2)

    assert actual.compressed.shape == (1, 2, 8)
    assert actual.compressed_valid.all()
    torch.testing.assert_close(actual.compressed, expected)


def test_learned_hca_compressor_masks_unused_fixed_output() -> None:
    generator = torch.Generator().manual_seed(59)
    module = LearnedNonOverlappingCompressor(
        hidden_size=6,
        head_dim=8,
        input_sequence_length=256,
        ratio=128,
        rope_dim=4,
    )
    hidden_states = torch.randn(1, 256, 6, generator=generator, dtype=torch.float16)
    kv_state, score_state = module.initial_state(1)
    angles = torch.randn(1, 2, 2, generator=generator, dtype=torch.float16)
    validity, new_count, offset, phase_indices = build_compressor_host_transition(
        input_sequence_length=256,
        past_length=0,
        current_length=78,
        ratio=128,
        dtype=torch.float16,
    )

    output = module(
        hidden_states,
        kv_state,
        score_state,
        offset,
        phase_indices,
        torch.tensor([78]),
        validity,
        new_count,
        angles.cos(),
        angles.sin(),
    )

    assert output.new_count.item() == 0
    assert not output.compressed_valid.any()
    assert torch.equal(output.compressed, torch.zeros_like(output.compressed))


def test_normalized_hadamard_preserves_dot_products() -> None:
    generator = torch.Generator().manual_seed(61)
    query = torch.randn(2, 4, 8, generator=generator, dtype=torch.float16)
    key = torch.randn(2, 5, 8, generator=generator, dtype=torch.float16)

    original = torch.einsum("bqd,bkd->bqk", query, key)
    rotated = torch.einsum(
        "bqd,bkd->bqk",
        normalized_hadamard(query),
        normalized_hadamard(key),
    )

    # The deployment matrix is explicitly FP16; orthogonality is therefore
    # preserved to FP16 rounding rather than an FP32 butterfly tolerance.
    torch.testing.assert_close(rotated, original, rtol=2e-3, atol=4e-3)


def _butterfly_hadamard_reference(x: torch.Tensor) -> torch.Tensor:
    work = x.float()
    width = work.shape[-1]
    stride = 1
    while stride < width:
        grouped = work.reshape(*work.shape[:-1], width // (2 * stride), 2, stride)
        left = grouped[..., 0, :]
        right = grouped[..., 1, :]
        work = torch.cat((left + right, left - right), dim=-1).reshape_as(work)
        stride *= 2
    return (work * (width**-0.5)).type_as(x)


def test_h128_matrix_matches_butterfly_and_exports_one_linear() -> None:
    generator = torch.Generator().manual_seed(63)
    value = torch.randn(1, 5, 7, 128, generator=generator, dtype=torch.float16)
    module = NormalizedHadamard(128).eval()

    actual = module(value)
    expected = _butterfly_hadamard_reference(value)
    graph = torch.export.export(module, (value,))

    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)
    assert isinstance(module.matmul, torch.nn.Linear)
    assert module.matmul.bias is None
    assert not module.matmul.weight.requires_grad
    assert sum(node.target == torch.ops.aten.linear.default for node in graph.graph.nodes) == 1
    assert not any(
        node.target
        in {
            torch.ops.aten.add.Tensor,
            torch.ops.aten.sub.Tensor,
            torch.ops.aten.cat.default,
            torch.ops.aten.reshape.default,
        }
        for node in graph.graph.nodes
    )


def test_static_indexer_first_chunk_causality_and_cache_update() -> None:
    generator = torch.Generator().manual_seed(67)
    module = StaticCSAIndexer(
        hidden_size=6,
        q_lora_rank=5,
        input_sequence_length=8,
        cache_capacity=16,
        num_heads=4,
        head_dim=8,
        topk=4,
        rope_dim=4,
    )
    hidden_states = torch.randn(1, 8, 6, generator=generator, dtype=torch.float16)
    q_residual = torch.randn(1, 8, 5, generator=generator, dtype=torch.float16)
    key_cache = torch.zeros(1, 1, 16, 8, dtype=torch.float16)
    kv_state, score_state = module.initial_state(1)
    query_angles = torch.randn(1, 8, 2, generator=generator, dtype=torch.float16)
    compressed_angles = torch.randn(1, 2, 2, generator=generator, dtype=torch.float16)

    output = module(
        hidden_states,
        q_residual,
        key_cache,
        torch.tensor([0]),
        torch.tensor([8]),
        *_transition(8),
        _index_validity(8),
        kv_state,
        score_state,
        query_angles.cos(),
        query_angles.sin(),
        compressed_angles.cos(),
        compressed_angles.sin(),
    )

    assert output.new_count.item() == 2
    assert output.topk_indices.shape == (1, 8, 4)
    assert _index_validity(8)[0].sum(dim=-1).tolist() == [0, 0, 0, 1, 1, 1, 1, 2]
    assert output.updated_key_cache.shape == (1, 16, 8)
    assert torch.count_nonzero(output.updated_key_cache[:, :2]) > 0
    assert torch.equal(
        output.updated_key_cache[:, 2:],
        key_cache.squeeze(1)[:, 2:],
    )


def test_compressors_and_indexer_export_as_static_graphs() -> None:
    overlap = LearnedOverlappingCompressor(
        hidden_size=6,
        head_dim=8,
        input_sequence_length=8,
        rope_dim=4,
    ).eval()
    hidden_states = torch.randn(1, 8, 6, dtype=torch.float16)
    state = overlap.initial_state(1)
    angles = torch.randn(1, 2, 2, dtype=torch.float16)
    validity, new_count, offset, phase_indices = _transition(7)
    overlap_graph = torch.export.export(
        overlap,
        (
            hidden_states,
            state[0],
            state[1],
            offset,
            phase_indices,
            torch.tensor([7]),
            validity,
            new_count,
            angles.cos(),
            angles.sin(),
        ),
    )

    indexer = StaticCSAIndexer(
        hidden_size=6,
        q_lora_rank=5,
        input_sequence_length=8,
        cache_capacity=16,
        num_heads=4,
        head_dim=8,
        topk=4,
        rope_dim=4,
    ).eval()
    index_state = indexer.initial_state(1)
    indexer_graph = torch.export.export(
        indexer,
        (
            hidden_states,
            torch.randn(1, 8, 5, dtype=torch.float16),
            torch.zeros(1, 1, 16, 8, dtype=torch.float16),
            torch.tensor([0]),
            torch.tensor([7]),
            *_transition(7),
            _index_validity(7),
            index_state[0],
            index_state[1],
            torch.randn(1, 8, 2, dtype=torch.float16).cos(),
            torch.randn(1, 8, 2, dtype=torch.float16).sin(),
            angles.cos(),
            angles.sin(),
        ),
    )

    assert any(node.target == torch.ops.xh.DynamicSlice.default for node in overlap_graph.graph.nodes)
    assert sum(node.target == torch.ops.aten.matmul.default for node in indexer_graph.graph.nodes) == 2
    # The remaining ReduceSum belongs to the C4 compressor pooling. The
    # indexer head weighting itself is the second MatMul above.
    assert sum(node.target == torch.ops.aten.sum.dim_IntList for node in indexer_graph.graph.nodes) == 1
    assert sum(node.target == torch.ops.aten.topk.default for node in indexer_graph.graph.nodes) == 1
