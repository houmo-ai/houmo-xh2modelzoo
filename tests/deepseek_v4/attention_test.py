from __future__ import annotations

import torch

from xhmodel_merak.xh_llm.models.deepseek_v4.attention import (
    CSALatentAttention,
    GroupedLatentOutputProjection,
    HCALatentAttention,
    InterleavedPartialRope,
    apply_partial_rope,
)
from xhmodel_merak.xh_llm.models.deepseek_v4.host_masks import (
    build_deepseek_v4_attention_masks,
)
from xhmodel_merak.xh_llm.models.deepseek_v4.static_cache import (
    DeepSeekV4StaticCacheSpec,
)


def _reference_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    valid: torch.Tensor,
    sinks: torch.Tensor,
) -> torch.Tensor:
    scores = torch.einsum("bphd,bptd->bhpt", query, key)
    scores = scores * query.shape[-1] ** -0.5
    scores = torch.where(valid.unsqueeze(1), scores, torch.full_like(scores, -65504.0))
    sink_logits = sinks.reshape(1, -1, 1, 1).expand(
        scores.shape[0],
        -1,
        scores.shape[2],
        -1,
    )
    probabilities = torch.softmax(torch.cat((scores, sink_logits), dim=-1), dim=-1)[..., :-1]
    return torch.einsum("bhpt,bptd->bphd", probabilities, value)


def _additive(valid: torch.Tensor) -> torch.Tensor:
    return torch.where(valid, 0.0, -65504.0).to(torch.float16).unsqueeze(1)


def test_partial_rope_inverse_is_exact_up_to_fp_roundoff() -> None:
    generator = torch.Generator().manual_seed(37)
    x = torch.randn(1, 7, 4, 12, generator=generator)
    angles = torch.randn(1, 7, 2, generator=generator)
    cos, sin = angles.cos(), angles.sin()

    rotated = apply_partial_rope(x, cos, sin, rope_dim=4)
    recovered = apply_partial_rope(rotated, cos, sin, rope_dim=4, inverse=True)

    torch.testing.assert_close(recovered, x, rtol=1e-5, atol=1e-6)


def test_interleaved_partial_rope_uses_xh_operator_and_matches_reference() -> None:
    generator = torch.Generator().manual_seed(39)
    x = torch.randn(1, 7, 4, 12, generator=generator)
    angles = torch.randn(1, 7, 2, generator=generator)
    cos, sin = angles.cos(), angles.sin()
    module = InterleavedPartialRope(rope_dim=4).eval()

    actual = module(x, cos, sin)
    expected = apply_partial_rope(x, cos, sin, rope_dim=4)
    graph = torch.export.export(module, (x, cos, sin))

    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    assert sum(node.target == torch.ops.xh.Rope.default for node in graph.graph.nodes) == 1


def test_host_swa_mask_handles_early_context_and_padding() -> None:
    masks = build_deepseek_v4_attention_masks(
        DeepSeekV4StaticCacheSpec(),
        input_sequence_length=256,
        past_length=0,
        current_length=78,
    )
    valid = masks.swa_attention_mask == 0

    assert valid.shape == (1, 1, 256, 384)
    assert valid[0, 0, 0].sum().item() == 1
    assert valid[0, 0, 77].sum().item() == 78
    assert not valid[0, 0, 78:].any()


def test_host_swa_mask_uses_last_128_rows_at_long_context() -> None:
    masks = build_deepseek_v4_attention_masks(
        DeepSeekV4StaticCacheSpec(),
        input_sequence_length=256,
        past_length=2359,
        current_length=55,
    )
    valid = masks.swa_attention_mask == 0

    assert valid[0, 0, 0].sum().item() == 128
    assert valid[0, 0, 54].sum().item() == 128
    assert not valid[0, 0, 55:].any()


def test_csa_attention_matches_explicit_reference_with_separate_kv() -> None:
    generator = torch.Generator().manual_seed(41)
    batch, query_count, heads, dim = 1, 3, 4, 8
    query = torch.randn(batch, query_count, heads, dim, generator=generator)
    compressed_k = torch.randn(batch, query_count, 3, dim, generator=generator)
    compressed_v = torch.randn(batch, query_count, 3, dim, generator=generator)
    swa_k = torch.randn(batch, 2, dim, generator=generator)
    swa_v = torch.randn(batch, 2, dim, generator=generator)
    compressed_valid = torch.tensor([[[1, 0, 0], [1, 1, 0], [1, 1, 1]]], dtype=torch.bool)
    swa_valid = torch.tensor([[[1, 0], [1, 1], [1, 1]]], dtype=torch.bool)
    sinks = torch.randn(heads, generator=generator)
    module = CSALatentAttention(head_dim=dim, num_heads=heads)

    actual = module(
        query,
        compressed_k,
        compressed_v,
        swa_k,
        swa_v,
        _additive(torch.cat((compressed_valid, swa_valid), dim=2)),
        sinks,
    )
    expanded_swa_k = swa_k.unsqueeze(1).expand(-1, query_count, -1, -1)
    expanded_swa_v = swa_v.unsqueeze(1).expand(-1, query_count, -1, -1)
    key = torch.cat((compressed_k, expanded_swa_k), dim=2)
    value = torch.cat((compressed_v, expanded_swa_v), dim=2)
    valid = torch.cat((compressed_valid, swa_valid), dim=2)
    expected = _reference_attention(query, key, value, valid, sinks)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    assert not torch.equal(compressed_k, compressed_v)


def test_hca_attention_matches_expanded_reference_without_expanding_in_module() -> None:
    generator = torch.Generator().manual_seed(43)
    batch, query_count, heads, dim = 1, 3, 4, 8
    query = torch.randn(batch, query_count, heads, dim, generator=generator)
    compressed_k = torch.randn(batch, 5, dim, generator=generator)
    compressed_v = torch.randn(batch, 5, dim, generator=generator)
    swa_k = torch.randn(batch, 2, dim, generator=generator)
    swa_v = torch.randn(batch, 2, dim, generator=generator)
    compressed_valid = torch.tensor(
        [[[1, 0, 0, 0, 0], [1, 1, 0, 0, 0], [1, 1, 1, 0, 0]]],
        dtype=torch.bool,
    )
    swa_valid = torch.ones(batch, query_count, 2, dtype=torch.bool)
    sinks = torch.randn(heads, generator=generator)
    module = HCALatentAttention(head_dim=dim, num_heads=heads)

    actual = module(
        query,
        compressed_k,
        compressed_v,
        swa_k,
        swa_v,
        _additive(torch.cat((compressed_valid, swa_valid), dim=2)),
        sinks,
    )
    expanded_k = compressed_k.unsqueeze(1).expand(-1, query_count, -1, -1)
    expanded_v = compressed_v.unsqueeze(1).expand(-1, query_count, -1, -1)
    expanded_swa_k = swa_k.unsqueeze(1).expand(-1, query_count, -1, -1)
    expanded_swa_v = swa_v.unsqueeze(1).expand(-1, query_count, -1, -1)
    key = torch.cat((expanded_k, expanded_swa_k), dim=2)
    value = torch.cat((expanded_v, expanded_swa_v), dim=2)
    valid = torch.cat((compressed_valid, swa_valid), dim=2)
    expected = _reference_attention(query, key, value, valid, sinks)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_grouped_output_projection_matches_hf_weight_layout() -> None:
    generator = torch.Generator().manual_seed(47)
    module = GroupedLatentOutputProjection(
        num_heads=4,
        head_dim=8,
        num_groups=2,
        o_lora_rank=5,
        hidden_size=7,
        rope_dim=4,
    )
    hf_o_a = torch.randn(10, 16, generator=generator, dtype=torch.float16)
    hf_o_b = torch.randn(7, 10, generator=generator, dtype=torch.float16)
    module.load_hf_weights(hf_o_a, hf_o_b)
    attention_output = torch.randn(1, 3, 4, 8, generator=generator, dtype=torch.float16)
    angles = torch.randn(1, 3, 2, generator=generator, dtype=torch.float16)
    cos, sin = angles.cos(), angles.sin()

    actual = module(attention_output, cos, sin)
    inverse = apply_partial_rope(
        attention_output,
        cos,
        sin,
        rope_dim=4,
        inverse=True,
    )
    grouped = inverse.reshape(1, 3, 2, 16)
    low_rank = torch.stack(
        [torch.nn.functional.linear(grouped[:, :, group], hf_o_a[group * 5 : (group + 1) * 5]) for group in range(2)],
        dim=2,
    )
    expected = torch.nn.functional.linear(low_rank.flatten(2), hf_o_b)

    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)


def test_plain_grouped_output_projection_has_finite_neutral_weights() -> None:
    module = GroupedLatentOutputProjection(
        num_heads=4,
        head_dim=8,
        num_groups=2,
        o_lora_rank=5,
        hidden_size=7,
        rope_dim=4,
    )

    assert torch.equal(module.o_a_weight, torch.zeros_like(module.o_a_weight))
    assert torch.equal(module.o_b_weight, torch.zeros_like(module.o_b_weight))


def test_csa_attention_export_consumes_host_mask_without_graph_comparisons() -> None:
    attention = CSALatentAttention(head_dim=8, num_heads=4).eval()
    query = torch.randn(1, 8, 4, 8)
    compressed_k = torch.randn(1, 8, 3, 8)
    compressed_v = torch.randn_like(compressed_k)
    compressed_valid = torch.ones(1, 8, 3, dtype=torch.bool)
    swa_k = torch.randn(1, 4, 8)
    swa_v = torch.randn_like(swa_k)
    swa_valid = torch.ones(1, 8, 4, dtype=torch.bool)
    attention_graph = torch.export.export(
        attention,
        (
            query,
            compressed_k,
            compressed_v,
            swa_k,
            swa_v,
            _additive(torch.cat((compressed_valid, swa_valid), dim=2)),
            torch.randn(4),
        ),
    )

    assert not any(
        node.target
        in {
            torch.ops.aten.clamp.default,
            torch.ops.aten.clamp.Tensor,
            torch.ops.aten.where.self,
            torch.ops.aten.lt.Tensor,
        }
        for node in attention_graph.graph.nodes
    )
    assert any(node.target == torch.ops.xh.SinksSoftmax.default for node in attention_graph.graph.nodes)
