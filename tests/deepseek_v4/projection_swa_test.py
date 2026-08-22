from __future__ import annotations

import torch
from torch import nn

from xhmodel_merak.xh_llm.models.deepseek_v4.attention import apply_partial_rope
from xhmodel_merak.xh_llm.models.deepseek_v4.host_masks import (
    build_deepseek_v4_attention_masks,
)
from xhmodel_merak.xh_llm.models.deepseek_v4.projection import (
    BoundGroupedOutputProjection,
    DeepSeekV4QKVProjection,
)
from xhmodel_merak.xh_llm.models.deepseek_v4.static_cache import (
    DeepSeekV4StaticCacheSpec,
)
from xhmodel_merak.xh_llm.models.deepseek_v4.swa import StaticSWAUpdate
from xhquant.core import HybridCacheTensor


def _tiny_masks(*, input_length: int, past: int, current: int):
    spec = DeepSeekV4StaticCacheSpec(
        max_context_length=16,
        prefill_chunk_length=8,
        sliding_window=4,
        csa_ratio=4,
        hca_ratio=4,
        index_topk=4,
    )
    return build_deepseek_v4_attention_masks(
        spec,
        input_sequence_length=input_length,
        past_length=past,
        current_length=current,
    )


class _GroupedLinear(nn.Module):
    def __init__(self, groups: int, input_width: int, output_width: int) -> None:
        super().__init__()
        self.groups = groups
        self.output_width = output_width
        self.weight = nn.Parameter(torch.randn(groups * output_width, input_width))
        self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            tuple(
                nn.functional.linear(
                    x[..., group, :],
                    self.weight[group * self.output_width : (group + 1) * self.output_width],
                )
                for group in range(self.groups)
            ),
            dim=-2,
        )


def test_shared_latent_qkv_projection_keeps_kv_compact() -> None:
    generator = torch.Generator().manual_seed(71)
    module = DeepSeekV4QKVProjection(
        hidden_size=6,
        q_lora_rank=5,
        num_heads=4,
        head_dim=8,
        rope_dim=4,
    )
    hidden_states = torch.randn(1, 7, 6, generator=generator, dtype=torch.float16)
    angles = torch.randn(1, 7, 2, generator=generator, dtype=torch.float16)

    output = module(hidden_states, angles.cos(), angles.sin())

    assert output.q_residual.shape == (1, 7, 5)
    assert output.query.shape == (1, 7, 4, 8)
    assert output.latent_kv.shape == (1, 7, 8)
    unrotated_latent = module.kv_norm(module.kv_proj(hidden_states))
    expected_latent = apply_partial_rope(
        unrotated_latent.unsqueeze(2),
        angles.cos(),
        angles.sin(),
        rope_dim=4,
    ).squeeze(2)
    torch.testing.assert_close(output.latent_kv, expected_latent)


def test_bound_output_projection_uses_grouped_modules_without_weight_expansion() -> None:
    generator = torch.Generator().manual_seed(73)
    o_a = _GroupedLinear(groups=2, input_width=16, output_width=3)
    o_b = nn.Linear(6, 7, bias=False)
    module = BoundGroupedOutputProjection(
        o_a,
        o_b,
        num_heads=4,
        head_dim=8,
        num_groups=2,
        rope_dim=4,
    )
    attention_output = torch.randn(1, 5, 4, 8, generator=generator)
    angles = torch.randn(1, 5, 2, generator=generator)

    actual = module(attention_output, angles.cos(), angles.sin())
    inverse = apply_partial_rope(
        attention_output,
        angles.cos(),
        angles.sin(),
        rope_dim=4,
        inverse=True,
    ).reshape(1, 5, 2, 16)
    expected = o_b(o_a(inverse).flatten(2))

    assert len(module.o_a_proj.projections) == 2
    torch.testing.assert_close(module.o_a_proj.projections[0].weight, o_a.weight[:3])
    torch.testing.assert_close(module.o_a_proj.projections[1].weight, o_a.weight[3:])
    assert module.o_b_proj is o_b
    torch.testing.assert_close(actual, expected)


def test_swa_update_has_one_shared_latent_cache_and_exact_per_query_windows() -> None:
    module = StaticSWAUpdate(input_sequence_length=8, window_size=4)
    latent = torch.arange(16, dtype=torch.float32).reshape(1, 8, 2)
    past_kv = torch.zeros(1, 1, module.attention_length, 2)

    output = module(
        latent,
        torch.tensor([0]),
        torch.tensor([7]),
        past_kv,
    )

    assert output.physical_kv.shape == (1, module.attention_length, 2)
    valid = _tiny_masks(input_length=8, past=0, current=7).swa_attention_mask == 0
    assert valid.shape == (1, 1, 8, module.attention_length)
    assert valid[0, 0].sum(dim=-1).tolist() == [1, 2, 3, 4, 4, 4, 4, 0]
    torch.testing.assert_close(output.physical_kv[0, 3:7, 0], torch.tensor([6.0, 8.0, 10.0, 12.0]))


def test_prefill_256_swa_cache_is_384_and_decode_cache_is_128() -> None:
    prefill = StaticSWAUpdate(input_sequence_length=256, window_size=128)
    decode = StaticSWAUpdate(
        input_sequence_length=1,
        window_size=128,
        backing_length=384,
    )

    assert prefill.attention_length == 384
    assert prefill.backing_length == 384
    assert decode.attention_length == 128
    assert decode.backing_length == 384

    output = decode(
        torch.randn(1, 1, 2),
        torch.tensor([256]),
        torch.tensor([1]),
        torch.zeros(1, 1, 384, 2),
    )
    assert output.physical_kv.shape == (1, 128, 2)


def test_swa_default_backing_is_window_plus_static_input() -> None:
    # The LLMCache attention view is align(window + input - 1), but its
    # persistent storage must independently reserve align(window + input).
    # These differ when the omitted row crosses an alignment boundary.
    module = StaticSWAUpdate(
        input_sequence_length=9,
        window_size=8,
        alignment=16,
    )

    assert module.attention_length == 16
    assert module.backing_length == 32


def test_swa_update_exports_one_llm_cache_node() -> None:
    module = StaticSWAUpdate(input_sequence_length=8, window_size=4).eval()
    graph = torch.export.export(
        module,
        (
            torch.randn(1, 8, 2),
            torch.tensor([9]),
            torch.tensor([7]),
            torch.randn(1, 1, module.attention_length, 2),
        ),
    )

    assert sum(node.target == torch.ops.xh.LLMCache.default for node in graph.graph.nodes) == 1
    assert not any(
        node.target in {torch.ops.aten.clamp.default, torch.ops.aten.clamp.Tensor} for node in graph.graph.nodes
    )


def test_swa_hybrid_backing_rolls_across_prefill_and_repeated_decode() -> None:
    prefill = StaticSWAUpdate(input_sequence_length=8, window_size=4).eval()
    decode = StaticSWAUpdate(
        input_sequence_length=1,
        window_size=4,
        backing_length=prefill.backing_length,
    ).eval()
    kv_cache = HybridCacheTensor(torch.zeros(1, 1, prefill.backing_length, 1))

    prefill(
        torch.arange(8, dtype=torch.float32).reshape(1, 8, 1),
        torch.tensor([0]),
        torch.tensor([8]),
        kv_cache,
    )
    assert kv_cache.cache_valid_len == 8

    for step in range(6):
        output = decode(
            torch.tensor([[[100.0 + step]]]),
            torch.tensor([8 + step]),
            torch.tensor([1]),
            kv_cache,
        )
        assert kv_cache.cache_valid_len == 4
        valid = _tiny_masks(input_length=1, past=8 + step, current=1).swa_attention_mask == 0
        assert valid.sum().item() == 4
    torch.testing.assert_close(
        output.physical_kv[0, :4, 0],
        torch.tensor([102.0, 103.0, 104.0, 105.0]),
    )
