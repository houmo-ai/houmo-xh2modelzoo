from __future__ import annotations

import torch

from xhmodel_merak.xh_llm.models.deepseek_v4.host_masks import (
    build_compressor_host_transition,
)
from xhmodel_merak.xh_llm.models.deepseek_v4.static_cache import (
    CSAIndexerTopK,
    DeepSeekV4StaticCacheSpec,
    FixedCapacityCacheWriter,
    LatentCacheGather,
    NonOverlappingCompressorStep,
    OverlappingCompressorStep,
    update_fixed_cache,
)


def _transition(*, input_length: int, past: int, current: int, ratio: int):
    return build_compressor_host_transition(
        input_sequence_length=input_length,
        past_length=past,
        current_length=current,
        ratio=ratio,
        dtype=torch.float32,
    )


def _offline_overlap(
    kv: torch.Tensor,
    score: torch.Tensor,
    ape: torch.Tensor,
    ratio: int = 4,
) -> torch.Tensor:
    outputs = []
    head_dim = kv.shape[-1] // 2
    complete = kv.shape[1] // ratio
    score = score + ape[torch.arange(kv.shape[1]) % ratio].unsqueeze(0)
    for group in range(complete):
        current = slice(group * ratio, (group + 1) * ratio)
        values = [kv[:, current, head_dim:]]
        gates = [score[:, current, head_dim:]]
        if group > 0:
            previous = slice((group - 1) * ratio, group * ratio)
            values.insert(0, kv[:, previous, :head_dim])
            gates.insert(0, score[:, previous, :head_dim])
        candidate_values = torch.cat(values, dim=1)
        candidate_gates = torch.cat(gates, dim=1)
        outputs.append(
            torch.sum(
                candidate_values * torch.softmax(candidate_gates, dim=1),
                dim=1,
            )
        )
    if not outputs:
        return kv.new_empty((kv.shape[0], 0, head_dim))
    return torch.stack(outputs, dim=1)


def _offline_non_overlap(
    kv: torch.Tensor,
    score: torch.Tensor,
    ape: torch.Tensor,
    ratio: int,
) -> torch.Tensor:
    complete = kv.shape[1] // ratio
    if complete == 0:
        return kv.new_empty((kv.shape[0], 0, kv.shape[-1]))
    token_count = complete * ratio
    kv = kv[:, :token_count].reshape(kv.shape[0], complete, ratio, -1)
    score = score[:, :token_count] + ape[torch.arange(token_count) % ratio].unsqueeze(0)
    score = score.reshape(score.shape[0], complete, ratio, -1)
    return torch.sum(kv * torch.softmax(score, dim=2), dim=2)


def test_static_cache_spec_uses_384_swa_and_unified_main_cache() -> None:
    spec = DeepSeekV4StaticCacheSpec()
    shapes = spec.tensor_shapes()

    assert spec.swa_logical_backing_length == spec.sliding_window + spec.prefill_chunk_length == 384
    assert spec.swa_physical_length == 384
    assert spec.csa_capacity == 65536
    assert spec.hca_capacity == 2048
    assert shapes["swa_kv"] == (1, 1, 384, 512)
    assert shapes["csa_main"] == (1, 1, 65536, 512)
    assert shapes["csa_index_k"] == (1, 1, 65536, 128)
    assert shapes["hca_main"] == (1, 1, 2048, 512)
    assert "csa_k" not in shapes and "csa_v" not in shapes
    assert "hca_k" not in shapes and "hca_v" not in shapes
    assert "csa_index_v" not in shapes


def test_swa_backing_uses_window_plus_input_not_attention_view_formula() -> None:
    # window + input = 17 crosses an alignment boundary while
    # window + input - 1 does not.  This catches accidental reuse of the
    # LLMCache attention-view formula for the persistent backing allocation.
    spec = DeepSeekV4StaticCacheSpec(
        max_context_length=64,
        prefill_chunk_length=9,
        sliding_window=8,
        latent_head_dim=16,
        index_head_dim=8,
        csa_ratio=4,
        hca_ratio=8,
        index_topk=4,
    )

    assert spec.swa_logical_backing_length == 17
    assert spec.swa_physical_length == 32


def test_csa_prefill_256_builds_64_eight_candidate_outputs() -> None:
    generator = torch.Generator().manual_seed(7)
    module = OverlappingCompressorStep(input_sequence_length=256, head_dim=5)
    kv = torch.randn(1, 256, 10, generator=generator)
    score = torch.randn(1, 256, 10, generator=generator)
    ape = torch.randn(4, 10, generator=generator)
    kv_state, score_state = module.initial_state(1)
    validity, new_count, offset, phase_indices = _transition(
        input_length=256,
        past=0,
        current=256,
        ratio=4,
    )

    assert offset.dtype == torch.int32

    result = module(
        kv,
        score,
        kv_state,
        score_state,
        offset,
        phase_indices,
        torch.tensor([256]),
        validity,
        new_count,
        ape,
    )

    expected = _offline_overlap(kv, score, ape)
    assert result.pooled.shape == (1, 64, 5)
    assert result.pooled_valid.shape == (64,)
    assert result.pooled_valid.all()
    assert result.new_count.item() == 64
    torch.testing.assert_close(result.pooled, expected, rtol=1e-5, atol=1e-6)


def test_csa_and_hca_steps_export_as_fixed_graphs() -> None:
    csa = OverlappingCompressorStep(input_sequence_length=256, head_dim=8).eval()
    csa_kv = torch.randn(1, 256, 16)
    csa_score = torch.randn_like(csa_kv)
    csa_state = csa.initial_state(1)
    csa_ape = torch.randn(4, 16)
    csa_validity, csa_count, csa_offset, csa_phases = _transition(
        input_length=256,
        past=0,
        current=78,
        ratio=4,
    )
    csa_graph = torch.export.export(
        csa,
        (
            csa_kv,
            csa_score,
            csa_state[0],
            csa_state[1],
            csa_offset,
            csa_phases,
            torch.tensor([78]),
            csa_validity,
            csa_count,
            csa_ape,
        ),
    )

    hca = NonOverlappingCompressorStep(
        input_sequence_length=256,
        head_dim=8,
    ).eval()
    hca_kv = torch.randn(1, 256, 8)
    hca_score = torch.randn_like(hca_kv)
    hca_state = hca.initial_state(1)
    hca_ape = torch.randn(128, 8)
    hca_validity, hca_count, hca_offset, hca_phases = _transition(
        input_length=256,
        past=0,
        current=235,
        ratio=128,
    )
    hca_graph = torch.export.export(
        hca,
        (
            hca_kv,
            hca_score,
            hca_state[0],
            hca_state[1],
            hca_offset,
            hca_phases,
            torch.tensor([235]),
            hca_validity,
            hca_count,
            hca_ape,
        ),
    )

    dynamic_slice = torch.ops.xh.DynamicSlice.default
    assert any(node.target == dynamic_slice for node in csa_graph.graph.nodes)
    assert any(node.target == dynamic_slice for node in hca_graph.graph.nodes)
    for graph in (csa_graph, hca_graph):
        # Absolute phase coordinates are Host ABI inputs; the device graph
        # uses offset only for DynamicSlice and performs no coordinate math.
        assert not any(
            node.target
            in {
                torch.ops.aten.arange.default,
                torch.ops.aten.sub.Tensor,
                torch.ops.aten.remainder.Tensor,
            }
            for node in graph.graph.nodes
        )
        assert not any(node.target == torch.ops.aten._to_copy.default for node in graph.graph.nodes)
        # Fixed batch-1 pad buffers must be consumed directly; anchoring them
        # through ``current[:, :1, :1] * 0 + pad`` adds two redundant chains
        # for every compressor state in every layer.
        assert not any(
            node.target == torch.ops.aten.mul.Tensor
            and len(node.args) == 2
            and node.args[1] == 0.0
            for node in graph.graph.nodes
        )


def test_csa_state_survives_78_then_256_then_decode_one() -> None:
    generator = torch.Generator().manual_seed(11)
    head_dim = 3
    total = 340
    all_kv = torch.randn(1, total, 2 * head_dim, generator=generator)
    all_score = torch.randn(1, total, 2 * head_dim, generator=generator)
    ape = torch.randn(4, 2 * head_dim, generator=generator)
    prefill = OverlappingCompressorStep(input_sequence_length=256, head_dim=head_dim)
    decode = OverlappingCompressorStep(input_sequence_length=1, head_dim=head_dim)
    kv_state, score_state = prefill.initial_state(1)
    outputs = []

    cursor = 0
    for valid_length in (78, 256):
        kv = torch.zeros(1, 256, 2 * head_dim)
        score = torch.zeros_like(kv)
        kv[:, :valid_length] = all_kv[:, cursor : cursor + valid_length]
        score[:, :valid_length] = all_score[:, cursor : cursor + valid_length]
        validity, new_count, offset, phase_indices = _transition(
            input_length=256,
            past=cursor,
            current=valid_length,
            ratio=4,
        )
        result = prefill(
            kv,
            score,
            kv_state,
            score_state,
            offset,
            phase_indices,
            torch.tensor([valid_length]),
            validity,
            new_count,
            ape,
        )
        outputs.append(result.pooled[:, : result.new_count.item()])
        kv_state, score_state = (
            result.next_kv_state,
            result.next_score_state,
        )
        cursor += valid_length

    while cursor < total:
        validity, new_count, offset, phase_indices = _transition(
            input_length=1,
            past=cursor,
            current=1,
            ratio=4,
        )
        result = decode(
            all_kv[:, cursor : cursor + 1],
            all_score[:, cursor : cursor + 1],
            kv_state,
            score_state,
            offset,
            phase_indices,
            torch.tensor([1]),
            validity,
            new_count,
            ape,
        )
        outputs.append(result.pooled[:, : result.new_count.item()])
        kv_state, score_state = (
            result.next_kv_state,
            result.next_score_state,
        )
        cursor += 1

    actual = torch.cat(outputs, dim=1)
    expected = _offline_overlap(all_kv, all_score, ape)
    assert actual.shape == (1, total // 4, head_dim)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_hca_state_survives_unaligned_prefill_chunks() -> None:
    generator = torch.Generator().manual_seed(13)
    ratio = 128
    head_dim = 4
    total = 384
    all_kv = torch.randn(1, total, head_dim, generator=generator)
    all_score = torch.randn(1, total, head_dim, generator=generator)
    ape = torch.randn(ratio, head_dim, generator=generator)
    module = NonOverlappingCompressorStep(
        input_sequence_length=256,
        head_dim=head_dim,
        ratio=ratio,
    )
    kv_state, score_state = module.initial_state(1)
    outputs = []
    cursor = 0
    for valid_length in (235, 149):
        kv = torch.zeros(1, 256, head_dim)
        score = torch.zeros_like(kv)
        kv[:, :valid_length] = all_kv[:, cursor : cursor + valid_length]
        score[:, :valid_length] = all_score[:, cursor : cursor + valid_length]
        validity, new_count, offset, phase_indices = _transition(
            input_length=256,
            past=cursor,
            current=valid_length,
            ratio=ratio,
        )
        result = module(
            kv,
            score,
            kv_state,
            score_state,
            offset,
            phase_indices,
            torch.tensor([valid_length]),
            validity,
            new_count,
            ape,
        )
        outputs.append(result.pooled[:, : result.new_count.item()])
        kv_state, score_state = (
            result.next_kv_state,
            result.next_score_state,
        )
        cursor += valid_length

    actual = torch.cat(outputs, dim=1)
    expected = _offline_non_overlap(all_kv, all_score, ape, ratio)
    assert actual.shape == (1, 3, head_dim)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_fixed_cache_update_does_not_commit_padded_outputs() -> None:
    cache = torch.arange(24, dtype=torch.float32).reshape(1, 6, 4)
    values = torch.full((1, 3, 4), 99.0)

    updated = update_fixed_cache(
        cache,
        values,
        write_start=torch.tensor([2]),
        valid_count=torch.tensor([1]),
    )

    torch.testing.assert_close(updated[:, :2], cache[:, :2])
    torch.testing.assert_close(updated[:, 2], values[:, 0])
    torch.testing.assert_close(updated[:, 3:], cache[:, 3:])


def test_fixed_capacity_writer_matches_reference_and_exports_llm_cache() -> None:
    cache = torch.arange(24, dtype=torch.float32).reshape(1, 1, 6, 4)
    values = torch.full((1, 3, 4), 99.0)
    reference = update_fixed_cache(
        cache.squeeze(1).clone(),
        values,
        write_start=torch.tensor([2]),
        valid_count=torch.tensor([1]),
    )
    writer = FixedCapacityCacheWriter()

    actual = writer(
        cache.clone(),
        values,
        torch.tensor([2]),
        torch.tensor([1]),
    )
    graph = torch.export.export(
        writer,
        (
            cache.clone(),
            values,
            torch.tensor([2]),
            torch.tensor([1]),
        ),
    )

    # LLMCache preserves the committed prefix and intentionally clears the
    # unused capacity tail.  Those rows are invalid by contract and must never
    # participate in TopK/attention.
    torch.testing.assert_close(actual[:, :3], reference[:, :3])
    assert torch.count_nonzero(actual[:, 3:]) == 0
    assert any(node.target == torch.ops.xh.LLMCache.default for node in graph.graph.nodes)


def test_direct_indexer_uses_matmul_for_weighted_head_reduction() -> None:
    generator = torch.Generator().manual_seed(17)
    batch, queries, heads, dim, capacity = 1, 3, 4, 8, 32
    query = torch.randn(batch, queries, heads, dim, generator=generator, dtype=torch.float16)
    weights = torch.randn(batch, queries, heads, generator=generator, dtype=torch.float16)
    index_k = torch.randn(batch, capacity, dim, generator=generator, dtype=torch.float16)
    valid = torch.zeros(batch, queries, capacity, dtype=torch.float16)
    valid[:, 0, :3] = True
    valid[:, 1, :17] = True
    valid[:, 2, :] = True
    module = CSAIndexerTopK(topk=4, cache_capacity=capacity)

    direct_values, direct_indices = module(
        query,
        weights,
        index_k,
        valid,
    )
    graph = torch.export.export(module, (query, weights, index_k, valid))
    full_per_head = torch.matmul(
        query,
        index_k.transpose(1, 2).unsqueeze(1),
    )
    legacy_score = (
        torch.sum(
            torch.relu(full_per_head) * weights.unsqueeze(-1),
            dim=2,
        )
        * (dim**-0.5)
        * (heads**-0.5)
    )
    full_score = torch.matmul(
        weights.unsqueeze(-2),
        torch.relu(full_per_head),
    ).squeeze(-2) * (dim * heads) ** -0.5
    invalid_floor = torch.finfo(torch.float16).min
    full_score = torch.where(valid.bool(), full_score, invalid_floor)
    legacy_score = torch.where(valid.bool(), legacy_score, invalid_floor)
    full_values, full_indices = torch.topk(full_score, 4, dim=-1)
    legacy_values, legacy_indices = torch.topk(legacy_score, 4, dim=-1)

    torch.testing.assert_close(direct_values, full_values)
    assert torch.equal(direct_indices, full_indices)
    torch.testing.assert_close(direct_values, legacy_values, rtol=2e-3, atol=5e-4)
    assert torch.equal(direct_indices, legacy_indices)
    mask_control_ops = {
        str(node.target)
        for node in graph.graph.nodes
        if any(name in str(node.target).lower() for name in ("where", "less", "clamp", "clip"))
    }
    assert not mask_control_ops
    assert torch.all(direct_values[0, 0, :3] > invalid_floor)
    assert direct_values[0, 0, 3].item() == invalid_floor
    assert sum(node.target == torch.ops.aten.matmul.default for node in graph.graph.nodes) == 2
    assert sum(node.target == torch.ops.aten.sum.dim_IntList for node in graph.graph.nodes) == 0
    assert sum(node.target == torch.ops.aten.topk.default for node in graph.graph.nodes) == 1
    assert sum(node.target == torch.ops.aten.rsub.Scalar for node in graph.graph.nodes) == 1
    assert not any(
        node.target == torch.ops.aten._to_copy.default and node.kwargs.get("dtype") == torch.float32
        for node in graph.graph.nodes
    )


def test_static_csa_latent_gather_does_not_expand_full_capacity_per_query() -> None:
    module = LatentCacheGather(feature_dim=5)
    latent = torch.randn(1, 32, 5)
    indices = torch.randint(0, 32, (1, 3, 4), dtype=torch.int64)

    selected = module(latent, indices)
    graph = torch.export.export(module, (latent, indices))

    torch.testing.assert_close(selected, latent[0][indices])
    assert selected.shape == (1, 3, 4, 5)
    targets = [node.target for node in graph.graph.nodes]
    assert targets.count(torch.ops.xh.Gather.default) == 1
    assert torch.ops.aten.expand.default not in targets
