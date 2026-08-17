from __future__ import annotations

import torch

from xhmodel_merak.xh_llm.models.deepseek_v4.mhc import (
    MHCPostMapping,
    MHCPreMapping,
    mhc_split_sinkhorn,
)
from xhquant import nn as xhnn


def _official_reference(
    mixes: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pre = torch.sigmoid(mixes[..., :4] * scale[0] + base[:4]) + 1e-6
    post = 2 * torch.sigmoid(mixes[..., 4:8] * scale[1] + base[4:8])
    combination = (mixes[..., 8:] * scale[2] + base[8:]).reshape(*mixes.shape[:-1], 4, 4)
    combination = torch.softmax(combination, dim=-1) + 1e-6
    combination = combination / (combination.sum(dim=-2, keepdim=True) + 1e-6)
    for _ in range(19):
        combination = combination / (combination.sum(dim=-1, keepdim=True) + 1e-6)
        combination = combination / (combination.sum(dim=-2, keepdim=True) + 1e-6)
    return pre, post, combination


def test_mhc_sinkhorn_matches_reference_in_deployment_fp16() -> None:
    generator = torch.Generator().manual_seed(19)
    mixes = torch.randn(2, 7, 24, generator=generator, dtype=torch.float16) * 20
    scale = torch.tensor([2.078125, 0.0194091796875, 0.23828125], dtype=torch.float16)
    base = torch.randn(24, generator=generator, dtype=torch.float16)

    actual = mhc_split_sinkhorn(mixes, scale, base)
    expected = _official_reference(mixes, scale, base)

    for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
        torch.testing.assert_close(
            actual_tensor,
            expected_tensor,
            rtol=2e-3,
            atol=2e-3,
        )


def test_fused_mhc_sinkhorn_is_bitwise_equal_to_unrolled_fp16() -> None:
    generator = torch.Generator().manual_seed(20260813)
    mixes = torch.randn(2, 256, 24, generator=generator, dtype=torch.float16) * 120
    mixes[0, 0, :4] = torch.tensor([-643.0, -321.0, 0.0, 85.0], dtype=torch.float16)
    scale = torch.tensor([2.078125, 0.0194091796875, 0.23828125], dtype=torch.float16)
    base = torch.randn(24, generator=generator, dtype=torch.float16)

    expanded = mhc_split_sinkhorn(mixes, scale, base)[2]
    logits = (mixes[..., 8:] * scale[2] + base[8:]).reshape(2, 256, 4, 4)
    fused = xhnn.MHCSinkhorn(20, 1e-6)(logits)

    assert torch.equal(expanded, fused)


def test_mhc_fp16_stays_finite_for_checkpoint_scale_range() -> None:
    generator = torch.Generator().manual_seed(23)
    mixes = torch.empty(256, 24, dtype=torch.float16).uniform_(-643.5, 85.3, generator=generator)
    scale = torch.tensor([2.078125, 0.0194091796875, 0.23828125], dtype=torch.float16)
    base = torch.randn(24, generator=generator, dtype=torch.float16)

    pre, post, combination = mhc_split_sinkhorn(mixes, scale, base)

    assert torch.isfinite(pre).all()
    assert torch.isfinite(post).all()
    assert torch.isfinite(combination).all()
    assert (combination.sum(dim=-2) - 1).abs().max() < 0.01


def test_mhc_pre_and_post_mapping_shapes() -> None:
    generator = torch.Generator().manual_seed(29)
    batch, sequence, hidden = 1, 8, 16
    residual = torch.randn(batch, sequence, 4, hidden, generator=generator, dtype=torch.float16)
    hc_fn = torch.randn(24, 4 * hidden, generator=generator, dtype=torch.float16)
    hc_scale = torch.randn(3, generator=generator, dtype=torch.float16)
    hc_base = torch.randn(24, generator=generator, dtype=torch.float16)
    pre_mapping = MHCPreMapping(hidden_size=4 * hidden)
    post_mapping = MHCPostMapping()

    collapsed, post, combination = pre_mapping(
        residual,
        hc_fn,
        hc_scale,
        hc_base,
    )
    output = post_mapping(collapsed, residual, post, combination)
    expected_output = post.unsqueeze(-1) * collapsed.unsqueeze(-2) + torch.matmul(
        combination.transpose(-1, -2), residual
    )

    assert collapsed.shape == (batch, sequence, hidden)
    assert post.shape == (batch, sequence, 4)
    assert combination.shape == (batch, sequence, 4, 4)
    assert output.shape == residual.shape
    torch.testing.assert_close(output, expected_output, rtol=3e-2, atol=2e-3)


def test_mhc_pre_and_post_export_as_fixed_graphs() -> None:
    generator = torch.Generator().manual_seed(31)
    residual = torch.randn(
        1,
        256,
        4,
        32,
        dtype=torch.float16,
        generator=generator,
    )
    hc_fn = torch.randn(24, 128, dtype=torch.float16, generator=generator)
    hc_scale = torch.tensor(
        [2.078125, 0.0194091796875, 0.23828125],
        dtype=torch.float16,
    )
    hc_base = torch.randn(24, dtype=torch.float16, generator=generator)
    pre_mapping = MHCPreMapping(hidden_size=128).eval()
    post_mapping = MHCPostMapping().eval()

    pre_graph = torch.export.export(
        pre_mapping,
        (residual, hc_fn, hc_scale, hc_base),
    )
    collapsed, post, combination = pre_mapping(
        residual,
        hc_fn,
        hc_scale,
        hc_base,
    )
    post_graph = torch.export.export(
        post_mapping,
        (collapsed, residual, post, combination),
    )

    targets = [node.target for node in pre_graph.graph.nodes if node.op == "call_function"]
    assert targets.count(torch.ops.xh.MHCSinkhorn.default) == 1
    assert len(list(pre_graph.graph.nodes)) < 60
    assert len(list(post_graph.graph.nodes)) > 10
