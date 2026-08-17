from __future__ import annotations

import torch
from torch import nn

from xhmodel_merak.xh_llm.models.deepseek_v4.norm import rms_norm_from_hf
from xhquant import nn as xhnn


class _SourceRMSNorm(nn.Module):
    def __init__(self, weight: torch.Tensor | None, eps: float) -> None:
        super().__init__()
        if weight is not None:
            self.weight = nn.Parameter(weight)
        self.variance_epsilon = float(eps)


def _reference(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return x * torch.rsqrt((x * x).mean(dim=-1, keepdim=True) + eps) * weight


def test_weighted_hf_rmsnorm_becomes_xh_leaf_without_copying_weight() -> None:
    generator = torch.Generator().manual_seed(211)
    source = _SourceRMSNorm(torch.randn(8, generator=generator), 1e-6)
    module = rms_norm_from_hf(source)
    x = torch.randn(2, 5, 8, generator=generator)

    assert isinstance(module, xhnn.RMSNorm)
    assert module.weight is source.weight
    torch.testing.assert_close(
        module(x),
        _reference(x, source.weight, source.variance_epsilon),
    )


def test_unweighted_hf_rmsnorm_uses_nonpersistent_ones_buffer() -> None:
    generator = torch.Generator().manual_seed(223)
    source = _SourceRMSNorm(None, 2e-6)
    module = rms_norm_from_hf(source, hidden_size=8)
    x = torch.randn(2, 5, 8, generator=generator)

    assert isinstance(module, xhnn.RMSNorm)
    assert "weight" not in module._parameters
    assert "weight" in module._buffers
    assert "weight" in module._non_persistent_buffers_set
    assert "weight" not in module.state_dict()
    torch.testing.assert_close(
        module(x),
        _reference(x, torch.ones(8), source.variance_epsilon),
    )
