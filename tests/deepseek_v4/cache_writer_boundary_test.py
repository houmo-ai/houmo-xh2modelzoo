import pytest
import torch

from xhmodel_merak.xh_llm.models.deepseek_v4.static_cache import (
    FixedCapacityCacheWriter,
)
from xhquant.core import CacheTensor


@pytest.mark.parametrize(
    ("compressor", "ratio"),
    (("csa", 4), ("hca", 128)),
)
def test_fixed_capacity_writer_zero_then_boundary_write(
    compressor: str,
    ratio: int,
) -> None:
    """Zero-count decode steps preserve storage; the boundary writes one row."""

    del compressor
    writer = FixedCapacityCacheWriter()
    initial = torch.arange(32, dtype=torch.float16).reshape(1, 1, 8, 4)
    cache = CacheTensor(initial.clone())
    values = torch.full((1, 1, 4), 123.0, dtype=torch.float16)
    write_start = torch.tensor([2], dtype=torch.int32)

    for _ in range(ratio - 1):
        before = cache.clone()
        output = writer(
            cache,
            values,
            write_start,
            torch.tensor([0], dtype=torch.int32),
        )

        assert output.shape == (1, 8, 4)
        torch.testing.assert_close(cache, before, rtol=0, atol=0)

    before_boundary = cache.clone()
    output = writer(
        cache,
        values,
        write_start,
        torch.tensor([1], dtype=torch.int32),
    )

    assert output.shape == (1, 8, 4)
    torch.testing.assert_close(cache[:, :, :2], before_boundary[:, :, :2], rtol=0, atol=0)
    torch.testing.assert_close(cache[:, :, 2], values, rtol=0, atol=0)
    torch.testing.assert_close(cache[:, :, 3:], before_boundary[:, :, 3:], rtol=0, atol=0)
