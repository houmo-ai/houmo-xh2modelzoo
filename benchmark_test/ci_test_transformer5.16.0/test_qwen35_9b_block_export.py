import pytest
from _transformer516_block_export import (
    QWEN35_9B,
    run_block_export,
)


@pytest.mark.transformer516_export
@pytest.mark.qwen35_block
def test_qwen35_9b_autoround_block_export() -> None:
    run_block_export(QWEN35_9B)
