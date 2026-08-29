import pytest
from _transformer516_block_export import (
    GEMMA4_E2B,
    run_block_export,
)


@pytest.mark.transformer516_export
@pytest.mark.gemma4_block
def test_gemma4_e2b_autoround_block_export() -> None:
    run_block_export(GEMMA4_E2B)
