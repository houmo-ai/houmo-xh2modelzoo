from __future__ import annotations

import pytest

from xhmodel_merak.xh_llm.models.deepseek_v4.xh_deepseek_v4_config import (
    XHDeepSeekV4ModelConfig,
)


def test_default_config_is_full43_prefill256_decode1_max256k() -> None:
    config = XHDeepSeekV4ModelConfig(model_name="deepseek-v4-flash")

    assert config.model_type == "DeepseekV4ForCausalLM"
    assert config.batch_size == 1
    assert config.prefill_chunk_length == 256
    assert config.context_max_length == 262144
    assert config.max_pe_length == 262144
    assert config.num_logits_to_keep == 1
    assert config.use_cache is True
    assert config.enable_prefill_chunk is True
    assert config.enable_auto_offload is False
    assert config.packed_weight_only is True
    assert config.get_max_decode_layers() == -1


def test_first_six_config_is_an_explicit_prefix() -> None:
    config = XHDeepSeekV4ModelConfig(
        model_name="deepseek-v4-flash-first6",
        max_layers=6,
    )

    assert config.get_max_decode_layers() == 6


def test_auto_offload_remains_an_explicit_opt_in() -> None:
    config = XHDeepSeekV4ModelConfig(
        model_name="deepseek-v4-flash-offload",
        enable_auto_offload=True,
    )

    assert config.enable_auto_offload is True


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"batch_size": 2}, "batch_size=1"),
        ({"prefill_chunk_length": 128}, "prefill_chunk_length=256"),
        ({"context_max_length": 262145}, "context_max_length"),
        ({"context_max_length": 1024}, "512 CSA"),
        ({"max_layers": 44}, "max_layers"),
        ({"use_cache": False}, "requires cache"),
    ],
)
def test_invalid_static_export_contracts_fail_early(overrides, message) -> None:
    with pytest.raises(ValueError, match=message):
        XHDeepSeekV4ModelConfig(model_name="invalid", **overrides)
