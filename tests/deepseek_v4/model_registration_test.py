from __future__ import annotations

from torch import nn

from xhmodel_merak.xh_llm.builder import get_config_class, get_model_class
from xhmodel_merak.xh_llm.models.deepseek_v4.deepseek_v4_model import XHDeepSeekV4Model
from xhmodel_merak.xh_llm.models.deepseek_v4.full_model import StaticDeepSeekV4Blocks
from xhmodel_merak.xh_llm.models.deepseek_v4.xh_deepseek_v4_config import (
    XHDeepSeekV4ModelConfig,
)


def _config(**kwargs) -> XHDeepSeekV4ModelConfig:
    values = {
        "model_name": "deepseek-v4-flash-first6",
        "hf_model": "/tmp/deepseek-v4-flash",
        "max_layers": 6,
    }
    values.update(kwargs)
    return XHDeepSeekV4ModelConfig(**values)


def test_model_is_registered_under_checkpoint_architecture() -> None:
    config = _config()

    assert get_model_class(config) is XHDeepSeekV4Model
    assert get_config_class(config) is XHDeepSeekV4ModelConfig


def test_first_six_export_names_flatten_every_cache_and_state() -> None:
    model = XHDeepSeekV4Model(_config())
    export = model.get_export_cfg()

    assert export["input_names"][:19] == [
        "inputs_embeds",
        "input_ids",
        "past_seq_length",
        "current_input_length",
        "last_token_index",
        "csa_write_start",
        "hca_write_start",
        "swa_attention_mask",
        "csa_index_validity",
        "csa_attention_mask",
        "hca_attention_mask",
        "csa_compressor_validity",
        "csa_compressor_new_count",
        "csa_compressor_offset",
        "csa_compressor_phase_indices",
        "hca_compressor_validity",
        "hca_compressor_new_count",
        "hca_compressor_offset",
        "hca_compressor_phase_indices",
    ]
    assert "layer_2_index_k_input" in export["input_names"]
    assert "layer_3_main_kv_state_input" in export["input_names"]
    assert "layer_4_index_kv_state_output" in export["output_names"]
    assert "layer_5_main_score_state_output" in export["output_names"]
    assert all("offset_output" not in name for name in export["output_names"])
    assert all("new_count_output" not in name for name in export["output_names"])
    assert "layer_2_main_input" in export["input_names"]
    assert all("main_k_input" not in name and "main_v_input" not in name for name in export["input_names"])
    assert len(export["input_names"]) == 19 + 2 * 1 + 2 * 7 + 2 * 4
    assert len(export["output_names"]) == 1 + 2 * 4 + 2 * 2


def test_model_config_keeps_auto_offload_disabled_by_default() -> None:
    model = XHDeepSeekV4Model(_config())

    assert model.config.enable_auto_offload is False
    assert model.cache_abi.persistent_swa_length == 384


def test_static_blocks_always_define_one_llm_tag_boundary_per_layer() -> None:
    blocks = StaticDeepSeekV4Blocks(
        [nn.Identity(), nn.Identity()],
        ("sliding_attention", "compressed_sparse_attention"),
        hc_mult=4,
    )

    assert [tag.tag_name for tag in blocks.layer_tags] == ["layer_0", "layer_1"]
    assert all(tag.tag_type == "LLM" for tag in blocks.layer_tags)


def test_full43_export_abi_covers_every_released_layer() -> None:
    model = XHDeepSeekV4Model(
        XHDeepSeekV4ModelConfig(
            model_name="deepseek-v4-flash-full43",
            hf_model="/tmp/deepseek-v4-flash",
            max_layers=43,
        )
    )
    export = model.get_export_cfg()

    assert len(model.layer_types) == 43
    assert len(model.cache_abi.csa_layers) == 21
    assert len(model.cache_abi.hca_layers) == 20
    assert len(export["input_names"]) == 19 + 2 * 1 + 21 * 7 + 20 * 4 == 248
    assert len(export["output_names"]) == 1 + 21 * 4 + 20 * 2 == 125
    assert "layer_42_index_k_input" in export["input_names"]
    assert "layer_42_index_score_state_output" in export["output_names"]
    assert "layer_41_main_score_state_output" in export["output_names"]
