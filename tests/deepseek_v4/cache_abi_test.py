from __future__ import annotations

from xhmodel_merak.xh_llm.models.deepseek_v4.cache_abi import (
    SLIDING,
    DeepSeekV4CacheABI,
    default_layer_types,
)
from xhmodel_merak.xh_llm.models.deepseek_v4.static_cache import (
    DeepSeekV4StaticCacheSpec,
)


def test_released_43_layer_schedule_and_capacities() -> None:
    abi = DeepSeekV4CacheABI()

    assert len(default_layer_types()) == 43
    assert abi.swa_layers == (0, 1)
    assert len(abi.csa_layers) == 21
    assert len(abi.hca_layers) == 20
    assert abi.csa_layers[0] == 2 and abi.csa_layers[-1] == 42
    assert abi.spec.csa_capacity == 65536
    assert abi.spec.hca_capacity == 2048


def test_prefill_and_decode_share_one_384_row_swa_backing() -> None:
    abi = DeepSeekV4CacheABI()

    assert abi.persistent_swa_length == 384
    assert abi.spec.swa_logical_backing_length == 128 + 256
    assert abi.graph_swa_length("prefill") == 384
    assert abi.graph_swa_length("decode") == 384
    assert abi.attention_swa_length("prefill") == 384
    assert abi.attention_swa_length("decode") == 128
    assert abi.attention_mask_shapes("prefill") == {
        "swa_attention_mask": (1, 1, 256, 384),
        "csa_index_validity": (1, 256, 65536),
        "csa_attention_mask": (1, 1, 256, 896),
        "hca_attention_mask": (1, 1, 256, 2432),
    }
    assert abi.attention_mask_shapes("decode") == {
        "swa_attention_mask": (1, 1, 1, 128),
        "csa_index_validity": (1, 1, 65536),
        "csa_attention_mask": (1, 1, 1, 640),
        "hca_attention_mask": (1, 1, 1, 2176),
    }


def test_swa_backing_and_attention_view_are_independent_at_alignment_boundary() -> None:
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
    abi = DeepSeekV4CacheABI(spec=spec, layer_types=(SLIDING,))

    assert abi.graph_swa_length("prefill") == 32
    assert abi.graph_swa_length("decode") == 32
    assert abi.attention_swa_length("prefill") == 16
    assert abi.attention_swa_length("decode") == 16


def test_graph_cache_input_order_matches_each_layer_type() -> None:
    abi = DeepSeekV4CacheABI()
    names = abi.graph_cache_input_names_by_layer()

    assert len(names) == 43
    assert names[0] == ("layer_0_swa_kv_input",)
    assert names[2] == (
        "layer_2_swa_kv_input",
        "layer_2_main_input",
        "layer_2_index_k_input",
        "layer_2_main_kv_state_input",
        "layer_2_main_score_state_input",
        "layer_2_index_kv_state_input",
        "layer_2_index_score_state_input",
    )
