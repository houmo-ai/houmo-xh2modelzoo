from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from xhmodel_merak.xh_llm.models.deepseek_v4.cache_abi import (
    CSAStateOutput,
    DeepSeekV4CacheABI,
    HCAStateOutput,
    default_layer_types,
)
from xhmodel_merak.xh_llm.models.deepseek_v4.deepseek_v4_hmonnx_inference import (
    _cache_input_devices_from_layer_infos,
)
from xhmodel_merak.xh_llm.models.deepseek_v4.runtime import (
    DeepSeekV4CacheMixin,
    DeepSeekV4DataPreprocess,
)
from xhmodel_merak.xh_llm.models.deepseek_v4.static_cache import DeepSeekV4StaticCacheSpec
from xhquant.core import CacheTensor, HybridCacheTensor


def _abi() -> DeepSeekV4CacheABI:
    return DeepSeekV4CacheABI(
        spec=DeepSeekV4StaticCacheSpec(
            max_context_length=16,
            prefill_chunk_length=12,
            sliding_window=4,
            latent_head_dim=8,
            index_head_dim=8,
            csa_ratio=4,
            hca_ratio=4,
            index_topk=4,
        ),
        layer_types=default_layer_types(6),
    )


def test_runtime_allocates_one_heterogeneous_request_state() -> None:
    runtime = DeepSeekV4CacheMixin(_abi())
    caches = runtime.cache_inputs()

    assert len(caches) == 6
    assert isinstance(caches[0].swa_kv, HybridCacheTensor)
    assert caches[0].swa_kv.shape == (1, 1, 16, 8)
    assert isinstance(caches[2].main, CacheTensor)
    assert caches[2].main.shape == (1, 1, 4, 8)
    assert caches[2].index_k.shape == (1, 1, 4, 8)
    assert caches[2].main_kv_state.shape == (1, 8, 16)
    assert caches[3].main_kv_state.shape == (1, 4, 8)
    assert torch.all(caches[2].main_score_state == -65504)
    assert runtime.residency_summary()["mismatches"] == []


def test_meta_export_keeps_only_persistent_caches_on_meta() -> None:
    runtime = DeepSeekV4CacheMixin(_abi())
    with torch.device("meta"):
        runtime.prepare_kv_cache()
    caches = runtime.cache_inputs()

    assert caches[0].swa_kv.device.type == "meta"
    assert caches[2].main.device.type == "meta"
    assert caches[2].index_k.device.type == "meta"
    assert caches[2].main_kv_state.device.type == "cpu"
    assert caches[2].main_score_state.device.type == "cpu"
    assert caches[3].main_kv_state.device.type == "cpu"


def test_runtime_preserves_explicit_per_input_cache_ownership() -> None:
    abi = _abi()
    input_devices = tuple(
        (torch.device("cpu"),) * len(layer_names) for layer_names in abi.graph_cache_input_names_by_layer()
    )
    runtime = DeepSeekV4CacheMixin(abi, input_devices=input_devices)
    runtime.to("meta")

    summary = runtime.residency_summary()

    assert summary == {
        "tensor_count": 24,
        "devices": {"cpu": 24},
        "mismatches": [],
    }


def test_runtime_rejects_incomplete_cache_device_map() -> None:
    abi = _abi()
    bad_devices = [()] * len(abi.layer_types)

    with pytest.raises(ValueError, match="layer 0 cache input devices"):
        DeepSeekV4CacheMixin(abi, input_devices=bad_devices)


def test_runtime_clear_resets_request_without_reallocating_cache() -> None:
    runtime = DeepSeekV4CacheMixin(_abi())
    caches = runtime.cache_inputs()
    pointers = tuple(tensor.data_ptr() for cache in caches for tensor in cache)
    caches[0].swa_kv.fill_(1)
    caches[0].swa_kv.cache_valid_len = 9
    caches[2].main.fill_(7)
    caches[2].index_k.fill_(6)
    caches[2].main_kv_state.fill_(3)
    caches[2].main_score_state.fill_(4)
    caches[2].index_kv_state.fill_(5)
    caches[2].index_score_state.fill_(6)
    caches[3].main.fill_(7)
    caches[3].main_kv_state.fill_(8)
    caches[3].main_score_state.fill_(9)

    runtime.clear_kv_cache()

    reset_caches = runtime.cache_inputs()
    assert tuple(tensor.data_ptr() for cache in reset_caches for tensor in cache) == pointers
    assert reset_caches[0].swa_kv.cache_valid_len == 0
    assert torch.count_nonzero(reset_caches[0].swa_kv) == 0
    assert torch.count_nonzero(reset_caches[2].main) == 0
    assert torch.count_nonzero(reset_caches[2].index_k) == 0
    assert torch.count_nonzero(reset_caches[2].main_kv_state) == 0
    assert torch.all(reset_caches[2].main_score_state == -65504)
    assert torch.count_nonzero(reset_caches[2].index_kv_state) == 0
    assert torch.all(reset_caches[2].index_score_state == -65504)
    assert torch.count_nonzero(reset_caches[3].main) == 0
    assert torch.count_nonzero(reset_caches[3].main_kv_state) == 0
    assert torch.all(reset_caches[3].main_score_state == -65504)


def test_runtime_clear_releases_meta_export_cache() -> None:
    runtime = DeepSeekV4CacheMixin(_abi())
    with torch.device("meta"):
        runtime.prepare_kv_cache()

    runtime.clear_kv_cache()

    assert runtime.layer_caches == []
    assert runtime._cache_initialized is False


def test_cache_ownership_follows_stable_layer_tags() -> None:
    abi = _abi()
    layer_infos = [
        SimpleNamespace(device=torch.device(f"cuda:{layer % 2}"), llm_tags=(f"layer_{layer}",))
        for layer in range(len(abi.layer_types))
    ]
    layer_infos.append(SimpleNamespace(device=torch.device("cuda:1"), llm_tags=()))

    devices = _cache_input_devices_from_layer_infos(abi, layer_infos)

    assert len(devices) == len(abi.layer_types)
    for layer, (actual, input_names) in enumerate(
        zip(devices, abi.graph_cache_input_names_by_layer(), strict=True)
    ):
        assert actual == (torch.device(f"cuda:{layer % 2}"),) * len(input_names)


def test_cache_ownership_rejects_missing_layer_tag() -> None:
    abi = _abi()
    layer_infos = [
        SimpleNamespace(device=torch.device("cuda:0"), llm_tags=(f"layer_{layer}",))
        for layer in range(len(abi.layer_types) - 1)
    ]

    with pytest.raises(RuntimeError, match="missing auto-offload placement"):
        _cache_input_devices_from_layer_infos(abi, layer_infos)


def test_processor_pads_hash_ids_and_builds_write_starts() -> None:
    runtime = DeepSeekV4CacheMixin(_abi())
    embedding = nn.Embedding(20, 6)
    processor = DeepSeekV4DataPreprocess(
        token_embedding=embedding,
        input_sequence_length=12,
        cache_mixin=runtime,
        pad_token_id=3,
    )

    processed = processor(
        {
            "input_ids": torch.tensor([[1, 2, 4, 5, 6]]),
            "past_seq_length": 7,
        }
    )

    assert processed[0].shape == (1, 12, 6)
    assert processed[1].shape == (1, 12)
    assert processed[1].dtype == torch.int32
    assert processed[1][0, 5:].tolist() == [3] * 7
    assert processed[2].item() == 7
    assert processed[3].item() == 5
    assert processed[4].item() == 4
    assert processed[5].item() == 1
    assert processed[6].item() == 1
    assert processed[7].shape == (1, 1, 12, 16)
    assert processed[8].shape == (1, 12, 4)
    assert processed[9].shape == (1, 1, 12, 20)
    assert processed[10].shape == (1, 1, 12, 20)
    assert processed[7].dtype == processed[8].dtype == torch.float16
    assert processed[11].tolist() == [1.0, 1.0, 0.0]
    assert processed[12].item() == 2
    assert processed[13].item() == 1
    assert processed[14].tolist() == [3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2]
    assert processed[15].tolist() == [1.0, 1.0, 0.0]
    assert processed[16].item() == 2
    assert processed[17].item() == 1
    assert processed[18].tolist() == [3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2]
    assert len(processed[19]) == 24


def test_explicit_compressor_outputs_are_fed_back_per_layer() -> None:
    runtime = DeepSeekV4CacheMixin(_abi())
    caches = runtime.cache_inputs()
    original_main_state = caches[2].main_kv_state
    csa_states = tuple(
        CSAStateOutput(
            torch.full_like(caches[layer].main_kv_state, float(layer)),
            caches[layer].main_score_state,
            caches[layer].index_kv_state,
            caches[layer].index_score_state,
        )
        for layer in runtime.abi.csa_layers
    )
    hca_states = tuple(
        HCAStateOutput(
            torch.full_like(caches[layer].main_kv_state, float(layer)),
            caches[layer].main_score_state,
        )
        for layer in runtime.abi.hca_layers
    )

    runtime.apply_state_outputs(csa_states, hca_states)

    assert runtime.layer_caches[4].main_kv_state[0, 0, 0].item() == 4
    assert runtime.layer_caches[2].main_kv_state is original_main_state


def test_flat_hmonnx_outputs_commit_compressor_states_and_return_logits() -> None:
    runtime = DeepSeekV4CacheMixin(_abi())
    caches = runtime.cache_inputs()
    logits = torch.randn(1, 1, 20)
    outputs = [logits]
    for layer in runtime.abi.csa_layers:
        cache = caches[layer]
        outputs.extend(
            (
                torch.full_like(cache.main_kv_state, float(layer)),
                cache.main_score_state.clone(),
                cache.index_kv_state.clone(),
                cache.index_score_state.clone(),
            )
        )
    for layer in runtime.abi.hca_layers:
        cache = caches[layer]
        outputs.extend(
            (
                torch.full_like(cache.main_kv_state, float(layer)),
                cache.main_score_state.clone(),
            )
        )

    returned = runtime.apply_flat_state_outputs(outputs)

    assert returned is logits
    assert runtime.layer_caches[5].main_kv_state[0, 0, 0].item() == 5


def test_static_pad_token_falls_back_to_eos_for_streamed_exports() -> None:
    from xhmodel_merak.xh_llm.models.deepseek_v4.xh_deepseek_v4_config import (
        resolve_deepseek_v4_pad_token_id,
    )

    assert resolve_deepseek_v4_pad_token_id(SimpleNamespace(pad_token_id=None, eos_token_id=7)) == 7
    assert resolve_deepseek_v4_pad_token_id(SimpleNamespace(pad_token_id=None, eos_token_id=[11, 12])) == 11
    assert resolve_deepseek_v4_pad_token_id(SimpleNamespace(pad_token_id=None, eos_token_id=None)) == 0
