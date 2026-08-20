from __future__ import annotations

import pytest
import torch
from transformers.cache_utils import Cache

from xhmodel_merak.xh_llm.models.minicpm_o_4_5.runtime_cache import (
    FixedCapacityDynamicCache,
    bind_cache_tensors,
)


def _single_layer_cache() -> FixedCapacityDynamicCache:
    return bind_cache_tensors(
        [torch.zeros((1, 2, 4, 4), dtype=torch.float32)],
        [torch.zeros((1, 2, 4, 4), dtype=torch.float32)],
        capacity=4,
        component="llm",
    )


def test_bind_cache_tensors_returns_fixed_capacity_hf_cache() -> None:
    cache = bind_cache_tensors(
        [torch.zeros((1, 2, 4, 4))],
        [torch.zeros((1, 2, 4, 4))],
        capacity=4,
        component="llm",
    )
    assert isinstance(cache, FixedCapacityDynamicCache)
    assert isinstance(cache, Cache)
    assert cache.get_max_cache_shape() == 4


def test_fixed_cache_rejects_overflow() -> None:
    cache = bind_cache_tensors(
        [torch.zeros((1, 2, 4, 4))],
        [torch.zeros((1, 2, 4, 4))],
        capacity=4,
        component="llm",
    )
    with pytest.raises(RuntimeError, match=r"llm.*current=0.*requested=5.*capacity=4"):
        cache.update(torch.ones((1, 2, 5, 4)), torch.ones((1, 2, 5, 4)), 0)


def test_fixed_cache_appends_into_backing_and_exposes_valid_views() -> None:
    backing_key = torch.zeros((1, 2, 4, 4), dtype=torch.float32)
    cache = bind_cache_tensors(
        [backing_key],
        [torch.zeros((1, 2, 4, 4), dtype=torch.float32)],
        capacity=4,
        component="llm",
    )
    keys, values = cache.update(torch.ones((1, 2, 2, 4)), torch.full((1, 2, 2, 4), 2.0), 0)

    assert cache.get_seq_length() == 2
    # returned tensors are step-length views into the backing storage
    assert keys.shape == (1, 2, 2, 4)
    assert values.shape == (1, 2, 2, 4)

    k_view, v_view = cache.key_cache[0], cache.value_cache[0]
    assert k_view.shape == (1, 2, 2, 4)
    assert torch.equal(k_view, torch.ones((1, 2, 2, 4)))
    assert torch.equal(v_view, torch.full((1, 2, 2, 4), 2.0))

    # region beyond the committed length stays zero in the backing buffer
    assert torch.all(backing_key[:, :, 2:4, :] == 0)


def test_fixed_cache_commits_length_only_after_last_layer_update() -> None:
    cache = bind_cache_tensors(
        [torch.zeros((1, 2, 4, 4)), torch.zeros((1, 2, 4, 4))],
        [torch.zeros((1, 2, 4, 4)), torch.zeros((1, 2, 4, 4))],
        capacity=4,
        component="llm",
    )
    layer0_keys = torch.full((1, 2, 2, 4), 1.0)
    layer1_keys = torch.full((1, 2, 2, 4), 2.0)

    keys0, _ = cache.update(layer0_keys, layer0_keys, 0)
    # layer 0 is not the last layer -> shared length is not yet committed
    assert cache.get_seq_length() == 0
    assert keys0.shape == (1, 2, 2, 4)

    cache.update(layer1_keys, layer1_keys, 1)
    assert cache.get_seq_length() == 2
    assert torch.equal(cache.key_cache[0], layer0_keys)
    assert torch.equal(cache.key_cache[1], layer1_keys)


def test_fixed_cache_crop_zeroes_removed_region_and_truncates_views() -> None:
    backing_key = torch.zeros((1, 2, 4, 4), dtype=torch.float32)
    cache = bind_cache_tensors(
        [backing_key],
        [torch.zeros((1, 2, 4, 4), dtype=torch.float32)],
        capacity=4,
        component="llm",
    )
    cache.update(torch.ones((1, 2, 4, 4)), torch.ones((1, 2, 4, 4)), 0)
    assert cache.get_seq_length() == 4

    cache.crop(2)

    assert cache.get_seq_length() == 2
    assert cache.key_cache[0].shape == (1, 2, 2, 4)
    assert torch.all(backing_key[:, :, 2:4, :] == 0)


def test_fixed_cache_reset_clears_valid_length_and_views() -> None:
    cache = _single_layer_cache()
    cache.update(torch.ones((1, 2, 4, 4)), torch.ones((1, 2, 4, 4)), 0)
    assert cache.get_seq_length() == 4

    cache.reset()

    assert cache.get_seq_length() == 0
    assert cache.key_cache[0].shape == (1, 2, 0, 4)
    assert cache.value_cache[0].shape == (1, 2, 0, 4)


def test_fixed_cache_to_legacy_cache_returns_valid_region_pairs() -> None:
    cache = _single_layer_cache()
    cache.update(torch.ones((1, 2, 2, 4)), torch.full((1, 2, 2, 4), 3.0), 0)

    legacy = cache.to_legacy_cache()

    assert len(legacy) == 1
    key, value = legacy[0]
    assert key.shape == (1, 2, 2, 4)
    assert torch.equal(key, torch.ones((1, 2, 2, 4)))
    assert torch.equal(value, torch.full((1, 2, 2, 4), 3.0))


def test_fixed_cache_len_is_layer_count() -> None:
    cache = bind_cache_tensors(
        [torch.zeros((1, 2, 4, 4)), torch.zeros((1, 2, 4, 4))],
        [torch.zeros((1, 2, 4, 4)), torch.zeros((1, 2, 4, 4))],
        capacity=4,
        component="llm",
    )
    assert len(cache) == 2


def test_fixed_cache_getitem_returns_valid_region_pair() -> None:
    cache = _single_layer_cache()
    cache.update(torch.ones((1, 2, 2, 4)), torch.full((1, 2, 2, 4), 3.0), 0)

    key, value = cache[0]

    assert key.shape == (1, 2, 2, 4)
    assert torch.equal(key, torch.ones((1, 2, 2, 4)))
    assert torch.equal(value, torch.full((1, 2, 2, 4), 3.0))


def test_fixed_cache_getitem_out_of_range_raises() -> None:
    cache = _single_layer_cache()

    with pytest.raises(KeyError, match=r"layer"):
        cache[1]


def test_fixed_cache_iteration_yields_valid_region_pairs() -> None:
    cache = bind_cache_tensors(
        [torch.zeros((1, 2, 4, 4)), torch.zeros((1, 2, 4, 4))],
        [torch.zeros((1, 2, 4, 4)), torch.zeros((1, 2, 4, 4))],
        capacity=4,
        component="llm",
    )
    cache.update(torch.full((1, 2, 2, 4), 10.0), torch.full((1, 2, 2, 4), 20.0), 0)
    cache.update(torch.full((1, 2, 2, 4), 30.0), torch.full((1, 2, 2, 4), 40.0), 1)

    pairs = list(cache)

    assert len(pairs) == 2
    assert torch.equal(pairs[0][0], torch.full((1, 2, 2, 4), 10.0))
    assert torch.equal(pairs[1][1], torch.full((1, 2, 2, 4), 40.0))


def test_fixed_cache_max_cache_len_is_capacity() -> None:
    assert _single_layer_cache().max_cache_len == 4


def test_fixed_cache_commit_length_updates_valid_view_without_writes() -> None:
    cache = _single_layer_cache()

    cache.commit_length(3)

    assert cache.get_seq_length() == 3
    assert cache.key_cache[0].shape == (1, 2, 3, 4)
    with pytest.raises(RuntimeError, match=r"valid_length=5.*capacity=4"):
        cache.commit_length(5)


def test_fixed_cache_rejects_key_value_layer_count_mismatch() -> None:
    with pytest.raises(ValueError, match=r"layer.*1.*2"):
        bind_cache_tensors(
            [torch.zeros((1, 2, 4, 4))],
            [torch.zeros((1, 2, 4, 4)), torch.zeros((1, 2, 4, 4))],
            capacity=4,
            component="llm",
        )


def test_fixed_cache_rejects_no_layers() -> None:
    with pytest.raises(ValueError, match=r"no layers"):
        bind_cache_tensors([], [], capacity=4, component="llm")


def test_fixed_cache_rejects_capacity_shape_mismatch() -> None:
    with pytest.raises(ValueError, match=r"capacity"):
        bind_cache_tensors(
            [torch.zeros((1, 2, 6, 4))],
            [torch.zeros((1, 2, 6, 4))],
            capacity=4,
            component="llm",
        )


def test_fixed_cache_rejects_key_value_shape_mismatch() -> None:
    with pytest.raises(ValueError, match=r"shape mismatch"):
        bind_cache_tensors(
            [torch.zeros((1, 2, 4, 4))],
            [torch.zeros((1, 2, 4, 8))],
            capacity=4,
            component="llm",
        )


def test_fixed_cache_rejects_out_of_range_layer_update() -> None:
    cache = _single_layer_cache()

    with pytest.raises(RuntimeError, match=r"layer_idx=1"):
        cache.update(torch.ones((1, 2, 1, 4)), torch.ones((1, 2, 1, 4)), 1)


def test_fixed_cache_rejects_duplicate_layer_update_in_step() -> None:
    cache = bind_cache_tensors(
        [torch.zeros((1, 2, 4, 4)), torch.zeros((1, 2, 4, 4))],
        [torch.zeros((1, 2, 4, 4)), torch.zeros((1, 2, 4, 4))],
        capacity=4,
        component="llm",
    )
    cache.update(torch.ones((1, 2, 2, 4)), torch.ones((1, 2, 2, 4)), 0)

    with pytest.raises(RuntimeError, match=r"out of order"):
        cache.update(torch.ones((1, 2, 2, 4)), torch.ones((1, 2, 2, 4)), 0)


def test_fixed_cache_rejects_out_of_order_layer_update() -> None:
    cache = bind_cache_tensors(
        [torch.zeros((1, 2, 4, 4)), torch.zeros((1, 2, 4, 4))],
        [torch.zeros((1, 2, 4, 4)), torch.zeros((1, 2, 4, 4))],
        capacity=4,
        component="llm",
    )

    with pytest.raises(RuntimeError, match=r"out of order"):
        cache.update(torch.ones((1, 2, 2, 4)), torch.ones((1, 2, 2, 4)), 1)


def test_fixed_cache_reset_then_reappend_does_not_leak_stale_data() -> None:
    cache = _single_layer_cache()
    cache.update(torch.ones((1, 2, 2, 4)), torch.ones((1, 2, 2, 4)), 0)
    cache.reset()

    cache.update(torch.full((1, 2, 1, 4), 7.0), torch.full((1, 2, 1, 4), 8.0), 0)

    assert cache.get_seq_length() == 1
    assert torch.equal(cache.key_cache[0], torch.full((1, 2, 1, 4), 7.0))
    assert torch.equal(cache.value_cache[0], torch.full((1, 2, 1, 4), 8.0))
    assert cache.key_cache[0].shape == (1, 2, 1, 4)

