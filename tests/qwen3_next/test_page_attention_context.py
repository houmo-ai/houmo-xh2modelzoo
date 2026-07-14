"""Qwen3-Next fixed-shape page-attention metadata contracts."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from xhmodel_merak.xh_llm.hmonnx.base_llm_hmonnx_model import BaseLLMHMONNXModel
from xhmodel_merak.xh_llm.models.qwen3_next.qwen3_next_hmonnx_inference import (
    XHQwen3NextHMONNXModel,
)


class _FakePagedKVCache:
    def __init__(self, *, num_blocks: int, device: torch.device):
        self.num_blocks = num_blocks
        self.device = device


def _make_runtime(monkeypatch, *, graph_length: int, prefill: bool):
    captured = []
    runtime = SimpleNamespace(
        enable_page_attention=True,
        kvcache_config=SimpleNamespace(
            kv_cache_shape=(1, 2, 256, 64),
            cache_axis=2,
            num_layers=2,
        ),
        device=torch.device("cpu"),
        _page_attention_block_size=64,
        _paged_kv_caches=None,
        _llm_prefill=prefill,
        get_input_sequence_length=lambda: graph_length,
    )

    def allocate_cache(*, num_blocks, device, **_kwargs):
        return _FakePagedKVCache(num_blocks=num_blocks, device=torch.device(device))

    def stage_context(caches, block_ids, slot_mapping, block_size):
        staged = BaseLLMHMONNXModel._stage_page_attention_context_by_device(
            runtime,
            caches,
            block_ids,
            slot_mapping,
        )
        captured.append((*staged["cpu"], block_size))

    runtime.set_page_attention_context = stage_context
    monkeypatch.setattr(
        "xhmodel_merak.xh_llm.models.qwen3_next.qwen3_next_hmonnx_inference.allocate_hmfp_paged_kv_cache",
        allocate_cache,
    )
    return runtime, captured


def _prepare(runtime, *, past: int, current: int):
    XHQwen3NextHMONNXModel.prepare_page_attention_context(
        runtime,
        past_seq_length=past,
        current_input_length=current,
    )


def test_mtp_decode_slots_keep_graph_shape_and_skip_padded_verify_tokens(monkeypatch):
    runtime, captured = _make_runtime(monkeypatch, graph_length=5, prefill=False)

    _prepare(runtime, past=17, current=1)

    slot_mapping = captured[-1][1]
    assert slot_mapping.shape == (5,)
    torch.testing.assert_close(slot_mapping, torch.tensor([17, -1, -1, -1, -1]))


def test_prefill_slots_fill_each_real_token_in_fixed_graph_chunk(monkeypatch):
    runtime, captured = _make_runtime(monkeypatch, graph_length=4, prefill=True)

    _prepare(runtime, past=0, current=4)

    slot_mapping = captured[-1][1]
    assert slot_mapping.shape == (4,)
    torch.testing.assert_close(slot_mapping, torch.tensor([0, 1, 2, 3]))


def test_page_metadata_reuses_fixed_addresses_across_mtp_decode_steps(monkeypatch):
    runtime, captured = _make_runtime(monkeypatch, graph_length=5, prefill=False)

    _prepare(runtime, past=23, current=1)
    first_block_ids, first_slots, _ = captured[-1]
    first_ptrs = (first_block_ids.data_ptr(), first_slots.data_ptr())

    _prepare(runtime, past=24, current=1)
    second_block_ids, second_slots, _ = captured[-1]

    assert second_block_ids.shape == first_block_ids.shape
    assert second_slots.shape == (5,)
    assert (second_block_ids.data_ptr(), second_slots.data_ptr()) == first_ptrs
    torch.testing.assert_close(second_slots, torch.tensor([24, -1, -1, -1, -1]))
