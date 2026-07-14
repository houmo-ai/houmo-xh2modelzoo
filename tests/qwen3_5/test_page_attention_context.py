"""Regression coverage for Qwen3.5 page-attention runtime context."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from xhmodel_merak.xh_llm.hmonnx.base_llm_hmonnx_model import BaseLLMHMONNXModel


class _FakePageAttentionContext:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _FakePageAttention:
    def __init__(self):
        self.contexts = []

    def set_context(self, context):
        self.contexts.append(context)


class _ClearSpy:
    def __init__(self):
        self.calls = []

    def clear(self, *, clear_disabled_reason):
        self.calls.append(clear_disabled_reason)


def _make_model(prefill_modules, decode_modules=()):
    clear_spy = _ClearSpy()
    prefill_model = SimpleNamespace(hmonnx_session=SimpleNamespace(interpreter=clear_spy))
    decode_model = SimpleNamespace(hmonnx_session=SimpleNamespace(interpreter=_ClearSpy()))
    model = SimpleNamespace(
        device=torch.device("cpu"),
        _llm_prefill=True,
        prefill_model=prefill_model,
        decode_model=decode_model,
    )
    modules_by_model = {id(prefill_model): list(prefill_modules), id(decode_model): list(decode_modules)}
    model._get_page_attention_modules = lambda active_model: modules_by_model[id(active_model)]
    return model, clear_spy


def _set_context(model, caches, block_ids, slot_mapping):
    BaseLLMHMONNXModel.set_page_attention_context(
        model,
        paged_kv_caches=caches,
        block_ids=block_ids,
        slot_mapping=slot_mapping,
        block_size=64,
    )


def test_page_attention_modules_initialize_and_unwrap_legacy_eager_session(monkeypatch):
    monkeypatch.setattr(
        "xhmodel_merak.xh_llm.hmonnx.base_llm_hmonnx_model.PageAttention",
        _FakePageAttention,
    )
    module = _FakePageAttention()
    graph_module = SimpleNamespace(
        graph=SimpleNamespace(
            nodes=[SimpleNamespace(op="call_module", target="page_attention")]
        ),
        get_submodule=lambda target: module,
    )

    class LazySession:
        def __init__(self):
            self._session = None

        def initialize(self):
            self._session = SimpleNamespace(graph_module=graph_module)

    hmonnx_model = SimpleNamespace(hmonnx_session=LazySession())

    modules = BaseLLMHMONNXModel._get_page_attention_modules(None, hmonnx_model)

    assert modules == [module]


def test_page_attention_modules_support_legacy_interpreter_node_list(monkeypatch):
    monkeypatch.setattr(
        "xhmodel_merak.xh_llm.hmonnx.base_llm_hmonnx_model.PageAttention",
        _FakePageAttention,
    )
    module = _FakePageAttention()
    eager_interpreter = SimpleNamespace(node_modules=[object(), module])
    wrapper = SimpleNamespace(
        _session=None,
        initialize=lambda: setattr(wrapper, "_session", eager_interpreter),
    )
    hmonnx_model = SimpleNamespace(hmonnx_session=wrapper)

    modules = BaseLLMHMONNXModel._get_page_attention_modules(None, hmonnx_model)

    assert modules == [module]


def test_page_attention_context_shares_stable_device_buffers_across_layers_and_requests(monkeypatch):
    """One model/stage/device owns stable metadata buffers shared by its PA layers."""
    monkeypatch.setattr(
        "xhmodel_merak.xh_llm.hmonnx.base_llm_hmonnx_model.PageAttentionContext",
        _FakePageAttentionContext,
    )
    modules = [_FakePageAttention(), _FakePageAttention()]
    model, clear_spy = _make_model(modules)
    caches = [
        SimpleNamespace(device=torch.device("cpu"), num_blocks=8),
        SimpleNamespace(device=torch.device("cpu"), num_blocks=8),
    ]

    _set_context(
        model,
        caches,
        block_ids=torch.tensor([1, 2], dtype=torch.int32),
        slot_mapping=torch.tensor([3, 4], dtype=torch.int32),
    )
    first = [module.contexts[-1] for module in modules]
    first_block_ptr = first[0].block_ids.data_ptr()
    first_slot_ptr = first[0].slot_mapping.data_ptr()

    assert first[0].block_ids is first[1].block_ids
    assert first[0].slot_mapping is first[1].slot_mapping
    assert first[0].block_ids.dtype is torch.int64
    assert first[0].slot_mapping.dtype is torch.int64
    assert first[0].block_ids.numel() == 8
    torch.testing.assert_close(first[0].block_ids[:2], torch.tensor([1, 2]))
    torch.testing.assert_close(first[0].block_ids[2:], torch.zeros(6, dtype=torch.int64))

    _set_context(
        model,
        caches,
        block_ids=torch.tensor([5, 6, 7], dtype=torch.int64),
        slot_mapping=torch.tensor([9, 10], dtype=torch.int64),
    )
    second = [module.contexts[-1] for module in modules]

    assert second[0].block_ids.data_ptr() == first_block_ptr
    assert second[0].slot_mapping.data_ptr() == first_slot_ptr
    torch.testing.assert_close(second[0].block_ids[:3], torch.tensor([5, 6, 7]))
    torch.testing.assert_close(second[0].block_ids[3:], torch.zeros(5, dtype=torch.int64))
    torch.testing.assert_close(second[0].slot_mapping, torch.tensor([9, 10]))
    assert clear_spy.calls == []


def test_page_attention_context_slot_growth_clears_only_active_graph(monkeypatch):
    """A rare metadata shape growth invalidates the graph owning the old pointer."""
    monkeypatch.setattr(
        "xhmodel_merak.xh_llm.hmonnx.base_llm_hmonnx_model.PageAttentionContext",
        _FakePageAttentionContext,
    )
    module = _FakePageAttention()
    model, prefill_clear_spy = _make_model([module])
    cache = SimpleNamespace(device=torch.device("cpu"), num_blocks=8)

    _set_context(model, [cache], torch.tensor([1]), torch.tensor([2, 3]))
    old_ptr = module.contexts[-1].slot_mapping.data_ptr()
    _set_context(model, [cache], torch.tensor([1]), torch.tensor([4, 5, 6]))

    assert module.contexts[-1].slot_mapping.data_ptr() != old_ptr
    assert prefill_clear_spy.calls == [True]
    assert model.decode_model.hmonnx_session.interpreter.calls == []


def test_page_attention_context_buffers_are_model_and_stage_local(monkeypatch):
    """Prefill, decode, and separate runtime instances never share mutable staging."""
    monkeypatch.setattr(
        "xhmodel_merak.xh_llm.hmonnx.base_llm_hmonnx_model.PageAttentionContext",
        _FakePageAttentionContext,
    )
    prefill_module = _FakePageAttention()
    decode_module = _FakePageAttention()
    model, _ = _make_model([prefill_module], [decode_module])
    cache = SimpleNamespace(device=torch.device("cpu"), num_blocks=8)

    _set_context(model, [cache], torch.tensor([1]), torch.tensor([2, 3]))
    prefill_ptrs = (
        prefill_module.contexts[-1].block_ids.data_ptr(),
        prefill_module.contexts[-1].slot_mapping.data_ptr(),
    )
    model._llm_prefill = False
    _set_context(model, [cache], torch.tensor([4]), torch.tensor([5]))
    decode_ptrs = (
        decode_module.contexts[-1].block_ids.data_ptr(),
        decode_module.contexts[-1].slot_mapping.data_ptr(),
    )
    assert decode_ptrs != prefill_ptrs

    other_module = _FakePageAttention()
    other_model, _ = _make_model([other_module])
    _set_context(other_model, [cache], torch.tensor([6]), torch.tensor([7, 8]))
    other_ptrs = (
        other_module.contexts[-1].block_ids.data_ptr(),
        other_module.contexts[-1].slot_mapping.data_ptr(),
    )
    assert other_ptrs != prefill_ptrs
    assert model._page_attention_context_device_buffers is not other_model._page_attention_context_device_buffers


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
def test_page_attention_context_device_stream_ordering_and_replay_reads_next_request(monkeypatch):
    """Target current streams order staging before capture/replay on both devices."""
    monkeypatch.setattr(
        "xhmodel_merak.xh_llm.hmonnx.base_llm_hmonnx_model.PageAttentionContext",
        _FakePageAttentionContext,
    )
    modules = [_FakePageAttention(), _FakePageAttention()]
    model, _ = _make_model(modules)
    caches = [
        SimpleNamespace(device=torch.device("cuda:0"), num_blocks=8),
        SimpleNamespace(device=torch.device("cuda:1"), num_blocks=8),
    ]
    source_device = torch.device("cuda:0")

    _set_context(
        model,
        caches,
        block_ids=torch.tensor([1, 2], dtype=torch.int64, device=source_device),
        slot_mapping=torch.tensor([3, 4], dtype=torch.int64, device=source_device),
    )
    contexts = [module.contexts[-1] for module in modules]
    assert contexts[0].block_ids.device == torch.device("cuda:0")
    assert contexts[1].block_ids.device == torch.device("cuda:1")
    first_ptrs = [(ctx.block_ids.data_ptr(), ctx.slot_mapping.data_ptr()) for ctx in contexts]

    graphs = []
    outputs = []
    for device_index, context in enumerate(contexts):
        with torch.cuda.device(device_index):
            current_stream = torch.cuda.current_stream(device_index)
            capture_stream = torch.cuda.Stream(device=device_index)
            capture_stream.wait_stream(current_stream)
            with torch.cuda.stream(capture_stream):
                context.slot_mapping * 2
            current_stream.wait_stream(capture_stream)
            capture_stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=capture_stream):
                output = context.slot_mapping * 2
            graphs.append(graph)
            outputs.append(output)

    _set_context(
        model,
        caches,
        block_ids=torch.tensor([5, 6], dtype=torch.int64, device=source_device),
        slot_mapping=torch.tensor([7, 11], dtype=torch.int64, device=source_device),
    )
    for device_index, module in enumerate(modules):
        context = module.contexts[-1]
        assert (context.block_ids.data_ptr(), context.slot_mapping.data_ptr()) == first_ptrs[device_index]
        with torch.cuda.device(device_index):
            graphs[device_index].replay()
            torch.cuda.synchronize(device_index)
            torch.testing.assert_close(outputs[device_index], torch.tensor([14, 22], device=device_index))
