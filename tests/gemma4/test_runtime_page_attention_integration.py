"""Real Gemma4 HMONNX runtime integration for contract-v2 PageAttention."""

from __future__ import annotations

from types import SimpleNamespace

import onnx
import pytest
import torch
from onnx import TensorProto, helper
from torch import fx, nn

import xhmodel_merak.xh_llm.models.gemma4_series.gemma4_series_hmonnx_inference as gemma_runtime
from xhmodel_merak.xh_llm.models.gemma4_series.data_preprocess import (
    Gemma4DataPreprocess,
    Gemma4PerLayerInputEmbedding,
)
from xhmodel_merak.xh_llm.models.gemma4_series.gemma4_series_hmonnx_inference import (
    XHGemma4SeriesHMONNXModel,
)
from xhquant.common import PrecisionMode
from xhquant.xhonnxruntime.convert_to_page_attention import (
    convert_to_page_attention,
)
from xhquant.xhonnxruntime.hmonnx_inference_v2 import (
    CudaGraphGraphModuleInterpreter,
)
from xhquant.xhonnxruntime.parsers.page_attention import PageAttention, PageAttentionContext


def test_series_propagates_cuda_graph_to_fixed_shape_visual_runtimes(monkeypatch):
    visual_calls = []

    def fake_parent_init(self, meta_info, **kwargs):
        self.meta_info = meta_info
        self.kvcache_config = SimpleNamespace()
        self.prefill_model = SimpleNamespace()
        self.decode_model = SimpleNamespace()

    class FakeVisual:
        def __init__(self, path, **kwargs):
            visual_calls.append((path, kwargs))

    monkeypatch.setattr(gemma_runtime.VisonLLMHMONNXModel, "__init__", fake_parent_init)
    monkeypatch.setattr(gemma_runtime, "Gemma4VisualHMONNXModel", FakeVisual)
    monkeypatch.setattr(
        gemma_runtime,
        "Gemma4KVCacheMixinHMONNX",
        lambda *_args, **_kwargs: SimpleNamespace(),
    )
    meta = SimpleNamespace(
        visual_config=SimpleNamespace(hmonnx="visual.onnx"),
        video_visual_config=SimpleNamespace(hmonnx="video.onnx"),
        audio_config=None,
        model_config=SimpleNamespace(prefill_chunk_length=320),
        layer_kv_shapes=[],
    )

    XHGemma4SeriesHMONNXModel(meta, enable_cuda_graph=True)

    assert visual_calls == [
        ("visual.onnx", {"enable_cuda_graph": True}),
        ("video.onnx", {"enable_cuda_graph": True}),
    ]


def _page_attention_module() -> PageAttention:
    module = object.__new__(PageAttention)
    nn.Module.__init__(module)
    module._page_attention_context = None
    return module


def _runtime(*, layer_cache_indices=(0, 0, 1), prefill_modules=()):
    runtime = object.__new__(XHGemma4SeriesHMONNXModel)
    runtime._llm_prefill = True
    runtime.enable_page_attention = True
    runtime.layer_cache_indices = list(layer_cache_indices)
    runtime.layer_types = [
        "sliding_attention",
        "sliding_attention",
        "full_attention",
    ]
    runtime.layer_cache_types = ["sliding_attention", "full_attention"]
    runtime.meta_info = SimpleNamespace(attention_contract_version=2)
    runtime.prefill_model = SimpleNamespace(
        enable_cuda_graph=True,
        hmonnx_session=SimpleNamespace(interpreter=None),
    )
    runtime.decode_model = SimpleNamespace(
        enable_cuda_graph=True,
        hmonnx_session=SimpleNamespace(interpreter=None),
    )
    runtime._active_prefill_model = runtime.prefill_model
    runtime._get_page_attention_modules = lambda active: (
        list(prefill_modules) if active is runtime.prefill_model else []
    )
    return runtime


def _context(cache_name: str, *, base: int | None) -> PageAttentionContext:
    return PageAttentionContext(
        paged_kv_cache=SimpleNamespace(name=cache_name),
        block_ids=torch.tensor([1, 2], dtype=torch.int64),
        slot_mapping=torch.tensor([64, 65], dtype=torch.int64),
        block_size=64,
        kv_window_start_abs=(torch.tensor([base], dtype=torch.int64) if base is not None else None),
        seq_lens=torch.tensor([66], dtype=torch.int64),
        mm_prefix_ranges=(torch.tensor([[[40, 319], [0, 0]]], dtype=torch.int32) if base is not None else None),
    )


def test_real_gemma_runtime_contract_v2_binds_contexts_by_authoritative_cache_index():
    modules = [_page_attention_module() for _ in range(3)]
    runtime = _runtime(prefill_modules=modules)
    local = _context("local", base=17)
    full = _context("full", base=None)

    runtime.set_page_attention_context(contexts_by_cache_index={0: local, 1: full})

    assert modules[0]._get_context() is local
    assert modules[1]._get_context() is local
    assert modules[2]._get_context() is full
    assert modules[2]._get_context().block_ids is not local.block_ids
    assert modules[0]._get_context().paged_kv_cache is modules[1]._get_context().paged_kv_cache
    assert modules[0]._get_context().kv_window_start_abs.tolist() == [17]
    assert modules[0]._get_context().seq_lens.tolist() == [66]
    assert modules[0]._get_context().mm_prefix_ranges.tolist() == [[[40, 319], [0, 0]]]


def test_real_fx_graph_discovers_page_attention_in_call_order_and_binds_repeated_map():
    layer0, layer1, layer2 = (_page_attention_module() for _ in range(3))
    root = nn.Module()
    # Registration order is intentionally different from graph execution order.
    root.add_module("layer2", layer2)
    root.add_module("layer0", layer0)
    root.add_module("layer1", layer1)
    graph = fx.Graph()
    value = graph.placeholder("value")
    value = graph.call_module("layer0", (value,))
    value = graph.call_module("layer1", (value,))
    value = graph.call_module("layer2", (value,))
    graph.output(value)
    graph_module = fx.GraphModule(root, graph)

    runtime = _runtime(layer_cache_indices=(1, 1, 0))
    del runtime._get_page_attention_modules
    runtime.prefill_model.hmonnx_session.graph_module = graph_module
    full = _context("full", base=None)
    local = _context("local", base=17)

    discovered = runtime._get_page_attention_modules(runtime.prefill_model)
    runtime.set_page_attention_context(contexts_by_cache_index={0: full, 1: local})

    assert discovered == [layer0, layer1, layer2]
    assert layer0._get_context() is local
    assert layer1._get_context() is local
    assert layer2._get_context() is full


def test_real_gemma_runtime_rejects_physical_cache_owner_list_metadata():
    modules = [_page_attention_module() for _ in range(3)]
    runtime = _runtime(layer_cache_indices=(0, 2), prefill_modules=modules)

    with pytest.raises(ValueError, match="one physical cache index per PageAttention layer"):
        runtime.set_page_attention_context(
            contexts_by_cache_index={
                0: _context("local", base=17),
                1: _context("full", base=None),
            }
        )


def test_real_gemma_runtime_contract_v2_rejects_incomplete_or_legacy_metadata():
    modules = [_page_attention_module() for _ in range(3)]
    runtime = _runtime(prefill_modules=modules)

    with pytest.raises(ValueError, match="contexts_by_cache_index"):
        runtime.set_page_attention_context(contexts_by_cache_index={0: _context("full", base=None)})

    runtime.meta_info.attention_contract_version = 1
    with pytest.raises(RuntimeError, match="FlashAttention lowering"):
        runtime.set_page_attention_context(
            contexts_by_cache_index={
                0: _context("local", base=0),
                1: _context("full", base=None),
            }
        )


def _write_contract_v2_graph(path) -> None:
    def value(name, dtype=TensorProto.FLOAT16):
        return helper.make_tensor_value_info(name, dtype, [1])

    inputs = [
        value("inputs_embeds"),
        value("past_seq_length", TensorProto.INT32),
        value("current_input_length", TensorProto.INT32),
        value("mm_prefix_ranges", TensorProto.INT32),
        value("kv_window_start_abs", TensorProto.INT64),
        value("kv_valid_length", TensorProto.INT64),
        value("per_layer_inputs"),
        value("compact_key"),
        value("compact_value"),
    ]
    nodes = [
        helper.make_node("Identity", ["inputs_embeds"], ["projected_key"]),
        helper.make_node("Identity", ["inputs_embeds"], ["projected_value"]),
        helper.make_node(
            "KVcache",
            [
                "projected_key",
                "compact_key",
                "past_seq_length",
                "current_input_length",
            ],
            ["key_cached"],
            name="model.layers.0.key_cache",
            domain="ai.houmo.xh2a",
        ),
        helper.make_node(
            "KVcache",
            [
                "projected_value",
                "compact_value",
                "past_seq_length",
                "current_input_length",
            ],
            ["value_cached"],
            name="model.layers.0.value_cache",
            domain="ai.houmo.xh2a",
        ),
        helper.make_node(
            "FlashAttention",
            [
                "inputs_embeds",
                "key_cached",
                "value_cached",
                "table",
                "past_seq_length",
                "current_input_length",
                "",
                "mm_prefix_ranges",
                "kv_window_start_abs",
                "kv_valid_length",
            ],
            ["attention_out"],
            name="model.layers.0.flash_attention",
            domain="ai.houmo.xh2a",
            is_causal=1,
            scale=-1.0,
            num_heads=1,
            num_kv_heads=1,
            sliding_window=16,
            q_bits=8,
            k_bits=8,
            v_bits=8,
            s_bits=8,
        ),
        helper.make_node(
            "Tag",
            ["attention_out"],
            ["logits"],
            name="layer_0",
            domain="ai.houmo.xh2a",
            tag_name="layer_0",
            tag_type="LLM",
            content="layer_0",
        ),
        helper.make_node("Identity", ["per_layer_inputs"], ["per_layer_out"]),
    ]
    graph = helper.make_graph(
        nodes,
        "gemma4_runtime_input_contract",
        inputs,
        [value("logits"), value("per_layer_out")],
        initializer=[helper.make_tensor("table", TensorProto.FLOAT16, [1], [0.0])],
    )
    onnx.save(
        helper.make_model(
            graph,
            opset_imports=[
                helper.make_opsetid("", 20),
                helper.make_opsetid("ai.houmo.xh2a", 1),
            ],
        ),
        path,
    )


def test_gemma_page_attention_preprocess_matches_converted_graph_inputs_with_ple(
    tmp_path,
):
    source = tmp_path / "gemma4-flash.onnx"
    converted = tmp_path / "gemma4-page.onnx"
    _write_contract_v2_graph(source)
    converted_model = convert_to_page_attention(source, converted)
    converted_input_names = [value.name for value in converted_model.graph.input]

    embedding = nn.Embedding(32, 8)
    per_layer = Gemma4PerLayerInputEmbedding(
        vocab_size_per_layer_input=32,
        num_hidden_layers=3,
        hidden_size_per_layer_input=4,
        pad_token_id=0,
    )
    key_caches = [torch.full((1,), 11)]
    value_caches = [torch.full((1,), 13)]
    preprocess = Gemma4DataPreprocess(
        token_embedding=embedding,
        input_sequence_length=4,
        context_length=32,
        past_key_caches=key_caches,
        past_value_caches=value_caches,
        per_layer_input_embedding=per_layer,
        attention_contract_version=2,
        max_mm_ranges_per_chunk=2,
        bidirectional_vision_attention=True,
    )
    preprocess.enable_page_attention = True

    outputs = preprocess(
        {
            "input_ids": torch.tensor([[1, 2]], dtype=torch.long),
            "past_seq_length": 7,
        }
    )

    assert len(outputs) == 7
    (
        inputs_embeds,
        past_seq_length,
        current_input_length,
        mm_prefix_ranges,
        kv_window_start_abs,
        kv_valid_length,
        per_layer_inputs,
    ) = outputs
    assert inputs_embeds.shape == (1, 4, 8)
    assert past_seq_length.tolist() == [7]
    assert current_input_length.tolist() == [2]
    assert mm_prefix_ranges.tolist() == [[[0, 0], [0, 0]]]
    assert kv_window_start_abs.tolist() == [0]
    assert kv_valid_length.tolist() == [9]
    assert per_layer_inputs.shape == (1, 4, 3, 4)
    assert all(value is not key_caches and value is not value_caches for value in outputs)
    assert converted_input_names == [
        "inputs_embeds",
        "past_seq_length",
        "current_input_length",
        "mm_prefix_ranges",
        "kv_window_start_abs",
        "kv_valid_length",
        "per_layer_inputs",
    ]
    assert len(outputs) == len(converted_input_names)


def test_gemma_runtime_builds_page_attention_preprocessor_from_export_metadata():
    runtime = object.__new__(XHGemma4SeriesHMONNXModel)
    runtime.meta_info = SimpleNamespace(
        attention_contract_version=2,
        max_mm_ranges_per_chunk=2,
        model_config=SimpleNamespace(
            context_max_length=64,
            bidirectional_vision_attention=True,
            image_token_id=7,
            audio_token_id=-1,
            video_token_id=-1,
        ),
    )
    runtime.embed_tokens = nn.Embedding(32, 8)
    runtime._kvcache_mixin = SimpleNamespace(
        past_key_caches=[torch.zeros(1)],
        past_value_caches=[torch.zeros(1)],
    )
    runtime.pad_token_id = 0
    runtime.per_layer_input_embedding = None
    runtime.sliding_window = 16
    runtime.get_input_sequence_length = lambda: 4
    runtime._uses_target_verify_decode_accepted_count = lambda: False

    preprocess = runtime._get_data_preprocessor()

    assert preprocess.attention_contract_version == 2
    assert preprocess.max_mm_ranges_per_chunk == 2


class _GraphInterpreterSpy:
    def __init__(self, *, captured=False, replayed=False):
        self._capture_state = object() if captured else None
        self._capture_disabled_reason = None
        self._replay_logged = replayed
        self.clear_calls = []

    @property
    def has_captured_graph(self):
        return self._capture_state is not None

    @property
    def capture_disabled_reason(self):
        return self._capture_disabled_reason

    def clear(self, *, clear_disabled_reason=True):
        self.clear_calls.append(clear_disabled_reason)
        self._capture_state = None
        self._replay_logged = False
        if clear_disabled_reason:
            self._capture_disabled_reason = None


def test_gemma_execution_mode_drives_eager_and_exposes_graph_state():
    runtime = _runtime(prefill_modules=[])
    prefill = _GraphInterpreterSpy(captured=True, replayed=True)
    decode = _GraphInterpreterSpy(captured=True, replayed=True)
    runtime.prefill_model.hmonnx_session.interpreter = prefill
    runtime.decode_model.hmonnx_session.interpreter = decode

    runtime.set_page_attention_execution_mode("eager")
    eager_state = runtime.get_page_attention_execution_state()

    assert prefill.clear_calls == [False]
    assert eager_state == {
        "stage": "prefill",
        "requested_mode": "eager",
        "cuda_graph_enabled": True,
        "has_captured_graph": False,
        "replay_active": False,
        "clear_count": 1,
        "capture_disabled_reason": "Gemma4 PageAttention eager execution explicitly requested",
    }
    runtime.set_page_attention_execution_mode("cuda_graph")
    assert prefill.capture_disabled_reason is None

    runtime._llm_prefill = False
    runtime.set_page_attention_execution_mode("cuda_graph")
    graph_state = runtime.get_page_attention_execution_state()

    assert decode.clear_calls == []
    assert graph_state == {
        "stage": "decode",
        "requested_mode": "cuda_graph",
        "cuda_graph_enabled": True,
        "has_captured_graph": True,
        "replay_active": True,
        "clear_count": 0,
        "capture_disabled_reason": None,
    }


def test_gemma_execution_mode_rejects_unknown_value():
    runtime = _runtime(prefill_modules=[])
    runtime.prefill_model.hmonnx_session.interpreter = _GraphInterpreterSpy()

    with pytest.raises(ValueError, match="eager.*cuda_graph"):
        runtime.set_page_attention_execution_mode("graph-ish")


def test_real_cuda_graph_interpreter_preserves_capture_failure_across_eager_cycle():
    graph_module = fx.symbolic_trace(nn.Identity())
    interpreter = CudaGraphGraphModuleInterpreter(
        graph_module,
        "cpu",
        owner=SimpleNamespace(
            precision_mode=PrecisionMode.ALIGNED,
            save_golden=False,
        ),
    )
    genuine_failure = "CUDA graph capture failed: real kernel error"
    interpreter._capture_state = object()
    interpreter._replay_logged = True
    interpreter._capture_disabled_reason = genuine_failure
    runtime = _runtime(prefill_modules=[])
    runtime.prefill_model.hmonnx_session.interpreter = interpreter

    runtime.set_page_attention_execution_mode("eager")
    eager_state = runtime.get_page_attention_execution_state()
    runtime.set_page_attention_execution_mode("cuda_graph")
    graph_state = runtime.get_page_attention_execution_state()

    assert not interpreter.has_captured_graph
    assert not interpreter._replay_logged
    assert eager_state["capture_disabled_reason"] == genuine_failure
    assert graph_state["capture_disabled_reason"] == genuine_failure
