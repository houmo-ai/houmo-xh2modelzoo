"""Regression tests for the current big-model HMONNX helper."""

import base64
import json
import operator
import sys
from types import ModuleType, SimpleNamespace

import onnx
import pytest
import torch
import torch.nn as nn
from safetensors.torch import save_file

import xhmodel_merak.xh_llm.big_hf_model_helper as big_hf_model_helper
from xhmodel_merak.xh_llm.base_model import XHBaseModel
from xhmodel_merak.xh_llm.big_hf_model_helper import (
    BigHFModelExportHelper,
    WeightMapping,
    _canonicalize_merged_hmonnx_model,
    _captured_placeholder_args_template,
    _cleanup_big_model_export_temporary_files,
    _make_flattened_input_adapter,
    _merge_external_data,
    _merge_opset_imports,
    _module_name_to_filename,
    _placeholder_input_fake_tensors,
    _placeholder_output_meta,
    _serializable_meta_template,
    _validate_exported_placeholder_hmonnx,
    replace_placeholder_node,
)
from xhmodel_merak.xh_llm.hmonnx.deterministic_export import (
    _canonical_node_names,
    canonicalize_hmonnx_artifact,
)


class NamedPlaceholder(nn.Module):
    pass


class MetaDerivedBuffer(nn.Module):
    def __init__(self, size: int = 3, device=None):
        super().__init__()
        self.size = size
        self.register_buffer("derived", torch.arange(size, device=device), persistent=False)


class TinyTiedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(tie_word_embeddings=True)
        self.input_embedding = nn.Embedding(3, 2, device="meta")
        self.output_embedding = nn.Linear(2, 3, bias=False, device="meta")
        self.tie_weights()

    def get_input_embeddings(self):
        return self.input_embedding

    def get_output_embeddings(self):
        return self.output_embedding

    def tie_weights(self):
        self.output_embedding.weight = self.input_embedding.weight


def test_placeholder_contract_falls_back_to_tensor_meta_without_leaking_parent_fx_nodes():
    graph = torch.fx.Graph()
    hidden_node = graph.placeholder("hidden_states")
    input_ids_node = graph.placeholder("input_ids")
    hidden_node.meta["tensor_meta"] = SimpleNamespace(
        shape=torch.Size((1, 256, 4096)),
        dtype=torch.float16,
    )
    input_ids_node.meta["tensor_meta"] = SimpleNamespace(
        shape=torch.Size((1, 256)),
        dtype=torch.int64,
    )

    contract = _placeholder_input_fake_tensors((hidden_node, input_ids_node))

    assert [(tuple(item.shape), item.dtype) for item in contract] == [
        ((1, 256, 4096), torch.float16),
        ((1, 256), torch.int64),
    ]

    class TwoInputModule(nn.Module):
        def forward(self, hidden_states, input_ids):
            return hidden_states + input_ids.unsqueeze(-1)

    adapter = _make_flattened_input_adapter(
        TwoInputModule(),
        (hidden_node, input_ids_node),
        expected_outputs=1,
    )
    hidden = torch.zeros(1, 256, 4096)
    input_ids = torch.ones(1, 256, dtype=torch.int64)
    assert torch.equal(adapter(hidden, input_ids), torch.ones_like(hidden))


def test_placeholder_subgraph_uses_contract_captured_before_parent_graph_rewrites():
    captured = (
        torch.empty(1, 256, 4096, device="meta", dtype=torch.float16),
        torch.empty(1, 256, device="meta", dtype=torch.int64),
    )
    placeholder = nn.Module()
    placeholder.__dict__["input_args_template"] = captured
    # Simulate a later frontend transform dropping the hidden-state metadata
    # from the current parent node.  The immutable captured contract must win.
    rewritten_node = SimpleNamespace(
        args=(torch.empty(1, 256, dtype=torch.int64),),
        kwargs={},
    )

    result = _captured_placeholder_args_template(placeholder, rewritten_node)

    assert result is captured
    assert [(tuple(item.shape), item.dtype) for item in _placeholder_input_fake_tensors(result)] == [
        ((1, 256, 4096), torch.float16),
        ((1, 256), torch.int64),
    ]


def test_post_ptq_placeholder_contract_refreshes_the_complete_main_graph(tmp_path):
    from xhquant import nn as xhnn

    root = nn.Module()
    root.mlp = xhnn.PlaceHolderModule(
        output_shapes=[[1, 4]],
        output_dtypes=[torch.float32],
        num_outputs=1,
        content="mlp",
    )
    graph = torch.fx.Graph()
    graph_input = graph.placeholder("hidden_states")
    graph_input.meta["val"] = torch.empty(1, 4, device="meta", dtype=torch.float32)
    placeholder = graph.call_module("mlp", args=(graph_input,))
    placeholder.meta["val"] = [torch.empty(1, 4, device="meta", dtype=torch.float32)]
    getitem = graph.call_function(operator.getitem, args=(placeholder, 0))
    getitem.meta["val"] = torch.empty(1, 4, device="meta", dtype=torch.float32)
    downstream = graph.call_function(torch.neg, args=(getitem,))
    downstream.meta["val"] = torch.empty(1, 4, device="meta", dtype=torch.float32)
    graph.output(downstream)
    main_graph = torch.fx.GraphModule(root, graph)

    subgraph_file = tmp_path / f"{_module_name_to_filename('mlp')}.onnx"
    subgraph_input = onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 4])
    subgraph_output = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT16, [1, 4])
    cast = onnx.helper.make_node("Cast", ["input"], ["output"], to=onnx.TensorProto.FLOAT16)
    onnx.save(
        onnx.helper.make_model(
            onnx.helper.make_graph([cast], "post_ptq_subgraph", [subgraph_input], [subgraph_output]),
            opset_imports=[onnx.helper.make_opsetid("", 18)],
        ),
        subgraph_file,
    )

    helper = BigHFModelExportHelper.__new__(BigHFModelExportHelper)
    synchronized = helper.synchronize_placeholder_output_contracts(main_graph, tmp_path)

    assert synchronized == 1
    assert root.mlp.output_shapes == "[[1, 4]]"
    assert root.mlp.output_dtypes == '["float16"]'
    assert placeholder.meta["val"][0].dtype == torch.float16
    assert getitem.meta["val"].dtype == torch.float16
    assert downstream.meta["val"].dtype == torch.float16


def test_big_model_export_temporary_files_are_cleaned_by_default(monkeypatch):
    monkeypatch.delenv("XH2MODELZOO_KEEP_EXPORT_TMP", raising=False)

    assert _cleanup_big_model_export_temporary_files() is True


@pytest.mark.parametrize("value", ["1", "true", "yes", "on"])
def test_big_model_export_temporary_file_cleanup_can_be_disabled_by_environment(monkeypatch, value):
    monkeypatch.setenv("XH2MODELZOO_KEEP_EXPORT_TMP", value)

    assert _cleanup_big_model_export_temporary_files() is False


def test_placeholder_export_worker_count_uses_export_workers_environment(monkeypatch):
    monkeypatch.setenv("XH2MODELZOO_EXPORT_WORKERS", "3")

    assert big_hf_model_helper._placeholder_export_worker_count() == 3


def test_spawn_worker_refreshes_registrations_after_dynamic_model_load(monkeypatch):
    events = []
    runtime_state = {"model_loaded": False}

    class DynamicModelBigExport(BigHFModelExportHelper):
        @classmethod
        def initialize_process_worker(cls):
            events.append(("register", runtime_state["model_loaded"]))

        @classmethod
        def _preprocess_quantized_hf_model(cls, model, hf_model_dir=None, **kwargs):
            del cls, model, hf_model_dir, kwargs
            events.append(("preprocess", runtime_state["model_loaded"]))

        def _read_weight_map(self):
            events.append(("read_weight_map", runtime_state["model_loaded"]))
            return {}

    def load_dynamic_model(_hf_model_dir):
        events.append(("load_model", runtime_state["model_loaded"]))
        runtime_state["model_loaded"] = True
        model = nn.Module()
        model.config = SimpleNamespace()
        return model

    monkeypatch.setattr(big_hf_model_helper, "_PROCESS_EMPTY_HF_MODEL", None)
    monkeypatch.setattr(big_hf_model_helper, "_PROCESS_PLACEHOLDER_EXPORTER", None)

    big_hf_model_helper._init_placeholder_export_process(
        DynamicModelBigExport,
        "/tmp/dynamic-model",
        load_dynamic_model,
    )

    assert events == [
        ("register", False),
        ("load_model", False),
        ("register", True),
        ("preprocess", True),
        ("register", True),
        ("read_weight_map", True),
    ]
    assert big_hf_model_helper._PROCESS_EMPTY_HF_MODEL is not None
    assert isinstance(
        big_hf_model_helper._PROCESS_PLACEHOLDER_EXPORTER,
        DynamicModelBigExport,
    )


@pytest.mark.parametrize("value", ["1", "true", "yes", "on"])
def test_get_wrap_cfg_enables_layer_tag_from_environment(monkeypatch, value):
    monkeypatch.setenv("LAYER_TAG_ENABLE", value)
    model = object.__new__(XHBaseModel)
    model.config = SimpleNamespace()
    model.wrap_cfg = SimpleNamespace()

    wrap_cfg = model.get_wrap_cfg()

    assert wrap_cfg.enable_layer_tag is True


@pytest.mark.parametrize("quant_method", ["fake", "awq", "auto-round"])
def test_preprocess_quantized_hf_model_creates_quantizer_buffers_on_meta(monkeypatch, quant_method):
    normalized_quant_config = object()

    class FakeQuantizer:
        def __init__(self):
            self.device_map = "unexpected"
            self.quantization_config = normalized_quant_config

        def preprocess_model(self, model):
            model.register_buffer("packed_weight", torch.zeros(1024, dtype=torch.int32))

    quantizer = FakeQuantizer()
    model = nn.Module()
    model.config = SimpleNamespace(quantization_config={"quant_method": quant_method})
    monkeypatch.setattr(
        big_hf_model_helper.AutoHfQuantizer,
        "from_config",
        lambda quant_config, pre_quantized: quantizer,
    )
    monkeypatch.setattr(big_hf_model_helper.XHBaseModel, "_trim_cpu_allocator", lambda: None)

    BigHFModelExportHelper._preprocess_quantized_hf_model(model)

    assert quantizer.device_map is None
    assert model.config.quantization_config is normalized_quant_config
    assert not hasattr(model, "hf_quantizer")
    assert model.packed_weight.is_meta


@pytest.mark.parametrize("model_dtype", [torch.float16, torch.bfloat16])
def test_preprocess_gptq_uses_custom_preprocessor_for_existing_empty_model(monkeypatch, model_dtype):
    calls = []

    class FakeGPTQModelPreprocessor:
        def __init__(self, model_dir, *, dtype):
            calls.append(("init", model_dir, dtype))

        def preprocess(self, model, skip_module_prefixes=None):
            calls.append(("preprocess", model, skip_module_prefixes))
            model.register_buffer("qweight", torch.zeros(4, dtype=torch.int32, device="meta"))
            return ("layer",)

    model = nn.Module()
    model.register_parameter(
        "floating_weight",
        nn.Parameter(torch.empty(1, dtype=model_dtype, device="meta")),
    )
    model.config = SimpleNamespace(quantization_config={"quant_method": "gptq"})
    model.to_empty = lambda *args, **kwargs: pytest.fail("GPTQ preprocessing must preserve its meta skeleton")
    monkeypatch.setattr(
        big_hf_model_helper,
        "GPTQModelQuantizedModelPreprocessor",
        FakeGPTQModelPreprocessor,
    )
    monkeypatch.setattr(
        big_hf_model_helper.AutoHfQuantizer,
        "from_config",
        lambda *args, **kwargs: pytest.fail("GPTQ must not use AutoHfQuantizer"),
    )
    monkeypatch.setattr(big_hf_model_helper.XHBaseModel, "_trim_cpu_allocator", lambda: None)

    BigHFModelExportHelper._preprocess_quantized_hf_model(model, "/tmp/gptq-model")

    assert calls == [("init", "/tmp/gptq-model", model_dtype), ("preprocess", model, None)]
    assert model.qweight.is_meta


def test_gptqmodel_load_disables_quantization_disk_offload_by_default(monkeypatch):
    calls = []
    loaded_model = SimpleNamespace(
        config=SimpleNamespace(quantization_config=None),
    )

    class FakeGPTQModel:
        @staticmethod
        def load(model_dir, **kwargs):
            calls.append((model_dir, kwargs))
            return SimpleNamespace(model=loaded_model)

    fake_module = ModuleType("gptqmodel")
    fake_module.GPTQModel = FakeGPTQModel
    monkeypatch.setitem(sys.modules, "gptqmodel", fake_module)

    result = XHBaseModel._load_gptqmodel("/tmp/packed-model", device_map="cpu")

    assert result is loaded_model
    assert calls[0][0] == "/tmp/packed-model"
    assert calls[0][1]["offload_to_disk"] is False
    assert calls[0][1]["offload_to_disk_path"] is None


def test_gptqmodel_load_preserves_explicit_disk_offload_choice(monkeypatch):
    calls = []
    loaded_model = SimpleNamespace(
        config=SimpleNamespace(quantization_config=None),
    )

    class FakeGPTQModel:
        @staticmethod
        def load(model_dir, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(model=loaded_model)

    fake_module = ModuleType("gptqmodel")
    fake_module.GPTQModel = FakeGPTQModel
    monkeypatch.setitem(sys.modules, "gptqmodel", fake_module)

    XHBaseModel._load_gptqmodel(
        "/tmp/packed-model",
        device_map="cpu",
        offload_to_disk=True,
    )

    assert calls[0]["offload_to_disk"] is True


def test_prepare_loaded_gptq_module_matches_checkpoint_level_bits(monkeypatch):
    calls = []

    class FakeBaseQuantLinear(nn.Module):
        REQUIRES_FORMAT_V2 = True

        def __init__(self, device="cpu"):
            super().__init__()
            self.bits = 8
            self._qzero_format = 1
            self.register_buffer("qzeros", torch.zeros(1, dtype=torch.int32, device=device))

        def qzero_format(self):
            return self._qzero_format

    class FakeFormat:
        GPTQ = SimpleNamespace(value="gptq")

    qlinear_module = ModuleType("gptqmodel.nn_modules.qlinear")
    qlinear_module.BaseQuantLinear = FakeBaseQuantLinear
    config_module = ModuleType("gptqmodel.quantization.config")
    config_module.FORMAT = FakeFormat
    model_module = ModuleType("gptqmodel.utils.model")
    model_module.convert_gptq_v1_to_v2_format_module = lambda module, bits, pack_dtype: calls.append(
        (module, bits, pack_dtype)
    )
    monkeypatch.setitem(sys.modules, "gptqmodel.nn_modules.qlinear", qlinear_module)
    monkeypatch.setitem(sys.modules, "gptqmodel.quantization.config", config_module)
    monkeypatch.setitem(sys.modules, "gptqmodel.utils.model", model_module)

    quant_linear = FakeBaseQuantLinear()
    unloaded_quant_linear = FakeBaseQuantLinear(device="meta")
    quant_linear._logical_bits = 8
    quant_config = SimpleNamespace(format="gptq", bits=4, pack_dtype=torch.int32)

    converted = big_hf_model_helper.GPTQModelQuantizedModelPreprocessor.prepare_loaded_module(
        nn.Sequential(quant_linear, unloaded_quant_linear),
        quant_config,
    )

    assert converted == 1
    assert calls == [(quant_linear, 4, torch.int32)]


def test_new_gptq_quant_linear_uses_requested_floating_dtype():
    class FakeQuantLinear(nn.Module):
        def __init__(self, *, in_features, out_features, **kwargs):
            del kwargs
            super().__init__()
            self.register_buffer(
                "scales",
                torch.zeros(in_features, out_features, dtype=torch.float16),
            )
            self.register_buffer("qweight", torch.zeros(in_features, out_features, dtype=torch.int32))

    preprocessor = object.__new__(big_hf_model_helper.GPTQModelQuantizedModelPreprocessor)
    preprocessor.dtype = torch.bfloat16
    preprocessor.quant_config = SimpleNamespace(adapter=None)
    preprocessor.model_definition = SimpleNamespace(lm_head="lm_head")
    preprocessor._backend = SimpleNamespace(AUTO="auto")
    preprocessor._effective_quant_config = lambda module_name: {
        "bits": 4,
        "group_size": 64,
        "desc_act": False,
        "sym": True,
        "pack_dtype": torch.int32,
    }
    preprocessor._quant_linear_class = lambda module_name, linear, effective: FakeQuantLinear
    linear = nn.Linear(4, 8, bias=False, dtype=torch.bfloat16, device="meta")

    with torch.device("meta"):
        quant_linear = preprocessor._new_quant_linear("model.layers.0.q_proj", linear)

    assert quant_linear.scales.is_meta
    assert quant_linear.scales.dtype == torch.bfloat16
    assert quant_linear.qweight.is_meta
    assert quant_linear.qweight.dtype == torch.int32


def test_prepare_loaded_gptq_module_skips_placeholder_prefixes(monkeypatch):
    calls = []

    class FakeBaseQuantLinear(nn.Module):
        REQUIRES_FORMAT_V2 = True

        def __init__(self):
            super().__init__()
            self._qzero_format = 1
            self.register_buffer("qzeros", torch.zeros(1, dtype=torch.int32))

        def qzero_format(self):
            return self._qzero_format

    class FakeFormat:
        GPTQ = SimpleNamespace(value="gptq")

    qlinear_module = ModuleType("gptqmodel.nn_modules.qlinear")
    qlinear_module.BaseQuantLinear = FakeBaseQuantLinear
    config_module = ModuleType("gptqmodel.quantization.config")
    config_module.FORMAT = FakeFormat
    model_module = ModuleType("gptqmodel.utils.model")
    model_module.convert_gptq_v1_to_v2_format_module = lambda module, bits, pack_dtype: calls.append(module)
    monkeypatch.setitem(sys.modules, "gptqmodel.nn_modules.qlinear", qlinear_module)
    monkeypatch.setitem(sys.modules, "gptqmodel.quantization.config", config_module)
    monkeypatch.setitem(sys.modules, "gptqmodel.utils.model", model_module)

    model = nn.Module()
    model.main = FakeBaseQuantLinear()
    model.placeholder = nn.Sequential(FakeBaseQuantLinear())
    quant_config = SimpleNamespace(format="gptq", bits=4, pack_dtype=torch.int32)

    converted = big_hf_model_helper.GPTQModelQuantizedModelPreprocessor.prepare_loaded_module(
        model,
        quant_config,
        skip_module_prefixes=["placeholder"],
    )

    assert converted == 1
    assert calls == [model.main]


def test_dequantize_gptq_hf_model_detects_quant_linear_without_hf_quantizer(monkeypatch):
    class FakeQuantLinear(nn.Module):
        pass

    model = nn.Sequential(FakeQuantLinear())
    converted = []

    def fake_converter(module):
        converted.append(module)

    monkeypatch.setattr(
        XHBaseModel,
        "_get_gptq_quant_linear_converters",
        classmethod(lambda cls: ({FakeQuantLinear: fake_converter}, ())),
    )

    result = XHBaseModel._dequantize_gptq_hf_model(model)

    assert result is model
    assert converted == [model[0]]
    assert not hasattr(model, "hf_quantizer")


def test_dequantize_gptq_hf_model_allows_model_without_quant_linear(monkeypatch):
    model = nn.Linear(2, 2)
    monkeypatch.setattr(
        XHBaseModel,
        "_get_gptq_quant_linear_converters",
        classmethod(lambda cls: ({}, ())),
    )

    assert XHBaseModel._dequantize_gptq_hf_model(model) is model


def test_load_placeholder_dequantizes_without_root_hf_quantizer(monkeypatch):
    root_model = nn.Module()
    root_model.layer = nn.Linear(2, 2)
    root_model.config = SimpleNamespace(quantization_config={"quant_method": "gptq"})
    setattr(
        root_model.config,
        big_hf_model_helper.GPTQModelQuantizedModelPreprocessor.CONFIG_MARKER,
        True,
    )
    calls = []
    prepared = []

    big_model = BigHFModelExportHelper.__new__(BigHFModelExportHelper)
    big_model._load_module_from_safetensor = lambda module, target: {"target": target}

    def fake_dequantize(cls, model, quantization_config):
        calls.append((model, quantization_config))
        return model

    monkeypatch.setattr(XHBaseModel, "dequantize_hf_model", classmethod(fake_dequantize))
    monkeypatch.setattr(
        big_hf_model_helper.GPTQModelQuantizedModelPreprocessor,
        "prepare_loaded_module",
        classmethod(
            lambda cls, model, quantization_config, skip_module_prefixes=None: prepared.append(
                (model, quantization_config)
            )
        ),
    )

    placeholder = big_model._load_place_holder_module_once(root_model, "layer")

    assert placeholder is not root_model.layer
    assert prepared == [(placeholder, root_model.config.quantization_config)]
    assert calls == [(placeholder, root_model.config.quantization_config)]
    assert placeholder._placeholder_load_audit == {"target": "layer"}


def test_remove_buffers_missing_from_weight_map_only_skips_non_persistent_buffers():
    model = nn.Module()
    model.register_buffer("derived", torch.ones(1), persistent=False)
    buffers = {"derived": model.derived}

    BigHFModelExportHelper._remove_buffers_missing_from_weight_map(model, buffers, {})

    assert buffers == {}


def test_remove_buffers_missing_from_weight_map_rejects_missing_persistent_buffer():
    model = nn.Module()
    model.register_buffer("required", torch.ones(1))

    with pytest.raises(KeyError, match="Persistent buffer 'required'"):
        BigHFModelExportHelper._remove_buffers_missing_from_weight_map(model, {"required": model.required}, {})


def test_placeholder_prefixes_use_configured_type_name_matching():
    model = nn.Sequential(NamedPlaceholder())
    big_model = BigHFModelExportHelper.__new__(BigHFModelExportHelper)
    big_model._placeholder_type_names = ("NamedPlaceholder",)
    big_model._placeholder_types = set()

    assert big_model._placeholder_prefixes(model) == ["0"]


def test_constructor_rejects_model_without_configured_placeholder_modules():
    model = nn.Sequential(nn.Linear(2, 2))

    with pytest.raises(RuntimeError, match="found no placeholder modules"):
        BigHFModelExportHelper("/tmp/model", model, ["NamedPlaceholder"])


def test_root_placeholder_prefix_covers_entire_module_tree():
    assert BigHFModelExportHelper._within_placeholder_module("", [""])
    assert BigHFModelExportHelper._within_placeholder_module("child.proj", [""])


def test_placeholder_output_meta_rejects_nested_outputs():
    nested_output = (torch.empty(1), (torch.empty(1), torch.empty(1)))

    with pytest.raises(RuntimeError, match="Nested placeholder outputs are not supported"):
        _placeholder_output_meta(nested_output)


def test_flattened_input_adapter_trims_outputs_to_placeholder_contract():
    class ReturnsAuxiliaryOutput(nn.Module):
        def forward(self, x):
            return x + 1, x.sum(dim=-1)

    x = torch.ones(2, 3)
    adapter = _make_flattened_input_adapter(ReturnsAuxiliaryOutput(), (x,), expected_outputs=1)

    output = adapter(x)

    assert torch.equal(output, x + 1)


def test_module_filename_is_stable_and_collision_resistant():
    assert _module_name_to_filename("a.b") == _module_name_to_filename("a.b")
    assert _module_name_to_filename("a.b") != _module_name_to_filename("a_b")


def test_serializable_meta_template_removes_fx_node_and_fake_tensor_state():
    fx_node = SimpleNamespace(meta={"val": torch.ones(2, 3, dtype=torch.float16)})

    template = _serializable_meta_template((fx_node, [fx_node], None, 7))

    assert template[0].device.type == "meta"
    assert template[0].shape == (2, 3)
    assert template[0].dtype == torch.float16
    assert template[1][0].device.type == "meta"
    assert template[2:] == (None, 7)


def test_validate_exported_placeholder_hmonnx_checks_io_contract(tmp_path):
    input_info = onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1])
    output_info = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1])
    model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [onnx.helper.make_node("Identity", ["input"], ["output"])],
            "placeholder",
            [input_info],
            [output_info],
        )
    )
    model_file = tmp_path / "placeholder.onnx"
    onnx.save(model, model_file)

    result = _validate_exported_placeholder_hmonnx(model_file, expected_inputs=1, expected_outputs=1)

    assert result["inputs"] == 1
    assert result["outputs"] == 1
    with pytest.raises(RuntimeError, match="input count mismatch"):
        _validate_exported_placeholder_hmonnx(model_file, expected_inputs=2, expected_outputs=1)
    with pytest.raises(RuntimeError, match="output count mismatch"):
        _validate_exported_placeholder_hmonnx(model_file, expected_inputs=1, expected_outputs=2)


def test_parallel_placeholder_export_raises_child_process_failure(monkeypatch, tmp_path):
    class PrefillPlaceholder(nn.Module):
        pass

    class DecodePlaceholder(nn.Module):
        pass

    class FakeFuture:
        def __init__(self, exc):
            self.exc = exc
            self.cancelled = False

        def result(self):
            raise self.exc

        def cancel(self):
            self.cancelled = True
            return True

    class FakeExecutor:
        submitted_tasks = []

        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def submit(self, fn, task):
            self.submitted_tasks.append(task)
            return FakeFuture(RuntimeError("CUDA out of memory"))

    class FakeTqdm:
        descriptions = []

        def __init__(self, iterable, **kwargs):
            self.iterable = iterable

        def __iter__(self):
            return iter(self.iterable)

        def set_description(self, value):
            self.description = value
            self.descriptions.append(value)

    helper = BigHFModelExportHelper.__new__(BigHFModelExportHelper)
    helper._hf_model_dir = str(tmp_path)
    placeholder_module = SimpleNamespace(num_outputs=1)
    prefill_graph = object()
    decode_graph = object()
    nodes_by_graph = {
        prefill_graph: SimpleNamespace(
            target="model.layers.0.mlp",
            args=(torch.empty(1, device="meta"),),
            kwargs={},
            meta={"val": torch.empty(1, device="meta")},
        ),
        decode_graph: SimpleNamespace(
            target="model.layers.1.mlp",
            args=(torch.empty(1, device="meta"),),
            kwargs={},
            meta={"val": torch.empty(1, device="meta")},
        ),
    }
    module_types_by_target = {
        "model.layers.0.mlp": PrefillPlaceholder(),
        "model.layers.1.mlp": DecodePlaceholder(),
    }
    empty_model = SimpleNamespace(get_submodule=module_types_by_target.__getitem__)
    monkeypatch.setattr(
        helper,
        "_collect_place_holder_modules",
        lambda graph: [(nodes_by_graph[graph], placeholder_module)],
    )
    monkeypatch.setattr(big_hf_model_helper, "ProcessPoolExecutor", FakeExecutor)
    monkeypatch.setattr(big_hf_model_helper, "as_completed", lambda futures: iter(futures))
    monkeypatch.setattr(big_hf_model_helper, "tqdm", FakeTqdm)
    monkeypatch.setattr(big_hf_model_helper, "_placeholder_export_execution_devices", lambda: ["cuda:0", "cuda:1"])

    with pytest.raises(
        RuntimeError,
        match=r"model\.layers\.0\.mlp.*CUDA out of memory.*Reduce.*WORKERS.*GPU",
    ):
        helper.export_prefill_decode_placeholder_layers(
            prefill_fronted_graph_module=prefill_graph,
            decode_fronted_graph_module=decode_graph,
            empty_hf_model_for_placeholder=empty_model,
            target_device="cuda:0",
            prefill_wrap_cfg=SimpleNamespace(),
            decode_wrap_cfg=SimpleNamespace(),
            quant_cfg=SimpleNamespace(),
            prefill_output_dir=tmp_path / "prefill",
            decode_output_dir=tmp_path / "decode",
            placeholder_export_workers=2,
            empty_hf_model_factory=lambda hf_model_dir: object(),
        )

    assert [task["execution_device"] for task in FakeExecutor.submitted_tasks] == ["cuda:0", "cuda:1"]
    assert not any(description.startswith("exported ") for description in FakeTqdm.descriptions)


def test_placeholder_export_execution_devices_fall_back_to_cpu(monkeypatch):
    monkeypatch.setattr(big_hf_model_helper.torch.cuda, "is_available", lambda: False)

    assert big_hf_model_helper._placeholder_export_execution_devices() == ["cpu"]


def test_placeholder_export_execution_devices_use_visible_cuda_devices(monkeypatch):
    monkeypatch.setattr(big_hf_model_helper.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(big_hf_model_helper.torch.cuda, "device_count", lambda: 3)

    assert big_hf_model_helper._placeholder_export_execution_devices() == ["cuda:0", "cuda:1", "cuda:2"]


def test_set_placeholder_export_execution_device_resolves_bare_cuda(monkeypatch):
    selected_devices = []
    monkeypatch.setattr(big_hf_model_helper.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(big_hf_model_helper.torch.cuda, "current_device", lambda: 2)
    monkeypatch.setattr(big_hf_model_helper.torch.cuda, "set_device", selected_devices.append)

    resolved_device = big_hf_model_helper._set_placeholder_export_execution_device("cuda")

    assert resolved_device == "cuda:2"
    assert selected_devices == [torch.device("cuda:2")]


def test_set_placeholder_export_execution_device_preserves_explicit_cuda_index(monkeypatch):
    selected_devices = []
    monkeypatch.setattr(big_hf_model_helper.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(big_hf_model_helper.torch.cuda, "set_device", selected_devices.append)

    resolved_device = big_hf_model_helper._set_placeholder_export_execution_device(torch.device("cuda:1"))

    assert resolved_device == "cuda:1"
    assert selected_devices == [torch.device("cuda:1")]


def test_detach_placeholder_modules_replaces_all_shared_aliases():
    shared_placeholder = NamedPlaceholder()
    model = nn.Module()
    model.first = shared_placeholder
    model.second = shared_placeholder
    big_model = BigHFModelExportHelper.__new__(BigHFModelExportHelper)
    big_model._placeholder_type_names = ("NamedPlaceholder",)
    big_model._placeholder_types = {NamedPlaceholder}

    with big_model._detach_placeholder_modules(model):
        assert model.first is not shared_placeholder
        assert model.second is not shared_placeholder

    assert model.first is shared_placeholder
    assert model.second is shared_placeholder


def test_load_module_materializes_missing_non_persistent_meta_buffer():
    module = MetaDerivedBuffer(device="meta")
    big_model = BigHFModelExportHelper.__new__(BigHFModelExportHelper)
    big_model._weight_mapping = WeightMapping()
    big_model._weight_mapping.weight_map = {}

    big_model._load_module_from_safetensor(module, "model.layers.0")

    assert module.derived.device.type == "cpu"
    torch.testing.assert_close(module.derived, torch.arange(3))


def test_load_module_rejects_non_meta_parameter_missing_from_weight_map():
    module = nn.Linear(2, 2, bias=False)
    big_model = BigHFModelExportHelper.__new__(BigHFModelExportHelper)
    big_model._weight_mapping = WeightMapping()
    big_model._weight_mapping.weight_map = {}

    with pytest.raises(RuntimeError, match="parameters=model.layers.0.weight"):
        big_model._load_module_from_safetensor(module, "model.layers.0")


def test_load_module_rejects_non_meta_persistent_buffer_missing_from_weight_map():
    module = nn.Module()
    module.register_buffer("packed_weight", torch.zeros(4, dtype=torch.int32))
    big_model = BigHFModelExportHelper.__new__(BigHFModelExportHelper)
    big_model._weight_mapping = WeightMapping()
    big_model._weight_mapping.weight_map = {}

    with pytest.raises(RuntimeError, match="persistent_buffers=model.layers.0.packed_weight"):
        big_model._load_module_from_safetensor(module, "model.layers.0")


def test_load_module_audits_all_checkpoint_backed_tensors(tmp_path):
    module = nn.Module()
    module.register_parameter("weight", nn.Parameter(torch.empty(2, 2, device="meta")))
    module.register_buffer("scale", torch.empty(2, device="meta"))
    weight = torch.arange(4, dtype=torch.float32).reshape(2, 2)
    scale = torch.tensor([0.5, 1.5], dtype=torch.float32)
    weight_file = tmp_path / "model.safetensors"
    save_file(
        {
            "model.layers.0.weight": weight,
            "model.layers.0.scale": scale,
        },
        weight_file,
    )
    big_model = BigHFModelExportHelper.__new__(BigHFModelExportHelper)
    big_model._weight_mapping = WeightMapping()
    big_model._weight_mapping.weight_map = {
        "model.layers.0.weight": str(weight_file),
        "model.layers.0.scale": str(weight_file),
    }

    audit = big_model._load_module_from_safetensor(module, "model.layers.0")

    torch.testing.assert_close(module.weight, weight)
    torch.testing.assert_close(module.scale, scale)
    assert audit == {
        "parameter_keys": 1,
        "buffer_keys": 1,
        "initialized_non_persistent_buffers": 0,
        "missing_parameter_keys": 0,
        "missing_persistent_buffer_keys": 0,
    }


def test_load_weight_mapping_restores_tied_output_weight(tmp_path, monkeypatch):
    weight = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    weight_file = tmp_path / "model.safetensors"
    save_file({"input_embedding.weight": weight}, weight_file)
    model = TinyTiedModel()
    big_model = BigHFModelExportHelper.__new__(BigHFModelExportHelper)
    big_model._hf_model_dir = str(tmp_path)
    big_model._placeholder_type_names = ()
    big_model._placeholder_types = set()
    big_model._weight_mapping = WeightMapping()
    monkeypatch.setattr(
        big_model,
        "_read_weight_map",
        lambda: {"input_embedding.weight": str(weight_file)},
    )

    big_model._load_weight_mapping(model)

    assert model.output_embedding.weight is model.input_embedding.weight
    torch.testing.assert_close(model.output_embedding.weight, weight)


def test_load_weight_mapping_prepares_gptq_after_loading_non_placeholder_tensors(monkeypatch):
    events = []
    model = nn.Module()
    model.config = SimpleNamespace(quantization_config=SimpleNamespace(quant_method="gptq"))
    model.placeholder = NamedPlaceholder()

    big_model = BigHFModelExportHelper.__new__(BigHFModelExportHelper)
    big_model._hf_model_dir = "/tmp/gptq-model"
    big_model._placeholder_type_names = ("NamedPlaceholder",)
    big_model._placeholder_types = {NamedPlaceholder}
    big_model._weight_mapping = WeightMapping()
    monkeypatch.setattr(big_model, "_read_weight_map", lambda: {})
    monkeypatch.setattr(big_model, "_materialize_non_placeholder_modules", lambda *args: 0)
    monkeypatch.setattr(big_model, "_remove_output_embedding_shared_tensors", lambda *args: None)
    monkeypatch.setattr(big_model, "_remove_buffers_missing_from_weight_map", lambda *args: None)
    monkeypatch.setattr(
        big_model,
        "_load_non_placeholder_modules_from_safetensor",
        lambda *args: events.append("load"),
    )

    def prepare(module, quantization_config, model_config=None, skip_module_prefixes=None):
        events.append(("prepare", skip_module_prefixes))
        return 0

    monkeypatch.setattr(big_model, "_prepare_loaded_gptqmodel_modules", prepare)

    big_model._load_weight_mapping(model)

    assert events == ["load", ("prepare", ["placeholder"])]


def test_replace_hmonnx_placeholders_with_subgraphs_rewires_graph(tmp_path):
    tensor_type = onnx.TensorProto.FLOAT
    input_info = onnx.helper.make_tensor_value_info("main_input", tensor_type, [1])
    output_info = onnx.helper.make_tensor_value_info("main_output", tensor_type, [1])
    placeholder = onnx.helper.make_node(
        "PlaceHolder",
        ["main_input"],
        ["placeholder_tuple"],
        name="placeholder",
        domain="ai.houmo.xh2a",
        content="model.layers.0.mlp",
    )
    tuple_getitem = onnx.helper.make_node(
        "TupleGetItem",
        ["placeholder_tuple"],
        ["main_output"],
        name="placeholder_getitem",
        domain="ai.houmo.xh2a",
        index=0,
    )
    main_model = onnx.helper.make_model(
        onnx.helper.make_graph([placeholder, tuple_getitem], "main", [input_info], [output_info]),
        opset_imports=[onnx.helper.make_opsetid("", 18), onnx.helper.make_opsetid("ai.houmo.xh2a", 1)],
    )
    main_file = tmp_path / "main.onnx"
    onnx.save(main_model, main_file)

    placeholders_dir = tmp_path / "placeholders"
    placeholders_dir.mkdir()
    sub_input = onnx.helper.make_tensor_value_info("sub_input", tensor_type, [1])
    sub_output = onnx.helper.make_tensor_value_info("sub_output", tensor_type, [1])
    identity = onnx.helper.make_node("KVcache", ["sub_input"], ["sub_output"], name="node_llmcache", axis=2)
    sub_model = onnx.helper.make_model(
        onnx.helper.make_graph([identity], "subgraph", [sub_input], [sub_output]),
        opset_imports=[onnx.helper.make_opsetid("", 18), onnx.helper.make_opsetid("ai.houmo.xh2a", 1)],
    )
    onnx.save(sub_model, placeholders_dir / f"{_module_name_to_filename('model.layers.0.mlp')}.onnx")

    replace_placeholder_node(
        main_model.graph.node[0],
        main_model,
        placeholders_dir / f"{_module_name_to_filename('model.layers.0.mlp')}.onnx",
        "main.data",
        tmp_path / "main.data",
    )
    onnx.save(main_model, main_file)

    merged_model = onnx.load(main_file)
    assert [(node.op_type, list(node.input), list(node.output)) for node in merged_model.graph.node] == [
        ("KVcache", ["main_input"], ["main_output"])
    ]
    attrs = {attr.name: onnx.helper.get_attribute_value(attr) for attr in merged_model.graph.node[0].attribute}
    assert "only_handle_old_cache" not in attrs


def test_regular_and_placeholder_exports_finalize_to_identical_artifacts(tmp_path):
    """Names, debug metadata, quant metadata and external layout all converge."""

    def external_tensor(name, location, offset, payload):
        tensor = onnx.TensorProto(name=name, data_type=onnx.TensorProto.FLOAT, dims=[1])
        tensor.data_location = onnx.TensorProto.EXTERNAL
        for key, value in (
            ("location", location),
            ("offset", str(offset)),
            ("length", str(len(payload))),
        ):
            entry = tensor.external_data.add()
            entry.key = key
            entry.value = value
        return tensor

    def attach_quant_metadata(model, quant_model):
        encoded = base64.b64encode(
            quant_model.SerializeToString(deterministic=True)
        ).decode("ascii")
        graph_meta = model.metadata_props.add()
        graph_meta.key = "graph_meta"
        graph_meta.value = json.dumps(
            {"quant_meta": {"quanted_info_onnx": encoded}},
            indent=2,
            sort_keys=True,
        ) + "\n"
        version = model.metadata_props.add()
        version.key = "hmquant_version"
        version.value = "test"

    def add_debug_metadata(proto, value):
        metadata = proto.metadata_props.add()
        metadata.key = "pkg.torch.onnx.original_node_name"
        metadata.value = value

    tensor_type = onnx.TensorProto.FLOAT
    weight_0 = b"\x00\x00\x80?"
    weight_1 = b"\x00\x00\x00@"
    full_dir = tmp_path / "full"
    low_dir = tmp_path / "low"
    full_dir.mkdir()
    low_dir.mkdir()
    external_location = "model_external_data"

    full_input = onnx.helper.make_tensor_value_info("main_input", tensor_type, [1])
    full_output = onnx.helper.make_tensor_value_info("main_output", tensor_type, [1])
    module_target = "model.layers.0.mlp"
    full_weight_0 = external_tensor(f"{module_target}.weight_0", external_location, 0, weight_0)
    full_weight_1 = external_tensor(f"{module_target}.weight_1", external_location, 4, weight_1)
    full_nodes = [
        onnx.helper.make_node(
            "Add",
            ["main_input", f"{module_target}.weight_0"],
            [f"{module_target}.hidden"],
            name="node_add",
        ),
        onnx.helper.make_node(
            "Add",
            [f"{module_target}.hidden", f"{module_target}.weight_1"],
            ["module_output"],
            name="node_add_1",
        ),
        onnx.helper.make_node(
            "Identity",
            ["module_output"],
            ["main_output"],
            name="node_identity",
        ),
    ]
    add_debug_metadata(full_nodes[0], "full_fx_node")
    full_value_info = onnx.helper.make_tensor_value_info(f"{module_target}.hidden", tensor_type, [1])
    add_debug_metadata(full_value_info, "full_hidden")
    full_model = onnx.helper.make_model(
        onnx.helper.make_graph(
            full_nodes,
            "model",
            [full_input],
            [full_output],
            [full_weight_0, full_weight_1],
            value_info=[full_value_info],
        ),
        opset_imports=[
            onnx.helper.make_opsetid("", 18),
            onnx.helper.make_opsetid("ai.houmo.xh2a", 1),
        ],
    )
    full_quant_nodes = [
        onnx.helper.make_node(
            "XH2aQuantQAdd",
            ["main_input", "full_export_capture"],
            ["model_layers_0_mlp_add_0"],
            name="model_layers_0_mlp_add_0",
        ),
        onnx.helper.make_node(
            "XH2aQuantQAdd",
            ["model_layers_0_mlp_add_0"],
            ["quant_module_output"],
            name="model_layers_0_mlp_add_1",
        ),
        onnx.helper.make_node(
            "XH2aQuantQIdentity",
            ["quant_module_output"],
            ["main_output"],
            name="quant_identity",
        ),
        # The regular and low-memory exporters are allowed to schedule this
        # independent node on opposite sides of the module subgraph.
        onnx.helper.make_node(
            "XH2aQuantQInputStub",
            ["main_input"],
            ["unused_input_stub"],
            name="late_input_stub",
        ),
    ]
    full_quant_model = onnx.helper.make_model(
        onnx.helper.make_graph(
            full_quant_nodes,
            "QuantedInfo",
            [onnx.helper.make_tensor_value_info("main_input", tensor_type, [1])],
            [onnx.helper.make_tensor_value_info("main_output", tensor_type, [1])],
        )
    )
    attach_quant_metadata(full_model, full_quant_model)
    full_file = full_dir / "model.onnx"
    onnx.save(full_model, full_file)
    (full_dir / external_location).write_bytes(weight_0 + weight_1)

    placeholder = onnx.helper.make_node(
        "PlaceHolder",
        ["main_input"],
        ["placeholder_tuple"],
        name="runtime_placeholder",
        domain="ai.houmo.xh2a",
        content=module_target,
    )
    tuple_getitem = onnx.helper.make_node(
        "TupleGetItem",
        ["placeholder_tuple"],
        ["module_output"],
        name="runtime_getitem",
        domain="ai.houmo.xh2a",
        index=0,
    )
    runtime_identity = onnx.helper.make_node(
        "Identity",
        ["module_output"],
        ["main_output"],
        name="node_identity",
    )
    low_model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [placeholder, tuple_getitem, runtime_identity],
            "model",
            [onnx.helper.make_tensor_value_info("main_input", tensor_type, [1])],
            [onnx.helper.make_tensor_value_info("main_output", tensor_type, [1])],
        ),
        opset_imports=[
            onnx.helper.make_opsetid("", 18),
            onnx.helper.make_opsetid("ai.houmo.xh2a", 1),
        ],
    )
    low_quant_input_stub = onnx.helper.make_node(
        "XH2aQuantQInputStub",
        ["main_input"],
        ["unused_input_stub"],
        name="early_input_stub",
    )
    low_quant_placeholder = onnx.helper.make_node(
        "XH2aQuantQPlaceHolderModule",
        ["main_input"],
        ["quant_module_output"],
        name="model_layers_0_mlp",
    )
    low_quant_identity = onnx.helper.make_node(
        "XH2aQuantQIdentity",
        ["quant_module_output"],
        ["main_output"],
        name="quant_identity",
    )
    low_quant_model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [low_quant_input_stub, low_quant_placeholder, low_quant_identity],
            "QuantedInfo",
            [onnx.helper.make_tensor_value_info("main_input", tensor_type, [1])],
            [onnx.helper.make_tensor_value_info("main_output", tensor_type, [1])],
        )
    )
    attach_quant_metadata(low_model, low_quant_model)

    sub_input = onnx.helper.make_tensor_value_info("arg0", tensor_type, [1])
    sub_output = onnx.helper.make_tensor_value_info("sub_output", tensor_type, [1])
    sub_weight_1 = external_tensor("module.weight_1", "sub.data", 0, weight_1)
    sub_weight_0 = external_tensor("module.weight_0", "sub.data", 4, weight_0)
    sub_nodes = [
        onnx.helper.make_node(
            "Add",
            ["arg0", "module.weight_0"],
            ["local_hidden"],
            name="node_add",
        ),
        onnx.helper.make_node(
            "Add",
            ["local_hidden", "module.weight_1"],
            ["sub_output"],
            name="node_add_1",
        ),
    ]
    add_debug_metadata(sub_nodes[0], "standalone_fx_node")
    sub_value_info = onnx.helper.make_tensor_value_info("local_hidden", tensor_type, [1])
    add_debug_metadata(sub_value_info, "standalone_hidden")
    sub_model = onnx.helper.make_model(
        onnx.helper.make_graph(
            sub_nodes,
            "subgraph",
            [sub_input],
            [sub_output],
            [sub_weight_1, sub_weight_0],
            value_info=[sub_value_info],
        ),
        opset_imports=[onnx.helper.make_opsetid("", 18)],
    )
    sub_quant_nodes = [
        onnx.helper.make_node(
            "XH2aQuantQInputStub",
            ["arg0"],
            ["arg0_quant_stub"],
            name="arg0_quant_stub",
        ),
        onnx.helper.make_node(
            "XH2aQuantQAdd",
            ["arg0_quant_stub", "module_capture"],
            ["module_add_0"],
            name="module_add_0",
        ),
        onnx.helper.make_node(
            "XH2aQuantQAdd",
            ["module_add_0"],
            ["module_add_1"],
            name="module_add_1",
        ),
        onnx.helper.make_node(
            "XH2aQuantQOutput",
            ["module_add_1"],
            ["module_output_stub"],
            name="module_output_stub",
        ),
    ]
    sub_quant_model = onnx.helper.make_model(
        onnx.helper.make_graph(
            sub_quant_nodes,
            "QuantedInfo",
            [onnx.helper.make_tensor_value_info("arg0", tensor_type, [1])],
            [onnx.helper.make_tensor_value_info("module_output_stub", tensor_type, [1])],
        )
    )
    attach_quant_metadata(sub_model, sub_quant_model)
    sub_file = low_dir / f"{_module_name_to_filename(module_target)}.onnx"
    onnx.save(sub_model, sub_file)
    (low_dir / "sub.data").write_bytes(weight_1 + weight_0)

    replace_placeholder_node(
        low_model.graph.node[0],
        low_model,
        sub_file,
        external_location,
        low_dir / external_location,
    )
    low_file = low_dir / "model.onnx"
    onnx.save(low_model, low_file)

    full_stats = canonicalize_hmonnx_artifact(full_file)
    low_stats = canonicalize_hmonnx_artifact(low_file)
    full_bytes = full_file.read_bytes()
    full_external_bytes = (full_dir / external_location).read_bytes()
    repeated_stats = canonicalize_hmonnx_artifact(full_file)

    assert full_bytes == low_file.read_bytes()
    assert full_external_bytes == (
        low_dir / external_location
    ).read_bytes()
    assert full_stats["model_sha256"] == low_stats["model_sha256"]
    assert repeated_stats["already_canonical"] is True
    assert full_file.read_bytes() == full_bytes
    assert (full_dir / external_location).read_bytes() == full_external_bytes

    # Simulate a deterministic-export version migration. Re-running the full
    # canonicalizer (rather than taking its version fast path) must still be
    # byte-idempotent, including topology-scoped node names and quant metadata.
    migration_model = onnx.load(full_file, load_external_data=False)
    retained_metadata = [
        (item.key, item.value)
        for item in migration_model.metadata_props
        if item.key != "hmonnx.deterministic_export.version"
    ]
    del migration_model.metadata_props[:]
    for key, value in retained_metadata:
        item = migration_model.metadata_props.add()
        item.key = key
        item.value = value
    onnx.save(migration_model, full_file)
    migrated_stats = canonicalize_hmonnx_artifact(full_file)
    assert migrated_stats.get("already_canonical", False) is False
    assert full_file.read_bytes() == full_bytes
    assert (full_dir / external_location).read_bytes() == full_external_bytes

    finalized = onnx.load(low_file, load_external_data=False)
    assert [node.name for node in finalized.graph.node] == [
        "node_000000_add",
        "node_000001_add",
        "node_000002_identity",
    ]
    assert all(not node.metadata_props for node in finalized.graph.node)
    assert all(not value.metadata_props for value in finalized.graph.value_info)
    graph_meta = json.loads(
        {item.key: item.value for item in finalized.metadata_props}["graph_meta"]
    )
    quant_model = onnx.ModelProto()
    quant_model.ParseFromString(
        base64.b64decode(graph_meta["quant_meta"]["quanted_info_onnx"])
    )
    assert sorted(node.op_type for node in quant_model.graph.node) == sorted(
        [
            "XH2aQuantQAdd",
            "XH2aQuantQAdd",
            "XH2aQuantQIdentity",
            "XH2aQuantQInputStub",
        ]
    )
    produced = {
        output
        for node in quant_model.graph.node
        for output in node.output
        if output
    }
    graph_inputs = {value.name for value in quant_model.graph.input}
    captures = {
        input_name
        for node in quant_model.graph.node
        for input_name in node.input
        if input_name and input_name not in produced and input_name not in graph_inputs
    }
    assert captures == {"capture_0"}


def test_canonical_node_names_ignore_exporter_local_paths_and_counters():
    def external_weight(name):
        tensor = onnx.TensorProto(name=name, data_type=onnx.TensorProto.FLOAT, dims=[1])
        tensor.data_location = onnx.TensorProto.EXTERNAL
        for key, value in (("location", "weights.bin"), ("offset", "0"), ("length", "4")):
            entry = tensor.external_data.add()
            entry.key = key
            entry.value = value
        return tensor

    def graph(mode):
        input_info = onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1])
        output_info = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1])
        weight = external_weight("model.layers.7.mlp.proj.weight")
        nodes = []
        source = "input"
        nodes.extend(
            [
                onnx.helper.make_node(
                    "Linear",
                    [source, weight.name],
                    ["projected"],
                    name=(
                        "model_layers_7_mlp_deadbeef00/node_linear"
                        if mode == "low_memory"
                        else "node_linear_401"
                    ),
                ),
                onnx.helper.make_node(
                    "Sigmoid",
                    ["projected"],
                    ["gated"],
                    name=(
                        "model_layers_7_mlp_deadbeef00/node_sigmoid"
                        if mode == "low_memory"
                        else "node_sigmoid_77"
                    ),
                ),
                onnx.helper.make_node(
                    "Identity",
                    ["gated"],
                    ["output"],
                    name=(
                        "model_layers_7_mlp_deadbeef00/node_identity"
                        if mode == "low_memory"
                        else "node_identity_44"
                    ),
                ),
            ]
        )
        return onnx.helper.make_graph(nodes, mode, [input_info], [output_info], [weight])

    regular = graph("regular")
    low_memory = graph("low_memory")
    regular_names = _canonical_node_names(
        regular,
        {item.name: item for item in regular.initializer},
    )
    low_memory_names = _canonical_node_names(
        low_memory,
        {item.name: item for item in low_memory.initializer},
    )

    assert regular_names == low_memory_names
    assert regular_names == [
        "node_000000_linear",
        "node_000001_sigmoid",
        "node_000002_identity",
    ]


def test_replace_placeholder_allows_unconsumed_subgraph_outputs(tmp_path):
    tensor_type = onnx.TensorProto.FLOAT
    input_info = onnx.helper.make_tensor_value_info("main_input", tensor_type, [1])
    output_info = onnx.helper.make_tensor_value_info("main_output", tensor_type, [1])
    placeholder = onnx.helper.make_node(
        "PlaceHolder",
        ["main_input"],
        ["placeholder_tuple"],
        name="placeholder",
        domain="ai.houmo.xh2a",
    )
    tuple_getitem = onnx.helper.make_node(
        "TupleGetItem",
        ["placeholder_tuple"],
        ["main_output"],
        name="placeholder_getitem",
        domain="ai.houmo.xh2a",
        index=0,
    )
    main_model = onnx.helper.make_model(
        onnx.helper.make_graph([placeholder, tuple_getitem], "main", [input_info], [output_info])
    )

    sub_input = onnx.helper.make_tensor_value_info("sub_input", tensor_type, [1])
    sub_output_0 = onnx.helper.make_tensor_value_info("sub_output_0", tensor_type, [1])
    sub_output_1 = onnx.helper.make_tensor_value_info("sub_output_1", tensor_type, [1])
    identity_0 = onnx.helper.make_node("Identity", ["sub_input"], ["sub_output_0"], name="identity_0")
    identity_1 = onnx.helper.make_node("Identity", ["sub_input"], ["sub_output_1"], name="identity_1")
    subgraph_file = tmp_path / "subgraph.onnx"
    onnx.save(
        onnx.helper.make_model(
            onnx.helper.make_graph(
                [identity_0, identity_1],
                "subgraph",
                [sub_input],
                [sub_output_0, sub_output_1],
            )
        ),
        subgraph_file,
    )

    replace_placeholder_node(
        main_model.graph.node[0],
        main_model,
        subgraph_file,
        "main.data",
        tmp_path / "main.data",
    )

    assert [(node.op_type, list(node.input), list(node.output)) for node in main_model.graph.node] == [
        ("Identity", ["main_input"], ["main_output"]),
        ("Identity", ["main_input"], ["subgraph/sub_output_1"]),
    ]
    onnx.checker.check_model(main_model)


def test_post_inline_canonicalization_restores_whole_graph_deduplication():
    input_info = onnx.helper.make_tensor_value_info("input", onnx.TensorProto.INT64, [1, 4])
    output_info = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT16, [2, 1, 4])
    axes_0 = onnx.helper.make_tensor("axes_0", onnx.TensorProto.INT64, [1], [0])
    axes_1 = onnx.helper.make_tensor("axes_1", onnx.TensorProto.INT64, [1], [0])
    unused = onnx.helper.make_tensor("unused", onnx.TensorProto.INT64, [], [1])
    nodes = [
        onnx.helper.make_node("Cast", ["input"], ["cast_0"], name="cast_0", to=onnx.TensorProto.FLOAT16),
        onnx.helper.make_node("Unsqueeze", ["cast_0", "axes_0"], ["unsqueeze_0"], name="unsqueeze_0"),
        onnx.helper.make_node("Cast", ["input"], ["cast_1"], name="cast_1", to=onnx.TensorProto.FLOAT16),
        onnx.helper.make_node("Unsqueeze", ["cast_1", "axes_1"], ["unsqueeze_1"], name="unsqueeze_1"),
        onnx.helper.make_node("Concat", ["unsqueeze_0", "unsqueeze_1"], ["output"], name="concat", axis=0),
    ]
    value_info = [
        onnx.helper.make_tensor_value_info("axes_1", onnx.TensorProto.INT64, [1]),
        onnx.helper.make_tensor_value_info("unused", onnx.TensorProto.INT64, []),
        onnx.helper.make_tensor_value_info("cast_1", onnx.TensorProto.FLOAT16, [1, 4]),
        onnx.helper.make_tensor_value_info("unsqueeze_1", onnx.TensorProto.FLOAT16, [1, 1, 4]),
    ]
    model = onnx.helper.make_model(
        onnx.helper.make_graph(
            nodes,
            "merged",
            [input_info],
            [output_info],
            [axes_0, axes_1, unused],
            value_info=value_info,
        ),
        opset_imports=[onnx.helper.make_opsetid("", 18)],
    )

    stats = _canonicalize_merged_hmonnx_model(model)

    assert stats == {
        "deduplicated_inline_initializers": 1,
        "deduplicated_pure_nodes": 2,
        "removed_unreferenced_inline_initializers": 0,
    }
    assert [initializer.name for initializer in model.graph.initializer] == ["axes_0", "unused"]
    assert [(node.op_type, list(node.input), list(node.output)) for node in model.graph.node] == [
        ("Cast", ["input"], ["cast_0"]),
        ("Unsqueeze", ["cast_0", "axes_0"], ["unsqueeze_0"]),
        ("Concat", ["unsqueeze_0", "unsqueeze_0"], ["output"]),
    ]
    assert {value.name for value in model.graph.value_info}.isdisjoint({"axes_1", "cast_1", "unsqueeze_1"})
    assert "unused" in {value.name for value in model.graph.value_info}
    onnx.checker.check_model(model)


def test_post_inline_canonicalization_never_deduplicates_external_weights_by_metadata():
    input_info = onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1])
    output_info = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1])
    weights = []
    for index in range(2):
        weight = onnx.TensorProto(name=f"weight_{index}", data_type=onnx.TensorProto.FLOAT, dims=[1])
        weight.data_location = onnx.TensorProto.EXTERNAL
        for key, value in {
            "location": "weights.bin",
            "offset": str(index * 4),
            "length": "4",
        }.items():
            entry = weight.external_data.add()
            entry.key = key
            entry.value = value
        weights.append(weight)
    nodes = [
        onnx.helper.make_node("Add", ["input", "weight_0"], ["sum_0"]),
        onnx.helper.make_node("Add", ["sum_0", "weight_1"], ["output"]),
    ]
    model = onnx.helper.make_model(
        onnx.helper.make_graph(nodes, "external", [input_info], [output_info], weights),
        opset_imports=[onnx.helper.make_opsetid("", 18)],
    )

    stats = _canonicalize_merged_hmonnx_model(model)

    assert stats == {
        "deduplicated_inline_initializers": 0,
        "deduplicated_pure_nodes": 0,
        "removed_unreferenced_inline_initializers": 0,
    }
    assert [initializer.name for initializer in model.graph.initializer] == ["weight_0", "weight_1"]


@pytest.mark.parametrize("cleanup_temporary_files", [False, True])
def test_replace_hmonnx_placeholders_optionally_cleans_temporary_files(tmp_path, cleanup_temporary_files):
    tensor_type = onnx.TensorProto.FLOAT
    input_info = onnx.helper.make_tensor_value_info("main_input", tensor_type, [1])
    valid_length_info = onnx.helper.make_tensor_value_info("valid_length", onnx.TensorProto.INT32, [1])
    current_length_info = onnx.helper.make_tensor_value_info("current_length", onnx.TensorProto.INT32, [1])
    old_cache_info = onnx.helper.make_tensor_value_info("old_cache", tensor_type, [1])
    output_info = onnx.helper.make_tensor_value_info("main_output", tensor_type, [1])
    kvcache = onnx.helper.make_node(
        "KVcache",
        ["main_input", "valid_length", "current_length", "old_cache"],
        ["cached_input"],
        name="main_kvcache",
        domain="ai.houmo.xh2a",
        attention_max_length=-1,
        axis=2,
    )
    placeholder = onnx.helper.make_node(
        "PlaceHolder",
        ["cached_input"],
        ["placeholder_tuple"],
        name="placeholder",
        domain="ai.houmo.xh2a",
        content="model.layers.0.mlp",
    )
    tuple_getitem = onnx.helper.make_node(
        "TupleGetItem",
        ["placeholder_tuple"],
        ["main_output"],
        name="placeholder_getitem",
        domain="ai.houmo.xh2a",
        index=0,
    )
    main_file = tmp_path / "main.onnx"
    onnx.save(
        onnx.helper.make_model(
            onnx.helper.make_graph(
                [kvcache, placeholder, tuple_getitem],
                "main",
                [input_info, valid_length_info, current_length_info, old_cache_info],
                [output_info],
            ),
            opset_imports=[onnx.helper.make_opsetid("", 18), onnx.helper.make_opsetid("ai.houmo.xh2a", 1)],
        ),
        main_file,
    )

    placeholders_dir = tmp_path / "placeholders"
    placeholders_dir.mkdir()
    sub_input = onnx.helper.make_tensor_value_info("sub_input", tensor_type, [1])
    sub_output = onnx.helper.make_tensor_value_info("sub_output", tensor_type, [1])
    identity = onnx.helper.make_node("Identity", ["sub_input"], ["sub_output"], name="identity")
    onnx.save(
        onnx.helper.make_model(
            onnx.helper.make_graph([identity], "subgraph", [sub_input], [sub_output]),
            opset_imports=[onnx.helper.make_opsetid("", 18)],
        ),
        placeholders_dir / f"{_module_name_to_filename('model.layers.0.mlp')}.onnx",
    )

    helper = BigHFModelExportHelper.__new__(BigHFModelExportHelper)
    helper._cleanup_temporary_files = cleanup_temporary_files
    helper.replace_hmonnx_placeholders_with_subgraphs(str(main_file))

    assert placeholders_dir.exists() is not cleanup_temporary_files
    assert main_file.with_suffix(".onnx.bak").exists() is not cleanup_temporary_files
    merged_model = onnx.load(main_file)
    assert [(node.op_type, list(node.input), list(node.output)) for node in merged_model.graph.node] == [
        (
            "KVcache",
            ["main_input", "valid_length", "current_length", "old_cache"],
            ["cached_input"],
        ),
        ("Identity", ["cached_input"], ["main_output"]),
    ]
    kvcache_attrs = {
        attr.name: onnx.helper.get_attribute_value(attr) for attr in merged_model.graph.node[0].attribute
    }
    assert "only_handle_old_cache" not in kvcache_attrs


def test_replace_hmonnx_placeholders_cleanup_ignores_missing_placeholder_dir_on_noop(tmp_path):
    tensor_type = onnx.TensorProto.FLOAT
    input_info = onnx.helper.make_tensor_value_info("main_input", tensor_type, [1])
    output_info = onnx.helper.make_tensor_value_info("main_output", tensor_type, [1])
    identity = onnx.helper.make_node("Identity", ["main_input"], ["main_output"], name="identity")
    main_file = tmp_path / "main.onnx"
    onnx.save(
        onnx.helper.make_model(
            onnx.helper.make_graph([identity], "main", [input_info], [output_info]),
            opset_imports=[onnx.helper.make_opsetid("", 18)],
        ),
        main_file,
    )

    helper = BigHFModelExportHelper.__new__(BigHFModelExportHelper)
    helper._cleanup_temporary_files = True
    helper.replace_hmonnx_placeholders_with_subgraphs(str(main_file))

    assert not (tmp_path / "placeholders").exists()


def test_merge_external_data_rejects_location_outside_subgraph_directory(tmp_path):
    subgraph_dir = tmp_path / "subgraphs"
    subgraph_dir.mkdir()
    outside_file = tmp_path / "outside.bin"
    outside_file.write_bytes(b"secret")
    initializer = onnx.helper.make_tensor("weight", onnx.TensorProto.FLOAT, [1], [0.0])
    initializer.ClearField("raw_data")
    initializer.data_location = onnx.TensorProto.EXTERNAL
    for key, value in {"location": "../outside.bin", "offset": "0", "length": "6"}.items():
        entry = initializer.external_data.add()
        entry.key = key
        entry.value = value
    model = onnx.helper.make_model(onnx.helper.make_graph([], "subgraph", [], [], [initializer]))

    with pytest.raises(RuntimeError, match="escapes model directory"):
        _merge_external_data(model, subgraph_dir / "subgraph.onnx", "main.data", tmp_path / "main.data")


def test_replace_placeholder_rejects_duplicate_tuple_getitem_indices(tmp_path):
    tensor_type = onnx.TensorProto.FLOAT
    input_info = onnx.helper.make_tensor_value_info("main_input", tensor_type, [1])
    output_0 = onnx.helper.make_tensor_value_info("main_output_0", tensor_type, [1])
    output_1 = onnx.helper.make_tensor_value_info("main_output_1", tensor_type, [1])
    placeholder = onnx.helper.make_node(
        "PlaceHolder",
        ["main_input"],
        ["placeholder_tuple"],
        name="placeholder",
        domain="ai.houmo.xh2a",
    )
    getitem_0 = onnx.helper.make_node(
        "TupleGetItem",
        ["placeholder_tuple"],
        ["main_output_0"],
        name="getitem_0",
        domain="ai.houmo.xh2a",
        index=0,
    )
    getitem_1 = onnx.helper.make_node(
        "TupleGetItem",
        ["placeholder_tuple"],
        ["main_output_1"],
        name="getitem_1",
        domain="ai.houmo.xh2a",
        index=0,
    )
    main_model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [placeholder, getitem_0, getitem_1],
            "main",
            [input_info],
            [output_0, output_1],
        )
    )

    sub_input = onnx.helper.make_tensor_value_info("sub_input", tensor_type, [1])
    sub_output_0 = onnx.helper.make_tensor_value_info("sub_output_0", tensor_type, [1])
    sub_output_1 = onnx.helper.make_tensor_value_info("sub_output_1", tensor_type, [1])
    identity_0 = onnx.helper.make_node("Identity", ["sub_input"], ["sub_output_0"], name="identity_0")
    identity_1 = onnx.helper.make_node("Identity", ["sub_input"], ["sub_output_1"], name="identity_1")
    subgraph_file = tmp_path / "subgraph.onnx"
    onnx.save(
        onnx.helper.make_model(
            onnx.helper.make_graph(
                [identity_0, identity_1],
                "subgraph",
                [sub_input],
                [sub_output_0, sub_output_1],
            )
        ),
        subgraph_file,
    )

    with pytest.raises(RuntimeError, match="must be unique valid"):
        replace_placeholder_node(
            placeholder,
            main_model,
            subgraph_file,
            "main.data",
            tmp_path / "main.data",
        )


def test_merge_opset_imports_adds_custom_domain_and_rejects_version_conflicts():
    main_model = onnx.helper.make_model(
        onnx.helper.make_graph([], "main", [], []),
        opset_imports=[onnx.helper.make_opsetid("", 18)],
    )
    subgraph_model = onnx.helper.make_model(
        onnx.helper.make_graph([], "subgraph", [], []),
        opset_imports=[
            onnx.helper.make_opsetid("", 18),
            onnx.helper.make_opsetid("ai.houmo.custom", 3),
        ],
    )

    _merge_opset_imports(main_model, subgraph_model)

    assert {opset.domain: opset.version for opset in main_model.opset_import}["ai.houmo.custom"] == 3

    incompatible_subgraph = onnx.helper.make_model(
        onnx.helper.make_graph([], "subgraph", [], []),
        opset_imports=[onnx.helper.make_opsetid("ai.houmo.custom", 4)],
    )
    with pytest.raises(RuntimeError, match="Incompatible ONNX opset versions"):
        _merge_opset_imports(main_model, incompatible_subgraph)
