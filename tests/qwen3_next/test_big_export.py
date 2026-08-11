import json
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from safetensors import safe_open
from safetensors.torch import save_file

from xhmodel_merak.xh_llm.models.qwen3_next import XHQwen3NextModel
from xhquant.utils.registry import _DMRegistryCls


def test_reduced_checkpoint_copies_mtp_shard_when_hard_link_is_unavailable(monkeypatch, tmp_path):
    from examples_merak.llm.qwen3_next.prepare_reduced_checkpoint import _link_or_copy

    source = tmp_path / "source.safetensors"
    destination = tmp_path / "output" / "mtp.safetensors"
    source.write_bytes(b"mtp-weights")
    destination.parent.mkdir()

    def reject_hard_link(*args, **kwargs):
        raise PermissionError("cross-device hard link")

    monkeypatch.setattr("os.link", reject_hard_link)

    _link_or_copy(source, destination)

    assert destination.read_bytes() == source.read_bytes()


def test_reduced_checkpoint_keeps_multiple_mtp_shards_without_overwriting(tmp_path):
    from examples_merak.llm.qwen3_next.prepare_reduced_checkpoint import reduce_checkpoint

    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    save_file({"mtp.layers.0.weight": torch.zeros(1)}, source / "model-00001.safetensors")
    save_file({"mtp.layers.1.weight": torch.ones(1)}, source / "model-00002.safetensors")
    (source / "config.json").write_text(json.dumps({"num_hidden_layers": 48}))
    (source / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "mtp.layers.0.weight": "model-00001.safetensors",
                    "mtp.layers.1.weight": "model-00002.safetensors",
                }
            }
        )
    )

    reduce_checkpoint(source, output, max_layers=4, include_mtp=True)

    target_index = json.loads((output / "model.safetensors.index.json").read_text())
    target_map = target_index["weight_map"]
    for tensor_name, shard_name in target_map.items():
        with safe_open(output / shard_name, framework="pt", device="cpu") as reader:
            assert tensor_name in reader.keys()
    assert len(set(target_map.values())) == 2
    assert target_index["metadata"]["total_size"] == 8


def test_reduced_checkpoint_drops_unselected_tensors_from_mixed_mtp_shard(tmp_path):
    from examples_merak.llm.qwen3_next.prepare_reduced_checkpoint import reduce_checkpoint

    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    save_file(
        {
            "mtp.layers.0.weight": torch.zeros(1),
            "model.layers.9.mlp.weight": torch.ones(2),
        },
        source / "mixed.safetensors",
    )
    (source / "config.json").write_text(json.dumps({"num_hidden_layers": 48}))
    (source / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "mtp.layers.0.weight": "mixed.safetensors",
                    "model.layers.9.mlp.weight": "mixed.safetensors",
                }
            }
        )
    )

    reduce_checkpoint(source, output, max_layers=4, include_mtp=True)

    target_index = json.loads((output / "model.safetensors.index.json").read_text())
    target_map = target_index["weight_map"]
    shard_name = target_map["mtp.layers.0.weight"]
    with safe_open(output / shard_name, framework="pt", device="cpu") as reader:
        assert list(reader.keys()) == ["mtp.layers.0.weight"]
    assert target_index["metadata"]["total_size"] == 4


@pytest.mark.parametrize(
    ("config", "expected_layer_types"),
    [
        ({"num_hidden_layers": 48, "full_attention_interval": 4}, None),
        (
            {
                "num_hidden_layers": 48,
                "layer_types": ["linear_attention", "full_attention"] * 24,
            },
            ["linear_attention", "full_attention"] * 2,
        ),
    ],
)
def test_reduced_checkpoint_preserves_implicit_layer_type_generation(config, expected_layer_types):
    from examples_merak.llm.qwen3_next.prepare_reduced_checkpoint import _reduce_config_layers

    _reduce_config_layers(config, max_layers=4)

    assert config["num_hidden_layers"] == 4
    if expected_layer_types is None:
        assert "layer_types" not in config
    else:
        assert config["layer_types"] == expected_layer_types


def test_qwen3_next_export_uses_big_model_path_when_enabled(monkeypatch, tmp_path):
    expected_meta = object()
    exported_info = SimpleNamespace(exported_dir=str(tmp_path), meta=SimpleNamespace())
    model = XHQwen3NextModel.__new__(XHQwen3NextModel)
    model.get_export_info = lambda output_dir: exported_info
    model._export_big_language_hmonnx = lambda info: expected_meta
    model._export_hmonnx = lambda info: (_ for _ in ()).throw(
        AssertionError("ordinary export path must not run")
    )

    monkeypatch.setenv("HUGE_MODEL_EXPORT_ENABLED", "1")

    assert model.export_hmonnx(str(tmp_path)) is expected_meta


def test_qwen3_next_export_uses_qwen3_next_language_finalizer_by_default(monkeypatch, tmp_path):
    expected_meta = object()
    exported_info = SimpleNamespace(exported_dir=str(tmp_path), meta=SimpleNamespace())
    model = XHQwen3NextModel.__new__(XHQwen3NextModel)
    model.get_export_info = lambda output_dir: exported_info
    model._export_language_hmonnx_impl = lambda info: expected_meta
    model._export_hmonnx = lambda info: (_ for _ in ()).throw(
        AssertionError("public export must delegate to the Qwen3-Next finalizer")
    )

    monkeypatch.delenv("HUGE_MODEL_EXPORT_ENABLED", raising=False)

    assert model.export_hmonnx(str(tmp_path)) is expected_meta


def test_qwen3_next_big_export_uses_qwen3_next_placeholder_components():
    from xhmodel_merak.xh_llm.models.qwen3_next._qwen3_next_big_export import (
        Qwen3NextBigHFModel,
    )

    model = XHQwen3NextModel.__new__(XHQwen3NextModel)

    big_hf_model_cls, placeholder_types = model._get_big_language_placeholder_export_components()

    assert big_hf_model_cls is Qwen3NextBigHFModel
    assert placeholder_types == [
        "Qwen3NextSparseMoeBlock",
        "LinearQwen3NextSparseMoeBlock",
    ]


def test_qwen3_next_big_export_registers_defused_moe_runtime_type():
    from xhmodel_merak.xh_llm.models.qwen3_next._qwen3_next_big_export import (
        Qwen3NextBigHFModel,
    )

    runtime_sparse_moe_type = type("LinearQwen3NextSparseMoeBlock", (nn.Module,), {})
    hf_model = nn.Sequential(runtime_sparse_moe_type())
    registry = _DMRegistryCls("Qwen3NextBigExportRuntimeAliasTest")

    Qwen3NextBigHFModel.register_placeholder(registry, hf_model=hf_model)

    assert runtime_sparse_moe_type in registry


def test_qwen3_next_big_export_registers_defused_moe_for_standalone_wrap():
    from defuser.modeling.unfused_moe.qwen3_next import LinearQwen3NextSparseMoeBlock

    from xhmodel_merak.xh_llm.models.qwen3_next._model import _Qwen3NextSparseMoeBlock
    from xhmodel_merak.xh_llm.models.qwen3_next._qwen3_next_big_export import (
        register_runtime_wrap_modules,
    )

    registry = _DMRegistryCls("Qwen3NextBigExportStandaloneWrapTest")

    register_runtime_wrap_modules(registry)

    assert LinearQwen3NextSparseMoeBlock in registry
    registered_type = registry.get(LinearQwen3NextSparseMoeBlock)
    assert issubclass(registered_type, _Qwen3NextSparseMoeBlock)
    assert issubclass(registered_type, LinearQwen3NextSparseMoeBlock)


def test_qwen3_next_big_export_replaces_defused_main_graph_module():
    from xhmodel_merak.xh_llm.models.qwen3_next._qwen3_next_big_export import (
        Qwen3NextBigHFModel,
        _Qwen3NextSparseMoeBlock_PlaceHolder,
    )

    runtime_sparse_moe_type = type("LinearQwen3NextSparseMoeBlock", (nn.Module,), {})
    hf_model = nn.Sequential(runtime_sparse_moe_type())
    hf_model[0].weight = nn.Parameter(torch.ones(2, 2))
    big_model = Qwen3NextBigHFModel.__new__(Qwen3NextBigHFModel)

    replaced = big_model.replace_runtime_placeholder_modules(hf_model)

    assert replaced == 1
    assert isinstance(hf_model[0], _Qwen3NextSparseMoeBlock_PlaceHolder)
    assert list(hf_model[0].parameters()) == []


def test_qwen3_next_big_export_preserves_fused_gptq_expert_layout(monkeypatch, tmp_path):
    from xhmodel_merak.xh_llm import big_hf_model_helper
    from xhmodel_merak.xh_llm.models.qwen3_next._qwen3_next_big_export import (
        Qwen3NextBigHFModel,
    )

    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "model.layers.0.mlp.experts.gate_up_proj": "model.safetensors",
                    "model.layers.0.mlp.experts.down_proj": "model.safetensors",
                }
            }
        )
    )
    calls = []

    class FakeGPTQModelPreprocessor:
        def __init__(self, model_dir):
            calls.append(("init", model_dir))

        def preprocess(self, model, skip_module_prefixes=None, convert_model_structure=True):
            calls.append(("preprocess", skip_module_prefixes, convert_model_structure))
            return ()

    model = nn.Module()
    model.config = SimpleNamespace(quantization_config={"quant_method": "gptq"})
    monkeypatch.setattr(
        big_hf_model_helper,
        "GPTQModelQuantizedModelPreprocessor",
        FakeGPTQModelPreprocessor,
    )
    monkeypatch.setattr(big_hf_model_helper.XHBaseModel, "_trim_cpu_allocator", lambda: None)

    Qwen3NextBigHFModel._preprocess_quantized_hf_model(model, tmp_path)

    assert calls == [
        ("init", tmp_path),
        ("preprocess", None, False),
    ]


def test_qwen3_next_big_export_support_check_accepts_required_modules():
    model = XHQwen3NextModel.__new__(XHQwen3NextModel)
    attention_type = type("Qwen3NextAttention", (nn.Module,), {})
    gated_delta_net_type = type("Qwen3NextGatedDeltaNet", (nn.Module,), {})
    sparse_moe_type = type("Qwen3NextSparseMoeBlock", (nn.Module,), {})
    empty_hf_model = SimpleNamespace(
        config=SimpleNamespace(model_type="qwen3_next"),
        model=nn.Sequential(
            attention_type(),
            gated_delta_net_type(),
            sparse_moe_type(),
        ),
    )

    model._check_big_language_placeholder_export_supported(empty_hf_model)


def test_qwen3_next_big_export_support_check_rejects_missing_modules():
    model = XHQwen3NextModel.__new__(XHQwen3NextModel)
    attention_type = type("Qwen3NextAttention", (nn.Module,), {})
    empty_hf_model = SimpleNamespace(
        config=SimpleNamespace(model_type="qwen3_next"),
        model=nn.Sequential(attention_type()),
    )

    with pytest.raises(NotImplementedError, match="Qwen3NextGatedDeltaNet"):
        model._check_big_language_placeholder_export_supported(empty_hf_model)