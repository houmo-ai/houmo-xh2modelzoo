from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from xhmodel_merak.xh_llm.models.deepseek_v4 import _deepseek_v4_big_export as big_export
from xhmodel_merak.xh_llm.models.deepseek_v4._deepseek_v4_big_export import (
    DeepSeekV4BigHFModel,
)
from xhmodel_merak.xh_llm.models.deepseek_v4.deepseek_v4_model import (
    XHDeepSeekV4Model,
)
from xhmodel_merak.xh_llm.models.deepseek_v4.moe import (
    DeepSeekV4MoE,
    DeepSeekV4MoEPlaceHolder,
)


def test_streaming_moe_placeholder_keeps_two_input_one_output_contract() -> None:
    placeholder = DeepSeekV4MoEPlaceHolder()
    hidden = torch.randn(1, 256, 8)
    input_ids = torch.randint(0, 16, (1, 256))

    output = placeholder(hidden, input_ids)

    assert output is hidden
    assert list(placeholder.parameters()) == []
    assert list(placeholder.buffers()) == []


def test_static_moe_target_maps_to_checkpoint_layer() -> None:
    assert DeepSeekV4BigHFModel.checkpoint_moe_target("blocks.layers.0.mlp") == "model.layers.0.mlp"
    assert DeepSeekV4BigHFModel.checkpoint_moe_target("blocks.layers.42.mlp") == "model.layers.42.mlp"


def test_marks_all_hf_sparse_moe_blocks_without_touching_other_modules() -> None:
    sparse_type = type("DeepseekV4SparseMoeBlock", (nn.Module,), {})
    model = nn.Module()
    model.layers = nn.ModuleList([sparse_type(), sparse_type(), sparse_type()])
    model.norm = nn.LayerNorm(4)

    count = DeepSeekV4BigHFModel.mark_streaming_placeholders(model)

    assert count == 3
    assert all(getattr(layer, "_xh_v4_streaming_placeholder", False) for layer in model.layers)
    assert not hasattr(model.norm, "_xh_v4_streaming_placeholder")


def test_marks_only_requested_prefixes_for_truncated_export() -> None:
    sparse_type = type("DeepseekV4SparseMoeBlock", (nn.Module,), {})
    model = nn.Module()
    model.model = nn.Module()
    model.model.layers = nn.ModuleList([nn.Module() for _ in range(4)])
    for layer in model.model.layers:
        layer.mlp = sparse_type()

    count = DeepSeekV4BigHFModel.mark_streaming_placeholders(
        model,
        ["model.layers.0.mlp", "model.layers.1.mlp"],
    )

    assert count == 2
    assert all(getattr(model.model.layers[index].mlp, "_xh_v4_streaming_placeholder", False) for index in (0, 1))
    assert all(not hasattr(model.model.layers[index].mlp, "_xh_v4_streaming_placeholder") for index in (2, 3))


def test_first_n_export_drops_unused_empty_layer_shells_without_changing_config() -> None:
    model = nn.Module()
    model.model = nn.Module()
    model.model.layers = nn.ModuleList([nn.Linear(1, 1) for _ in range(43)])
    model.config = SimpleNamespace(num_hidden_layers=43)
    retained = tuple(model.model.layers[:6])

    XHDeepSeekV4Model._retain_requested_layer_prefix(model, 6)

    assert tuple(model.model.layers) == retained
    assert len(model.model.layers) == 6
    assert model.config.num_hidden_layers == 43
    assert not any(name.startswith("model.layers.6") for name, _ in model.named_modules())


def test_full_export_keeps_complete_empty_layer_module_list() -> None:
    model = nn.Module()
    model.model = nn.Module()
    model.model.layers = nn.ModuleList([nn.Linear(1, 1) for _ in range(43)])
    original_layers = model.model.layers

    XHDeepSeekV4Model._retain_requested_layer_prefix(model, 43)

    assert model.model.layers is original_layers
    assert len(model.model.layers) == 43


def test_autoround_empty_model_uses_gptqmodel_meta_preprocessor(monkeypatch) -> None:
    calls = {}

    class _Preprocessor:
        def __init__(self, model_dir, *, dtype):
            calls["model_dir"] = model_dir
            calls["dtype"] = dtype

        def preprocess(
            self,
            model,
            include_module_prefixes=None,
            skip_module_prefixes=None,
        ):
            calls["model"] = model
            calls["include"] = include_module_prefixes
            calls["skip"] = skip_module_prefixes
            return ("linear0", "linear1")

    model = nn.Module()
    model.register_parameter(
        "weight",
        nn.Parameter(torch.empty(1, dtype=torch.bfloat16, device="meta")),
    )
    model.config = SimpleNamespace(
        quantization_config={"quant_method": "auto_round"},
        _name_or_path="/ignored",
    )
    monkeypatch.setattr(
        big_export,
        "GPTQModelQuantizedModelPreprocessor",
        _Preprocessor,
    )
    monkeypatch.setattr(big_export.XHBaseModel, "_trim_cpu_allocator", lambda: None)

    DeepSeekV4BigHFModel._preprocess_quantized_hf_model(
        model,
        "/packed",
        skip_module_prefixes=["model.layers.0.mlp"],
    )

    assert calls == {
        "model_dir": "/packed",
        "dtype": torch.bfloat16,
        "model": model,
        "include": None,
        "skip": ["model.layers.0.mlp"],
    }


def test_streamed_loader_maps_target_and_builds_static_moe(monkeypatch) -> None:
    template = nn.Module()
    template.model = nn.Module()
    template.model.layers = nn.ModuleList([nn.Module() for _ in range(3)])
    template.model.layers[2].mlp = nn.Module()
    template.config = SimpleNamespace(quantization_config={"quant_method": "gptq"})

    helper = DeepSeekV4BigHFModel.__new__(DeepSeekV4BigHFModel)
    calls = {}

    def fake_load(module, prefix):
        calls["loaded"] = (module, prefix)
        return {"parameter_keys": 7, "buffer_keys": 1}

    def fake_prepare(module, quantization_config, model_config=None, **kwargs):
        calls["prepared"] = (module, quantization_config, model_config)
        return 0

    monkeypatch.setattr(helper, "_load_module_from_safetensor", fake_load)
    monkeypatch.setattr(helper, "_prepare_loaded_gptqmodel_modules", fake_prepare)
    monkeypatch.setattr(helper, "_convert_loaded_packed_linears", lambda module: 771)

    static_moe = nn.Module()

    def fake_from_hf(cls, source, config, fast_mode=True):
        calls["from_hf"] = (source, config, fast_mode)
        return static_moe

    monkeypatch.setattr(DeepSeekV4MoE, "from_hf", classmethod(fake_from_hf))

    result = helper._load_place_holder_module_once(
        template,
        "blocks.layers.2.mlp",
    )

    assert result is static_moe
    assert calls["loaded"][1] == "model.layers.2.mlp"
    assert calls["prepared"][1] == template.config.quantization_config
    assert calls["from_hf"][1] is template.config
    assert not hasattr(result, "_placeholder_load_audit")
