from __future__ import annotations

import json
import re
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from xhmodel_merak.xh_llm.models.ling_3_flash.quant_adapter import (
    Ling3FlashQuantSpec,
    _install_autoround_nonfinite_loss_guard,
    autoround_calibration_source,
    build_autoround_block_names,
    build_autoround_layer_config,
    canonicalize_autoround_gptqmodel_config,
    quantize_ling3_flash,
)
from xhmodel_merak.xh_llm.models.ling_3_flash.workflow import Ling3FlashWorkflow


class _Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(64, 64, bias=False)
        self.f_proj = nn.Linear(64, 64, bias=False)
        self.b_proj = nn.Linear(64, 4, bias=False)


class _Expert(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(64, 64, bias=False)
        self.up_proj = nn.Linear(64, 64, bias=False)
        self.down_proj = nn.Linear(64, 64, bias=False)


class _MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.experts = nn.ModuleList([_Expert(), _Expert()])
        self.shared_experts = _Expert()


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = _Attention()
        self.mlp = _MLP()


class _TinyLing(nn.Module):
    def __init__(self, *, include_mtp: bool = False):
        super().__init__()
        self.config = SimpleNamespace(num_hidden_layers=1)
        self.model = nn.Module()
        layers = [_Layer()]
        if include_mtp:
            layers.append(_Layer())
        self.model.layers = nn.ModuleList(layers)


def _resolve_autoround_layer_config(layer_config, module_name):
    from auto_round.utils import to_standard_regex

    for pattern, config in layer_config.items():
        if re.search(to_standard_regex(pattern), module_name):
            return config
    return None


def test_autoround_profile_excludes_kda_and_only_routes_experts_to_w4():
    layer_config = build_autoround_layer_config(_TinyLing())

    assert len(layer_config) == 2
    assert _resolve_autoround_layer_config(
        layer_config, "model.layers.0.attention.f_proj"
    ) == {
        "bits": 16,
        "data_type": "fp",
    }
    assert _resolve_autoround_layer_config(
        layer_config, "model.layers.0.attention.b_proj"
    ) == {
        "bits": 16,
        "data_type": "fp",
    }
    assert _resolve_autoround_layer_config(
        layer_config, "model.layers.0.mlp.experts.0.gate_proj"
    ) == {
        "bits": 4,
        "group_size": 64,
    }
    assert _resolve_autoround_layer_config(
        layer_config, "model.layers.0.attention.q_proj"
    ) is None
    assert _resolve_autoround_layer_config(
        layer_config, "model.layers.0.mlp.shared_experts.gate_proj"
    ) is None
    assert not any(key.startswith("model.layers.0.") for key in layer_config)


def test_autoround_excludes_mtp_from_blocks_and_keeps_its_linears_float():
    model = _TinyLing(include_mtp=True)

    assert build_autoround_block_names(model) == [["model.layers.0"]]
    layer_config = build_autoround_layer_config(model)
    mtp_linears = [
        name
        for name, module in model.named_modules()
        if name.startswith("model.layers.1.") and isinstance(module, nn.Linear)
    ]
    assert mtp_linears
    assert len(layer_config) == 3
    assert all(
        _resolve_autoround_layer_config(layer_config, name)
        == {"bits": 16, "data_type": "fp"}
        for name in mtp_linears
    )


def test_gptq_profile_is_symmetric_w8_w4_g64():
    spec = Ling3FlashQuantSpec(
        model_dir="input",
        output_dir="output",
        method="auto-round",
    )
    spec.validate()
    assert spec.method == "autoround"
    assert (spec.base_bits, spec.expert_bits, spec.group_size) == (8, 4, 64)
    assert spec.moe_batch_size is None
    assert spec.offload_to_disk is False
    assert spec.wait_for_submodule_finalizers is True


def test_workflow_rejects_asymmetric_quantization():
    workflow = object.__new__(Ling3FlashWorkflow)
    workflow.model_dir = "input"
    workflow.seed = 42

    with pytest.raises(ValueError, match="sym=true"):
        workflow._quant_ling(
            output_dir="output",
            device="cuda:0",
            quant_cfg={"sym": False},
            method="gptq",
        )


def test_gptq_adapter_delegates_to_recipe_without_offload(tmp_path, monkeypatch):
    import gptqmodel.recipes.ling3_flash as recipe

    captured = {}
    monkeypatch.setattr(
        recipe,
        "quantize_ling3_flash",
        lambda **kwargs: captured.update(kwargs),
    )
    calibration_texts = ["first", "second"]
    spec = Ling3FlashQuantSpec(
        model_dir="input",
        output_dir=str(tmp_path / "output"),
        method="gptq",
        nsamples=2,
        seqlen=4,
    )

    quantize_ling3_flash(spec, calibration_texts=calibration_texts)

    assert captured["bits"] == 8
    assert captured["expert_bits"] == 4
    assert captured["group_size"] == 64
    assert captured["calibration_data"] is calibration_texts
    assert captured["offload_to_disk"] is False
    assert captured["offload_path"] is None
    assert captured["wait_for_submodule_finalizers"] is True


def test_autoround_uses_full_dataset_source_instead_of_first_raw_rows(tmp_path):
    calibration_path = tmp_path / "calibration.jsonl"
    calibration_path.write_text('{"text": "example"}\n', encoding="utf-8")
    spec = Ling3FlashQuantSpec(
        model_dir="input",
        output_dir="output",
        method="autoround",
        calibration_jsonl=str(calibration_path),
    )

    assert autoround_calibration_source(spec, None) == str(
        calibration_path.resolve()
    )


def test_autoround_nonfinite_loss_guard_fails_before_export():
    class _AutoRound:
        @staticmethod
        def _get_loss(*args, **kwargs):
            return torch.tensor(float("nan"))

    autoround = _AutoRound()
    _install_autoround_nonfinite_loss_guard(autoround)

    with pytest.raises(FloatingPointError, match="non-finite tuning loss"):
        autoround._get_loss()


def test_autoround_canonicalizes_native_regex_export(tmp_path):
    from auto_round.utils import to_standard_regex

    output_dir = tmp_path / "quantized"
    output_dir.mkdir()
    config_path = output_dir / "config.json"
    layer_config = build_autoround_layer_config(_TinyLing(include_mtp=True))
    dynamic = {}
    regex_extra = {}
    for pattern, config in layer_config.items():
        standardized = to_standard_regex(pattern)
        regex_extra[standardized] = config
        if config["bits"] >= 16:
            dynamic[f"-:{standardized}"] = {}
        else:
            dynamic[f"+:{standardized}"] = {"bits": config["bits"]}
    expanded_quantization_config = {
        "bits": 8,
        "group_size": 64,
        "sym": True,
        "block_name_to_quantize": ["model.layers"],
        "dynamic": dynamic,
        "extra_config": {
            **regex_extra,
            "model.layers.0.mlp.experts.0.gate_proj": {"bits": 4},
            "model.layers.0.mlp.experts.0.up_proj": {"bits": 4},
            "model.layers.0.attention.f_proj": {
                "bits": 16,
                "data_type": "fp",
            },
            "model.layers.0.attention.b_proj": {
                "bits": 16,
                "data_type": "fp",
            },
            "model.layers.1.attention.q_proj": {
                "bits": 16,
                "data_type": "fp",
            },
            "model.layers.1.mlp.experts.0.gate_proj": {
                "bits": 16,
                "data_type": "fp",
            },
        },
    }
    config_path.write_text(
        json.dumps(
            {
                "num_hidden_layers": 1,
                "num_nextn_predict_layers": 1,
                "quantization_config": expanded_quantization_config,
            }
        ),
        encoding="utf-8",
    )
    standalone_path = output_dir / "quantization_config.json"
    standalone_path.write_text(
        json.dumps(expanded_quantization_config),
        encoding="utf-8",
    )

    canonicalize_autoround_gptqmodel_config(output_dir)

    quantization_config = json.loads(config_path.read_text(encoding="utf-8"))[
        "quantization_config"
    ]
    assert quantization_config["dynamic"] == dynamic
    assert "extra_config" not in quantization_config
    assert len(dynamic) == 3
    standalone_quantization_config = json.loads(
        standalone_path.read_text(encoding="utf-8")
    )
    assert standalone_quantization_config["dynamic"] == dynamic
    assert "extra_config" not in standalone_quantization_config
