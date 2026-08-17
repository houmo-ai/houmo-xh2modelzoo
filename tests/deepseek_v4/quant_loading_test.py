from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from xhmodel_merak.xh_llm.models.deepseek_v4 import deepseek_v4_model
from xhmodel_merak.xh_llm.models.deepseek_v4.deepseek_v4_model import XHDeepSeekV4Model


class _FakePacked(nn.Module):
    def __init__(self, bits: int) -> None:
        super().__init__()
        self.bits = int(bits)


class _FakeExpert(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = _FakePacked(4)
        self.up_proj = _FakePacked(4)
        self.down_proj = _FakePacked(4)


class _FakeLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = nn.Module()
        for name in ("q_a_proj", "q_b_proj", "kv_proj", "o_b_proj"):
            setattr(self.self_attn, name, _FakePacked(8))
        self.mlp = nn.Module()
        self.mlp.experts = nn.ModuleList(_FakeExpert() for _ in range(256))
        self.mlp.shared_experts = nn.Module()
        for name in ("gate_proj", "up_proj", "down_proj"):
            setattr(self.mlp.shared_experts, name, _FakePacked(8))


class _FakePackedModel(nn.Module):
    def __init__(self, layer_count: int) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(_FakeLayer() for _ in range(layer_count))
        self.config = SimpleNamespace(num_hidden_layers=layer_count)


def test_packed_inventory_requires_w4_experts_and_w8_base() -> None:
    model = _FakePackedModel(2)

    inventory = XHDeepSeekV4Model._validate_packed_quant_inventory(
        model,
        packed_type=_FakePacked,
    )

    assert inventory == {
        "layers": 2,
        "packed_total": 1550,
        "routed_expert_w4": 1536,
        "base_w8": 14,
    }


def test_packed_inventory_rejects_missing_or_wrong_precision() -> None:
    model = _FakePackedModel(1)
    model.model.layers[0].mlp.experts[0].gate_proj.bits = 8

    with pytest.raises(RuntimeError, match="wrong_bits"):
        XHDeepSeekV4Model._validate_packed_quant_inventory(
            model,
            packed_type=_FakePacked,
        )

    model = _FakePackedModel(1)
    del model.model.layers[0].self_attn.q_a_proj
    with pytest.raises(RuntimeError, match="missing"):
        XHDeepSeekV4Model._validate_packed_quant_inventory(
            model,
            packed_type=_FakePacked,
        )


def test_autoround_uses_gptqmodel_loader_instead_of_transformers(monkeypatch) -> None:
    calls: list[tuple[str, object]] = []
    packed_model = nn.Module()
    converted_model = nn.Module()
    converted_model.linear = nn.Linear(1, 1, bias=False)
    converted_model.linear.register_buffer("quant_weight", torch.ones(1, 1, dtype=torch.int8))

    monkeypatch.setattr(
        deepseek_v4_model.AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(quantization_config={"quant_method": "auto_round"}),
    )

    def fake_load(cls, path, **kwargs):
        calls.append(("load", (path, kwargs)))
        return packed_model

    def fake_inventory(cls, model, **kwargs):
        assert model is packed_model
        calls.append(("inventory", kwargs))
        return {"layers": 1, "packed_total": 1, "routed_expert_w4": 0, "base_w8": 1}

    def fake_dequantize(cls, model):
        assert model is packed_model
        calls.append(("dequantize", model))
        return converted_model

    def fake_postprocess(cls, model, **kwargs):
        assert model is converted_model
        calls.append(("postprocess", kwargs))
        return model

    monkeypatch.setattr(XHDeepSeekV4Model, "_load_gptqmodel", classmethod(fake_load))
    monkeypatch.setattr(
        XHDeepSeekV4Model,
        "_validate_packed_quant_inventory",
        classmethod(fake_inventory),
    )
    monkeypatch.setattr(
        XHDeepSeekV4Model,
        "_dequantize_gptqmodel_hf_model",
        classmethod(fake_dequantize),
    )
    monkeypatch.setattr(
        XHDeepSeekV4Model,
        "_postprocess_gptqmodel_structure",
        classmethod(fake_postprocess),
    )

    result = XHDeepSeekV4Model.get_hf_model("/tmp/packed", device_map="cpu")

    assert result is converted_model
    assert [name for name, _ in calls] == ["load", "inventory", "dequantize", "postprocess"]
    assert calls[0][1] == ("/tmp/packed", {"device_map": "cpu"})


def test_packed_weight_only_dispatch_skips_dense_dequantization(monkeypatch) -> None:
    calls: list[str] = []
    load_kwargs: dict[str, object] = {}
    packed_model = nn.Module()
    converted_model = nn.Module()

    monkeypatch.setattr(
        deepseek_v4_model.AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(quantization_config={"quant_method": "auto_round"}),
    )
    def fake_load(cls, *args, **kwargs):
        load_kwargs.update(kwargs)
        return packed_model

    monkeypatch.setattr(XHDeepSeekV4Model, "_load_gptqmodel", classmethod(fake_load))
    monkeypatch.setattr(
        XHDeepSeekV4Model,
        "_validate_packed_quant_inventory",
        classmethod(
            lambda cls, model, **kwargs: {
                "layers": 1,
                "packed_total": 0,
                "routed_expert_w4": 0,
                "base_w8": 0,
            }
        ),
    )

    def fake_packed(cls, model):
        assert model is packed_model
        calls.append("packed")
        return converted_model

    def fail_dense(cls, model):
        raise AssertionError("dense dequantization must not run")

    monkeypatch.setattr(
        XHDeepSeekV4Model,
        "_retain_gptqmodel_packed_hf_model",
        classmethod(fake_packed),
    )
    monkeypatch.setattr(
        XHDeepSeekV4Model,
        "_dequantize_gptqmodel_hf_model",
        classmethod(fail_dense),
    )
    monkeypatch.setattr(
        XHDeepSeekV4Model,
        "_postprocess_gptqmodel_structure",
        classmethod(lambda cls, model, **kwargs: model),
    )

    result = XHDeepSeekV4Model.get_hf_model(
        "/tmp/packed",
        device_map="cpu",
        packed_weight_only=True,
    )

    assert result is converted_model
    assert calls == ["packed"]
    assert load_kwargs == {"device_map": "cpu", "use_meta_shell": True}
