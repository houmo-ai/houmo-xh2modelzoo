from types import SimpleNamespace

import pytest
import torch

from xhmodel_merak.configuration_utils import HFModelConfig
from xhmodel_merak.xh_llm.base_model import XHBaseModel
from xhmodel_merak.xh_llm.utils import hf_auto_offload, resolve_model_dtype


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, torch.float16),
        ("float16", torch.float16),
        ("fp16", torch.float16),
        ("torch.float16", torch.float16),
        ("bf16", torch.bfloat16),
        (torch.float32, torch.float32),
    ],
)
def test_resolve_model_dtype(value, expected):
    assert resolve_model_dtype(value) is expected


def test_resolve_model_dtype_rejects_non_floating_dtype():
    with pytest.raises(ValueError, match="Unsupported model dtype"):
        resolve_model_dtype(torch.int8)


def test_model_config_and_xh_model_default_to_fp16():
    config = HFModelConfig(model_name="dummy", hf_model="/tmp/dummy")
    model = XHBaseModel(config)

    assert config.dtype == "float16"
    assert model.dtype is torch.float16


def test_xh_model_uses_explicit_config_dtype():
    config = HFModelConfig(
        model_name="dummy",
        hf_model="/tmp/dummy",
        dtype="bfloat16",
    )
    model = XHBaseModel(config)

    assert model.dtype is torch.bfloat16


def test_load_hf_model_respects_explicit_dtype(monkeypatch):
    captured = {}

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(model_dir, **kwargs):
            captured.update(kwargs)
            return torch.nn.Linear(1, 1)

    class DummyModel(XHBaseModel):
        HF_AUTO_MODEL_CLS = FakeAutoModel

    DummyModel._load_hf_model("/tmp/dummy", dtype=torch.float16)

    assert captured["dtype"] is torch.float16
    assert "torch_dtype" not in captured


def test_wrap_observes_xh_dtype_before_dtype_sensitive_preprocessing(monkeypatch):
    observed = {}

    class DummyModel(XHBaseModel):
        def _wraped_pre(self, hf_model):
            observed["pre_dtype"] = hf_model.weight.dtype

    config = HFModelConfig(
        model_name="dummy",
        hf_model="/tmp/dummy",
        dtype="float16",
    )
    model = DummyModel(config)
    hf_model = torch.nn.Linear(2, 2, dtype=torch.bfloat16)
    monkeypatch.setattr(
        "xhmodel_merak.xh_llm.base_model.wrap_llm_model",
        lambda native_model, wrap_cfg: native_model,
    )

    model.init_wrap_model(hf_model)

    assert observed["pre_dtype"] is torch.float16
    assert hf_model.weight.dtype is torch.float16


def test_empty_and_compatible_native_models_receive_xh_dtype(monkeypatch):
    class DummyHFModel:
        pass

    class DummyModel(XHBaseModel):
        HF_MODEL_CLS = DummyHFModel

    config = HFModelConfig(
        model_name="dummy",
        hf_model="/tmp/dummy",
        dtype="float16",
    )
    model = DummyModel(config)
    calls = []

    monkeypatch.setattr(
        DummyModel,
        "get_empty_hf_model",
        classmethod(
            lambda cls, model_dir, **kwargs: calls.append(("empty", kwargs))
            or DummyHFModel()
        ),
    )
    monkeypatch.setattr(
        DummyModel,
        "get_compatible_model",
        classmethod(
            lambda cls, model_dir, **kwargs: calls.append(("compatible", kwargs))
            or SimpleNamespace()
        ),
    )

    assert isinstance(model.get_empty_native_model(), DummyHFModel)
    model.get_compatible_native_model()

    assert calls == [
        ("empty", {"dtype": torch.float16}),
        ("compatible", {"dtype": torch.float16}),
    ]


def test_hf_auto_offload_uses_configured_dtype(monkeypatch):
    captured = {}

    def fake_get_device_map(
        model,
        device_map,
        max_memory,
        hf_quantizer,
        dtype,
        keep_in_fp32_regex,
    ):
        captured["dtype"] = dtype
        return {"": "cpu"}

    def fake_dispatch_model(model, **kwargs):
        captured["dispatch"] = kwargs

    monkeypatch.setattr(
        "transformers.modeling_utils._get_device_map",
        fake_get_device_map,
    )
    monkeypatch.setattr("accelerate.dispatch_model", fake_dispatch_model)

    model = torch.nn.Linear(2, 2)
    model._skip_keys_device_placement = None
    hf_auto_offload(model, dtype="bfloat16")

    assert captured["dtype"] is torch.bfloat16
    assert captured["dispatch"]["device_map"] == {"": "cpu"}
