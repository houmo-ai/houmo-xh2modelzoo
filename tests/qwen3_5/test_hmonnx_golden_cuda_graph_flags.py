"""Regression tests for HMONNX golden/cuda-graph flag handling."""

from __future__ import annotations

import pytest
import torch

from xhmodel_merak.xh_llm.hmonnx import hmonnx_model as hmonnx_module
from xhmodel_merak.xh_llm.hmonnx.vision_llm_hmonnx_model import VisonLLMHMONNXModel


class _FakeSession:
    def __init__(self, hmonnx_path: str):
        self.hmonnx_path = hmonnx_path
        self.save_golden = False
        self.reset_count = 0
        self.device = None

    def reset_step(self):
        self.reset_count += 1

    def to(self, device):
        self.device = device

    def forward(self, *args):
        return args

    def to_fast_mode(self):
        pass


class _FakeGoldenSession(_FakeSession):
    pass


class _FakeCudaGraphSession(_FakeSession):
    pass


class _FakeV2Config:
    def __init__(self):
        self.enable_cuda_graph = False
        self.enable_auto_offload = False
        self.exec_devices = []
        self.enable_golden = False


class _FakeV2Session(_FakeSession):
    def __init__(self, hmonnx_path: str, session_config: _FakeV2Config, onnx_graph=None):
        super().__init__(hmonnx_path)
        self.session_config = session_config
        self.onnx_graph = onnx_graph
        self.device = torch.device("cuda:1")
        self.auto_offload_calls = []

    @classmethod
    def from_onnx_graph(cls, hmonnx_path: str, onnx_graph, session_config: _FakeV2Config):
        return cls(hmonnx_path, session_config, onnx_graph=onnx_graph)

    def enable_auto_offload(self, enable=True):
        self.auto_offload_calls.append(enable)


@pytest.fixture(autouse=True)
def fake_hmonnx_sessions(monkeypatch):
    monkeypatch.setattr(hmonnx_module, "HMONNXGolden", _FakeGoldenSession)
    monkeypatch.setattr(hmonnx_module, "HMONNXCUDAGraphInference", _FakeCudaGraphSession)


@pytest.fixture
def fake_hmonnx_v2(monkeypatch):
    monkeypatch.setattr(hmonnx_module, "HMONNXInferenceConfig", _FakeV2Config)
    monkeypatch.setattr(hmonnx_module, "HMONNXInferenceV2", _FakeV2Session)


def test_hmonnx_model_uses_golden_session_when_cuda_graph_disabled():
    model = hmonnx_module.HMONNXModel("prefill.onnx", enable_cuda_graph=False)

    assert model.enable_cuda_graph is False
    assert isinstance(model.hmonnx_session, _FakeGoldenSession)
    assert model.hmonnx_session.device == model.device
    model.enable_golden = True
    assert model.hmonnx_session.save_golden is True
    assert model.hmonnx_session.reset_count == 1


def test_hmonnx_model_uses_cuda_graph_session_when_enabled():
    model = hmonnx_module.HMONNXModel("decode.onnx", enable_cuda_graph=True)

    assert model.enable_cuda_graph is True
    assert isinstance(model.hmonnx_session, _FakeCudaGraphSession)
    assert model.hmonnx_session.device == model.device


def test_hmonnx_model_legacy_to_moves_session_device():
    model = hmonnx_module.HMONNXModel("prefill.onnx", enable_cuda_graph=False)

    model.to("cpu")

    assert model.hmonnx_session.device == model.device


def test_hmonnx_model_v2_configures_session_from_env(monkeypatch, fake_hmonnx_v2):
    monkeypatch.setenv(hmonnx_module.HMONNXModel.ENV_ENABLE_INFERENCE_V2, "1")

    model = hmonnx_module.HMONNXModel(
        "prefill.onnx",
        onnx_graph=object(),
        enable_cuda_graph=True,
        enable_auto_offload=True,
        enable_golden=True,
        device_map=[0, 1],
    )

    assert isinstance(model.hmonnx_session, _FakeV2Session)
    assert model.device == torch.device("cuda:1")
    assert model.hmonnx_session.session_config.enable_cuda_graph is True
    assert model.hmonnx_session.session_config.enable_auto_offload is True
    assert model.hmonnx_session.session_config.enable_golden is True
    assert model.hmonnx_session.session_config.exec_devices == [0, 1]


def test_hmonnx_model_v2_to_keeps_session_owned_device(monkeypatch, fake_hmonnx_v2):
    monkeypatch.setenv(hmonnx_module.HMONNXModel.ENV_ENABLE_INFERENCE_V2, "1")
    model = hmonnx_module.HMONNXModel("decode.onnx")

    model.to("cpu")

    assert model.device == torch.device("cuda:1")
    assert model.hmonnx_session.device == torch.device("cuda:1")


def test_hmonnx_model_v2_enable_auto_offload_calls_session(monkeypatch, fake_hmonnx_v2):
    monkeypatch.setenv(hmonnx_module.HMONNXModel.ENV_ENABLE_INFERENCE_V2, "1")
    model = hmonnx_module.HMONNXModel("decode.onnx")

    model.enable_auto_offload()

    assert model.hmonnx_session.auto_offload_calls == [True]


def test_hmonnx_model_v2_enable_golden_calls_session(monkeypatch, fake_hmonnx_v2):
    monkeypatch.setenv(hmonnx_module.HMONNXModel.ENV_ENABLE_INFERENCE_V2, "1")
    model = hmonnx_module.HMONNXModel("decode.onnx")

    model.enable_golden = True

    assert model.hmonnx_session.save_golden is True
    assert model.hmonnx_session.reset_count == 1


def test_hmonnx_base_model_enable_golden_does_not_require_missing_cuda_graph_attr():
    base = hmonnx_module.HMONNXBaseModel()
    base.prefill_model = hmonnx_module.HMONNXModel("prefill.onnx", enable_cuda_graph=False)

    base.enable_golden = True

    assert base.enable_golden is True
    assert base.prefill_model.hmonnx_session.save_golden is True


def test_hmonnx_base_model_rejects_golden_when_child_uses_cuda_graph():
    base = hmonnx_module.HMONNXBaseModel()
    base.decode_model = hmonnx_module.HMONNXModel("decode.onnx", enable_cuda_graph=True)

    with pytest.raises(RuntimeError, match="CUDA Graph"):
        base.enable_golden = True


def test_qwen3_5_hmonnx_forward_splits_split_conv_outputs_by_cache_counts(monkeypatch):
    """split_conv_cache exports 3 conv outputs per linear layer plus 1 recurrent output."""
    qwen35_module = pytest.importorskip("xhmodel_merak.xh_llm.models.qwen3_5.qwen3_5_hmonnx_inference")
    qwen35_moe_module = pytest.importorskip("xhmodel_merak.xh_llm.models.qwen3_5_moe.qwen3_5_moe_hmonnx_inference")

    class _FakeKVCacheMixin:
        split_conv_cache = True

        def __init__(self):
            self.past_conv_caches = [
                (torch.zeros(1), torch.zeros(1), torch.zeros(1)),
                (torch.zeros(1), torch.zeros(1), torch.zeros(1)),
            ]
            self.past_recurrent_states = [torch.zeros(1), torch.zeros(1)]

    logits = torch.tensor([0.0])
    conv_outputs = [torch.tensor([float(idx + 1)]) for idx in range(6)]
    recurrent_outputs = [torch.tensor([101.0]), torch.tensor([102.0])]

    def fake_base_forward(self, *args):
        return [logits, *conv_outputs, *recurrent_outputs]

    monkeypatch.setattr(VisonLLMHMONNXModel, "forward", fake_base_forward)

    for model_cls in (qwen35_module.XHQwen3_5_HMONNXModel, qwen35_moe_module.XHQwen3_5MoeHMONNXModel):
        model = model_cls.__new__(model_cls)
        model._kvcache_mixin = _FakeKVCacheMixin()

        out_logits, out_conv, out_recurrent = model.forward()

        assert out_logits is logits
        assert out_conv == conv_outputs
        assert out_recurrent == recurrent_outputs
        assert [cache[0].item() for cache in model._kvcache_mixin.past_conv_caches] == [1.0, 4.0]
        assert [cache.item() for cache in model._kvcache_mixin.past_recurrent_states] == [101.0, 102.0]
