"""Regression tests for HMONNX golden/cuda-graph flag handling."""
from __future__ import annotations

import pytest
import torch

from xhmodel_merak.xh_llm.hmonnx.vision_llm_hmonnx_model import VisonLLMHMONNXModel
from xhmodel_merak.xh_llm.hmonnx import hmonnx_model as hmonnx_module
from xhmodel_merak.xh_llm.models.qwen3_5.qwen3_5_hmonnx_inference import XHQwen3_5_HMONNXModel
from xhmodel_merak.xh_llm.models.qwen3_5_moe.qwen3_5_moe_hmonnx_inference import (
    XHQwen3_5MoeHMONNXModel,
)


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


@pytest.fixture(autouse=True)
def fake_hmonnx_sessions(monkeypatch):
    monkeypatch.setattr(hmonnx_module, "HMONNXGolden", _FakeGoldenSession)
    monkeypatch.setattr(hmonnx_module, "HMONNXCUDAGraphInference", _FakeCudaGraphSession)


def test_hmonnx_model_uses_golden_session_when_cuda_graph_disabled():
    model = hmonnx_module.HMONNXModel("prefill.onnx", enable_cuda_graph=False)

    assert model.enable_cuda_graph is False
    assert isinstance(model.hmonnx_session, _FakeGoldenSession)
    model.enable_golden = True
    assert model.hmonnx_session.save_golden is True
    assert model.hmonnx_session.reset_count == 1


def test_hmonnx_model_uses_cuda_graph_session_when_enabled():
    model = hmonnx_module.HMONNXModel("decode.onnx", enable_cuda_graph=True)

    assert model.enable_cuda_graph is True
    assert isinstance(model.hmonnx_session, _FakeCudaGraphSession)


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


@pytest.mark.parametrize("model_cls", [XHQwen3_5_HMONNXModel, XHQwen3_5MoeHMONNXModel])
def test_qwen3_5_hmonnx_forward_splits_split_conv_outputs_by_cache_counts(monkeypatch, model_cls):
    """split_conv_cache exports 3 conv outputs per linear layer plus 1 recurrent output."""

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

    model = model_cls.__new__(model_cls)
    model._kvcache_mixin = _FakeKVCacheMixin()

    out_logits, out_conv, out_recurrent = model.forward()

    assert out_logits is logits
    assert out_conv == conv_outputs
    assert out_recurrent == recurrent_outputs
    assert [cache[0].item() for cache in model._kvcache_mixin.past_conv_caches] == [1.0, 4.0]
    assert [cache.item() for cache in model._kvcache_mixin.past_recurrent_states] == [101.0, 102.0]
