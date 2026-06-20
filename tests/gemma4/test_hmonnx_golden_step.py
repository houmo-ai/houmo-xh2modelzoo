import torch

from xhmodel_merak.xh_llm.hmonnx.base_llm_hmonnx_model import BaseLLMHMONNXModel


class _FakeHMONNXModel:
    def __init__(self):
        self.calls = 0
        self.update_step_calls = 0
        self.devices = []

    def to(self, device):
        self.devices.append(device)
        return self

    def __call__(self, *args):
        self.calls += 1
        return (torch.tensor([self.calls]),)

    def update_step(self):
        self.update_step_calls += 1


def _make_model(prefill: bool):
    model = BaseLLMHMONNXModel.__new__(BaseLLMHMONNXModel)
    model._llm_prefill = prefill
    model._enable_golden = True
    model._device = torch.device("cpu")
    model.prefill_model = _FakeHMONNXModel()
    model.decode_model = _FakeHMONNXModel()
    return model


def test_text_hmonnx_prefill_does_not_double_increment_golden_step():
    model = _make_model(prefill=True)

    out = BaseLLMHMONNXModel.forward(model, torch.tensor([1]))

    assert out.tolist() == [1]
    assert model.prefill_model.calls == 1
    assert model.prefill_model.update_step_calls == 0


def test_text_hmonnx_decode_does_not_double_increment_golden_step():
    model = _make_model(prefill=False)

    out = BaseLLMHMONNXModel.forward(model, torch.tensor([1]))

    assert out.tolist() == [1]
    assert model.decode_model.calls == 1
    assert model.decode_model.update_step_calls == 0
