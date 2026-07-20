import torch
import torch.nn as nn

from xhmodel_merak.xh_llm._dequant_converter import gptqmodel_torch_qlinear_converter


class LazyUnpackGPTQLinear(nn.Module):
    """Minimal GPTQModel 5.8-style module with lazily initialized unpack buffers."""

    def __init__(self):
        super().__init__()
        self.bits = 4
        self.pack_factor = 8
        self.dequant_dtype = torch.int8
        self.maxq = 15
        self.unpack_init_calls = 0
        self.register_buffer("qweight", torch.zeros((8, 64), dtype=torch.int32))
        self.register_buffer("qzeros", torch.zeros((1, 8), dtype=torch.int32))
        self.register_buffer("scales", torch.ones((1, 64), dtype=torch.float16))
        self.register_buffer("g_idx", torch.zeros(64, dtype=torch.int32))

    def _init_wf_unsqueeze_buffers(self):
        self.unpack_init_calls += 1
        wf = torch.arange(0, 32, self.bits, dtype=torch.int32).unsqueeze(0)
        self.register_buffer("wf_unsqueeze_zero", wf.unsqueeze(0), persistent=False)
        self.register_buffer("wf_unsqueeze_neg_one", wf.unsqueeze(-1), persistent=False)


def test_gptqmodel_converter_initializes_lazy_unpack_buffers():
    module = LazyUnpackGPTQLinear()

    assert not hasattr(module, "wf_unsqueeze_zero")
    assert not hasattr(module, "wf_unsqueeze_neg_one")

    gptqmodel_torch_qlinear_converter(module)

    assert isinstance(module, nn.Linear)
    assert module.unpack_init_calls == 1
    assert module.weight.shape == (64, 64)
    assert module.quant_weight.shape == (64, 64)
