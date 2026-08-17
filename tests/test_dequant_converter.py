from types import SimpleNamespace

import torch
import torch.nn as nn

from xhmodel_merak.xh_llm._dequant_converter import (
    gptqmodel_torch_qlinear_converter,
    gptqmodel_torch_qlinear_packed_converter,
)
from xhmodel_merak.xh_llm.base_model import XHBaseModel
from xhquant.nn import GPTQPackedLinear


class LazyUnpackGPTQLinear(nn.Module):
    """Minimal GPTQModel 5.8-style module with lazily initialized unpack buffers."""

    def __init__(self):
        super().__init__()
        self.bits = 4
        self.pack_factor = 8
        self.dequant_dtype = torch.int8
        self.maxq = 15
        self.group_size = 64
        self.infeatures = 64
        self.outfeatures = 64
        self.in_features = self.infeatures
        self.out_features = self.outfeatures
        self.pack_dtype_bits = 32
        self.sym = True
        self.desc_act = False
        self.adapter = None
        self.bias = None
        self.name = "fake.linear"
        self._qzero_format = 2
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

    def qzero_format(self):
        return self._qzero_format


def test_gptqmodel_converter_initializes_lazy_unpack_buffers():
    module = LazyUnpackGPTQLinear()

    assert not hasattr(module, "wf_unsqueeze_zero")
    assert not hasattr(module, "wf_unsqueeze_neg_one")

    gptqmodel_torch_qlinear_converter(module)

    assert isinstance(module, nn.Linear)
    assert module.unpack_init_calls == 1
    assert module.weight.shape == (64, 64)
    assert module.quant_weight.shape == (64, 64)


def test_gptqmodel_packed_converter_transfers_ownership_without_unpacking():
    dense = LazyUnpackGPTQLinear()
    source = LazyUnpackGPTQLinear()
    source_qweight = source.qweight
    source_qzeros = source.qzeros
    source_scales = source.scales
    source_g_idx = source.g_idx

    gptqmodel_torch_qlinear_converter(dense)
    packed = gptqmodel_torch_qlinear_packed_converter(source)

    assert isinstance(packed, GPTQPackedLinear)
    assert packed.weight is None
    assert (packed.out_features, packed.in_features) == dense.weight.shape
    assert packed.packed_weight.qweight is source_qweight
    assert packed.packed_weight.qzeros is source_qzeros
    assert packed.packed_weight.scales is source_scales
    assert packed.packed_weight.g_idx is source_g_idx
    assert source.unpack_init_calls == 0

    qweight, scale = packed.packed_weight.to_hm_ssfp(nshare=64)

    assert qweight.dtype == torch.int8
    assert qweight.shape == (1, 64, 64)
    assert scale.dtype == torch.float16
    assert scale.shape == (1, 1, 64)
    assert torch.equal(
        qweight.reshape(64, 64).T,
        dense.quant_weight.to(torch.int8),
    )
    assert torch.equal(scale[:, 0], torch.ones(1, 64, dtype=torch.float16))


def test_get_hf_model_can_retain_gptq_words_without_dense_dequantization(monkeypatch):
    calls = []
    quant_config = {"quant_method": "gptq"}
    packed_model = SimpleNamespace(
        config=SimpleNamespace(quantization_config=quant_config)
    )

    monkeypatch.setattr(
        "xhmodel_merak.xh_llm.base_model.AutoConfig.from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(
            quantization_config=quant_config
        ),
    )
    monkeypatch.setattr(
        XHBaseModel,
        "_load_gptqmodel",
        classmethod(lambda cls, *args, **kwargs: packed_model),
    )
    monkeypatch.setattr(
        XHBaseModel,
        "_retain_gptqmodel_packed_hf_model",
        classmethod(lambda cls, model: calls.append("packed") or model),
    )
    monkeypatch.setattr(
        XHBaseModel,
        "_dequantize_gptqmodel_hf_model",
        classmethod(
            lambda cls, model: (_ for _ in ()).throw(
                AssertionError("dense GPTQ dequantization must not run")
            )
        ),
    )
    monkeypatch.setattr(
        XHBaseModel,
        "_postprocess_gptqmodel_structure",
        classmethod(lambda cls, model, **kwargs: model),
    )

    result = XHBaseModel.get_hf_model(
        "/tmp/packed",
        gptq_weight_mode="packed",
        device_map="cpu",
    )

    assert result is packed_model
    assert calls == ["packed"]
