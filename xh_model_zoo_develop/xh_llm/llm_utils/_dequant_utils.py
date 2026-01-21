import torch
import torch.nn as nn
from typing import Optional, Callable


def qlinear_cuda_old_converter(self: nn.Module):
    from auto_gptq.nn_modules.qlinear.qlinear_cuda_old import QuantLinear as CudaOldQuantLinear

    assert isinstance(self, CudaOldQuantLinear)
    if self.bits in [2, 4, 8]:
        zeros = torch.bitwise_right_shift(
            torch.unsqueeze(self.qzeros, 2).expand(-1, -1, 32 // self.bits),
            self.wf.unsqueeze(0),
        ).to(torch.int16 if self.bits == 8 else torch.int8)

        zeros = zeros + 1
        zeros = torch.bitwise_and(
            zeros, (2**self.bits) - 1
        )  # NOTE: It appears that casting here after the `zeros = zeros + 1` is important.

        zeros = zeros.reshape(-1, 1, zeros.shape[1] * zeros.shape[2])

        scales = self.scales
        scales = scales.reshape(-1, 1, scales.shape[-1])

        weight = torch.bitwise_right_shift(
            torch.unsqueeze(self.qweight, 1).expand(-1, 32 // self.bits, -1),
            self.wf.unsqueeze(-1),
        ).to(torch.int16 if self.bits == 8 else torch.int8)
        weight = torch.bitwise_and(weight, (2**self.bits) - 1)
        weight = weight.reshape(-1, self.group_size, weight.shape[2])
    elif self.bits == 3:
        zeros = self.qzeros.reshape(self.qzeros.shape[0], self.qzeros.shape[1] // 3, 3, 1).expand(-1, -1, -1, 12)
        zeros = zeros >> self.wf.unsqueeze(0)
        zeros[:, :, 0, 10] = (zeros[:, :, 0, 10] & 0x3) | ((zeros[:, :, 1, 0] << 2) & 0x4)
        zeros[:, :, 1, 11] = (zeros[:, :, 1, 11] & 0x1) | ((zeros[:, :, 2, 0] << 1) & 0x6)
        zeros = zeros & 0x7
        zeros = torch.cat(
            [zeros[:, :, 0, :11], zeros[:, :, 1, 1:12], zeros[:, :, 2, 1:11]],
            dim=2,
        )

        zeros = zeros + 1
        zeros = zeros.reshape(-1, 1, zeros.shape[1] * zeros.shape[2])

        scales = self.scales
        scales = scales.reshape(-1, 1, scales.shape[-1])

        weight = self.qweight.reshape(self.qweight.shape[0] // 3, 3, 1, self.qweight.shape[1]).expand(-1, -1, 12, -1)
        weight = (weight >> self.wf.unsqueeze(-1)) & 0x7
        weight[:, 0, 10] = (weight[:, 0, 10] & 0x3) | ((weight[:, 1, 0] << 2) & 0x4)
        weight[:, 1, 11] = (weight[:, 1, 11] & 0x1) | ((weight[:, 2, 0] << 1) & 0x6)
        weight = weight & 0x7
        weight = torch.cat([weight[:, 0, :11], weight[:, 1, 1:12], weight[:, 2, 1:11]], dim=1)
        weight = weight.reshape(-1, self.group_size, weight.shape[2])
    else:
        raise NotImplementedError("Only 2,3,4,8 bits are supported.")

    quant_weight = weight - zeros
    weight = scales * quant_weight
    weight = weight.reshape(weight.shape[0] * weight.shape[1], weight.shape[2])
    quant_weight = quant_weight.reshape(quant_weight.shape[0] * quant_weight.shape[1], quant_weight.shape[2])

    max_val = 2 ** (self.bits - 1)
    min_val = -max_val

    assert quant_weight.max() < max_val and quant_weight.min() >= min_val, f"{quant_weight.max()} {quant_weight}.min()"
    if hasattr(self, "qweight"):
        delattr(self, "qweight")
    if hasattr(self, "qzeros"):
        delattr(self, "qzeros")
    if hasattr(self, "scales"):
        delattr(self, "scales")
    if hasattr(self, "g_idx"):
        delattr(self, "g_idx")
    weight = weight.t()
    quant_weight = quant_weight.t()
    self.register_parameter("weight", nn.Parameter(weight))
    self.register_buffer("quant_weight", quant_weight)
    self.__class__ = nn.Linear
    self.out_features = self.outfeatures
    self.in_features = self.infeatures


def _dequantize_awq_hf_model(native_hf_model: nn.Module):
    from awq.modules.linear.gemm import WQLinear_GEMM
    from awq.utils.packing_utils import reverse_awq_order, unpack_awq
    from transformers.utils.quantization_config import QuantizationMethod

    assert hf_model.config.quantization_config.quant_method == QuantizationMethod.AWQ
    hf_model = native_hf_model
    for name, module in hf_model.named_modules():
        if isinstance(module, WQLinear_GEMM):
            if hasattr(module, "weight"):
                continue
            bits = module.w_bit
            group_size = module.group_size
            iweight = module.qweight
            izeros = module.qzeros
            scales = module.scales

            iweight, izeros = unpack_awq(iweight, izeros, bits)
            # Reverse the order of the iweight and izeros tensors
            iweight, izeros = reverse_awq_order(iweight, izeros, bits)

            # overflow checks
            iweight = torch.bitwise_and(iweight, (2**bits) - 1)
            izeros = torch.bitwise_and(izeros, (2**bits) - 1)

            # fp16 weights
            scales = scales.repeat_interleave(group_size, dim=0)
            izeros = izeros.repeat_interleave(group_size, dim=0)

            # quant weight and weight
            quant_weight = iweight - izeros
            weight = quant_weight * scales
            quant_weight = quant_weight.t().contiguous()
            weight = weight.t().contiguous()

            iweight = None
            izeros = None
            scales = None
            if hasattr(module, "qweight"):
                delattr(module, "qweight")
            if hasattr(module, "qzeros"):
                delattr(module, "qzeros")
            if hasattr(module, "scales"):
                delattr(module, "scales")

            module.register_parameter("weight", nn.Parameter(weight))
            module.register_buffer("quant_weight", quant_weight)
            quant_weight = None
            weight = None
            # module.forward = types.MethodType(linear_forward, module)
            module.__class__ = nn.Linear

    if hf_model.config.tie_word_embeddings:
        hf_model.config.torchscript = True
        hf_model.tie_weights()
        hf_model.config.tie_word_embeddings = False

    hf_model.quantization_method = None  # type: ignore
    hf_model._is_hf_initialized = False  # type: ignore
    return hf_model


def _dequantize_gptqmodel_hf_model(native_hf_model):
    from transformers.utils import is_gptqmodel_available

    assert is_gptqmodel_available(), "We need gptqmodel to dequantize auto-gptq model"
    converter = gptqmodel_torch_qlinear_converter
    from gptqmodel.nn_modules.qlinear import PackableQuantLinear

    for name, module in native_hf_model.named_modules():  # type: ignore
        if isinstance(module, PackableQuantLinear):
            converter(module)
    return native_hf_model


def _dequantize_gptq_hf_model(native_hf_model: nn.Module):
    hf_model = native_hf_model
    from transformers.utils.quantization_config import QuantizationMethod
    from transformers.quantizers.quantizer_gptq import GptqHfQuantizer

    # assert hf_model.config.quantization_config["quant_method"] == QuantizationMethod.GPTQ
    hf_quantizer: GptqHfQuantizer = hf_model.hf_quantizer

    from transformers.utils import is_auto_gptq_available, is_gptqmodel_available

    converter: Optional[Callable] = None

    QuantLinear = hf_quantizer.optimum_quantizer.quant_linear  # type: ignore
    if is_auto_gptq_available():
        from auto_gptq.nn_modules.qlinear.qlinear_cuda import QuantLinear as GeneralQuantLinear
        from auto_gptq.nn_modules.qlinear.qlinear_cuda_old import QuantLinear as CudaOldQuantLinear
        from auto_gptq.nn_modules.qlinear.qlinear_exllama import QuantLinear as ExllamaQuantLinear
        from auto_gptq.nn_modules.qlinear.qlinear_exllamav2 import QuantLinear as Exllamav2QuantLinear
        from auto_gptq.nn_modules.qlinear.qlinear_marlin import QuantLinear as MarlinQuantLinear

        if QuantLinear is GeneralQuantLinear:
            converter = general_qlinear_converter
        elif QuantLinear is CudaOldQuantLinear:
            converter = qlinear_cuda_old_converter
        elif QuantLinear is ExllamaQuantLinear:
            converter = None
        elif QuantLinear is Exllamav2QuantLinear:
            converter = None
        elif QuantLinear is MarlinQuantLinear:
            converter = None

    if is_gptqmodel_available():
        if hasattr(QuantLinear, "dequantize_weight"):
            converter = gptqmodel_torch_qlinear_converter
        else:
            raise NotImplementedError(f"Not implemented for {QuantLinear} yet")
        if False:
            from gptqmodel.nn_modules.qlinear.marlin import MarlinQuantLinear
            from gptqmodel.nn_modules.qlinear.torch import TorchQuantLinear

            if QuantLinear is TorchQuantLinear:
                converter = gptqmodel_torch_qlinear_converter
            elif QuantLinear is MarlinQuantLinear:
                converter = None

    assert converter is not None, f"Not implemented for {QuantLinear} yet"

    for name, module in hf_model.named_modules():  # type: ignore
        if isinstance(module, QuantLinear):
            if converter is not None:
                converter(module)
            else:
                raise NotImplementedError(f"Not implemented for {type(QuantLinear)} yet")

    hf_model.quantization_method = None  # type: ignore
    hf_model._is_hf_initialized = False  # type: ignore
    return hf_model


def general_qlinear_converter(self: nn.Module):
    if self.bits in [2, 4, 8]:
        zeros = torch.bitwise_right_shift(
            torch.unsqueeze(self.qzeros, 2).expand(-1, -1, 32 // self.bits),
            self.wf.unsqueeze(0),
        ).to(torch.int16 if self.bits == 8 else torch.int8)
        zeros = torch.bitwise_and(zeros, (2**self.bits) - 1)

        zeros = zeros + 1
        zeros = zeros.reshape(self.scales.shape)

        weight = torch.bitwise_right_shift(
            torch.unsqueeze(self.qweight, 1).expand(-1, 32 // self.bits, -1),
            self.wf.unsqueeze(-1),
        ).to(torch.int16 if self.bits == 8 else torch.int8)
        weight = torch.bitwise_and(weight, (2**self.bits) - 1)
    elif self.bits == 3:
        zeros = self.qzeros.reshape(self.qzeros.shape[0], self.qzeros.shape[1] // 3, 3, 1).expand(-1, -1, -1, 12)
        zeros = zeros >> self.wf.unsqueeze(0)
        zeros[:, :, 0, 10] = (zeros[:, :, 0, 10] & 0x3) | ((zeros[:, :, 1, 0] << 2) & 0x4)
        zeros[:, :, 1, 11] = (zeros[:, :, 1, 11] & 0x1) | ((zeros[:, :, 2, 0] << 1) & 0x6)
        zeros = zeros & 0x7
        zeros = torch.cat(
            [zeros[:, :, 0, :11], zeros[:, :, 1, 1:12], zeros[:, :, 2, 1:11]],
            dim=2,
        )
        zeros = zeros + 1
        zeros = zeros.reshape(self.scales.shape)

        weight = self.qweight.reshape(self.qweight.shape[0] // 3, 3, 1, self.qweight.shape[1]).expand(-1, -1, 12, -1)
        weight = (weight >> self.wf.unsqueeze(-1)) & 0x7
        weight[:, 0, 10] = (weight[:, 0, 10] & 0x3) | ((weight[:, 1, 0] << 2) & 0x4)
        weight[:, 1, 11] = (weight[:, 1, 11] & 0x1) | ((weight[:, 2, 0] << 1) & 0x6)
        weight = weight & 0x7
        weight = torch.cat([weight[:, 0, :11], weight[:, 1, 1:12], weight[:, 2, 1:11]], dim=1)
    else:
        raise NotImplementedError("Only 2,3,4,8 bits are supported.")

    weight = weight.reshape(weight.shape[0] * weight.shape[1], weight.shape[2])
    # weights = self.scales[self.g_idx.long()] * (weight - zeros[self.g_idx.long()])

    quant_weight = weight - zeros[self.g_idx.long()]
    weight = self.scales[self.g_idx.long()] * quant_weight
    # weight = weight.reshape(weight.shape[0] * weight.shape[1], weight.shape[2])
    # quant_weight = quant_weight.reshape(quant_weight.shape[0] * quant_weight.shape[1], quant_weight.shape[2])

    maxq = (2**self.bits) / 2

    assert quant_weight.max() < maxq and quant_weight.min() >= -maxq, f"{quant_weight.max()} {quant_weight}.min()"
    if hasattr(self, "qweight"):
        delattr(self, "qweight")
    if hasattr(self, "qzeros"):
        delattr(self, "qzeros")
    if hasattr(self, "scales"):
        delattr(self, "scales")
    if hasattr(self, "g_idx"):
        delattr(self, "g_idx")
    weight = weight.t()
    quant_weight = quant_weight.t()
    self.register_parameter("weight", nn.Parameter(weight))
    self.register_buffer("quant_weight", quant_weight)
    self.__class__ = nn.Linear
    self.out_features = self.outfeatures
    self.in_features = self.infeatures


def gptqmodel_torch_qlinear_converter(self: nn.Module):
    import torch as t  # conflict with torch.py
    ori_device = self.g_idx.device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    self.to(device)

    if self.bits in [2, 4, 8]:
        zeros = t.bitwise_right_shift(
            t.unsqueeze(self.qzeros, 2).expand(-1, -1, self.pack_factor),
            self.wf_unsqueeze_zero,  # self.wf.unsqueeze(0),
        ).to(self.dequant_dtype)
        zeros = t.bitwise_and(zeros, self.maxq).reshape(self.scales.shape)

        weight = t.bitwise_and(
            t.bitwise_right_shift(
                t.unsqueeze(self.qweight, 1).expand(-1, self.pack_factor, -1),
                self.wf_unsqueeze_neg_one,  # self.wf.unsqueeze(-1)
            ).to(self.dequant_dtype),
            self.maxq,
        )
    elif self.bits == 3:
        zeros = self.qzeros.reshape(self.qzeros.shape[0], self.qzeros.shape[1] // 3, 3, 1).expand(-1, -1, -1, 12)
        zeros = zeros >> self.wf_unsqueeze_zero  # self.wf.unsqueeze(0)
        zeros[:, :, 0, 10] = (zeros[:, :, 0, 10] & 0x3) | ((zeros[:, :, 1, 0] << 2) & 0x4)
        zeros[:, :, 1, 11] = (zeros[:, :, 1, 11] & 0x1) | ((zeros[:, :, 2, 0] << 1) & 0x6)
        zeros = zeros & 0x7
        zeros = t.cat(
            [zeros[:, :, 0, :11], zeros[:, :, 1, 1:12], zeros[:, :, 2, 1:11]],
            dim=2,
        ).reshape(self.scales.shape)

        weight = self.qweight.reshape(self.qweight.shape[0] // 3, 3, 1, self.qweight.shape[1]).expand(-1, -1, 12, -1)
        weight = (weight >> self.wf_unsqueeze_neg_one) & 0x7  # self.wf.unsqueeze(-1)
        weight[:, 0, 10] = (weight[:, 0, 10] & 0x3) | ((weight[:, 1, 0] << 2) & 0x4)
        weight[:, 1, 11] = (weight[:, 1, 11] & 0x1) | ((weight[:, 2, 0] << 1) & 0x6)
        weight = weight & 0x7
        weight = t.cat([weight[:, 0, :11], weight[:, 1, 1:12], weight[:, 2, 1:11]], dim=1)
    weight = weight.reshape(weight.shape[0] * weight.shape[1], weight.shape[2])

    quant_weight = weight - zeros[self.g_idx.long()]
    weight = self.scales[self.g_idx.long()] * quant_weight
    maxq = 2 ** (self.bits - 1)
    # diff = quant_weight.to(torch.int32) - quant_weight
    # error = diff.abs().float()
    # assert torch.allclose(error, t.tensor(0.0), atol=1e-3), f"{error.max()}"
    assert (
        quant_weight.max() < maxq and quant_weight.min() >= -maxq
    ), f"min={quant_weight.min()}, max={quant_weight.max()}, not in [{-maxq}, {maxq})"
    if hasattr(self, "qweight"):
        delattr(self, "qweight")
    if hasattr(self, "qzeros"):
        delattr(self, "qzeros")
    if hasattr(self, "scales"):
        delattr(self, "scales")
    if hasattr(self, "g_idx"):
        delattr(self, "g_idx")
    weight = weight.t()
    quant_weight = quant_weight.t()
    self.register_parameter("weight", nn.Parameter(weight))
    self.register_buffer("quant_weight", quant_weight)
    self.__class__ = nn.Linear
    self.to(ori_device)
