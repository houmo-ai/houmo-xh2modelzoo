import math
from logging import getLogger

import numpy as np
import torch
import torch.nn as nn
import transformers

logger = getLogger(__name__)
try:
    import autogptq_cuda_64
    import autogptq_cuda_256

    _autogptq_cuda_available = True
except ImportError:
    logger.warning("CUDA extension not installed.")
    autogptq_cuda_256 = None
    autogptq_cuda_64 = None
    _autogptq_cuda_available = False


class QuantLinear_non_zero(nn.Module):
    QUANT_TYPE = "cuda-old"

    def __init__(
        self,
        bits,
        group_size,
        infeatures,
        outfeatures,
        bias,
        use_cuda_fp16=True,
        kernel_switch_threshold=128,
        trainable=False,
        weight_dtype=torch.float16,
    ):
        super().__init__()
        global _autogptq_cuda_available
        if bits not in [2, 3, 4, 8]:
            raise NotImplementedError("Only 2,3,4,8 bits are supported.")
        if trainable:
            _autogptq_cuda_available = False
        self.infeatures = infeatures
        self.outfeatures = outfeatures
        self.bits = bits
        self.group_size = group_size if group_size != -1 else infeatures
        self.maxq = 2**self.bits - 1

        self.quant_type = 0
        self.save_weight = False
        self.dequant_weight = None
        self.quant_val = None

        self.register_buffer("qweight", torch.zeros((infeatures // 32 * self.bits, outfeatures), dtype=torch.int32))
        self.register_buffer(
            "qzeros", torch.zeros((math.ceil(infeatures / self.group_size), outfeatures), dtype=weight_dtype)
        )
        self.register_buffer(
            "scales", torch.zeros((math.ceil(infeatures / self.group_size), outfeatures), dtype=weight_dtype)
        )
        self.register_buffer(
            "g_idx", torch.tensor([i // self.group_size for i in range(infeatures)], dtype=torch.int32)
        )

        if bias:
            self.register_buffer("bias", torch.zeros((outfeatures), dtype=weight_dtype))
        else:
            self.bias = None
        self.half_indim = self.infeatures // 2

        self.use_cuda_fp16 = use_cuda_fp16 if bits != 8 else False

        # is performed by unpacking the weights and using torch.matmul
        if self.bits in [2, 4, 8]:
            self.wf = torch.tensor(list(range(0, 32, self.bits)), dtype=torch.int32).unsqueeze(0)
        elif self.bits == 3:
            self.wf = torch.tensor(
                [
                    [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 0],
                    [0, 1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31],
                    [0, 2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 0],
                ],
                dtype=torch.int32,
            ).reshape(1, 3, 12)

        self.kernel_switch_threshold = kernel_switch_threshold
        self.autogptq_cuda_available = _autogptq_cuda_available
        self.autogptq_cuda = autogptq_cuda_256
        if infeatures % 256 != 0 or outfeatures % 256 != 0:
            self.autogptq_cuda = autogptq_cuda_64
        if infeatures % 64 != 0 or outfeatures % 64 != 0:
            self.autogptq_cuda_available = False

        self.trainable = trainable

    def post_init(self):
        pass

    def pack(self, linear, scales, zeros, g_idx):
        W = linear.weight.data.clone()
        if isinstance(linear, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(linear, transformers.pytorch_utils.Conv1D):
            W = W.t()

        scales = scales.t().contiguous()
        zeros = zeros.t().contiguous()
        scale_zeros = zeros * scales
        self.scales = scales.clone().to(dtype=linear.weight.dtype)
        self.qzeros = zeros.clone().to(dtype=linear.weight.dtype)
        if linear.bias is not None:
            self.bias = linear.bias.clone().to(dtype=linear.weight.dtype)

        intweight = []
        for idx in range(self.infeatures):
            g_idx = idx // self.group_size
            intweight.append(torch.round((W[:, idx] + scale_zeros[g_idx]) / self.scales[g_idx]).to(torch.int)[:, None])
        intweight = torch.cat(intweight, dim=1)
        intweight = intweight.t().contiguous()
        intweight = intweight.numpy().astype(np.uint32)

        i = 0
        row = 0
        qweight = np.zeros((intweight.shape[0] // 32 * self.bits, intweight.shape[1]), dtype=np.uint32)
        while row < qweight.shape[0]:
            if self.bits in [2, 4, 8]:
                for j in range(i, i + (32 // self.bits)):
                    qweight[row] |= intweight[j] << (self.bits * (j - i))
                i += 32 // self.bits
                row += 1
            elif self.bits == 3:
                for j in range(i, i + 10):
                    qweight[row] |= intweight[j] << (3 * (j - i))
                i += 10
                qweight[row] |= intweight[i] << 30
                row += 1
                qweight[row] |= (intweight[i] >> 2) & 1
                i += 1
                for j in range(i, i + 10):
                    qweight[row] |= intweight[j] << (3 * (j - i) + 1)
                i += 10
                qweight[row] |= intweight[i] << 31
                row += 1
                qweight[row] |= (intweight[i] >> 1) & 0x3
                i += 1
                for j in range(i, i + 10):
                    qweight[row] |= intweight[j] << (3 * (j - i) + 2)
                i += 10
                row += 1
            else:
                raise NotImplementedError("Only 2,3,4,8 bits are supported.")

        qweight = qweight.astype(np.int32)
        self.qweight = torch.from_numpy(qweight)

    def forward(self, x):
        x_dtype = x.dtype
        origin_shape = x.shape
        out_shape = x.shape[:-1] + (self.outfeatures,)
        x = x.reshape(-1, x.shape[-1])
        if self.wf.device != self.qzeros.device:
            self.wf = self.wf.to(self.qzeros.device)

        if self.bits in [2, 4, 8]:
            zeros = self.qzeros
            zeros = zeros.reshape(-1, 1, zeros.shape[-1])

            scales = self.scales
            scales = scales.reshape(-1, 1, scales.shape[-1])

            weight = torch.bitwise_right_shift(
                torch.unsqueeze(self.qweight, 1).expand(-1, 32 // self.bits, -1), self.wf.unsqueeze(-1)
            ).to(torch.int16 if self.bits == 8 else torch.int8)
            weight = torch.bitwise_and(weight, (2**self.bits) - 1)
            weight = weight.reshape(-1, self.group_size, weight.shape[2])

        elif self.bits == 3:
            zeros = self.zeros
            zeros = zeros.reshape(-1, 1, zeros.shape[-1])

            scales = self.scales
            scales = scales.reshape(-1, 1, scales.shape[-1])

            weight = self.qweight.reshape(self.qweight.shape[0] // 3, 3, 1, self.qweight.shape[1]).expand(
                -1, -1, 12, -1
            )
            weight = (weight >> self.wf.unsqueeze(-1)) & 0x7
            weight[:, 0, 10] = (weight[:, 0, 10] & 0x3) | ((weight[:, 1, 0] << 2) & 0x4)
            weight[:, 1, 11] = (weight[:, 1, 11] & 0x1) | ((weight[:, 2, 0] << 1) & 0x6)
            weight = weight & 0x7
            weight = torch.cat([weight[:, 0, :11], weight[:, 1, 1:12], weight[:, 2, 1:11]], dim=1)
            weight = weight.reshape(-1, self.group_size, weight.shape[2])
        else:
            raise NotImplementedError("Only 2,3,4,8 bits are supported.")

        quant_type = self.quant_type
        # quant_type = 2
        if quant_type == 0:
            weight = scales * (weight - zeros)  # type 0: original
        elif quant_type == 1:
            qweight = (weight - zeros).to(torch.int16)
            weight = scales * qweight  # type 1: 9bit
            qweight = None
        elif quant_type == 2:
            qweight = (weight - zeros).to(torch.float16)
            shape = qweight.shape
            qweight = qweight.reshape(-1, 64, self.outfeatures)
            extra_scales = torch.ones(qweight.shape, dtype=torch.float16, device=qweight.device)
            min_, _ = qweight.min(dim=1, keepdim=True)
            max_, _ = qweight.max(dim=1, keepdim=True)
            if self.bits == 8:  # 8bit
                extra_scales[min_.expand(-1, 64, -1) < -128] = 0.5
                extra_scales[max_.expand(-1, 64, -1) >= 128] = 0.5
            elif self.bits == 2:  # 4bit
                extra_scales = torch.floor(128 / qweight.abs().max(dim=1, keepdim=True)[0])
                extra_scales = extra_scales.expand(-1, 64, -1).clone()
                # extra_scales[min_.expand(-1, 64, -1) < -8] = 0.5
                # extra_scales[max_.expand(-1, 64, -1) >= 8] = 0.5

            qweight.mul_(extra_scales)
            qweight = qweight.to(torch.int16).reshape(shape)
            if self.bits in [2, 8]:
                # assert qweight.min().item() >= -128 and qweight.max().item() < 128
                qweight.clip_(min=-128, max=127)
            # elif self.bits == 2:
            #     assert qweight.min().item() >= -8 and qweight.max().item() <= 8
            extra_scales = extra_scales.reciprocal().reshape(shape).mul_(scales)

            weight = extra_scales * qweight  # type 2: 8bit
            self.quant_weight = qweight
            extra_scales = None
            qweight = None

        weight = weight.reshape(weight.shape[0] * weight.shape[1], weight.shape[2])
        # for t5 fp16 wo layer https://github.com/huggingface/transformers/issues/20287
        if x_dtype == torch.float32:
            weight = weight.to(dtype=torch.float32)
            if self.bias is not None:
                self.bias = self.bias.to(dtype=torch.float32)
        if self.save_weight:
            self.dequant_weight = weight.t()
        out = torch.matmul(x, weight)

        out = out.to(dtype=x_dtype).reshape(
            out_shape
        )  # A cast is needed here as for some reason the vecquant2matmul_faster_old still allocate a float32 output.
        out = out + self.bias if self.bias is not None else out
        if False:
            val = torch.nn.functional.linear(x.reshape(origin_shape), self.dequant_weight.to(x.device), bias=None)
            val = val + self.bias if self.bias is not None else val
            print(val.shape, out.shape)
            print((val - out).abs().sum().item())
            val2 = torch.matmul(x.reshape(origin_shape), self.dequant_weight.t().to(x.device))
            val2 = val2 + self.bias if self.bias is not None else val2
            print((val2 - out).abs().sum().item())
        return out
