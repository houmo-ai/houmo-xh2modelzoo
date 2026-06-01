import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from xhquant.ops.xh import torch_ops_xh_qlinear
from xhquant.utils.registry import DynamicModule

from ..builder import XHLLM_TRACEABLE_MODULES


@XHLLM_TRACEABLE_MODULES.register_module(
    {
        nn.Linear: "nn.Linear",
    }
)
class _DynamicLinear(DynamicModule):
    def _setup(self, cfg=None):
        return self

    def forward(self, input: Tensor) -> Tensor:
        if torch.compiler.is_compiling() and hasattr(self, "quant_weight"):
            return torch_ops_xh_qlinear(input, self.weight, self.quant_weight, self.bias)
        if torch.onnx.is_in_onnx_export() and hasattr(self, "quant_weight"):
            return torch_ops_xh_qlinear(input, self.weight, self.quant_weight, self.bias)
        if input.device != self.weight.device:
            input = input.to(self.weight.device)
        if input.dtype != self.weight.dtype:
            input = input.to(self.weight.dtype)
        return F.linear(input, self.weight, self.bias)
