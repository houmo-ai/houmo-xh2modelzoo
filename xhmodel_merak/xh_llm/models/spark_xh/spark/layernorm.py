from typing import Optional, Union

import torch
import torch.nn as nn
from vllm.model_executor.custom_op import CustomOp


@CustomOp.register("fp32_rms_norm")
class FP32RMSNorm(CustomOp):
    """Root mean square normalization.

    Computes x -> w * x / sqrt(E[x^2] + eps) where w is the learned weight.
    Refer to https://arxiv.org/abs/1910.07467
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        var_hidden_size: Optional[int] = None,
        has_weight: bool = True,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.variance_epsilon = eps
        self.variance_size_override = None if var_hidden_size == hidden_size else var_hidden_size
        self.has_weight = has_weight
        if dtype is not None:
            self.weight = torch.ones(hidden_size, dtype=dtype)
        else:
            self.weight = torch.ones(hidden_size)
        if self.has_weight:
            self.weight = nn.Parameter(self.weight)

    def forward_native(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """PyTorch-native implementation equivalent to forward()."""
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        if residual is not None:
            x = x + residual.to(torch.float32)
            residual = x.to(orig_dtype)

        hidden_size = x.shape[-1]
        if hidden_size != self.hidden_size:
            raise ValueError(f"Expected hidden_size to be {self.hidden_size}, but found: {hidden_size}")

        if self.variance_size_override is None:
            x_var = x
        else:
            if hidden_size < self.variance_size_override:
                raise ValueError(
                    f"Expected hidden_size to be at least {self.variance_size_override}, but found: {hidden_size}"
                )

            x_var = x[:, :, : self.variance_size_override]

        variance = x_var.pow(2).mean(dim=-1, keepdim=True)

        x = x * torch.rsqrt(variance + self.variance_epsilon)
        if self.has_weight:
            x = x * self.weight
        x = x.to(orig_dtype)
        if residual is None:
            return x
        else:
            return x, residual

    def forward_cuda(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if self.variance_size_override is not None:
            return self.forward_native(x, residual)

        add_residual = residual is not None
        from vllm.model_executor.layers.layernorm import rms_norm

        orig_dtype = x.dtype
        x = x.to(torch.float32)

        if add_residual:
            from vllm.model_executor.layers.layernorm import fused_add_rms_norm

            residual = residual.to(torch.float32)
            x, residual = fused_add_rms_norm(x, residual, self.weight.data, self.variance_epsilon)
            residual = residual.to(orig_dtype)
        else:
            from vllm.model_executor.layers.layernorm import rms_norm

            x = rms_norm(x, self.weight.data, self.variance_epsilon)

        x = x.to(orig_dtype)
        if residual is None:
            return x
        else:
            return x, residual

    def extra_repr(self) -> str:
        s = f"hidden_size={self.weight.data.size(0)}"
        s += f", eps={self.variance_epsilon}"
        return s
