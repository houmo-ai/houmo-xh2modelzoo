"""Static mHC pre/post mappings for DeepSeek-V4 Flash.

The trained contract uses twenty Sinkhorn iterations on a 4x4 matrix.  The
deployment graph preserves that exact order through one dedicated XH2 op.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from xhquant import nn as xhnn

from ._trace import is_fx_proxy
from .norm import build_xh_rms_norm, rms_norm_from_hf


def _split_mhc_coefficients(
    mixes: Tensor,
    hc_scale: Tensor,
    hc_base: Tensor,
    *,
    hc_mult: int = 4,
    eps: float = 1e-6,
) -> tuple[Tensor, Tensor, Tensor]:
    hc_mult = int(hc_mult)
    if hc_mult <= 0 or float(eps) <= 0:
        raise ValueError("hc_mult and eps must be positive")
    expected = (2 + hc_mult) * hc_mult
    if not is_fx_proxy(mixes):
        if mixes.shape[-1] != expected:
            raise ValueError(f"mixes must end in {expected} values")
        if tuple(hc_scale.shape) != (3,) or tuple(hc_base.shape) != (expected,):
            raise ValueError("hc_scale/hc_base shapes do not match the mHC layout")
        deployment_tensors = {
            "mixes": mixes,
            "hc_scale": hc_scale,
            "hc_base": hc_base,
        }
        wrong_dtype = {name: value.dtype for name, value in deployment_tensors.items() if value.dtype != torch.float16}
        if wrong_dtype:
            raise TypeError(f"DeepSeek-V4 deployment mHC tensors must all be torch.float16; got {wrong_dtype}")

    work = mixes
    scale = hc_scale
    base = hc_base
    epsilon = eps

    pre = torch.sigmoid(work[..., :hc_mult] * scale[0] + base[:hc_mult]) + epsilon
    post = 2 * torch.sigmoid(work[..., hc_mult : 2 * hc_mult] * scale[1] + base[hc_mult : 2 * hc_mult])
    combination = work[..., 2 * hc_mult :] * scale[2] + base[2 * hc_mult :]
    if not is_fx_proxy(combination) and combination.ndim == 2:
        combination = combination.reshape(combination.shape[0], hc_mult, hc_mult)
    else:
        combination = combination.reshape(
            combination.shape[0],
            combination.shape[1],
            hc_mult,
            hc_mult,
        )
    return pre, post, combination


def mhc_split_sinkhorn(
    mixes: Tensor,
    hc_scale: Tensor,
    hc_base: Tensor,
    *,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
) -> tuple[Tensor, Tensor, Tensor]:
    """Reference implementation of the checkpoint's FP16 mHC mapping.

    Production graphs use :class:`xhquant.nn.MHCSinkhorn`; this function stays
    as the small eager reference used for numerical tests.
    """

    sinkhorn_iters = int(sinkhorn_iters)
    if sinkhorn_iters <= 0:
        raise ValueError("sinkhorn_iters must be positive")
    pre, post, combination = _split_mhc_coefficients(
        mixes,
        hc_scale,
        hc_base,
        hc_mult=hc_mult,
        eps=eps,
    )

    combination = combination - combination.amax(dim=-1, keepdim=True)
    combination = torch.exp(combination)
    combination = combination / combination.sum(dim=-1, keepdim=True) + eps
    combination = combination / (combination.sum(dim=-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        combination = combination / (combination.sum(dim=-1, keepdim=True) + eps)
        combination = combination / (combination.sum(dim=-2, keepdim=True) + eps)
    return pre, post, combination


class MHCPreMapping(nn.Module):
    """Collapse four hyper-connection streams before attention or FFN."""

    def __init__(
        self,
        *,
        hidden_size: int,
        hc_mult: int = 4,
        sinkhorn_iters: int = 20,
        norm_eps: float = 1e-6,
        hc_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.hc_mult = int(hc_mult)
        self.sinkhorn_iters = int(sinkhorn_iters)
        self.norm_eps = float(norm_eps)
        self.hc_eps = float(hc_eps)
        self.input_norm = build_xh_rms_norm(
            int(hidden_size),
            self.norm_eps,
        )
        self.sinkhorn = xhnn.MHCSinkhorn(self.sinkhorn_iters, self.hc_eps)

    def from_mixes(
        self,
        hidden_states: Tensor,
        mixes: Tensor,
        hc_scale: Tensor,
        hc_base: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if not is_fx_proxy(hidden_states) and (hidden_states.ndim != 4 or hidden_states.shape[-2] != self.hc_mult):
            raise ValueError(f"hidden_states must be [B,S,{self.hc_mult},D]")
        pre, post, logits = _split_mhc_coefficients(
            mixes,
            hc_scale,
            hc_base,
            hc_mult=self.hc_mult,
            eps=self.hc_eps,
        )
        combination = self.sinkhorn(logits)
        collapsed = torch.sum(
            pre.unsqueeze(-1) * hidden_states,
            dim=-2,
        )
        return collapsed, post, combination

    def forward(
        self,
        hidden_states: Tensor,
        hc_fn: Tensor,
        hc_scale: Tensor,
        hc_base: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if not is_fx_proxy(hidden_states) and (hidden_states.ndim != 4 or hidden_states.shape[-2] != self.hc_mult):
            raise ValueError(f"hidden_states must be [B,S,{self.hc_mult},D]")
        flattened = hidden_states.flatten(2)
        normalized = self.input_norm(flattened)
        mixes = F.linear(normalized, hc_fn)
        return self.from_mixes(
            hidden_states,
            mixes,
            hc_scale,
            hc_base,
        )


class MHCPostMapping(nn.Module):
    """Expand one branch output and mix the four residual streams."""

    def forward(
        self,
        branch_output: Tensor,
        residual: Tensor,
        post: Tensor,
        combination: Tensor,
    ) -> Tensor:
        if not is_fx_proxy(residual) and (
            residual.ndim != 4 or branch_output.shape != residual.shape[:2] + residual.shape[3:]
        ):
            raise ValueError("branch_output/residual shapes do not match mHC")
        output = post.unsqueeze(-1) * branch_output.unsqueeze(-2)
        output = output + torch.sum(
            combination.unsqueeze(-1) * residual.unsqueeze(-2),
            # output[k] = sum_j combination[j, k] * residual[j]
            dim=-3,
        )
        return output


class BoundMHC(nn.Module):
    """Bind one checkpoint HyperConnection's trained tensors to static maps."""

    def __init__(
        self,
        hc_fn: Tensor,
        hc_scale: Tensor,
        hc_base: Tensor,
        *,
        hc_mult: int = 4,
        sinkhorn_iters: int = 20,
        norm_eps: float = 1e-6,
        hc_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.hc_proj = nn.Linear(
            int(hc_fn.shape[1]),
            int(hc_fn.shape[0]),
            bias=False,
            device="meta",
            dtype=hc_fn.dtype,
        )
        self.hc_proj.weight = nn.Parameter(
            hc_fn,
            requires_grad=hc_fn.requires_grad if isinstance(hc_fn, nn.Parameter) else True,
        )
        self.hc_scale = nn.Parameter(hc_scale) if not isinstance(hc_scale, nn.Parameter) else hc_scale
        self.hc_base = nn.Parameter(hc_base) if not isinstance(hc_base, nn.Parameter) else hc_base
        self.pre = MHCPreMapping(
            hidden_size=int(hc_fn.shape[1]),
            hc_mult=hc_mult,
            sinkhorn_iters=sinkhorn_iters,
            norm_eps=norm_eps,
            hc_eps=hc_eps,
        )
        self.post = MHCPostMapping()

    @classmethod
    def from_hf(
        cls,
        hyper_connection: nn.Module,
    ) -> "BoundMHC":
        return cls(
            hyper_connection.fn,
            hyper_connection.scale,
            hyper_connection.base,
            hc_mult=hyper_connection.hc_mult,
            sinkhorn_iters=hyper_connection.hc_sinkhorn_iters,
            norm_eps=getattr(
                hyper_connection.input_norm,
                "variance_epsilon",
                getattr(hyper_connection.input_norm, "eps", 1e-6),
            ),
            hc_eps=hyper_connection.hc_eps,
        )

    def collapse(self, hidden_states: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        flattened = hidden_states.flatten(2)
        normalized = self.pre.input_norm(flattened)
        mixes = self.hc_proj(normalized)
        return self.pre.from_mixes(
            hidden_states,
            mixes,
            self.hc_scale,
            self.hc_base,
        )

    def expand(
        self,
        branch_output: Tensor,
        residual: Tensor,
        post: Tensor,
        combination: Tensor,
    ) -> Tensor:
        return self.post(branch_output, residual, post, combination)


class BoundHCHead(nn.Module):
    """Final trained hyper-stream collapse with an explicit Linear node."""

    def __init__(
        self,
        *,
        input_norm: nn.Module,
        hc_fn: Tensor,
        hc_base: Tensor,
        hc_scale: Tensor,
        hc_mult: int,
        eps: float,
    ) -> None:
        super().__init__()
        self.input_norm = input_norm
        self.hc_mult = int(hc_mult)
        self.eps = float(eps)
        self.hc_proj = nn.Linear(
            int(hc_fn.shape[1]),
            int(hc_fn.shape[0]),
            bias=False,
            device="meta",
            dtype=hc_fn.dtype,
        )
        self.hc_proj.weight = nn.Parameter(
            hc_fn,
            requires_grad=hc_fn.requires_grad if isinstance(hc_fn, nn.Parameter) else True,
        )
        self.hc_base = nn.Parameter(hc_base) if not isinstance(hc_base, nn.Parameter) else hc_base
        self.hc_scale = nn.Parameter(hc_scale) if not isinstance(hc_scale, nn.Parameter) else hc_scale

    @classmethod
    def from_hf(cls, head: nn.Module) -> "BoundHCHead":
        return cls(
            input_norm=rms_norm_from_hf(
                head.input_norm,
                hidden_size=int(head.hc_fn.shape[1]),
            ),
            hc_fn=head.hc_fn,
            hc_base=head.hc_base,
            hc_scale=head.hc_scale,
            hc_mult=head.hc_mult,
            eps=head.eps,
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        if not is_fx_proxy(hidden_states) and (hidden_states.ndim != 4 or hidden_states.shape[-2] != self.hc_mult):
            raise ValueError(f"hidden_states must be [B,S,{self.hc_mult},D]")
        flat = self.input_norm(hidden_states.flatten(2))
        mixes = self.hc_proj(flat)
        pre = torch.sigmoid(mixes * self.hc_scale + self.hc_base) + self.eps
        return (pre.unsqueeze(-1) * hidden_states).sum(dim=2)


__all__ = [
    "BoundHCHead",
    "BoundMHC",
    "MHCPostMapping",
    "MHCPreMapping",
    "mhc_split_sinkhorn",
]
