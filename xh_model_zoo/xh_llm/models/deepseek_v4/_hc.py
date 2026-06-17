# ================================================================== #
#  File: _hc.py                                                       #
#  Description:                                                       #
#    DeepSeek-V4 HyperConnection + HyperHead DynamicModule wrappers  #
#                                                                     #
#    Manifold-Constrained Hyper-Connections (mHC) maintain hc_mult=4  #
#    parallel residual streams throughout the network. Each HC module #
#    computes pre (stream collapse), post (sublayer output), and comb #
#    (stream mixer) via Sinkhorn doubly-stochastic projection.        #
# ================================================================== #

from typing import Dict, Optional

import torch
import torch.nn as nn

from xhquant.utils.registry import DynamicModule

from ..builder import XHLLM_TRACEABLE_MODULES


try:
    from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
        DeepseekV4HyperConnection,
        DeepseekV4HyperHead,
    )
except ImportError:
    DeepseekV4HyperConnection = None
    DeepseekV4HyperHead = None


# ================================================================== #
#  HyperConnection: mHC residual stream mixing                       #
# ================================================================== #


if DeepseekV4HyperConnection is not None:
    _hc_registry = {DeepseekV4HyperConnection: "DeepseekV4HyperConnection"}
else:
    _hc_registry = {}


@XHLLM_TRACEABLE_MODULES.register_module(_hc_registry)
class _DeepseekV4HyperConnection(DynamicModule):
    """Wrap HyperConnection for FX tracing.

    Forward produces (post, comb, collapsed) from hc_mult parallel streams.
    Sinkhorn projection is unrolled for tracing compatibility.
    """

    def forward(self, hidden_streams: torch.Tensor):
        """hidden_streams: [B, S, hc_mult, D]"""
        hc = self.hc_mult
        eps = self.hc_eps

        flat = self.input_norm(hidden_streams.flatten(start_dim=2).float())
        mix_logits = self.fn_linear(flat)
        pre_w, post_w, comb_w = mix_logits.split([hc, hc, hc * hc], dim=-1)

        pre_b, post_b, comb_b = self.base.split([hc, hc, hc * hc])
        pre_scale = self.scale[0]
        post_scale = self.scale[1]
        comb_scale = self.scale[2]

        # -- pre: stream collapse weights --
        pre = torch.sigmoid(pre_w * pre_scale + pre_b) + eps

        # -- post: sublayer output placement --
        post = 2 * torch.sigmoid(post_w * post_scale + post_b)

        # -- comb: Sinkhorn doubly-stochastic projection --
        comb_logits = comb_w.view(comb_w.shape[0], comb_w.shape[1], hc, hc) * comb_scale + comb_b.view(hc, hc)
        comb = torch.softmax(comb_logits, dim=-1) + eps
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
        for _ in range(self.hc_sinkhorn_iters - 1):
            comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
            comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)

        # -- collapse streams --
        collapsed = (pre.unsqueeze(-1) * hidden_streams).sum(dim=2)
        collapsed = collapsed.to(hidden_streams.dtype)

        return post, comb, collapsed

    def _setup(self, cfg: Optional[Dict] = None):
        fn_w = self.fn.data
        self.fn_linear = nn.Linear(fn_w.shape[1], fn_w.shape[0], bias=False)
        self.fn_linear.weight.data.copy_(fn_w)
        del self.fn
        return self


# ================================================================== #
#  HyperHead: final HC stream collapse                               #
# ================================================================== #


if DeepseekV4HyperHead is not None:
    _head_registry = {DeepseekV4HyperHead: "DeepseekV4HyperHead"}
else:
    _head_registry = {}


@XHLLM_TRACEABLE_MODULES.register_module(_head_registry)
class _DeepseekV4HyperHead(DynamicModule):
    """Wrap HyperHead for FX tracing.

    Collapses hc_mult streams to a single hidden_states tensor.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, S, hc_mult, D] -> [B, S, D]"""
        flat = self.input_norm(x.flatten(2).float())
        mixes = self.hc_linear(flat)
        pre = torch.sigmoid(mixes * self.hc_scale.float() + self.hc_base.float()) + self.eps
        out = (pre.unsqueeze(-1) * x).sum(dim=2)
        return out.to(x.dtype)

    def _setup(self, cfg: Optional[Dict] = None):
        w = self.hc_fn.data
        self.hc_linear = nn.Linear(w.shape[1], w.shape[0], bias=False)
        self.hc_linear.weight.data.copy_(w)
        del self.hc_fn
        return self
