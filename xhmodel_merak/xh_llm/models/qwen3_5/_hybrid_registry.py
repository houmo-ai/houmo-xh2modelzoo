# -*- coding: utf-8 -*-
"""Unique registrations for optional modules shared by hybrid Qwen families."""

from xhquant.utils.registry import DynamicModule

from ...register import XHLLM_TRACEABLE_MODULES
from ._hybrid_gated_delta_net import HybridRMSNormGatedMixin


try:
    from fla.modules import FusedRMSNormGated
except ImportError:
    FusedRMSNormGated = None


if FusedRMSNormGated is not None:

    @XHLLM_TRACEABLE_MODULES.register_module({FusedRMSNormGated: "FusedRMSNormGated_HybridQwen"})
    class _HybridFusedRMSNormGated(HybridRMSNormGatedMixin, DynamicModule):
        """One family-neutral wrapper for FLA's shared fused norm class."""

        pass


def ensure_hybrid_fused_rms_norm_registered() -> None:
    """Import-time registration barrier used by each model-family module."""
