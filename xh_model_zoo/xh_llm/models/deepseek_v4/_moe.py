# ================================================================== #
#  File: _moe.py                                                      #
#  Description:                                                       #
#    DeepSeek-V4 SparseMoeBlock DynamicModule wrapper.               #
#                                                                     #
#    Handles both Hash-MoE and standard top-K MoE routing.           #
#    Packs 256 experts into xhquant MoeBlock format.                 #
#    Includes shared expert with SwiGLU activation.                  #
# ================================================================== #

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from xhquant.nn.modules.moeblock import MoeBlock
from xhquant.utils.registry import DynamicModule

from ..builder import XHLLM_TRACEABLE_MODULES


try:
    from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
        DeepseekV4SparseMoeBlock,
    )
except ImportError:
    DeepseekV4SparseMoeBlock = None


# NOTE: 死代码，待删除。qwen3_next/qwen3_5_moe 用了这个，deepseek_v4 没用。
# def _get_activation_name(act_fn) -> str:
#     if hasattr(act_fn, "_get_name"):
#         return act_fn._get_name().lower()
#     return act_fn.__class__.__name__.lower()


# ================================================================== #
#  SparseMoeBlock: 256 experts + shared expert                       #
# ================================================================== #


if DeepseekV4SparseMoeBlock is not None:
    _moe_registry = {DeepseekV4SparseMoeBlock: "DeepseekV4SparseMoeBlock"}
else:
    _moe_registry = {}


@XHLLM_TRACEABLE_MODULES.register_module(_moe_registry)
class _DeepseekV4SparseMoeBlock(DynamicModule):
    """Wrap DeepSeek-V4 SparseMoeBlock for FX tracing.

    Supports both hash routing and top-K routing.
    Packs 256 routed experts into MoeBlock, keeps shared expert separate.
    """

    def forward(self, hidden_states: torch.Tensor, input_ids=None):
        batch, seq_len, hidden_dim = hidden_states.shape

        # -- Router --
        flat = hidden_states.reshape(-1, hidden_dim)
        logits = self.gate_linear(flat)
        scores = self._score_fn(logits)

        if self.is_hash:
            # Hash routing: tid2eid lookup gives fixed expert IDs per token.
            # Pass full 256-dim scores + expert IDs (0-255) to MoeBlock.
            # MoeBlock(topk_outside=True) gathers internally, then
            # normalizes (normalize_routing_weights=True).
            indices = self.tid2eid[input_ids.reshape(-1)].to(torch.int64)
            routing_weights = scores.view(batch, seq_len, -1)
        else:
            # Top-K routing: external TopK picks expert IDs.
            # Pass full 256-dim scores (with bias) + expert IDs.
            indices = torch.topk(
                scores + self.e_score_correction_bias,
                self.top_k,
                dim=-1,
                sorted=False,
            )[1]
            routing_weights = scores.view(batch, seq_len, -1)

        selected_experts = indices.view(batch, seq_len, -1)
        moe_out = self.moeblock(
            hidden_states,
            routing_weights,
            selected_experts=selected_experts,
            fast_mode=False,
        )
        moe_out = moe_out * self.routed_scaling_factor

        # -- Shared expert --
        shared_out = self._shared_expert_forward(hidden_states)

        return moe_out + shared_out

    def _shared_expert_forward(self, hidden_states):
        """SwiGLU shared expert: down(silu(gate) * up)"""
        gate = self.shared_gate_proj(hidden_states)
        up = self.shared_up_proj(hidden_states)
        return self.shared_down_proj(F.silu(gate) * up)

    def _score_fn(self, logits):
        """sqrt(softplus(x)) activation for router."""
        return torch.sqrt(F.softplus(logits).to(logits.dtype))

    def _setup(self, cfg: Optional[Dict] = None):
        # Config lives on experts submodule
        config = getattr(self, "config", None)
        if config is None:
            config = getattr(self.experts, "config", None)
        if config is None:
            config = getattr(self.gate, "config", None)
        if config is None:
            raise ValueError("Cannot find config for DeepseekV4SparseMoeBlock")

        # Detect hash routing from gate type (is_hash is not a stored attribute)
        try:
            from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4HashRouter

            self.is_hash = isinstance(self.gate, DeepseekV4HashRouter)
        except ImportError:
            self.is_hash = hasattr(self.gate, "tid2eid")
        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.5)
        top_k = config.num_experts_per_tok
        self.top_k = top_k
        self.device = self.gate.weight.device

        # -- Extract gate (router) weights --
        if self.is_hash:
            gate_weight = self.gate.weight.data.clone()
            tid2eid = self.gate.tid2eid.data.clone()
            num_experts_dim, hidden_dim = gate_weight.shape
            self.gate_linear = nn.Linear(hidden_dim, num_experts_dim, bias=False)
            self.gate_linear.weight.data.copy_(gate_weight)
            self.register_buffer("tid2eid", tid2eid, persistent=True)
        else:
            gate_weight = self.gate.weight.data.clone()
            num_experts_dim, hidden_dim = gate_weight.shape
            self.gate_linear = nn.Linear(hidden_dim, num_experts_dim, bias=False)
            self.gate_linear.weight.data.copy_(gate_weight)
            e_score_bias = getattr(self.gate, "e_score_correction_bias", None)
            if e_score_bias is not None:
                self.register_buffer(
                    "e_score_correction_bias",
                    e_score_bias.data.clone(),
                    persistent=True,
                )
            else:
                self.register_buffer(
                    "e_score_correction_bias",
                    torch.zeros(num_experts_dim, device=self.device),
                    persistent=True,
                )
        del self.gate

        # -- Pack experts into MoeBlock --
        experts = self.experts
        intermediate_dim = config.moe_intermediate_size

        self.moeblock = MoeBlock("silu", top_k, topk_outside=True)
        # MoeBlock normalizes after internal gather (both hash & top-K).
        self.moeblock.normalize_routing_weights = True

        gate_up = experts.gate_up_proj.data.to(self.device)
        gate_proj_weight = gate_up[:, :intermediate_dim, :].contiguous()
        up_proj_weight = gate_up[:, intermediate_dim:, :].contiguous()
        down_proj_weight = experts.down_proj.data.to(self.device).contiguous()

        self.moeblock.expert_gate_proj_weight = nn.Parameter(gate_proj_weight)
        self.moeblock.expert_gate_proj_bias = None
        self.moeblock.expert_up_proj_weight = nn.Parameter(up_proj_weight)
        self.moeblock.expert_up_proj_bias = None
        self.moeblock.expert_down_proj_weight = nn.Parameter(down_proj_weight)
        self.moeblock.expert_down_proj_bias = None

        del self.experts

        # -- Shared expert: extract linear weights --
        shared = self.shared_experts
        self.shared_gate_proj = nn.Linear(
            config.hidden_size,
            config.moe_intermediate_size,
            bias=False,
            device=self.device,
            dtype=shared.gate_proj.weight.dtype,
        )
        self.shared_gate_proj.weight.data.copy_(shared.gate_proj.weight.data)

        self.shared_up_proj = nn.Linear(
            config.hidden_size,
            config.moe_intermediate_size,
            bias=False,
            device=self.device,
            dtype=shared.up_proj.weight.dtype,
        )
        self.shared_up_proj.weight.data.copy_(shared.up_proj.weight.data)

        self.shared_down_proj = nn.Linear(
            config.moe_intermediate_size,
            config.hidden_size,
            bias=False,
            device=self.device,
            dtype=shared.down_proj.weight.dtype,
        )
        self.shared_down_proj.weight.data.copy_(shared.down_proj.weight.data)

        del self.shared_experts
        torch.cuda.empty_cache()

        return self
