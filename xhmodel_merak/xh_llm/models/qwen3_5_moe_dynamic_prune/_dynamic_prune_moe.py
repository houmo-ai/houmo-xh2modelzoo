"""Merak trace wrapper for Qwen3.5-MoE dynamic pruning."""

from __future__ import annotations

import torch
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeSparseMoeBlock

from ...register import XHLLM_TRACEABLE_MODULES
from ..qwen3_5_moe._moe_model import _Qwen3_5MoeSparseMoeBlock


class Qwen35MoeDynamicPruneBlock(_Qwen3_5MoeSparseMoeBlock):
    """Reuse canonical weight packing and add the threshold input."""

    def _setup(self, cfg=None):
        super()._setup(cfg)
        threshold = 0.0
        if cfg is not None:
            threshold = float(cfg.get("dynamic_prune_threshold", 0.0))
        expert_device = self.moeblock.expert_gate_proj_weight.device
        self.moeblock.prune_threshold = threshold
        expert_count = self.moeblock.expert_gate_proj_weight.shape[0]
        self.moeblock.s_scalar = torch.ones(
            expert_count,
            dtype=torch.float32,
            device=expert_device,
        )
        return self

    def forward(self, hidden_states):
        router_logits = self.gate(hidden_states)
        routing_weights = torch.softmax(router_logits, dim=-1)
        moe_out = self.moeblock(
            hidden_states,
            routing_weights,
        )
        shared_out = self.shared_expert(hidden_states)
        shared_out = torch.sigmoid(self.shared_expert_gate(hidden_states)) * shared_out
        return moe_out + shared_out, router_logits


def register_dynamic_prune_block() -> None:
    """Replace the regular Qwen3.5-MoE SparseMoeBlock trace mapping."""

    current = XHLLM_TRACEABLE_MODULES._registry.get(Qwen3_5MoeSparseMoeBlock)
    if current is Qwen35MoeDynamicPruneBlock:
        return
    if Qwen3_5MoeSparseMoeBlock in XHLLM_TRACEABLE_MODULES._registry:
        del XHLLM_TRACEABLE_MODULES._registry[Qwen3_5MoeSparseMoeBlock]
    if Qwen3_5MoeSparseMoeBlock in XHLLM_TRACEABLE_MODULES._key_registry:
        del XHLLM_TRACEABLE_MODULES._key_registry[Qwen3_5MoeSparseMoeBlock]
    if Qwen3_5MoeSparseMoeBlock in XHLLM_TRACEABLE_MODULES._dynamic_classes:
        del XHLLM_TRACEABLE_MODULES._dynamic_classes[Qwen3_5MoeSparseMoeBlock]
    XHLLM_TRACEABLE_MODULES.register_module(
        {Qwen3_5MoeSparseMoeBlock: "Qwen3_5MoeSparseMoeBlock"},
        Qwen35MoeDynamicPruneBlock,
    )
