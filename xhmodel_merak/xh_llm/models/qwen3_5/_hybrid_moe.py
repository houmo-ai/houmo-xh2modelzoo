# -*- coding: utf-8 -*-
"""Registry-free canonical sparse-MoE adapter for hybrid Qwen models."""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from xhquant.nn.modules.moeblock import MoeBlock


def _get_activation_name(act_fn) -> str:
    if hasattr(act_fn, "_get_name"):
        return act_fn._get_name().lower()
    return act_fn.__class__.__name__.lower()


def _move_parameter_to_meta(module: nn.Module, parameter_name: str) -> None:
    parameter = getattr(module, parameter_name)
    meta_parameter = nn.Parameter(
        torch.empty_like(parameter.data, device="meta"),
        requires_grad=parameter.requires_grad,
    )
    setattr(module, parameter_name, meta_parameter)


def _move_buffer_to_meta(module: nn.Module, buffer_name: str) -> None:
    buffer = getattr(module, buffer_name)
    module._buffers[buffer_name] = torch.empty_like(buffer, device="meta")


def _init_single_moe_linear_storage(moeblock: MoeBlock, experts, linear, linear_name: str, device: torch.device):
    weight = linear.weight

    setattr(
        moeblock,
        f"expert_{linear_name}_weight",
        nn.Parameter(
            torch.empty(
                len(experts),
                weight.shape[0],
                weight.shape[1],
                device=device,
                dtype=weight.dtype,
            )
        ),
    )

    quant_weight = getattr(linear, "quant_weight", None)
    if quant_weight is not None:
        setattr(
            moeblock,
            f"expert_{linear_name}_quant_weight",
            torch.empty(
                len(experts),
                quant_weight.shape[0],
                quant_weight.shape[1],
                device=device,
                dtype=quant_weight.dtype,
            ),
        )

    bias = linear.bias
    if bias is not None:
        setattr(
            moeblock,
            f"expert_{linear_name}_bias",
            nn.Parameter(
                torch.empty(
                    len(experts),
                    bias.shape[0],
                    device=device,
                    dtype=bias.dtype,
                )
            ),
        )
    else:
        setattr(moeblock, f"expert_{linear_name}_bias", None)


def _pack_defused_expert_linear_to_moeblock(moeblock: MoeBlock, experts, linear_name: str, device: torch.device):
    _init_single_moe_linear_storage(moeblock, experts, getattr(experts[0], linear_name), linear_name, device)

    with torch.no_grad():
        for expert_idx, expert in enumerate(experts):
            linear = getattr(expert, linear_name)
            weight_data = linear.weight.data
            if weight_data.device != device:
                weight_data = weight_data.to(device)
            getattr(moeblock, f"expert_{linear_name}_weight")[expert_idx].copy_(weight_data)

            quant_weight = getattr(linear, "quant_weight", None)
            if quant_weight is not None and hasattr(moeblock, f"expert_{linear_name}_quant_weight"):
                quant_weight_data = quant_weight.data
                if quant_weight_data.device != device:
                    quant_weight_data = quant_weight_data.to(device)
                getattr(moeblock, f"expert_{linear_name}_quant_weight")[expert_idx].copy_(quant_weight_data)

            bias = linear.bias
            if bias is not None and getattr(moeblock, f"expert_{linear_name}_bias") is not None:
                bias_data = bias.data
                if bias_data.device != device:
                    bias_data = bias_data.to(device)
                getattr(moeblock, f"expert_{linear_name}_bias")[expert_idx].copy_(bias_data)

            _move_parameter_to_meta(linear, "weight")
            if bias is not None:
                _move_parameter_to_meta(linear, "bias")
            if quant_weight is not None:
                _move_buffer_to_meta(linear, "quant_weight")


class HybridSparseMoeMixin:
    """Wrap Qwen3.5-MoE SparseMoeBlock.

    The HF Qwen3_5MoeExperts uses packed weight tensors:
      - gate_up_proj: (num_experts, 2*intermediate_dim, hidden_dim)
      - down_proj: (num_experts, hidden_dim, intermediate_dim)

    We split gate_up into gate + up and pack into standard MoeBlock format.
    Also handles shared_expert + shared_expert_gate.
    """

    # def forward(self, hidden_states):
    #     batch_size, sequence_length, hidden_dim = hidden_states.shape
    #     hidden_states_reshaped = hidden_states.view(-1, hidden_dim)

    #     # Router: softmax + topk done inside MoeBlock
    #     router_logits = self.gate(hidden_states_reshaped)
    #     routing_weights = F.softmax(router_logits, dim=-1)
    #     # MoeBlock expects 3D: [batch, seq, num_experts]
    #     routing_weights = routing_weights.view(batch_size, sequence_length, -1)

    #     # MoE block forward
    #     moe_out = self.moeblock(hidden_states, routing_weights)

    #     # Shared expert
    #     shared_out = self.shared_expert(hidden_states_reshaped)
    #     shared_out = torch.sigmoid(self.shared_expert_gate(hidden_states_reshaped)) * shared_out
    #     shared_out = shared_out.reshape(batch_size, sequence_length, hidden_dim)

    #     return moe_out + shared_out

    def forward(self, hidden_states):
        if hasattr(self, "moeblock") and hasattr(self.moeblock, "expert_gate_proj_weight"):
            hidden_states = hidden_states.to(self.moeblock.expert_gate_proj_weight.dtype)
        router_logits = self.gate(hidden_states)
        routing_weights = F.softmax(router_logits, dim=-1)
        moe_out = self.moeblock(hidden_states, routing_weights)
        shared_out = self.shared_expert(hidden_states)
        shared_out = torch.sigmoid(self.shared_expert_gate(hidden_states)) * shared_out
        return moe_out + shared_out, router_logits

    def _setup(self, cfg: Optional[Dict] = None):
        self.input_seq_len = cfg.input_sequence_length
        self.batch_size = cfg.batch_size
        self.device = self.gate.weight.device

        # Save gate weight as buffer and keep TopKRouter info
        gate_weight_data = self.gate.weight.data.clone()
        top_k = self.gate.top_k
        num_experts_dim, hidden_dim = gate_weight_data.shape

        # Replace TopKRouter with standard nn.Linear so it traces as call_module
        self.gate = nn.Linear(
            hidden_dim,
            num_experts_dim,
            bias=False,
            device=self.device,
            dtype=gate_weight_data.dtype,
        )
        self.gate.weight.data.copy_(gate_weight_data)

        # Determine expert dimensions from the packed Experts
        experts = self.experts
        if hasattr(experts, "act_fn"):
            act_fn_name = _get_activation_name(experts.act_fn)
        else:
            act_fn_name = _get_activation_name(experts[0].act_fn)

        # Qwen3_5Moe normalizes routing weights inside gate, so norm_topk_prob=True
        self.moeblock = MoeBlock(act_fn_name, top_k, True)

        if hasattr(experts, "gate_up_proj"):
            # Split packed gate_up_proj -> gate_proj + up_proj
            # gate_up_proj shape: (num_experts, 2*intermediate_dim, hidden_dim)
            intermediate_dim = experts.intermediate_dim
            gate_up = experts.gate_up_proj.data
            num_experts = gate_up.shape[0]
            target_dtype = gate_up.dtype

            self.moeblock.expert_gate_proj_weight = nn.Parameter(
                torch.empty(num_experts, intermediate_dim, hidden_dim, device=self.device, dtype=target_dtype)
            )
            self.moeblock.expert_gate_proj_bias = None
            self.moeblock.expert_gate_proj_weight.data.copy_(gate_up[:, :intermediate_dim, :].to(self.device))

            self.moeblock.expert_up_proj_weight = nn.Parameter(
                torch.empty(num_experts, intermediate_dim, hidden_dim, device=self.device, dtype=target_dtype)
            )
            self.moeblock.expert_up_proj_bias = None
            self.moeblock.expert_up_proj_weight.data.copy_(gate_up[:, intermediate_dim:, :].to(self.device))

            # Release the packed gate_up tensor before wiring the remaining
            # expert weights so the wrap peak does not hold both layouts longer
            # than necessary.
            _move_parameter_to_meta(experts, "gate_up_proj")
            del gate_up

            down_proj = experts.down_proj
            if down_proj.device == self.device and down_proj.is_contiguous():
                self.moeblock.expert_down_proj_weight = down_proj
            else:
                self.moeblock.expert_down_proj_weight = nn.Parameter(down_proj.data.to(self.device).contiguous())
            _move_parameter_to_meta(experts, "down_proj")

            self.moeblock.expert_down_proj_bias = None
            del down_proj
            # Release original packed experts to free memory
            del self.experts
        elif (
            len(experts) > 0
            and hasattr(experts[0], "gate_proj")
            and hasattr(experts[0], "up_proj")
            and hasattr(experts[0], "down_proj")
        ):
            for linear_name in ("gate_proj", "up_proj", "down_proj"):
                _pack_defused_expert_linear_to_moeblock(self.moeblock, experts, linear_name, self.device)

            # Release expert modules after packing to reduce memory.
            self.experts = nn.ModuleList()
        else:
            raise RuntimeError(f"Unsupported Qwen3.5-MoE experts structure: {type(experts)}")
        torch.cuda.empty_cache()
        return self
