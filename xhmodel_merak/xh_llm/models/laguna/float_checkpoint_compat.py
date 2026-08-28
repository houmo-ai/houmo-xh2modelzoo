import sys
from contextlib import contextmanager
from typing import Any

import torch
import torch.nn as nn
from transformers.activations import ACT2FN
from transformers.dynamic_module_utils import get_class_from_dynamic_module


class _LagunaSplitExpert(nn.Module):
    def __init__(self, config: Any):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.moe_intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.moe_intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.moe_intermediate_size, config.hidden_size, bias=False)


class LagunaSplitExperts(nn.ModuleList):
    """Load the official per-expert floating-point checkpoint layout."""

    def __init__(self, config: Any):
        super().__init__([_LagunaSplitExpert(config) for _ in range(config.num_experts)])
        self.config = config


@contextmanager
def split_expert_checkpoint_loader(model_dir: str):
    """Temporarily replace the remote fused experts class while loading weights."""

    native_experts_cls = get_class_from_dynamic_module(
        "modeling_laguna.LagunaExperts",
        model_dir,
    )
    remote_module = sys.modules[native_experts_cls.__module__]
    original_cls = remote_module.LagunaExperts
    pretrained_model_cls = remote_module.LagunaPreTrainedModel
    original_init_weights = pretrained_model_cls._init_weights

    def split_compatible_init_weights(self, module):
        # The checkpoint-compatible split shell intentionally has no fused
        # gate_up_proj/down_proj attributes. Its parameters are populated from
        # safetensors, so the remote fused-expert initializer must skip it.
        if isinstance(module, LagunaSplitExperts):
            return None
        return original_init_weights(self, module)

    remote_module.LagunaExperts = LagunaSplitExperts
    pretrained_model_cls._init_weights = split_compatible_init_weights
    try:
        yield original_cls
    finally:
        pretrained_model_cls._init_weights = original_init_weights
        remote_module.LagunaExperts = original_cls


def fuse_split_experts(model: nn.Module, native_experts_cls: type[nn.Module]) -> int:
    """Convert loaded split floating-point expert weights to Laguna's fused layout."""

    converted = 0
    for parent in model.modules():
        experts = getattr(parent, "experts", None)
        if not isinstance(experts, LagunaSplitExperts):
            continue

        config = experts.config
        fused = native_experts_cls.__new__(native_experts_cls)
        nn.Module.__init__(fused)
        fused.config = config
        fused.num_experts = int(config.num_experts)
        fused.hidden_dim = int(config.hidden_size)
        fused.intermediate_dim = int(config.moe_intermediate_size)
        fused.act_fn = ACT2FN[config.hidden_act]
        fused.gate_up_proj = nn.Parameter(
            torch.stack(
                [torch.cat((expert.gate_proj.weight, expert.up_proj.weight), dim=0) for expert in experts],
                dim=0,
            )
        )
        fused.down_proj = nn.Parameter(torch.stack([expert.down_proj.weight for expert in experts], dim=0))
        if all(
            hasattr(proj, "quant_weight")
            for expert in experts
            for proj in (expert.gate_proj, expert.up_proj, expert.down_proj)
        ):
            fused.register_buffer(
                "gate_up_proj_quant_weight",
                torch.stack(
                    [
                        torch.cat((expert.gate_proj.quant_weight, expert.up_proj.quant_weight), dim=0)
                        for expert in experts
                    ],
                    dim=0,
                ),
                persistent=False,
            )
            fused.register_buffer(
                "down_proj_quant_weight",
                torch.stack([expert.down_proj.quant_weight for expert in experts], dim=0),
                persistent=False,
            )
        gate = getattr(parent, "gate", None)
        if gate is None or not hasattr(gate, "e_score_correction_bias"):
            raise AttributeError("Laguna sparse MoE block is missing gate.e_score_correction_bias")
        parent.experts = fused
        converted += 1
    return converted
