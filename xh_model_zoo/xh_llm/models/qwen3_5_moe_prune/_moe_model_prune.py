from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeSparseMoeBlock
from xhquant.nn.modules.moeblock_prune import PruningRouter, MoeBlockPrune
from xhquant.utils.registry import DynamicModule

from ..builder import XHLLM_TRACEABLE_MODULES
from ..qwen3_5_moe._moe_model import (  # noqa: F401
    _Qwen3_5MoeAttention,
    _Qwen3_5MoeDecoderLayer,
    _Qwen3_5MoeForCausalLM,
    _Qwen3_5MoeForConditionalGeneration,
    _Qwen3_5MoeGatedDeltaNet,
    _Qwen3_5MoeRMSNorm,
    _Qwen3_5MoeRMSNormGated,
    _Qwen3_5MoeTextModel,
    _Qwen3_5MoeTextRotaryEmbedding,
    _copy_defused_expert_weights_to_moeblock,
    _get_activation_name,
    _init_moe_linear_storage,
)

if Qwen3_5MoeSparseMoeBlock in XHLLM_TRACEABLE_MODULES._registry:
    del XHLLM_TRACEABLE_MODULES._registry[Qwen3_5MoeSparseMoeBlock]
if Qwen3_5MoeSparseMoeBlock in XHLLM_TRACEABLE_MODULES._key_registry:
    del XHLLM_TRACEABLE_MODULES._key_registry[Qwen3_5MoeSparseMoeBlock]
if Qwen3_5MoeSparseMoeBlock in XHLLM_TRACEABLE_MODULES._dynamic_classes:
    del XHLLM_TRACEABLE_MODULES._dynamic_classes[Qwen3_5MoeSparseMoeBlock]


@XHLLM_TRACEABLE_MODULES.register_module({Qwen3_5MoeSparseMoeBlock: "Qwen3_5MoeSparseMoeBlock"})
class _Qwen3_5MoeSparseMoeBlockPrune(DynamicModule):
    def forward(self, hidden_states):
        router_logits = self.gate(hidden_states)
        routing_weights = F.softmax(router_logits, dim=-1)
        routing_weights, selected_experts, k_tensor = self.prune_router(
            routing_weights,
            self.s_scalar,
            self.threshold,
        )
        moe_out = self.moeblock_prune(hidden_states, routing_weights, k_tensor, selected_experts)
        shared_out = self.shared_expert(hidden_states)
        shared_out = torch.sigmoid(self.shared_expert_gate(hidden_states)) * shared_out
        return moe_out + shared_out, router_logits

    def _setup(self, cfg: Optional[Dict] = None):
        self.input_seq_len = cfg.input_sequence_length
        self.batch_size = cfg.batch_size
        self.device = self.gate.weight.device

        gate_weight_data = self.gate.weight.data.clone()
        top_k = self.gate.top_k
        num_experts_dim, hidden_dim = gate_weight_data.shape

        self.gate = nn.Linear(
            hidden_dim,
            num_experts_dim,
            bias=False,
            device=self.device,
            dtype=gate_weight_data.dtype,
        )
        self.gate.weight.data.copy_(gate_weight_data)

        experts = self.experts
        if hasattr(experts, "act_fn"):
            act_fn_name = _get_activation_name(experts.act_fn)
        else:
            act_fn_name = _get_activation_name(experts[0].act_fn)

        self.moeblock_prune = MoeBlockPrune(
            act_fn_name,
            top_k,
            True,
            topk_outside=True,
        )

        if hasattr(experts, "gate_up_proj"):
            intermediate_dim = experts.intermediate_dim
            gate_up = experts.gate_up_proj.data.to(self.device)
            gate_proj_weight = gate_up[:, :intermediate_dim, :].contiguous()
            up_proj_weight = gate_up[:, intermediate_dim:, :].contiguous()
            down_proj_weight = experts.down_proj.data.to(self.device).contiguous()

            self.moeblock_prune.expert_gate_proj_weight = nn.Parameter(gate_proj_weight)
            self.moeblock_prune.expert_gate_proj_bias = None
            self.moeblock_prune.expert_up_proj_weight = nn.Parameter(up_proj_weight)
            self.moeblock_prune.expert_up_proj_bias = None
            self.moeblock_prune.expert_down_proj_weight = nn.Parameter(down_proj_weight)
            self.moeblock_prune.expert_down_proj_bias = None
            del self.experts
        elif (
            len(experts) > 0
            and hasattr(experts[0], "gate_proj")
            and hasattr(experts[0], "up_proj")
            and hasattr(experts[0], "down_proj")
        ):
            for linear_name in ("gate_proj", "up_proj", "down_proj"):
                _init_moe_linear_storage(self.moeblock_prune, experts, linear_name, self.device)
            _copy_defused_expert_weights_to_moeblock(self.moeblock_prune, experts, self.device)
            self.experts = nn.ModuleList()
        else:
            raise RuntimeError(f"Unsupported Qwen3.5-MoE experts structure: {type(experts)}")

        self.prune_router = PruningRouter(top_k, True)
        self.register_buffer("s_scalar", torch.ones(num_experts_dim, device=self.device))
        self.threshold = 0.0
        torch.cuda.empty_cache()
        return self


def register_wrap_modules(hf_model=None):
    pass
