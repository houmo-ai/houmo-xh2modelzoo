from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeRMSNorm,
    Qwen3OmniMoeRotaryEmbedding,
    Qwen3OmniMoeTalkerDecoderLayer,
    Qwen3OmniMoeTalkerForConditionalGeneration,
    Qwen3OmniMoeTalkerModel,
    Qwen3OmniMoeTalkerTextSparseMoeBlock,
    Qwen3OmniMoeTextRMSNorm,
)

from xhquant.nn.modules.moeblock import MoeBlock
from xhquant.utils.registry import DynamicModule

from ...register import XHLLM_TRACEABLE_MODULES
from ._text_model import (
    _Qwen3MoeDecoderLayer,
    _Qwen3MoeModel,
    _Qwen3MoeRMSNorm,
    _Qwen3MoeRotaryEmbedding,
)


class _Qwen3OmniTalkerDynamicModule(DynamicModule):
    """Base wrapper for Qwen3 Omni talker modules."""

    def _setup(self, cfg: Optional[Dict] = None):
        return None


@XHLLM_TRACEABLE_MODULES.register_module(
    {
        Qwen3OmniMoeRMSNorm: "Qwen3OmniMoeRMSNorm",
    }
)
class _Qwen3OmniTalkerRMSNorm(_Qwen3MoeRMSNorm):
    """Register RMSNorm wrapper for talker stack."""


@XHLLM_TRACEABLE_MODULES.register_module(
    {
        Qwen3OmniMoeTextRMSNorm: "Qwen3OmniMoeTextRMSNorm",
    }
)
class _Qwen3OmniTalkerTextRMSNorm(_Qwen3MoeRMSNorm):
    """Register RMSNorm wrapper for talker stack."""


@XHLLM_TRACEABLE_MODULES.register_module(
    {
        Qwen3OmniMoeRotaryEmbedding: "Qwen3OmniMoeRotaryEmbedding",
    }
)
class _Qwen3OmniTalkerRotaryEmbedding(_Qwen3MoeRotaryEmbedding):
    """Register rotary embedding wrapper for talker stack."""


# @XHLLM_TRACEABLE_MODULES.register_module({Qwen3OmniMoeThinkerTextAttention: "Qwen3OmniMoeThinkerTextAttention",})
# class _Qwen3OmniMoeTalkerAttention(_Qwen3MoeAttention):
#     """Register thinker text attention wrapper for talker stack."""
#     pass


@XHLLM_TRACEABLE_MODULES.register_module(
    {
        Qwen3OmniMoeTalkerTextSparseMoeBlock: "Qwen3OmniMoeTalkerTextSparseMoeBlock",
    }
)
class _Qwen3OmniMoeTalkerTextSparseMoeBlock(_Qwen3OmniTalkerDynamicModule):
    def forward(self, hidden_states):
        ori_hidden_states = hidden_states
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        # hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=-1)
        final_hidden_states = self.moeblock(
            ori_hidden_states,
            routing_weights,
        )

        shared_expert_output = self.shared_expert(hidden_states)
        shared_expert_output = F.sigmoid(self.shared_expert_gate(hidden_states)) * shared_expert_output

        final_hidden_states = final_hidden_states + shared_expert_output

        return final_hidden_states

    def _setup(self, cfg: Optional[Dict] = None):
        self.input_seq_len = cfg.input_sequence_length
        self.batch_size = cfg.batch_size
        self.device = self.gate.weight.device

        self.moeblock = MoeBlock(self.experts[0].act_fn._get_name().lower(), self.top_k, self.norm_topk_prob)

        self.moeblock.expert_gate_proj_weight = torch.nn.Parameter(
            torch.zeros(
                self.num_experts,
                self.experts[0].gate_proj.weight.shape[0],
                self.experts[0].gate_proj.weight.shape[1],
                device=self.device,
                dtype=self.experts[0].gate_proj.weight.dtype,
            )
        )
        if hasattr(self.experts[0].gate_proj, "quant_weight") and self.experts[0].gate_proj.quant_weight is not None:
            self.moeblock.expert_gate_proj_quant_weight = torch.tensor(
                torch.zeros(
                    self.num_experts,
                    self.experts[0].gate_proj.quant_weight.shape[0],
                    self.experts[0].gate_proj.quant_weight.shape[1],
                    device=self.device,
                    dtype=self.experts[0].gate_proj.quant_weight.dtype,
                )
            )

        if self.experts[0].gate_proj.bias is not None:
            self.moeblock.expert_gate_proj_bias = torch.nn.Parameter(
                torch.zeros(self.num_experts, self.experts[0].gate_proj.bias.shape[0], device=self.device)
            )
        else:
            self.moeblock.expert_gate_proj_bias = None

        self.moeblock.expert_up_proj_weight = torch.nn.Parameter(
            torch.zeros(
                self.num_experts,
                self.experts[0].up_proj.weight.shape[0],
                self.experts[0].up_proj.weight.shape[1],
                device=self.device,
                dtype=self.experts[0].up_proj.weight.dtype,
            )
        )

        if hasattr(self.experts[0].up_proj, "quant_weight") and self.experts[0].up_proj.quant_weight is not None:
            self.moeblock.expert_up_proj_quant_weight = torch.tensor(
                torch.zeros(
                    self.num_experts,
                    self.experts[0].up_proj.quant_weight.shape[0],
                    self.experts[0].up_proj.quant_weight.shape[1],
                    device=self.device,
                    dtype=self.experts[0].up_proj.quant_weight.dtype,
                )
            )

        if self.experts[0].up_proj.bias is not None:
            self.moeblock.expert_up_proj_bias = torch.nn.Parameter(
                torch.zeros(self.num_experts, self.experts[0].up_proj.bias.shape[0], device=self.device)
            )
        else:
            self.moeblock.expert_up_proj_bias = None

        self.moeblock.expert_down_proj_weight = torch.nn.Parameter(
            torch.zeros(
                self.num_experts,
                self.experts[0].down_proj.weight.shape[0],
                self.experts[0].down_proj.weight.shape[1],
                device=self.device,
                dtype=self.experts[0].down_proj.weight.dtype,
            )
        )

        if hasattr(self.experts[0].down_proj, "quant_weight") and self.experts[0].down_proj.quant_weight is not None:
            self.moeblock.expert_down_proj_quant_weight = torch.tensor(
                torch.zeros(
                    self.num_experts,
                    self.experts[0].down_proj.quant_weight.shape[0],
                    self.experts[0].down_proj.quant_weight.shape[1],
                    device=self.device,
                    dtype=self.experts[0].down_proj.quant_weight.dtype,
                )
            )

        if self.experts[0].down_proj.bias is not None:
            self.moeblock.expert_down_proj_bias = torch.nn.Parameter(
                torch.zeros(self.num_experts, self.experts[0].down_proj.bias.shape[0], device=self.device)
            )
        else:
            self.moeblock.expert_down_proj_bias = None

        with torch.no_grad():
            self.moeblock.expert_gate_proj_weight.copy_(
                torch.cat([expert.gate_proj.weight.data.to(self.device).unsqueeze(0) for expert in self.experts], dim=0)
            )
            if (
                hasattr(self.experts[0].gate_proj, "quant_weight")
                and self.experts[0].gate_proj.quant_weight is not None
            ):
                self.moeblock.expert_gate_proj_quant_weight.copy_(
                    torch.cat(
                        [expert.gate_proj.quant_weight.data.to(self.device).unsqueeze(0) for expert in self.experts],
                        dim=0,
                    )
                )
            if self.moeblock.expert_gate_proj_bias is not None:
                self.moeblock.expert_gate_proj_bias.copy_(
                    torch.cat(
                        [expert.gate_proj.bias.data.to(self.device).unsqueeze(0) for expert in self.experts], dim=0
                    )
                )

            self.moeblock.expert_up_proj_weight.copy_(
                torch.cat([expert.up_proj.weight.data.to(self.device).unsqueeze(0) for expert in self.experts], dim=0)
            )
            if hasattr(self.experts[0].up_proj, "quant_weight") and self.experts[0].up_proj.quant_weight is not None:
                self.moeblock.expert_up_proj_quant_weight.copy_(
                    torch.cat(
                        [expert.up_proj.quant_weight.data.to(self.device).unsqueeze(0) for expert in self.experts],
                        dim=0,
                    )
                )
            if self.moeblock.expert_up_proj_bias is not None:
                self.moeblock.expert_up_proj_bias.copy_(
                    torch.cat([expert.up_proj.bias.data.to(self.device).unsqueeze(0) for expert in self.experts], dim=0)
                )

            self.moeblock.expert_down_proj_weight.copy_(
                torch.cat([expert.down_proj.weight.data.to(self.device).unsqueeze(0) for expert in self.experts], dim=0)
            )
            if (
                hasattr(self.experts[0].down_proj, "quant_weight")
                and self.experts[0].down_proj.quant_weight is not None
            ):
                self.moeblock.expert_down_proj_quant_weight.copy_(
                    torch.cat(
                        [expert.down_proj.quant_weight.data.to(self.device).unsqueeze(0) for expert in self.experts],
                        dim=0,
                    )
                )
            if self.moeblock.expert_down_proj_bias is not None:
                self.moeblock.expert_down_proj_bias.copy_(
                    torch.cat(
                        [expert.down_proj.bias.data.to(self.device).unsqueeze(0) for expert in self.experts], dim=0
                    )
                )

        del self.experts
        torch.cuda.empty_cache()

        return self


@XHLLM_TRACEABLE_MODULES.register_module(
    {
        Qwen3OmniMoeTalkerDecoderLayer: "Qwen3OmniMoeTalkerDecoderLayer",
    }
)
class _Qwen3OmniMoeTalkerDecoderLayer(_Qwen3MoeDecoderLayer):
    """Register code predictor decoder layer wrapper."""


@XHLLM_TRACEABLE_MODULES.register_module(
    {
        Qwen3OmniMoeTalkerModel: "Qwen3OmniMoeTalkerModel",
    }
)
class _Qwen3OmniMoeTalkerModel(_Qwen3MoeModel):
    """Register code predictor model wrapper."""


@XHLLM_TRACEABLE_MODULES.register_module(
    {
        Qwen3OmniMoeTalkerForConditionalGeneration: "Qwen3OmniMoeTalkerForConditionalGeneration",
    }
)
class _Qwen3OmniMoeTalkerForConditionalGeneration(_Qwen3OmniTalkerDynamicModule):
    """Register code predictor model for conditional generation wrapper."""

    def forward(
        self,
        inputs_embeds: Optional[Tensor] = None,
        past_seq_length: Optional[Tensor] = None,
        current_input_length: Optional[Tensor] = None,
        # position_ids: Optional[Tensor] = None,
        past_key_cache: Optional[List[Tensor]] = None,
        past_value_cache: Optional[List[Tensor]] = None,
    ):
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            past_seq_length=past_seq_length,
            current_input_length=current_input_length,
            # position_ids=position_ids,
            past_key_cache=past_key_cache,
            past_value_cache=past_value_cache,
        )

        hidden_states = outputs.last_hidden_state
        logits = self.codec_head(hidden_states)
        return logits

    def _setup(self, cfg: Optional[Dict] = None):
        return self


def register_wrap_modules():
    """Compatibility helper for API parity with thinker registration."""
    return None
