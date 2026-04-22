from typing import Dict, List, Optional

import torch
from torch import Tensor
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeRMSNorm,
    Qwen3OmniMoeRotaryEmbedding,
    Qwen3OmniMoeTextRMSNorm,
    Qwen3OmniMoeTalkerCodePredictorAttention,
    Qwen3OmniMoeTalkerCodePredictorDecoderLayer,
    Qwen3OmniMoeTalkerCodePredictorModel,
    Qwen3OmniMoeTalkerCodePredictorModelForConditionalGeneration,
)
from xhquant.utils.registry import DynamicModule

from ._text_model import (
    _Qwen3MoeAttention,
    _Qwen3MoeDecoderLayer,
    _Qwen3MoeModel,
    _Qwen3MoeRMSNorm,
    _Qwen3MoeRotaryEmbedding,
)
from ..builder import XHLLM_TRACEABLE_MODULES


class _Qwen3OmniTalkerDynamicModule(DynamicModule):
    """Base wrapper for Qwen3 Omni talker modules."""

    def _setup(self, cfg: Optional[Dict] = None):
        return None


@XHLLM_TRACEABLE_MODULES.register_module(
    {Qwen3OmniMoeTalkerCodePredictorAttention: "Qwen3OmniMoeTalkerCodePredictorAttention"}
)
class _Qwen3OmniMoeTalkerCodePredictorAttention(_Qwen3MoeAttention):
    """Register code predictor attention wrapper."""


@XHLLM_TRACEABLE_MODULES.register_module(
    {Qwen3OmniMoeRMSNorm: "Qwen3OmniMoeRMSNorm"}
)
class _Qwen3OmniTalkerRMSNorm(_Qwen3MoeRMSNorm):
    """Register RMSNorm wrapper for code predictor stack."""


@XHLLM_TRACEABLE_MODULES.register_module(
    {Qwen3OmniMoeTextRMSNorm: "Qwen3OmniMoeTextRMSNorm"}
)
class _Qwen3OmniTalkerTextRMSNorm(_Qwen3MoeRMSNorm):
    """Register text RMSNorm wrapper for code predictor stack."""


@XHLLM_TRACEABLE_MODULES.register_module(
    {Qwen3OmniMoeRotaryEmbedding: "Qwen3OmniMoeRotaryEmbedding"}
)
class _Qwen3OmniTalkerRotaryEmbedding(_Qwen3MoeRotaryEmbedding):
    """Register rotary embedding wrapper for code predictor stack."""


@XHLLM_TRACEABLE_MODULES.register_module(
    {Qwen3OmniMoeTalkerCodePredictorDecoderLayer: "Qwen3OmniMoeTalkerCodePredictorDecoderLayer"}
)
class _Qwen3OmniMoeTalkerCodePredictorDecoderLayer(_Qwen3MoeDecoderLayer):
    """Register code predictor decoder layer wrapper."""


@XHLLM_TRACEABLE_MODULES.register_module(
    {Qwen3OmniMoeTalkerCodePredictorModel: "Qwen3OmniMoeTalkerCodePredictorModel"}
)
class _Qwen3OmniMoeTalkerPrediction(_Qwen3MoeModel):
    """Register code predictor model wrapper."""

    def forward(
        self,
        inputs_embeds: Optional[Tensor] = None,
        past_seq_length: Optional[Tensor] = None,
        current_input_length: Optional[Tensor] = None,
        past_key_cache: Optional[List[Tensor]] = None,
        past_value_cache: Optional[List[Tensor]] = None,
    ):
        return super().forward(
            inputs_embeds=inputs_embeds,
            past_seq_length=past_seq_length,
            current_input_length=current_input_length,
            past_key_cache=past_key_cache,
            past_value_cache=past_value_cache,
        )


@XHLLM_TRACEABLE_MODULES.register_module(
    {
        Qwen3OmniMoeTalkerCodePredictorModelForConditionalGeneration:
        "Qwen3OmniMoeTalkerCodePredictorModelForConditionalGeneration"
    }
)
class _Qwen3OmniMoeTalkerCodePredictorForConditionalGeneration(_Qwen3OmniTalkerDynamicModule):
    """Register code predictor model-for-generation wrapper.

    All 15 lm_heads are unfolded into the graph.  A ``head_mask`` tensor
    (float16, shape ``[B,S,15,1]``) selects which head(s) contribute to the
    output logits via element-wise multiplication and sum reduction.

        Both prefill and decode should use one-hot ``head_mask`` to match
        the HF predictor semantics where only one lm_head is active for
        each forward call.
    """

    def forward(
        self,
        inputs_embeds: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
        past_seq_length: Optional[Tensor] = None,
        current_input_length: Optional[Tensor] = None,
        past_key_cache: Optional[List[Tensor]] = None,
        past_value_cache: Optional[List[Tensor]] = None,
    ):
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            past_seq_length=past_seq_length,
            current_input_length=current_input_length,
            past_key_cache=past_key_cache,
            past_value_cache=past_value_cache,
        )
        hidden_states = outputs.last_hidden_state
        # Unfold every lm_head into the graph.
        # hidden_states: [B, S, H]
        # head_logits_list[i]: [B, S, V]
        head_logits_list = [head(hidden_states) for head in self.lm_head]
        # Stack to [B, S, num_heads, V]
        all_logits = torch.stack(head_logits_list, dim=2)
        # head_mask: [B, S, num_heads, 1] -> broadcast over V
        selected_logits = (all_logits * head_mask).sum(dim=2)
        return selected_logits, hidden_states

    def _setup(self, cfg: Optional[Dict] = None):
        return self


def register_wrap_modules():
    """Compatibility helper for API parity with thinker registration."""
    return None
