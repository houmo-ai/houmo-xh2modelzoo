from typing import Optional

from xhmodel_merak.xh_other_model.builder import register_other_model

# Qwen3-ForcedAligner uses the same Qwen3-ASR architecture and Transformers
# backend. Its only model-side difference is retaining logits for every token.
from xhmodel_merak.xh_other_model.models.qwen3_asr._llm_model_impl import (
    _Qwen3ASRThinkerTextModel,
)
from xhmodel_merak.xh_other_model.models.qwen3_asr._qwen3_asr_llm_model import (
    XHQwen3ASRLLMModel,
)
from xhmodel_merak.xh_other_model.models.qwen3_asr.modeling_qwen3_asr import (
    Qwen3ASRForConditionalGeneration,
)


class _Qwen3ForceAlignerThinkerTextModel(_Qwen3ASRThinkerTextModel):
    """Forced-aligner head that retains logits for every input position."""

    def forward(
        self,
        inputs_embeds=None,
        past_seq_length=None,
        current_input_length=None,
        past_key_cache=None,
        past_value_cache=None,
    ):
        hidden_states = inputs_embeds
        position_embeddings = (
            self.cos_slice(self.rotary_emb.cos_cached, past_seq_length),
            self.sin_slice(self.rotary_emb.sin_cached, past_seq_length),
        )

        for index, decoder_layer in enumerate(self.layers):
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=None,
                past_seq_length=past_seq_length,
                current_input_length=current_input_length,
                past_k_cache=past_key_cache[index] if self.use_cache else None,
                past_v_cache=past_value_cache[index] if self.use_cache else None,
                position_embeddings=position_embeddings,
            )
            hidden_states = layer_outputs[0]
            if self.only_first_block:
                break

        if self.num_logits_to_keep != 0:
            hidden_states = self.llm_gather(hidden_states, current_input_length - 1)
        hidden_states = self.norm(hidden_states)
        return self.lm_head(hidden_states)


@register_other_model("XHQwen3ForceAlignerLLMModel")
class XHQwen3ForceAlignerLLMModel(XHQwen3ASRLLMModel):
    WORKFLOW_CLS = (
        "xhmodel_merak.xh_other_model.models.qwen3_forcealigner.workflow:"
        "Qwen3ForceAlignerWorkflow"
    )

    def init_wrap_model(self, hf_model: Optional[_Qwen3ASRThinkerTextModel] = None):
        super().init_wrap_model(hf_model)
        # The ASR wrapper is already fully initialized here. Only its output-head
        # behavior differs: forced alignment consumes one logit vector per token.
        self.wrap_model.__class__ = _Qwen3ForceAlignerThinkerTextModel
