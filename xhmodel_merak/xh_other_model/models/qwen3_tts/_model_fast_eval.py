from typing import Optional

import torch
from qwen_tts.core.models.modeling_qwen3_tts import (
    Qwen3TTSTalkerCodePredictorModelForConditionalGeneration,
    Qwen3TTSTalkerForConditionalGeneration,
    Qwen3TTSTalkerOutputWithPast,
)
from torch import Tensor
from transformers import Cache
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.utils import can_return_tuple

from xhquant.patch import FUNCTION_REWRITER
from xhquant.utils.registry import DynamicModule

from ...builder import XHLLM_TRACEABLE_MODULES_TORCH_COMPILE


@XHLLM_TRACEABLE_MODULES_TORCH_COMPILE.register_module(
    {
        Qwen3TTSTalkerCodePredictorModelForConditionalGeneration: "Qwen3TTSTalkerCodePredictorModelForConditionalGeneration",
    }
)
class _Qwen3TTSTalkerCodePredictorModelForConditionalGeneration(DynamicModule):
    def _setup(self, cfg: dict | None = None):
        return self

    def forward(self, *args, **kwargs):
        # if "generation_steps" in kwargs and isinstance(kwargs["generation_steps"], Tensor):
        #     kwargs["generation_steps"] = kwargs["generation_steps"].item()
        output = super().forward(*args, **kwargs)
        if not isinstance(output.generation_steps, Tensor):
            output.generation_steps = torch.tensor(output.generation_steps)  # 转成Tensor，阻止出发dynamo的graph guard
        return output


@XHLLM_TRACEABLE_MODULES_TORCH_COMPILE.register_module(
    {
        Qwen3TTSTalkerForConditionalGeneration: "Qwen3TTSTalkerForConditionalGeneration",
    }
)
class _Qwen3TTSTalkerForConditionalGeneration(DynamicModule):
    def _setup(self, cfg: dict | None = None):
        return self

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[Cache] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            **kwargs,
        )

        inputs_embeds = inputs.get("inputs_embeds", None)
        if inputs_embeds is None:
            input_ids = inputs["input_ids"]
            past_hidden = inputs["past_hidden"]
            subtalker_dosample = inputs["subtalker_dosample"]
            subtalker_top_p = inputs["subtalker_top_p"]
            subtalker_top_k = inputs["subtalker_top_k"]
            subtalker_temperature = inputs["subtalker_temperature"]
            generation_step = inputs["generation_step"]
            trailing_text_hidden = inputs["trailing_text_hidden"]
            tts_pad_embed = inputs["tts_pad_embed"]

            last_id_hidden = self.get_input_embeddings()(input_ids)
            predictor_result = self.code_predictor.generate(
                inputs_embeds=torch.cat((past_hidden, last_id_hidden), dim=1),
                max_new_tokens=self.config.num_code_groups - 1,
                do_sample=subtalker_dosample,
                top_p=subtalker_top_p,
                top_k=subtalker_top_k,
                temperature=subtalker_temperature,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )
            codec_ids = torch.cat((input_ids, predictor_result.sequences), dim=-1)
            codec_hiddens = torch.cat(
                [last_id_hidden]
                + [
                    self.code_predictor.get_input_embeddings()[i](predictor_result.sequences[..., i : i + 1])
                    for i in range(self.config.num_code_groups - 1)
                ],
                dim=1,
            )
            inputs_embeds = codec_hiddens.sum(1, keepdim=True)

            if generation_step < trailing_text_hidden.shape[1]:
                inputs_embeds = inputs_embeds + trailing_text_hidden[:, generation_step].unsqueeze(1)
            else:
                inputs_embeds = inputs_embeds + tts_pad_embed
            inputs["inputs_embeds"] = inputs_embeds
            inputs["input_ids"] = None
            inputs["codec_ids"] = codec_ids
        return inputs


@FUNCTION_REWRITER.register_rewriter(
    func_name="qwen_tts.core.models.modeling_qwen3_tts.Qwen3TTSTalkerForConditionalGeneration.forward"
)
@can_return_tuple
def _forward(
    self,
    input_ids=None,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    inputs_embeds=None,
    labels=None,
    use_cache=None,
    output_attentions=None,
    output_hidden_states=None,
    cache_position=None,
    past_hidden=None,
    trailing_text_hidden=None,
    tts_pad_embed=None,
    generation_step=None,
    subtalker_dosample=None,
    subtalker_top_p=None,
    subtalker_top_k=None,
    subtalker_temperature=None,
    **kwargs,
) -> CausalLMOutputWithPast:
    """Forward pass for Qwen3TTS Talker model.

    Args:
        labels: Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
    """
    # Prefill
    if inputs_embeds is not None and inputs_embeds.shape[1] > 1:
        generation_step = -1
        codec_ids = None
    elif inputs_embeds is not None and inputs_embeds.shape[1] == 1:
        codec_ids = kwargs["codec_ids"]
    # Generate
    # else:
    #     last_id_hidden = self.get_input_embeddings()(input_ids)
    #     predictor_result = self.code_predictor.generate(
    #         inputs_embeds=torch.cat((past_hidden, last_id_hidden), dim=1),
    #         max_new_tokens=self.config.num_code_groups - 1,
    #         do_sample=subtalker_dosample,
    #         top_p=subtalker_top_p,
    #         top_k=subtalker_top_k,
    #         temperature=subtalker_temperature,
    #         output_hidden_states=True,
    #         return_dict_in_generate=True,
    #     )
    #     codec_ids = torch.cat((input_ids, predictor_result.sequences), dim=-1)
    #     codec_hiddens = torch.cat(
    #         [last_id_hidden]
    #         + [
    #             self.code_predictor.get_input_embeddings()[i](predictor_result.sequences[..., i : i + 1])
    #             for i in range(self.config.num_code_groups - 1)
    #         ],
    #         dim=1,
    #     )
    #     inputs_embeds = codec_hiddens.sum(1, keepdim=True)

    #     if generation_step < trailing_text_hidden.shape[1]:
    #         inputs_embeds = inputs_embeds + trailing_text_hidden[:, generation_step].unsqueeze(1)
    #     else:
    #         inputs_embeds = inputs_embeds + tts_pad_embed
    if attention_mask is not None:
        if (
            cache_position is None
            or (cache_position is not None and cache_position[0] == 0)
            or self.rope_deltas is None
        ):
            if attention_mask.dim() != 2:
                _attention_mask = attention_mask.any(dim=-1).long().squeeze(1)  # [1, 78]
            else:
                _attention_mask = attention_mask
            delta0 = (1 - _attention_mask).sum(dim=-1).unsqueeze(1)
            position_ids, rope_deltas = self.get_rope_index(_attention_mask)
            rope_deltas = rope_deltas - delta0
            self.rope_deltas = rope_deltas
        else:
            # batch_size, seq_length = input_ids.shape
            batch_size, seq_length = inputs_embeds.shape[:2]
            delta = cache_position[0] + self.rope_deltas if cache_position is not None else 0
            # position_ids = torch.arange(seq_length, device=input_ids.device)
            position_ids = torch.arange(seq_length, device=inputs_embeds.device)
            position_ids = position_ids.view(1, -1).expand(batch_size, -1)
            position_ids = position_ids.add(delta)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

    outputs: BaseModelOutputWithPast = self.model(
        input_ids=None,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        cache_position=cache_position,
        **kwargs,
    )

    hidden_states = outputs.last_hidden_state
    logits = self.codec_head(hidden_states)

    loss = None
    if labels is not None:
        loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

    return Qwen3TTSTalkerOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=([outputs.last_hidden_state], codec_ids),
        attentions=outputs.attentions,
        past_hidden=hidden_states[:, -1:, :],
        generation_step=generation_step + 1,
        trailing_text_hidden=trailing_text_hidden,
        tts_pad_embed=tts_pad_embed,
    )
