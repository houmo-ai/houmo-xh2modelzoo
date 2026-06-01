import inspect
from typing import Optional, Union

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import DynamicCache
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast

from xhquant.utils.registry import DynamicModule

from ..base_llm_model import LLMBaseModel


class LLM_HFCompatible(DynamicModule):  # noqa: N801
    def __init__(self, *args, **kwargs):
        """Initializing a dynamic module is not allowed!"""
        raise RuntimeError("DynamicModule cannot be initialized directly; use convert instead!")

    def _setup(self, llm_model: LLMBaseModel):
        self._prefill = True
        self._llm_model = llm_model
        self.embed_tokens = llm_model.token_embedding
        self._past_seq_length = 0
        self._dynamic_input: bool = False

        return self

    @property
    def dynamic_input(self) -> bool:
        return self._dynamic_input

    @dynamic_input.setter
    def dynamic_input(self, value: bool):
        self._dynamic_input = value

    @dynamic_input.setter
    def dynamic_input(self, value: bool) -> None:
        self._dynamic_input = value

    def get_output_embeddings(self):
        raise NotImplementedError

    def get_input_embeddings(self):
        return self.embed_tokens

    @property
    def prefill(self) -> bool:
        return self._prefill

    @prefill.setter
    def prefill(self, value: bool) -> None:
        self._prefill = value
        if hasattr(self._llm_model, "_prefill"):
            self._llm_model._prefill = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        if past_key_values is None:
            pass

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if not isinstance(past_key_values, (type(None), Cache)):
            raise ValueError("The `past_key_values` should be either a `Cache` object or `None`.")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        assert inputs_embeds.shape[0] == 1
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        past_seq_length = torch.tensor([self._past_seq_length], dtype=torch.int32).to(inputs_embeds.device)

        seq_length = None
        if input_ids is not None:
            seq_length = input_ids.shape[-1]
        elif inputs_embeds is not None:
            seq_length = inputs_embeds.shape[1]
        assert seq_length is not None, (
            "Failed to infer seq_length from inputs, please specify it explicitly if you are using dynamic input sequence length"
        )
        assert inputs_embeds is not None, "inputs_embeds cannot be None"

        current_input_length = torch.tensor([seq_length], dtype=torch.int32).to(inputs_embeds.device)

        past_key_caches = self._llm_model.past_value_caches
        past_value_caches = self._llm_model.past_key_caches

        if past_key_values is None or (hasattr(self, "use_cache") and not self.use_cache):
            past_key_caches = [torch.tensor([])] * len(past_key_caches)
            past_value_caches = [torch.tensor([])] * len(past_value_caches)

        if self.dynamic_input and isinstance(self._llm_model, LLMBaseModel):
            self._llm_model.set_input_sequence_length(seq_length)

        net_input_seq_len = self._llm_model.get_input_sequence_length()
        steps = (seq_length + net_input_seq_len - 1) // net_input_seq_len

        padding_len = steps * net_input_seq_len - seq_length
        if padding_len > 0:
            if isinstance(self.embed_tokens, (list, tuple, nn.ModuleList)):
                embed_token = self.embed_tokens[0]
            else:
                embed_token = self.embed_tokens
            padding_embeds = embed_token(
                torch.zeros(inputs_embeds.shape[0], padding_len, dtype=torch.long, device=inputs_embeds.device)
            )
            inputs_embeds = torch.cat([inputs_embeds, padding_embeds], dim=1)
        num_logits_to_keep = self._llm_model.get_num_logits_to_keep()
        extra_kwargs = {}
        if "generation_steps" in kwargs:  # for qwen3-tts
            extra_kwargs["generation_steps"] = kwargs["generation_steps"]
        if steps > 1:
            outputs_logits = []
            for i in tqdm(range(steps)):
                start = i * net_input_seq_len
                end = (i + 1) * net_input_seq_len
                sub_inputs_embeds = inputs_embeds[:, start:end, :]
                sub_past_seq_length = past_seq_length + start
                sub_current_input_length = torch.tensor(
                    [min(end, seq_length) - start], dtype=current_input_length.dtype
                ).to(current_input_length.device)
                self._llm_model.set_input_sequence_length(int(sub_current_input_length.item()))
                output = self._llm_model.forward(
                    sub_inputs_embeds,
                    sub_past_seq_length,
                    sub_current_input_length,
                    past_key_caches,
                    past_value_caches,
                    **extra_kwargs,
                )
                if isinstance(output, torch.Tensor):
                    logits = output
                else:
                    logits = output.logits
                outputs_logits.append(logits)
            if num_logits_to_keep != 0:
                logits = outputs_logits[-1]
            else:
                logits = torch.cat(outputs_logits, dim=1)[:, :seq_length, :]
        else:
            outputs = self._llm_model.forward(
                inputs_embeds, past_seq_length, current_input_length, past_key_caches, past_value_caches, **extra_kwargs
            )
            if isinstance(outputs, torch.Tensor):
                logits = outputs
            else:
                logits = outputs["logits"]
            if num_logits_to_keep == 0:
                logits = logits[:, :seq_length, :]

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return CausalLMOutputWithPast(
            logits=logits,
            past_key_values=past_key_values,
        )

    def _sample_forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        if input_ids is not None:
            seq_length = input_ids.shape[-1]
        elif inputs_embeds is not None:
            seq_length = inputs_embeds.shape[-2]
        else:
            raise ValueError("You must specify either input_ids or inputs_embeds")
        if self._prefill:
            if self.dynamic_input:
                self._llm_model.set_input_sequence_length(seq_length)
        else:
            self._llm_model.set_input_sequence_length(1)

        out = self._xh_orig_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )

        self._past_seq_length += seq_length
        if self.prefill:
            self.prefill = False
        return out

    def generate(self, *args, **kwargs):
        self.prefill = True
        self._past_seq_length = 0
        self._xh_orig_forward = self.forward
        self.forward = self._sample_forward
        self.prefill_input_sequence_length = self._llm_model.get_input_sequence_length()
        out = super().generate(*args, **kwargs)
        self._llm_model.set_input_sequence_length(self.prefill_input_sequence_length)
        self.forward = self._xh_orig_forward
        del self._xh_orig_forward
        return out
