# Copyright 2025 HOUMO AI
#
# File: text_llm_hf_compatible.py
# Description:
#   Text LLM HuggingFace-compatible wrapper for xh2modelzoo.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Union

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import DynamicCache
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast

from xhquant.utils.registry import DynamicModule, _DMRegistryCls

from .infer_mixin import TextLLMInferenceMixin


class TextLLMHFCompatible(DynamicModule):  # noqa: N801
    def __init__(self, *args, **kwargs):
        """Initializing a dynamic module is not allowed!"""
        raise RuntimeError("DynamicModule cannot be initialized directly; use convert instead!")

    def _setup(self, text_llm_model: TextLLMInferenceMixin):
        """"""
        """
        初始化模型
        """
        self._llm_prefill: bool = True
        self._llm_model = text_llm_model
        self.embed_tokens = text_llm_model.get_input_embeddings()
        self._past_seq_length = 0
        return self

    def get_input_embeddings(self):
        return self._llm_model.get_input_embeddings()

    @property
    def is_support_dynamic_input(self) -> bool:
        if not hasattr(self._llm_model, "is_support_dynamic_input"):
            return True  # 默认支持动态输入
        else:
            return self._llm_model.is_support_dynamic_input()

    @property
    def prefill(self) -> bool:
        return self._llm_prefill

    @prefill.setter
    def prefill(self, value: bool) -> None:
        self._llm_prefill = value
        if value:
            self._llm_model.set_prefill()
        else:
            self._llm_model.set_decode()

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

        r"""
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            logits_to_keep (`int` or `torch.Tensor`, *optional*):
                If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
                If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension.
                This is useful when using packed tensor format (single dimension for batch and sequence length).

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, Qwen3ForCausalLM

        >>> model = Qwen3ForCausalLM.from_pretrained("Qwen/Qwen3-8B")
        >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
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

        # if self.gradient_checkpointing and self.training and use_cache:
        #     # logger.warning_once(
        #     #     "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
        #     # )
        #     use_cache = False

        # TODO (joao): remove this exception in v4.56 -- it exists for users that try to pass a legacy cache
        if not isinstance(past_key_values, (type(None), Cache)):
            raise ValueError("The `past_key_values` should be either a `Cache` object or `None`.")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

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

        # TODO: 需要根据attention mask计算seq_length
        seq_length = None
        if input_ids is not None:
            seq_length = input_ids.shape[-1]
        elif inputs_embeds is not None:
            seq_length = inputs_embeds.shape[1]
        assert seq_length is not None, (
            "Failed to infer seq_length from inputs, please specify it explicitly if you are using dynamic input sequence length"
        )
        assert inputs_embeds is not None, "inputs_embeds cannot be None"  # for mypy

        current_input_length = torch.tensor([seq_length], dtype=torch.int32).to(inputs_embeds.device)
        # data_preprocessor = self._llm_model.get_data_preprocessor()

        past_key_caches = self._llm_model.past_key_caches
        past_value_caches = self._llm_model.past_value_caches

        if past_key_values is None or (hasattr(self, "use_cache") and not self.use_cache):
            past_key_caches = [torch.tensor([])] * len(past_key_caches)
            past_value_caches = [torch.tensor([])] * len(past_value_caches)

        if self.is_support_dynamic_input:
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
        # TODO: 需要根据attention mask计算seq_length
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
        # TODO: 需要根据attention mask计算seq_length
        if input_ids is not None:
            seq_length = input_ids.shape[-1]
        elif inputs_embeds is not None:
            seq_length = inputs_embeds.shape[-2]
        else:
            raise ValueError("You must specify either input_ids or inputs_embeds")
        if self._llm_prefill:
            self._llm_model.set_prefill()
            if self.is_support_dynamic_input:
                self._llm_model.set_input_sequence_length(seq_length)
        else:
            self._llm_model.set_decode()
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


class _TextLLMLegacyHFCompatible_(TextLLMHFCompatible):  # noqa: N801
    def _setup(self, text_llm_model):
        m = super()._setup(text_llm_model)
        if text_llm_model is not None:
            if hasattr(m, "model"):
                del m.model
            if hasattr(m, "lm_head"):
                del m.lm_head
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return m


def build_text_llm_hf_compatible(hf_model, text_llm_model) -> TextLLMHFCompatible:
    LLM_COMPATIBLE_MODULES = _DMRegistryCls("XHCompatible")  # noqa: N806
    hf_model_cls = type(hf_model)
    if hf_model_cls not in LLM_COMPATIBLE_MODULES:
        LLM_COMPATIBLE_MODULES.register_module(
            {
                hf_model_cls: hf_model_cls.__name__,
            },
            _TextLLMLegacyHFCompatible_,
        )
    return LLM_COMPATIBLE_MODULES.convert(hf_model, text_llm_model=text_llm_model)
