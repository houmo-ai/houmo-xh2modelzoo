# Copyright 2018-2025 The Hugging Face Team. All rights reserved.
# Copyright 2025 HOUMO AI. All rights reserved.
#
# Modifications:
# - Adapted HuggingFace transformers Qwen2 implementation for xh2 hardware
# - Modified for CosyVoice3 LLM integration and HMONNX export
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# File: qwen2_hf_compatible.py
# Description:
#   Qwen2 HuggingFace compatibility layer for CosyVoice3 HMONNX inference.

from typing import Any, Optional, Union

import torch
import torch.nn as nn
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM, DynamicCache, Qwen2ForCausalLM
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import no_init_weights
from typing_extensions import Self

from ...base_llm_model import BaseModel


def get_empty_hf_model(hf_model_dir, device_map="cpu", **kwargs) -> Any:
    """
    仅仅加载模型结构,不初始化权重,不占用显存
    """
    config = AutoConfig.from_pretrained(hf_model_dir)
    with no_init_weights(), init_empty_weights():
        hf_model: nn.Module = AutoModelForCausalLM.from_config(
            config,
            torch_dtype=torch.float16,
            **kwargs,
        )
    return hf_model


def create_llm_wraped_cls(cls):
    class _Qwen2ForCausalLM(cls):
        def _setup(self, *args, **kwargs):
            pass

        @property
        def prefill(self):
            return self._prefill

        @prefill.setter
        def prefill(self, prefill: bool):
            self._prefill = prefill
            if hasattr(self._llm_model, "set_phase_prefill"):
                self._llm_model.set_phase_prefill(prefill)

        def generate(self, min_len, max_len, *args, **kwargs):
            stop_token = list(range(6561, 6761))
            out_tokens = []
            self.prefill = True
            self._past_seq_length = 0
            self.prefill_input_sequence_length = self._llm_model.get_input_sequence_length()
            out = self.forward(*args, **kwargs)
            logp = self._llm_model.llm_decoder_session(out.logits.squeeze(0))
            top_ids = self._llm_model.sampling_ids(
                logp.squeeze(dim=0), out_tokens, 25, ignore_eos=True if 0 < min_len else False
            ).item()
            if top_ids in stop_token:
                return out_tokens
            out_tokens.append(top_ids)
            lm_input = self._llm_model.speech_embedding.weight[top_ids].reshape(1, 1, -1)
            for i in range(1, max_len):
                out = self.forward(inputs_embeds=lm_input)
                logp = self._llm_model.llm_decoder_session(out.logits.squeeze(0))
                top_ids = self._llm_model.sampling_ids(
                    logp.squeeze(dim=0), out_tokens, 25, ignore_eos=True if i < min_len else False
                ).item()
                if top_ids in stop_token:
                    break
                out_tokens.append(top_ids)
                lm_input = self._llm_model.speech_embedding.weight[top_ids].reshape(1, 1, -1)
            return out_tokens

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
            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )

            use_cache = use_cache if use_cache is not None else self.config.use_cache

            if (input_ids is None) ^ (inputs_embeds is not None):
                raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

            if not isinstance(past_key_values, (type(None), Cache)):
                raise ValueError("The `past_key_values` should be either a `Cache` object or `None`.")

            if use_cache and past_key_values is None:
                past_key_values = DynamicCache()

            data = {}
            if input_ids is not None:
                data["input_ids"] = input_ids
                seq_length = input_ids.shape[-1]
            else:
                data["inputs_embeds"] = inputs_embeds
                seq_length = inputs_embeds.shape[1]

            if self._prefill:
                past_seq_length = [0]
            else:
                past_seq_length = [self._past_seq_length]

            data["past_seq_length"] = past_seq_length

            input_sequence_length = self._llm_model.get_input_sequence_length()

            pad_input_seq_length = (
                (seq_length + input_sequence_length - 1) // input_sequence_length
            ) * input_sequence_length
            self._llm_model.set_input_sequence_length(pad_input_seq_length)
            (
                inputs_embeds,
                past_seq_length,
                current_input_length,
                position_ids,
                past_key_caches,
                past_value_caches,
            ) = self._llm_model.prepare_inputs(data)
            self._llm_model.set_input_sequence_length(input_sequence_length)

            pad_seq_lenght = inputs_embeds.shape[1]
            assert pad_seq_lenght % input_sequence_length == 0, (
                "pad_seq_lenght must be divisible by input_sequence_length"
            )
            steps = pad_seq_lenght // input_sequence_length
            for i in range(steps):
                start = i * input_sequence_length
                end = (i + 1) * input_sequence_length
                sub_inputs_embeds = inputs_embeds[:, start:end, :]
                sub_past_seq_length = past_seq_length + start
                sub_current_input_length = torch.tensor(
                    [min(end, seq_length) - start], dtype=current_input_length.dtype
                ).to(current_input_length.device)
                sub_position_ids = position_ids[:, start:end]
                outputs = self._llm_model(
                    sub_inputs_embeds,
                    sub_past_seq_length,
                    sub_current_input_length,
                    sub_position_ids,
                    past_key_caches,
                    past_value_caches,
                )

            self._past_seq_length = (past_seq_length + current_input_length)[0].item()
            if isinstance(outputs, torch.Tensor):
                logits = outputs
            else:
                logits = outputs.logits

            if self.prefill:
                self.prefill = False
                self._llm_model.set_input_sequence_length(1)

            return CausalLMOutputWithPast(
                logits=logits,
            )

    return _Qwen2ForCausalLM


class Qwen2_HFCompatible(Qwen2ForCausalLM):  # noqa: N801
    def __init__(self, *args, **kwargs):
        """Initializing a dynamic module is not allowed!"""
        raise RuntimeError("DynamicModule cannot be initialized directly; use convert instead!")

    def set_input_sequence_length(self, input_seq_length: int) -> None:
        self._llm_model.set_input_sequence_length(input_seq_length)

    def __setup__(self, llm_model: BaseModel) -> Self:
        self._prefill = True
        self._llm_model = llm_model
        self.embed_tokens = llm_model.token_embedding
        self._past_seq_length = 0
        return self

    def get_output_embeddings(self):
        """
        lm_eval需要这个接口
        """
        return None

    @property
    def prefill(self) -> bool:
        return self._prefill

    @prefill.setter
    def prefill(self, value: bool) -> None:
        self._prefill = value

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

        seq_length = input_ids.shape[-1]
        current_input_length = torch.tensor([seq_length], dtype=torch.int32).to(inputs_embeds.device)

        past_key_caches = self._llm_model.past_value_caches
        past_value_caches = self._llm_model.past_key_caches

        if past_key_values is None or (hasattr(self, "use_cache") and not self.use_cache):
            past_key_caches = [torch.tensor([])] * len(past_key_caches)
            past_value_caches = [torch.tensor([])] * len(past_value_caches)

        outputs = self._llm_model._forward(
            inputs_embeds, past_seq_length, current_input_length, past_key_caches, past_value_caches
        )

        logits = outputs.logits
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return CausalLMOutputWithPast(
            logits=logits,
            past_key_values=past_key_values,
        )

    @classmethod
    def to_hf_compatible(
        cls,
        hf_model_or_path: Union[Qwen2ForCausalLM, str],
        llm_model: Optional[BaseModel] = None,
    ) -> Self:
        """
        将改写后的模型转换为兼容 Hugging Face 的模型
        """
        if isinstance(hf_model_or_path, str):
            hf_model = Qwen2ForCausalLM.from_pretrained(hf_model_or_path, torch_dtype=torch.float16, device_map="auto")
        elif isinstance(hf_model_or_path, Qwen2ForCausalLM):
            hf_model = hf_model_or_path

        if llm_model is not None:
            if not isinstance(llm_model, BaseModel):
                _llm_cls = create_llm_wraped_cls(type(hf_model))
                hf_model.__class__ = _llm_cls
                hf_model._llm_model = llm_model
                hf_model._prefill = True
            else:
                hf_model.__class__ = cls
                hf_model.__setup__(llm_model)
            del hf_model.model
            del hf_model.lm_head
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return hf_model

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
        self._llm_model.set_num_logits_to_keep(1)
        self._xh_orig_forward = self.forward
        self.forward = self._sample_forward
        self.prefill_input_sequence_length = self._llm_model.get_input_sequence_length()
        out = super().generate(*args, **kwargs)
        self._llm_model.set_input_sequence_length(self.prefill_input_sequence_length)
        self.forward = self._xh_orig_forward
        del self._xh_orig_forward
        return out
