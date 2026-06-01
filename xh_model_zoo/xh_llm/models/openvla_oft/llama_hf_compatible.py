# Copyright 2025 HOUMO AI
#
# File: llama_hf_compatible.py
# Description:
#   OpenVLA-OFT Llama Hugging Face-compatible wrapper for xh2modelzoo generation.
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

from typing import Any, Optional, Union

import torch
import torch.nn as nn
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM, DynamicCache, LlamaForCausalLM
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import no_init_weights

# from transformers.models.qwen2.modeling_qwen2 import KwargsForCausalLM
from typing_extensions import Self

from ..base_llm_model import BaseModel

def create_llm_wraped_cls(cls):
    class _LlamaForCausalLM(cls):

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

        def generate(self, *args, **kwargs):
            self.prefill = True
            self._past_seq_length = 0
            self.prefill_input_sequence_length = self._llm_model.get_input_sequence_length()
            out = super().generate(*args, **kwargs, max_new_tokens=7, do_sample=False)
            self._llm_model.set_input_sequence_length(self.prefill_input_sequence_length)
            return out

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
            
            if attention_mask is not None:
                data["attention_mask"] = attention_mask

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
                attention_mask,
                past_key_caches,
                past_value_caches,
            ) = self._llm_model.prepare_inputs(data)
            self._llm_model.set_input_sequence_length(input_sequence_length)

            pad_seq_lenght = inputs_embeds.shape[1]
            assert (
                pad_seq_lenght % input_sequence_length == 0
            ), "pad_seq_lenght must be divisible by input_sequence_length"
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
                sub_attention_mask = attention_mask[:, :, start:end, :]
                outputs = self._llm_model(
                    sub_inputs_embeds,
                    sub_past_seq_length,
                    sub_current_input_length,
                    sub_position_ids,
                    sub_attention_mask,
                    past_key_caches,
                    past_value_caches,
                )

            self._past_seq_length = (past_seq_length + current_input_length)[0].item()
            if isinstance(outputs, torch.Tensor):
                logits = outputs
            else:
                logits = outputs.last_hidden_state

            if self.prefill:
                self.prefill = False
                self._llm_model.set_input_sequence_length(1)

            return logits

    return _LlamaForCausalLM

class Llama_HFCompatible(LlamaForCausalLM):
    def __init__(self, *args, **kwargs):
        """Initializing a dynamic module is not allowed!"""
        raise RuntimeError("DynamicModule cannot be initialized directly; use convert instead!")

    def set_input_sequence_length(self, input_seq_length: int) -> None:
        self._llm_model.set_input_sequence_length(input_seq_length)

    def __setup__(self, llm_model: BaseModel) -> Self:
        """"""
        """
        初始化模型
        """
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

    @classmethod
    def to_hf_compatible(
        cls,
        hf_model_or_path: Union[LlamaForCausalLM, str],
        llm_model: Optional[BaseModel] = None,
    ) -> Self:
        """
        将改写后的模型转换为兼容 Hugging Face 的模型
        """
        if isinstance(hf_model_or_path, str):
            hf_model = LlamaForCausalLM.from_pretrained(hf_model_or_path, torch_dtype=torch.float16, device_map="auto")
        elif isinstance(hf_model_or_path, LlamaForCausalLM):
            hf_model = hf_model_or_path

        if llm_model is not None:
            #assert isinstance(llm_model, BaseModel)
            if not isinstance(llm_model, BaseModel):
                _llm_cls = create_llm_wraped_cls(type(hf_model))
                hf_model.__class__ = _llm_cls
                hf_model._llm_model = llm_model
                hf_model.prefill = True
            else:
                hf_model.__class__ = cls
                hf_model.__setup__(llm_model)
            # hf_model.embed_tokens = hf_model.model.embed_tokens
            # del hf_model.model
            del hf_model.lm_head
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return hf_model

    def generate(self, *args, **kwargs):
        self.prefill = True
        self._past_seq_length = 0
        return super().generate(*args, **kwargs)
