# Copyright 2025 HOUMO AI
#
# File: fm9g_hf_compatible.py
# Description:
#   FM9G HF Compatible implementation.
#   FM9G is a custom model loaded via trust_remote_code=True, so its model
#   class is only available at runtime. The HFCompatible class is created
#   dynamically to inherit from the actual FM9G model class.
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
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Optional, Union

import torch
import torch.nn as nn
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM, DynamicCache, GenerationMixin, PreTrainedModel
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import no_init_weights

from .inference import FM9GInference


def get_empty_hf_model(hf_model_dir, device_map="cpu", **kwargs) -> PreTrainedModel:
    """
    仅仅加载模型结构,不初始化权重,不占用显存.
    FM9G 需要 trust_remote_code=True 来加载自定义模型代码.
    """
    config = AutoConfig.from_pretrained(hf_model_dir, trust_remote_code=True)
    with no_init_weights(), init_empty_weights():
        hf_model: PreTrainedModel = AutoModelForCausalLM.from_config(
            config,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            **kwargs,
        )
    return hf_model


# Cache for the dynamically created HFCompatible class
_fm9g_hf_compatible_cls = None


class FM9GHFCompatibleMixin:
    """
    Mixin class providing HF-compatible forward/generate for FM9G.

    Since FM9G's model class is only available at runtime (loaded via
    trust_remote_code=True), we cannot statically inherit from it. Instead,
    we use this mixin which is dynamically combined with the actual FM9G
    model class in `to_hf_compatible()`.
    """

    def setup(self, llm_model: FM9GInference, embed_tokens: torch.nn.Module):
        self._prefill = False
        self._past_seq_length = 0
        self._embed_tokens = embed_tokens
        self._llm_model = llm_model

    def get_output_embeddings(self):
        """
        lm_eval需要这个接口
        """
        return None

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

        # TODO (joao): remove this exception in v4.56 -- it exists for users that try to pass a legacy cache
        if not isinstance(past_key_values, (type(None), Cache)):
            raise ValueError("The `past_key_values` should be either a `Cache` object or `None`.")

        if inputs_embeds is None:
            if self._embed_tokens is not None:
                inputs_embeds = self._embed_tokens(input_ids)
            else:
                raise ValueError("embed_tokens is not initialized")

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()  # type: ignore

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            ).long()  # type: ignore

        if position_ids is None and cache_position is not None:
            position_ids = cache_position.unsqueeze(0).long()  # type: ignore

        past_seq_length = torch.tensor([self._past_seq_length], dtype=torch.int32).to(inputs_embeds.device)

        # TODO: 需要根据attention mask计算seq_length
        if input_ids is not None:
            seq_length = input_ids.shape[-1]
        else:
            seq_length = inputs_embeds.shape[1] if inputs_embeds is not None else 0
        current_input_length = torch.tensor([seq_length], dtype=torch.int32).to(inputs_embeds.device)

        if self._llm_model is None:
            raise ValueError("llm_model is not initialized")

        input_sequence_length = self._llm_model.get_input_sequence_length()

        pad_input_seq_length = (
            (seq_length + input_sequence_length - 1) // input_sequence_length
        ) * input_sequence_length

        input_data = dict(
            input_ids=input_ids,
            past_seq_length=past_seq_length,
        )
        (
            inputs_embeds,
            past_seq_length,
            current_input_length,
            past_key_caches,
            past_value_caches,
        ) = self._llm_model.prepare_inputs(input_data, pad_input_seq_length)

        pad_seq_lenght = inputs_embeds.shape[1]
        assert pad_seq_lenght % input_sequence_length == 0, "pad_seq_lenght must be divisible by input_sequence_length"
        steps = pad_seq_lenght // input_sequence_length
        for i in range(steps):
            start = i * input_sequence_length
            end = (i + 1) * input_sequence_length
            sub_inputs_embeds = inputs_embeds[:, start:end, :]
            sub_past_seq_length = past_seq_length + start
            sub_current_input_length = torch.tensor(
                [min(end, seq_length) - start], dtype=current_input_length.dtype
            ).to(current_input_length.device)

            outputs = self._llm_model(
                sub_inputs_embeds,
                sub_past_seq_length,
                sub_current_input_length,
                past_key_caches,
                past_value_caches,
            )

        logits: Optional[torch.FloatTensor] = None
        if isinstance(outputs, torch.Tensor):
            logits = outputs
        else:
            logits = outputs.logits

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return CausalLMOutputWithPast(
            logits=logits,
        )

    @property
    def past_seq_length(self):
        return self._past_seq_length

    @past_seq_length.setter
    def past_seq_length(self, past_seq_length: int):
        self._past_seq_length = past_seq_length

    @property
    def prefill(self):
        return self._prefill

    @prefill.setter
    def prefill(self, prefill: bool):
        self._prefill = prefill
        if self._llm_model is not None and hasattr(self._llm_model, "set_phase_prefill"):
            self._llm_model.set_phase_prefill(prefill)

    @classmethod
    def to_hf_compatible(
        cls,
        hf_model_or_path: Union[PreTrainedModel, str],
        llm_model: Optional[FM9GInference] = None,
    ) -> PreTrainedModel:
        """
        将改写后的模型转换为兼容 Hugging Face 的模型.

        由于 FM9G 是通过 trust_remote_code=True 加载的自定义模型,
        其模型类仅在运行时可用。此方法动态创建一个同时继承
        FM9GHFCompatibleMixin 和原始 FM9G 模型类的兼容类。
        """
        global _fm9g_hf_compatible_cls

        if isinstance(hf_model_or_path, str):
            hf_model = get_empty_hf_model(hf_model_or_path, device_map="auto")
        else:
            hf_model = hf_model_or_path

        if llm_model is not None:
            assert isinstance(llm_model, FM9GInference)

            # Dynamically create HFCompatible class inheriting from both
            # the mixin and the actual FM9G model class
            original_cls = type(hf_model)
            if _fm9g_hf_compatible_cls is None or not issubclass(_fm9g_hf_compatible_cls, original_cls):
                _fm9g_hf_compatible_cls = type(
                    "FM9GHFCompatible",
                    (FM9GHFCompatibleMixin, original_cls),
                    {},
                )

            hf_model.__class__ = _fm9g_hf_compatible_cls
            hf_model.setup(llm_model=llm_model, embed_tokens=llm_model.token_embedding)

            if hasattr(hf_model, "model"):
                del hf_model.model
            if hasattr(hf_model, "lm_head"):
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
        if past_key_values is None:
            self.prefill = True
            self._past_seq_length = 0
        assert hasattr(self, "_original_forward"), "self._original_forward is not set"
        out = self._original_forward(
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
        # TODO: 需要根据attention mask计算seq_length
        if input_ids is not None:
            seq_length = input_ids.shape[-1]
        elif inputs_embeds is not None:
            seq_length = inputs_embeds.shape[-2]
        else:
            raise ValueError("You must specify either input_ids or inputs_embeds")
        self._past_seq_length += seq_length
        if self.prefill:
            self.prefill = False
        return out

    def generate(self, *args: Any, **kwargs: Any) -> torch.LongTensor:
        self.prefill = True
        self.past_seq_length = 0

        model_ref = self

        class ForwardContext:
            def __init__(self, model):
                self._model = model
                self._model._original_forward = self._model.forward
                self._prefill_input_sequence_length = self._model._llm_model.get_input_sequence_length()

            def __enter__(self):
                self._model.forward = self._model._sample_forward
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                self._model.forward = self._model._original_forward
                self._model._llm_model.set_input_sequence_length(self._prefill_input_sequence_length)

        with ForwardContext(model_ref):
            out = GenerationMixin.generate(self, *args, **kwargs)

        return out
