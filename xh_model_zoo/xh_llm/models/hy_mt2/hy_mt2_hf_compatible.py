# Copyright 2025 HOUMO AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Optional, TypeVar, Union

import torch
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM, DynamicCache, GenerationMixin
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import no_init_weights
from transformers.models.hunyuan_v1_dense.modeling_hunyuan_v1_dense import HunYuanDenseV1ForCausalLM

from .inference import HyMT2Inference

T = TypeVar("T", bound=HunYuanDenseV1ForCausalLM)


def get_empty_hf_model(hf_model_dir, device_map="cpu", **kwargs) -> HunYuanDenseV1ForCausalLM:
    config = AutoConfig.from_pretrained(hf_model_dir, trust_remote_code=True)
    with no_init_weights(), init_empty_weights():
        hf_model: HunYuanDenseV1ForCausalLM = AutoModelForCausalLM.from_config(
            config,
            trust_remote_code=True,
            dtype=torch.float16,
            **kwargs,
        )
    return hf_model


class HyMT2HFCompatible(HunYuanDenseV1ForCausalLM):
    def setup(self, llm_model: HyMT2Inference, embed_tokens: torch.nn.Module):
        self._prefill = False
        self._past_seq_length = 0
        self._embed_tokens = embed_tokens
        self._llm_model = llm_model

    def __init__(self, *args, **kwargs):
        raise RuntimeError("DynamicModule cannot be initialized directly; use convert instead!")

    def get_output_embeddings(self):
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
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if not isinstance(past_key_values, (type(None), Cache)):
            raise ValueError("The `past_key_values` should be either a `Cache` object or `None`.")
        if inputs_embeds is None:
            if self._embed_tokens is not None:
                inputs_embeds = self._embed_tokens(input_ids)
            else:
                raise ValueError("embed_tokens is not initialized")
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)
        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            ).long()
        if position_ids is None and cache_position is not None:
            position_ids = cache_position.unsqueeze(0).long()

        if input_ids is not None:
            seq_length = input_ids.shape[-1]
        else:
            seq_length = inputs_embeds.shape[1] if inputs_embeds is not None else 0

        if self._llm_model is None:
            raise ValueError("llm_model is not initialized")
        input_sequence_length = self._llm_model.get_input_sequence_length()
        pad_input_seq_length = ((seq_length + input_sequence_length - 1) // input_sequence_length) * input_sequence_length
        input_data = dict(input_ids=input_ids, past_seq_length=self._past_seq_length)
        (
            inputs_embeds,
            past_seq_length,
            current_input_length,
            past_key_caches,
            past_value_caches,
        ) = self._llm_model.prepare_inputs(input_data, pad_input_seq_length)

        steps = inputs_embeds.shape[1] // input_sequence_length
        outputs = None
        for i in range(steps):
            start = i * input_sequence_length
            end = (i + 1) * input_sequence_length
            sub_current_input_length = torch.tensor(
                [min(end, seq_length) - start], dtype=current_input_length.dtype, device=current_input_length.device
            )
            outputs = self._llm_model(
                inputs_embeds[:, start:end, :],
                past_seq_length + start,
                sub_current_input_length,
                past_key_caches,
                past_value_caches,
            )

        logits = outputs if isinstance(outputs, torch.Tensor) else outputs.logits
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return CausalLMOutputWithPast(logits=logits)

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
        hf_model_or_path: Union[HunYuanDenseV1ForCausalLM, str],
        llm_model: Optional[HyMT2Inference] = None,
    ) -> HunYuanDenseV1ForCausalLM:
        if isinstance(hf_model_or_path, str):
            hf_model = get_empty_hf_model(hf_model_or_path, device_map="auto")
        elif isinstance(hf_model_or_path, HunYuanDenseV1ForCausalLM):
            hf_model = hf_model_or_path
        else:
            raise TypeError(f"Unsupported model type: {type(hf_model_or_path)}")

        if llm_model is not None:
            assert isinstance(llm_model, HyMT2Inference)
            hf_model.__class__ = cls
            assert isinstance(hf_model, HyMT2HFCompatible)
            hf_model.setup(llm_model=llm_model, embed_tokens=llm_model.token_embedding)
            if hasattr(hf_model, "model"):
                del hf_model.model
            if hasattr(hf_model, "lm_head"):
                del hf_model.lm_head
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return hf_model

    def _sample_forward(self, *args: Any, **kwargs: Any) -> CausalLMOutputWithPast:
        past_key_values = kwargs.get("past_key_values", None)
        if past_key_values is None:
            self.prefill = True
            self._past_seq_length = 0
        assert hasattr(self, "_original_forward"), "self._original_forward is not set"
        out = self._original_forward(*args, **kwargs)
        input_ids = kwargs.get("input_ids", args[0] if len(args) > 0 else None)
        inputs_embeds = kwargs.get("inputs_embeds", None)
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

        class ForwardContext:
            def __init__(self, model: HyMT2HFCompatible):
                self._model = model
                self._model._original_forward = self._model.forward
                self._prefill_input_sequence_length = self._model._llm_model.get_input_sequence_length()

            def __enter__(self):
                self._model.forward = self._model._sample_forward
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                self._model.forward = self._model._original_forward
                self._model._llm_model.set_input_sequence_length(self._prefill_input_sequence_length)

        with ForwardContext(self):
            out = GenerationMixin.generate(self, *args, **kwargs)
        return out
