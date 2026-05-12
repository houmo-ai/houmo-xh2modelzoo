# Copyright 2025 HOUMO AI
#
# File: qwen_moe_hf_compatible.py
# Description:
#   Qwen Moe Hf Compatible implementation.
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

from typing import Any, Optional, Type, TypeVar, Union

import torch
import torch.nn as nn
from accelerate import init_empty_weights
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, DynamicCache, GenerationMixin, Qwen3MoeForCausalLM
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast
try:
    from transformers.modeling_utils import no_init_weights
except ImportError:
    import contextlib

    @contextlib.contextmanager
    def no_init_weights(_enable=True):
        yield

from .inference import Qwen3MoeInference

T = TypeVar("T", bound=Qwen3MoeForCausalLM)


def get_empty_hf_model(hf_model_dir, device_map="cpu", **kwargs) -> Qwen3MoeForCausalLM:
    """
    仅仅加载模型结构,不初始化权重,不占用显存
    """
    config = AutoConfig.from_pretrained(hf_model_dir)
    with no_init_weights(), init_empty_weights():
        hf_model: Qwen3MoeForCausalLM = AutoModelForCausalLM.from_config(
            config,
            torch_dtype=torch.float16,
            **kwargs,
        )
    return hf_model


class Qwen3MoeHFCompatible(Qwen3MoeForCausalLM):
    # _prefill: bool = True
    # _past_seq_length: int = 0
    # _embed_tokens: Optional[torch.nn.Module] = None
    # _llm_model: Optional[Qwen3LegacyInference] = None

    def setup(self, llm_model: Qwen3MoeInference, embed_tokens: torch.nn.Module):
        self._prefill = False
        self._past_seq_length = 0
        self._embed_tokens = embed_tokens
        self._llm_model = llm_model

    def __init__(self, *args, **kwargs):
        """Initializing a dynamic module is not allowed!"""
        raise RuntimeError("DynamicModule cannot be initialized directly; use convert instead!")

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

        # causal_mask = self._update_causal_mask(
        #     attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        # )

        # # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        # outputs: BaseModelOutputWithPast = self.model(
        #     input_ids=input_ids,
        #     attention_mask=attention_mask,
        #     position_ids=position_ids,
        #     past_key_values=past_key_values,
        #     inputs_embeds=inputs_embeds,
        #     use_cache=use_cache,
        #     output_attentions=output_attentions,
        #     output_hidden_states=output_hidden_states,
        #     cache_position=cache_position,
        #     **kwargs,
        # )

        # hidden_states = outputs.last_hidden_state
        # # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        # slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        # logits = self.lm_head(hidden_states[:, slice_indices, :])
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

        # Store all outputs to concatenate later
        all_outputs = []

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

            # Debug: print output shapes
            print(f"Step {i}: outputs type: {type(outputs)}")
            if isinstance(outputs, torch.Tensor):
                print(f"Step {i}: outputs shape: {outputs.shape}")
                all_outputs.append(outputs)
            else:
                # This should not happen for Qwen3MoeInference
                print(f"Step {i}: outputs.logits shape: {outputs.logits.shape}")
                all_outputs.append(outputs.logits)

        # Concatenate all outputs along sequence dimension
        print(f"Number of outputs: {len(all_outputs)}")
        if all_outputs:
            print(f"First output shape: {all_outputs[0].shape}")
            print(f"Expected seq_length: {seq_length}")

            # Check if we need to reshape the outputs
            reshaped_outputs = []
            for i, output in enumerate(all_outputs):
                if output.shape[1] == 1:  # If output has only 1 token
                    # Expand to match the expected sequence length for this step
                    step_seq_length = min(input_sequence_length, seq_length - i * input_sequence_length)
                    if step_seq_length > 1:
                        # Repeat the single token output to match expected length
                        output = output.expand(-1, step_seq_length, -1)
                        print(f"Step {i}: expanded output shape: {output.shape}")
                reshaped_outputs.append(output)

            if isinstance(reshaped_outputs[0], torch.Tensor):
                logits = torch.cat(reshaped_outputs, dim=1)  # Concatenate along seq_len dimension
            else:
                logits = torch.cat(reshaped_outputs, dim=1)
            print(f"Final concatenated logits shape: {logits.shape}")
        else:
            # Fallback: create dummy logits with correct shape
            batch_size = inputs_embeds.shape[0] if inputs_embeds is not None else 1
            vocab_size = self.config.vocab_size if hasattr(self.config, "vocab_size") else 151936
            logits = torch.zeros(
                (batch_size, seq_length, vocab_size),
                dtype=inputs_embeds.dtype if inputs_embeds is not None else torch.float16,
                device=inputs_embeds.device if inputs_embeds is not None else "cpu",
            )
            print(f"Fallback logits shape: {logits.shape}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return CausalLMOutputWithPast(
            # loss=loss,
            logits=logits,
            # past_key_values=outputs.past_key_values,
            # hidden_states=outputs.hidden_states,
            # attentions=outputs.attentions,
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
        hf_model_or_path: Union[Qwen3MoeForCausalLM, str],
        llm_model: Optional[Qwen3MoeInference] = None,
    ) -> Qwen3MoeForCausalLM:
        """
        将改写后的模型转换为兼容 Hugging Face 的模型
        """
        if isinstance(hf_model_or_path, str):
            hf_model = get_empty_hf_model(hf_model_or_path, device_map="auto")
        elif isinstance(hf_model_or_path, Qwen3MoeForCausalLM):
            hf_model = hf_model_or_path

        if llm_model is not None:
            assert isinstance(llm_model, Qwen3MoeInference)
            hf_model.__class__ = cls
            assert isinstance(hf_model, Qwen3MoeHFCompatible)
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

        class ForwardContext:
            def __init__(self, model: Qwen3MoeHFCompatible):
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
