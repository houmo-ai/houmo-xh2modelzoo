# -*- coding: utf-8 -*-
# Copyright 2025 The Qwen Team, Alibaba Group and The HuggingFace Inc. team. All rights reserved.
# Copyright 2025 HOUMO AI. All rights reserved.
#
# Modifications:
# - Portions of this file have been modified by HOUMO AI.
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
# File: qwen3_next_hf_compatible.py
# Description:
#   Qwen3-Next HuggingFace compatibility layer adapted for the xh2 model zoo (xh2modelzoo).

"""
Qwen3Next HF-compatible wrapper for lm_eval / evaluate_wikitext / generate().

Wraps the quantized/exported model so it can be used with standard
Hugging Face model(input_ids) interface.
"""

import re
from typing import Optional, Union

import torch
import transformers
from torch import Tensor
from transformers import DynamicCache
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextForCausalLM
from tqdm import tqdm
from xhquant.core import CacheTensor

from ..base_llm_model import BaseModel


class Qwen3NextHFCompatible(Qwen3NextForCausalLM):
    def __init__(self, *args, **kwargs):
        """Initializing a dynamic module is not allowed!"""
        raise RuntimeError(
            "DynamicModule cannot be initialized directly; use to_hf_compatible instead!"
        )

    def get_output_embeddings(self):
        return None

    @staticmethod
    def _ensure_transformers_52_plus() -> None:
        parts = [int(x) for x in re.findall(r"\d+", transformers.__version__)[:3]]
        while len(parts) < 3:
            parts.append(0)
        if tuple(parts[:3]) < (5, 2, 0):
            raise RuntimeError(
                "Qwen3Next requires transformers>=5.2.0; transformers 4.57 is not supported."
            )

    def _get_language_model_module(self):
        if hasattr(self._llm_model, "_get_language_model_module"):
            return self._llm_model._get_language_model_module()
        wrap_model = getattr(self._llm_model, "wrap_model", None)
        model = getattr(wrap_model, "model", None)
        if model is None:
            return None
        return getattr(model, "language_model", model)

    def _prepare_linear_state_caches(self, device: torch.device):
        if not hasattr(self._llm_model, "past_conv_caches"):
            self._llm_model.past_conv_caches = []
        if not hasattr(self._llm_model, "past_recurrent_states"):
            self._llm_model.past_recurrent_states = []

        past_conv_caches = self._llm_model.past_conv_caches
        past_recurrent_states = self._llm_model.past_recurrent_states

        if len(past_conv_caches) == 0 or len(past_recurrent_states) == 0:
            language_model = self._get_language_model_module()
            if language_model is not None and hasattr(language_model, "layers"):
                layer_indices = getattr(
                    self._llm_model, "linear_attention_layer_indices", None
                )
                if layer_indices is None:
                    layer_indices = [
                        i
                        for i, layer in enumerate(language_model.layers)
                        if getattr(layer, "layer_type", None) == "linear_attention"
                    ]

                batch_size = self._llm_model.wrap_cfg.get("batch_size", 1)
                past_conv_caches = []
                past_recurrent_states = []
                for layer_idx in layer_indices:
                    layer = language_model.layers[layer_idx]
                    linear_attn = layer.linear_attn
                    cache_dtype = linear_attn.conv1d.weight.dtype
                    past_conv_caches.append(
                        CacheTensor(
                            torch.zeros(
                                [
                                    batch_size,
                                    linear_attn.conv_dim,
                                    linear_attn.conv_kernel_size,
                                ],
                                dtype=cache_dtype,
                                device=device,
                            )
                        )
                    )
                    past_recurrent_states.append(
                        CacheTensor(
                            torch.zeros(
                                [
                                    batch_size,
                                    linear_attn.num_v_heads,
                                    linear_attn.head_k_dim,
                                    linear_attn.head_v_dim,
                                ],
                                dtype=cache_dtype,
                                device=device,
                            )
                        )
                    )
                self._llm_model.past_conv_caches = past_conv_caches
                self._llm_model.past_recurrent_states = past_recurrent_states

        past_conv_caches = self._llm_model.past_conv_caches
        past_recurrent_states = self._llm_model.past_recurrent_states

        if len(past_conv_caches) > 0 and past_conv_caches[0].device != device:
            past_conv_caches = [t.to(device) for t in past_conv_caches]
            self._llm_model.past_conv_caches = past_conv_caches
        if (
            len(past_recurrent_states) > 0
            and past_recurrent_states[0].device != device
        ):
            past_recurrent_states = [t.to(device) for t in past_recurrent_states]
            self._llm_model.past_recurrent_states = past_recurrent_states

        return past_conv_caches, past_recurrent_states

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
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
        self._ensure_transformers_52_plus()

        config_use_cache = getattr(self.config, "use_cache", None)
        if config_use_cache is None and hasattr(self.config, "text_config"):
            config_use_cache = getattr(self.config.text_config, "use_cache", False)
        use_cache = use_cache if use_cache is not None else config_use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        if not isinstance(past_key_values, (type(None), Cache)):
            raise ValueError(
                "The `past_key_values` should be either a `Cache` object or `None`."
            )

        if inputs_embeds is None:
            self.embed_tokens.weight = torch.nn.Parameter(
                self.embed_tokens.weight.to(input_ids.device)
            )
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        past_seq_length = torch.tensor(
            [self._past_seq_length], dtype=torch.int32
        ).to(inputs_embeds.device)

        seq_length = (
            input_ids.shape[-1]
            if input_ids is not None
            else inputs_embeds.shape[-2]
        )
        current_input_length = torch.tensor(
            [seq_length], dtype=torch.int32
        ).to(inputs_embeds.device)

        past_key_caches = self._llm_model.past_value_caches
        past_value_caches = self._llm_model.past_key_caches

        # When cache is disabled (PPL mode), nullify KV caches
        if not getattr(self._llm_model, "use_cache", True):
            past_key_caches = None
            past_value_caches = None

        past_conv_caches, past_recurrent_states = self._prepare_linear_state_caches(
            inputs_embeds.device
        )

        language_model = self._get_language_model_module()
        old_text_model_use_cache = None
        if (
            not getattr(self._llm_model, "use_cache", True)
            and language_model is not None
            and hasattr(language_model, "use_cache")
        ):
            old_text_model_use_cache = language_model.use_cache
            language_model.use_cache = True

        net_input_seq_len = self._llm_model.get_input_sequence_length()
        steps = (seq_length + net_input_seq_len - 1) // net_input_seq_len

        # Pad inputs_embeds to multiple of net_input_seq_len
        padding_len = steps * net_input_seq_len - seq_length
        if padding_len > 0:
            padding_embeds = self.embed_tokens(
                torch.zeros(
                    inputs_embeds.shape[0],
                    padding_len,
                    dtype=torch.long,
                    device=inputs_embeds.device,
                )
            )
            inputs_embeds = torch.cat([inputs_embeds, padding_embeds], dim=1)

        try:
            if steps > 1:
                outputs = []
                for i in tqdm(range(steps)):
                    start = i * net_input_seq_len
                    end = (i + 1) * net_input_seq_len
                    sub_inputs_embeds = inputs_embeds[:, start:end, :]
                    sub_past_seq_length = past_seq_length + start
                    actual_len = min(end, seq_length) - start
                    sub_current_input_length = torch.tensor(
                        [actual_len],
                        dtype=current_input_length.dtype,
                    ).to(current_input_length.device)

                    # Linear attention mask for this chunk
                    mask_dtype = inputs_embeds.dtype
                    if actual_len < net_input_seq_len:
                        lin_mask = torch.cat(
                            [
                                torch.ones(
                                    actual_len,
                                    device=inputs_embeds.device,
                                    dtype=mask_dtype,
                                ),
                                torch.zeros(
                                    net_input_seq_len - actual_len,
                                    device=inputs_embeds.device,
                                    dtype=mask_dtype,
                                ),
                            ],
                            dim=0,
                        ).unsqueeze(0)
                    else:
                        lin_mask = torch.ones(
                            net_input_seq_len,
                            device=inputs_embeds.device,
                            dtype=mask_dtype,
                        ).unsqueeze(0)

                    self._llm_model.set_input_sequence_length(
                        int(sub_current_input_length.item())
                    )
                    output = self._llm_model._forward(
                        sub_inputs_embeds,
                        sub_past_seq_length,
                        sub_current_input_length,
                        lin_mask,
                        past_key_caches,
                        past_value_caches,
                        past_conv_caches,
                        past_recurrent_states,
                    )
                    if isinstance(output, torch.Tensor):
                        logits = output
                    else:
                        logits = output.logits
                    outputs.append(logits)
                outputs = torch.concat(outputs, dim=1)
            else:
                # Single step
                mask_dtype = inputs_embeds.dtype
                if seq_length < net_input_seq_len:
                    lin_mask = torch.cat(
                        [
                            torch.ones(
                                seq_length, device=inputs_embeds.device, dtype=mask_dtype
                            ),
                            torch.zeros(
                                net_input_seq_len - seq_length,
                                device=inputs_embeds.device,
                                dtype=mask_dtype,
                            ),
                        ],
                        dim=0,
                    ).unsqueeze(0)
                else:
                    lin_mask = torch.ones(
                        net_input_seq_len, device=inputs_embeds.device, dtype=mask_dtype
                    ).unsqueeze(0)

                outputs = self._llm_model._forward(
                    inputs_embeds,
                    past_seq_length,
                    current_input_length,
                    lin_mask,
                    past_key_caches,
                    past_value_caches,
                    past_conv_caches,
                    past_recurrent_states,
                )
        finally:
            if old_text_model_use_cache is not None:
                language_model.use_cache = old_text_model_use_cache

        if isinstance(outputs, torch.Tensor):
            logits = outputs
        else:
            logits = outputs.logits
        logits = logits[:, :seq_length, :]

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return CausalLMOutputWithPast(
            logits=logits,
            past_key_values=past_key_values,
        )

    @classmethod
    def to_hf_compatible(
        cls,
        hf_model_or_path: Union[Qwen3NextForCausalLM, str],
        llm_model: Optional[BaseModel] = None,
    ) -> Qwen3NextForCausalLM:
        """Convert the adapted model to a HF-compatible model."""
        if isinstance(hf_model_or_path, str):
            hf_model = Qwen3NextForCausalLM.from_pretrained(
                hf_model_or_path, torch_dtype=torch.float16, device_map="auto"
            )
        elif isinstance(hf_model_or_path, Qwen3NextForCausalLM):
            hf_model = hf_model_or_path

        if llm_model is not None:
            assert isinstance(llm_model, BaseModel)
            hf_model.__class__ = cls
            hf_model._llm_model = llm_model
            hf_model.embed_tokens = llm_model.token_embedding
            hf_model._past_seq_length = 0
            del hf_model.model
            del hf_model.lm_head
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return hf_model
