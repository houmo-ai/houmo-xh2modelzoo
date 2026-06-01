# Copyright 2025 HOUMO AI
#
# File: glm_ocr_hf_compatible.py
# Description:
#   GLM OCR Hugging Face-compatible wrapper for xh2modelzoo generation.
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

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
from transformers.models.glm_ocr.modeling_glm_ocr import (
    GlmOcrCausalLMOutputWithPast,
    GlmOcrForConditionalGeneration,
)


class GlmOcrHFCompatible(GlmOcrForConditionalGeneration):
    def __init__(self, *args, **kwargs):
        raise RuntimeError("DynamicModule cannot be initialized directly; use convert instead!")

    def _reset_llm_cache_state(self):
        llm_model = getattr(self, "_llm_model", None)
        if llm_model is None:
            return

        for cache_name in ("past_key_caches", "past_value_caches"):
            caches = getattr(llm_model, cache_name, None)
            if caches is None:
                continue
            for cache in caches:
                if isinstance(cache, torch.Tensor):
                    cache.zero_()

        if hasattr(llm_model, "rope_deltas"):
            llm_model.rope_deltas = None

    def _reset_state(self):
        # Avoid clashing with `GenerationMixin._prefill(...)` method name in transformers>=5.
        self._is_prefill_step = True
        self._past_seq_length = 0
        self._cached_image_embeds = None
        self._cached_image_grid_thw = None
        self._reset_llm_cache_state()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> Union[Tuple, GlmOcrCausalLMOutputWithPast]:
        if input_ids is None:
            raise ValueError("input_ids is required")

        seq_length = int(input_ids.shape[-1])
        if self._is_prefill_step:
            if hasattr(self, "_llm_model") and hasattr(self._llm_model, "set_input_sequence_length"):
                self._llm_model.set_input_sequence_length(seq_length)

            image_embeds = None
            if pixel_values is not None:
                image_features = self.get_image_features(pixel_values, image_grid_thw, return_dict=True).pooler_output
                image_embeds = torch.cat(image_features, dim=0).to(input_ids.device)

            self._cached_image_embeds = image_embeds
            self._cached_image_grid_thw = image_grid_thw
            past_seq_length = 0
            self._is_prefill_step = False
        else:
            if hasattr(self, "_llm_model") and hasattr(self._llm_model, "set_input_sequence_length"):
                self._llm_model.set_input_sequence_length(1)
            past_seq_length = int(self._past_seq_length)

        outputs = self._llm_model.test_step(
            {
                "input_ids": input_ids,
                "image_embeds": self._cached_image_embeds,
                "image_grid_thw": self._cached_image_grid_thw,
                "attention_mask": attention_mask,
                "past_seq_length": past_seq_length,
            }
        )
        logits = outputs.logits
        self._past_seq_length += seq_length

        return GlmOcrCausalLMOutputWithPast(logits=logits)

    @classmethod
    def to_hf_compatible(
        cls,
        hf_model: GlmOcrForConditionalGeneration,
        llm_model=None,
    ):
        if llm_model is not None:
            hf_model.__class__ = cls
            hf_model._llm_model = llm_model
            hf_model._reset_state()
        return hf_model

    def generate(self, *args, **kwargs):
        prefill_input_sequence_length = None
        if hasattr(self, "_llm_model") and hasattr(self._llm_model, "get_input_sequence_length"):
            prefill_input_sequence_length = self._llm_model.get_input_sequence_length()
        self._reset_state()
        try:
            return super().generate(*args, **kwargs)
        finally:
            if (
                prefill_input_sequence_length is not None
                and hasattr(self, "_llm_model")
                and hasattr(self._llm_model, "set_input_sequence_length")
            ):
                self._llm_model.set_input_sequence_length(prefill_input_sequence_length)
