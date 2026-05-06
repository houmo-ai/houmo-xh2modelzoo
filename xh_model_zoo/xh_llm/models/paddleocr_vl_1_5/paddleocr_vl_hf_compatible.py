# Copyright 2025 HOUMO AI
#
# File: paddleocr_vl_hf_compatible.py
# Description:
#   PaddleOCR-VL Hugging Face-compatible wrapper for xh2modelzoo multimodal inference.
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

from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from ..base_llm_model import BaseModel
from .modeling_paddleocr_vl import (
    PaddleOCRVLCausalLMOutputWithPast,
    PaddleOCRVLForConditionalGeneration,
)


class PaddleOCRVL_HFCompatible(PaddleOCRVLForConditionalGeneration):
    def __init__(self, *args, **kwargs):
        """Initializing a dynamic module is not allowed!"""
        raise RuntimeError(
            "DynamicModule cannot be initialized directly; use convert instead!"
        )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
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
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, PaddleOCRVLCausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        >>> model = Qwen3VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
        >>> processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

        >>> messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What is shown in this image?"},
                ],
            },
        ]
        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        >>> inputs = processor(text=[text], images=[image], vision_infos=[vision_infos])

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "The image shows a street scene with a red stop sign in the foreground. In the background, there is a large red gate with Chinese characters ..."
        ```"""

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        if inputs_embeds is None:
            inputs_embeds = self._llm_model.token_embedding(input_ids)

        image_mask = None
        video_mask = None

        if pixel_values is not None:
            image_embeds = self.get_image_features(pixel_values, image_grid_thw)
            image_embeds = torch.cat(image_embeds, dim=0).to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
            n_image_features = image_embeds.shape[0]
            if n_image_tokens != n_image_features:
                raise ValueError(
                    f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                )

            mask = input_ids == self.config.image_token_id
            mask_unsqueezed = mask.unsqueeze(-1)
            mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
            image_mask = mask_expanded.to(inputs_embeds.device)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            # TODO
            raise NotImplementedError("")

        if attention_mask is not None:
            attention_mask = attention_mask.to(inputs_embeds.device)

        # if we get 4D attention mask we cannot calculate rope deltas anymore. TODO @raushan fixme
        if position_ids is None:
            attention_mask_tensor = (
                attention_mask
                if not isinstance(attention_mask, dict)
                else attention_mask["full_attention"]
            )
            if attention_mask_tensor is not None and attention_mask_tensor.ndim == 4:
                attention_mask_tensor = torch.diagonal(
                    attention_mask_tensor[:, 0], dim1=1, dim2=2
                )
                # Only apply conversion for floating point tensors (inverted masks)
                if attention_mask_tensor.dtype.is_floating_point:
                    attention_mask_tensor = (
                        attention_mask_tensor
                        / torch.finfo(attention_mask_tensor.dtype).min
                    )
                    attention_mask_tensor = (1.0 - attention_mask_tensor).int()

            # Calculate RoPE index once per generation in the pre-fill stage only.
            if self._prefill:
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    attention_mask=attention_mask_tensor,
                )
                self.rope_deltas = rope_deltas
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (
                    (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                    if cache_position is not None
                    else 0
                )
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        if self._prefill:
            past_seq_length = torch.tensor([0], dtype=torch.int32).to(
                inputs_embeds.device
            )
            self._prefill = False
        else:
            past_seq_length = torch.tensor(
                [self._past_seq_length], dtype=torch.int32
            ).to(inputs_embeds.device)

        seq_length = input_ids.shape[-1]
        current_input_length = torch.tensor([seq_length], dtype=torch.int32).to(
            inputs_embeds.device
        )

        past_key_caches = self._llm_model.past_key_caches
        past_value_caches = self._llm_model.past_value_caches

        max_cache_length = getattr(self._llm_model, "max_sequence_length", None)
        if max_cache_length is None:
            max_cache_length = getattr(self._llm_model, "cache_length", None)
        if max_cache_length is not None:
            total_len = int(past_seq_length.item()) + int(current_input_length.item())
            if total_len > max_cache_length:
                if hasattr(self._llm_model, "reset_kvcache"):
                    self._llm_model.reset_kvcache()
                self._past_seq_length = 0
                past_seq_length = torch.tensor([0], dtype=torch.int32).to(
                    inputs_embeds.device
                )

        time_position_ids = position_ids[0, 0, :].to(torch.float16)
        hight_position_ids = position_ids[1, 0, :].to(torch.float16)
        width_position_ids = position_ids[2, 0, :].to(torch.float16)
        past_seq_length = past_seq_length.to(torch.float16)
        current_input_length = current_input_length.to(torch.float16)

        logits = self._llm_model(
            inputs_embeds,
            time_position_ids,
            hight_position_ids,
            width_position_ids,
            past_seq_length,
            current_input_length,
            past_key_caches,
            past_value_caches,
        )

        self._past_seq_length += seq_length
        return PaddleOCRVLCausalLMOutputWithPast(logits=logits)

    def get_image_features(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: Optional[torch.LongTensor],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        if image_grid_thw is None:
            raise ValueError("image_grid_thw is required when pixel_values is provided")

        if pixel_values.dim() == 4:
            pixel_values = pixel_values.unsqueeze(0)

        siglip_position_ids = []
        image_grid_hws = []
        sample_indices = []
        cu_seqlens = [0]

        for idx, thw in enumerate(image_grid_thw):
            thw_tuple = tuple(thw.detach().cpu().numpy().tolist())
            numel = np.prod(thw_tuple)
            image_grid_hws.append(thw_tuple)
            image_position_ids = torch.arange(numel) % np.prod(thw_tuple[1:])
            siglip_position_ids.append(image_position_ids)
            sample_indices.append(torch.full((numel,), idx, dtype=torch.int64))
            cu_seqlens.append(cu_seqlens[-1] + numel)

        siglip_position_ids = torch.concat(siglip_position_ids, dim=0).to(
            pixel_values.device
        )
        cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32).to(pixel_values.device)
        sample_indices = torch.concat(sample_indices, dim=0).to(pixel_values.device)

        vision_outputs = self.visual(
            pixel_values=pixel_values,
            image_grid_thw=image_grid_hws,
            position_ids=siglip_position_ids,
            vision_return_embed_list=True,
            interpolate_pos_encoding=True,
            sample_indices=sample_indices,
            cu_seqlens=cu_seqlens,
            return_pooler_output=False,
            use_rope=True,
            window_size=-1,
        )

        image_embeds = vision_outputs.last_hidden_state
        image_embeds = self.mlp_AR(image_embeds, image_grid_thw)
        if not isinstance(image_embeds, (list, tuple)):
            image_embeds = [image_embeds]

        return list(image_embeds)

    @classmethod
    def to_hf_compatible(
        cls,
        hf_model: PaddleOCRVLForConditionalGeneration,
        llm_model: Optional[BaseModel] = None,
        visual: Optional[BaseModel] = None,
    ):
        """
        将改写后的模型转换为兼容 Hugging Face 的模型
        """
        if llm_model is not None:
            assert isinstance(llm_model, BaseModel)
            hf_model.__class__ = cls
            hf_model._prefill = True
            hf_model._llm_model = llm_model
            if hasattr(hf_model, "model") and hasattr(hf_model.model, "language_model"):
                del hf_model.model.language_model
            if hasattr(hf_model, "lm_head"):
                del hf_model.lm_head
        if visual is not None:
            assert isinstance(visual, BaseModel)
            # del hf_model.visual
            hf_model.model.visual = visual
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return hf_model

    def generate(self, *args, **kwargs):
        self._prefill = True
        self._past_seq_length = 0
        return super().generate(*args, **kwargs)
