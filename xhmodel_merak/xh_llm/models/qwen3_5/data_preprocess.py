# Copyright 2025 HOUMO AI
#
# File: data_preprocess.py
# Description:
#   Data preprocessing utilities for Qwen3 VL models.
#   This module provides Qwen3_VLDataPreprocess class for
#   preprocessing multimodal inputs.
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
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from ...llm_data_processor import BaseInputProcessorConfig, BaseLLMInputProcessor


class Qwen3_5_DataPreprocess(BaseLLMInputProcessor):  # noqa: N801
    def __init__(
        self,
        *,
        token_embedding: nn.Embedding,
        input_sequence_length: int = 256,
        image_size_w: int = 1204,
        image_size_h: int = 1204,
        past_key_caches=None,
        past_value_caches=None,
        past_conv_caches=None,
        past_recurrent_states=None,
        image_token_id: int = -1,
        video_token_id: int = -1,
        vision_start_token_id: int = -1,
        vision_end_token_id: int = -1,
        spatial_merge_size: int = 2,
        patch_size: int = 16,
    ):
        config = BaseInputProcessorConfig(token_embedding, input_sequence_length, past_key_caches, past_value_caches)
        super().__init__(config)
        self.token_embedding = token_embedding
        self.input_sequence_length = input_sequence_length
        self.rope_deltas = None
        self.pad_token_id = 0

        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id
        self.spatial_merge_size = spatial_merge_size

        self.patch_size = patch_size
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size
        self.image_size_w = image_size_w
        self.image_size_h = image_size_h

        self.past_key_caches = past_key_caches
        self.past_value_caches = past_value_caches
        self.past_conv_caches = past_conv_caches
        self.past_recurrent_states = past_recurrent_states

    def get_rope_index(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the 3D rope index based on image and video's temporal, height and width in LLM.

        Explanation:
            Each embedding sequence contains vision embedding and text embedding or just contains text embedding.

            For pure text embedding sequence, the rotary position embedding has no difference with mordern LLMs.
            Examples:
                input_ids: [T T T T T], here T is for text.
                temporal position_ids: [0, 1, 2, 3, 4]
                height position_ids: [0, 1, 2, 3, 4]
                width position_ids: [0, 1, 2, 3, 4]

            For vision and text embedding sequence, we calculate 3D rotary position embedding for vision part
            and 1D rotary position embeddin for text part.
            Examples:
                Assume we have a video input with 3 temporal patches, 2 height patches and 2 width patches.
                input_ids: [V V V V V V V V V V V V T T T T T], here V is for vision.
                vision temporal position_ids: [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
                vision height position_ids: [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
                vision width position_ids: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
                text temporal position_ids: [3, 4, 5, 6, 7]
                text height position_ids: [3, 4, 5, 6, 7]
                text width position_ids: [3, 4, 5, 6, 7]
                Here we calculate the text start position_ids as the max vision position_ids plus 1.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
                it.
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
            video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
                The temporal, height and width of feature shape of each video in LLM.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

        Returns:
            position_ids (`torch.LongTensor` of shape `(3, batch_size, sequence_length)`)
            mrope_position_deltas (`torch.Tensor` of shape `(batch_size)`)
        """
        spatial_merge_size = self.spatial_merge_size
        image_token_id = self.image_token_id
        video_token_id = self.video_token_id
        vision_start_token_id = self.vision_start_token_id
        mrope_position_deltas = []
        if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
            total_input_ids = input_ids
            if attention_mask is None:
                attention_mask = torch.ones_like(total_input_ids)
            position_ids = torch.ones(
                3, input_ids.shape[0], input_ids.shape[1], dtype=input_ids.dtype, device=input_ids.device
            )
            image_index, video_index = 0, 0
            for i, input_ids in enumerate(total_input_ids):
                input_ids = input_ids[attention_mask[i] == 1]
                image_nums, video_nums = 0, 0
                vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
                vision_tokens = input_ids[vision_start_indices + 1]
                image_nums = (vision_tokens == image_token_id).sum()
                video_nums = (vision_tokens == video_token_id).sum()
                input_tokens = input_ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_images, remain_videos = image_nums, video_nums
                for _ in range(image_nums + video_nums):
                    if image_token_id in input_tokens and remain_images > 0:
                        ed_image = input_tokens.index(image_token_id, st)
                    else:
                        ed_image = len(input_tokens) + 1
                    if video_token_id in input_tokens and remain_videos > 0:
                        ed_video = input_tokens.index(video_token_id, st)
                    else:
                        ed_video = len(input_tokens) + 1
                    if ed_image < ed_video:
                        t, h, w = (
                            image_grid_thw[image_index][0],
                            image_grid_thw[image_index][1],
                            image_grid_thw[image_index][2],
                        )
                        image_index += 1
                        remain_images -= 1
                        ed = ed_image
                    else:
                        t, h, w = (
                            video_grid_thw[video_index][0],
                            video_grid_thw[video_index][1],
                            video_grid_thw[video_index][2],
                        )
                        video_index += 1
                        remain_videos -= 1
                        ed = ed_video
                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        t.item(),
                        h.item() // spatial_merge_size,
                        w.item() // spatial_merge_size,
                    )
                    text_len = ed - st

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                    t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
                    h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                    w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                    llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w

                if st < len(input_tokens):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
                mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
            mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
            return position_ids, mrope_position_deltas
        else:
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(input_ids.device)
                max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
                mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
            else:
                B, S = inputs_embeds.shape[:2]
                position_ids = torch.arange(S, device=inputs_embeds.device).view(1, 1, -1).expand(3, B, -1)
                mrope_position_deltas = torch.zeros(
                    [B, 1],
                    device=inputs_embeds.device,
                    dtype=torch.long,
                )

            return position_ids, mrope_position_deltas

    def forward(self, data: Union[dict, tuple, list]):
        assert isinstance(data, dict)
        input_ids = data.get("input_ids", None)
        inputs_embeds = data.get("inputs_embeds", None)
        assert (input_ids is not None) or (inputs_embeds is not None), (
            "Either input_ids or inputs_embeds should be provided, but not both."
        )

        device = self._device

        if input_ids is not None:
            assert input_ids.shape[0] == 1, "Batch size should be 1 in inference mode."
            seq_length = input_ids.shape[1]
            assert seq_length <= self.input_sequence_length, (
                f"Input sequence length is too long. "
                f"max input sequence length is {self.input_sequence_length} but got {seq_length}"
            )
            if self.input_sequence_length > seq_length:
                padding_input_ids = torch.zeros(
                    (1, self.input_sequence_length - seq_length),
                    dtype=torch.long,
                    device=input_ids.device,
                )
                padding_input_ids.fill_(self.pad_token_id)
                input_ids = torch.cat([input_ids, padding_input_ids], dim=-1)
            inputs_embeds = self.embed_tokens(input_ids)
        elif inputs_embeds is not None:
            assert inputs_embeds.shape[0] == 1, "Batch size should be 1 in inference mode."
            seq_length = inputs_embeds.shape[1]
            inputs_embeds = inputs_embeds
            assert seq_length <= self.input_sequence_length, (
                "Input sequence length should be larger than input_sequence_length."
            )
            if self.input_sequence_length > seq_length:
                padding_token_id = self.pad_token_id
                padding_input_ids = torch.ones(
                    (1, self.input_sequence_length - seq_length),
                    dtype=torch.long,
                    device=inputs_embeds.device,
                )
                padding_input_ids = padding_input_ids * padding_token_id
                padding_embedding = self.embed_tokens(padding_input_ids)
                inputs_embeds = torch.cat([inputs_embeds, padding_embedding], dim=1)

        attention_mask = None

        # input_ids = data["input_ids"].to(device)
        # seq_length = input_ids.shape[1]

        # assert self.token_embedding is not None, "Token embedding is not available."
        # assert input_ids.shape[0] == 1, "Batch size should be 1 in inference mode."

        # assert seq_length <= self.input_sequence_length, (
        #     f"Input sequence length is too long. max input sequence length is {self.input_sequence_length} but got {seq_length}"
        # )
        # if self.input_sequence_length > seq_length:
        #     padding_input_ids = torch.zeros((1, self.input_sequence_length - seq_length), dtype=torch.long).to(device)
        #     padding_input_ids.fill_(self.pad_token_id)
        #     input_ids = torch.cat([input_ids, padding_input_ids], dim=-1)

        # Build linear attention mask
        linear_attn_mask = []
        if self.input_sequence_length > seq_length:
            pad_len = self.input_sequence_length - seq_length
            mask = torch.cat(
                [
                    torch.ones(seq_length, device=device),
                    torch.zeros(pad_len, device=device),
                ],
                dim=0,
            )
        else:
            mask = torch.ones(seq_length, device=device)
        linear_attn_mask.append(mask.unsqueeze(0))
        linear_attn_mask = torch.cat(linear_attn_mask, dim=0).to(device=device, dtype=torch.float16)

        image_embeds = data.get("image_embeds", None)
        if image_embeds is not None:
            n_image_tokens = (input_ids == self.image_token_id).sum().item()
            image_embeds = data["image_embeds"]
            n_image_features = image_embeds.shape[0]
            if n_image_tokens != n_image_features:
                raise ValueError(
                    f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                )
            image_mask = (
                (input_ids == self.image_token_id).unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
            )
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        past_seq_length = data["past_seq_length"]
        assert past_seq_length >= 0, "past_seq_length should be non-negative."

        if past_seq_length == 0:
            # prefill
            image_grid_thw = data.get("image_grid_thw", None)
            video_grid_thw = data.get("video_grid_thw", None)
            position_ids, rope_deltas = self.get_rope_index(
                input_ids, inputs_embeds, image_grid_thw, video_grid_thw, attention_mask
            )
            self.rope_deltas = rope_deltas
        else:
            assert self.rope_deltas is not None, f"rope_deltas is None, but past_seq_length is {past_seq_length}"
            batch_size, seq_length, _ = inputs_embeds.shape
            delta = past_seq_length + self.rope_deltas
            position_ids = torch.arange(seq_length, device=inputs_embeds.device)
            position_ids = position_ids.view(1, -1).expand(batch_size, -1)
            delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
            position_ids = position_ids.add(delta)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        time_position_ids = position_ids[0, 0].to(torch.int64)
        height_position_ids = position_ids[1, 0].to(torch.int64)
        width_position_ids = position_ids[2, 0].to(torch.int64)

        return (
            inputs_embeds.to(self._device),
            time_position_ids,
            height_position_ids,
            width_position_ids,
            torch.tensor([past_seq_length], dtype=torch.int32).to(self._device),
            torch.tensor([seq_length], dtype=torch.int32).to(self._device),
            linear_attn_mask.to(device),
            self.past_key_caches,
            self.past_value_caches,
            self.past_conv_caches,
            self.past_recurrent_states,
        )
