# Copyright 2025 HOUMO AI
#
# File: _static_export.py
# Description:
#   Static-shape export wrappers for FunAudioChat encoder/decoder.
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

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FunAudioChatStaticEncoderWarp(nn.Module):
    """Static-shape frontend warp for FunAudioChat.

    Notes:
    - Designed for export with batch size 1.
    - All dynamic padding/chunk metadata must be precomputed outside.
    - Avoids Python control flow based on traced tensors.
    """

    def __init__(
        self,
        model: nn.Module,
        feature_lens: torch.LongTensor,
        aftercnn_lens: torch.LongTensor,
        speech_maxlen: int,
        feature_exist_mask: torch.Tensor,
        audio_token_positions: torch.LongTensor,
        chunk_lengths: List[int],
        pooled_lengths: List[int],
    ):
        super().__init__()
        self.audio_encoder = model.continuous_audio_tower
        self.audio_tower = model.audio_tower
        self.speech_maxlen = int(speech_maxlen)
        self.group_size = int(model.config.audio_config.group_size)
        self.max_audio_tokens = len(pooled_lengths)
        self.avg_pool2d = nn.AvgPool2d(kernel_size=(2, 1), stride=(2, 1))
        self.fixed_pooled_len = self.max_aftercnn_len if self.max_aftercnn_len < 2 else self.max_aftercnn_len // 2
        self.total_pooled_length = sum(self.pooled_lengths)
        self.num_chunks = len(self.aftercnn_split_lengths)
        self.total_aftercnn_length = self.num_chunks * self.max_aftercnn_len
        self.fixed_total_pooled_length = self.num_chunks * self.fixed_pooled_len
        self.max_conv_output_len = (self.max_chunk_len - 1) // 2 + 1
        cu_seqlens = torch.cat(
            [
                torch.zeros(1, dtype=torch.int32),
                torch.tensor(self.aftercnn_split_lengths, dtype=torch.int32).cumsum(0),
            ]
        )
        self.register_buffer("cu_seqlens", cu_seqlens, persistent=False)
        self.register_buffer(
            "positional_embedding_cache",
            self.audio_encoder.positional_embedding.positional_embedding[: self.max_conv_output_len, :].unsqueeze(0).to(torch.float16),
            persistent=False,
        )

        self.register_buffer("feature_lens", feature_lens.to(torch.int32), persistent=False)
        self.register_buffer("aftercnn_lens", aftercnn_lens.to(torch.int32), persistent=False)
        self.register_buffer("feature_exist_mask", feature_exist_mask.to(torch.bool), persistent=False)
        self.register_buffer("audio_token_positions", audio_token_positions.to(torch.int64), persistent=False)
        self.register_buffer("continuous_audio_output_lengths", torch.tensor(pooled_lengths, dtype=torch.int32), persistent=False)

        self.chunk_lengths = [int(x) for x in chunk_lengths]
        self.aftercnn_split_lengths = [int(x) for x in aftercnn_lens.tolist()]
        self.pooled_lengths = [int(x) for x in pooled_lengths]
        self.max_chunk_len = max(self.chunk_lengths)
        self.max_aftercnn_len = max(self.aftercnn_split_lengths)
        self.fixed_pooled_len = self.max_aftercnn_len if self.max_aftercnn_len < 2 else self.max_aftercnn_len // 2
        self.total_pooled_length = sum(self.pooled_lengths)
        self.num_chunks = len(self.aftercnn_split_lengths)
        self.total_aftercnn_length = self.num_chunks * self.max_aftercnn_len
        self.fixed_total_pooled_length = self.num_chunks * self.fixed_pooled_len
        self.max_conv_output_len = (self.max_chunk_len - 1) // 2 + 1
        cu_seqlens = torch.cat(
            [
                torch.zeros(1, dtype=torch.int32),
                torch.tensor(self.aftercnn_split_lengths, dtype=torch.int32).cumsum(0),
            ]
        )
        self.register_buffer("cu_seqlens", cu_seqlens, persistent=False)
        self.register_buffer(
            "positional_embedding_cache",
            self.audio_encoder.positional_embedding.positional_embedding[: self.max_conv_output_len, :].unsqueeze(0).to(torch.float16),
            persistent=False,
        )

    def _encode_continuous_audio(
        self,
        padded_feature: torch.Tensor,
        chunk_padded_mask: torch.Tensor,
        aftercnn_valid_mask: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        padded_mask = chunk_padded_mask

        padded_embed = F.gelu(self.audio_encoder.conv1(padded_feature)) * padded_mask
        padded_embed = F.gelu(self.audio_encoder.conv2(padded_embed)).transpose(1, 2)
        padded_embed = padded_embed + self.positional_embedding_cache
        hidden_states = (padded_embed * aftercnn_valid_mask).reshape(-1, padded_embed.shape[-1])
        flat_valid_mask = aftercnn_valid_mask.reshape(-1, 1)

        cu_seqlens = self.cu_seqlens

        for encoder_layer in self.audio_encoder.layers:
            hidden_states = encoder_layer(
                hidden_states,
                cu_seqlens=cu_seqlens,
                attention_mask=attention_mask,
            )[0]
            hidden_states = hidden_states * flat_valid_mask

        if self.max_aftercnn_len < 2:
            pooled_concat = hidden_states
        else:
            pooled_concat = hidden_states.reshape(self.num_chunks, self.max_aftercnn_len, -1)
            pooled_concat = pooled_concat[:, : self.fixed_pooled_len * 2, :]
            pooled_concat = pooled_concat.reshape(
                self.num_chunks, self.fixed_pooled_len, 2, -1
            ).mean(dim=2)
            pooled_concat = pooled_concat.reshape(self.num_chunks * self.fixed_pooled_len, -1)
        processed_concat = self.audio_encoder.proj(self.audio_encoder.ln_post(pooled_concat))
        return processed_concat.unsqueeze(0)

    def forward(
        self,
        speech_ids: torch.LongTensor,
        audio_inputs_embeds: torch.FloatTensor,
        padded_input_features: torch.FloatTensor,
        chunk_padded_mask: torch.Tensor,
        aftercnn_valid_mask: torch.Tensor,
        attention_mask: torch.Tensor,
        continuous_audio_valid_mask: torch.Tensor,
    ):
        processed_concat = self._encode_continuous_audio(
            padded_input_features,
            chunk_padded_mask,
            aftercnn_valid_mask,
            attention_mask,
        )
        # continuous_audio_features = torch.zeros(
        #     (1, self.speech_maxlen, processed_concat.shape[-1]),
        #     dtype=processed_concat.dtype,
        #     device=processed_concat.device,
        # )
        # copy_length = min(self.fixed_total_pooled_length, processed_concat.shape[1], self.speech_maxlen)
        # continuous_audio_features[:, :copy_length, :] = processed_concat[:, :copy_length, :]
        copy_length = min(self.fixed_total_pooled_length, self.speech_maxlen)
        continuous_audio_features = processed_concat[:, :copy_length, :]
        if copy_length < self.speech_maxlen:
            continuous_audio_features = F.pad(
                continuous_audio_features,
                (0, 0, 0, self.speech_maxlen - copy_length),
            )

        # grouped_valid_mask = continuous_audio_valid_mask.reshape(1, -1, self.group_size, 1).to(processed_concat.dtype)
        # grouped_features = continuous_audio_features.reshape(1, -1, self.group_size, processed_concat.shape[-1])
        # grouped_features = grouped_features * grouped_valid_mask
        # continuous_audio_features = grouped_features

        continuous_audio_features = continuous_audio_features * continuous_audio_valid_mask

        audio_features, *_ = self.audio_tower(
            speech_ids,
            inputs_embeds=audio_inputs_embeds,
            continuous_audio_features=continuous_audio_features,
            # continuous_audio_valid_mask=grouped_valid_mask,
            continuous_audio_output_lengths=self.continuous_audio_output_lengths,
            feature_exist_mask=self.feature_exist_mask,
        )

        return audio_features


class FunAudioChatStaticAudioTowerWarp(nn.Module):
    """Static-shape discrete audio tower wrapper for generation-time last-group input."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.audio_tower = model.audio_tower

    def forward(self, speech_ids: torch.LongTensor):
        audio_features = self.audio_tower(speech_ids)[0]
        return audio_features


class FunAudioChatStaticDecoderPrefillWarp(nn.Module):
    """Static-shape decoder prefill export wrapper."""

    def __init__(self, decoder: nn.Module):
        super().__init__()
        self.decoder = decoder
        if not hasattr(self.decoder, "crq_audio_embeds"):
            self.decoder.crq_audio_embeds =  self.decoder.get_embeddings(self.decoder.config.bos_token_id)[None, None, :]

    def forward(
        self,
        crq_inputs_embeds: torch.FloatTensor,
        past_seq_length: torch.IntTensor,
        current_input_length: torch.IntTensor,
        attention_mask: Optional[torch.Tensor],
        past_key_cache: Optional[List[torch.Tensor]] = None,
        past_value_cache: Optional[List[torch.Tensor]] = None,
    ):
        # bs = crq_inputs_embeds.shape[0]
        # start_audio_embeds = (
        #     self.decoder.get_embeddings(self.decoder.config.bos_token_id)[None, None, :]
        #     .repeat(bs, 1, 1)
        #     .to(dtype=crq_inputs_embeds.dtype, device=crq_inputs_embeds.device)
        # )
        crq_inputs_embeds = crq_inputs_embeds + self.decoder.crq_audio_embeds
        crq_inputs_embeds = self.decoder.input_matching(crq_inputs_embeds)

        outputs = self.decoder.crq_transformer(
            inputs_embeds=crq_inputs_embeds,
            attention_mask=attention_mask,
            past_seq_length=past_seq_length,
            current_input_length=current_input_length,
            past_key_cache=past_key_cache,
            past_value_cache=past_value_cache,
        )
        hidden_states = self.decoder.output_matching(outputs.last_hidden_state)
        logits = self.decoder.lm_head(hidden_states)

        return logits


class FunAudioChatStaticDecoderDecodeWarp(nn.Module):
    """Static-shape decoder generate-branch export wrapper.

    This wrapper mirrors `crq_generate_forward` for one text token using greedy
    step-by-step CRQ generation, without Python-side mutable state.
    """

    def __init__(self, decoder: nn.Module):
        super().__init__()
        self.decoder = decoder
        if not hasattr(self.decoder, "crq_audio_embeds"):
            self.decoder.crq_audio_embeds = self.decoder.get_embeddings(self.decoder.config.bos_token_id)[None, None, :]
        elif self.decoder.crq_audio_embeds.ndim != 3:
            self.decoder.crq_audio_embeds = self.decoder.crq_audio_embeds.unsqueeze(0)
        
    def forward(
        self,
        crq_inputs_embeds: torch.FloatTensor,
        past_seq_length: torch.IntTensor,
        current_input_length: torch.IntTensor,
        attention_mask: Optional[torch.Tensor],
        past_key_cache: Optional[List[torch.Tensor]] = None,
        past_value_cache: Optional[List[torch.Tensor]] = None,
    ):
        crq_inputs_embeds = crq_inputs_embeds + self.decoder.crq_audio_embeds
        crq_inputs_embeds = self.decoder.input_matching(crq_inputs_embeds)

        outputs = self.decoder.crq_transformer(
            inputs_embeds=crq_inputs_embeds,
            attention_mask=attention_mask,
            past_seq_length=past_seq_length,
            current_input_length=current_input_length,
            past_key_cache=past_key_cache,
            past_value_cache=past_value_cache,
        )
        step_hidden = self.decoder.output_matching(outputs.last_hidden_state)
        logits = self.decoder.lm_head(step_hidden)

        return logits


FunAudioChatStaticDecoderWarp = FunAudioChatStaticDecoderPrefillWarp
