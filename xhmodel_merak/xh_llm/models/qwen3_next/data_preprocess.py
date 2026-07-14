from __future__ import annotations

import torch
import torch.nn as nn

from ...llm_data_processor import BaseInputProcessorConfig, BaseLLMInputProcessor
from ...types import CacheList
from ..qwen3_5.split_conv_cache_utils import _flatten_split_conv_cache_outputs


class Qwen3NextDataPreprocess(BaseLLMInputProcessor):
    """Prepare text and hybrid full/GDN cache inputs for Qwen3-Next."""

    def __init__(
        self,
        *,
        token_embedding: nn.Embedding,
        input_sequence_length: int,
        past_key_caches=None,
        past_value_caches=None,
        past_conv_caches=None,
        past_recurrent_states=None,
        enable_page_attention: bool = False,
        pad_token_id: int = 0,
    ):
        config = BaseInputProcessorConfig(
            token_embedding,
            input_sequence_length,
            past_key_caches,
            past_value_caches,
            enable_page_attention=enable_page_attention,
            pad_token_id=pad_token_id,
        )
        super().__init__(config)
        self.past_conv_caches = past_conv_caches
        self.past_recurrent_states = past_recurrent_states

    def forward(self, data: dict | tuple | list):
        if not isinstance(data, dict):
            raise TypeError(f"Qwen3NextDataPreprocess expects dict input, got {type(data)}")
        input_ids = data.get("input_ids")
        inputs_embeds = data.get("inputs_embeds")
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")

        if input_ids is not None:
            if input_ids.shape[0] != 1:
                raise ValueError("Qwen3-Next inference currently supports batch size 1")
            seq_length = input_ids.shape[1]
            if seq_length > self.input_sequence_length:
                raise ValueError(
                    f"Input sequence length {seq_length} exceeds graph length {self.input_sequence_length}"
                )
            if seq_length < self.input_sequence_length:
                padding = torch.full(
                    (1, self.input_sequence_length - seq_length),
                    self.pad_token_id,
                    dtype=torch.long,
                    device=input_ids.device,
                )
                input_ids = torch.cat([input_ids, padding], dim=-1)
            inputs_embeds = self.embed_tokens(input_ids)
        else:
            if inputs_embeds.shape[0] != 1:
                raise ValueError("Qwen3-Next inference currently supports batch size 1")
            seq_length = inputs_embeds.shape[1]
            if seq_length > self.input_sequence_length:
                raise ValueError(
                    f"Input sequence length {seq_length} exceeds graph length {self.input_sequence_length}"
                )
            if seq_length < self.input_sequence_length:
                padding_ids = torch.full(
                    (1, self.input_sequence_length - seq_length),
                    self.pad_token_id,
                    dtype=torch.long,
                    device=inputs_embeds.device,
                )
                inputs_embeds = torch.cat([inputs_embeds, self.embed_tokens(padding_ids)], dim=1)

        linear_mask = torch.zeros(
            1, self.input_sequence_length, dtype=torch.float16, device=inputs_embeds.device
        )
        linear_mask[:, :seq_length] = 1
        past_seq_length = int(data.get("past_seq_length", 0))
        if past_seq_length < 0:
            raise ValueError("past_seq_length must be non-negative")
        prefix = (
            inputs_embeds.to(self._device),
            torch.tensor([past_seq_length], dtype=torch.int32, device=self._device),
            torch.tensor([seq_length], dtype=torch.int32, device=self._device),
            linear_mask.to(self._device),
        )
        if not isinstance(self.past_conv_caches, CacheList):
            raise ValueError("past_conv_caches must be CacheList")
        if not isinstance(self.past_recurrent_states, CacheList):
            raise ValueError("past_recurrent_states must be CacheList")
        conv_caches = _flatten_split_conv_cache_outputs(self.past_conv_caches)
        # Page attention replaces only ordinary KV-cache inputs. GDN cache
        # inputs remain explicit because they are recurrent states, not pages.
        if self.enable_page_attention:
            return (*prefix, conv_caches, self.past_recurrent_states)
        return (
            *prefix,
            self.past_key_caches,
            self.past_value_caches,
            conv_caches,
            self.past_recurrent_states,
        )
