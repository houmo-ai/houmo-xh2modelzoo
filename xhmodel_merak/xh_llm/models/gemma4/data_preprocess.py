from __future__ import annotations

import torch
import torch.nn as nn

from ...llm_data_processor import BaseInputProcessorConfig, BaseLLMInputProcessor


class Gemma4DataPreprocess(BaseLLMInputProcessor):
    def __init__(
        self,
        *,
        token_embedding: nn.Embedding,
        input_sequence_length: int,
        context_length: int,
        past_key_caches=None,
        past_value_caches=None,
        pad_token_id: int = 0,
        image_token_id: int = -1,
        sliding_window: int = 1024,
    ):
        super().__init__(
            BaseInputProcessorConfig(
                embed_tokens=token_embedding,
                input_sequence_length=input_sequence_length,
                past_key_caches=past_key_caches,
                past_value_caches=past_value_caches,
                pad_token_id=pad_token_id,
            )
        )
        self.token_embedding = token_embedding
        self.context_length = context_length
        self.image_token_id = image_token_id
        self.sliding_window = sliding_window

    def _pad_1d(self, tensor: torch.Tensor, value: int = 0):
        if tensor.shape[-1] >= self.input_sequence_length:
            return tensor[..., : self.input_sequence_length]
        pad = torch.full(
            (*tensor.shape[:-1], self.input_sequence_length - tensor.shape[-1]),
            value,
            dtype=tensor.dtype,
            device=tensor.device,
        )
        return torch.cat([tensor, pad], dim=-1)

    def _build_attention_masks(
        self,
        current_input_length: int,
        past_seq_length: int,
        mm_token_type_ids: torch.Tensor,
        device: torch.device,
    ):
        q_len = self.input_sequence_length
        ctx = self.context_length
        neg = torch.tensor(torch.finfo(torch.float16).min, dtype=torch.float16, device=device)
        full_mask = torch.full((1, 1, q_len, ctx), neg, dtype=torch.float16, device=device)
        sliding_mask = torch.full((1, 1, q_len, ctx), neg, dtype=torch.float16, device=device)

        valid_k = min(ctx, max(1, past_seq_length + current_input_length))
        for q in range(q_len):
            if q < current_input_length:
                abs_q = past_seq_length + q
                full_end = min(valid_k, abs_q + 1)
                full_mask[0, 0, q, :full_end] = 0

                slide_start = max(0, abs_q - self.sliding_window + 1)
                slide_end = full_end
                sliding_mask[0, 0, q, slide_start:slide_end] = 0
            else:
                full_mask[0, 0, q, 0] = 0
                sliding_mask[0, 0, q, 0] = 0

        if mm_token_type_ids.numel() > 0:
            mm = mm_token_type_ids[:current_input_length]
            is_vision = (mm == 1) | (mm == 2)
            group_start = None
            for idx in range(current_input_length):
                if bool(is_vision[idx]) and group_start is None:
                    group_start = idx
                if group_start is not None and (idx == current_input_length - 1 or not bool(is_vision[idx + 1])):
                    group_end = idx + 1
                    abs_start = past_seq_length + group_start
                    abs_end = past_seq_length + group_end
                    full_mask[0, 0, group_start:group_end, abs_start:abs_end] = 0
                    sliding_mask[0, 0, group_start:group_end, abs_start:abs_end] = 0
                    group_start = None

        return full_mask, sliding_mask

    def forward(self, data: dict | tuple | list):
        assert isinstance(data, dict)
        device = self._device
        input_ids = data["input_ids"].to(device)
        assert input_ids.shape[0] == 1, "Batch size should be 1 in inference mode."
        seq_length = input_ids.shape[1]
        assert seq_length <= self.input_sequence_length, (
            f"Input sequence length is too long. max input sequence length is {self.input_sequence_length} but got {seq_length}"
        )
        input_ids = self._pad_1d(input_ids, value=self.pad_token_id)
        inputs_embeds = self.token_embedding.to(device)(input_ids)

        image_embeds = data.get("image_embeds", None)
        if image_embeds is not None:
            n_image_tokens = (input_ids == self.image_token_id).sum().item()
            if n_image_tokens > 0:
                image_mask = (input_ids == self.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                if image_embeds.shape[0] != n_image_tokens:
                    raise ValueError(
                        f"Image features and image tokens do not match: tokens={n_image_tokens}, features={image_embeds.shape[0]}"
                    )
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        mm_token_type_ids = data.get("mm_token_type_ids", None)
        if mm_token_type_ids is None:
            mm_token_type_ids = torch.zeros((1, seq_length), dtype=torch.long, device=device)
        else:
            mm_token_type_ids = mm_token_type_ids.to(device)
        mm_token_type_ids = self._pad_1d(mm_token_type_ids, value=0)[0]

        past_seq_length = int(data["past_seq_length"])
        current_input_length = int(seq_length)

        full_attention_mask, sliding_attention_mask = self._build_attention_masks(
            current_input_length=current_input_length,
            past_seq_length=past_seq_length,
            mm_token_type_ids=mm_token_type_ids,
            device=device,
        )

        return (
            inputs_embeds,
            torch.tensor([past_seq_length], dtype=torch.int32, device=device),
            torch.tensor([current_input_length], dtype=torch.int32, device=device),
            full_attention_mask,
            sliding_attention_mask,
            self.past_key_caches,
            self.past_value_caches,
        )
