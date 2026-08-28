from __future__ import annotations

import torch
from torch import Tensor

from ...llm_data_processor import BaseInputProcessorConfig, BaseLLMInputProcessor


def aligned(size: int, alignment: int = 16) -> int:
    return ((size + alignment - 1) // alignment) * alignment


def _gen_causal_mask(
    inputs: Tensor,
    past_seq_length: int | Tensor,
    attention_max_length: int = -1,
) -> Tensor:
    if isinstance(past_seq_length, int):
        past_seq_length = torch.tensor([past_seq_length], device=inputs.device)
    past_seq_length = past_seq_length.reshape(-1)
    batch_size, query_length, key_length = inputs.shape
    masks = []
    for batch_index in range(batch_size):
        valid_length = int(past_seq_length[batch_index].item())
        if attention_max_length > 0:
            valid_length = min(valid_length, attention_max_length - 1)
        attention_mask = torch.tril(
            torch.ones(query_length, key_length, dtype=torch.bool, device=inputs.device),
            diagonal=valid_length,
        ).logical_not()
        if attention_max_length > 0:
            expired_window_mask = torch.tril(
                torch.ones_like(attention_mask),
                diagonal=valid_length - attention_max_length,
            )
            attention_mask = torch.where(expired_window_mask, True, attention_mask)
        masks.append(attention_mask.unsqueeze(0).unsqueeze(0))
    return torch.cat(masks, dim=0)


class LagunaDataPreprocess(BaseLLMInputProcessor):
    def __init__(
        self,
        config: BaseInputProcessorConfig,
        *,
        sliding_window: int,
    ):
        super().__init__(config)
        self.sliding_window = int(sliding_window)

    @staticmethod
    def _to_additive_mask(inputs: Tensor, mask: Tensor) -> Tensor:
        additive_mask = torch.zeros_like(mask, dtype=inputs.dtype, device=inputs.device)
        return additive_mask.masked_fill(mask, torch.finfo(inputs.dtype).min)

    def build_attention_mask(
        self,
        inputs_embeds: Tensor,
        past_seq_length: int | Tensor,
    ) -> Tensor:
        batch_size, query_length = inputs_embeds.shape[:2]
        sliding_context_length = aligned(self.sliding_window + query_length - 1)
        sliding_inputs = torch.empty(
            (batch_size, query_length, sliding_context_length),
            dtype=inputs_embeds.dtype,
            device=inputs_embeds.device,
        )
        sliding_attention_mask = self._to_additive_mask(
            inputs_embeds,
            _gen_causal_mask(
                sliding_inputs,
                past_seq_length,
                self.sliding_window,
            ),
        )
        return sliding_attention_mask

    def forward(self, data: dict | tuple | list) -> list[Tensor]:
        inputs_embeds, past_seq_length, current_input_length, past_key_caches, past_value_caches = super().forward(data)
        sliding_attention_mask = self.build_attention_mask(
            inputs_embeds,
            past_seq_length,
        )
        return [
            inputs_embeds,
            past_seq_length,
            current_input_length,
            sliding_attention_mask,
            past_key_caches,
            past_value_caches,
        ]


class LagunaSlidingMaskForwardMixin:
    uses_explicit_sliding_attention_mask: bool

    def forward(self, inputs_embeds, past_seq_length, current_input_length, *args, **kwargs):
        if (
            getattr(self, "uses_explicit_sliding_attention_mask", False)
            and len(args) >= 2
            and isinstance(args[0], (list, tuple))
            and isinstance(args[1], (list, tuple))
        ):
            data_processor = self.get_data_preprocessor()
            sliding_attention_mask = data_processor.build_attention_mask(
                inputs_embeds,
                past_seq_length,
            )
            args = (sliding_attention_mask, *args)
        return super().forward(
            inputs_embeds,
            past_seq_length,
            current_input_length,
            *args,
            **kwargs,
        )
