from typing import Union, cast

import torch
from torch import Tensor

from ..configuration_utils import BaseConfig
from .types import CacheList


class BaseInputProcessorConfig(BaseConfig):
    def __init__(
        self, embed_tokens, input_sequence_length: int = 2048, past_key_caches=None, past_value_caches=None, **kwargs
    ):
        super().__init__()
        self.input_sequence_length = input_sequence_length
        self.past_key_caches = past_key_caches
        self.past_value_caches = past_value_caches
        self.enable_page_attention = kwargs.get("enable_page_attention", False)
        if "pad_token_id" in kwargs:
            self.pad_token_id = kwargs["pad_token_id"]
        else:
            self.pad_token_id = 0
        self.embed_tokens = embed_tokens


class BaseLLMInputProcessor:
    def __init__(self, config: BaseInputProcessorConfig):
        self.config = config
        self.input_sequence_length = config.input_sequence_length
        self.past_key_caches = config.past_key_caches
        self.past_value_caches = config.past_value_caches
        self.enable_page_attention = config.enable_page_attention
        self.embed_tokens = config.embed_tokens
        self.pad_token_id = config.pad_token_id

        self._device = torch.device("cpu")
        self._dtype = torch.float16

    def set_page_attention_context(
        self,
        past_key_caches,
        past_value_caches,
        block_ids,
        slot_mapping,
        block_size: int,
    ) -> None:
        self.enable_page_attention = True
        self.past_key_caches = past_key_caches
        self.past_value_caches = past_value_caches
        self.page_attention_block_ids = block_ids
        self.page_attention_slot_mapping = slot_mapping
        self.page_attention_block_size = block_size

    def to(self, *args, **kwargs) -> "BaseLLMInputProcessor":
        device, dtype = torch._C._nn._parse_to(*args, **kwargs)[:2]
        if device is not None:
            self._device = device
        if dtype is not None:
            self._dtype = dtype

        return self

    def __call__(self, data: dict | tuple | list) -> list[torch.Tensor]:
        if isinstance(data, dict):
            data = {k: v.to(self._device) if isinstance(v, torch.Tensor) else v for k, v in data.items()}
        elif isinstance(data, (tuple, list)):
            data = type(data)([d.to(self._device) if isinstance(d, torch.Tensor) else d for d in data])

        return self.forward(data)

    def forward(self, data: dict | tuple | list) -> list[torch.Tensor]:
        assert isinstance(data, dict)
        input_ids = data.get("input_ids", None)
        inputs_embeds = data.get("inputs_embeds", None)
        assert (input_ids is not None) or (inputs_embeds is not None), (
            "Either input_ids or inputs_embeds should be provided, but not both."
        )
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

        assert self.embed_tokens is not None, "Token embedding is not available."

        past_seq_length = data["past_seq_length"]
        assert past_seq_length >= 0, "past_seq_length should be non-negative."
        if self.enable_page_attention:
            return (
                inputs_embeds,
                torch.tensor([past_seq_length], dtype=torch.int32, device=self._device),
                torch.tensor([seq_length], dtype=torch.int32, device=self._device),
            )
        past_key_caches = self.past_key_caches
        past_value_caches = self.past_value_caches
        assert past_key_caches is not None and past_value_caches is not None, "KV cache is not available."
        assert isinstance(past_key_caches, CacheList) and isinstance(past_value_caches, CacheList), (
            "KV cache should be of type CacheList."
        )
        return (
            inputs_embeds,
            torch.tensor([past_seq_length], dtype=torch.int32, device=self._device),
            torch.tensor([seq_length], dtype=torch.int32, device=self._device),
            past_key_caches,
            past_value_caches,
        )


class BaseVisualProcessor:
    def __init__(self) -> None:
        self._device = torch.device("cpu")
        self._dtype = torch.float16

    def to(self, *args, **kwargs) -> "BaseLLMInputProcessor":
        device, dtype = torch._C._nn._parse_to(*args, **kwargs)[:2]
        if device is not None:
            self._device = device
        if dtype is not None:
            self._dtype = dtype

        return self

    def __call__(self, data: dict) -> list[torch.Tensor]:
        return self.forward(data)

    def forward(self, data: dict) -> list[torch.Tensor]:
        assert isinstance(data, dict) and "image" in data, "Input data should be a dictionary with key 'image'."
        return (data["image"],)


def _prepare_window_attention_mask(inputs_tensor: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
    nq, nk = inputs_tensor.size(-2), inputs_tensor.size(-1)
    attention_mask = torch.ones(
        [1, nq, nk],
        device=inputs_tensor.device,
        dtype=torch.bool,
    )
    for i in range(1, len(cu_seqlens)):
        attention_mask[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = 0
    return attention_mask


def _gen_mask_v2(x: Tensor, valid_length: Union[int, Tensor], attention_max_length: int = -1):
    if isinstance(valid_length, int):
        valid_length: Tensor = torch.tensor(valid_length).to(x.device)
    valid_length = valid_length.reshape(-1)
    if x.shape[0] != valid_length.size().numel() or (valid_length[0].item() == 0 and valid_length.shape[0] == 2):
        return _prepare_window_attention_mask(x, valid_length)
    # assert (
    #     x.shape[0] == valid_length.size().numel()
    # ), f"x batch size = {x.shape[0]}, but valid_length size = {valid_length.size().numel()}"

    bsz, nq, nk = x.size(0), x.size(-2), x.size(-1)
    bsz_attention_mask = []
    for i in range(bsz):
        b_valid_length = int(valid_length[i].item())
        if attention_max_length > 0:
            b_valid_length = min(b_valid_length, attention_max_length - 1)
        attention_mask = torch.tril(
            torch.ones(nq, nk, dtype=torch.bool, device=x.device), diagonal=b_valid_length
        ).logical_not()
        if attention_max_length > 0:
            sliding_window_mask = torch.tril(
                torch.ones_like(attention_mask, dtype=torch.bool), diagonal=b_valid_length - attention_max_length
            )
            attention_mask = torch.where(sliding_window_mask, True, attention_mask)
        bsz_attention_mask.append(attention_mask.unsqueeze(dim=0).unsqueeze(dim=0))

    mask = torch.cat(bsz_attention_mask, dim=0)
    return mask


def aligned(size, align):
    return ((size + align - 1) // align) * align


class SlidingWindowInputProcessorConfig(BaseInputProcessorConfig):
    def __init__(
        self,
        embed_tokens,
        input_sequence_length: int = 2048,
        past_key_caches=None,
        past_value_caches=None,
        sliding_window: int = -1,
        **kwargs,
    ):
        super().__init__(embed_tokens, input_sequence_length, past_key_caches, past_value_caches, **kwargs)
        self.sliding_window = sliding_window


class SlidingWindowLLMInputProcessor(BaseLLMInputProcessor):
    def __init__(self, config: SlidingWindowInputProcessorConfig):
        super().__init__(config)

    def prepare_casual_mask(self, x: Tensor, valid_length: int, attention_max_length: int):
        mask = _gen_mask_v2(x, valid_length, attention_max_length)
        attention_mask = torch.zeros_like(mask, dtype=x.dtype, device=x.device)

        attention_mask = attention_mask.masked_fill(mask, torch.finfo(x.dtype).min)
        return attention_mask

    def forward(self, data: dict | tuple | list) -> list[torch.Tensor]:
        inputs_embeds, past_seq_length, seq_length, past_key_caches, past_value_caches = super().forward(data)
        bz, nq = inputs_embeds.shape[:2]
        config = cast(SlidingWindowInputProcessorConfig, self.config)
        local_attention_window_size = config.sliding_window + nq - 1
        local_attention_window_size = aligned(local_attention_window_size, 16)
        x = torch.empty((bz, nq, local_attention_window_size), dtype=inputs_embeds.dtype, device=inputs_embeds.device)
        sliding_attention_mask = self.prepare_casual_mask(x, past_seq_length, config.sliding_window)
        return (
            inputs_embeds,
            past_seq_length,
            seq_length,
            sliding_attention_mask,
            past_key_caches,
            past_value_caches,
        )
