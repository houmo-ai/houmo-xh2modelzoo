import torch

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
        self.embed_tokens = config.embed_tokens
        self.pad_token_id = config.pad_token_id

        self._device = torch.device("cpu")
        self._dtype = torch.float16

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
                padding_input_ids = torch.zeros((1, self.input_sequence_length - seq_length), dtype=torch.long)
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
                padding_input_ids = (
                    torch.ones((1, self.input_sequence_length - seq_length), dtype=torch.long) * padding_token_id
                )
                padding_embedding = self.embed_tokens(padding_input_ids)
                inputs_embeds = torch.cat([inputs_embeds, padding_embedding], dim=1)

        assert self.embed_tokens is not None, "Token embedding is not available."

        past_seq_length = data["past_seq_length"]
        assert past_seq_length >= 0, "past_seq_length should be non-negative."
        past_key_caches = self.past_key_caches
        past_value_caches = self.past_value_caches
        assert past_key_caches is not None and past_value_caches is not None, "KV cache is not available."
        assert isinstance(past_key_caches, CacheList) and isinstance(past_value_caches, CacheList), (
            "KV cache should be of type CacheList."
        )
        return (
            # position_ids.to(self.device),
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
