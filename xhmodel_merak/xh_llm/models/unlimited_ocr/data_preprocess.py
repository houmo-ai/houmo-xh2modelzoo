from __future__ import annotations

import math
from typing import Any, Optional

import torch
import torch.nn as nn

from ...llm_data_processor import BaseInputProcessorConfig, BaseLLMInputProcessor
from ...types import CacheList


class UnlimitedOCRDataPreprocess(BaseLLMInputProcessor):
    """Data processor for Unlimited-OCR base/no-crop multimodal prefill."""

    image_token = "<image>"

    def __init__(
        self,
        *,
        token_embedding: nn.Embedding,
        input_sequence_length: int = 256,
        past_key_caches=None,
        past_value_caches=None,
        image_token_id: int = 128815,
        image_size: int = 1024,
        patch_size: int = 16,
        downsample_ratio: int = 4,
        crop_mode: bool = False,
        pad_token_id: int = 0,
    ):
        config = BaseInputProcessorConfig(
            token_embedding,
            input_sequence_length,
            past_key_caches,
            past_value_caches,
            pad_token_id=0 if pad_token_id is None else pad_token_id,
        )
        super().__init__(config)
        self.token_embedding = token_embedding
        self.image_token_id = int(image_token_id)
        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        self.downsample_ratio = int(downsample_ratio)
        self.crop_mode = bool(crop_mode)
        self.last_position_ids: Optional[torch.Tensor] = None

    @property
    def image_grid_size(self) -> int:
        return math.ceil((self.image_size // self.patch_size) / self.downsample_ratio)

    @property
    def image_token_count(self) -> int:
        grid_size = self.image_grid_size
        return (grid_size + 1) * grid_size + 1

    def build_image_token_ids(self, num_images: int = 1) -> list[int]:
        if self.crop_mode:
            raise NotImplementedError(
                "Crop/gundam prompt building lives in XHUnlimitedOCRProcessor; "
                "this base helper only builds no-crop image tokens."
            )
        if num_images < 1:
            raise ValueError(f"num_images must be >= 1, got {num_images}.")
        return [self.image_token_id] * (self.image_token_count * num_images)

    def build_inputs_from_prompt(self, prompt: str, tokenizer: Any, num_images: int = 1) -> dict[str, torch.Tensor]:
        """Build base/no-crop input ids and masks from a prompt containing one or more <image> markers."""
        if tokenizer is None:
            raise ValueError("tokenizer must be provided when building inputs from prompt.")
        if self.crop_mode:
            raise NotImplementedError(
                "Crop/gundam prompt building lives in XHUnlimitedOCRProcessor.process()."
            )
        image_markers = prompt.count(self.image_token)
        if image_markers == 0 and num_images > 0:
            raise ValueError("prompt must contain '<image>' when num_images > 0.")
        if image_markers not in (1, num_images):
            raise ValueError(
                f"prompt/image count mismatch: found {image_markers} '<image>' markers for {num_images} image(s)."
            )

        text_splits = prompt.split(self.image_token)
        token_ids: list[int] = [0]
        images_seq_mask: list[bool] = [False]

        def _encode(text: str) -> list[int]:
            if hasattr(tokenizer, "encode"):
                return list(tokenizer.encode(text, add_special_tokens=False))
            return list(tokenizer(text, add_special_tokens=False)["input_ids"])

        for idx, text in enumerate(text_splits):
            text_ids = _encode(text)
            token_ids.extend(text_ids)
            images_seq_mask.extend([False] * len(text_ids))
            if idx < len(text_splits) - 1:
                image_tokens = self.build_image_token_ids(1 if image_markers == num_images else num_images)
                token_ids.extend(image_tokens)
                images_seq_mask.extend([True] * len(image_tokens))

        input_ids = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)
        images_seq_mask_tensor = torch.tensor(images_seq_mask, dtype=torch.bool).unsqueeze(0)
        images_spatial_crop = torch.ones(num_images, 2, dtype=torch.long)
        return {
            "input_ids": input_ids,
            "images_seq_mask": images_seq_mask_tensor,
            "images_spatial_crop": images_spatial_crop,
        }

    def _normalize_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if input_ids.shape[0] != 1:
            raise AssertionError("Batch size should be 1 in inference mode.")
        return input_ids.to(torch.long)

    def _normalize_image_embeds(self, image_embeds: torch.Tensor) -> torch.Tensor:
        if image_embeds.dim() == 3:
            image_embeds = image_embeds.reshape(-1, image_embeds.shape[-1])
        elif image_embeds.dim() != 2:
            raise ValueError(f"image_embeds should be rank 2 or 3, got shape {tuple(image_embeds.shape)}.")
        return image_embeds

    def _scatter_image_embeds(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        image_embeds: Optional[torch.Tensor],
        images_seq_mask: Optional[torch.Tensor],
        past_seq_length: int,
    ) -> torch.Tensor:
        if images_seq_mask is None and input_ids is not None:
            images_seq_mask = input_ids == self.image_token_id
        elif images_seq_mask is None:
            token_len = int(inputs_embeds.shape[1])
            images_seq_mask = torch.zeros(1, token_len, dtype=torch.bool, device=inputs_embeds.device)
        else:
            if images_seq_mask.dim() == 1:
                images_seq_mask = images_seq_mask.unsqueeze(0)
            images_seq_mask = images_seq_mask.to(torch.bool)

        if images_seq_mask.shape[1] < inputs_embeds.shape[1]:
            pad_len = inputs_embeds.shape[1] - images_seq_mask.shape[1]
            mask_padding = torch.zeros(1, pad_len, dtype=torch.bool, device=images_seq_mask.device)
            images_seq_mask = torch.cat([images_seq_mask, mask_padding], dim=1)
        elif images_seq_mask.shape[1] > inputs_embeds.shape[1]:
            raise ValueError(
                f"images_seq_mask length {images_seq_mask.shape[1]} exceeds input length {inputs_embeds.shape[1]}."
            )

        n_image_tokens = int(images_seq_mask.sum().item())
        if past_seq_length > 0:
            if image_embeds is not None or n_image_tokens > 0:
                raise ValueError("decode inputs must not carry image_embeds or image tokens.")
            return inputs_embeds

        if image_embeds is None:
            if n_image_tokens > 0:
                raise ValueError(f"image_embeds is required for {n_image_tokens} image token(s).")
            return inputs_embeds

        image_embeds = self._normalize_image_embeds(image_embeds)
        n_image_features = int(image_embeds.shape[0])
        if n_image_tokens != n_image_features:
            raise ValueError(
                "Unlimited-OCR image features and image tokens do not match: "
                f"tokens={n_image_tokens}, features={n_image_features}."
            )

        image_mask = images_seq_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        return inputs_embeds.masked_scatter(image_mask, image_embeds)

    def make_position_ids(self, seq_length: int, past_seq_length: int, device: torch.device) -> torch.Tensor:
        return torch.arange(past_seq_length, past_seq_length + seq_length, dtype=torch.long, device=device)

    def forward(self, data: dict | tuple | list):
        assert isinstance(data, dict)
        if data.get("input_ids", None) is None and data.get("prompt", None) is not None:
            built = self.build_inputs_from_prompt(
                data["prompt"],
                data.get("tokenizer"),
                int(data.get("num_images", 1)),
            )
            data = {**built, **data}

        input_ids = data.get("input_ids", None)
        inputs_embeds = data.get("inputs_embeds", None)
        assert (input_ids is not None) or (inputs_embeds is not None), (
            "Either input_ids or inputs_embeds should be provided."
        )

        past_seq_length = int(data["past_seq_length"])
        if past_seq_length < 0:
            raise AssertionError("past_seq_length should be non-negative.")

        if input_ids is not None:
            input_ids = self._normalize_input_ids(input_ids)
            seq_length = int(input_ids.shape[1])
            if seq_length > self.input_sequence_length:
                raise AssertionError(
                    f"Input sequence length is too long. max input sequence length is "
                    f"{self.input_sequence_length} but got {seq_length}"
                )
            if self.input_sequence_length > seq_length:
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
                raise AssertionError("Batch size should be 1 in inference mode.")
            seq_length = int(inputs_embeds.shape[1])
            if seq_length > self.input_sequence_length:
                raise AssertionError(
                    f"Input sequence length is too long. max input sequence length is "
                    f"{self.input_sequence_length} but got {seq_length}"
                )
            if self.input_sequence_length > seq_length:
                padding_ids = torch.full(
                    (1, self.input_sequence_length - seq_length),
                    self.pad_token_id,
                    dtype=torch.long,
                    device=inputs_embeds.device,
                )
                padding_embedding = self.embed_tokens(padding_ids)
                inputs_embeds = torch.cat([inputs_embeds, padding_embedding], dim=1)

        assert inputs_embeds is not None
        inputs_embeds = self._scatter_image_embeds(
            input_ids,
            inputs_embeds,
            data.get("image_embeds", None),
            data.get("images_seq_mask", None),
            past_seq_length,
        )
        self.last_position_ids = self.make_position_ids(seq_length, past_seq_length, inputs_embeds.device)

        past_key_caches = self.past_key_caches
        past_value_caches = self.past_value_caches
        assert past_key_caches is not None and past_value_caches is not None, "KV cache is not available."
        assert isinstance(past_key_caches, CacheList) and isinstance(past_value_caches, CacheList), (
            "KV cache should be of type CacheList."
        )
        return (
            inputs_embeds.to(self._device),
            torch.tensor([past_seq_length], dtype=torch.int32, device=self._device),
            torch.tensor([seq_length], dtype=torch.int32, device=self._device),
            past_key_caches,
            past_value_caches,
        )
