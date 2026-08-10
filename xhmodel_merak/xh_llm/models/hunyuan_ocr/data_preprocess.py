# Copyright (C) 2026 THL A29 Limited, a Tencent company and the HuggingFace Inc. team.
# Copyright 2025 HOUMO AI (modifications)
# Licensed under the Apache License, Version 2.0.
# See licenses/transformers/LICENSE for the full license text.
# Source: https://github.com/huggingface/transformers/blob/v5.13.0/src/transformers/models/hunyuan_vl/modular_hunyuan_vl.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from ...llm_data_processor import BaseInputProcessorConfig, BaseLLMInputProcessor
from ...types import CacheList


@dataclass(frozen=True)
class HunyuanOCRPrefillPlan:
    inputs_embeds: Tensor
    position_ids: Tensor
    valid_length: int
    rope_delta: Tensor


class HunyuanOCRTextDataPreprocess(BaseLLMInputProcessor):
    """Prepare the explicit four-axis input contract for the text backend."""

    NUM_POSITION_AXES = 4

    def __init__(
        self,
        *,
        token_embedding: nn.Embedding,
        input_sequence_length: int,
        context_max_length: int,
        past_key_caches: CacheList | None,
        past_value_caches: CacheList | None,
        pad_token_id: int,
        image_token_id: int = 120120,
        image_start_token_id: int = 120118,
        image_end_token_id: int = 120119,
        spatial_merge_size: int = 2,
    ) -> None:
        super().__init__(
            BaseInputProcessorConfig(
                embed_tokens=token_embedding,
                input_sequence_length=input_sequence_length,
                past_key_caches=past_key_caches,
                past_value_caches=past_value_caches,
                pad_token_id=pad_token_id,
            )
        )
        self.context_max_length = int(context_max_length)
        self.image_token_id = int(image_token_id)
        self.image_start_token_id = int(image_start_token_id)
        self.image_end_token_id = int(image_end_token_id)
        self.spatial_merge_size = int(spatial_merge_size)
        self.rope_deltas: Tensor | None = None
        if self.spatial_merge_size <= 0:
            raise ValueError(f"spatial_merge_size must be positive, got {self.spatial_merge_size}")

    def _multimodal_positions_and_spans(
        self,
        input_ids: Tensor,
        mm_token_type_ids: Tensor,
        image_grid_thw: Tensor | None,
    ) -> tuple[Tensor, list[tuple[int, int]], list[int]]:
        mm_token_type_ids = torch.as_tensor(mm_token_type_ids, dtype=torch.long, device=input_ids.device)
        if mm_token_type_ids.shape != input_ids.shape:
            raise ValueError(
                "mm_token_type_ids must match input_ids shape, "
                f"got {tuple(mm_token_type_ids.shape)} and {tuple(input_ids.shape)}"
            )
        if not bool(((mm_token_type_ids == 0) | (mm_token_type_ids == 1)).all()):
            raise ValueError("mm_token_type_ids must contain only 0 for text or 1 for image placeholders")

        image_mask = mm_token_type_ids[0].bool()
        transitions = torch.diff(
            torch.cat(
                (
                    torch.zeros(1, dtype=torch.int8),
                    image_mask.cpu().to(torch.int8),
                    torch.zeros(1, dtype=torch.int8),
                )
            )
        )
        boundaries = transitions.nonzero().flatten().tolist()
        spans = list(zip(boundaries[::2], boundaries[1::2], strict=True))
        if bool(((input_ids[0] == self.image_token_id) & ~image_mask).any()):
            raise ValueError("Found image token outside an mm_token_type_ids image placeholder span")

        if image_grid_thw is None:
            if spans:
                raise ValueError("image_grid_thw is required when image placeholder spans are present")
            grids = torch.empty((0, 3), dtype=torch.long, device=input_ids.device)
        else:
            grids = torch.as_tensor(image_grid_thw, dtype=torch.long, device=input_ids.device)
            if grids.ndim != 2 or grids.shape[1] != 3:
                raise ValueError(f"image_grid_thw must have shape [num_images, 3], got {tuple(grids.shape)}")
        if len(spans) != len(grids):
            raise ValueError(
                "Image placeholder span count does not match image_grid_thw: "
                f"spans={len(spans)}, images={len(grids)}"
            )

        positions = torch.arange(input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        positions = positions.view(1, 1, -1).expand(self.NUM_POSITION_AXES, 1, -1).clone()
        expected_counts: list[int] = []
        for image_index, ((span_start, span_end), grid) in enumerate(zip(spans, grids, strict=True)):
            temporal, grid_h, grid_w = (int(value) for value in grid)
            if temporal != 1 or grid_h <= 0 or grid_w <= 0:
                raise ValueError(
                    f"image_grid_thw[{image_index}] must be [1, positive_h, positive_w], got {grid.tolist()}"
                )
            if grid_h % self.spatial_merge_size or grid_w % self.spatial_merge_size:
                raise ValueError(
                    f"image_grid_thw[{image_index}] spatial dimensions must be divisible by "
                    f"spatial_merge_size={self.spatial_merge_size}, got {grid_h}x{grid_w}"
                )
            if span_start == 0 or span_end >= input_ids.shape[1]:
                raise ValueError(f"Image placeholder span {image_index} is missing image start/end wrapper tokens")
            if int(input_ids[0, span_start - 1]) != self.image_start_token_id:
                raise ValueError(f"Image placeholder span {image_index} is missing image start token")
            if int(input_ids[0, span_end]) != self.image_end_token_id:
                raise ValueError(f"Image placeholder span {image_index} is missing image end token")
            if not bool((input_ids[0, span_start:span_end] == self.image_token_id).all()):
                raise ValueError(f"Image placeholder span {image_index} contains a non-image token")

            merged_h = grid_h // self.spatial_merge_size
            merged_w = grid_w // self.spatial_merge_size
            grid_token_count = merged_h * (merged_w + 1)
            expected_count = grid_token_count + 2
            span_length = span_end - span_start
            if span_length != expected_count:
                raise ValueError(
                    f"Image placeholder span {image_index} length does not match grid: "
                    f"span={span_length}, expected={expected_count}"
                )
            expected_counts.append(expected_count)

            height, width = torch.meshgrid(
                torch.arange(merged_h, device=input_ids.device),
                torch.arange(merged_w + 1, device=input_ids.device),
                indexing="ij",
            )
            grid_start = span_start + 1
            grid_end = span_end - 1
            positions[1, 0, grid_start:grid_end] = width.flatten()
            positions[2, 0, grid_start:grid_end] = height.flatten()
            positions[3, 0, grid_start:grid_end] = image_index
        image_start_count = int((input_ids == self.image_start_token_id).sum().item())
        image_end_count = int((input_ids == self.image_end_token_id).sum().item())
        if image_start_count != len(spans) or image_end_count != len(spans):
            raise ValueError(
                "Image wrapper token count does not match image placeholder spans: "
                f"spans={len(spans)}, starts={image_start_count}, ends={image_end_count}"
            )
        return positions, spans, expected_counts

    @staticmethod
    def _scalar_length(value: Any, field: str) -> int:
        tensor = torch.as_tensor(value).flatten()
        if tensor.numel() != 1:
            raise ValueError(f"{field} must contain exactly one value, got shape {tuple(tensor.shape)}")
        result = int(tensor.item())
        if result < 0:
            raise ValueError(f"{field} must be non-negative, got {result}")
        return result

    def _normalize_position_ids(
        self,
        position_ids: Tensor | None,
        *,
        past_seq_length: int,
        physical_length: int,
        valid_length: int,
        device: torch.device,
    ) -> Tensor:
        if position_ids is None:
            positions = torch.arange(
                past_seq_length,
                past_seq_length + physical_length,
                dtype=torch.long,
                device=device,
            ).view(1, 1, physical_length)
            return positions.expand(self.NUM_POSITION_AXES, 1, physical_length).clone()

        position_ids = torch.as_tensor(position_ids, dtype=torch.long, device=device)
        if position_ids.ndim == 2 and position_ids.shape[0] == self.NUM_POSITION_AXES:
            position_ids = position_ids.unsqueeze(1)
        if position_ids.ndim != 3 or position_ids.shape[:2] != (self.NUM_POSITION_AXES, 1):
            raise ValueError(
                "position_ids must have shape [4, 1, sequence] or [4, sequence], "
                f"got {tuple(position_ids.shape)}"
            )
        if position_ids.shape[-1] not in (valid_length, physical_length):
            raise ValueError(
                "position_ids length must match current_input_length or graph input length, "
                f"got {position_ids.shape[-1]}, {valid_length}, and {physical_length}"
            )
        if position_ids.shape[-1] < physical_length:
            padding = torch.zeros(
                self.NUM_POSITION_AXES,
                1,
                physical_length - position_ids.shape[-1],
                dtype=position_ids.dtype,
                device=device,
            )
            position_ids = torch.cat((position_ids, padding), dim=-1)
        return position_ids

    def build_prefill_plan(self, data: dict[str, Any]) -> HunyuanOCRPrefillPlan:
        """Build the complete unpadded semantic prefill before graph chunking."""

        input_ids = data.get("input_ids")
        inputs_embeds = data.get("inputs_embeds")
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")
        if input_ids is not None:
            input_ids = torch.as_tensor(input_ids, device=self._device)
            if input_ids.ndim != 2 or input_ids.shape[0] != 1:
                raise ValueError(f"input_ids must have shape [1, sequence], got {tuple(input_ids.shape)}")
            inputs_embeds = self.embed_tokens(input_ids)
        else:
            inputs_embeds = torch.as_tensor(inputs_embeds, device=self._device)
            if inputs_embeds.ndim != 3 or inputs_embeds.shape[0] != 1:
                raise ValueError(
                    f"inputs_embeds must have shape [1, sequence, hidden], got {tuple(inputs_embeds.shape)}"
                )
        valid_length = self._scalar_length(
            data.get("current_input_length", inputs_embeds.shape[1]),
            "current_input_length",
        )
        if valid_length != inputs_embeds.shape[1] or valid_length > self.context_max_length:
            raise ValueError("A prefill plan must contain the complete input within context_max_length")
        requested_position_ids = data.get("position_ids")
        image_embeds = data.get("image_embeds")
        mm_token_type_ids = data.get("mm_token_type_ids")
        image_grid_thw = data.get("image_grid_thw")
        expected_image_counts: list[int] = []
        if any(value is not None for value in (image_embeds, mm_token_type_ids, image_grid_thw)):
            if input_ids is None or mm_token_type_ids is None:
                raise ValueError("Multimodal prefill requires input_ids and mm_token_type_ids")
            generated_positions, _, expected_image_counts = self._multimodal_positions_and_spans(
                input_ids,
                mm_token_type_ids,
                image_grid_thw,
            )
            if requested_position_ids is None:
                requested_position_ids = generated_positions
        expected_image_tokens = sum(expected_image_counts)
        if expected_image_tokens:
            image_embeds = torch.as_tensor(image_embeds, device=inputs_embeds.device)
            if image_embeds.ndim == 3 and image_embeds.shape[0] == 1:
                image_embeds = image_embeds[0]
            expected_shape = (expected_image_tokens, inputs_embeds.shape[-1])
            if tuple(image_embeds.shape) != expected_shape:
                raise ValueError(
                    "Image features and image placeholders do not match: "
                    f"expected={expected_shape}, features={tuple(image_embeds.shape)}"
                )
            image_mask = (input_ids == self.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds.to(inputs_embeds.dtype))
        elif image_embeds is not None:
            raise ValueError("image_embeds were provided but no image placeholder span was found")
        if requested_position_ids is None:
            requested_position_ids = torch.arange(
                valid_length,
                dtype=torch.long,
                device=self._device,
            ).view(1, 1, valid_length).expand(self.NUM_POSITION_AXES, 1, valid_length)
        position_ids = self._normalize_position_ids(
            requested_position_ids,
            past_seq_length=0,
            physical_length=valid_length,
            valid_length=valid_length,
            device=self._device,
        )
        rope_delta = int(position_ids[:, :, :valid_length].max().item()) + 1 - valid_length if valid_length else 0
        return HunyuanOCRPrefillPlan(
            inputs_embeds=inputs_embeds.to(dtype=self._dtype),
            position_ids=position_ids,
            valid_length=valid_length,
            rope_delta=torch.tensor([[rope_delta]], dtype=torch.long, device=self._device),
        )

    def forward(self, data: dict | tuple | list):
        if not isinstance(data, dict):
            raise TypeError(f"HunyuanOCR text inputs must be a dictionary, got {type(data).__name__}")
        input_ids = data.get("input_ids")
        inputs_embeds = data.get("inputs_embeds")
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")

        if input_ids is not None:
            input_ids = input_ids.to(self._device)
            if input_ids.ndim != 2 or input_ids.shape[0] != 1:
                raise ValueError(f"input_ids must have shape [1, sequence], got {tuple(input_ids.shape)}")
            source_length = int(input_ids.shape[1])
        else:
            inputs_embeds = inputs_embeds.to(self._device)
            if inputs_embeds.ndim != 3 or inputs_embeds.shape[0] != 1:
                raise ValueError(
                    f"inputs_embeds must have shape [1, sequence, hidden], got {tuple(inputs_embeds.shape)}"
                )
            source_length = int(inputs_embeds.shape[1])

        valid_length = self._scalar_length(data.get("current_input_length", source_length), "current_input_length")
        if valid_length > source_length:
            raise ValueError(
                f"current_input_length ({valid_length}) exceeds provided sequence length ({source_length})"
            )
        if source_length > self.input_sequence_length:
            raise ValueError(
                f"Input sequence length {source_length} exceeds graph input length {self.input_sequence_length}"
            )
        past_seq_length = self._scalar_length(data.get("past_seq_length", 0), "past_seq_length")
        if past_seq_length + valid_length > self.context_max_length:
            raise ValueError(
                f"past_seq_length + current_input_length ({past_seq_length + valid_length}) "
                f"exceeds context_max_length ({self.context_max_length})"
            )

        requested_position_ids = data.get("position_ids")
        caller_supplied_position_ids = requested_position_ids is not None
        image_embeds = data.get("image_embeds")
        mm_token_type_ids = data.get("mm_token_type_ids")
        image_grid_thw = data.get("image_grid_thw")
        has_multimodal_contract = any(value is not None for value in (image_embeds, mm_token_type_ids, image_grid_thw))
        expected_image_counts: list[int] = []
        if has_multimodal_contract:
            if input_ids is None:
                raise ValueError("Multimodal prefill requires input_ids so placeholder spans can be validated")
            if past_seq_length != 0:
                raise ValueError("Decode inputs must not carry image embeddings, grids, or multimodal token types")
            if mm_token_type_ids is None:
                raise ValueError("mm_token_type_ids is required for multimodal prefill")
            generated_positions, _, expected_image_counts = self._multimodal_positions_and_spans(
                input_ids,
                mm_token_type_ids,
                image_grid_thw,
            )
            if requested_position_ids is None:
                requested_position_ids = generated_positions

        if input_ids is not None:
            if source_length < self.input_sequence_length:
                padding = torch.full(
                    (1, self.input_sequence_length - source_length),
                    self.pad_token_id,
                    dtype=input_ids.dtype,
                    device=input_ids.device,
                )
                input_ids = torch.cat((input_ids, padding), dim=1)
            inputs_embeds = self.embed_tokens(input_ids)
        elif source_length < self.input_sequence_length:
            padding_ids = torch.full(
                (1, self.input_sequence_length - source_length),
                self.pad_token_id,
                dtype=torch.long,
                device=inputs_embeds.device,
            )
            inputs_embeds = torch.cat((inputs_embeds, self.embed_tokens(padding_ids)), dim=1)

        expected_image_tokens = sum(expected_image_counts)
        if expected_image_tokens:
            if image_embeds is None:
                raise ValueError(f"image_embeds is required for {expected_image_tokens} image placeholders")
            if isinstance(image_embeds, (list, tuple)):
                if len(image_embeds) != len(expected_image_counts):
                    raise ValueError(
                        "Per-image feature count does not match image spans: "
                        f"features={len(image_embeds)}, spans={len(expected_image_counts)}"
                    )
                normalized_images = []
                for image_index, (features, expected_count) in enumerate(
                    zip(image_embeds, expected_image_counts, strict=True)
                ):
                    features = torch.as_tensor(features, device=inputs_embeds.device)
                    if features.ndim == 3 and features.shape[0] == 1:
                        features = features[0]
                    expected_shape = (expected_count, inputs_embeds.shape[-1])
                    if features.shape != expected_shape:
                        raise ValueError(
                            f"Image features for span {image_index} do not match: "
                            f"expected={expected_shape}, features={tuple(features.shape)}"
                        )
                    normalized_images.append(features)
                image_embeds = torch.cat(normalized_images, dim=0)
            else:
                image_embeds = torch.as_tensor(image_embeds, device=inputs_embeds.device)
                if image_embeds.ndim == 3 and image_embeds.shape[0] == 1:
                    image_embeds = image_embeds[0]
                if image_embeds.ndim != 2:
                    raise ValueError(f"image_embeds must have shape [tokens, hidden], got {tuple(image_embeds.shape)}")
            if image_embeds.shape != (expected_image_tokens, inputs_embeds.shape[-1]):
                raise ValueError(
                    "Image features and image placeholders do not match: "
                    f"expected=({expected_image_tokens}, {inputs_embeds.shape[-1]}), "
                    f"features={tuple(image_embeds.shape)}"
                )
            image_mask = (input_ids == self.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(
                image_mask,
                image_embeds.to(dtype=inputs_embeds.dtype),
            )
        elif image_embeds is not None:
            raise ValueError("image_embeds were provided but no image placeholder span was found")

        if past_seq_length > 0 and requested_position_ids is None:
            rope_deltas = data.get("rope_deltas", self.rope_deltas)
            if rope_deltas is None:
                rope_delta = 0
            else:
                rope_deltas = torch.as_tensor(rope_deltas).flatten()
                if rope_deltas.numel() != 1:
                    raise ValueError(f"rope_deltas must contain exactly one value, got {tuple(rope_deltas.shape)}")
                rope_delta = int(rope_deltas.item())
            decode_start = past_seq_length + rope_delta
            if decode_start < 0:
                raise ValueError(
                    f"past_seq_length + rope_delta must be non-negative, got {past_seq_length} + {rope_delta}"
                )
            requested_position_ids = torch.arange(
                decode_start,
                decode_start + valid_length,
                dtype=torch.long,
                device=self._device,
            ).view(1, 1, valid_length).expand(self.NUM_POSITION_AXES, 1, valid_length).clone()

        position_ids = self._normalize_position_ids(
            requested_position_ids,
            past_seq_length=past_seq_length,
            physical_length=self.input_sequence_length,
            valid_length=valid_length,
            device=self._device,
        )
        if past_seq_length == 0 and not caller_supplied_position_ids:
            if valid_length:
                rope_delta = int(position_ids[:, :, :valid_length].max().item()) + 1 - valid_length
            else:
                rope_delta = 0
            self.rope_deltas = torch.tensor([[rope_delta]], dtype=torch.long, device=self._device)
        elif past_seq_length == 0:
            self.rope_deltas = None
        if not isinstance(self.past_key_caches, CacheList) or not isinstance(self.past_value_caches, CacheList):
            raise ValueError("HunyuanOCR KV cache lists are not initialized")

        return (
            inputs_embeds.to(dtype=self._dtype),
            position_ids[0],
            position_ids[1],
            position_ids[2],
            position_ids[3],
            torch.tensor([past_seq_length], dtype=torch.int32, device=self._device),
            torch.tensor([valid_length], dtype=torch.int32, device=self._device),
            self.past_key_caches,
            self.past_value_caches,
        )


__all__ = ["HunyuanOCRPrefillPlan", "HunyuanOCRTextDataPreprocess"]
