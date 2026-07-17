from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_transforms import convert_to_rgb, resize, to_channel_dimension_format
from transformers.image_utils import (
    ChannelDimension,
    get_image_size,
    infer_channel_dimension_format,
    make_flat_list_of_images,
    make_list_of_images,
    to_numpy_array,
)
from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor, smart_resize
from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor, Qwen3VLProcessorKwargs


class LingBotQwen3VLDataPreprocess(nn.Module):
    def __init__(
        self,
        *,
        token_embedding: nn.Embedding,
        input_sequence_length: int,
        image_token_id: int,
        video_token_id: int,
        vision_start_token_id: int,
        spatial_merge_size: int,
    ):
        super().__init__()
        self.token_embedding = token_embedding
        self.input_sequence_length = int(input_sequence_length)
        self.image_token_id = int(image_token_id)
        self.video_token_id = int(video_token_id)
        self.vision_start_token_id = int(vision_start_token_id)
        self.spatial_merge_size = int(spatial_merge_size)
        self.pad_token_id = 0

    def get_rope_index(
        self,
        input_ids: torch.LongTensor,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if image_grid_thw is None and video_grid_thw is None:
            if attention_mask is None:
                position_ids = (
                    torch.arange(input_ids.shape[1], device=input_ids.device)
                    .view(1, 1, -1)
                    .expand(3, input_ids.shape[0], -1)
                )
                rope_deltas = torch.zeros(
                    (input_ids.shape[0], 1),
                    device=input_ids.device,
                    dtype=input_ids.dtype,
                )
                return position_ids, rope_deltas
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(input_ids.device)
            max_position_ids = position_ids.max(0)[0].max(-1, keepdim=True)[0]
            return position_ids, max_position_ids + 1 - attention_mask.shape[-1]

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        position_ids = torch.ones(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        rope_deltas = []
        image_index = 0
        video_index = 0
        for batch_index, sequence in enumerate(input_ids):
            sequence = sequence[attention_mask[batch_index] == 1]
            vision_start_indices = torch.argwhere(sequence == self.vision_start_token_id).squeeze(1)
            vision_tokens = sequence[vision_start_indices + 1]
            image_count = int((vision_tokens == self.image_token_id).sum().item())
            video_count = int((vision_tokens == self.video_token_id).sum().item())
            tokens = sequence.tolist()
            position_parts = []
            start = 0
            remaining_images = image_count
            remaining_videos = video_count
            for _ in range(image_count + video_count):
                image_end = (
                    tokens.index(self.image_token_id, start)
                    if self.image_token_id in tokens[start:] and remaining_images
                    else len(tokens) + 1
                )
                video_end = (
                    tokens.index(self.video_token_id, start)
                    if self.video_token_id in tokens[start:] and remaining_videos
                    else len(tokens) + 1
                )
                if image_end < video_end:
                    grid_t, grid_h, grid_w = image_grid_thw[image_index]
                    image_index += 1
                    remaining_images -= 1
                    vision_end = image_end
                else:
                    grid_t, grid_h, grid_w = video_grid_thw[video_index]
                    video_index += 1
                    remaining_videos -= 1
                    vision_end = video_end

                grid_t = int(grid_t.item())
                grid_h = int(grid_h.item()) // self.spatial_merge_size
                grid_w = int(grid_w.item()) // self.spatial_merge_size
                text_length = vision_end - start
                start_position = int(position_parts[-1].max().item() + 1) if position_parts else 0
                position_parts.append(torch.arange(text_length).view(1, -1).expand(3, -1) + start_position)
                time_index = torch.arange(grid_t).view(-1, 1).expand(-1, grid_h * grid_w).flatten()
                height_index = torch.arange(grid_h).view(1, -1, 1).expand(grid_t, -1, grid_w).flatten()
                width_index = torch.arange(grid_w).view(1, 1, -1).expand(grid_t, grid_h, -1).flatten()
                position_parts.append(
                    torch.stack((time_index, height_index, width_index)) + text_length + start_position
                )
                start = vision_end + grid_t * grid_h * grid_w

            if start < len(tokens):
                start_position = int(position_parts[-1].max().item() + 1) if position_parts else 0
                text_length = len(tokens) - start
                position_parts.append(torch.arange(text_length).view(1, -1).expand(3, -1) + start_position)
            positions = torch.cat(position_parts, dim=1).reshape(3, -1)
            position_ids[..., batch_index, attention_mask[batch_index] == 1] = positions.to(position_ids.device)
            rope_deltas.append(positions.max() + 1 - len(input_ids[batch_index]))
        return position_ids, torch.tensor(rope_deltas, device=input_ids.device).unsqueeze(1)

    def forward(self, data: dict[str, Any]) -> tuple[Any, ...]:
        input_ids = data["input_ids"].to(self.token_embedding.weight.device)
        if input_ids.shape[0] != 1:
            raise ValueError("LingBot Qwen3-VL export supports batch_size=1")
        current_length = int(input_ids.shape[1])
        if current_length > self.input_sequence_length:
            raise ValueError(
                f"Input sequence length {current_length} exceeds static length {self.input_sequence_length}"
            )
        if current_length < self.input_sequence_length:
            padding = torch.full(
                (1, self.input_sequence_length - current_length),
                self.pad_token_id,
                device=input_ids.device,
                dtype=input_ids.dtype,
            )
            input_ids = torch.cat((input_ids, padding), dim=-1)

        inputs_embeds = self.token_embedding(input_ids)
        deepstack_embeds = [torch.zeros_like(inputs_embeds) for _ in range(3)]
        image_token_count = int((input_ids == self.image_token_id).sum().item())
        if image_token_count:
            image_embeds = data["image_embeds"]
            if image_token_count != image_embeds.shape[0]:
                raise ValueError(
                    f"Image token count {image_token_count} does not match visual features {image_embeds.shape[0]}"
                )
            image_mask = (input_ids == self.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(
                image_mask,
                image_embeds.to(inputs_embeds.device, inputs_embeds.dtype),
            )
            for index, feature in enumerate(data["deepstack_image_embeds"]):
                deepstack_embeds[index] = deepstack_embeds[index].masked_scatter(
                    image_mask,
                    feature.to(inputs_embeds.device, inputs_embeds.dtype),
                )

        past_sequence_length = int(data.get("past_seq_length", 0))
        if past_sequence_length != 0:
            raise ValueError("LingBot Qwen3-VL exports a prefill-only language graph")
        position_ids, _ = self.get_rope_index(
            input_ids,
            image_grid_thw=data.get("image_grid_thw"),
            video_grid_thw=None,
            attention_mask=None,
        )
        return (
            inputs_embeds,
            position_ids[0, 0].to(torch.int64),
            position_ids[1, 0].to(torch.int64),
            position_ids[2, 0].to(torch.int64),
            torch.tensor([0], dtype=torch.int32, device=input_ids.device),
            torch.tensor([current_length], dtype=torch.int32, device=input_ids.device),
            deepstack_embeds[0],
            deepstack_embeds[1],
            deepstack_embeds[2],
        )


class LingBotQwen3VLImageProcessor(Qwen2VLImageProcessor):
    def _raw_pixel_values(
        self,
        images: Any,
        *,
        do_resize: bool,
        resample: Any,
        do_convert_rgb: bool,
        input_data_format: ChannelDimension | str | None,
    ) -> list[torch.Tensor]:
        raw_values = []
        for image in make_flat_list_of_images(self.fetch_images(images)):
            frames = make_list_of_images(image)
            if do_convert_rgb:
                frames = [convert_to_rgb(frame) for frame in frames]
            frames = [to_numpy_array(frame) for frame in frames]
            frame_format = input_data_format or infer_channel_dimension_format(frames[0])
            height, width = get_image_size(frames[0], channel_dim=frame_format)
            if do_resize:
                height, width = smart_resize(
                    height,
                    width,
                    factor=self.patch_size * self.merge_size,
                    min_pixels=self.min_pixels,
                    max_pixels=self.max_pixels,
                )
            processed = []
            for frame in frames:
                if do_resize:
                    frame = resize(
                        frame,
                        size=(height, width),
                        resample=resample,
                        input_data_format=frame_format,
                    )
                processed.append(
                    to_channel_dimension_format(
                        frame,
                        ChannelDimension.FIRST,
                        input_channel_dim=frame_format,
                    )
                )
            tensor = torch.from_numpy(np.asarray(processed))
            if tensor.shape[0] == 1:
                tensor = tensor.repeat(self.temporal_patch_size, 1, 1, 1)
            if tensor.shape[0] != self.temporal_patch_size:
                raise ValueError(
                    f"LingBot visual graph expects {self.temporal_patch_size} frames, got {tensor.shape[0]}"
                )
            raw_values.append(tensor.permute(1, 0, 2, 3).unsqueeze(0))
        return raw_values

    def preprocess(self, images: Any, videos: Any = None, **kwargs: Any) -> BatchFeature:
        do_resize = kwargs.get("do_resize", self.do_resize)
        resample = kwargs.get("resample", self.resample)
        do_convert_rgb = kwargs.get("do_convert_rgb", self.do_convert_rgb)
        input_data_format = kwargs.get("input_data_format")
        hm_pixel_values = self._raw_pixel_values(
            images,
            do_resize=do_resize,
            resample=resample,
            do_convert_rgb=do_convert_rgb,
            input_data_format=input_data_format,
        )
        outputs = super().preprocess(images=images, videos=videos, **kwargs)
        outputs["hm_pixel_values"] = hm_pixel_values
        return outputs


class LingBotQwen3VLProcessor(Qwen3VLProcessor):
    def __init__(self, image_processor=None, tokenizer=None, video_processor=None, chat_template=None, **kwargs):
        super().__init__(image_processor, tokenizer, video_processor, chat_template=chat_template, **kwargs)
        self.image_processor = LingBotQwen3VLImageProcessor(**vars(self.image_processor))

    def __call__(self, images=None, text=None, videos=None, **kwargs) -> BatchFeature:
        output_kwargs = self._merge_kwargs(
            Qwen3VLProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
        if images is not None:
            image_inputs = self.image_processor(images=images, **output_kwargs["images_kwargs"])
            image_grid_thw = image_inputs["image_grid_thw"]
        else:
            image_inputs = {}
            image_grid_thw = None
        if videos is not None:
            raise ValueError("LingBot Video Qwen3-VL runtime accepts image or text conditioning, not video input")

        if not isinstance(text, list):
            text = [text]
        text = text.copy()
        if image_grid_thw is not None:
            merge_length = self.image_processor.merge_size**2
            image_index = 0
            for text_index in range(len(text)):
                while self.image_token in text[text_index]:
                    image_token_count = int(image_grid_thw[image_index].prod().item() // merge_length)
                    text[text_index] = text[text_index].replace(
                        self.image_token,
                        "<|placeholder|>" * image_token_count,
                        1,
                    )
                    image_index += 1
                text[text_index] = text[text_index].replace("<|placeholder|>", self.image_token)

        output_kwargs["text_kwargs"].pop("return_tensors", None)
        return_mm_token_type_ids = output_kwargs["text_kwargs"].pop("return_mm_token_type_ids", None)
        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"], return_tensors="pt")
        self._check_special_mm_tokens(text, text_inputs, modalities=["image", "video"])
        if return_mm_token_type_ids:
            input_ids = np.asarray(text_inputs["input_ids"])
            mm_token_type_ids = np.zeros_like(input_ids)
            mm_token_type_ids[input_ids == self.image_token_id] = 1
            text_inputs["mm_token_type_ids"] = mm_token_type_ids.tolist()
        return BatchFeature(data={**text_inputs, **image_inputs})
