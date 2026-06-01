# Copyright 2025 HOUMO AI
#
# File: qwen3_vl_onnx_model.py
# Description:
#   Qwen3 VL ONNX model implementation.
#   This module provides Qwen3VLONNXModel class for running
#   Qwen3 VL models with ONNX runtime.
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
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor
from transformers.video_utils import VideoMetadata
from xhquant.api import HMONNXGoldenInference, HMONNXInference
from .postprocess import VLLMPresencePenaltyLogitsProcessor
from .llm_onnx_model import LLMONNXModel


def decode_next_token(tokenizer, logits: torch.Tensor, do_sample = False):
    if do_sample:
        probs = nn.functional.softmax(logits[:, -1, :].float(), dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        next_tokens = next_tokens.unsqueeze(0)
    else:
        next_tokens = torch.argmax(logits, dim=-1)
    # logits: (batch_size, 1, vocab_size)
    next_token_str = tokenizer.batch_decode(next_tokens, skip_special_tokens=True)
    return next_tokens, next_token_str


class Qwen3VLONNXModel(LLMONNXModel):
    def __init__(
        self,
        image_feature,
        prefill,
        decode,
        kv_cache,
        cache_len=2048,
        image_size_w=448,
        image_size_h=448,
        max_size_t=2,
        resize_v1 = True,
        presence_penalty = 0,
    ):
        super().__init__(prefill, decode, kv_cache)
        self.pad_token_id = 0
        self.image_token_id = 151655
        self.video_token_id = 151656
        self.vision_start_token_id = 151652
        self.vision_end_token_id = 151653
        self.vision_token_id = 151654
        self.eos_token_id = [151645, 151643]
        self.spatial_merge_size = 2
        self.patch_size = 16
        self.max_size_t = max_size_t
        self.temporal_patch_size = 2
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size
        self.image_feature_session = None
        self.image_feature_config = image_feature
        self.input_sequence_length = 0
        self.cache_len = cache_len
        self.image_size_w = image_size_w
        self.image_size_h = image_size_h
        self.batch_size = 1
        self.resize_v1 = resize_v1
        self.presence_penalty = presence_penalty
        self.logits_processor = VLLMPresencePenaltyLogitsProcessor(presence_penalty, 0)

    def _build_video_raw_clip(self, video_tensor: torch.Tensor) -> torch.Tensor:
        if video_tensor.dim() != 4:
            raise ValueError(f"Expected sampled video tensor with shape [T, C, H, W], but got {tuple(video_tensor.shape)}")

        video_tensor = video_tensor.float()
        target_t = self.max_size_t
        if video_tensor.shape[0] != target_t:
            if video_tensor.shape[0] > target_t:
                indices = torch.linspace(0, video_tensor.shape[0] - 1, target_t).round().long()
                video_tensor = video_tensor.index_select(0, indices)
            else:
                pad_count = target_t - video_tensor.shape[0]
                pad_frames = video_tensor[-1:].repeat(pad_count, 1, 1, 1)
                video_tensor = torch.cat([video_tensor, pad_frames], dim=0)

        if video_tensor.shape[-2:] != (self.image_size_h, self.image_size_w):
            video_tensor = F.interpolate(
                video_tensor,
                size=(self.image_size_h, self.image_size_w),
                mode="bilinear",
                align_corners=False,
            )

        return video_tensor.permute(1, 0, 2, 3).unsqueeze(0).contiguous()

    def _build_sampled_video_metadata(self, video_tensor: torch.Tensor, sample_fps: float) -> list[VideoMetadata]:
        num_frames = int(video_tensor.shape[0])
        duration = None if sample_fps <= 0 else num_frames / sample_fps
        return [
            VideoMetadata(
                total_num_frames=num_frames,
                fps=sample_fps,
                width=int(video_tensor.shape[-1]),
                height=int(video_tensor.shape[-2]),
                duration=duration,
                video_backend="sampled_clip",
                frames_indices=list(range(num_frames)),
            )
        ]


    def preprocess_visual(self, inputs):
        visual_inputs = dict()
        visual_inputs["hidden_states"] = inputs["hm_pixel_values"][0]
        visual_inputs["hidden_states"] = visual_inputs["hidden_states"].to(self.device)
        return (visual_inputs["hidden_states"].half(),)


    def init_image_feature(self):
        self.image_feature_session = HMONNXGoldenInference(self.image_feature_config.onnx)
        self.image_feature_session.exec_device = self._exec_device
        self.image_feature_session.to(self.device)


    def save_image_feature_golden(self, output_dir):
        self.image_feature_session.save_golden = True
        self.image_feature_session.golden_dir = output_dir


    def release_image_feature(self):
        self.image_feature_session = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


    def get_rope_index(
        self,
        input_ids: torch.LongTensor,
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
        if video_grid_thw is not None:
            video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
            video_grid_thw[:, 0] = 1

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
                position_ids = (
                    torch.arange(input_ids.shape[1], device=input_ids.device)
                    .view(1, 1, -1)
                    .expand(3, input_ids.shape[0], -1)
                )
                mrope_position_deltas = torch.zeros(
                    [input_ids.shape[0], 1],
                    device=input_ids.device,
                    dtype=input_ids.dtype,
                )

            return position_ids, mrope_position_deltas

    def prepare_inputs(self, data: Union[dict, tuple, list]):
        device = self._exec_device

        input_ids = data["input_ids"].to(device)

        attention_mask = None

        seq_length = input_ids.shape[1]

        assert self.token_embedding is not None, "Token embedding is not available."
        assert input_ids.shape[0] == 1, "Batch size should be 1 in inference mode."

        assert (
            seq_length <= self.input_sequence_length
        ), f"Input sequence length is too long. max input sequence length is {self.input_sequence_length} but got {seq_length}"
        if self.input_sequence_length > seq_length:
            padding_input_ids = torch.zeros((1, self.input_sequence_length - seq_length), dtype=torch.long).to(device)
            padding_input_ids.fill_(self.pad_token_id)
            input_ids = torch.cat([input_ids, padding_input_ids], dim=-1)

        inputs_embeds = self.token_embedding.to(device)(input_ids.to(device))

        def _normalize_visual_embeds(visual_embeds):
            if visual_embeds is None:
                return None
            if visual_embeds.dim() == 3 and visual_embeds.shape[0] == 1:
                return visual_embeds.squeeze(0)
            return visual_embeds

        def _normalize_deepstack_embeds(deepstack_embeds):
            if deepstack_embeds is None:
                return None
            normalized = []
            for deepstack_embed in deepstack_embeds:
                if deepstack_embed.dim() == 3 and deepstack_embed.shape[0] == 1:
                    normalized.append(deepstack_embed.squeeze(0))
                else:
                    normalized.append(deepstack_embed)
            return normalized

        n_image_tokens = torch.sum(input_ids == self.image_token_id).item()
        n_video_tokens = torch.sum(input_ids == self.video_token_id).item()

        image_mask = None
        video_mask = None

        if n_image_tokens > 0:
            image_embeds = _normalize_visual_embeds(data["image_embeds"])
            n_image_features = image_embeds.shape[0]
            if n_image_tokens != n_image_features:
                raise ValueError(
                    f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                )
            image_mask = (
                (input_ids == self.image_token_id)
                .unsqueeze(-1)
                .expand_as(inputs_embeds)
                .to(inputs_embeds.device)
            )
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if n_video_tokens > 0:
            video_embeds = _normalize_visual_embeds(data["video_embeds"])
            n_video_features = video_embeds.shape[0]
            if n_video_tokens != n_video_features:
                raise ValueError(
                    f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                )
            video_mask = (
                (input_ids == self.video_token_id)
                .unsqueeze(-1)
                .expand_as(inputs_embeds)
                .to(inputs_embeds.device)
            )
            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        deepstack_image_embeds = _normalize_deepstack_embeds(data.get("deepstack_image_embeds"))
        deepstack_video_embeds = _normalize_deepstack_embeds(data.get("deepstack_video_embeds"))
        deepstack_outputs = []
        for layer_index in range(3):
            layer_embed = torch.zeros_like(inputs_embeds)
            if image_mask is not None and deepstack_image_embeds is not None:
                layer_embed = layer_embed.masked_scatter(
                    image_mask.to(deepstack_image_embeds[layer_index].device),
                    deepstack_image_embeds[layer_index],
                )
            if video_mask is not None and deepstack_video_embeds is not None:
                layer_embed = layer_embed.masked_scatter(
                    video_mask.to(deepstack_video_embeds[layer_index].device),
                    deepstack_video_embeds[layer_index],
                )
            deepstack_outputs.append(layer_embed.to(self._exec_device))

        deepstack_image_embed_0 = deepstack_outputs[0]
        deepstack_image_embed_1 = deepstack_outputs[1]
        deepstack_image_embed_2 = deepstack_outputs[2]

        past_seq_length = data["past_seq_length"]
        assert past_seq_length >= 0, "past_seq_length should be non-negative."

        if past_seq_length == 0:
            # prefill
            image_grid_thw = data.get("image_grid_thw")
            video_grid_thw = data.get("video_grid_thw")
            position_ids, rope_deltas = self.get_rope_index(input_ids, image_grid_thw, video_grid_thw, attention_mask)
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

        past_key_caches = []
        past_value_caches = []
        for i in range(self.num_hidden_layers):
            past_key_caches.append(getattr(self, f"past_k_cache_{i}"))
            past_value_caches.append(getattr(self, f"past_v_cache_{i}"))

        time_position_ids = position_ids[0, 0].to(torch.int32)
        height_position_ids = position_ids[1, 0].to(torch.int32)
        width_position_ids = position_ids[2, 0].to(torch.int32)

        return (
            inputs_embeds.to(self._exec_device),
            time_position_ids.to(self._exec_device),
            height_position_ids.to(self._exec_device),
            width_position_ids.to(self._exec_device),
            torch.tensor([past_seq_length], dtype=torch.int32).to(self._exec_device),
            torch.tensor([seq_length], dtype=torch.int32).to(self._exec_device),
            deepstack_image_embed_0,
            deepstack_image_embed_1,
            deepstack_image_embed_2,
            past_key_caches,
            past_value_caches,
        )

    def extract_image_features(self, visual_inputs: List[Tensor]) -> Tensor:
        return self.image_feature_session(*visual_inputs)

    def prefill(self, data, save_golden=True):
        input_ids = data["input_ids"]
        input_seq_len = input_ids.shape[-1]

        steps = (input_seq_len + self.prefill_input_sequence_length - 1) // self.prefill_input_sequence_length

        self.input_sequence_length = self.prefill_input_sequence_length * steps
        inputs = self.prepare_inputs(data)

        (
            inputs_embeds,
            time_position_ids,
            height_position_ids,
            width_position_ids,
            past_seq_length,
            _,
            deepstack_image_embed_0,
            deepstack_image_embed_1,
            deepstack_image_embed_2,
            past_key_caches,
            past_value_caches,
        ) = inputs

        # breakpoint()
        golden_dir = self.prefill_session.golden_dir
        for i in range(steps):
            if save_golden:
                step_golden_dir = Path(golden_dir) / f"prefill_step_{i}"
                step_golden_dir.mkdir(exist_ok=True, parents=True)
                self.save_prefill_golden(step_golden_dir)

            start = i * self.prefill_input_sequence_length
            end = (i + 1) * self.prefill_input_sequence_length
            current_input_length = min(end, input_seq_len) - start
            output = self.prefill_session(
                inputs_embeds[:, start:end, :],
                time_position_ids[start:end],
                height_position_ids[start:end],
                width_position_ids[start:end],
                past_seq_length,
                torch.tensor([current_input_length], dtype=torch.int32).to(inputs_embeds.device),
                deepstack_image_embed_0[:, start:end, :],
                deepstack_image_embed_1[:, start:end, :],
                deepstack_image_embed_2[:, start:end, :],
                *past_key_caches,
                *past_value_caches,
            )
            past_seq_length += current_input_length
        if save_golden:
            self.save_prefill_golden(golden_dir)
        return output

    @torch.no_grad()
    def decode(self, data: Union[dict, tuple, list]):
        self.input_sequence_length = 1
        inputs = self.prepare_inputs(data)
        (
            inputs_embeds,
            time_position_ids,
            height_position_ids,
            width_position_ids,
            past_seq_length,
            current_seq_length,
            deepstack_image_embed_0,
            deepstack_image_embed_1,
            deepstack_image_embed_2,
            past_key_caches,
            past_value_caches,
        ) = inputs
        output = self.decode_session(
            inputs_embeds,
            time_position_ids,
            height_position_ids,
            width_position_ids,
            past_seq_length,
            current_seq_length,
            deepstack_image_embed_0,
            deepstack_image_embed_1,
            deepstack_image_embed_2,
            *past_key_caches,
            *past_value_caches,
        )
        return output

    def create_template(self, prompt, media_path=None, media_type="image"):
        if media_path is None:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
        else:
            media_key = "image" if media_type == "image" else "video"
            media_content = {
                "type": media_type,
                media_key: media_path,
            }
            if media_type == "video":
                media_content["nframes"] = self.max_size_t
                media_content["resized_height"] = self.image_size_h
                media_content["resized_width"] = self.image_size_w
            messages = [
                {
                    "role": "user",
                    "content": [
                        media_content,
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
        return messages

    def preprocess(self, prompt, media_path, processor, cus_temp=False, media_type="image"):
        from qwen_vl_utils import process_vision_info
        if not cus_temp:    
            messages = self.create_template(prompt, media_path, media_type=media_type)
        else:
            messages = prompt
        print(messages)
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        if media_path is not None and media_type == "video":
            image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
            sampled_video = video_inputs[0]
            sampled_metadata = self._build_sampled_video_metadata(sampled_video, float(video_kwargs["fps"][0]))
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
                videos_kwargs={"video_metadata": sampled_metadata, "return_metadata": True},
            )
            inputs["hm_pixel_values"] = [self._build_video_raw_clip(sampled_video)]
        elif media_path is not None:
            image_inputs, video_inputs = process_vision_info(messages, image_patch_size=self.patch_size)
            inputs = processor(
                text=[text],    
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
        else:
            image_inputs, video_inputs = None, None
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
        return inputs

    def load_and_process_image(self, image_path):
        """
        Loads an image from the given path, converts to RGB, resizes proportionally if needed,
        and pads to (self.image_size_w, self.image_size_h) with (114,114,114) background.
        Returns the processed PIL image.
        """
        from PIL import Image, ImageOps
        target_w, target_h = self.image_size_w, self.image_size_h
        image = Image.open(image_path).convert("RGB")
        orig_w, orig_h = image.size
        if (orig_w, orig_h) != (target_w, target_h):
            # Resize while keeping aspect ratio
            scale = min(target_w / orig_w, target_h / orig_h)
            new_w = int(orig_w * scale)
            new_h = int(orig_h * scale)
            image = image.resize((new_w, new_h), Image.BICUBIC)
            # Pad to target size
            pad_w = target_w - new_w
            pad_h = target_h - new_h
            left = 0
            top = 0
            right = pad_w
            bottom = pad_h
            image = ImageOps.expand(image, border=(left, top, right, bottom), fill=(114, 114, 114))
        return image

    def load_and_process_image_v2(self, image_path):
        """
        Loads an image from the given path, converts to RGB, resizes proportionally if needed,
        and pads to (self.image_size_w, self.image_size_h) with (114,114,114) background.
        Returns the processed PIL image.
        """
        from PIL import Image, ImageOps
        target_w, target_h = self.image_size_w, self.image_size_h
        image = Image.open(image_path).convert("RGB")
        orig_w, orig_h = image.size
        if (orig_w, orig_h) != (target_w, target_h):
            image = image.resize((target_w, target_h), Image.BICUBIC)
        return image

    @torch.no_grad()
    def chat(self, prompt, media_path, processor, logger, use_fast=False, do_sample=False, cus_temp=False, media_type="image"):
        from tqdm import tqdm

        if media_path is not None and media_type == "image":
            if self.resize_v1:
                media_input = self.load_and_process_image(media_path)
            else:
                media_input = self.load_and_process_image_v2(media_path)
        elif media_path is not None:
            media_input = media_path
        else:
            media_input = None
        inputs = self.preprocess(prompt, media_input, processor, cus_temp=cus_temp, media_type=media_type)
        inputs = inputs.to(self.device)

        if media_path is not None:
            self.init_image_feature()
            self.to(self.device)
            self.set_exec_device(self._exec_device)
            if use_fast:
                self.image_feature_session.initialize()
                self.image_feature_session._session.to_fast_mode()
            visual_inputs = self.preprocess_visual(inputs)
            visual_features, deepstack_feature_0, deepstack_feature_1, deepstack_feature_2 = self.extract_image_features(visual_inputs)
            self.release_image_feature()
        else:
            visual_features = None
            deepstack_feature_0 = None
            deepstack_feature_1 = None
            deepstack_feature_2 = None

        deepstack_visual_features = (deepstack_feature_0, deepstack_feature_1, deepstack_feature_2)
        decoder_ids = list()
        
        data_prefill = {
            "input_ids": inputs["input_ids"],
            "past_seq_length": 0,
            "image_grid_thw": inputs.get("image_grid_thw", None),
            "video_grid_thw": inputs.get("video_grid_thw", None),
        }
        if media_type == "video":
            data_prefill["video_embeds"] = visual_features
            data_prefill["deepstack_video_embeds"] = deepstack_visual_features
        else:
            data_prefill["image_embeds"] = visual_features
            data_prefill["deepstack_image_embeds"] = deepstack_visual_features
        
        input_ids = data_prefill["input_ids"].cuda()

        self.init_prefill()
        self.to(self.device)
        self.set_exec_device(self._exec_device)
        if use_fast:
            self.prefill_session.initialize()
            self.prefill_session._session.to_fast_mode()
        prefill_logits = self.prefill(data_prefill, save_golden=False)
        
        self.logits_processor.prompt_length = input_ids.shape[1]

        # prefill_logits = self.repetition_penalty_logits_processor(input_ids, prefill_logits[:, -1, :].float()).unsqueeze(1)
        prefill_logits = self.logits_processor(input_ids, prefill_logits[:, -1, :].float()).unsqueeze(1)
        next_token_id, next_token_text = decode_next_token(processor.tokenizer, prefill_logits, do_sample=do_sample)
        input_ids = torch.cat([input_ids, next_token_id], dim=-1)
        decoder_ids.append(next_token_id)
        logger.info(f"Prefill next token: {next_token_id} {next_token_text}")
        self.release_prefill_session()
        
        self.init_decode()
        self.to(self.device)
        self.set_exec_device(self._exec_device)
        if use_fast:
            self.decode_session.initialize()
            self.decode_session._session.to_fast_mode()
        current_length = inputs["input_ids"].shape[-1]
        for decoder_index in tqdm(range(current_length, self.cache_len), desc="Decoder"):
            data_decode = {
                "input_ids": next_token_id,
                "past_seq_length": decoder_index,
            }

            decode_logits = self.decode(data_decode)
            decode_logits = self.logits_processor(input_ids, decode_logits[:, -1, :].float()).unsqueeze(1)
            # decode_logits = self.repetition_penalty_logits_processor(input_ids, decode_logits[:, -1, :].float()).unsqueeze(1)
            next_token_id, next_token_text = decode_next_token(processor.tokenizer, decode_logits, do_sample=do_sample)
            input_ids = torch.cat([input_ids, next_token_id], dim=-1)
            decoder_ids.append(next_token_id)
            logger.info(f"Decode Quanted Model next token: {next_token_id} {next_token_text}")
            if next_token_id.cpu().item() in self.eos_token_id:
                break
            out = processor.decode(torch.cat(decoder_ids, dim=-1).view(-1)).strip()
            logger.info(f"Output: {out}")
        
        decoder_ids = torch.cat(decoder_ids, dim=-1).view(-1)
        out = processor.decode(decoder_ids).strip()
        logger.info(f"Output: {out}")
        self.release_decode_session()
        return out
        # breakpoint()