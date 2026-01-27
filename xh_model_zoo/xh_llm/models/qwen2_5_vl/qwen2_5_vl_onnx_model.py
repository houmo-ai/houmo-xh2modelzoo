# Copyright 2025 HOUMO AI
#
# File: qwen2_5_vl_onnx_model.py
# Description:
#   Qwen2 5 Vl Onnx Model model implementation.
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
import torch.nn.functional as F
from torch import Tensor
from torch import nn
from xhquant.api import HMONNXGoldenInference, HMONNXInference

from .llm_onnx_model import LLMONNXModel

class RepetitionPenaltyLogitsProcessor:
    def __init__(self, penalty: float):
        if not isinstance(penalty, float) or not (penalty > 0):
            raise ValueError(f"`penalty` has to be a strictly positive float, but is {penalty}")
        self.penalty = penalty

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        score = torch.gather(scores, 1, input_ids)
        score = torch.where(score < 0, score * self.penalty, score / self.penalty)
        scores_processed = scores.scatter(1, input_ids, score)
        return scores_processed


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

class Qwen2_5_VLONNXModel(LLMONNXModel):
    def __init__(
        self,
        image_feature,
        prefill,
        decode,
        kv_cache,
        cache_len=2048,
        image_size_w=1204,
        image_size_h=1204,
        max_size_t=2,
        resize_v1 = True,
        repetition_penalty = 1.0,
        chat_template = None,
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
        self.window_size = 112
        self.patch_size = 14
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
        self.window_optimizer = not (self.image_size_w % self.window_size and self.image_size_h % self.window_size)
        self.resize_v1 = resize_v1
        self.repetition_penalty_logits_processor = RepetitionPenaltyLogitsProcessor(repetition_penalty)
        self.chat_template = chat_template

    def get_window_index(self, grid_thw):
        window_index: list = []
        cu_window_seqlens: list = [0]
        window_index_id = 0
        vit_merger_window_size = self.window_size // self.spatial_merge_size // self.patch_size

        for grid_t, grid_h, grid_w in grid_thw:
            llm_grid_h, llm_grid_w = (
                grid_h // self.spatial_merge_size,
                grid_w // self.spatial_merge_size,
            )
            index = torch.arange(grid_t * llm_grid_h * llm_grid_w).reshape(grid_t, llm_grid_h, llm_grid_w)
            pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size
            pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size
            num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size
            num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size
            index_padded = F.pad(index, (0, pad_w, 0, pad_h), "constant", -100)
            index_padded = index_padded.reshape(
                grid_t,
                num_windows_h,
                vit_merger_window_size,
                num_windows_w,
                vit_merger_window_size,
            )
            index_padded = index_padded.permute(0, 1, 3, 2, 4).reshape(
                grid_t,
                num_windows_h * num_windows_w,
                vit_merger_window_size,
                vit_merger_window_size,
            )
            seqlens = (index_padded != -100).sum([2, 3]).reshape(-1)
            index_padded = index_padded.reshape(-1)
            index_new = index_padded[index_padded != -100]
            window_index.append(index_new + window_index_id)
            cu_seqlens_tmp = seqlens.cumsum(0) * self.spatial_merge_unit + cu_window_seqlens[-1]
            cu_window_seqlens.extend(cu_seqlens_tmp.tolist())
            window_index_id += (grid_t * llm_grid_h * llm_grid_w).item()
        window_index = torch.cat(window_index, dim=0)

        return window_index, cu_window_seqlens

    def preprocess_visual(self, inputs):
        visual_inputs = dict()
        visual_inputs["hidden_states"] = inputs["hm_pixel_values"][0]
        visual_inputs["hidden_states"] = visual_inputs["hidden_states"].repeat(self.batch_size, 1, 1, 1)
        visual_inputs["hidden_states"] = visual_inputs["hidden_states"].unsqueeze(2).repeat(1, 1, self.max_size_t, 1, 1)
        inputs["image_grid_thw"][0][0] = self.max_size_t // self.temporal_patch_size
        window_index, cu_window_seqlens = self.get_window_index(inputs["image_grid_thw"])
        cu_window_seqlens = torch.tensor(cu_window_seqlens, device=self.device, dtype=torch.int32)
        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

        seq_len = cu_window_seqlens[-1]
        attention_mask = torch.full(
            [1, seq_len, seq_len],
            torch.finfo(torch.float16).min,
            device=visual_inputs["hidden_states"].device,
            dtype=torch.float16,
        )

        for i in range(1, len(cu_window_seqlens)):
            attention_mask[
                ...,
                cu_window_seqlens[i - 1] : cu_window_seqlens[i],
                cu_window_seqlens[i - 1] : cu_window_seqlens[i],
            ] = 0

        visual_inputs["window_index"] = window_index.to(self.device)
        visual_inputs["window_mask"] = attention_mask.to(self.device)
        visual_inputs["hidden_states"] = visual_inputs["hidden_states"].to(self.device)
        if self.window_optimizer:
            return (
                visual_inputs["hidden_states"].half(),
                visual_inputs["window_index"].to(torch.int32),
            )
        else:
            return (
                visual_inputs["hidden_states"].half(),
                visual_inputs["window_index"].to(torch.int32),
                visual_inputs["window_mask"].half(),
            )

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

        n_image_tokens = (input_ids == self.image_token_id).sum().item()
        if n_image_tokens > 0:
            image_embeds = data["image_embeds"]
            dim = image_embeds.shape[-1]
            image_embeds = image_embeds.reshape(-1, dim)
            n_image_features = image_embeds.shape[0]
            if n_image_tokens != n_image_features:
                raise ValueError(
                    f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                )
            image_mask = (
                (input_ids == self.image_token_id).unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
            )
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        past_seq_length = data["past_seq_length"]
        assert past_seq_length >= 0, "past_seq_length should be non-negative."

        if past_seq_length == 0:
            # prefill
            image_grid_thw = data["image_grid_thw"]
            video_grid_thw = None
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
            time_position_ids,
            height_position_ids,
            width_position_ids,
            torch.tensor([past_seq_length], dtype=torch.int32).to(self._exec_device),
            torch.tensor([seq_length], dtype=torch.int32).to(self._exec_device),
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
            past_key_caches,
            past_value_caches,
        ) = inputs
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
            *past_key_caches,
            *past_value_caches,
        )
        return output

    def create_template(self, prompt, image_dir = None):
        if image_dir is None:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
        else:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image_dir,
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
        return messages

    def preprocess(self, prompt, image_dir, processor):
        from qwen_vl_utils import process_vision_info
        messages = self.create_template(prompt, image_dir)
        print(messages)
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        if self.chat_template is not None:
            text = text.replace("You are a helpful assistant.", self.chat_template)
        if image_dir is not None:
            image_inputs, video_inputs = process_vision_info(messages)
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
    def chat(self, prompt, image_path, processor, logger, use_fast=False, do_sample=False):
        from tqdm import tqdm

        if image_path is not None:
            if self.resize_v1:
                pil_image = self.load_and_process_image(image_path)
            else:
                pil_image = self.load_and_process_image_v2(image_path)
        else:
            pil_image = None
        inputs = self.preprocess(prompt, pil_image, processor)
        inputs = inputs.to(self.device)

        if image_path is not None:
            self.init_image_feature()
            self.to(self.device)
            self.set_exec_device(self._exec_device)
            if use_fast:
                self.image_feature_session.initialize()
                self.image_feature_session._session.to_fast_mode()
            visual_inputs = self.preprocess_visual(inputs)
            image_features = self.extract_image_features(visual_inputs)
            self.release_image_feature()
        else:
            image_features = None

        decoder_ids = list()
        
        data_prefill = {
            "input_ids": inputs["input_ids"],
            "image_embeds": image_features,
            "past_seq_length": 0,
            "image_grid_thw": inputs.get("image_grid_thw", None),
        }
        
        input_ids = data_prefill["input_ids"].cuda()

        self.init_prefill()
        self.to(self.device)
        self.set_exec_device(self._exec_device)
        if use_fast:
            self.prefill_session.initialize()
            self.prefill_session._session.to_fast_mode()
        prefill_logits = self.prefill(data_prefill, save_golden=False)
        
        prefill_logits = self.repetition_penalty_logits_processor(input_ids, prefill_logits[:, -1, :].float()).unsqueeze(1)
        
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
            decode_logits = self.repetition_penalty_logits_processor(input_ids, decode_logits[:, -1, :].float()).unsqueeze(1)
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




