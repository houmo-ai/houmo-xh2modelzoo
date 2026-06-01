# -*- coding: utf-8 -*-
# Copyright 2025 HOUMO AI
#
# File: qwen3_5_vl_hmonnx_demo.py
# Description:
#   Qwen3.5-VL HMONNX inference demo glue script for xh2modelzoo.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""
Qwen3.5-9B vision + LLM HMONNX inference demo.

Loads:
- Vision HMONNX exported by qwen3_5_vision_xh2a_export_hmonnx.py
- Prefill/Decode HMONNX exported by qwen3_5_xh2a_export.py

Usage:
    python examples/llm/qwen3_5/qwen3_5_vl_hmonnx_demo.py

    python examples/llm/qwen3_5/qwen3_5_vl_hmonnx_demo.py \
        --image-path data/images/qwen2_vl_demo.jpeg \
        --prompt "描述这张照片"
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import xhquant.utils.suppress_printing
from qwen_vl_utils import process_vision_info
from transformers import AutoConfig, AutoTokenizer
from transformers.image_processing_utils import BatchFeature
from transformers.video_processing_utils import BaseVideoProcessor
from xhquant.api import HMONNXInference

from xh_model_zoo.api import ConfigDict, get_root_logger, xhquant_llm_init
from xh_model_zoo.xh_llm.models.qwen3_5 import Qwen3_5ONNXModel, Qwen3_5Processor
from xh_model_zoo.xh_llm.models.qwen3_5.image_processing_qwen3_5 import Qwen3_5ImageProcessor
from transformers import TextStreamer
from xh_model_zoo.xh_llm.models.qwen3_5.qwen3_5_onnx_model import (
    _alloc_cache_inputs,
    _apply_presence_penalty,
    _apply_repetition_penalty,
    _build_linear_attn_mask,
    _resolve_input_name,
    _sample_next_token,
    _select_last_valid_logits,
)


DTYPE_NAME_MAP = {
    "fp16": torch.float16,
    "float16": torch.float16,
    "half": torch.float16,
    "fp32": torch.float32,
    "float32": torch.float32,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
}


def parse_arguments():
    import argparse

    parser = argparse.ArgumentParser(
        description="Qwen3.5-9B vision + LLM HMONNX inference demo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-config",
        type=str,
        default=(
            "/data01/home/chenzx/project/xhquant_llm/work_dirs/qwen3_5/"
            "hmquant_xh2_qwen3.5_9b_w4_a8_256_2k_448_20260324_"
            "Qwen3.5-9B-quarot-gptq-4bit-mse24-hessian_20260324_103723/"
            "export_meta_info.json"
        ),
        help="Path to LLM export_meta_info.json",
    )
    parser.add_argument(
        "--vision-onnx",
        type=str,
        default=(
            "/data01/home/chenzx/project/xh2modelzoo/work_dirs/qwen3_5_9B/"
            "qwen3_5_instruct_vision_config_1_2_448_448_use_gptq_model_False_Qwen3/"
            "vision/qwen3_5_instruct_vision_config.onnx"
        ),
        help="Path to vision HMONNX model",
    )
    parser.add_argument("--image-path", type=str, default="data/images/qwen2_vl_demo.jpeg")
    parser.add_argument("--prompt", type=str, default="描述这张照片")
    parser.add_argument("--system-prompt", type=str, default="")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-context-tokens", type=int, default=None)
    parser.add_argument("--max-size-w", type=int, default=448)
    parser.add_argument("--max-size-h", type=int, default=448)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--do-sample", dest="do_sample", action="store_true", default=False)
    parser.add_argument("--no-sample", dest="do_sample", action="store_false")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--presence-penalty", type=float, default=0.0)
    parser.add_argument("--stream-output", dest="stream_output", action="store_true", default=True)
    parser.add_argument("--no-stream-output", dest="stream_output", action="store_false")
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Enable tokenizer thinking mode when supported by the tokenizer/chat template.",
    )
    parser.add_argument("--dtype", type=str, default="fp16", choices=sorted(DTYPE_NAME_MAP.keys()))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--exec-device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--disable-auto-offload", action="store_true")
    parser.add_argument(
        "--auto-offload-max-memory",
        type=str,
        default=None,
        help='JSON string, e.g. {"0":"35GB","cpu":"120GB"}',
    )
    parser.add_argument(
        "--prefill-auto-offload-max-memory",
        type=str,
        default=None,
        help='JSON string for prefill only, e.g. {"0":"40GB","cpu":"120GB"}',
    )
    parser.add_argument(
        "--decode-auto-offload-max-memory",
        type=str,
        default=None,
        help='JSON string for decode only, e.g. {"1":"40GB","cpu":"120GB"}',
    )
    parser.add_argument(
        "--resource-tight-mode",
        action="store_true",
        help="Load prefill/decode lazily to reduce peak memory",
    )
    parser.add_argument("--debug", action="store_true")
    return parser


def _resolve_path(base_dir: Path, path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def _parse_dtype(dtype_name: str) -> torch.dtype:
    key = dtype_name.strip().lower()
    if key not in DTYPE_NAME_MAP:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    return DTYPE_NAME_MAP[key]


def _load_token_embedding(embed_path: Path) -> nn.Module:
    torch.serialization.add_safe_globals([nn.Embedding])
    try:
        try:
            token_embedding = torch.load(str(embed_path), map_location="cpu", weights_only=False)
        except TypeError:
            token_embedding = torch.load(str(embed_path), map_location="cpu")
    finally:
        torch.serialization.clear_safe_globals()

    if isinstance(token_embedding, dict):
        if "weight" not in token_embedding:
            raise ValueError(f"Unsupported embedding state dict format: {embed_path}")
        embedding = nn.Embedding(token_embedding["weight"].shape[0], token_embedding["weight"].shape[1])
        embedding.load_state_dict(token_embedding)
        token_embedding = embedding

    token_embedding.eval()
    return token_embedding


def _parse_auto_offload_max_memory(max_memory_json: Optional[str]):
    if max_memory_json is None or max_memory_json.strip() == "":
        return None
    parsed = json.loads(max_memory_json)
    if not isinstance(parsed, dict):
        raise ValueError("auto_offload_max_memory must be a JSON object")
    fixed = {}
    for key, value in parsed.items():
        try:
            normalized_key = int(key)
        except Exception:
            normalized_key = key
        fixed[normalized_key] = value
    return fixed


def _resolve_processor_source(hf_model_config_dir: Path, hf_model_dir: Optional[str]) -> Path:
    if (hf_model_config_dir / "preprocessor_config.json").exists():
        return hf_model_config_dir
    if hf_model_dir is None:
        return hf_model_config_dir
    candidate = Path(hf_model_dir).resolve()
    if (candidate / "preprocessor_config.json").exists():
        return candidate
    return hf_model_config_dir


class _DummyVideoProcessor(BaseVideoProcessor):
    model_input_names = ["pixel_values_videos", "video_grid_thw"]

    def __call__(self, videos=None, **kwargs):
        del videos, kwargs
        return BatchFeature(data={})


def _build_processor(processor_source: Path, args) -> Qwen3_5Processor:
    tokenizer = AutoTokenizer.from_pretrained(str(processor_source), trust_remote_code=True)
    chat_template = getattr(tokenizer, "chat_template", None)
    if chat_template is None:
        chat_template_file = processor_source / "chat_template.jinja"
        if chat_template_file.exists():
            chat_template = chat_template_file.read_text(encoding="utf-8")
    image_processor = Qwen3_5ImageProcessor(
        patch_size=args.patch_size,
        merge_size=2,
        temporal_patch_size=2,
        min_pixels=args.max_size_h * args.max_size_w,
        max_pixels=args.max_size_h * args.max_size_w,
    )
    video_processor = _DummyVideoProcessor()
    return Qwen3_5Processor(
        image_processor=image_processor,
        tokenizer=tokenizer,
        video_processor=video_processor,
        chat_template=chat_template,
    )


def _scatter_image_embeds(
    input_ids: torch.Tensor,
    token_embeds: torch.Tensor,
    image_embeds: torch.Tensor,
    image_token_id: int,
) -> torch.Tensor:
    n_image_tokens = int((input_ids == image_token_id).sum().item())
    if n_image_tokens == 0:
        return token_embeds

    if image_embeds.dim() != 2:
        raise ValueError(f"image_embeds must be rank-2, got shape={tuple(image_embeds.shape)}")
    if image_embeds.shape[0] != n_image_tokens:
        raise ValueError(
            f"Image tokens/features mismatch, tokens={n_image_tokens}, features={image_embeds.shape[0]}"
        )

    image_mask = (input_ids == image_token_id).unsqueeze(-1).expand_as(token_embeds)
    image_embeds = image_embeds.to(token_embeds.device, token_embeds.dtype)
    return token_embeds.masked_scatter(image_mask, image_embeds)


def _build_messages(
    prompt: str,
    image_path: str,
    system_prompt: str = "",
    history: Optional[List[Dict[str, str]]] = None,
    max_size_h: int = 448,
    max_size_w: int = 448,
) -> list[dict]:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    if history:
        messages.extend(history)

    messages.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                    "resized_height": max_size_h,
                    "resized_width": max_size_w,
                },
                {"type": "text", "text": prompt},
            ],
        }
    )
    return messages


def _prepare_multimodal_inputs(
    processor: Qwen3_5Processor,
    prompt: str,
    image_path: str,
    args,
    history: Optional[List[Dict[str, str]]] = None,
    system_prompt: str = "",
):
    messages = _build_messages(
        prompt=prompt,
        image_path=image_path,
        system_prompt=system_prompt,
        history=history,
        max_size_h=args.max_size_h,
        max_size_w=args.max_size_w,
    )
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=args.enable_thinking,
    )
    image_inputs, video_inputs = process_vision_info(messages, image_patch_size=args.patch_size)
    model_inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        min_pixels=args.max_size_h * args.max_size_w,
        max_pixels=args.max_size_h * args.max_size_w,
        patch_size=args.patch_size,
        merge_size=2,
        padding=True,
        return_tensors="pt",
    )
    logger = get_root_logger()
    logger.info(f"processor image_grid_thw: {model_inputs.get('image_grid_thw')}")
    logger.info(
        "processor image token count: %s",
        int((model_inputs["input_ids"] == processor.image_token_id).sum().item()),
    )
    return model_inputs


class Qwen35VLHMONNXModel(Qwen3_5ONNXModel):
    def __init__(
        self,
        vision_onnx: str,
        image_token_id: int,
        video_token_id: int,
        vision_start_token_id: int,
        spatial_merge_size: int,
        prefill,
        decode,
        max_context_tokens: Optional[int] = None,
        auto_offload: bool = True,
        auto_offload_max_memory=None,
        prefill_auto_offload_max_memory=None,
        decode_auto_offload_max_memory=None,
        resource_tight_mode: bool = False,
        pad_token_id: int = 0,
    ):
        super().__init__(
            prefill=prefill,
            decode=decode,
            max_context_tokens=max_context_tokens,
            auto_offload=auto_offload,
            auto_offload_max_memory=auto_offload_max_memory,
            prefill_auto_offload_max_memory=prefill_auto_offload_max_memory,
            decode_auto_offload_max_memory=decode_auto_offload_max_memory,
            resource_tight_mode=resource_tight_mode,
            pad_token_id=pad_token_id,
        )
        self.vision_onnx = str(vision_onnx)
        self.image_token_id = int(image_token_id)
        self.video_token_id = int(video_token_id)
        self.vision_start_token_id = int(vision_start_token_id)
        self.spatial_merge_size = int(spatial_merge_size)
        self.rope_deltas: Optional[torch.Tensor] = None
        self.vision_session: Optional[HMONNXInference] = None

        self._prefill_time_pos_name = None
        self._prefill_height_pos_name = None
        self._prefill_width_pos_name = None
        self._decode_time_pos_name = None
        self._decode_height_pos_name = None
        self._decode_width_pos_name = None

    def _create_prefill_session(self):
        super()._create_prefill_session()
        self._prefill_time_pos_name = _resolve_input_name(self.prefill_session, ("time_position_ids",))
        self._prefill_height_pos_name = _resolve_input_name(
            self.prefill_session,
            ("hight_position_ids", "height_position_ids"),
        )
        self._prefill_width_pos_name = _resolve_input_name(self.prefill_session, ("width_position_ids",))

    def _create_decode_session(self):
        super()._create_decode_session()
        self._decode_time_pos_name = _resolve_input_name(self.decode_session, ("time_position_ids",))
        self._decode_height_pos_name = _resolve_input_name(
            self.decode_session,
            ("hight_position_ids", "height_position_ids"),
        )
        self._decode_width_pos_name = _resolve_input_name(self.decode_session, ("width_position_ids",))

    def _release_prefill_session(self):
        super()._release_prefill_session()
        self._prefill_time_pos_name = None
        self._prefill_height_pos_name = None
        self._prefill_width_pos_name = None

    def _release_decode_session(self):
        super()._release_decode_session()
        self._decode_time_pos_name = None
        self._decode_height_pos_name = None
        self._decode_width_pos_name = None

    def _init_vision_session(self):
        if self.vision_session is None:
            self.vision_session = HMONNXInference(self.vision_onnx)
            self.vision_session.exec_device = self.exec_device
            self.vision_session.to(self.device)

    def release_vision_session(self):
        self.vision_session = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def extract_image_features(self, hm_pixel_values: torch.Tensor) -> torch.Tensor:
        self._init_vision_session()
        hm_pixel_values = hm_pixel_values.to(device=self.device, dtype=torch.float16)
        outputs = self.vision_session(hm_pixel_values)
        if isinstance(outputs, (tuple, list)):
            image_embeds = outputs[0]
        else:
            image_embeds = outputs
        if image_embeds.dim() == 3 and image_embeds.shape[0] == 1:
            image_embeds = image_embeds.squeeze(0)
        return image_embeds.to(device=self.device, dtype=self.dtype)

    def extract_all_image_features(self, hm_pixel_values) -> torch.Tensor:
        if hm_pixel_values is None:
            raise ValueError("hm_pixel_values is required")

        if isinstance(hm_pixel_values, torch.Tensor):
            return self.extract_image_features(hm_pixel_values)

        if not isinstance(hm_pixel_values, Sequence) or len(hm_pixel_values) == 0:
            raise ValueError("hm_pixel_values must be a non-empty tensor or sequence of tensors")

        image_embeds_list = []
        for pixel_values in hm_pixel_values:
            image_embeds_list.append(self.extract_image_features(pixel_values))
        return torch.cat(image_embeds_list, dim=0)

    def _maybe_fix_image_token_id(self, input_ids: torch.Tensor, image_embeds: torch.Tensor):
        token_count = int((input_ids == self.image_token_id).sum().item())
        if token_count == int(image_embeds.shape[0]):
            return
        unique_ids, counts = torch.unique(input_ids, return_counts=True)
        matched = unique_ids[counts == int(image_embeds.shape[0])]
        if matched.numel() == 1:
            self.image_token_id = int(matched[0].item())
            return
        raise ValueError(
            f"Image token count mismatch: configured={token_count}, feature_tokens={image_embeds.shape[0]}"
        )

    def get_rope_index(
        self,
        input_ids: torch.LongTensor,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mrope_position_deltas = []
        if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
            total_input_ids = input_ids
            if attention_mask is None:
                attention_mask = torch.ones_like(total_input_ids)
            position_ids = torch.zeros(
                3,
                input_ids.shape[0],
                input_ids.shape[1],
                dtype=input_ids.dtype,
                device=input_ids.device,
            )

            image_index, video_index = 0, 0
            image_grid_thw_list = image_grid_thw.tolist() if image_grid_thw is not None else None
            video_grid_thw_list = video_grid_thw.tolist() if video_grid_thw is not None else None

            for batch_index, ids in enumerate(total_input_ids):
                ids = ids[attention_mask[batch_index] == 1]
                vision_start_indices = torch.argwhere(ids == self.vision_start_token_id).squeeze(1)
                vision_tokens = ids[vision_start_indices + 1]
                image_nums = (vision_tokens == self.image_token_id).sum()
                video_nums = (vision_tokens == self.video_token_id).sum()
                input_tokens = ids.tolist()

                llm_pos_ids_list = []
                start = 0
                remain_images, remain_videos = image_nums, video_nums

                for _ in range(image_nums + video_nums):
                    end_image = (
                        input_tokens.index(self.image_token_id, start)
                        if self.image_token_id in input_tokens and remain_images > 0
                        else len(input_tokens) + 1
                    )
                    end_video = (
                        input_tokens.index(self.video_token_id, start)
                        if self.video_token_id in input_tokens and remain_videos > 0
                        else len(input_tokens) + 1
                    )

                    if end_image < end_video:
                        t, h, w = image_grid_thw_list[image_index]
                        image_index += 1
                        remain_images -= 1
                        end = end_image
                    else:
                        t, h, w = video_grid_thw_list[video_index]
                        video_index += 1
                        remain_videos -= 1
                        end = end_video

                    llm_grid_t = t
                    llm_grid_h = h // self.spatial_merge_size
                    llm_grid_w = w // self.spatial_merge_size
                    text_len = end - start
                    start_index = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0

                    llm_pos_ids_list.append(
                        torch.arange(text_len, device=input_ids.device).view(1, -1).expand(3, -1) + start_index
                    )

                    t_index = (
                        torch.arange(llm_grid_t, device=input_ids.device)
                        .view(-1, 1)
                        .expand(-1, llm_grid_h * llm_grid_w)
                        .flatten()
                    )
                    h_index = (
                        torch.arange(llm_grid_h, device=input_ids.device)
                        .view(1, -1, 1)
                        .expand(llm_grid_t, -1, llm_grid_w)
                        .flatten()
                    )
                    w_index = (
                        torch.arange(llm_grid_w, device=input_ids.device)
                        .view(1, 1, -1)
                        .expand(llm_grid_t, llm_grid_h, -1)
                        .flatten()
                    )
                    llm_pos_ids_list.append(
                        torch.stack([t_index, h_index, w_index]) + text_len + start_index
                    )
                    start = end + llm_grid_t * llm_grid_h * llm_grid_w

                if start < len(input_tokens):
                    start_index = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
                    text_len = len(input_tokens) - start
                    llm_pos_ids_list.append(
                        torch.arange(text_len, device=input_ids.device).view(1, -1).expand(3, -1) + start_index
                    )

                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                position_ids[..., batch_index, attention_mask[batch_index] == 1] = llm_positions.to(position_ids.device)
                mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[batch_index]))

            deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
            return position_ids, deltas

        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(input_ids.device)
            max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
            deltas = max_position_ids + 1 - attention_mask.shape[-1]
            return position_ids, deltas

        position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).view(1, 1, -1).expand(3, input_ids.shape[0], -1)
        deltas = torch.zeros([input_ids.shape[0], 1], device=input_ids.device, dtype=input_ids.dtype)
        return position_ids, deltas

    def _prepare_prefill_tensors(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        image_embeds: torch.Tensor,
        image_grid_thw: Optional[torch.Tensor],
    ):
        prompt_len = int(input_ids.shape[1])
        prefill_chunk_len = int(self._prefill_inputs_info.shape[1])
        target_seq_len = ((prompt_len + prefill_chunk_len - 1) // prefill_chunk_len) * prefill_chunk_len
        pad_len = target_seq_len - prompt_len

        input_ids = input_ids.to(device=self.device, dtype=torch.long)
        if pad_len > 0:
            padding_ids = torch.full((1, pad_len), self.pad_token_id, dtype=torch.long, device=self.device)
            input_ids = torch.cat([input_ids, padding_ids], dim=-1)

        if attention_mask is None:
            attention_mask = torch.ones((1, prompt_len), dtype=torch.long, device=self.device)
        else:
            attention_mask = attention_mask.to(device=self.device, dtype=torch.long)
        if pad_len > 0:
            padding_mask = torch.zeros((1, pad_len), dtype=attention_mask.dtype, device=self.device)
            attention_mask = torch.cat([attention_mask, padding_mask], dim=-1)

        inputs_embeds = self.token_embedding(input_ids).to(dtype=self._prefill_inputs_info.dtype)

        self._maybe_fix_image_token_id(input_ids, image_embeds)
        inputs_embeds = _scatter_image_embeds(input_ids, inputs_embeds, image_embeds, self.image_token_id)

        if image_grid_thw is not None:
            image_grid_thw = image_grid_thw.to(device=self.device, dtype=torch.long)
        position_ids, rope_deltas = self.get_rope_index(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            attention_mask=attention_mask,
        )
        if pad_len > 0:
            rope_deltas = rope_deltas + pad_len
        self.rope_deltas = rope_deltas

        return (
            inputs_embeds,
            position_ids[0, 0],
            position_ids[1, 0],
            position_ids[2, 0],
            prompt_len,
            target_seq_len,
        )

    def _build_mm_prefill_feed(
        self,
        chunk_embeds: torch.Tensor,
        time_position_ids: torch.Tensor,
        height_position_ids: torch.Tensor,
        width_position_ids: torch.Tensor,
        valid_len: int,
        past_seq_len: int,
        cache_state: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        batch_size = self._prefill_inputs_info.shape[0]
        past_seq_length = torch.full(
            (batch_size,),
            int(past_seq_len),
            dtype=self._prefill_past_seq_info.dtype,
            device=self.device,
        )
        current_input_length = torch.full(
            (batch_size,),
            int(valid_len),
            dtype=self._prefill_current_seq_info.dtype,
            device=self.device,
        )
        linear_attn_mask = _build_linear_attn_mask(valid_len, self._prefill_mask_info, self.device)

        feed = {}
        for name in self.prefill_session.get_input_names():
            if name == self._prefill_inputs_name:
                feed[name] = chunk_embeds
            elif name == self._prefill_time_pos_name:
                info = self.prefill_session.get_input(name)
                feed[name] = time_position_ids.to(dtype=info.dtype, device=self.device)
            elif name == self._prefill_height_pos_name:
                info = self.prefill_session.get_input(name)
                feed[name] = height_position_ids.to(dtype=info.dtype, device=self.device)
            elif name == self._prefill_width_pos_name:
                info = self.prefill_session.get_input(name)
                feed[name] = width_position_ids.to(dtype=info.dtype, device=self.device)
            elif name == self._prefill_past_seq_name:
                feed[name] = past_seq_length
            elif name == self._prefill_current_seq_name:
                feed[name] = current_input_length
            elif name == self._prefill_mask_name:
                feed[name] = linear_attn_mask
            elif name in cache_state:
                feed[name] = cache_state[name]
            else:
                info = self.prefill_session.get_input(name)
                feed[name] = torch.zeros(info.shape, dtype=info.dtype, device=self.device)
        return feed

    def _build_mm_decode_feed(
        self,
        token_id: torch.Tensor,
        past_seq_len: int,
        cache_state: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        if self.rope_deltas is None:
            raise RuntimeError("rope_deltas is not initialized")

        inputs_embeds = self.token_embedding(token_id.to(device=self.device, dtype=torch.long)).to(
            dtype=self._decode_inputs_info.dtype
        )
        linear_attn_mask = _build_linear_attn_mask(1, self._decode_mask_info, self.device)
        delta = (past_seq_len + self.rope_deltas).view(-1)

        batch_size = self._decode_inputs_info.shape[0]
        past_seq_length = torch.full(
            (batch_size,),
            int(past_seq_len),
            dtype=self._decode_past_seq_info.dtype,
            device=self.device,
        )
        current_input_length = torch.full(
            (batch_size,),
            1,
            dtype=self._decode_current_seq_info.dtype,
            device=self.device,
        )

        feed = {}
        for name in self.decode_session.get_input_names():
            if name == self._decode_inputs_name:
                feed[name] = inputs_embeds
            elif name == self._decode_time_pos_name:
                info = self.decode_session.get_input(name)
                feed[name] = delta.to(dtype=info.dtype, device=self.device)
            elif name == self._decode_height_pos_name:
                info = self.decode_session.get_input(name)
                feed[name] = delta.to(dtype=info.dtype, device=self.device)
            elif name == self._decode_width_pos_name:
                info = self.decode_session.get_input(name)
                feed[name] = delta.to(dtype=info.dtype, device=self.device)
            elif name == self._decode_past_seq_name:
                feed[name] = past_seq_length
            elif name == self._decode_current_seq_name:
                feed[name] = current_input_length
            elif name == self._decode_mask_name:
                feed[name] = linear_attn_mask
            elif name in cache_state:
                feed[name] = cache_state[name]
            else:
                info = self.decode_session.get_input(name)
                feed[name] = torch.zeros(info.shape, dtype=info.dtype, device=self.device)
        return feed

    @torch.no_grad()
    def generate_multimodal(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        image_embeds: torch.Tensor,
        image_grid_thw: Optional[torch.Tensor],
        tokenizer,
        max_new_tokens: int,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        stream_output: bool = False,
    ) -> str:
        if max_new_tokens <= 0:
            return ""
        if input_ids.dim() != 2 or input_ids.shape[0] != 1:
            raise ValueError(f"input_ids must be [1, seq], got {tuple(input_ids.shape)}")
        if self.token_embedding is None:
            raise ValueError("token_embedding is not set, call set_input_embeddings first.")

        prompt_len = int(input_ids.shape[1])
        if self.max_context_tokens is not None and prompt_len > self.max_context_tokens:
            raise ValueError(
                f"Prompt too long for multimodal HMONNX demo: prompt_len={prompt_len}, max_context_tokens={self.max_context_tokens}"
            )

        self._ensure_prefill_session()
        inputs_embeds, time_pos, height_pos, width_pos, prompt_len, target_seq_len = self._prepare_prefill_tensors(
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_embeds=image_embeds,
            image_grid_thw=image_grid_thw,
        )

        prefill_cache_state = _alloc_cache_inputs(self.prefill_session, self.device)
        prefill_chunk_len = int(self._prefill_inputs_info.shape[1])
        last_prefill_logits = None
        past_seq_len = 0

        for start in range(0, target_seq_len, prefill_chunk_len):
            end = start + prefill_chunk_len
            valid_len = min(prefill_chunk_len, max(0, prompt_len - start))
            if valid_len <= 0:
                break

            prefill_feed = self._build_mm_prefill_feed(
                chunk_embeds=inputs_embeds[:, start:end, :],
                time_position_ids=time_pos[start:end],
                height_position_ids=height_pos[start:end],
                width_position_ids=width_pos[start:end],
                valid_len=valid_len,
                past_seq_len=past_seq_len,
                cache_state=prefill_cache_state,
            )
            _, prefill_output_map = self._run_hmonnx(self.prefill_session, prefill_feed)
            prefill_logits = self._extract_logits(prefill_output_map)
            last_prefill_logits = _select_last_valid_logits(prefill_logits, valid_len)
            self._update_linear_cache(prefill_cache_state, prefill_output_map)
            past_seq_len += valid_len

        if last_prefill_logits is None:
            return ""

        history_token_ids = input_ids[0].tolist()
        first_step_logits = _apply_repetition_penalty(
            last_prefill_logits, history_token_ids, repetition_penalty
        )
        first_step_logits = _apply_presence_penalty(
            first_step_logits, history_token_ids, presence_penalty
        )
        next_token_id = _sample_next_token(
            first_step_logits,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        logger = get_root_logger()
        logger.info(f"First generated token id: {int(next_token_id[0][0].item())}")

        if self.resource_tight_mode:
            self._release_prefill_session()

        self._ensure_decode_session()
        decode_cache_state = _alloc_cache_inputs(self.decode_session, self.device)
        for name in decode_cache_state:
            if name in prefill_cache_state:
                decode_cache_state[name] = prefill_cache_state[name]

        stop_token_ids = set()
        tokenizer_eos = getattr(tokenizer, "eos_token_id", None)
        tokenizer_pad = getattr(tokenizer, "pad_token_id", None)
        if isinstance(tokenizer_eos, (list, tuple)):
            stop_token_ids.update(int(token_id) for token_id in tokenizer_eos)
        elif tokenizer_eos is not None:
            stop_token_ids.add(int(tokenizer_eos))
        if tokenizer_pad is not None:
            stop_token_ids.add(int(tokenizer_pad))

        generated_ids = []
        streamer: Optional[TextStreamer] = None
        if stream_output:
            streamer = TextStreamer(tokenizer, skip_prompt=False, skip_special_tokens=True)

        token_val = int(next_token_id[0][0].item())
        if token_val in stop_token_ids:
            logger.info("Generation stopped immediately on EOS/PAD token.")
            if streamer is not None:
                streamer.end()
            return ""
        generated_ids.append(token_val)
        history_token_ids.append(token_val)
        if streamer is not None:
            streamer.put(next_token_id.detach().cpu())

        current_token = next_token_id.to(self.device)
        for _ in range(max_new_tokens - 1):
            decode_feed = self._build_mm_decode_feed(current_token, past_seq_len, decode_cache_state)
            _, decode_output_map = self._run_hmonnx(self.decode_session, decode_feed)
            decode_logits = self._extract_logits(decode_output_map)
            decode_logits = _select_last_valid_logits(decode_logits, 1)

            decode_logits = _apply_repetition_penalty(
                decode_logits, history_token_ids, repetition_penalty
            )
            decode_logits = _apply_presence_penalty(
                decode_logits, history_token_ids, presence_penalty
            )
            next_token_id = _sample_next_token(
                decode_logits,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
            self._update_linear_cache(decode_cache_state, decode_output_map)

            token_val = int(next_token_id[0][0].item())
            if token_val in stop_token_ids:
                break
            generated_ids.append(token_val)
            history_token_ids.append(token_val)
            if streamer is not None:
                streamer.put(next_token_id.detach().cpu())
            current_token = next_token_id.to(self.device)
            past_seq_len += 1

        if streamer is not None:
            streamer.end()
        if self.resource_tight_mode:
            self._release_decode_session()

        output_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        logger.info(f"Generated {len(generated_ids)} tokens")
        return output_text

    @torch.no_grad()
    def chat(
        self,
        prompt: str,
        image_path: str,
        processor: Qwen3_5Processor,
        args,
        history: Optional[List[Dict[str, str]]] = None,
        system_prompt: str = "",
    ) -> str:
        model_inputs = _prepare_multimodal_inputs(
            processor,
            prompt=prompt,
            image_path=image_path,
            args=args,
            history=history,
            system_prompt=system_prompt,
        )
        input_ids = model_inputs["input_ids"].to(self.device)
        attention_mask = model_inputs.get("attention_mask", None)
        image_grid_thw = model_inputs.get("image_grid_thw", None)
        hm_pixel_values = model_inputs.get("hm_pixel_values", None)
        if hm_pixel_values is None:
            raise ValueError("Processor output does not contain hm_pixel_values")

        image_embeds = self.extract_all_image_features(hm_pixel_values)
        self.release_vision_session()
        return self.generate_multimodal(
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_embeds=image_embeds,
            image_grid_thw=image_grid_thw,
            tokenizer=processor.tokenizer,
            max_new_tokens=args.max_new_tokens,
            do_sample=getattr(args, "do_sample", False),
            temperature=getattr(args, "temperature", 1.0),
            top_p=getattr(args, "top_p", 1.0),
            top_k=getattr(args, "top_k", 0),
            repetition_penalty=getattr(args, "repetition_penalty", 1.0),
            presence_penalty=getattr(args, "presence_penalty", 0.0),
            stream_output=getattr(args, "stream_output", False),
        )


def main():
    parser = parse_arguments()
    args = parser.parse_args()

    model_meta_file_path = Path(args.model_config).resolve()
    model_dir = model_meta_file_path.parent
    meta_info = json.load(open(model_meta_file_path, "r", encoding="utf-8"))

    prefill_onnx = _resolve_path(model_dir, meta_info["prefill_onnx_file"])
    decode_onnx = _resolve_path(model_dir, meta_info["decode_onnx_file"])
    hf_model_config_dir = _resolve_path(model_dir, meta_info["hf_config"])
    embed_tokens_file = _resolve_path(model_dir, meta_info["token_embedding_file"])
    vision_onnx = Path(args.vision_onnx).resolve()
    hf_model_dir = meta_info.get("hf_model", None)
    processor_source = _resolve_processor_source(hf_model_config_dir, hf_model_dir)

    max_context_tokens = args.max_context_tokens
    if max_context_tokens is None:
        kv_cache_shape = meta_info.get("kv_cache_shape", None)
        if isinstance(kv_cache_shape, list) and len(kv_cache_shape) >= 3:
            max_context_tokens = int(kv_cache_shape[2])

    cfg_name = "qwen3_5_9b_vl_hmonnx_demo"
    work_dir = Path("./work_dirs") / cfg_name
    work_dir.mkdir(exist_ok=True, parents=True)
    log_file = work_dir / f"{cfg_name}.log"
    xhquant_llm_init(log_file, args.debug)
    logger = get_root_logger()

    xhquant.utils.suppress_printing.disable_printing = True

    dtype = _parse_dtype(args.dtype)
    auto_offload_max_memory = _parse_auto_offload_max_memory(args.auto_offload_max_memory)
    prefill_auto_offload_max_memory = _parse_auto_offload_max_memory(args.prefill_auto_offload_max_memory)
    decode_auto_offload_max_memory = _parse_auto_offload_max_memory(args.decode_auto_offload_max_memory)

    processor = _build_processor(processor_source, args)
    model_config = AutoConfig.from_pretrained(str(hf_model_config_dir), trust_remote_code=True)
    token_embedding = _load_token_embedding(embed_tokens_file).to(dtype=dtype)

    xh_model = Qwen35VLHMONNXModel(
        vision_onnx=str(vision_onnx),
        image_token_id=getattr(processor, "image_token_id", getattr(model_config, "image_token_id", 248056)),
        video_token_id=getattr(processor, "video_token_id", getattr(model_config, "video_token_id", 248057)),
        vision_start_token_id=getattr(
            processor,
            "vision_start_token_id",
            getattr(model_config, "vision_start_token_id", 248053),
        ),
        spatial_merge_size=getattr(getattr(model_config, "vision_config", None), "spatial_merge_size", 2),
        prefill=ConfigDict(onnx=str(prefill_onnx)),
        decode=ConfigDict(onnx=str(decode_onnx)),
        max_context_tokens=max_context_tokens,
        auto_offload=not args.disable_auto_offload,
        auto_offload_max_memory=auto_offload_max_memory,
        prefill_auto_offload_max_memory=prefill_auto_offload_max_memory,
        decode_auto_offload_max_memory=decode_auto_offload_max_memory,
        resource_tight_mode=args.resource_tight_mode,
        pad_token_id=(
            processor.tokenizer.pad_token_id
            if processor.tokenizer.pad_token_id is not None
            else processor.tokenizer.eos_token_id
        ),
    )
    xh_model.set_input_embeddings(token_embedding)

    if args.disable_auto_offload:
        xh_model.to(torch.device(args.device))
    else:
        target_device = torch.device(args.device)
        xh_model._device = target_device
        if xh_model.token_embedding is not None:
            xh_model.token_embedding.to(target_device)

    xh_model.set_exec_device(torch.device(args.exec_device))
    if args.disable_auto_offload:
        xh_model.to(dtype)
    else:
        if xh_model.token_embedding is not None:
            xh_model.token_embedding.to(dtype)
        xh_model._dtype = dtype

    logger.info(f"vision hmonnx: {vision_onnx}")
    logger.info(f"prefill hmonnx: {prefill_onnx}")
    logger.info(f"decode  hmonnx: {decode_onnx}")
    logger.info(f"hf config: {hf_model_config_dir}")
    logger.info(f"processor source: {processor_source}")
    logger.info(f"token embedding: {embed_tokens_file}")
    logger.info(f"image_path: {Path(args.image_path).resolve()}")
    logger.info(f"prompt: {args.prompt}")
    logger.info(f"system_prompt: {args.system_prompt}")
    logger.info(f"max_context_tokens={max_context_tokens}, max_new_tokens={args.max_new_tokens}")
    logger.info(f"resource_tight_mode={args.resource_tight_mode}")
    logger.info(f"enable_thinking={args.enable_thinking}")

    out = xh_model.chat(
        prompt=args.prompt,
        image_path=args.image_path,
        processor=processor,
        args=args,
        history=None,
        system_prompt=args.system_prompt,
    )
    if args.stream_output:
        print("", flush=True)
    print(f"\n[Output]: {repr(out)}", flush=True)


if __name__ == "__main__":
    main()