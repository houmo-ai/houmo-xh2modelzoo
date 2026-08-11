"""Inference-time processor for Unlimited-OCR base/no-crop HMONNX runtime.

This processor reuses the reference logic from ``modeling_unlimitedocr`` (prompt
formatting, PIL image loading, image transform) so the exported runtime builds
the exact same ``input_ids`` / ``images_seq_mask`` / global-view image tensor the
original model produces for base/no-crop single-image OCR.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Optional

import torch
from PIL import ImageOps
from transformers import AutoTokenizer

from .modeling_unlimitedocr import (
    BasicImageTransform,
    dynamic_preprocess,
    format_messages,
    load_pil_images,
    text_encode,
)


class XHUnlimitedOCRProcessor:
    """Build runtime inputs for Unlimited-OCR base/no-crop single-image prompts."""

    image_token = "<image>"

    def __init__(
        self,
        tokenizer: Any,
        *,
        image_token_id: int = 128815,
        image_size: int = 1024,
        base_size: int = 1024,
        patch_size: int = 16,
        downsample_ratio: int = 4,
        crop_mode: bool = False,
        max_crop_num: int = 32,
        image_mean: tuple[float, float, float] = (0.5, 0.5, 0.5),
        image_std: tuple[float, float, float] = (0.5, 0.5, 0.5),
        normalize: bool = True,
    ):
        self.tokenizer = tokenizer
        self.image_token_id = int(image_token_id)
        self.image_size = int(image_size)
        self.base_size = int(base_size)
        self.patch_size = int(patch_size)
        self.downsample_ratio = int(downsample_ratio)
        self.crop_mode = bool(crop_mode)
        self.max_crop_num = int(max_crop_num)
        self.image_mean = tuple(image_mean)
        self.image_std = tuple(image_std)
        self.normalize = bool(normalize)
        self._image_transform = BasicImageTransform(
            mean=self.image_mean, std=self.image_std, normalize=self.normalize
        )

    @classmethod
    def from_pretrained(cls, hf_model_dir: str | Path, **kwargs: Any) -> "XHUnlimitedOCRProcessor":
        tokenizer = AutoTokenizer.from_pretrained(str(hf_model_dir), trust_remote_code=True)
        return cls(tokenizer, **kwargs)

    @property
    def num_queries(self) -> int:
        return math.ceil((self.image_size // self.patch_size) / self.downsample_ratio)

    @property
    def image_token_count(self) -> int:
        num_queries = self.num_queries
        return (num_queries + 1) * num_queries + 1

    @property
    def num_queries_base(self) -> int:
        return math.ceil((self.base_size // self.patch_size) / self.downsample_ratio)

    def _build_image_tokens(self) -> list[int]:
        num_queries = self.num_queries
        tokens = ([self.image_token_id] * num_queries + [self.image_token_id]) * num_queries
        tokens += [self.image_token_id]
        return tokens

    def _build_crop_image_tokens(self, width_crop_num: int, height_crop_num: int) -> list[int]:
        """Base(global)+separator[+local] image token layout for crop/gundam.

        Matches the original ``infer()`` token construction: a base-grid block
        with per-row newlines, one view separator, and (when cropped) a
        local-grid block with per-row newlines.
        """
        nqb = self.num_queries_base
        nq = self.num_queries
        tokens = ([self.image_token_id] * nqb + [self.image_token_id]) * nqb
        tokens += [self.image_token_id]
        if width_crop_num > 1 or height_crop_num > 1:
            tokens += ([self.image_token_id] * (nq * width_crop_num) + [self.image_token_id]) * (nq * height_crop_num)
        return tokens

    def build_conversation(self, prompt: str, image_file: Optional[str]) -> list[dict[str, Any]]:
        if not prompt:
            raise ValueError("prompt must be provided for Unlimited-OCR inference.")
        user_message: dict[str, Any] = {"role": "<|User|>", "content": prompt}
        if image_file:
            user_message["images"] = [image_file]
        return [user_message, {"role": "<|Assistant|>", "content": ""}]

    def process(
        self,
        prompt: str,
        image_file: Optional[str] = None,
        *,
        device: str | torch.device = "cpu",
    ) -> dict[str, torch.Tensor]:
        """Return input_ids, images_seq_mask, images_spatial_crop and image tensors.

        Supports base/no-crop (single global view) and crop/gundam (global view
        plus dynamic local crops). For crop mode the returned dict also carries
        ``images_crop`` (local patches padded to ``base_size``); for base mode
        ``images_crop`` is an all-zero placeholder so the HF forward takes the
        global-only branch.
        """
        conversation = self.build_conversation(prompt, image_file)
        formatted = format_messages(conversations=conversation, sft_format="plain", system_prompt="")
        images = load_pil_images(conversation)

        text_splits = formatted.split(self.image_token)
        if image_file is not None and len(images) == 0:
            raise ValueError(f"Could not load image from {image_file!r}.")
        if len(text_splits) - 1 != len(images):
            raise ValueError(
                f"prompt/image mismatch: found {len(text_splits) - 1} '<image>' marker(s) "
                f"for {len(images)} image(s)."
            )
        if self.crop_mode and len(images) > 1:
            raise NotImplementedError("Unlimited-OCR crop/gundam runtime currently supports a single image.")

        tokenized_str: list[int] = []
        images_seq_mask: list[bool] = []
        images_list: list[torch.Tensor] = []
        images_crop_list: list[torch.Tensor] = []
        images_spatial_crop: list[list[int]] = []

        for text_sep, image in zip(text_splits, images, strict=False):
            tokenized_sep = text_encode(self.tokenizer, text_sep, bos=False, eos=False)
            tokenized_str += tokenized_sep
            images_seq_mask += [False] * len(tokenized_sep)

            if self.crop_mode:
                image_tokens = self._process_crop_image(
                    image, images_list, images_crop_list, images_spatial_crop
                )
            else:
                global_view = ImageOps.pad(
                    image,
                    (self.image_size, self.image_size),
                    color=tuple(int(x * 255) for x in self.image_mean),
                )
                images_list.append(self._image_transform(global_view).to(torch.float32))
                images_spatial_crop.append([1, 1])
                image_tokens = self._build_image_tokens()

            tokenized_str += image_tokens
            images_seq_mask += [True] * len(image_tokens)

        tokenized_sep = text_encode(self.tokenizer, text_splits[-1], bos=False, eos=False)
        tokenized_str += tokenized_sep
        images_seq_mask += [False] * len(tokenized_sep)

        bos_id = 0
        tokenized_str = [bos_id] + tokenized_str
        images_seq_mask = [False] + images_seq_mask

        input_ids = torch.tensor(tokenized_str, dtype=torch.long, device=device).unsqueeze(0)
        images_seq_mask_tensor = torch.tensor(images_seq_mask, dtype=torch.bool, device=device).unsqueeze(0)

        global_size = self.base_size if self.crop_mode else self.image_size
        if images_list:
            images_ori = torch.stack(images_list, dim=0).to(device)
            images_spatial_crop_tensor = torch.tensor(images_spatial_crop, dtype=torch.long, device=device)
        else:
            images_ori = torch.zeros((0, 3, global_size, global_size), device=device)
            images_spatial_crop_tensor = torch.zeros((0, 2), dtype=torch.long, device=device)

        if images_crop_list:
            images_crop = torch.stack(images_crop_list, dim=0).to(device)
        else:
            images_crop = torch.zeros((0, 3, self.image_size, self.image_size), device=device)

        return {
            "input_ids": input_ids,
            "images_seq_mask": images_seq_mask_tensor,
            "images_spatial_crop": images_spatial_crop_tensor,
            "images_ori": images_ori,
            "images_crop": images_crop,
        }

    def _process_crop_image(
        self,
        image: Any,
        images_list: list[torch.Tensor],
        images_crop_list: list[torch.Tensor],
        images_spatial_crop: list[list[int]],
    ) -> list[int]:
        """Build crop/gundam global view + local crops and the matching tokens."""
        if image.size[0] <= self.image_size and image.size[1] <= self.image_size:
            crop_ratio = [1, 1]
            images_crop_raw: list[Any] = []
        else:
            images_crop_raw, crop_ratio = dynamic_preprocess(
                image, image_size=self.image_size, max_num=self.max_crop_num
            )

        global_view = ImageOps.pad(
            image,
            (self.base_size, self.base_size),
            color=tuple(int(x * 255) for x in self.image_mean),
        )
        images_list.append(self._image_transform(global_view).to(torch.float32))

        width_crop_num, height_crop_num = crop_ratio
        images_spatial_crop.append([width_crop_num, height_crop_num])

        if width_crop_num > 1 or height_crop_num > 1:
            for crop in images_crop_raw:
                images_crop_list.append(self._image_transform(crop).to(torch.float32))

        return self._build_crop_image_tokens(width_crop_num, height_crop_num)
