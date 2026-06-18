#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Shared static ViT bucket routing utilities for MinerU2.5 examples."""

import importlib.util
import math
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from qwen_vl_utils.vision_process import SPATIAL_MERGE_SIZE
from transformers.models.qwen2_vl.processing_qwen2_vl import Qwen2VLProcessor


DEFAULT_VISUAL_BUCKETS_CONFIG = (
    "configs_merak/xh2a/llm_models/mineru2.5/1_2b/"
    "mineru2_5_visual_buckets_1_2b_xh2a.py"
)


class StaticBucketProcessorAdapter:
    """Route MinerU layout/content images to a finite set of static visual buckets."""

    def __init__(
        self,
        processor,
        buckets: list[tuple[int, int]],
        fallback_bucket: tuple[int, int],
        patch_size: int,
        max_upscale: float = 2.5,
        score_mode: str = "fit_padding",
        alpha_down: float = 10.0,
        beta_up: float = 1.0,
        gamma_pad: float = 3.0,
        ref_area: float = 448 * 448,
        allow_content_fallback_bucket: bool = False,
        logger=None,
        add_hm_pixel_values: bool = False,
        log_prefix: str = "static-vit",
    ) -> None:
        if max_upscale <= 0:
            raise ValueError("max_upscale must be positive.")
        if ref_area <= 0:
            raise ValueError("ref_area must be positive.")
        if score_mode == "ratio":
            raise ValueError(
                "Static ViT score mode 'ratio' is disabled: OmniDocBench tests showed unstable accuracy, "
                "especially after reducing bucket count. Use 'fit_padding' instead."
            )
        if score_mode != "fit_padding":
            raise ValueError(f"Unsupported static ViT score mode: {score_mode}")
        self.processor = processor
        self.buckets = sorted(set(buckets), key=lambda item: (item[0] * item[1], item[0], item[1]))
        self.fallback_bucket = fallback_bucket
        self.patch_size = patch_size
        self.max_upscale = max_upscale
        self.score_mode = score_mode
        self.alpha_down = alpha_down
        self.beta_up = beta_up
        self.gamma_pad = gamma_pad
        self.ref_area = ref_area
        self.allow_content_fallback_bucket = allow_content_fallback_bucket
        self.logger = logger
        self.add_hm_pixel_values = add_hm_pixel_values
        self.log_prefix = log_prefix
        self.bucket_counts: Counter[tuple[int, int]] = Counter()
        self.fallback_count = 0

    def __getattr__(self, name):
        return getattr(self.processor, name)

    def apply_chat_template(self, conversation, chat_template=None, **kwargs):
        return Qwen2VLProcessor.apply_chat_template(
            self.processor,
            conversation,
            chat_template=chat_template,
            **kwargs,
        )

    def __call__(self, *args, **kwargs):
        expected_buckets = None
        images = kwargs.get("images")
        if images is not None:
            kwargs["images"], expected_buckets = self._bucket_images(images)
        model_inputs = self.processor(*args, **kwargs)
        if expected_buckets:
            self._validate_image_grid(model_inputs, expected_buckets)
        if self.add_hm_pixel_values:
            pixel_values = model_inputs.get("pixel_values")
            if isinstance(pixel_values, torch.Tensor):
                model_inputs["hm_pixel_values"] = [pixel_values.contiguous().float()]
        return model_inputs

    def _bucket_images(self, images):
        single_image = isinstance(images, Image.Image)
        image_list = [images] if single_image else list(images)
        bucketed_images = []
        expected_buckets = []
        for image in image_list:
            if not isinstance(image, Image.Image):
                bucketed_images.append(image)
                continue
            bucketed_image, bucket, native_size, render_size, score = self._bucket_image(image)
            bucketed_images.append(bucketed_image)
            expected_buckets.append(bucket)
            self.bucket_counts[bucket] += 1
            if bucket == self.fallback_bucket:
                self.fallback_count += 1
            self._log_bucket(native_size, bucket, render_size, score)
        return (bucketed_images[0] if single_image else bucketed_images), expected_buckets

    def _bucket_image(
        self,
        image: Image.Image,
    ) -> tuple[Image.Image, tuple[int, int], tuple[int, int], tuple[int, int], float]:
        image = image.convert("RGB")
        input_h, input_w = image.height, image.width
        is_layout_input = (input_h, input_w) == self.fallback_bucket
        allow_fallback = is_layout_input or self.allow_content_fallback_bucket
        bucket, score = self._select_bucket(input_h, input_w, allow_fallback=allow_fallback)
        bucketed, render_size = self._letterbox(image, bucket)
        return bucketed, bucket, (input_h, input_w), render_size, score

    def _select_bucket(
        self,
        native_h: int,
        native_w: int,
        allow_fallback: bool,
    ) -> tuple[tuple[int, int], float]:
        candidates = [
            bucket
            for bucket in self.buckets
            if allow_fallback or bucket != self.fallback_bucket
        ]
        if not candidates:
            raise RuntimeError("No static visual buckets available after excluding fallback bucket.")
        scored = [(self._bucket_score(native_h, native_w, bucket), bucket) for bucket in candidates]
        score, bucket = min(scored, key=lambda item: (item[0], item[1][0] * item[1][1], item[1]))
        return bucket, score

    def _bucket_score(self, native_h: int, native_w: int, bucket: tuple[int, int]) -> float:
        if self.score_mode == "fit_padding":
            return self._fit_padding_bucket_score(native_h, native_w, bucket)
        raise ValueError(f"Unsupported static ViT score mode: {self.score_mode}")

    def _ratio_bucket_score(self, native_h: int, native_w: int, bucket: tuple[int, int]) -> float:
        bucket_h, bucket_w = bucket
        native_ratio = native_w / native_h
        bucket_ratio = bucket_w / bucket_h
        aspect_cost = abs(math.log(native_ratio / bucket_ratio))
        scale = min(bucket_h / native_h, bucket_w / native_w)
        effective_scale = min(scale, self.max_upscale)
        render_h = max(1, min(bucket_h, round(native_h * effective_scale)))
        render_w = max(1, min(bucket_w, round(native_w * effective_scale)))
        padding_cost = 1.0 - (render_h * render_w) / (bucket_h * bucket_w)
        downscale_cost = max(0.0, -math.log(scale))
        return 2.0 * aspect_cost + 0.8 * downscale_cost + 0.2 * padding_cost

    def _fit_padding_bucket_score(self, native_h: int, native_w: int, bucket: tuple[int, int]) -> float:
        bucket_h, bucket_w = bucket
        s_fit = min(bucket_w / native_w, bucket_h / native_h)
        scale = min(s_fit, self.max_upscale)
        render_h = max(1, min(bucket_h, round(native_h * scale)))
        render_w = max(1, min(bucket_w, round(native_w * scale)))
        pad_pixels = max(0, bucket_h * bucket_w - render_h * render_w)

        downscale_penalty = max(0.0, 1.0 / scale - 1.0) ** 2
        upscale_penalty = max(0.0, scale - 1.0) ** 2
        padding_penalty = pad_pixels / self.ref_area
        return (
            self.alpha_down * downscale_penalty
            + self.beta_up * upscale_penalty
            + self.gamma_pad * padding_penalty
        )

    def _letterbox(self, image: Image.Image, bucket: tuple[int, int]) -> tuple[Image.Image, tuple[int, int]]:
        bucket_h, bucket_w = bucket
        scale = min(bucket_h / image.height, bucket_w / image.width)
        scale = min(scale, self.max_upscale)
        render_w = max(1, min(bucket_w, round(image.width * scale)))
        render_h = max(1, min(bucket_h, round(image.height * scale)))
        if (render_w, render_h) != image.size:
            image = image.resize((render_w, render_h), Image.Resampling.BICUBIC)
        canvas = Image.new("RGB", (bucket_w, bucket_h), (255, 255, 255))
        canvas.paste(image, ((bucket_w - image.width) // 2, (bucket_h - image.height) // 2))
        return canvas, (render_h, render_w)

    def _validate_image_grid(self, model_inputs, expected_buckets: list[tuple[int, int]]) -> None:
        image_grid_thw = model_inputs.get("image_grid_thw")
        if image_grid_thw is None:
            raise RuntimeError("Expected image_grid_thw after static bucket preprocessing.")
        if image_grid_thw.shape[0] != len(expected_buckets):
            raise RuntimeError(
                f"image_grid_thw count {image_grid_thw.shape[0]} does not match bucket count {len(expected_buckets)}."
            )
        for idx, bucket in enumerate(expected_buckets):
            expected_h = bucket[0] // self.patch_size
            expected_w = bucket[1] // self.patch_size
            actual_h = int(image_grid_thw[idx, 1].item())
            actual_w = int(image_grid_thw[idx, 2].item())
            if (actual_h, actual_w) != (expected_h, expected_w):
                raise RuntimeError(
                    f"Static bucket grid mismatch: bucket={bucket}, "
                    f"expected grid={(expected_h, expected_w)}, actual grid={(actual_h, actual_w)}"
                )

    def _log_bucket(
        self,
        native_size: tuple[int, int],
        bucket: tuple[int, int],
        render_size: tuple[int, int],
        score: float,
    ) -> None:
        message = (
            f"[{self.log_prefix}] input={native_size}, bucket={bucket}, "
            f"render={render_size}, score={score:.4f}"
        )
        if self.logger is not None:
            self.logger.info(message)
        else:
            print(message, flush=True)


def load_python_config(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load config from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_bucket(bucket) -> tuple[int, int]:
    if isinstance(bucket, dict):
        return int(bucket["max_size_h"]), int(bucket["max_size_w"])
    return int(bucket[0]), int(bucket[1])


def validate_static_buckets(buckets: list[tuple[int, int]], patch_size: int, spatial_merge_size: int = SPATIAL_MERGE_SIZE):
    factor = patch_size * spatial_merge_size
    for bucket_h, bucket_w in buckets:
        if bucket_h <= 0 or bucket_w <= 0:
            raise ValueError(f"Invalid static visual bucket {(bucket_h, bucket_w)}")
        if bucket_h % factor != 0 or bucket_w % factor != 0:
            raise ValueError(f"Static visual bucket {(bucket_h, bucket_w)} must be divisible by {factor}")


def describe_static_vit(adapter: StaticBucketProcessorAdapter) -> str:
    return (
        "Static ViT score: "
        f"mode={adapter.score_mode}, "
        f"max_upscale={adapter.max_upscale}, "
        f"alpha_down={adapter.alpha_down}, "
        f"beta_up={adapter.beta_up}, "
        f"gamma_pad={adapter.gamma_pad}, "
        f"ref_area={adapter.ref_area}, "
        f"allow_content_fallback_bucket={adapter.allow_content_fallback_bucket}"
    )
