from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image


DEFAULT_IMAGE_SIZE = (448, 448)
OFFICIAL_POOLING_KERNEL_SIZE = 3
DIRECT_TOKEN_POOLING_KERNEL_SIZE = 1
MAX_SOFT_TOKENS = 280

try:
    BICUBIC = Image.Resampling.BICUBIC
except AttributeError:
    BICUBIC = Image.BICUBIC


def build_visual_variant_name(
    upsample_token: bool,
    target_image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
) -> str:
    token_tag = "upsample_token" if upsample_token else "no_upsample_token"
    return f"{token_tag}_{target_image_size[0]}x{target_image_size[1]}"


def build_vision_variant_name(
    upsample_token: bool,
    target_image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
) -> str:
    return build_visual_variant_name(upsample_token, target_image_size)


def resolve_visual_output_dir(
    base_dir: Path,
    upsample_token: bool,
    target_image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
) -> Path:
    variant_name = build_visual_variant_name(upsample_token, target_image_size)
    if base_dir.name.endswith(variant_name):
        return base_dir
    return base_dir.with_name(f"{base_dir.name}_{variant_name}")


def resolve_variant_output_dir(
    base_dir: Path,
    upsample_token: bool,
    target_image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
) -> Path:
    return resolve_visual_output_dir(base_dir, upsample_token, target_image_size)


def resolve_pooling_kernel_size(upsample_token: bool) -> int:
    return OFFICIAL_POOLING_KERNEL_SIZE if upsample_token else DIRECT_TOKEN_POOLING_KERNEL_SIZE


def configure_gemma4_visual_processor(processor: Any, upsample_token: bool) -> int:
    pooling_kernel_size = resolve_pooling_kernel_size(upsample_token)
    processor.image_processor.max_soft_tokens = MAX_SOFT_TOKENS
    processor.image_processor.pooling_kernel_size = pooling_kernel_size
    processor.image_seq_length = (
        MAX_SOFT_TOKENS
        if upsample_token
        else DEFAULT_IMAGE_SIZE[0] // 28 * DEFAULT_IMAGE_SIZE[1] // 28
    )
    return pooling_kernel_size


def configure_gemma4_vision_processor(processor: Any, upsample_token: bool) -> int:
    return configure_gemma4_visual_processor(processor, upsample_token)


def prepare_visual_input_image(
    image_path: str | Path,
    upsample_token: bool,
    target_image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
) -> tuple[Image.Image, dict[str, Any]]:
    resolved_image_path = Path(image_path).resolve()
    image = Image.open(resolved_image_path).convert("RGB")
    original_size = image.size

    if image.size != target_image_size:
        image = image.resize(target_image_size, BICUBIC)

    preprocess_meta = {
        "source_image": str(resolved_image_path),
        "original_image_size": list(original_size),
        "target_image_size": list(target_image_size),
        "upsample_token": upsample_token,
        "processor_pooling_kernel_size": resolve_pooling_kernel_size(upsample_token),
        "max_soft_tokens": MAX_SOFT_TOKENS,
        "vision_variant": build_visual_variant_name(upsample_token, target_image_size),
    }
    return image, preprocess_meta


def prepare_vision_input_image(
    image_path: str | Path,
    upsample_token: bool,
    target_image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
) -> tuple[Image.Image, dict[str, Any]]:
    return prepare_visual_input_image(image_path, upsample_token, target_image_size)


def extract_valid_patch_tokens(
    pixel_values: torch.Tensor,
    pixel_position_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    padding_positions = (pixel_position_ids == -1).all(dim=-1)
    valid_mask = ~padding_positions
    return pixel_values[:, valid_mask[0]], pixel_position_ids[:, valid_mask[0]], valid_mask[0]