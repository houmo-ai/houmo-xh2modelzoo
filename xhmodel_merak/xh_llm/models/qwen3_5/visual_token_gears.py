"""Static token-gear helpers for the Qwen3.5 vision encoder.

The processor emits flattened, normalized temporal patches in the same
merge-major order consumed by the native Qwen3.5 ViT.  A token-gear graph
pads those patches to a fixed capacity while keeping the image geometry in
compact position/RoPE inputs.
"""

from __future__ import annotations

import math
from typing import Iterable

import torch


VISUAL_INPUT_IMAGE = "image"
VISUAL_INPUT_PATCHES = "patches"
SUPPORTED_VISUAL_INPUT_MODES = frozenset({VISUAL_INPUT_IMAGE, VISUAL_INPUT_PATCHES})
VISUAL_ATTENTION_MASK_FORMAT = "additive_key_padding_bias"
VISUAL_ATTENTION_MASK_SHAPE = "[1,1,1,patch_token_capacity]"
VISUAL_ATTENTION_MASK_OPERATOR = "xhquant.nn.MaskedAdd+Softmax"
VISUAL_ROTARY_POSITION_FORMAT = "height_width_2d_ids"


def normalize_image_token_gears(gears: Iterable[int]) -> tuple[int, ...]:
    """Return a sorted, unique and validated post-merge gear sequence."""

    normalized = tuple(sorted({int(gear) for gear in gears}))
    if not normalized or normalized[0] <= 0:
        raise ValueError(f"image token gears must be positive, got {normalized}")
    return normalized


def select_image_token_gear(valid_image_tokens: int, gears: Iterable[int]) -> int:
    """Select the smallest post-merge gear that can hold one image."""

    valid_image_tokens = int(valid_image_tokens)
    if valid_image_tokens <= 0:
        raise ValueError(f"valid image token count must be positive, got {valid_image_tokens}")
    normalized = normalize_image_token_gears(gears)
    for gear in normalized:
        if valid_image_tokens <= gear:
            return gear
    raise ValueError(
        f"image needs {valid_image_tokens} tokens, which exceeds the largest configured gear {normalized[-1]}"
    )


def build_visual_gear_manifest(
    gear_artifacts: Iterable[dict],
    *,
    spatial_merge_size: int = 2,
    visual_rope_cache_length: int,
) -> dict:
    """Build the versioned runtime contract for a set of static HMONNX gears."""

    artifacts = sorted(
        (dict(artifact) for artifact in gear_artifacts),
        key=lambda artifact: int(artifact["image_token_capacity"]),
    )
    gears = normalize_image_token_gears(artifact["image_token_capacity"] for artifact in artifacts)
    if len(gears) != len(artifacts):
        raise ValueError("visual gear artifacts must have unique image_token_capacity values")
    visual_rope_cache_length = int(visual_rope_cache_length)
    minimum_cache_length = gears[-1] * int(spatial_merge_size)
    if visual_rope_cache_length < minimum_cache_length:
        raise ValueError(
            "visual_rope_cache_length does not cover the largest gear: "
            f"got {visual_rope_cache_length}, need at least {minimum_cache_length}"
        )
    for gear, artifact in zip(gears, artifacts, strict=True):
        expected_patch_capacity = patch_token_capacity(gear, spatial_merge_size)
        if int(artifact["patch_token_capacity"]) != expected_patch_capacity:
            raise ValueError(
                f"gear {gear} declares patch capacity {artifact['patch_token_capacity']}, "
                f"expected {expected_patch_capacity}"
            )
        if not str(artifact.get("hmonnx", "")).strip():
            raise ValueError(f"gear {gear} is missing its HMONNX path")
    return {
        "schema_version": 1,
        "visual_input_mode": VISUAL_INPUT_PATCHES,
        "token_unit": "post_spatial_merge",
        "routing_policy": "smallest_fit",
        "overflow_policy": "reject",
        "attention_mask_format": VISUAL_ATTENTION_MASK_FORMAT,
        "attention_mask_shape": VISUAL_ATTENTION_MASK_SHAPE,
        "attention_mask_operator": VISUAL_ATTENTION_MASK_OPERATOR,
        "rotary_position_format": VISUAL_ROTARY_POSITION_FORMAT,
        "visual_rope_cache_length": visual_rope_cache_length,
        "spatial_merge_size": int(spatial_merge_size),
        "shared_weight_loader": "MultiHMONNXLoader",
        "gears": artifacts,
    }


def patch_token_capacity(image_token_capacity: int, spatial_merge_size: int = 2) -> int:
    """Convert a post-merger image-token capacity to a pre-merger patch capacity."""

    image_token_capacity = int(image_token_capacity)
    spatial_merge_size = int(spatial_merge_size)
    if image_token_capacity <= 0:
        raise ValueError(f"image_token_capacity must be positive, got {image_token_capacity}")
    if spatial_merge_size <= 0:
        raise ValueError(f"spatial_merge_size must be positive, got {spatial_merge_size}")
    return image_token_capacity * spatial_merge_size * spatial_merge_size


def factor_image_token_grid(image_token_capacity: int, spatial_merge_size: int = 2) -> torch.Tensor:
    """Build a valid, near-square ``[t, h, w]`` grid for export dummy data."""

    image_token_capacity = int(image_token_capacity)
    merged_h = math.isqrt(image_token_capacity)
    while image_token_capacity % merged_h != 0:
        merged_h -= 1
    merged_w = image_token_capacity // merged_h
    return torch.tensor(
        [[1, merged_h * spatial_merge_size, merged_w * spatial_merge_size]],
        dtype=torch.int64,
    )


def pad_flattened_patches(pixel_values: torch.Tensor, capacity: int) -> tuple[torch.Tensor, int]:
    """Right-pad native flattened patches to ``[1, capacity, patch_dim]``."""

    if pixel_values.ndim == 2:
        pixel_values = pixel_values.unsqueeze(0)
    if pixel_values.ndim != 3 or pixel_values.shape[0] != 1:
        raise ValueError(
            f"flattened Qwen3.5 patches must have shape [N, D] or [1, N, D], got {tuple(pixel_values.shape)}"
        )
    valid_tokens = int(pixel_values.shape[1])
    capacity = int(capacity)
    if valid_tokens > capacity:
        raise ValueError(f"patch token count {valid_tokens} exceeds gear capacity {capacity}")
    if valid_tokens == capacity:
        return pixel_values, valid_tokens
    padding = pixel_values.new_zeros((1, capacity - valid_tokens, pixel_values.shape[2]))
    return torch.cat((pixel_values, padding), dim=1), valid_tokens


def _merge_major(tensor: torch.Tensor, t: int, h: int, w: int, merge_size: int) -> torch.Tensor:
    """Reorder ``[..., h, w]`` values to Qwen3.5 merge-major patch order."""

    leading_shape = tensor.shape[:-2]
    tensor = tensor.unsqueeze(-3).expand(*leading_shape, t, h, w)
    tensor = tensor.reshape(*leading_shape, t, h // merge_size, merge_size, w // merge_size, merge_size)
    leading_dims = list(range(len(leading_shape)))
    base = len(leading_shape)
    tensor = tensor.permute(*leading_dims, base, base + 1, base + 3, base + 2, base + 4)
    return tensor.flatten(len(leading_shape))


def _position_interpolation_inputs(
    *,
    t: int,
    h: int,
    w: int,
    capacity: int,
    num_position_embeddings: int,
    spatial_merge_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    grid_side = math.isqrt(num_position_embeddings)
    if grid_side * grid_side != num_position_embeddings:
        raise ValueError(f"Qwen3.5 num_position_embeddings must be a perfect square, got {num_position_embeddings}")

    h_coords = torch.linspace(0, grid_side - 1, h, device=device, dtype=torch.float32)
    w_coords = torch.linspace(0, grid_side - 1, w, device=device, dtype=torch.float32)
    h_floor = h_coords.to(torch.int64)
    w_floor = w_coords.to(torch.int64)
    h_ceil = (h_floor + 1).clamp(max=grid_side - 1)
    w_ceil = (w_floor + 1).clamp(max=grid_side - 1)
    dh = h_coords - h_floor
    dw = w_coords - w_floor

    base_h = h_floor[:, None] * grid_side
    base_h_ceil = h_ceil[:, None] * grid_side
    ids = torch.stack(
        (
            base_h + w_floor[None, :],
            base_h + w_ceil[None, :],
            base_h_ceil + w_floor[None, :],
            base_h_ceil + w_ceil[None, :],
        )
    )
    weights = torch.stack(
        (
            (1 - dh)[:, None] * (1 - dw)[None, :],
            (1 - dh)[:, None] * dw[None, :],
            dh[:, None] * (1 - dw)[None, :],
            dh[:, None] * dw[None, :],
        )
    )
    ids = _merge_major(ids, t, h, w, spatial_merge_size).reshape(4, -1)
    weights = _merge_major(weights, t, h, w, spatial_merge_size).reshape(4, -1).to(dtype)

    padded_ids = torch.zeros((4, capacity), dtype=torch.int64, device=device)
    padded_weights = torch.zeros((4, capacity), dtype=dtype, device=device)
    padded_ids[:, : ids.shape[1]] = ids
    padded_weights[:, : weights.shape[1]] = weights
    return padded_ids, padded_weights


def _rotary_position_inputs(
    *,
    t: int,
    h: int,
    w: int,
    capacity: int,
    spatial_merge_size: int,
    device: torch.device,
) -> torch.Tensor:
    rows = torch.arange(h, device=device).view(h, 1).expand(h, w)
    cols = torch.arange(w, device=device).view(1, w).expand(h, w)
    rows = _merge_major(rows, t, h, w, spatial_merge_size).reshape(-1)
    cols = _merge_major(cols, t, h, w, spatial_merge_size).reshape(-1)

    rotary_position_ids = torch.zeros((2, capacity), dtype=torch.int64, device=device)
    rotary_position_ids[0, : rows.shape[0]] = rows
    rotary_position_ids[1, : cols.shape[0]] = cols
    return rotary_position_ids


def build_visual_token_gear_inputs(
    grid_thw: torch.Tensor,
    *,
    patch_capacity: int,
    num_position_embeddings: int,
    spatial_merge_size: int = 2,
    dtype: torch.dtype = torch.float16,
    device: torch.device | str | None = None,
    rotary_cache_length: int | None = None,
) -> dict[str, torch.Tensor | int]:
    """Build compact geometry inputs for one padded static token gear.

    Qwen3.5's current image path has ``grid_t == 1`` and each graph invocation
    handles one image.  A compact broadcast bias ``[1, 1, 1, Ncap]`` is
    therefore sufficient: valid queries see every real patch and no right
    padding.  ``xhquant.nn.MaskedAdd`` applies this bias twice with saturation
    before ``xhquant.nn.Softmax``.
    """

    if grid_thw.numel() != 3:
        raise ValueError(f"one visual gear invocation handles exactly one image, got {tuple(grid_thw.shape)}")
    t, h, w = (int(value) for value in grid_thw.reshape(-1).tolist())
    if t != 1:
        raise ValueError(f"Qwen3.5 image token gears currently require grid_t == 1, got {t}")
    if h <= 0 or w <= 0:
        raise ValueError(f"grid height and width must be positive, got {(h, w)}")
    if h % spatial_merge_size or w % spatial_merge_size:
        raise ValueError(f"grid {(t, h, w)} must be divisible by spatial_merge_size={spatial_merge_size}")

    valid_patch_tokens = t * h * w
    patch_capacity = int(patch_capacity)
    merge_unit = spatial_merge_size * spatial_merge_size
    if patch_capacity % merge_unit:
        raise ValueError(f"patch capacity {patch_capacity} must be divisible by spatial merge unit {merge_unit}")
    if valid_patch_tokens > patch_capacity:
        raise ValueError(f"image grid needs {valid_patch_tokens} patches, which exceeds gear capacity {patch_capacity}")

    target_device = torch.device(device) if device is not None else grid_thw.device
    position_ids, position_weights = _position_interpolation_inputs(
        t=t,
        h=h,
        w=w,
        capacity=patch_capacity,
        num_position_embeddings=num_position_embeddings,
        spatial_merge_size=spatial_merge_size,
        dtype=dtype,
        device=target_device,
    )
    rotary_position_ids = _rotary_position_inputs(
        t=t,
        h=h,
        w=w,
        capacity=patch_capacity,
        spatial_merge_size=spatial_merge_size,
        device=target_device,
    )
    if rotary_cache_length is not None:
        rotary_cache_length = int(rotary_cache_length)
        if rotary_cache_length <= 0:
            raise ValueError(f"rotary_cache_length must be positive, got {rotary_cache_length}")
        max_position = max(h, w) - 1
        if max_position >= rotary_cache_length:
            raise ValueError(
                f"image grid {(h, w)} needs rotary position {max_position}, "
                f"but cache length is {rotary_cache_length}"
            )
    if not torch.empty((), dtype=dtype).is_floating_point():
        raise ValueError(f"attention mask dtype must be floating point, got {dtype}")
    attention_mask = torch.zeros((1, 1, 1, patch_capacity), dtype=dtype, device=target_device)
    attention_mask[..., valid_patch_tokens:] = -torch.finfo(dtype).max
    return {
        "position_ids": position_ids,
        "position_weights": position_weights,
        "rotary_position_ids": rotary_position_ids,
        "attention_mask": attention_mask,
        "valid_patch_tokens": valid_patch_tokens,
        "valid_image_tokens": valid_patch_tokens // merge_unit,
    }
