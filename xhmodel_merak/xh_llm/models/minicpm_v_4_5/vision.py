"""Static-capacity Vision model and input helpers for MiniCPM-V-4.5.

MiniCPM-V-4.5 encodes each image slice with SigLIP2 (``vpm``) followed by a
64-query 2D perceiver ``resampler``.  The exported graph keeps the slice grid
fixed at ``patch_capacity``: the host pads the packed pixel tensor, computes the
bucketized 2D position ids, the SigLIP prefix attention bias, the resampler
2D-sincos position embedding, and the resampler key-padding bias.  All five
inputs are static tensors, so the graph contains no dynamic-shape operation and
stays numerically identical to the native encoder (no padding leakage).
"""

from __future__ import annotations

import torch
import torch.nn as nn


PATCH_SIZE = 14
DEFAULT_PATCH_CAPACITY = 1600  # 40 x 40 patch grid
DEFAULT_GROUP_CAPACITY = 6  # official 3D-Resampler packing range is 1-6 frames


def _vision_attention(
    attention: nn.Module,
    hidden_states: torch.Tensor,
    attention_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Eager SigLIP attention with an optional prefix bias on the key axis."""
    batch_size, sequence_length, hidden_size = hidden_states.shape
    num_heads = int(attention.num_heads)
    head_dim = int(attention.head_dim)
    head_shape = (batch_size, sequence_length, num_heads, head_dim)

    query = attention.q_proj(hidden_states).view(head_shape).transpose(1, 2)
    key = attention.k_proj(hidden_states).view(head_shape).transpose(1, 2)
    value = attention.v_proj(hidden_states).view(head_shape).transpose(1, 2)
    scores = torch.matmul(query, key.transpose(2, 3)) * float(attention.scale)
    if attention_bias is not None:
        scores = scores + attention_bias.to(dtype=scores.dtype)
    probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    output = torch.matmul(probabilities, value)
    output = (
        output.transpose(1, 2)
        .contiguous()
        .view(
            batch_size,
            sequence_length,
            hidden_size,
        )
    )
    return attention.out_proj(output)


def _vision_encoder_layer(
    layer: nn.Module,
    hidden_states: torch.Tensor,
    attention_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    residual = hidden_states
    hidden_states = layer.layer_norm1(hidden_states)
    hidden_states = residual + _vision_attention(
        layer.self_attn,
        hidden_states,
        attention_bias,
    )

    residual = hidden_states
    hidden_states = layer.layer_norm2(hidden_states)
    return residual + layer.mlp(hidden_states)


def _resampler_attention(
    attention: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    key_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Eager scaled dot-product attention for the 64-query resampler.

    ``query`` is the learnable 64 queries ``[Q, D]``; ``key``/``value`` are the
    fixed-capacity patch embeddings ``[1, P, D]``.
    """
    batch_size, sequence_length, hidden_size = key.shape
    num_heads = int(attention.num_heads)
    head_dim = int(attention.embed_dim // attention.num_heads)
    q_head_shape = (1, query.shape[0], num_heads, head_dim)
    kv_head_shape = (batch_size, sequence_length, num_heads, head_dim)

    in_proj_weight = attention.in_proj_weight
    in_proj_bias = attention.in_proj_bias
    w_q, w_k, w_v = in_proj_weight.chunk(3, dim=0)
    b_q, b_k, b_v = None if in_proj_bias is None else in_proj_bias.chunk(3, dim=0)

    q_states = nn.functional.linear(query, w_q, b_q).view(q_head_shape).transpose(1, 2)
    k_states = nn.functional.linear(key, w_k, b_k).view(kv_head_shape).transpose(1, 2)
    v_states = nn.functional.linear(value, w_v, b_v).view(kv_head_shape).transpose(1, 2)

    scores = torch.matmul(q_states, k_states.transpose(2, 3)) * (float(head_dim) ** -0.5)
    if key_bias is not None:
        scores = scores + key_bias.to(dtype=scores.dtype)
    probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q_states.dtype)
    output = torch.matmul(probabilities, v_states)
    output = output.transpose(1, 2).contiguous().view(1, query.shape[0], hidden_size)
    return attention.out_proj(output)


class MiniCPMV45VisionExportModel(nn.Module):
    """Static wrapper whose shape is defined by a fixed patch-token capacity."""

    _vision_attention = staticmethod(_vision_attention)
    _vision_encoder_layer = staticmethod(_vision_encoder_layer)
    _resampler_attention = staticmethod(_resampler_attention)

    def __init__(
        self,
        vpm: nn.Module,
        resampler: nn.Module,
        patch_capacity: int,
    ):
        super().__init__()
        if patch_capacity <= 0:
            raise ValueError(f"patch_capacity must be positive, got {patch_capacity}")
        self.patch_capacity = int(patch_capacity)
        self.positions_per_side = int(vpm.embeddings.num_patches_per_side)

        self.patch_embedding = vpm.embeddings.patch_embedding
        self.position_embedding = vpm.embeddings.position_embedding
        self.encoder_layers = vpm.encoder.layers
        self.post_layernorm = vpm.post_layernorm

        # Resampler submodules; the 2D sincos position embedding and the key
        # padding mask are supplied by the host as static inputs.
        self.resampler_query = resampler.query
        self.resampler_ln_q = resampler.ln_q
        self.resampler_kv_proj = resampler.kv_proj
        self.resampler_ln_kv = resampler.ln_kv
        self.resampler_attn = resampler.attn
        self.resampler_ln_post = resampler.ln_post
        self.resampler_proj = resampler.proj

    def forward(
        self,
        pixel_values: torch.Tensor,
        position_ids: torch.Tensor,
        attention_bias: torch.Tensor,
        resampler_pos_embed: torch.Tensor,
        resampler_key_bias: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.patch_embedding(pixel_values.to(dtype=self.patch_embedding.weight.dtype))
        hidden_states = hidden_states.flatten(2).transpose(1, 2)
        hidden_states = hidden_states + self.position_embedding(position_ids)

        for layer in self.encoder_layers:
            hidden_states = self._vision_encoder_layer(
                layer,
                hidden_states,
                attention_bias,
            )
        hidden_states = self.post_layernorm(hidden_states)

        hidden_states = self.resampler_kv_proj(hidden_states)
        hidden_states = self.resampler_ln_kv(hidden_states)
        key = hidden_states + resampler_pos_embed.to(dtype=hidden_states.dtype)
        query = self.resampler_ln_q(self.resampler_query)
        output = self._resampler_attention(
            self.resampler_attn,
            query,
            key,
            hidden_states,
            resampler_key_bias,
        )
        output = self.resampler_ln_post(output)
        return output @ self.resampler_proj


def build_vision_export_model(
    full_model,
    patch_capacity: int,
) -> MiniCPMV45VisionExportModel:
    return MiniCPMV45VisionExportModel(
        vpm=full_model.vpm,
        resampler=full_model.resampler,
        patch_capacity=patch_capacity,
    ).eval()


class MiniCPMV45VideoGroupVisionExportModel(nn.Module):
    """Static wrapper for one temporal group of up to ``group_capacity`` frames.

    The official 3D-Resampler merges the patches of all frames in a temporal
    group into one key sequence and jointly compresses them into 64 tokens.
    This wrapper keeps the SigLIP encoder per-frame (batched, no cross-frame
    attention, matching the native per-slice batch semantics) and merges only
    in the Resampler: keys are ``kv + 2D sincos + 1D temporal sincos`` and the
    host supplies the merged per-frame-block key-padding bias.
    """

    def __init__(
        self,
        vpm: nn.Module,
        resampler: nn.Module,
        patch_capacity: int,
        group_capacity: int = DEFAULT_GROUP_CAPACITY,
    ):
        super().__init__()
        if patch_capacity <= 0 or group_capacity <= 0:
            raise ValueError(
                f"patch_capacity and group_capacity must be positive, got {patch_capacity}, {group_capacity}"
            )
        self.patch_capacity = int(patch_capacity)
        self.group_capacity = int(group_capacity)
        self.positions_per_side = int(vpm.embeddings.num_patches_per_side)

        self.patch_embedding = vpm.embeddings.patch_embedding
        self.position_embedding = vpm.embeddings.position_embedding
        self.encoder_layers = vpm.encoder.layers
        self.post_layernorm = vpm.post_layernorm

        self.resampler_query = resampler.query
        self.resampler_ln_q = resampler.ln_q
        self.resampler_kv_proj = resampler.kv_proj
        self.resampler_ln_kv = resampler.ln_kv
        self.resampler_attn = resampler.attn
        self.resampler_ln_post = resampler.ln_post
        self.resampler_proj = resampler.proj

    def forward(
        self,
        pixel_values: torch.Tensor,
        position_ids: torch.Tensor,
        attention_bias: torch.Tensor,
        resampler_pos_embed: torch.Tensor,
        resampler_temporal_pos_embed: torch.Tensor,
        resampler_key_bias: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.patch_embedding(pixel_values.to(dtype=self.patch_embedding.weight.dtype))
        hidden_states = hidden_states.flatten(2).transpose(1, 2)
        hidden_states = hidden_states + self.position_embedding(position_ids)

        for layer in self.encoder_layers:
            hidden_states = _vision_encoder_layer(
                layer,
                hidden_states,
                attention_bias,
            )
        hidden_states = self.post_layernorm(hidden_states)

        hidden_states = self.resampler_kv_proj(hidden_states)
        hidden_states = self.resampler_ln_kv(hidden_states)
        merged_length = self.group_capacity * self.patch_capacity
        key = (
            hidden_states
            + resampler_pos_embed.to(dtype=hidden_states.dtype)
            + resampler_temporal_pos_embed.to(dtype=hidden_states.dtype)
        ).reshape(1, merged_length, -1)
        value = hidden_states.reshape(1, merged_length, -1)
        query = self.resampler_ln_q(self.resampler_query)
        output = _resampler_attention(
            self.resampler_attn,
            query,
            key,
            value,
            resampler_key_bias,
        )
        output = self.resampler_ln_post(output)
        return output @ self.resampler_proj


def build_video_group_vision_export_model(
    full_model,
    patch_capacity: int,
    group_capacity: int = DEFAULT_GROUP_CAPACITY,
) -> MiniCPMV45VideoGroupVisionExportModel:
    return MiniCPMV45VideoGroupVisionExportModel(
        vpm=full_model.vpm,
        resampler=full_model.resampler,
        patch_capacity=patch_capacity,
        group_capacity=group_capacity,
    ).eval()


def validate_patch_capacity(target_size: tuple[int, int], patch_capacity: int) -> None:
    """Validate one native patch grid against the fixed capacity."""
    height, width = target_size
    if min(height, width, patch_capacity) <= 0:
        raise ValueError(
            f"Target dimensions and capacity must be positive, got "
            f"target={target_size}, patch_capacity={patch_capacity}"
        )
    token_count = height * width
    if token_count > patch_capacity:
        raise ValueError(f"Target {target_size} needs {token_count} patch tokens, exceeding capacity {patch_capacity}")


def build_position_ids(
    target_size: tuple[int, int],
    patch_capacity: int,
    positions_per_side: int,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Build bucketized 2D position ids in raster order (native Navit)."""
    validate_patch_capacity(target_size, patch_capacity)
    height, width = target_size
    boundaries = torch.arange(
        1.0 / positions_per_side,
        1.0,
        1.0 / positions_per_side,
        device=device,
    )
    fractional_h = torch.arange(0, 1 - 1e-6, 1.0 / height, device=device)
    fractional_w = torch.arange(0, 1 - 1e-6, 1.0 / width, device=device)
    bucket_h = torch.bucketize(fractional_h, boundaries, right=True)
    bucket_w = torch.bucketize(fractional_w, boundaries, right=True)
    raster_position_ids = (bucket_h[:, None] * positions_per_side + bucket_w).flatten().to(torch.int32)
    position_ids = torch.zeros(
        (patch_capacity,),
        dtype=torch.int32,
        device=device,
    )
    position_ids[: height * width] = raster_position_ids
    return position_ids.unsqueeze(0)


def build_prefix_bias(
    valid_length: int,
    capacity: int,
    dtype: torch.dtype,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Mask the padded tail with a [1, 1, 1, C] broadcastable prefix bias."""
    if valid_length < 0 or valid_length > capacity:
        raise ValueError(f"valid_length {valid_length} outside [0, {capacity}]")
    bias = torch.full(
        (capacity,),
        torch.finfo(dtype).min,
        dtype=dtype,
        device=device,
    )
    bias[:valid_length] = 0
    return bias.reshape(1, 1, 1, -1)


def build_resampler_pos_embed_cache(
    embed_dim: int,
    max_side: int,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Deterministic 2D sincos cache identical to ``Resampler._set_2d_pos_cache``.

    The cache is a pure function of ``embed_dim`` and the grid side, so the
    runtime can rebuild it without loading the checkpoint.
    """
    import numpy as np

    grid_h = np.arange(max_side, dtype=np.float32)
    grid_w = np.arange(max_side, dtype=np.float32)
    grid = np.stack(np.meshgrid(grid_w, grid_h), axis=0)  # [2, H, W]
    half_dim = embed_dim // 2
    omega = np.arange(half_dim // 2, dtype=np.float32)
    omega /= half_dim / 2.0
    omega = 1.0 / 10000.0**omega  # [D/4]

    def half(grid_axis: np.ndarray) -> np.ndarray:
        out = np.einsum("hw,d->hwd", grid_axis, omega)  # [H, W, D/4]
        return np.concatenate([np.sin(out), np.cos(out)], axis=-1)

    pos_embed = np.concatenate([half(grid[0]), half(grid[1])], axis=-1)
    return torch.from_numpy(pos_embed).float().to(device)


def build_resampler_pos_embed(
    target_size: tuple[int, int],
    patch_capacity: int,
    pos_embed_cache: torch.Tensor,
    dtype: torch.dtype,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Slice the resampler 2D-sincos cache to the real grid and pad it.

    Matches the official Resampler semantics: when the grid exceeds the base
    cache side (``_adjust_pos_cache``), the cache is rebuilt at the larger
    side instead of slicing out of bounds.
    """
    validate_patch_capacity(target_size, patch_capacity)
    height, width = target_size
    cache_h, cache_w = pos_embed_cache.shape[0], pos_embed_cache.shape[1]
    if height > cache_h or width > cache_w:
        max_side = max(height, width)
        pos_embed_cache = build_resampler_pos_embed_cache(
            pos_embed_cache.shape[-1],
            max_side,
            device=pos_embed_cache.device,
        )
    embed = pos_embed_cache[:height, :width, :].reshape(height * width, -1)
    padded = torch.zeros(
        (patch_capacity, embed.shape[-1]),
        dtype=dtype,
        device=device,
    )
    padded[: height * width, :] = embed.to(dtype=dtype)
    return padded.unsqueeze(0)


def pad_packed_pixel_values(
    pixel_values: torch.Tensor,
    patch_capacity: int,
) -> torch.Tensor:
    """Pad one packed slice tensor ``[3, 14, W]`` to ``[1, 3, 14, C*14]``."""
    if pixel_values.dim() == 4:
        if pixel_values.shape[0] != 1:
            raise ValueError(f"Expected a single slice, got shape {tuple(pixel_values.shape)}")
        pixel_values = pixel_values[0]
    if pixel_values.dim() != 3 or tuple(pixel_values.shape[:2]) != (3, PATCH_SIZE):
        raise ValueError(f"Expected packed slice [3, 14, W], got {tuple(pixel_values.shape)}")
    token_count = pixel_values.shape[-1] // PATCH_SIZE
    if token_count * PATCH_SIZE != pixel_values.shape[-1]:
        raise ValueError(f"Packed width {pixel_values.shape[-1]} is not a multiple of {PATCH_SIZE}")
    if token_count > patch_capacity:
        raise ValueError(f"Slice needs {token_count} patch tokens, exceeding capacity {patch_capacity}")
    padded = pixel_values.new_zeros((3, PATCH_SIZE, patch_capacity * PATCH_SIZE))
    padded[:, :, : pixel_values.shape[-1]] = pixel_values
    return padded.unsqueeze(0)


def prepare_vision_inputs(
    pixel_values: torch.Tensor,
    target_size: tuple[int, int],
    patch_capacity: int,
    positions_per_side: int,
    pos_embed_cache: torch.Tensor,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare the fixed-capacity Vision ABI for one image slice."""
    validate_patch_capacity(target_size, patch_capacity)
    padded_pixels = pad_packed_pixel_values(pixel_values, patch_capacity)
    position_ids = build_position_ids(
        target_size,
        patch_capacity,
        positions_per_side,
        device=pixel_values.device,
    )
    valid_tokens = target_size[0] * target_size[1]
    attention_bias = build_prefix_bias(
        valid_tokens,
        patch_capacity,
        dtype,
        device=pixel_values.device,
    )
    resampler_pos_embed = build_resampler_pos_embed(
        target_size,
        patch_capacity,
        pos_embed_cache,
        dtype,
        device=pixel_values.device,
    )
    resampler_key_bias = build_prefix_bias(
        valid_tokens,
        patch_capacity,
        dtype,
        device=pixel_values.device,
    )
    return (
        padded_pixels.to(device=pixel_values.device, dtype=dtype),
        position_ids,
        attention_bias,
        resampler_pos_embed,
        resampler_key_bias,
    )


def build_resampler_temporal_pos_embed_cache(
    embed_dim: int,
    max_temporal_size: int,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Deterministic 1D temporal sincos cache of ``Resampler.temporal_pos_embed``."""
    import numpy as np

    positions = np.arange(max_temporal_size, dtype=np.float32)
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000.0**omega  # [D/2]
    out = np.einsum("m,d->md", positions, omega)  # [M, D/2]
    pos_embed = np.concatenate([np.sin(out), np.cos(out)], axis=-1)  # [M, D]
    return torch.from_numpy(pos_embed).float().to(device)


def build_temporal_pos_embed(
    temporal_id: int,
    temporal_pos_embed_cache: torch.Tensor,
    patch_capacity: int,
    dtype: torch.dtype,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Broadcast one frame's 1D temporal embedding over its patch slots.

    Temporal id -1 (static image) maps to the zero embedding, matching the
    native Resampler behavior.
    """
    if temporal_id == -1:
        return torch.zeros(
            (1, patch_capacity, temporal_pos_embed_cache.shape[-1]),
            dtype=dtype,
            device=device,
        )
    frame_embed = temporal_pos_embed_cache[temporal_id].to(device=device, dtype=dtype)
    return frame_embed.unsqueeze(0).expand(1, patch_capacity, -1)


def prepare_video_group_inputs(
    pixel_slices: list[torch.Tensor],
    tgt_sizes: list[tuple[int, int]],
    temporal_ids: list[int],
    patch_capacity: int,
    group_capacity: int,
    positions_per_side: int,
    pos_embed_cache: torch.Tensor,
    temporal_pos_embed_cache: torch.Tensor,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare the fixed-capacity temporal-group Vision ABI.

    Frames beyond the actual group size are zero-padded and fully masked.
    ``temporal_ids`` must contain one entry per frame (use -1 for static
    images).  Returns ``(pixel_values, position_ids, attention_bias,
    resampler_pos_embed, resampler_temporal_pos_embed, resampler_key_bias)``.
    """
    if not (1 <= len(pixel_slices) <= group_capacity):
        raise ValueError(f"Temporal group needs 1..{group_capacity} frames, got {len(pixel_slices)}")
    if len(pixel_slices) != len(tgt_sizes) or len(pixel_slices) != len(temporal_ids):
        raise ValueError("pixel_slices/tgt_sizes/temporal_ids must have equal length")
    device = pixel_slices[0].device
    padded_pixels = pixel_slices[0].new_zeros((group_capacity, 3, PATCH_SIZE, patch_capacity * PATCH_SIZE))
    position_ids = torch.zeros(
        (group_capacity, patch_capacity),
        dtype=torch.int32,
        device=device,
    )
    attention_bias = torch.full(
        (group_capacity, 1, 1, patch_capacity),
        torch.finfo(dtype).min,
        dtype=dtype,
        device=device,
    )
    resampler_pos_embed = torch.zeros(
        (group_capacity, patch_capacity, pos_embed_cache.shape[-1]),
        dtype=dtype,
        device=device,
    )
    temporal_pos_embed = torch.zeros(
        (group_capacity, patch_capacity, temporal_pos_embed_cache.shape[-1]),
        dtype=dtype,
        device=device,
    )
    key_bias = torch.full(
        (1, 1, 1, group_capacity * patch_capacity),
        torch.finfo(dtype).min,
        dtype=dtype,
        device=device,
    )

    for frame_index, (pixel_slice, tgt, temporal_id) in enumerate(
        zip(pixel_slices, tgt_sizes, temporal_ids, strict=True)
    ):
        validate_patch_capacity(tgt, patch_capacity)
        padded = pad_packed_pixel_values(pixel_slice, patch_capacity)[0]
        padded_pixels[frame_index, :, :, :] = padded
        position_ids[frame_index, : tgt[0] * tgt[1]] = build_position_ids(
            tgt,
            patch_capacity,
            positions_per_side,
            device=device,
        )[0, : tgt[0] * tgt[1]]
        valid_tokens = tgt[0] * tgt[1]
        attention_bias[frame_index, 0, 0, :valid_tokens] = 0
        resampler_pos_embed[frame_index, :valid_tokens, :] = build_resampler_pos_embed(
            tgt,
            patch_capacity,
            pos_embed_cache,
            dtype,
            device=device,
        )[0, :valid_tokens, :]
        temporal_pos_embed[frame_index, :, :] = build_temporal_pos_embed(
            temporal_id,
            temporal_pos_embed_cache,
            patch_capacity,
            dtype,
            device=device,
        )[0]
        block_start = frame_index * patch_capacity
        key_bias[0, 0, 0, block_start : block_start + valid_tokens] = 0

    return (
        padded_pixels.to(device=device, dtype=dtype),
        position_ids,
        attention_bias,
        resampler_pos_embed,
        temporal_pos_embed,
        key_bias,
    )


__all__ = [
    "PATCH_SIZE",
    "DEFAULT_PATCH_CAPACITY",
    "DEFAULT_GROUP_CAPACITY",
    "MiniCPMV45VisionExportModel",
    "MiniCPMV45VideoGroupVisionExportModel",
    "build_vision_export_model",
    "build_video_group_vision_export_model",
    "build_position_ids",
    "build_prefix_bias",
    "build_resampler_pos_embed",
    "build_resampler_pos_embed_cache",
    "build_resampler_temporal_pos_embed_cache",
    "build_temporal_pos_embed",
    "pad_packed_pixel_values",
    "prepare_vision_inputs",
    "prepare_video_group_inputs",
    "validate_patch_capacity",
]
