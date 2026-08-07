"""Token-capacity Vision model and input helpers for MiniCPM-V-4.6."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


PATCH_SIZE = 14
DEFAULT_TOKEN_CAPACITY = 1536


def _vision_attention(
    attention: nn.Module,
    hidden_states: torch.Tensor,
    attention_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    batch_size, sequence_length, hidden_size = hidden_states.shape
    num_heads = int(attention.num_heads)
    head_dim = int(attention.head_dim)
    head_shape = (batch_size, sequence_length, num_heads, head_dim)

    query = attention.q_proj(hidden_states).view(head_shape).transpose(1, 2)
    key = attention.k_proj(hidden_states).view(head_shape).transpose(1, 2)
    value = attention.v_proj(hidden_states).view(head_shape).transpose(1, 2)
    scores = torch.matmul(query, key.transpose(2, 3)) * float(attention.scaling)
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


class MiniCPMV46TokenCapacityVisionExportModel(nn.Module):
    """Static Vision wrapper whose shape is defined by patch-token capacity."""

    _attention = staticmethod(_vision_attention)
    _encoder_layer = staticmethod(_vision_encoder_layer)

    def __init__(
        self,
        vision_tower: nn.Module,
        merger: nn.Module,
        token_capacity: int,
        downsample_mode: str = "16x",
    ):
        super().__init__()
        if downsample_mode not in {"16x", "4x"}:
            raise ValueError(f"Unsupported downsample mode: {downsample_mode}")
        if token_capacity <= 0:
            raise ValueError(f"token_capacity must be positive, got {token_capacity}")

        self.token_capacity = int(token_capacity)
        self.downsample_mode = downsample_mode
        self.insert_layer_id = int(vision_tower.config.insert_layer_id)

        self.patch_embedding = vision_tower.embeddings.patch_embedding
        self.position_embedding = vision_tower.embeddings.position_embedding
        self.encoder_layers = vision_tower.encoder.layers
        self.vit_merger = vision_tower.vit_merger
        self.post_layernorm = vision_tower.post_layernorm
        self.output_merger = merger

        window_size = tuple(int(value) for value in self.vit_merger.window_kernel_size)
        if window_size != (2, 2):
            raise ValueError(f"Token-capacity Vision expects a (2, 2) ViT merger, got {window_size}")
        merge_kernel = tuple(int(value) for value in self.output_merger.merge_kernel_size)
        if merge_kernel != (2, 2):
            raise ValueError(f"Token-capacity Vision expects a (2, 2) output merger, got {merge_kernel}")

        output_merge_rounds = len(self.output_merger.mlp)
        if output_merge_rounds != 1:
            raise ValueError(
                f"MiniCPM-V-4.6 token-capacity export currently expects one output merger, got {output_merge_rounds}"
            )
        total_merge_rounds = output_merge_rounds + (1 if downsample_mode == "16x" else 0)
        output_divisor = 4**total_merge_rounds
        if self.token_capacity % output_divisor:
            raise ValueError(
                f"token_capacity={self.token_capacity} must be divisible by {output_divisor} for {downsample_mode}"
            )

    def _packed_window_merge(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Merge consecutive groups of four already ordered as 2x2 windows."""

        merger = self.vit_merger
        residual = hidden_states
        hidden_states = merger.layer_norm1(hidden_states)

        hidden_size = hidden_states.shape[-1]
        num_windows = hidden_states.shape[1] // 4
        windows = hidden_states.reshape(num_windows, 4, hidden_size)
        attended = self._attention(merger.self_attn, windows)
        hidden_states = residual + attended.reshape(1, self.token_capacity, hidden_size)

        windows = hidden_states.reshape(num_windows, 4, hidden_size)
        residual = windows.mean(dim=1)
        hidden_states = merger.pre_norm(windows.reshape(num_windows, 4 * hidden_size))
        hidden_states = merger.linear_1(hidden_states)
        hidden_states = merger.act(hidden_states)
        hidden_states = merger.linear_2(hidden_states)
        return (hidden_states + residual).unsqueeze(0)

    @staticmethod
    def _merge_consecutive(hidden_states: torch.Tensor) -> torch.Tensor:
        """Flatten each consecutive four-token 2x2 group."""

        hidden_size = hidden_states.shape[-1]
        return hidden_states[0].reshape(hidden_states.shape[1] // 4, 4 * hidden_size)

    def forward(
        self,
        pixel_values: torch.Tensor,
        position_ids: torch.Tensor,
        pre_attention_bias: torch.Tensor,
        post_attention_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = self.patch_embedding(pixel_values.to(dtype=self.patch_embedding.weight.dtype))
        hidden_states = hidden_states.flatten(2).transpose(1, 2)
        hidden_states = hidden_states + self.position_embedding(position_ids)

        use_vit_merger = self.downsample_mode == "16x"
        attention_bias = pre_attention_bias
        for layer_index, layer in enumerate(self.encoder_layers):
            hidden_states = self._encoder_layer(layer, hidden_states, attention_bias)
            if use_vit_merger and layer_index == self.insert_layer_id:
                hidden_states = self._packed_window_merge(hidden_states)
                if post_attention_bias is None:
                    raise ValueError("16x token-capacity Vision requires post_attention_bias")
                attention_bias = post_attention_bias

        hidden_states = self.post_layernorm(hidden_states)
        for downsample_mlp in self.output_merger.mlp:
            hidden_states = self._merge_consecutive(hidden_states)
            hidden_states = downsample_mlp(hidden_states).unsqueeze(0)
        return hidden_states


def build_token_capacity_vision_export_model(
    full_model,
    token_capacity: int,
    downsample_mode: str = "16x",
) -> MiniCPMV46TokenCapacityVisionExportModel:
    return MiniCPMV46TokenCapacityVisionExportModel(
        vision_tower=full_model.model.vision_tower,
        merger=full_model.model.merger,
        token_capacity=token_capacity,
        downsample_mode=downsample_mode,
    ).eval()


def validate_token_capacity(
    target_size: tuple[int, int],
    token_capacity: int,
    downsample_mode: str,
) -> None:
    """Validate one native patch grid against a fixed patch-token capacity."""

    height, width = target_size
    factor = merge_factor(downsample_mode)
    output_divisor = factor**2
    if min(height, width, token_capacity) <= 0:
        raise ValueError(
            f"Target dimensions and token capacity must be positive, got "
            f"target={target_size}, token_capacity={token_capacity}"
        )
    if height % factor or width % factor:
        raise ValueError(f"Target {target_size} must be divisible by {factor} for {downsample_mode}")
    if token_capacity % output_divisor:
        raise ValueError(f"Token capacity {token_capacity} must be divisible by {output_divisor} for {downsample_mode}")
    token_count = height * width
    if token_count > token_capacity:
        raise ValueError(f"Target {target_size} needs {token_count} patch tokens, exceeding capacity {token_capacity}")


def hierarchical_merge_permutation(
    target_size: tuple[int, int],
    downsample_mode: str,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Return raster indices ordered for consecutive 2x2 merge operations.

    Global Vision attention is permutation equivariant when patch tokens and
    position ids are permuted together. Arranging the valid prefix in this
    order lets every merger consume consecutive groups of four without a
    runtime shape-dependent Gather.
    """

    height, width = target_size
    factor = merge_factor(downsample_mode)
    if min(height, width) <= 0 or height % factor or width % factor:
        raise ValueError(
            f"Target {target_size} must have positive dimensions divisible by {factor} for {downsample_mode}"
        )
    merge_rounds = 2 if downsample_mode == "16x" else 1

    def recurse(grid_height: int, grid_width: int, rounds: int) -> torch.Tensor:
        raster = torch.arange(
            grid_height * grid_width,
            dtype=torch.long,
            device=device,
        ).reshape(grid_height, grid_width)
        windows = (
            raster.reshape(
                grid_height // 2,
                2,
                grid_width // 2,
                2,
            )
            .permute(0, 2, 1, 3)
            .reshape(-1, 4)
        )
        if rounds == 1:
            return windows.reshape(-1)
        outer_order = recurse(
            grid_height // 2,
            grid_width // 2,
            rounds - 1,
        )
        return windows.index_select(0, outer_order).reshape(-1)

    return recurse(height, width, merge_rounds)


def pad_packed_patch_tokens(
    pixel_values: torch.Tensor,
    target_size: tuple[int, int],
    token_capacity: int,
    downsample_mode: str,
) -> torch.Tensor:
    """Reorder native packed patches and append zero patch-token slots."""

    validate_token_capacity(target_size, token_capacity, downsample_mode)
    height, width = target_size
    token_count = height * width
    expected_width = token_count * PATCH_SIZE
    if pixel_values.dim() != 4 or tuple(pixel_values.shape[:3]) != (1, 3, PATCH_SIZE):
        raise ValueError(f"Expected one packed pixel tensor shaped [1, 3, 14, H*W*14], got {tuple(pixel_values.shape)}")
    if pixel_values.shape[-1] != expected_width:
        raise ValueError(f"Target {target_size} requires packed width {expected_width}, got {pixel_values.shape[-1]}")

    patches = pixel_values.reshape(
        1,
        3,
        PATCH_SIZE,
        token_count,
        PATCH_SIZE,
    )
    permutation = hierarchical_merge_permutation(
        target_size,
        downsample_mode,
        device=pixel_values.device,
    )
    ordered = patches.index_select(3, permutation)
    padded = pixel_values.new_zeros((1, 3, PATCH_SIZE, token_capacity, PATCH_SIZE))
    padded[:, :, :, :token_count, :] = ordered
    return padded.reshape(
        1,
        3,
        PATCH_SIZE,
        token_capacity * PATCH_SIZE,
    )


def build_token_capacity_position_ids(
    target_size: tuple[int, int],
    token_capacity: int,
    downsample_mode: str,
    positions_per_side: int,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Build native 2D position ids in hierarchical token order."""

    validate_token_capacity(target_size, token_capacity, downsample_mode)
    if positions_per_side <= 0:
        raise ValueError(f"positions_per_side must be positive, got {positions_per_side}")
    height, width = target_size
    fractional_h = (
        torch.arange(
            height,
            device=device,
            dtype=torch.float32,
        )
        / height
    )
    fractional_w = (
        torch.arange(
            width,
            device=device,
            dtype=torch.float32,
        )
        / width
    )
    position_h = torch.floor(fractional_h * positions_per_side).to(torch.int32)
    position_w = torch.floor(fractional_w * positions_per_side).to(torch.int32)
    raster_position_ids = (position_h[:, None] * positions_per_side + position_w).reshape(-1)
    permutation = hierarchical_merge_permutation(
        target_size,
        downsample_mode,
        device=device,
    )
    valid_position_ids = raster_position_ids.index_select(0, permutation)
    position_ids = torch.zeros(
        (token_capacity,),
        dtype=torch.int32,
        device=device,
    )
    position_ids[: height * width] = valid_position_ids
    return position_ids.unsqueeze(0)


def build_token_capacity_attention_biases(
    target_size: tuple[int, int],
    token_capacity: int,
    downsample_mode: str,
    *,
    dtype: torch.dtype,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mask the padded tail before and after the optional ViT merger."""

    validate_token_capacity(target_size, token_capacity, downsample_mode)
    height, width = target_size
    valid_tokens = height * width
    mask_value = torch.finfo(dtype).min

    def make_prefix_bias(length: int, valid_length: int) -> torch.Tensor:
        bias = torch.full(
            (length,),
            mask_value,
            dtype=dtype,
            device=device,
        )
        bias[:valid_length] = 0
        return bias.reshape(1, 1, 1, -1)

    pre_bias = make_prefix_bias(token_capacity, valid_tokens)
    post_bias = make_prefix_bias(token_capacity // 4, valid_tokens // 4)
    return pre_bias, post_bias


def prepare_token_capacity_vision_inputs(
    pixel_values: torch.Tensor,
    target_size: tuple[int, int],
    token_capacity: int,
    downsample_mode: str,
    positions_per_side: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare the fixed-capacity Vision ABI for one native image/slice."""

    padded_pixels = pad_packed_patch_tokens(
        pixel_values,
        target_size,
        token_capacity,
        downsample_mode,
    )
    position_ids = build_token_capacity_position_ids(
        target_size,
        token_capacity,
        downsample_mode,
        positions_per_side,
        device=pixel_values.device,
    )
    pre_bias, post_bias = build_token_capacity_attention_biases(
        target_size,
        token_capacity,
        downsample_mode,
        dtype=pixel_values.dtype,
        device=pixel_values.device,
    )
    return padded_pixels, position_ids, pre_bias, post_bias


def crop_token_capacity_image_embeds(
    image_embeds: torch.Tensor,
    target_size: tuple[int, int],
    token_capacity: int,
    downsample_mode: str,
) -> torch.Tensor:
    """Take the valid output prefix from a fixed patch-token-capacity graph."""

    validate_token_capacity(target_size, token_capacity, downsample_mode)
    factor = merge_factor(downsample_mode)
    output_divisor = factor**2
    expected_tokens = token_capacity // output_divisor
    valid_tokens = target_size[0] * target_size[1] // output_divisor
    if image_embeds.dim() != 3 or image_embeds.shape[0] != 1:
        raise ValueError(f"Expected token-capacity image embeddings [1, L, D], got {tuple(image_embeds.shape)}")
    if image_embeds.shape[1] != expected_tokens:
        raise ValueError(
            f"Capacity {token_capacity}/{downsample_mode} requires "
            f"{expected_tokens} output tokens, got {image_embeds.shape[1]}"
        )
    return image_embeds[:, :valid_tokens, :].contiguous()


def split_packed_pixel_values(
    pixel_values: torch.Tensor,
    target_sizes: torch.Tensor,
) -> list[torch.Tensor]:
    """Split MiniCPM-V's packed pixel tensor back into independent images/slices."""

    if pixel_values.dim() != 4 or tuple(pixel_values.shape[:3]) != (1, 3, PATCH_SIZE):
        raise ValueError(
            f"Expected packed pixel_values shaped [1, 3, 14, packed_width], got {tuple(pixel_values.shape)}"
        )
    if target_sizes.dim() != 2 or target_sizes.shape[1] != 2:
        raise ValueError(f"Expected target_sizes [N, 2], got {tuple(target_sizes.shape)}")

    packed_widths = [int(height) * int(width) * PATCH_SIZE for height, width in target_sizes.tolist()]
    if sum(packed_widths) != pixel_values.shape[-1]:
        raise ValueError(
            f"Packed width mismatch: target_sizes require {sum(packed_widths)}, "
            f"pixel_values has {pixel_values.shape[-1]}"
        )
    return list(torch.split(pixel_values, packed_widths, dim=-1))


def image_token_count(target_sizes: torch.Tensor, downsample_mode: str) -> int:
    if downsample_mode not in {"16x", "4x"}:
        raise ValueError(f"Unsupported downsample mode: {downsample_mode}")
    merge_factor = 4 if downsample_mode == "16x" else 2
    divisor = merge_factor**2
    total = 0
    for height, width in target_sizes.tolist():
        height, width = int(height), int(width)
        if height % merge_factor or width % merge_factor:
            raise ValueError(f"Patch grid {(height, width)} is incompatible with {downsample_mode}")
        total += height * width // divisor
    return total


def merge_factor(downsample_mode: str) -> int:
    if downsample_mode == "16x":
        return 4
    if downsample_mode == "4x":
        return 2
    raise ValueError(f"Unsupported downsample mode: {downsample_mode}")


def token_capacity_artifact_name(
    token_capacity: int,
    downsample_mode: str,
) -> str:
    return f"minicpm_v_4_6_vision_tokens_{token_capacity}_{downsample_mode}.onnx"


def token_capacity_hmonnx_path(
    capacity_root: Path,
    token_capacity: int,
    downsample_mode: str,
) -> Path:
    return capacity_root / downsample_mode / "hmonnx" / token_capacity_artifact_name(token_capacity, downsample_mode)
