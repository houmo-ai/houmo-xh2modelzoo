from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import onnx
import torch
from torch import Tensor

from ...hmonnx.hmonnx_model import HMONNXModel


TensorSession = Callable[..., Tensor | Sequence[Tensor]]


def _input_shapes(path: Path) -> tuple[tuple[int, ...], ...]:
    graph = onnx.load(str(path), load_external_data=False).graph
    return tuple(tuple(int(dim.dim_value or 1) for dim in value.type.tensor_type.shape.dim) for value in graph.input)


def prepare_vision_inputs(
    data: dict[str, Any],
    *,
    patch_size: int,
    num_patches_per_side: int,
    image_slice_max_size: tuple[int, int],
) -> tuple[Tensor, Tensor, Tensor, Tensor, list[int]]:
    all_pixel_values: list[Tensor] = []
    image_counts: list[int] = []
    max_height, max_width = image_slice_max_size
    max_patches = max_height * max_width
    max_length = max_patches * patch_size
    for images in data["pixel_values"]:
        image_counts.append(len(images))
        for image in images:
            flattened = image.flatten(end_dim=1).permute(1, 0)
            padded = torch.zeros(max_length, flattened.shape[1], dtype=flattened.dtype, device=flattened.device)
            padded[: flattened.shape[0]] = flattened
            all_pixel_values.append(padded)
    target_sizes = torch.vstack([value for value in data["tgt_sizes"] if isinstance(value, Tensor)]).to(torch.int32)
    if target_sizes.ndim == 1:
        target_sizes = target_sizes.unsqueeze(0)
    elif target_sizes.ndim > 2:
        target_sizes = target_sizes.reshape(target_sizes.shape[0], -1)
    target_sizes = target_sizes[:, :2]
    pixels = torch.nn.utils.rnn.pad_sequence(all_pixel_values, batch_first=True, padding_value=0.0)
    batch = pixels.shape[0]
    pixels = pixels.permute(0, 2, 1).reshape(batch, 3, patch_size, max_length)
    patch_mask = torch.zeros((batch, max_patches), dtype=torch.bool, device=pixels.device)
    for index in range(batch):
        patch_mask[index, : int(target_sizes[index, 0] * target_sizes[index, 1])] = True
    attention_mask = 1.0 - patch_mask[:, None, None, :].expand(batch, 1, max_patches, max_patches).to(torch.float16)
    attention_mask = attention_mask.masked_fill(attention_mask.to(torch.bool), torch.finfo(torch.float16).min)
    boundaries = torch.arange(1 / num_patches_per_side, 1.0, 1 / num_patches_per_side)
    position_ids = torch.zeros((batch, max_patches), dtype=torch.int64)
    for index in range(batch):
        height, width = int(target_sizes[index, 0]), int(target_sizes[index, 1])
        height_coords = torch.arange(0, 1 - 1e-6, 1 / height)
        width_coords = torch.arange(0, 1 - 1e-6, 1 / width)
        positions = (
            torch.bucketize(height_coords, boundaries, right=True)[:, None] * num_patches_per_side
            + torch.bucketize(width_coords, boundaries, right=True)
        ).flatten()
        position_ids[index, patch_mask[index].cpu()] = positions
    return pixels.to(torch.float16), position_ids, attention_mask, target_sizes, image_counts


def run_vision_slices(
    session: TensorSession,
    pixel_values: Tensor,
    position_ids: Tensor,
    attention_mask: Tensor,
    target_sizes: Tensor,
) -> Tensor:
    outputs = [
        torch.as_tensor(
            session(
                pixel_values[index : index + 1],
                position_ids[index : index + 1].to(torch.int32),
                attention_mask[index : index + 1],
                target_sizes[index : index + 1].to(torch.int32),
            )
        )
        for index in range(pixel_values.shape[0])
    ]
    return torch.cat(outputs, dim=0)


class MiniCPMO45VisionHMONNXRuntime(HMONNXModel):
    """MiniCPM input shaping on the repository's single-graph runtime."""

    def __init__(
        self,
        graph: Path,
        *,
        enable_cuda_graph: bool = False,
        enable_auto_offload: bool = False,
        enable_golden: bool = False,
        device_map: str | torch.device | list[str | torch.device] | None = None,
    ) -> None:
        self.graph = graph
        self.input_shapes = _input_shapes(graph)
        super().__init__(
            str(graph),
            enable_cuda_graph=enable_cuda_graph,
            enable_auto_offload=enable_auto_offload,
            enable_golden=enable_golden,
            device_map=device_map,
        )

    def prepare_inputs(self, data: dict[str, Any]) -> tuple[Tensor, Tensor, Tensor, Tensor, list[int]]:
        return prepare_vision_inputs(
            data,
            patch_size=self.patch_size,
            num_patches_per_side=self.num_patches_per_side,
            image_slice_max_size=tuple(self.wrap_cfg.image_slice_max_size),
        )

    def _wrap_model(
        self, pixel_values: Tensor, position_ids: Tensor, attention_mask: Tensor, target_sizes: Tensor
    ) -> Tensor:
        return self.forward(pixel_values, position_ids, attention_mask, target_sizes)

    def forward(
        self, pixel_values: Tensor, position_ids: Tensor, attention_mask: Tensor, target_sizes: Tensor
    ) -> Tensor:
        # The per-slice loop is a MiniCPM graph-shape constraint.  Graph
        # ownership and device/golden/fast behaviour remain in HMONNXModel.
        return run_vision_slices(super().forward, pixel_values, position_ids, attention_mask, target_sizes)


__all__ = ["MiniCPMO45VisionHMONNXRuntime", "prepare_vision_inputs", "run_vision_slices"]
