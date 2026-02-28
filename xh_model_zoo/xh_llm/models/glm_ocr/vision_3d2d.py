from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class GlmOcrVisionPatchEmbed2D(nn.Module):
    """
    Replace Conv3d patch embedding with equivalent stacked Conv2d ops.

    Original Conv3d kernel shape: [out_channels, in_channels, t, p, p]
    For an input chunk [C, t, p, p], output is:
      sum_i Conv2d_i(frame_i) + bias
    """

    def __init__(self, patch_embed):
        super().__init__()
        self.patch_size = patch_embed.patch_size
        self.temporal_patch_size = patch_embed.temporal_patch_size
        self.in_channels = patch_embed.in_channels
        self.embed_dim = patch_embed.embed_dim

        proj3d = patch_embed.proj
        if not isinstance(proj3d, nn.Conv3d):
            raise TypeError(f"Expected Conv3d, got {type(proj3d)}")

        self.proj2d = nn.ModuleList(
            [
                nn.Conv2d(
                    self.in_channels,
                    self.embed_dim,
                    kernel_size=(self.patch_size, self.patch_size),
                    stride=(self.patch_size, self.patch_size),
                    bias=False,
                    device=proj3d.weight.device,
                    dtype=proj3d.weight.dtype,
                )
                for _ in range(self.temporal_patch_size)
            ]
        )
        for i in range(self.temporal_patch_size):
            self.proj2d[i].weight.data.copy_(proj3d.weight[:, :, i, :, :].contiguous())

        if proj3d.bias is not None:
            self.bias = nn.Parameter(proj3d.bias.data.clone())
        else:
            self.bias = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj2d[0].weight.dtype
        hidden_states = hidden_states.view(
            -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
        )
        hidden_states = hidden_states.to(dtype=target_dtype)

        out = None
        for i in range(self.temporal_patch_size):
            term = self.proj2d[i](hidden_states[:, :, i, :, :])
            out = term if out is None else (out + term)
        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1, 1)
        return out.view(-1, self.embed_dim)


def _resolve_visual_module(root_model: Any):
    # HF multimodal wrapper: model.model.visual
    if hasattr(root_model, "model") and hasattr(root_model.model, "visual"):
        return root_model.model.visual
    # vision-only model wrapper: model.visual
    if hasattr(root_model, "visual"):
        return root_model.visual
    return None


def replace_patch_embed_3d_with_2d_(root_model, logger=None) -> bool:
    visual = _resolve_visual_module(root_model)
    if visual is None:
        if logger is not None:
            logger.warning("visual module not found, skip 3d->2d replacement.")
        return False

    patch_embed = getattr(visual, "patch_embed", None)
    if patch_embed is None:
        if logger is not None:
            logger.warning("visual.patch_embed not found, skip 3d->2d replacement.")
        return False

    proj = getattr(patch_embed, "proj", None)
    if not isinstance(proj, nn.Conv3d):
        if logger is not None:
            logger.info("visual.patch_embed.proj is not Conv3d, skip 3d->2d replacement.")
        return False

    visual.patch_embed = GlmOcrVisionPatchEmbed2D(patch_embed).to(proj.weight.device, dtype=proj.weight.dtype)
    if logger is not None:
        logger.info(
            f"Replace visual.patch_embed Conv3d(t={patch_embed.temporal_patch_size}, p={patch_embed.patch_size}) "
            "with Conv2d stack."
        )
    return True

