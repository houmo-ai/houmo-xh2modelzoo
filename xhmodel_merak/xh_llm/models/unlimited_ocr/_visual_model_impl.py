from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from xhquant.utils.registry import DynamicModule

from ...register import XHLLM_TRACEABLE_MODULES
from .deepencoder import Attention, Block, CLIPVisionEmbeddings, ImageEncoderViT, NoTPAttention, NoTPFeedForward
from .unlimited_ocr_visual_model import UnlimitedOCRBaseVisualModel


def _as_dict(cfg: Any) -> Dict[str, Any]:
    if cfg is None:
        return {}
    if isinstance(cfg, dict):
        return cfg
    if hasattr(cfg, "to_dict"):
        return cfg.to_dict()
    return dict(cfg)


def _get_cfg_int(cfg: Dict[str, Any], name: str, default: int) -> int:
    return int(cfg.get(name, default))


def _resize_sam_abs_pos(abs_pos: Tensor, src_size: int, tgt_size: int) -> Tensor:
    if src_size == tgt_size:
        return abs_pos
    dtype = abs_pos.dtype
    pos = abs_pos.permute(0, 3, 1, 2).float()
    pos = F.interpolate(pos, size=(tgt_size, tgt_size), mode="bicubic", antialias=True, align_corners=False)
    return pos.to(dtype).permute(0, 2, 3, 1)


def _resize_clip_abs_pos(abs_pos: Tensor, src_tokens: int, tgt_tokens: int, dim: int) -> Tensor:
    if src_tokens == tgt_tokens:
        return abs_pos

    abs_pos = abs_pos.squeeze(0)
    cls_token, old_pos_embed = abs_pos[:1], abs_pos[1:]
    src_size = int(math.sqrt(src_tokens - 1))
    tgt_size = int(math.sqrt(tgt_tokens - 1))
    dtype = old_pos_embed.dtype

    old_pos_embed = old_pos_embed.view(1, src_size, src_size, dim).permute(0, 3, 1, 2).float()
    new_pos_embed = F.interpolate(
        old_pos_embed,
        size=(tgt_size, tgt_size),
        mode="bicubic",
        antialias=True,
        align_corners=False,
    ).to(dtype)
    new_pos_embed = new_pos_embed.permute(0, 2, 3, 1).reshape(tgt_size * tgt_size, dim)
    return torch.cat([cls_token, new_pos_embed], dim=0).view(1, tgt_size * tgt_size + 1, dim)


def _window_partition_static(x: Tensor, window_size: int, input_hw: Tuple[int, int]) -> Tuple[Tensor, Tuple[int, int]]:
    batch_size, _, _, channels = x.shape
    height, width = input_hw
    pad_h = (window_size - height % window_size) % window_size
    pad_w = (window_size - width % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))

    padded_h, padded_w = height + pad_h, width + pad_w
    x = x.view(batch_size, padded_h // window_size, window_size, padded_w // window_size, window_size, channels)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, channels)
    return windows, (padded_h, padded_w)


def _window_unpartition_static(
    windows: Tensor,
    window_size: int,
    pad_hw: Tuple[int, int],
    input_hw: Tuple[int, int],
    batch_size: int,
) -> Tensor:
    padded_h, padded_w = pad_hw
    height, width = input_hw
    x = windows.view(batch_size, padded_h // window_size, padded_w // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(batch_size, padded_h, padded_w, -1)
    return x[:, :height, :width, :].contiguous()


def _get_rel_pos(q_size: int, k_size: int, rel_pos: Tensor, rel_pos_len: int) -> Tensor:
    max_rel_dist = 2 * max(q_size, k_size) - 1
    if rel_pos_len != max_rel_dist:
        dtype = rel_pos.dtype
        rel_pos = F.interpolate(
            rel_pos.reshape(1, rel_pos_len, -1).permute(0, 2, 1).float(),
            size=max_rel_dist,
            mode="linear",
        ).to(dtype)
        rel_pos = rel_pos.reshape(-1, max_rel_dist).permute(1, 0)

    q_coords = torch.arange(q_size)[:, None].to(rel_pos) * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :].to(rel_pos) * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)
    return rel_pos[relative_coords.long()]


def _add_decomposed_rel_pos(
    q: Tensor,
    rel_pos_h: Tensor,
    rel_pos_w: Tensor,
    q_size: Tuple[int, int],
    k_size: Tuple[int, int],
    rel_pos_h_len: int,
    rel_pos_w_len: int,
) -> Tuple[Tensor, Tensor]:
    q_h, q_w = q_size
    k_h, k_w = k_size
    rh = _get_rel_pos(q_h, k_h, rel_pos_h, rel_pos_h_len)
    rw = _get_rel_pos(q_w, k_w, rel_pos_w, rel_pos_w_len)

    batch_heads, _, dim = q.shape
    r_q = q.reshape(batch_heads, q_h, q_w, dim)
    # Height bias is indexed by query-height: broadcasting matmul aligns the
    # q_h axis (dim 1) against rh's leading q_h, matching einsum "bhwc,hkc->bhwk".
    rel_h = torch.matmul(r_q, rh.permute(0, 2, 1)).unsqueeze(-1)
    # Width bias is indexed by query-width, so the contraction axis is q_w (dim 2).
    # A plain matmul on r_q would wrongly reuse the q_h alignment; move q_w to the
    # leading batch position so matmul aligns it against rw's leading q_w
    # (einsum "bhwc,wkc->bhwk"), then restore the original layout.
    r_q_w = r_q.permute(0, 2, 1, 3)  # (bh, q_w, q_h, dim)
    rel_w = torch.matmul(r_q_w, rw.permute(0, 2, 1))  # (bh, q_w, q_h, k_w)
    rel_w = rel_w.permute(0, 2, 1, 3).unsqueeze(-2)  # (bh, q_h, q_w, 1, k_w)
    rel_h = rel_h.reshape(batch_heads, q_h * q_w, k_h, 1)
    rel_w = rel_w.reshape(batch_heads, q_h * q_w, 1, k_w)
    return rel_h, rel_w


@XHLLM_TRACEABLE_MODULES.register_module({UnlimitedOCRBaseVisualModel: "UnlimitedOCRBaseVisualModel"})
class _UnlimitedOCRBaseVisualModel(DynamicModule):
    def _setup(self, cfg: Optional[Dict[str, Any]] = None):
        cfg = _as_dict(cfg)
        image_size = _get_cfg_int(cfg, "image_size", 1024)
        patch_size = _get_cfg_int(cfg, "patch_size", 16)
        downsample_ratio = _get_cfg_int(cfg, "downsample_ratio", 4)
        self.output_grid_size = image_size // patch_size // downsample_ratio
        self.batch_size = _get_cfg_int(cfg, "batch_size", 1)
        self.hidden_size = int(self.image_newline.shape[0])
        return self

    def forward(self, image: Tensor) -> Tensor:
        sam_features = self.sam_model(image)
        clip_features = self.vision_model(image, sam_features)
        sam_tokens = sam_features.flatten(2).permute(0, 2, 1)
        visual_features = torch.cat((clip_features[:, 1:], sam_tokens), dim=-1)
        visual_features = self.projector(visual_features)

        grid_size = self.output_grid_size
        batch_size = self.batch_size
        hidden_size = self.hidden_size
        visual_features = visual_features.view(batch_size, grid_size, grid_size, hidden_size)
        newline = self.image_newline.view(1, 1, 1, hidden_size).expand(batch_size, grid_size, 1, hidden_size)
        visual_features = torch.cat([visual_features, newline], dim=2).reshape(batch_size, -1, hidden_size)
        separator = self.view_seperator.view(1, 1, hidden_size).expand(batch_size, 1, hidden_size)
        return torch.cat([visual_features, separator], dim=1)


@XHLLM_TRACEABLE_MODULES.register_module({ImageEncoderViT: "unlimited_ocr.sam.ImageEncoderViT"})
class _ImageEncoderViT(DynamicModule):
    def _setup(self, cfg: Optional[Dict[str, Any]] = None):
        cfg = _as_dict(cfg)
        self.input_grid_size = _get_cfg_int(cfg, "image_size", 1024) // _get_cfg_int(cfg, "patch_size", 16)
        self.pos_embed_grid_size = int(self.pos_embed.shape[1]) if self.pos_embed is not None else self.input_grid_size
        return self

    def forward(self, x: Tensor) -> Tensor:
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + _resize_sam_abs_pos(self.pos_embed, self.pos_embed_grid_size, self.input_grid_size)

        for block in self.blocks:
            x = block(x)

        x = self.neck(x.permute(0, 3, 1, 2))
        x2 = self.net_2(x)
        return self.net_3(x2)


@XHLLM_TRACEABLE_MODULES.register_module({Block: "unlimited_ocr.sam.Block"})
class _Block(DynamicModule):
    def _setup(self, cfg: Optional[Dict[str, Any]] = None):
        cfg = _as_dict(cfg)
        image_size = _get_cfg_int(cfg, "image_size", 1024)
        patch_size = _get_cfg_int(cfg, "patch_size", 16)
        grid_size = image_size // patch_size
        self.input_hw = (grid_size, grid_size)
        self.batch_size = _get_cfg_int(cfg, "batch_size", 1)
        return self

    def forward(self, x: Tensor) -> Tensor:
        shortcut = x
        x = self.norm1(x)
        if self.window_size > 0:
            x, pad_hw = _window_partition_static(x, self.window_size, self.input_hw)

        x = self.attn(x)

        if self.window_size > 0:
            x = _window_unpartition_static(x, self.window_size, pad_hw, self.input_hw, self.batch_size)

        x = shortcut + x
        return x + self.mlp(self.norm2(x))


@XHLLM_TRACEABLE_MODULES.register_module({Attention: "unlimited_ocr.sam.Attention"})
class _Attention(DynamicModule):
    def _setup(self, cfg: Optional[Dict[str, Any]] = None):
        cfg = _as_dict(cfg)
        image_size = _get_cfg_int(cfg, "image_size", 1024)
        patch_size = _get_cfg_int(cfg, "patch_size", 16)
        grid_size = image_size // patch_size
        self.rel_pos_h_len = int(self.rel_pos_h.shape[0]) if self.use_rel_pos else 0
        self.rel_pos_w_len = int(self.rel_pos_w.shape[0]) if self.use_rel_pos else 0
        if self.use_rel_pos:
            self.input_hw = ((self.rel_pos_h_len + 1) // 2, (self.rel_pos_w_len + 1) // 2)
        else:
            self.input_hw = (grid_size, grid_size)

        height, width = self.input_hw
        batch_size = _get_cfg_int(cfg, "batch_size", 1)
        if height != grid_size or width != grid_size:
            num_windows_h = math.ceil(grid_size / height)
            num_windows_w = math.ceil(grid_size / width)
            batch_size *= num_windows_h * num_windows_w
        self.input_batch_size = batch_size
        self.input_batch_heads = batch_size * self.num_heads
        return self

    def forward(self, x: Tensor) -> Tensor:
        batch_size = self.input_batch_size
        batch_heads = self.input_batch_heads
        height, width = self.input_hw
        qkv = self.qkv(x).reshape(batch_size, height * width, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        qkv = qkv.reshape(3, batch_heads, height * width, -1)
        q = qkv[0]
        k = qkv[1]
        v = qkv[2]

        rel_h, rel_w = None, None
        if self.use_rel_pos:
            rel_h, rel_w = _add_decomposed_rel_pos(
                q,
                self.rel_pos_h,
                self.rel_pos_w,
                self.input_hw,
                self.input_hw,
                self.rel_pos_h_len,
                self.rel_pos_w_len,
            )

        q = q.view(batch_size, self.num_heads, height * width, -1)
        k = k.view(batch_size, self.num_heads, height * width, -1)
        v = v.view(batch_size, self.num_heads, height * width, -1)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if self.use_rel_pos:
            rel_h = rel_h.view(batch_size, self.num_heads, height * width, height, 1)
            rel_w = rel_w.view(batch_size, self.num_heads, height * width, 1, width)
            attn_bias = (rel_h + rel_w).view(batch_size, self.num_heads, height * width, height * width)
            attn_weights = attn_weights + attn_bias

        attn_weights = torch.softmax(attn_weights, dim=-1)
        x = torch.matmul(attn_weights, v)
        x = x.view(batch_size, self.num_heads, height, width, -1).permute(0, 2, 3, 1, 4).reshape(batch_size, height, width, -1)
        return self.proj(x)


@XHLLM_TRACEABLE_MODULES.register_module({CLIPVisionEmbeddings: "unlimited_ocr.clip.CLIPVisionEmbeddings"})
class _CLIPVisionEmbeddings(DynamicModule):
    def _setup(self, cfg: Optional[Dict[str, Any]] = None):
        cfg = _as_dict(cfg)
        self.batch_size = _get_cfg_int(cfg, "batch_size", 1)
        self.num_positions_static = int(self.position_ids.shape[-1])
        self.src_num_positions_static = int(self.position_embedding.num_embeddings)
        self.position_embed_dim_static = int(self.position_embedding.embedding_dim)
        return self

    def forward(self, pixel_values: Tensor, patch_embeds: Optional[Tensor]) -> Tensor:
        batch_size = self.batch_size
        if patch_embeds is None:
            patch_embeds = self.patch_embedding(pixel_values)
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)
        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        position_embeddings = _resize_clip_abs_pos(
            self.position_embedding(self.position_ids),
            self.src_num_positions_static,
            self.num_positions_static,
            self.position_embed_dim_static,
        )
        return embeddings + position_embeddings


@XHLLM_TRACEABLE_MODULES.register_module({NoTPFeedForward: "unlimited_ocr.clip.NoTPFeedForward"})
class _NoTPFeedForward(DynamicModule):
    def _setup(self, cfg: Optional[Dict[str, Any]] = None):
        return self

    def forward(self, x: Tensor) -> Tensor:
        hidden = self.fc1(x)
        return self.fc2(hidden * torch.sigmoid(1.702 * hidden))


@XHLLM_TRACEABLE_MODULES.register_module({NoTPAttention: "unlimited_ocr.clip.NoTPAttention"})
class _NoTPAttention(DynamicModule):
    def _setup(self, cfg: Optional[Dict[str, Any]] = None):
        cfg = _as_dict(cfg)
        self.batch_size = _get_cfg_int(cfg, "batch_size", 1)
        self.seq_len = int(self.max_seq_len) + 1
        self.attention_scale = 1.0 / math.sqrt(self.head_dim)
        return self

    def forward(self, x: Tensor) -> Tensor:
        batch_size = self.batch_size
        seq_len = self.seq_len
        qkv = self.qkv_proj(x).view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        query, key, value = torch.split(qkv, 1, dim=2)
        query = query.squeeze(2).permute(0, 2, 1, 3)
        key = key.squeeze(2).permute(0, 2, 1, 3)
        value = value.squeeze(2).permute(0, 2, 1, 3)

        attn_weights = torch.matmul(query, key.transpose(-2, -1)) * self.attention_scale
        attn_weights = torch.softmax(attn_weights, dim=-1)
        output = torch.matmul(attn_weights, value)
        output = output.permute(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        return self.out_proj(output)


def register_wrap_cls(hf_model=None):
    return hf_model
