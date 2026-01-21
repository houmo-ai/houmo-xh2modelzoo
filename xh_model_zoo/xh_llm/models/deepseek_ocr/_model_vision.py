import torch.nn as nn
import torch
import torch.nn.functional as F
import copy

from contextlib import nullcontext
import math
from typing import Optional, Tuple
# from megatron.model import LayerNorm

from einops import rearrange
from easydict import EasyDict as adict


from typing import Optional, Tuple, Type
from functools import partial


from ..builder import XHLLM_TRACEABLE_MODULES, DynamicRegister
from xhquant.nn.modules import Permute, Resize
from xh_model_zoo.xh_llm.models.deepseek_ocr.deepencoder import (
    NoTPFeedForward,
    NoTPAttention,
    CLIPVisionEmbeddings,
    Block,
    ImageEncoderViT,
    Attention
)


DType = torch.dtype

def window_partition_1(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    B, H, W, C = x.shape

    pad_h = 10
    pad_w = 10
    x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows, (Hp, Wp)

def window_partition(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Partition into non-overlapping windows with padding if needed.
    Args:
        x (tensor): input tokens with [B, H, W, C].
        window_size (int): window size.

    Returns:
        windows: windows after partition with [B * num_windows, window_size, window_size, C].
        (Hp, Wp): padded height and width before partition
    """
    B, H, W, C = x.shape

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows, (Hp, Wp)

def window_unpartition(
    windows: torch.Tensor, window_size: int, pad_hw: Tuple[int, int], hw: Tuple[int, int]
) -> torch.Tensor:
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)

    x = x[:, :H, :W, :].contiguous()
    return x

# @XHLLM_TRACEABLE_MODULES.register_module(
#     {
#         Block: "deepseek_ocr.sam.Block",
#     }
# )
class _Block(DynamicRegister):
    """Transformer blocks with support of window attention and residual propagation blocks"""

    def _setup(self, cfg):
        self.input_resolution = cfg.input_resolution
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)
        # Window partition
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            if self.input_resolution in [512,]:
                x, pad_hw = window_partition_1(x, self.window_size)
            else:
                raise NotImplementedError

        x = self.attn(x)
        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        return x


def get_abs_pos_sam(abs_pos, tgt_size):

    dtype = abs_pos.dtype

    src_size = abs_pos.size(1)

    if src_size != tgt_size:
        old_pos_embed = abs_pos.permute(0, 3, 1, 2)
        old_pos_embed = old_pos_embed.to(torch.float32)
        new_pos_embed = F.interpolate(
            old_pos_embed,
            size=(tgt_size, tgt_size),
            mode='bilinear',
            # antialias=True,
            align_corners=False,
        ).to(dtype)
        new_pos_embed = new_pos_embed.permute(0, 2, 3, 1)
        return new_pos_embed
    else:
        return abs_pos



# @XHLLM_TRACEABLE_MODULES.register_module(
#     {
#         ImageEncoderViT: "deepseek_ocr.sam.ImageEncoderViT",
#     }
# )
class _ImageEncoderViT(DynamicRegister):
    def _setup(self, cfg):
        self.input_resolution = cfg.input_resolution
        self.permute_1 = Permute(0, 3, 1, 2)
        self.permute_2 = Permute(0, 2, 3, 1)
        if self.input_resolution in [512,]:
            self.resize = Resize(size=(32, 32),resize_mode='bilinear',align_corners=False,)
        else:
            raise NotImplementedError
        


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            # x = x + self.pos_embed
            x = x + get_abs_pos_sam(self.pos_embed, x.size(1))

        for blk in self.blocks:
            x = blk(x)

        x = self.neck(x.permute(0, 3, 1, 2))
        x2 = self.net_2(x)
        x3 = self.net_3(x2.clone())

        return x3
    
# @XHLLM_TRACEABLE_MODULES.register_module(
#     {
#         Attention: "deepseek_ocr.sam.Attention",
#     }
# )
class _Attention(DynamicRegister):
    def _setup(self, cfg):
        self.input_resolution = cfg.input_resolution

    def add_decomposed_rel_pos(self,
        q: torch.Tensor,
        rel_pos_h: torch.Tensor,
        rel_pos_w: torch.Tensor,
        q_size: Tuple[int, int],
        k_size: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q_h, q_w = q_size
        k_h, k_w = k_size
        if self.input_resolution in [512,]:
            Rh = self.get_rel_pos_2(q_h, k_h, rel_pos_h)
            Rw = self.get_rel_pos_2(q_w, k_w, rel_pos_w)

        B, _, dim = q.shape
        r_q = q.reshape(B, q_h, q_w, dim)
        Rh_T = Rh.permute(0, 2, 1)
        rel_h = torch.matmul(r_q, Rh_T)
        # rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
        Rw_T = Rw.permute(0, 2, 1)
        rel_w = torch.matmul(r_q, Rw_T)
        # rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)
        rel_h = rel_h.unsqueeze(-1)
        rel_w = rel_w.unsqueeze(-2)
        rel_h = rel_h.reshape(B, q_h * q_w, k_h, 1)
        rel_w = rel_w.reshape(B, q_h * q_w, 1, k_w)

        return rel_h, rel_w
    
    def get_rel_pos_1(self, q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
        max_rel_dist = int(2 * max(q_size, k_size) - 1)


        dtype = rel_pos.dtype
        rel_pos = rel_pos.to(torch.float32)
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        ).to(dtype)
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)


        # Scale the coords with short length if shapes for q and k are different.
        q_coords = torch.arange(q_size, device=rel_pos.device)[:, None] * max(k_size / q_size, 1.0)
        k_coords = torch.arange(k_size, device=rel_pos.device)[None, :] * max(q_size / k_size, 1.0)
        relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

        return rel_pos_resized[relative_coords.long()]

    def get_rel_pos_2(self, q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
        # Interpolate rel pos if needed.
        rel_pos_resized = rel_pos

        # Scale the coords with short length if shapes for q and k are different.
        q_coords = torch.arange(q_size)[:, None].to(rel_pos) * max(k_size / q_size, 1.0)
        k_coords = torch.arange(k_size)[None, :].to(rel_pos) * max(q_size / k_size, 1.0)
        relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

        return rel_pos_resized[relative_coords.long()]
   

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)

        rel_h, rel_w = None, None
        if self.use_rel_pos:
            if self.input_resolution in [512,]:
                rel_h, rel_w = self.add_decomposed_rel_pos(q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))
            else:
                raise NotImplementedError
            
        q = q.view(B, self.num_heads, H * W, -1)
        k = k.view(B, self.num_heads, H * W, -1)
        v = v.view(B, self.num_heads, H * W, -1)

        if self.use_rel_pos:
            rel_h = rel_h.view(B, self.num_heads, rel_h.size(1), rel_h.size(2), rel_h.size(3))
            rel_w = rel_w.view(B, self.num_heads, rel_w.size(1), rel_w.size(2), rel_w.size(3))
            attn_bias = (rel_h + rel_w).view(B, self.num_heads, rel_h.size(2), rel_h.size(3) * rel_w.size(4))

            attn_weights = torch.matmul(q, k.transpose(-2, -1))
            attn_weights = attn_weights * self.scale
            if attn_bias is not None:
                attn_weights = attn_weights + attn_bias
            attn_weights = torch.softmax(attn_weights, dim=-1)
            x = torch.matmul(attn_weights, v)
            # x = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)
        else:
            attn_weights = torch.matmul(q, k.transpose(-2, -1))
            attn_weights = attn_weights * self.scale
            attn_weights = torch.softmax(attn_weights, dim=-1)
            x = torch.matmul(attn_weights, v)
            # x = torch.nn.functional.scaled_dot_product_attention(q, k, v)

        x = x.view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)

        x = self.proj(x)

        return x


# @XHLLM_TRACEABLE_MODULES.register_module(
#     {
#         NoTPFeedForward: "deepseek_ocr.vision.NoTPFeedForward",
#     }
# )
class _NoTPFeedForward(DynamicRegister):
    def _setup(self, config):
        pass
    def forward(self, x):
        output = self.fc2(torch.nn.functional.gelu(self.fc1(x)))
        return output


# @XHLLM_TRACEABLE_MODULES.register_module(
#     {
#         NoTPAttention: "deepseek_ocr.vision.NoTPAttention",
#     }
# )
class _NoTPAttention(DynamicRegister):
    def _setup(self, cfg):
        pass

    def forward(
            self,
            x: torch.Tensor,
    ):
        bsz, seqlen, _ = x.shape
        xqkv = self.qkv_proj(x)
        xqkv = xqkv.view(bsz, seqlen, 3, self.num_heads, self.head_dim)

        xq, xk, xv = torch.split(xqkv, 1, dim=2)
        xq = xq.squeeze(2)
        xk = xk.squeeze(2)
        xv = xv.squeeze(2)
        # xq, xk, xv = xqkv[:, :, 0, ...], xqkv[:, :, 1, ...], xqkv[:, :, 2, ...]

        # （B, num_head, S, head_size)
        xq = xq.permute(0, 2, 1, 3)
        xk = xk.permute(0, 2, 1, 3)
        xv = xv.permute(0, 2, 1, 3)
        # with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
        attn_weights = torch.matmul(xq, xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = torch.softmax(attn_weights, dim=-1)
        output = torch.matmul(attn_weights, xv)
        # output = torch.nn.functional.scaled_dot_product_attention(xq, xk, xv, attn_mask=None)
        output = output.permute(0, 2, 1, 3).reshape(bsz, seqlen, -1)
        # output = output.permute(0, 2, 1, 3).contiguous().view(bsz, seqlen, -1)
        
        output = self.out_proj(output)
        return output

def get_abs_pos(abs_pos, tgt_size):
    # abs_pos: L, C
    # tgt_size: M
    # return: M, C

    # print(tgt_size)
    # print(abs_pos.shape)
    # exit()
    dim = abs_pos.size(-1)
    # print(dim)
    abs_pos_new = abs_pos.squeeze(0)
    cls_token, old_pos_embed = abs_pos_new[:1], abs_pos_new[1:]



    src_size = int(math.sqrt(abs_pos_new.shape[0] - 1))
    tgt_size = int(math.sqrt(tgt_size))
    dtype = abs_pos.dtype

    if src_size != tgt_size:
        old_pos_embed = old_pos_embed.view(1, src_size, src_size, dim).permute(0, 3, 1,
                                                                                    2).contiguous()
        old_pos_embed = old_pos_embed.to(torch.float32)
        new_pos_embed = F.interpolate(
            old_pos_embed,
            size=(tgt_size, tgt_size),
            mode='bilinear',
            # antialias=True,
            align_corners=False,
        ).to(dtype)
        new_pos_embed = new_pos_embed.permute(0, 2, 3, 1)
        new_pos_embed = new_pos_embed.view(tgt_size * tgt_size, dim)
        vision_pos_embed = torch.cat([cls_token, new_pos_embed], dim=0)
        vision_pos_embed = vision_pos_embed.view(1, tgt_size * tgt_size + 1, dim)
        return vision_pos_embed
    else:
        return abs_pos


# @XHLLM_TRACEABLE_MODULES.register_module(
#     {
#         CLIPVisionEmbeddings: "deepseek_ocr.vision.CLIPVisionEmbeddings",
#     }
# )
class _CLIPVisionEmbeddings(DynamicRegister):
    def _setup(self, cfg):
        self.input_resolution = cfg.input_resolution
        pass

    def forward(self, pixel_values, patch_embeds):
        batch_size = pixel_values.shape[0]
        # patch_embeds = self.patch_embedding(
        #     pixel_values
        # )  # shape = [*, width, grid, grid]


        if patch_embeds is not None:
            patch_embeds = patch_embeds
            # print(patch_embeds.shape)
        else:
            patch_embeds = self.patch_embedding(pixel_values)  
            # print(111111)
        # shape = [*, width, grid, grid]
        # patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)


        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)

        # x = torch.cat([cls_token, x], dim=1)
        embeddings = embeddings + get_abs_pos(self.position_embedding(self.position_ids), embeddings.size(1))
        # embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings



def register_wrap_modules(hf_model = None):
    pass

