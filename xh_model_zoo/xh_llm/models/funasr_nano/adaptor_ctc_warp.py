"""Adaptor and CTC decoder warp modules for FunASR-Nano export."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from xhquant import nn as xhnn

from funasr.models.llm_asr.adaptor import Transformer as FunASRAdaptorTransformer
from funasr.models.transformer.attention import MultiHeadedAttention


class XHMultiHeadedAttention(nn.Module):
    def __init__(self, src: MultiHeadedAttention):
        super().__init__()
        self.d_k = src.d_k
        self.h = src.h
        self.linear_q = src.linear_q
        self.linear_k = src.linear_k
        self.linear_v = src.linear_v
        self.linear_out = src.linear_out
        self.attn = None
        self.dropout = nn.Identity()
        self.maskedadd = xhnn.MaskedAdd()
        self.kv_scale = self.d_k**-0.5

    def _proj_heads(self, proj: nn.Linear, x: torch.Tensor) -> torch.Tensor:
        b = x.size(0)
        return proj(x).reshape(b, -1, self.h, self.d_k).transpose(1, 2)

    def forward_qkv(self, query, key, value):
        return (
            self._proj_heads(self.linear_q, query),
            self._proj_heads(self.linear_k, key),
            self._proj_heads(self.linear_v, value),
        )

    def forward_attention(self, value, scores, att_mask):
        n_batch = value.size(0)
        if att_mask is not None:
            scores = self.maskedadd(scores, att_mask)
        attn = torch.softmax(scores, dim=-1)
        x = torch.matmul(attn, value)
        x = x.transpose(1, 2).contiguous().reshape(n_batch, -1, self.h * self.d_k)
        return self.linear_out(x)

    def forward(self, query, key, value, mask, att_mask):
        q, k, v = self.forward_qkv(query, key, value)
        scores = torch.matmul(q * self.kv_scale, k.transpose(-2, -1))
        return self.forward_attention(v, scores, att_mask)


class XHTransformerEncoderLayer(nn.Module):
    def __init__(self, src: nn.Module):
        super().__init__()
        self.self_attn = src.self_attn
        self.feed_forward = src.feed_forward
        self.norm1 = src.norm1
        self.norm2 = src.norm2
        self.dropout = nn.Identity()
        self.size = src.size
        self.normalize_before = src.normalize_before
        self.concat_after = src.concat_after
        if self.concat_after:
            self.concat_linear = src.concat_linear

    def forward(self, x, mask, att_mask):
        residual = x
        if self.normalize_before:
            x = self.norm1(x)
        if self.concat_after:
            x_concat = torch.cat((x, self.self_attn(x, x, x, mask, att_mask)), dim=-1)
            x = residual + self.concat_linear(x_concat)
        else:
            x = residual + self.self_attn(x, x, x, mask, att_mask)
        if not self.normalize_before:
            x = self.norm1(x)

        residual = x
        if self.normalize_before:
            x = self.norm2(x)
        x = residual + self.feed_forward(x)
        if not self.normalize_before:
            x = self.norm2(x)
        return x, mask


class XHAdaptorTransformer(nn.Module):
    """FunASR LLM-ASR Transformer adaptor with externally supplied masks."""

    def __init__(self, src: FunASRAdaptorTransformer):
        super().__init__()
        self.k = src.k
        self.encoder_dim = src.encoder_dim
        self.llm_dim = src.llm_dim
        self.linear1 = src.linear1
        self.relu = src.relu
        self.linear2 = src.linear2
        self.blocks = src.blocks

    def forward(self, x: torch.Tensor, masks: torch.Tensor, att_masks: torch.Tensor):
        batch_size, seq_len, dim = x.size()
        chunk_num = (seq_len - 1) // self.k + 1
        pad_num = chunk_num * self.k - seq_len
        x = F.pad(x, (0, 0, 0, pad_num, 0, 0), value=0.0)
        x = x.contiguous().reshape(batch_size, chunk_num, dim * self.k)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)

        if self.blocks is not None:
            for block in self.blocks:
                x, masks = block(x, masks, att_masks)
        olens = masks.squeeze(1).sum(1).int()
        return x, olens
