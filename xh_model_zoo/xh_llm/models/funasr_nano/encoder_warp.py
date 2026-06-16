"""SenseVoice encoder warp modules for FunASR-Nano export."""

from __future__ import annotations

import torch
import torch.nn as nn
from xhquant import nn as xhnn

from funasr.models.sense_voice.model import MultiHeadedAttentionSANM, SenseVoiceEncoderSmall


class XHMultiHeadedAttentionSANM(nn.Module):
    def __init__(self, src: MultiHeadedAttentionSANM):
        super().__init__()
        self.d_k = src.d_k
        self.h = src.h
        self.linear_out = src.linear_out
        self.attn = None
        self.dropout = nn.Identity()
        self.fsmn_block = src.fsmn_block
        self.pad_fn = src.pad_fn
        self.maskedadd = xhnn.MaskedAdd()
        self.kv_scale = self.d_k**-0.5

        in_feat = src.linear_q_k_v.in_features
        out_feat = self.h * self.d_k
        has_bias = src.linear_q_k_v.bias is not None
        self.q_proj = nn.Linear(in_feat, out_feat, bias=has_bias)
        self.k_proj = nn.Linear(in_feat, out_feat, bias=has_bias)
        self.v_proj = nn.Linear(in_feat, out_feat, bias=has_bias)

        weight = src.linear_q_k_v.weight.data
        self.q_proj.weight.data.copy_(weight[:out_feat])
        self.k_proj.weight.data.copy_(weight[out_feat : out_feat * 2])
        self.v_proj.weight.data.copy_(weight[out_feat * 2 : out_feat * 3])
        if has_bias:
            bias = src.linear_q_k_v.bias.data
            self.q_proj.bias.data.copy_(bias[:out_feat])
            self.k_proj.bias.data.copy_(bias[out_feat : out_feat * 2])
            self.v_proj.bias.data.copy_(bias[out_feat * 2 : out_feat * 3])

    def forward_fsmn(self, inputs, mask, mask_shfit_chunk=None):
        b, _t, _d = inputs.size()
        if mask is not None:
            mask = torch.reshape(mask, (b, -1, 1))
            if mask_shfit_chunk is not None:
                mask = mask * mask_shfit_chunk
            inputs = inputs * mask
        x = inputs.transpose(1, 2)
        x = self.pad_fn(x)
        x = self.fsmn_block(x)
        x = x.transpose(1, 2)
        x = x + inputs
        if mask is not None:
            x = x * mask
        return x

    def _proj_heads(self, proj: nn.Linear, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.size()
        return proj(x).reshape(b, t, self.h, self.d_k).transpose(1, 2)

    def forward_qkv(self, x):
        q_h = self._proj_heads(self.q_proj, x)
        k_h = self._proj_heads(self.k_proj, x)
        v = self.v_proj(x)
        b, t, _ = v.size()
        v_h = v.reshape(b, t, self.h, self.d_k).transpose(1, 2)
        return q_h, k_h, v_h, v

    def forward_attention(self, value, scores, att_mask):
        n_batch = value.size(0)
        if att_mask is not None:
            scores = self.maskedadd(scores, att_mask)
        attn = torch.softmax(scores, dim=-1)
        x = torch.matmul(attn, value)
        x = x.transpose(1, 2).contiguous().reshape(n_batch, -1, self.h * self.d_k)
        return self.linear_out(x)

    def forward(self, x, mask, att_mask, mask_shfit_chunk=None):
        q_h, k_h, v_h, v = self.forward_qkv(x)
        fsmn_memory = self.forward_fsmn(v, mask, mask_shfit_chunk)
        scores = torch.matmul(q_h * self.kv_scale, k_h.transpose(-2, -1))
        att_outs = self.forward_attention(v_h, scores, att_mask)
        return att_outs + fsmn_memory


class XHSenseVoiceEncoderLayer(nn.Module):
    def __init__(self, src: nn.Module):
        super().__init__()
        self.self_attn = src.self_attn
        self.feed_forward = src.feed_forward
        self.norm1 = src.norm1
        self.norm2 = src.norm2
        self.dropout = nn.Identity()
        self.size = src.size
        self.in_size = src.in_size
        self.normalize_before = src.normalize_before
        self.concat_after = src.concat_after
        if self.concat_after:
            self.concat_linear = src.concat_linear

    def forward(self, x, mask, att_mask, mask_shfit_chunk=None):
        residual = x
        if self.normalize_before:
            x = self.norm1(x)
        if self.concat_after:
            x_concat = torch.cat((x, self.self_attn(x, mask, att_mask, mask_shfit_chunk)), dim=-1)
            if self.in_size == self.size:
                x = residual + self.concat_linear(x_concat)
            else:
                x = self.concat_linear(x_concat)
        else:
            if self.in_size == self.size:
                x = residual + self.self_attn(x, mask, att_mask, mask_shfit_chunk)
            else:
                x = self.self_attn(x, mask, att_mask, mask_shfit_chunk)
        if not self.normalize_before:
            x = self.norm1(x)

        residual = x
        if self.normalize_before:
            x = self.norm2(x)
        x = residual + self.feed_forward(x)
        if not self.normalize_before:
            x = self.norm2(x)
        return x, mask


class XHSenseVoiceEncoderSmall(nn.Module):
    def __init__(self, src: SenseVoiceEncoderSmall):
        super().__init__()
        self.src = src
        self.embed = src.embed
        self.encoders0 = src.encoders0
        self.encoders = src.encoders
        self.tp_encoders = src.tp_encoders
        self.after_norm = src.after_norm
        self.tp_norm = src.tp_norm

    def output_size(self) -> int:
        return self.src.output_size()

    def forward(self, xs_pad: torch.Tensor, masks: torch.Tensor, att_masks: torch.Tensor):
        xs_pad = xs_pad * (self.output_size() ** 0.5)
        xs_pad = self.embed(xs_pad)
        for encoder_layer in self.encoders0:
            xs_pad, masks = encoder_layer(xs_pad, masks, att_masks)[:2]
        for encoder_layer in self.encoders:
            xs_pad, masks = encoder_layer(xs_pad, masks, att_masks)[:2]
        xs_pad = self.after_norm(xs_pad)
        olens = masks.squeeze(1).sum(1).int()
        for encoder_layer in self.tp_encoders:
            xs_pad, masks = encoder_layer(xs_pad, masks, att_masks)[:2]
        xs_pad = self.tp_norm(xs_pad)
        return xs_pad, olens
