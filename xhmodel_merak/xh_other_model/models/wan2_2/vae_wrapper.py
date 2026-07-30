# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: E402, I001

# pyright: reportMissingImports=false

from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from xhquant import nn as xhnn
from xhquant.utils.registry.dynamic_module import DynamicModule

from .common import ensure_wan2_2_repo

ensure_wan2_2_repo()
from wan.modules.vae2_1 import (  # noqa: E402
    AttentionBlock,
    CACHE_T,
    CausalConv3d,
    Decoder3d,
    Encoder3d,
    RMS_norm,
    ResidualBlock,
    Resample,
    WanVAE_,
)


def _is_causal_conv3d_module(module: nn.Module) -> bool:
    return type(module).__name__ in {"CausalConv3d", "_WanCausalConv3d"}


def _is_residual_block_module(module: nn.Module) -> bool:
    return type(module).__name__ in {"ResidualBlock", "_WanResidualBlock"}


_REP_CACHE_SENTINEL = 1111


def _is_rep_cache(cache_value) -> bool:
    return isinstance(cache_value, int) and cache_value == _REP_CACHE_SENTINEL


class _WanCausalConv3d(DynamicModule):
    def _setup(self, *args, **kwargs):
        del args, kwargs
        self.conv = nn.Conv3d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            bias=self.bias is not None,
            padding_mode=self.padding_mode,
        )
        self.conv.weight = nn.Parameter(deepcopy(self.weight.data))
        if self.bias is not None:
            self.conv.bias = nn.Parameter(deepcopy(self.bias.data))
        return self

    def forward(self, x, cache_x=None):
        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            x = torch.cat([cache_x, x], dim=2)
            padding[4] -= cache_x.shape[2]
        x = F.pad(x, padding)
        return self.conv(x)


class _WanRMSNorm(DynamicModule):
    def _setup(self, *args, **kwargs):
        del args, kwargs
        if isinstance(self.bias, nn.Parameter):
            assert torch.count_nonzero(self.bias).item() == 0, "Wan2.2 VAE RMS_norm bias must be zero"
        else:
            assert self.bias == 0.0, "Wan2.2 VAE RMS_norm bias must be zero"
        hidden_size = self.gamma.data.shape[0]
        self.norm = xhnn.RMSNorm(hidden_size, 0.0, reduce_dim=1)
        self.norm.weight = nn.Parameter(deepcopy(self.gamma.data.squeeze(-1).squeeze(-1).squeeze(-1)))
        return self

    def forward(self, x):
        return self.norm(x)


class _WanAttentionBlock(DynamicModule):
    def _setup(self, *args, **kwargs):
        del args, kwargs
        self.kv_scale = 1.0 / (self.dim**0.5)
        self.q_proj = nn.Linear(self.dim, self.dim, bias=True)
        self.k_proj = nn.Linear(self.dim, self.dim, bias=True)
        self.v_proj = nn.Linear(self.dim, self.dim, bias=True)

        qkv_weight = self.to_qkv.weight.detach().reshape(self.dim * 3, self.dim)
        qkv_bias = self.to_qkv.bias.detach()
        self.q_proj.weight.data.copy_(qkv_weight[: self.dim])
        self.k_proj.weight.data.copy_(qkv_weight[self.dim : 2 * self.dim])
        self.v_proj.weight.data.copy_(qkv_weight[2 * self.dim :])
        self.q_proj.bias.data.copy_(qkv_bias[: self.dim])
        self.k_proj.bias.data.copy_(qkv_bias[self.dim : 2 * self.dim])
        self.v_proj.bias.data.copy_(qkv_bias[2 * self.dim :])

        self.q_proj.to(device=qkv_weight.device, dtype=qkv_weight.dtype)
        self.k_proj.to(device=qkv_weight.device, dtype=qkv_weight.dtype)
        self.v_proj.to(device=qkv_weight.device, dtype=qkv_weight.dtype)
        return self

    def forward(self, x):
        identity = x
        b, c, t, h, w = x.size()
        x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        x = self.norm(x)
        x_tokens = x.permute(0, 2, 3, 1).reshape(b * t, h * w, c)
        q = self.q_proj(x_tokens).unsqueeze(1)
        k = self.k_proj(x_tokens).unsqueeze(1)
        v = self.v_proj(x_tokens).unsqueeze(1)
        scores = torch.matmul(q * self.kv_scale, k.transpose(-1, -2))
        x = torch.matmul(torch.softmax(scores, dim=-1), v)
        x = x.squeeze(1).reshape(b * t, h, w, c).permute(0, 3, 1, 2)
        x = self.proj(x)
        x = x.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4)
        return x + identity


class _WanResidualBlock(DynamicModule):
    def _setup(self, *args, **kwargs):
        del args, kwargs
        return self

    def forward(self, x, feat_cache=None, feat_idx=None):
        if feat_idx is None:
            feat_idx = [0]
        h = self.shortcut(x)
        for layer in self.residual:
            if _is_causal_conv3d_module(layer) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :]  # .clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    cache_x = torch.cat(
                        [
                            feat_cache[idx][:, :, -1, :, :].unsqueeze(2),
                            cache_x,
                        ],
                        dim=2,
                    )
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x + h


class _WanResample(DynamicModule):
    def _setup(self, *args, **kwargs):
        del args, kwargs
        return self

    def forward(self, x, feat_cache=None, feat_idx=None):
        if feat_idx is None:
            feat_idx = [0]
        b, c, t, h, w = x.size()
        if self.mode == "upsample3d":
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = _REP_CACHE_SENTINEL
                    feat_idx[0] += 1
                else:
                    cache_x = x[:, :, -CACHE_T:, :, :]  # .clone()
                    if cache_x.shape[2] < 2 and feat_cache[idx] is not None and not _is_rep_cache(feat_cache[idx]):
                        cache_x = torch.cat(
                            [feat_cache[idx][:, :, -1, :, :].unsqueeze(2), cache_x], dim=2
                        )
                    if cache_x.shape[2] < 2 and feat_cache[idx] is not None and _is_rep_cache(feat_cache[idx]):
                        cache_x = torch.cat([torch.zeros_like(cache_x), cache_x], dim=2)
                    if _is_rep_cache(feat_cache[idx]):
                        x = self.time_conv(x)
                    else:
                        x = self.time_conv(x, feat_cache[idx])
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1

                    x = x.reshape(b, 2, c, t, h, w)
                    x = torch.stack((x[:, 0, :, :, :, :], x[:, 1, :, :, :, :]), 3)
                    x = x.reshape(b, c, t * 2, h, w)
        t = x.shape[2]
        x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        if isinstance(self.resample, nn.Sequential):
            for layer in self.resample:
                if isinstance(layer, nn.ZeroPad2d):
                    x = F.pad(x, layer.padding)
                else:
                    x = layer(x)
        else:
            x = self.resample(x)
        _, c_out, h_out, w_out = x.shape
        x = x.reshape(b, t, c_out, h_out, w_out).permute(0, 2, 1, 3, 4)

        if self.mode == "downsample3d":
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = x  # .clone()
                    feat_idx[0] += 1
                else:
                    cache_x = x[:, :, -1:, :, :]  # .clone()
                    x = self.time_conv(torch.cat([feat_cache[idx][:, :, -1:, :, :], x], 2))
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1
        return x


class _WanEncoder3d(DynamicModule):
    def _setup(self, *args, **kwargs):
        del args, kwargs
        return self

    def forward(self, x, feat_cache=None, feat_idx=None):
        if feat_idx is None:
            feat_idx = [0]
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :]  # .clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat(
                    [feat_cache[idx][:, :, -1, :, :].unsqueeze(2), cache_x],
                    dim=2,
                )
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        for layer in self.downsamples:
            x = layer(x, feat_cache, feat_idx) if feat_cache is not None else layer(x)

        for layer in self.middle:
            if _is_residual_block_module(layer) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        for layer in self.head:
            if _is_causal_conv3d_module(layer) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :]  # .clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    cache_x = torch.cat(
                        [feat_cache[idx][:, :, -1, :, :].unsqueeze(2), cache_x],
                        dim=2,
                    )
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x


class _WanDecoder3d(DynamicModule):
    def _setup(self, *args, **kwargs):
        del args, kwargs
        return self

    def forward(self, x, feat_cache=None, feat_idx=None):
        if feat_idx is None:
            feat_idx = [0]
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :]  # .clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat(
                    [feat_cache[idx][:, :, -1, :, :].unsqueeze(2), cache_x],
                    dim=2,
                )
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        for layer in self.middle:
            if _is_residual_block_module(layer) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        for layer in self.upsamples:
            x = layer(x, feat_cache, feat_idx) if feat_cache is not None else layer(x)

        for layer in self.head:
            if _is_causal_conv3d_module(layer) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :]  # .clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    cache_x = torch.cat(
                        [feat_cache[idx][:, :, -1, :, :].unsqueeze(2), cache_x],
                        dim=2,
                    )
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x


def count_conv3d(model):
    count = 0
    for m in model.modules():
        if _is_causal_conv3d_module(m):
            count += 1
    return count


class _WanVAE(DynamicModule):
    def _setup(self, *args, **kwargs):
        del args, kwargs
        return self

    def encode(self, x, scale):
        self.clear_cache()
        t = x.shape[2]
        iter_ = 1 + (t - 1) // 4
        for i in range(iter_):
            self._enc_conv_idx = [0]
            if i == 0:
                out = self.encoder(x[:, :, :1, :, :], feat_cache=self._enc_feat_map, feat_idx=self._enc_conv_idx)
            else:
                out_ = self.encoder(
                    x[:, :, 1 + 4 * (i - 1) : 1 + 4 * i, :, :],
                    feat_cache=self._enc_feat_map,
                    feat_idx=self._enc_conv_idx,
                )
                out = torch.cat([out, out_], 2)
        conv1_out = self.conv1(out)
        mu = conv1_out[:, : self.z_dim, ...]
        log_var = conv1_out[:, self.z_dim :, ...]
        if isinstance(scale[0], torch.Tensor):
            mu = (mu - scale[0].view(1, self.z_dim, 1, 1, 1)) * scale[1].view(1, self.z_dim, 1, 1, 1)
        else:
            mu = (mu - scale[0]) * scale[1]
        self.clear_cache()
        del log_var
        return mu

    def decode(self, z, scale):
        self.clear_cache()
        if isinstance(scale[0], torch.Tensor):
            z = z / scale[1].view(1, self.z_dim, 1, 1, 1) + scale[0].view(1, self.z_dim, 1, 1, 1)
        else:
            z = z / scale[1] + scale[0]
        iter_ = z.shape[2]
        x = self.conv2(z)
        for i in range(iter_):
            self._conv_idx = [0]
            if i == 0:
                out = self.decoder(x[:, :, i : i + 1, :, :], feat_cache=self._feat_map, feat_idx=self._conv_idx)
            else:
                out_ = self.decoder(x[:, :, i : i + 1, :, :], feat_cache=self._feat_map, feat_idx=self._conv_idx)
                out = torch.cat([out, out_], 2)
        self.clear_cache()
        return out

    def clear_cache(self):
        self._conv_num = count_conv3d(self.decoder)
        self._conv_idx = [0]
        self._feat_map = [None] * self._conv_num
        # cache encode
        self._enc_conv_num = count_conv3d(self.encoder)
        self._enc_conv_idx = [0]
        self._enc_feat_map = [None] * self._enc_conv_num


def wrap_vae_model(vae_model: nn.Module) -> nn.Module:
    wrapped_classes = {
        AttentionBlock: _WanAttentionBlock,
        CausalConv3d: _WanCausalConv3d,
        Decoder3d: _WanDecoder3d,
        Encoder3d: _WanEncoder3d,
        RMS_norm: _WanRMSNorm,
        ResidualBlock: _WanResidualBlock,
        Resample: _WanResample,
        WanVAE_: _WanVAE,
    }
    for _, module in list(vae_model.named_modules()):
        dynamic_cls = wrapped_classes.get(type(module))
        if dynamic_cls is not None and not isinstance(module, DynamicModule):
            dynamic_cls.convert(module)
    return vae_model


def wrap_vae_encoder_model(vae_model: nn.Module) -> nn.Module:
    return wrap_vae_model(vae_model)


def wrap_vae_decoder_model(vae_model: nn.Module) -> nn.Module:
    return wrap_vae_model(vae_model)


class Wan22VAEEncoderExportWrapper(nn.Module):
    """Wrap Wan2.2 VAE encoder path for HMONNX export."""

    def __init__(self, vae_model: nn.Module, scale):
        super().__init__()
        self.vae_model = wrap_vae_encoder_model(vae_model)
        self.scale = scale

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        return self.vae_model.encode(video.unsqueeze(0), self.scale).squeeze(0)


class Wan22VAEDecoderExportWrapper(nn.Module):
    """Wrap Wan2.2 VAE decoder path for HMONNX export."""

    def __init__(self, vae_model: nn.Module, scale):
        super().__init__()
        self.vae_model = wrap_vae_decoder_model(vae_model)
        self.scale = scale

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.vae_model.decode(latent.unsqueeze(0), self.scale).squeeze(0)


Wan2_2VAEEncoderExportWrapper = Wan22VAEEncoderExportWrapper
Wan2_2VAEDecoderExportWrapper = Wan22VAEDecoderExportWrapper
