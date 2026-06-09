from typing import Optional

import accelerate
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention
from diffusers.models.autoencoders.autoencoder_kl_flux2 import AutoencoderKLFlux2
from diffusers.models.autoencoders.vae import Decoder, DecoderOutput, Encoder
from diffusers.models.resnet import ResnetBlock2D
from diffusers.models.unets.unet_2d_blocks import UNetMidBlock2D, UpDecoderBlock2D
from diffusers.models.unets.unet_2d_blocks import DownEncoderBlock2D
from diffusers.models.upsampling import Upsample2D
from diffusers.models.downsampling import Downsample2D
from xhquant import nn as xhnn
from xhquant.utils.registry.dynamic_module import DynamicModule


def register_vae_wrap_modules(vae: nn.Module) -> nn.Module:
    vae = accelerate.hooks.remove_hook_from_module(vae, recurse=True)
    wrapped_classes = {
        AutoencoderKLFlux2: _AutoencoderKLFlux2,
        Encoder: _Encoder,
        DownEncoderBlock2D: _DownEncoderBlock2D,
        Decoder: _Decoder,
        UNetMidBlock2D: _UNetMidBlock2D,
        UpDecoderBlock2D: _UpDecoderBlock2D,
        ResnetBlock2D: _ResnetBlock2D,
        Upsample2D: _Upsample2D,
        Attention: _Attention,
        Downsample2D: _Downsample2D,
    }
    for _, module in list(vae.named_modules()):
        dynamic_cls = wrapped_classes.get(type(module))
        if dynamic_cls is not None and not isinstance(module, DynamicModule):
            dynamic_cls.convert(module)
    return vae


class _Downsample2D(DynamicModule):
    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if self.norm is not None:
            hidden_states = self.norm(hidden_states.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        if self.use_conv and self.padding == 0:
            pad = (0, 1, 0, 1)
            hidden_states = F.pad(hidden_states, pad, mode="constant", value=0)

        hidden_states = self.conv(hidden_states)

        return hidden_states

    def _setup(self):
        return self
  

class _AutoencoderKLFlux2(DynamicModule):
    def _decode(self, z: torch.Tensor, return_dict: bool = True):
        if self.post_quant_conv is not None:
            z = self.post_quant_conv(z)
        dec = self.decoder(z)
        if not return_dict:
            return (dec,)
        return DecoderOutput(sample=dec)

    def decode(self, z: torch.FloatTensor, return_dict: bool = True, generator=None):
        decoded = self._decode(z, return_dict=True).sample
        if not return_dict:
            return (decoded,)
        return DecoderOutput(sample=decoded)

    def _setup(self):
        return self


class _Encoder(DynamicModule):
    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        sample = self.conv_in(sample)
        for down_block in self.down_blocks:
            sample = down_block(sample)
        sample = self.mid_block(sample)
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)
        return sample

    def _setup(self):
        return self


class _DownEncoderBlock2D(DynamicModule):
    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        del args, kwargs
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb=None)
        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)
        return hidden_states

    def _setup(self):
        return self


class _Decoder(DynamicModule):
    def forward(self, sample: torch.Tensor, latent_embeds: Optional[torch.Tensor] = None) -> torch.Tensor:
        sample = self.conv_in(sample)
        sample = self.mid_block(sample, latent_embeds)
        for up_block in self.up_blocks:
            sample = up_block(sample, latent_embeds)
        if latent_embeds is None:
            sample = self.conv_norm_out(sample)
        else:
            sample = self.conv_norm_out(sample, latent_embeds)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)
        return sample

    def _setup(self):
        return self


class _UNetMidBlock2D(DynamicModule):
    def forward(self, hidden_states: torch.Tensor, temb: Optional[torch.Tensor] = None) -> torch.Tensor:
        hidden_states = self.resnets[0](hidden_states, temb)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            if attn is not None:
                hidden_states = attn(hidden_states, temb=temb)
            hidden_states = resnet(hidden_states, temb)
        return hidden_states

    def _setup(self):
        return self


class _UpDecoderBlock2D(DynamicModule):
    def forward(self, hidden_states: torch.Tensor, temb: Optional[torch.Tensor] = None) -> torch.Tensor:
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb=temb)
        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)
        return hidden_states

    def _setup(self):
        return self


class _ResnetBlock2D(DynamicModule):
    def forward(self, input_tensor: torch.Tensor, temb: Optional[torch.Tensor], *args, **kwargs) -> torch.Tensor:
        hidden_states = input_tensor
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)

        if self.upsample is not None:
            input_tensor = self.upsample(input_tensor)
            hidden_states = self.upsample(hidden_states)
        elif self.downsample is not None:
            input_tensor = self.downsample(input_tensor)
            hidden_states = self.downsample(hidden_states)

        hidden_states = self.conv1(hidden_states)

        if self.time_emb_proj is not None and temb is not None:
            if not self.skip_time_act:
                temb = self.nonlinearity(temb)
            temb = self.time_emb_proj(temb)[:, :, None, None]

        if self.time_embedding_norm == "default":
            if temb is not None:
                hidden_states = hidden_states + temb
            hidden_states = self.norm2(hidden_states)
        elif self.time_embedding_norm == "scale_shift":
            time_scale, time_shift = torch.chunk(temb, 2, dim=1)
            hidden_states = self.norm2(hidden_states)
            hidden_states = hidden_states * (1 + time_scale) + time_shift
        else:
            hidden_states = self.norm2(hidden_states)

        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor.contiguous())

        return input_tensor + hidden_states

    def _setup(self):
        return self


class _Upsample2D(DynamicModule):
    def forward(self, hidden_states: torch.Tensor, output_size: Optional[int] = None, *args, **kwargs) -> torch.Tensor:
        if self.norm is not None:
            hidden_states = self.norm(hidden_states.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        if self.use_conv_transpose:
            return self.conv(hidden_states)

        if self.interpolate:
            if output_size is None:
                hidden_states = F.interpolate(hidden_states, scale_factor=2.0, mode="nearest")
            else:
                hidden_states = F.interpolate(hidden_states, size=output_size, mode="nearest")

        if self.use_conv:
            if self.name == "conv":
                hidden_states = self.conv(hidden_states)
            else:
                hidden_states = self.Conv2d_0(hidden_states)
        return hidden_states

    def _setup(self):
        return self


class _Attention(DynamicModule):
    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, **cross_attention_kwargs):
        residual = hidden_states
        input_ndim = hidden_states.ndim

        if getattr(self, "spatial_norm", None) is not None:
            hidden_states = self.spatial_norm(hidden_states, None)

        # if input_ndim == 4:
        # batch_size, channel, height, width = hidden_states.shape # 1, 512, 128, 128
        # hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        hidden_states = hidden_states.view(1, 512, 128 * 128).transpose(1, 2)
        batch_size, channel, height, width = 1, 512, 128, 128
        # else:
        #     batch_size = hidden_states.shape[0]
        #     height = width = None

        if getattr(self, "group_norm", None) is not None:
            hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        encoder_hidden_states = hidden_states if encoder_hidden_states is None else encoder_hidden_states
        query = self.to_q(hidden_states)
        key = self.to_k(encoder_hidden_states)
        value = self.to_v(encoder_hidden_states)

        # inner_dim = key.shape[-1]
        # head_dim = inner_dim // self.heads
        head_dim = 512
        query = query.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)

        attn_weights = torch.matmul(query * (head_dim**-0.5), key.transpose(2, 3))
        attn_weights = torch.softmax(attn_weights, dim=-1)
        hidden_states = torch.matmul(attn_weights, value)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        hidden_states = self.to_out[0](hidden_states)
        hidden_states = self.to_out[1](hidden_states)

        # if input_ndim == 4:
        hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if self.residual_connection:
            hidden_states = hidden_states + residual
        # hidden_states = hidden_states / self.rescale_output_factor
        return hidden_states

    def _setup(self):
        return self


class Flux2VAEDecoderWrapper(nn.Module):
    def __init__(self, vae: nn.Module):
        super().__init__()
        self.vae = vae
        self.batch_norm_eps = float(getattr(getattr(vae, "config", None), "batch_norm_eps", vae.bn.eps))
        self._setup()

    @staticmethod
    def _unpack_latents_with_ids(latents: torch.Tensor, latent_ids: torch.Tensor) -> torch.Tensor:
        """Graph 外预处理：按 diffusers pipeline 的 latent_ids 还原 packed latent grid。

        这里包含 Python list、动态 max、scatter_，不能放进 xhquant trace 图。
        converter/runtime 先调用该函数得到固定 NCHW latents，再交给 forward 只 trace decode 计算。
        """
        outputs = []
        for data, pos in zip(latents, latent_ids):
            h_ids = pos[:, 1].to(torch.int64)
            w_ids = pos[:, 2].to(torch.int64)
            height = torch.max(h_ids) + 1
            width = torch.max(w_ids) + 1
            flat_ids = h_ids * width + w_ids

            out = torch.zeros((height * width, data.shape[-1]), device=data.device, dtype=data.dtype)
            out.scatter_(0, flat_ids.unsqueeze(1).expand(-1, data.shape[-1]), data)
            out = out.view(height, width, data.shape[-1]).permute(2, 0, 1)
            outputs.append(out)

        return torch.stack(outputs, dim=0)

    def _unpatchify_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Graph 外预处理：把 packed VAE latent channel unpatchify 成 decode 输入。"""
        # batch_size, num_channels_latents, height, width = latents.shape
        latents = latents.reshape(self.batch_size, self.num_channels_latents // 4, 2, 2, self.height, self.width)
        latents = latents.permute(0, 1, 4, 2, 5, 3)
        latents = latents.reshape(self.batch_size, self.num_channels_latents // 4, self.height * 2, self.width * 2)
        return latents

    def preprocess_decode_latents(self, packed_latents: torch.Tensor, latent_ids: torch.Tensor) -> torch.Tensor:
        latents = self._unpack_latents_with_ids(packed_latents, latent_ids)
        # latents_bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
        # latents_bn_std = torch.sqrt(
        #     self.vae.bn.running_var.view(1, -1, 1, 1).to(latents.device, latents.dtype) + self.batch_norm_eps
        # )
        return latents

        latents = latents * self.latents_bn_std +  self.latents_bn_mean
        return self._unpatchify_latents(latents)

    def _setup(self):
        self.latents_bn_std =  torch.sqrt(
            self.vae.bn.running_var.view(1, -1, 1, 1) + self.batch_norm_eps
        )
        self.latents_bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1)
        self.batch_size, self.num_channels_latents, self.height, self.width =  1, 128, 64, 64
        return self

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        latents = latents * self.latents_bn_std +  self.latents_bn_mean
        latents = self._unpatchify_latents(latents)
        return self.vae.decode(latents, return_dict=False)[0]


class Flux2VAEEncoderWrapper(nn.Module):
    def __init__(self, vae: nn.Module):
        super().__init__()
        self.vae = vae
        self.batch_norm_eps = float(getattr(getattr(vae, "config", None), "batch_norm_eps", vae.bn.eps))
        self._setup()

    @staticmethod
    def _patchify_latents(latents: torch.Tensor) -> torch.Tensor:
        # batch_size, num_channels, height, width = latents.shape
        latents = latents.view(1, 32, 64, 2, 64, 2)
        latents = latents.permute(0, 1, 3, 5, 2, 4)
        latents = latents.reshape(1, 32 * 4, 64, 64)
        return latents

    def _setup(self):
        self.latents_bn_std = torch.sqrt(
            self.vae.bn.running_var.view(1, -1, 1, 1) + self.batch_norm_eps
        )
        self.latents_bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1)
        self.slice_mean = xhnn.Slice([0], [32], [1], [1])
        self.slice_logvar = xhnn.Slice([32], [64], [1], [1])
        return self

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        enc = self.vae.encoder(image)
        if self.vae.quant_conv is not None:
            enc = self.vae.quant_conv(enc)
        mean = self.slice_mean(enc)
        # _ = self.slice_logvar(enc)
        image_latents = self._patchify_latents(mean)
        return (image_latents - self.latents_bn_mean) / self.latents_bn_std
