import math
from pickle import NONE
from re import X
import sys
from turtle import forward
import types
from copy import deepcopy
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from transformers.modeling_outputs import BaseModelOutputWithPast
from xhquant import nn as xhnn
from xhquant.api import ConfigDict
from xhquant.nn import LLMCacheV2, MaskedSoftmax, RMSNorm
from xhquant.nn.modules import none_parameter_module
from xhquant.utils.registry import DynamicModule

from ..builder import XHLLM_TRACEABLE_MODULES
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.models.autoencoders.vae import Decoder
from diffusers.models.unets.unet_2d_blocks import UNetMidBlock2D
from diffusers.models.unets.unet_2d_blocks import UpDecoderBlock2D
from diffusers.models.resnet import ResnetBlock2D
from diffusers.models.upsampling import Upsample2D

from diffusers.models.attention_processor import Attention

from diffusers.models.autoencoders.autoencoder_kl_qwenimage import QwenImageCausalConv3d
from diffusers.models.autoencoders.autoencoder_kl_qwenimage import QwenImageRMS_norm
from diffusers.models.autoencoders.autoencoder_kl_qwenimage import QwenImageResidualBlock
from diffusers.models.autoencoders.autoencoder_kl_qwenimage import QwenImageMidBlock
from diffusers.models.autoencoders.autoencoder_kl_qwenimage import QwenImageAttentionBlock
from diffusers.models.autoencoders.autoencoder_kl_qwenimage import QwenImageUpBlock
from diffusers.models.autoencoders.autoencoder_kl_qwenimage import QwenImageResample

import torch.nn.functional as F
import torch.nn as nn

@XHLLM_TRACEABLE_MODULES.register_module({Attention: "Attention"})
class _Attention(DynamicModule):
    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, **cross_attention_kwargs):
        residual = hidden_states
        batch_size = 1
        channel = 512
        height = 128
        width = 128

        hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = self.to_q(hidden_states)

        encoder_hidden_states = hidden_states

        key = self.to_k(encoder_hidden_states)
        value = self.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // self.heads

        query = query.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)


        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        # hidden_states = F.scaled_dot_product_attention(
        #     query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        # )
        query = query * self.kv_scale
        key = key.transpose(2, 3)
        attn_weights = torch.matmul(query, key) 
        attn_weights = F.softmax(attn_weights, dim=-1)
        hidden_states = torch.matmul(attn_weights, value) # [1, 1, 16384, 512]

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = self.to_out[0](hidden_states)
        # dropout
        hidden_states = self.to_out[1](hidden_states)

        hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        hidden_states = hidden_states + residual

        hidden_states = hidden_states / self.rescale_output_factor

        return hidden_states
    def _setup(self, cfg: Optional[Dict] = None):

        _kv_scale = 1 / math.sqrt(512)
        self.kv_scale = _kv_scale
        return self


@XHLLM_TRACEABLE_MODULES.register_module({Upsample2D: "Upsample2D"})
class _Upsample2D(DynamicModule):
    def forward(self, hidden_states: torch.Tensor, output_size: Optional[int] = None, *args, **kwargs) -> torch.Tensor:
        if self.norm is not None:
            hidden_states = self.norm(hidden_states.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        if self.use_conv_transpose:
            return self.conv(hidden_states)


        # if `output_size` is passed we force the interpolation output
        # size and do not make use of `scale_factor=2`
        if self.interpolate:
            # upsample_nearest_nhwc also fails when the number of output elements is large
            # https://github.com/pytorch/pytorch/issues/141831
            scale_factor = 2
            hidden_states = F.interpolate(hidden_states, scale_factor=2.0, mode="nearest")


        # TODO(Suraj, Patrick) - clean up after weight dicts are correctly renamed
        if self.use_conv:
            if self.name == "conv":
                hidden_states = self.conv(hidden_states)
            else:
                hidden_states = self.Conv2d_0(hidden_states)

        return hidden_states

    def _setup(self, cfg: ConfigDict):
        self.cfg = cfg

@XHLLM_TRACEABLE_MODULES.register_module({AutoencoderKL: "AutoencoderKL"})
class _AutoencoderKL(DynamicModule):
    def _setup(self, cfg: ConfigDict):
        self.cfg = cfg
        del self.encoder

    def decode(self, z: torch.Tensor, return_dict: bool = True):
        decoded = self._decode(z).sample
        return decoded
    
    def _decode(self, z: torch.Tensor, return_dict: bool = True):
        # _, _, num_frame, height, width = z.shape
        if self.post_quant_conv is not None:
            z = self.post_quant_conv(z)
        # assert num_frame == 1
        # for i in range(num_frame):
        #     self._conv_idx = [0]
        #     if i == 0:
        #         out = self.decoder(x[:, :, i : i + 1, :, :], feat_cache=self._feat_map, feat_idx=self._conv_idx)
        #     else:
        #         out_ = self.decoder(x[:, :, i : i + 1, :, :], feat_cache=self._feat_map, feat_idx=self._conv_idx)
        #         out = torch.cat([out, out_], 2)
        out = self.decoder(z)
        # out = torch.clamp(out, min=-1.0, max=1.0)
        return out
    
    def forward(self, x: torch.Tensor):
        decoded = self._decode(x)
        return decoded

@XHLLM_TRACEABLE_MODULES.register_module({Decoder: "Decoder"})
class _Decoder(DynamicModule):
    def _setup(self, cfg: ConfigDict):
        self.cfg = cfg

    def forward(self, sample, latent_embeds=None):
        ## conv1
        sample = self.conv_in(sample)

        ## middle
        sample = self.mid_block(sample, latent_embeds)

        ## upsamples
        for up_block in self.up_blocks:
            sample = up_block(sample, latent_embeds)

        ## head
        sample = self.conv_norm_out(sample)

        sample = self.conv_act(sample)
        sample = self.conv_out(sample)
        return sample

@XHLLM_TRACEABLE_MODULES.register_module({UNetMidBlock2D: "UNetMidBlock2D"})
class _UNetMidBlock2D(DynamicModule):
    def _setup(self, cfg: ConfigDict):
        self.cfg = cfg

    def forward(self, hidden_states, temb=None):
        hidden_states = self.resnets[0](hidden_states, temb)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            hidden_states = attn(hidden_states, temb=temb)
            hidden_states = resnet(hidden_states, temb)

        return hidden_states


@XHLLM_TRACEABLE_MODULES.register_module({UpDecoderBlock2D: "UpDecoderBlock2D"})
class _UpDecoderBlock2D(DynamicModule):
    def _setup(self, cfg: ConfigDict):
        self.cfg = cfg

    def forward(self, hidden_states: torch.Tensor, temb: Optional[torch.Tensor] = None) -> torch.Tensor:
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb=temb)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)

        return hidden_states

@XHLLM_TRACEABLE_MODULES.register_module({ResnetBlock2D: "ResnetBlock2D"})
class _ResnetBlock2D(DynamicModule):
    def _setup(self, cfg: ConfigDict):
        self.cfg = cfg
    
    def forward(self, input_tensor, temb, *args, **kwargs):
        hidden_states = input_tensor

        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)

        # if self.upsample is not None:
        #     # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
        #     if hidden_states.shape[0] >= 64:
        #         input_tensor = input_tensor.contiguous()
        #         hidden_states = hidden_states.contiguous()
        #     input_tensor = self.upsample(input_tensor)
        #     hidden_states = self.upsample(hidden_states)
        # elif self.downsample is not None:
        #     input_tensor = self.downsample(input_tensor)
        #     hidden_states = self.downsample(hidden_states)

        hidden_states = self.conv1(hidden_states)

        if self.time_emb_proj is not None:
            if not self.skip_time_act:
                temb = self.nonlinearity(temb)
            temb = self.time_emb_proj(temb)[:, :, None, None]

        if self.time_embedding_norm == "default":
            if temb is not None:
                hidden_states = hidden_states + temb
            hidden_states = self.norm2(hidden_states)
        elif self.time_embedding_norm == "scale_shift":
            if temb is None:
                raise ValueError(
                    f" `temb` should not be None when `time_embedding_norm` is {self.time_embedding_norm}"
                )
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

        output_tensor = (input_tensor + hidden_states) / self.output_scale_factor

        return output_tensor


def register_wrap_cls(hf_model):
    pass