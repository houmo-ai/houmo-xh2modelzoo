# Copyright 2025 HOUMO AI
#
# File: _vae_model_imp.py
# Description:
#   Vae Model Imp model implementation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

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
from diffusers.models.autoencoders.autoencoder_kl_qwenimage import AutoencoderKLQwenImage
from diffusers.models.autoencoders.autoencoder_kl_qwenimage import QwenImageDecoder3d
from diffusers.models.autoencoders.autoencoder_kl_qwenimage import QwenImageCausalConv3d
from diffusers.models.autoencoders.autoencoder_kl_qwenimage import QwenImageRMS_norm
from diffusers.models.autoencoders.autoencoder_kl_qwenimage import QwenImageResidualBlock
from diffusers.models.autoencoders.autoencoder_kl_qwenimage import QwenImageMidBlock
from diffusers.models.autoencoders.autoencoder_kl_qwenimage import QwenImageAttentionBlock
from diffusers.models.autoencoders.autoencoder_kl_qwenimage import QwenImageUpBlock
from diffusers.models.autoencoders.autoencoder_kl_qwenimage import QwenImageResample

import torch.nn.functional as F
import torch.nn as nn

@XHLLM_TRACEABLE_MODULES.register_module({QwenImageAttentionBlock: "QwenImageAttentionBlock"})
class _QwenImageAttentionBlock(DynamicModule):
    def forward(self, x):
        identity = x
        # batch_size, channels, time, height, width = x.size()
        # assert channels == 384 and height == 116 and width==208
        
        x = x.permute(0, 2, 1, 3, 4).reshape(1, self.channels, self.height, self.width)
        x = self.norm(x)

        x = x.permute(0,2,3,1)

        query_states = self.q_proj(x)
        key_states = self.k_proj(x)
        value_states = self.v_proj(x)

        query_states = query_states.permute(0,3,1,2).view(1, 1, self.channels, -1)
        key_states = key_states.permute(0,3,1,2).view(1, 1, self.channels, -1)
        value_states = value_states.permute(0,3,1,2).view(1, 1, self.channels, -1)

        query_states = query_states.transpose(2, 3)
        key_states = key_states.transpose(2, 3)
        value_states = value_states.transpose(2, 3)

        # # compute query, key, value
        # qkv = self.to_qkv(x)
        # qkv = qkv.reshape(batch_size * time, 1, channels * 3, -1)
        # qkv = qkv.permute(0, 1, 3, 2).contiguous()
        # q, k, v = qkv.chunk(3, dim=-1)

        query_states = query_states * self.kv_scale
        key_states = key_states.transpose(2, 3)
        attn_weights = torch.matmul(query_states, key_states)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, value_states)
        # attn_output = attn_output.transpose(1, 2)
        # attn_output = attn_output.reshape(bsz, -1, self.hidden_size)

        # # apply attention
        # x = F.scaled_dot_product_attention(q, k, v)

        x = attn_output.squeeze(1).permute(0, 2, 1).reshape(1, self.channels, self.height, self.width)

        # output projection
        x = self.proj(x)

        # Reshape back: [(b*t), c, h, w] -> [b, c, t, h, w]
        x = x.view(1, 1, self.channels, self.height, self.width)
        x = x.permute(0, 2, 1, 3, 4)

        return x + identity

    def _setup(self, cfg: Optional[Dict] = None):
        self.masked_softmax = MaskedSoftmax(dim=-1)

        self.channels = 384
        self.height = 128
        self.width = 128
        
        weight = self.to_qkv.weight.data.clone()
        bias = self.to_qkv.bias.data.clone()

        dim = weight.shape[0] // 3

        weight = weight.permute(1, 0, 2, 3).reshape(dim, 3, dim).permute(1, 2, 0)
        bias = bias.reshape(3, dim)
        self.q_proj = nn.Linear(dim, dim, bias=True)
        self.k_proj = nn.Linear(dim, dim, bias=True)
        self.v_proj = nn.Linear(dim, dim, bias=True)

        self.q_proj.weight.data = weight[0]
        self.q_proj.bias.data = bias[0]
        self.k_proj.weight.data = weight[1]
        self.k_proj.bias.data = bias[1]
        self.v_proj.weight.data = weight[2]
        self.v_proj.bias.data = bias[2]
        self.maskedadd = xhnn.MaskedAdd()

        _kv_scale = 1 / math.sqrt(dim)
        self.kv_scale = _kv_scale

        self.masked_softmax = MaskedSoftmax(dim=-1)

        self.to_qkv = None
        del self.to_qkv
        torch.cuda.empty_cache()
        return self

@XHLLM_TRACEABLE_MODULES.register_module({QwenImageRMS_norm: "QwenImageRMS_norm"})
class _QwenImageRMS_norm(DynamicModule):
    def forward(self, hidden_states):
        return self.norm(hidden_states)

    def _setup(self, cfg: Optional[Dict] = None):
        assert self.bias == 0
        dim = 1 if self.channel_first else -1
        weight = (self.gamma).reshape(-1) # self.scale * 
        hidden_size = weight.shape[0]
        self.norm = RMSNorm(hidden_size, reduce_dim=dim)
        self.norm.weight = nn.Parameter(deepcopy(weight.data))
        return self
    
@XHLLM_TRACEABLE_MODULES.register_module({QwenImageMidBlock: "QwenImageMidBlock"})
class _QwenImageMidBlock(DynamicModule):
    def forward(self, x, feat_cache=None, feat_idx=[0]):
        # First residual block
        x = self.resnets[0](x, feat_cache, feat_idx)

        # Process through attention and residual blocks
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            x = attn(x)
            x = resnet(x, feat_cache, feat_idx)

        return x

    def _setup(self, cfg: ConfigDict):
        self.cfg = cfg

@XHLLM_TRACEABLE_MODULES.register_module({QwenImageResidualBlock: "QwenImageResidualBlock"})
class _QwenImageResidualBlock(DynamicModule):
    def forward(self, x, feat_cache=None, feat_idx=[0]):
        # Apply shortcut connection
        h = self.conv_shortcut(x)

        # First normalization and activation
        x = self.norm1(x)
        x = self.nonlinearity(x)

        x = self.conv1(x)

        # Second normalization and activation
        x = self.norm2(x)
        x = self.nonlinearity(x)

        # # Dropout
        # x = self.dropout(x)

        x = self.conv2(x)
        # Add residual connection
        return x + h

    def _setup(self, cfg: ConfigDict):
        self.cfg = cfg 

@XHLLM_TRACEABLE_MODULES.register_module({QwenImageResample: "QwenImageResample"})
class _QwenImageResample(DynamicModule):
    def forward(self, x, feat_cache=None, feat_idx=[0]):
        b, c, t, h, w = x.size()
        # assert c == 384 and h == 116 and w==208  # 232 416
        # print(c, h, w)
        t = x.shape[2]
        x = x.permute(0, 2, 1, 3, 4).reshape(1, self.b_c, self.b_h, self.b_w)
        x = self.resample(x)
        x = x.view(1, 1, self.a_c, self.a_h, self.a_w).permute(0, 2, 1, 3, 4)
        return x

    def _setup(self, cfg: ConfigDict):
        self.cfg = cfg
        self.resample[0].mode = "nearest"

        self.b_c = self.inp_shape[0]
        self.b_h = self.inp_shape[1]
        self.b_w = self.inp_shape[2]

        self.a_c = int(self.inp_shape[0] / 2)
        self.a_h = int(self.inp_shape[1] * 2)
        self.a_w = int(self.inp_shape[2] * 2)   



@XHLLM_TRACEABLE_MODULES.register_module({AutoencoderKLQwenImage: "AutoencoderKLQwenImage"})
class _AutoencoderKLQwenImage(DynamicModule):
    def _setup(self, cfg: ConfigDict):
        self.cfg = cfg
        del self.encoder

    def decode(self, z: torch.Tensor, return_dict: bool = True):
        decoded = self._decode(z).sample
        return decoded
    
    def _decode(self, z: torch.Tensor, return_dict: bool = True):
        _, _, num_frame, height, width = z.shape
        x = self.post_quant_conv(z)
        # assert num_frame == 1
        # for i in range(num_frame):
        #     self._conv_idx = [0]
        #     if i == 0:
        #         out = self.decoder(x[:, :, i : i + 1, :, :], feat_cache=self._feat_map, feat_idx=self._conv_idx)
        #     else:
        #         out_ = self.decoder(x[:, :, i : i + 1, :, :], feat_cache=self._feat_map, feat_idx=self._conv_idx)
        #         out = torch.cat([out, out_], 2)
        out = self.decoder(x, feat_cache=None, feat_idx=[0])
        out = torch.clamp(out, min=-1.0, max=1.0)
        return out
    
    def forward(self, x: torch.Tensor):
        decoded = self._decode(x)
        return decoded

@XHLLM_TRACEABLE_MODULES.register_module({QwenImageDecoder3d: "QwenImageDecoder3d"})
class _QwenImageDecoder3d(DynamicModule):
    def _setup(self, cfg: ConfigDict):
        self.cfg = cfg
        self.up_blocks[0].upsamplers[0].inp_shape = [384, 128, 128] # [192, 232, 416]
        self.up_blocks[1].upsamplers[0].inp_shape = [384, 256, 256] # [192, 464, 832]
        self.up_blocks[2].upsamplers[0].inp_shape = [192, 512, 512] # [96, 928, 1664]
        # 384 116 208
        # 384 232 416
        # 192 464 832
    def forward(self, x, feat_cache=None, feat_idx=[0]):
        ## conv1
        x = self.conv_in(x)

        ## middle
        x = self.mid_block(x, feat_cache, feat_idx)

        ## upsamples
        for up_block in self.up_blocks:
            x = up_block(x, feat_cache, feat_idx)

        ## head
        x = self.norm_out(x)
        x = self.nonlinearity(x)

        x = self.conv_out(x)
        return x

@XHLLM_TRACEABLE_MODULES.register_module({QwenImageCausalConv3d: "QwenImageCausalConv3d"})
class _QwenImageCausalConv3d(DynamicModule):
    def _setup(self, cfg: Optional[Dict] = None):

        weight = self.weight.clone()  # [C_out, C_in/groups, kD, kH, kW]
        bias = self.bias.clone()  # [C_out] 或 None
        stride = self.stride  # int 或 3-tuple
        padding = self.padding  # int 或 3-tuple
        dilation = self.dilation  # int 或 3-tuple
        groups = self.groups

        def to_3(x):
            if isinstance(x, int):
                return (x, x, x)
            return x

        stride = to_3(stride)
        padding = to_3(padding)
        dilation = to_3(dilation)

        sd, sh, sw = stride
        pd, ph, pw = padding
        dd, dh, dw = dilation     

        assert padding == (0, 0, 0) 
        assert stride == (1, 1, 1) 
        assert dilation == (1, 1, 1)

        C_out, Cin_g, kD, kH, kW = weight.shape

        # 3D 卷积输出尺寸公式
        def out_size(L_in, k, stride, pad, dil):
            return (L_in + 2 * pad - dil * (k - 1) - 1) // stride + 1

        # D_out = out_size(D_in, kD, sd, pd, dd)
        # H_out = out_size(H_in, kH, sh, ph, dh)
        # W_out = out_size(W_in, kW, sw, pw, dw)
        self.pad_add = self._padding[4]
    
        w_slice = weight[:, :, 0+self.pad_add , :, :]

        self.conv_2dd = nn.Conv2d(
            Cin_g,
            C_out,
            kernel_size=(kH, kW),
            stride=(sh, sw),
            padding=(ph, pw),
            dilation=(dh, dw),
            groups=groups,
            # bias=False
        )
        self.conv_2dd.weight.data = w_slice
        self.conv_2dd.bias.data = bias

        self.weight = None
        self.bias = None
        del self.weight
        del self.bias
        torch.cuda.empty_cache()
        
        return self

    def forward(self, x, cache_x=None):
        # assert x.shape[2] == 1
        x = F.pad(x, self._padding)
        # 取一个 depth slice: [N, C_in, H, W]
        x_slice = x[:, :, self.pad_add, :, :]

        # 对应的 2D 卷积 kernel: [C_out, C_in/groups, kH, kW]
        out = self.conv_2dd(x_slice)
        out = out.unsqueeze(2)
        return out



def register_wrap_cls(hf_model):
    pass