# Copyright 2025 HOUMO AI
#
# File: _tts_dvae_model_impl.py
# Description:
#   Tts Dvae Model Impl model implementation.
#
# Portions of this file adapt upstream OpenBMB MiniCPM / MiniCPM-V TTS / DVAE
# components for local tracing, wrapping, and export workflows.
# Original project: https://github.com/OpenBMB/MiniCPM-V
# Original license: Apache License, Version 2.0
# Modifications by HOUMO AI: local tracing, wrapping, and export integration.
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
import sys
import types
from typing import Dict, List, Optional, Tuple, Union, Any

import torch
import torch.nn as nn
import transformers
from torch import Tensor
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
)
from xhquant import nn as xhnn
from xhquant.nn import LLMCacheV2, MaskedSoftmax, RMSNorm, LLMCache, Rope
from xhquant.utils.registry import DynamicModule
from xhquant.api import ConfigDict
from xhquant.utils.version_utils import digit_version
from vector_quantize_pytorch.residual_fsq import ResidualFSQ, GroupedResidualFSQ

from ..builder import XHLLM_TRACEABLE_MODULES

if digit_version(transformers.__version__) < digit_version("4.48.0"):
    from transformers.models.llama.modeling_llama import LlamaSdpaAttention
else:

    class LlamaSdpaAttention(LlamaAttention):
        pass


def gather_codebooks_gather(codebooks: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    b, n, q = indices.shape
    q2, c, d = codebooks.shape
    cb = codebooks.unsqueeze(0).unsqueeze(0).expand(b, n, q, c, d)                                  # (b,n,q,c,d) 视图
    idx = indices.to(torch.long).unsqueeze(-1).unsqueeze(-1).expand(b, n, q, 1, d)                  # (b,n,q,1,d)
    out = torch.gather(cb, dim=3, index=idx).squeeze(3)                                             # (b,n,q,d)
    return out.permute(2, 0, 1, 3).contiguous()  


@XHLLM_TRACEABLE_MODULES.register_module({ResidualFSQ: "ResidualFSQ"})
class _ResidualFSQ(DynamicModule):
    def _setup(self, cfg: Optional[Dict] = None):
        pass
    
    def get_codes_from_indices(self, indices: torch.Tensor) -> torch.Tensor:
        batch, quantize_dim = indices.shape[0], indices.shape[-1]

        all_codes = gather_codebooks_gather(self.codebooks, indices)
        scale = self.scales.unsqueeze(1).unsqueeze(1)
        all_codes = all_codes * scale
        return all_codes

    def get_output_from_indices(self, indices: torch.Tensor) -> torch.Tensor:
        codes = self.get_codes_from_indices(indices)
        codes_summed = torch.sum(codes, dim=0)
        return self.project_out(codes_summed)

@XHLLM_TRACEABLE_MODULES.register_module({GroupedResidualFSQ: "GroupedResidualFSQ"})
class _GroupedResidualFSQ(DynamicModule):
    def _setup(self, cfg: Optional[Dict] = None):
        pass
    
    def get_output_from_indices(self, indices: torch.Tensor) -> torch.Tensor:
        # indices: [num_rvq, ...]
        outputs = []
        for i, rvq in enumerate(self.rvqs):
            outputs.append(rvq.get_output_from_indices(indices[i]))
        return torch.cat(outputs, dim=self.split_dim)

@XHLLM_TRACEABLE_MODULES.register_module({"transformers_modules.MiniCPM-o-2_6.modeling_minicpmo.GFSQ": "GFSQ"})
class _GFSQ(DynamicModule):
    def _setup(self, cfg: Optional[Dict] = None):
        pass

@XHLLM_TRACEABLE_MODULES.register_module({"transformers_modules.MiniCPM-o-2_6.modeling_minicpmo.ConvNeXtBlock": "ConvNeXtBlock"})
class _ConvNeXtBlock(DynamicModule):
    def _setup(self, cfg: Optional[Dict] = None):
        pass
    
    def forward(self, x: torch.Tensor, cond=None, mask=None, mask1=None) -> torch.Tensor:
        residual = x

        y = self.dwconv(x)
        # y *= mask
        y.transpose_(1, 2)  # (B, C, T) -> (B, T, C)
        x = self.norm(y)
        del y
        y = self.pwconv1(x)
        del x
        x = self.act(y)
        # x *= mask1
        del y
        y = self.pwconv2(x)
        # y *= mask1
        del x
        if self.coef is not None:
            y *= self.coef
        y.transpose_(1, 2)  # (B, T, C) -> (B, C, T)
        y *= mask
        x = y + residual
        del y

        return x

@XHLLM_TRACEABLE_MODULES.register_module({"transformers_modules.MiniCPM-o-2_6.modeling_minicpmo.DVAEDecoder": "minicpmo.DVAEDecoder"})
class _DVAEDecoder(DynamicModule):
    def _setup(self, cfg: Optional[Dict] = None):
        pass
    
    def forward(self, x: torch.Tensor, conditioning=None, mask=None, mask1=None) -> torch.Tensor:
        # B, C, T
        y = self.conv_in[0](x)
        y = self.conv_in[1](y)
        y *= mask
        y = self.conv_in[2](y)
        y *= mask
        del x
        for f in self.decoder_block:
            y = f(y, conditioning, mask=mask, mask1=mask1)

        x = self.conv_out(y)
        x *= mask
        del y
        return x

@XHLLM_TRACEABLE_MODULES.register_module({"transformers_modules.MiniCPM-o-2_6.modeling_minicpmo.DVAE": "DVAE"})
class _DVAE(DynamicModule):
    def _setup(self, cfg: Optional[Dict] = None):
        pass

    @torch.inference_mode()
    def forward(self, inp: torch.Tensor, mode = "decode", mask=None) -> torch.Tensor:
        if mode == "encode" and hasattr(self, "encoder") and self.vq_layer is not None:
            mel = inp.clone()
            x: torch.Tensor = self.downsample_conv(
                torch.div(mel, self.coef.view(100, 1).expand(mel.shape), out=mel),
            ).unsqueeze_(0)
            del mel
            x = self.encoder(x)
            ind = self.vq_layer(x)
            del x
            return ind

        if mask is None:
            current_length = inp.shape[2] * 2
            mask = torch.ones(current_length).float().to(inp.device)
            mask = mask.unsqueeze(0).unsqueeze(0)

        vq_feats = self.forward_part1(inp)

        vp_feats = vq_feats.flatten(2)

        dec_out = self.forward_part2(vp_feats, mask=mask)
        return dec_out

    def forward_part1(self, inp: torch.Tensor) -> torch.Tensor:
        if self.vq_layer is not None:
            vq_feats = self.vq_layer._embed(inp)
        else:
            vq_feats = inp

        vq_feats = (
            vq_feats.view(
                (vq_feats.size(0), 2, vq_feats.size(1) // 2, vq_feats.size(2)),
            )
            .permute(0, 2, 3, 1)
        )
        return vq_feats

    def forward_part2(self, vq_feats: torch.Tensor, mask=None) -> torch.Tensor:
        mask1 = mask.transpose(1, 2)
        dec_out = self.out_conv(
            self.decoder(
                x=vq_feats,
                mask=mask,
                mask1=mask1,
            ),
        )
        dec_out *= mask
        dec_out = torch.mul(dec_out, self.coef)
        return dec_out

def register_wrap_cls(hf_model):
    pass
