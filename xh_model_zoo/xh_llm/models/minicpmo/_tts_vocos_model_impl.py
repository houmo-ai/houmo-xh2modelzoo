# Copyright 2025 HOUMO AI
#
# File: _tts_vocos_model_impl.py
# Description:
#   Tts Vocos Model Impl model implementation.
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
from vocos.models import VocosBackbone
from xhquant import nn as xhnn
from xhquant.nn import LLMCacheV2, MaskedSoftmax, RMSNorm, LLMCache, Rope
from xhquant.utils.registry import DynamicModule
from xhquant.api import ConfigDict
from xhquant.utils.version_utils import digit_version
from vector_quantize_pytorch.residual_fsq import ResidualFSQ, GroupedResidualFSQ
from vocos.pretrained import Vocos
from vocos.heads import ISTFTHead
from vocos.modules import ConvNeXtBlock

from ..builder import XHLLM_TRACEABLE_MODULES

if digit_version(transformers.__version__) < digit_version("4.48.0"):
    from transformers.models.llama.modeling_llama import LlamaSdpaAttention
else:

    class LlamaSdpaAttention(LlamaAttention):
        pass


@XHLLM_TRACEABLE_MODULES.register_module({ISTFTHead: "ISTFTHead"})
class _ISTFTHead(DynamicModule):
    def _setup(self, cfg: Optional[Dict] = None):
        pass

    def forward(self, x: torch.Tensor, mask) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.out(x).transpose(1, 2)
        x *= mask
        mag, p = x.chunk(2, dim=1)
        mag = torch.exp(mag)
        mag = torch.clip(mag, max=1e2)  # safeguard to prevent excessively large magnitudes
        # wrapping happens here. These two lines produce real and imaginary value
        x = torch.cos(p)
        y = torch.sin(p)
        return x, y, mag

@XHLLM_TRACEABLE_MODULES.register_module({ConvNeXtBlock: "ConvNeXtBlock"})
class _ConvNeXtBlock(DynamicModule):
    def _setup(self, cfg: Optional[Dict] = None):
        pass
    
    def forward(self, x: torch.Tensor, cond_embedding_id: Optional[torch.Tensor] = None, mask: torch.Tensor = None, mask1: torch.Tensor = None) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        # x *= mask
        x = x.transpose(1, 2)  # (B, C, T) -> (B, T, C)
        if self.adanorm:
            assert cond_embedding_id is not None
            x = self.norm(x, cond_embedding_id)
        else:
            x = self.norm(x)
        # x *= mask1

        x = self.pwconv1(x)
        x = self.act(x)
        # x *= mask1
        x = self.pwconv2(x)
        # x *= mask1
        if self.gamma is not None:
            x = self.gamma * x
        x = x.transpose(1, 2)  # (B, T, C) -> (B, C, T)
        x *= mask
        x = residual + x
        return x

@XHLLM_TRACEABLE_MODULES.register_module({VocosBackbone: "VocosBackbone"})
class _VocosBackbone(DynamicModule):
    def _setup(self, cfg: Optional[Dict] = None):
        pass

    def forward(self, x: torch.Tensor, mask: torch.Tensor, **kwargs) -> torch.Tensor:
        mask1 = mask.transpose(1, 2)
        bandwidth_id = kwargs.get('bandwidth_id', None)
        x = self.embed(x)
        if self.adanorm:
            assert bandwidth_id is not None
            x = self.norm(x.transpose(1, 2), cond_embedding_id=bandwidth_id)
        else:
            x = self.norm(x.transpose(1, 2))
        x = x.transpose(1, 2)
        x *= mask
        for conv_block in self.convnext:
            x = conv_block(x, cond_embedding_id=bandwidth_id, mask=mask, mask1=mask1)
        x = self.final_layer_norm(x.transpose(1, 2))
        # x *= mask1
        return x


@XHLLM_TRACEABLE_MODULES.register_module({Vocos: "VOCOS"})
class _VOCOS(DynamicModule):
    def _setup(self, cfg: Optional[Dict] = None):
        pass

    @torch.inference_mode()
    def decode(self, features_input: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """
        Method to decode audio waveform from already calculated features. The features input is passed through
        the backbone and the head to reconstruct the audio output.

        Args:
            features_input (Tensor): The input tensor of features of shape (B, C, L), where B is the batch size,
                                     C denotes the feature dimension, and L is the sequence length.

        Returns:
            Tensor: The output tensor representing the reconstructed audio waveform of shape (B, T).
        """
        current_length = features_input.shape[2]
        mask = torch.ones(current_length).to(features_input).unsqueeze(0).unsqueeze(0)
        audio_output = self.decode_mask(features_input, mask=mask, **kwargs)
        return audio_output

    def decode_mask(self, features_input: torch.Tensor, mask: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        x = self.backbone(features_input, mask=mask, **kwargs)
        audio_output = self.head(x, mask=mask)
        return audio_output

def register_wrap_cls(hf_model):
    pass
