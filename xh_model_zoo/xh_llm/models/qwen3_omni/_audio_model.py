import math
import sys
import types
from copy import deepcopy
from typing import Callable, Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as F
from transformers.modeling_outputs import BaseModelOutputWithPast
from xhquant import nn as xhnn
from xhquant.api import ConfigDict
from xhquant.nn import (
    BfpFlashAttention,
    LLMCache,
    MaskedSoftmax,
    SinksMaskedSoftmax,
    RMSNorm,
    Rope,
    MoeBlock,
)
from xhquant.utils.registry import DynamicModule

from xhquant.backend.xh2a import generate_lut_table_v2, verbose_lut

from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeAudioAttention,
    Qwen3OmniMoeAudioEncoder,
    Qwen3OmniMoeAudioEncoderLayer,
)
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    MoeCausalLMOutputWithPast,
    MoeModelOutputWithPast,
)

from xhquant.utils.registry import DynamicModule

from ..builder import XHLLM_TRACEABLE_MODULES


class _Qwen3OmniAudioDynamicModule(DynamicModule):
    """Base wrapper for Qwen3 Omni audio modules."""

    def _setup(self, cfg=None):
        return None


@XHLLM_TRACEABLE_MODULES.register_module(
    {
        Qwen3OmniMoeAudioAttention: "Qwen3OmniMoeAudioAttention",
    }
)
class _Qwen3OmniAudioAttention(_Qwen3OmniAudioDynamicModule):
    """Register attention wrapper for audio encoder."""
    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # Use static shape constants to avoid data-dependent branching in torch.export
        query_states = self.q_proj(hidden_states).reshape(-1, self.num_heads, self._head_dim)
        key_states = self.k_proj(hidden_states).reshape(-1, self.num_heads, self._head_dim)
        value_states = self.v_proj(hidden_states).reshape(-1, self.num_heads, self._head_dim)

        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)
        max_seqlen = cu_seqlens[1:] - cu_seqlens[:-1]

        key_states = torch.repeat_interleave(
            key_states,
            self.num_key_value_groups,
            dim=1,
        )
        value_states = torch.repeat_interleave(
            value_states,
            self.num_key_value_groups,
            dim=1,
        )

        attn_weights = torch.matmul(query_states * self.scaling, key_states.transpose(2, 3))
        attn_weights: Optional[Tensor] = self.masked_softmax(attn_weights, max_seqlen)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(-1, self._embed_dim_out).contiguous()
        attn_output = self.out_proj(attn_output)

        return attn_output

    def _setup(self, cfg=None):
        self.masked_softmax = MaskedSoftmax(dim=-1)
        self._head_dim = int(self.head_dim)
        self._embed_dim_out = int(self.num_heads * self.head_dim)


@XHLLM_TRACEABLE_MODULES.register_module(
    {
        Qwen3OmniMoeAudioEncoderLayer: "Qwen3OmniMoeAudioEncoderLayer",
    }
)
class _Qwen3OmniAudioEncoderLayer(_Qwen3OmniAudioDynamicModule):
    """Register encoder layer wrapper for audio encoder."""
    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            cu_seqlens=cu_seqlens,
            attention_mask=attention_mask,
            **kwargs,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states

        # if hidden_states.dtype == torch.float16:
        #     clamp_value = torch.finfo(hidden_states.dtype).max - 1000
        #     hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        return outputs

    def _setup(self, cfg=None):
        self.activation_fn = nn.GELU()

def _get_feat_extract_output_lengths(input_lengths):
    """
    Computes the output length of the convolutional layers and the output length of the audio encoder
    """

    input_lengths_leave = input_lengths % 100
    feat_lengths = (input_lengths_leave - 1) // 2 + 1
    output_lengths = ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13
    return output_lengths

@XHLLM_TRACEABLE_MODULES.register_module(
    {
        Qwen3OmniMoeAudioEncoder: "Qwen3OmniMoeAudioEncoder",
    }
)
class _Qwen3OmniMoeAudioEncoder(_Qwen3OmniAudioDynamicModule):
    def forward(
        self,
        padded_feature,
        cu_seqlens=None,
    ):
        r"""
        feature_lens (`torch.LongTensor` of shape `(batch_size,)`):
            mel length
        aftercnn_lens (`torch.LongTensor` of shape `(batch_size,)`):
            mel length after cnn
        """

        padded_feature = padded_feature.unsqueeze(1)
        # Split to chunk to avoid OOM during convolution
        # padded_embeds = []
        # for chunk in padded_feature.split(self.conv_chunksize, dim=0):
        #     padded_embed = F.gelu(self.conv2d1(chunk))
        #     padded_embed = F.gelu(self.conv2d2(padded_embed))
        #     padded_embed = F.gelu(self.conv2d3(padded_embed))
        #     padded_embeds.append(padded_embed)
        # padded_embed = torch.cat(padded_embeds, dim=0)
        padded_embed = F.gelu(self.conv2d1(padded_feature))
        chip_len = math.ceil(self.n_window*2/2)
        padded_embed = F.gelu(self.conv2d2(padded_embed))
        chip_len = math.ceil(chip_len/2)
        padded_embed = F.gelu(self.conv2d3(padded_embed))
        chip_len = math.ceil(chip_len/2)


        # Use pre-computed static shape constants to avoid data-dependent branching
        padded_embed = self.conv_out(padded_embed.permute(0, 3, 1, 2).contiguous().view(
            self._static_view_b, self._static_view_t, self._static_view_cf
        ))

        positional_embedding = (
            self.positional_embedding.positional_embedding[: chip_len, :]
            .unsqueeze(0)
            .to(padded_embed.dtype)
        )
        padded_embed = padded_embed + positional_embedding
        hidden_states = padded_embed.reshape(-1, self._static_embed_dim)
        # cu_chunk_lens = [0]
        # window_aftercnn = padded_mask_after_cnn.shape[-1] * (self.n_window_infer // (self.n_window * 2))
        # for cnn_len in aftercnn_lens:
        #     cu_chunk_lens += [window_aftercnn] * (cnn_len // window_aftercnn)
        #     remainder = cnn_len % window_aftercnn
        #     if remainder != 0:
        #         cu_chunk_lens += [remainder]
        # cu_seqlens = torch.tensor(cu_chunk_lens, device=aftercnn_lens.device).cumsum(-1, dtype=torch.int32)

        for encoder_layer in self.layers:
            layer_outputs = encoder_layer(
                hidden_states,
                cu_seqlens,
            )

            hidden_states = layer_outputs[0]

        hidden_states = self.ln_post(hidden_states)
        hidden_states = self.proj1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.proj2(hidden_states)
        return hidden_states

    def _setup(self, cfg=None):
        # Pre-compute conv output shape as static Python ints for export
        mel_bins = int(cfg.get('mel_bins', 128)) if cfg else 128
        mel_length = int(cfg.get('mel_length', 100)) if cfg else 100
        with torch.no_grad():
            dummy = torch.zeros(1, 1, mel_bins, mel_length, dtype=torch.float16)
            x = self.conv2d1(dummy)
            x = self.conv2d2(x)
            x = self.conv2d3(x)
            self._static_view_b = 1
            self._static_view_t = int(x.shape[3])
            self._static_view_cf = int(x.shape[1] * x.shape[2])
        # embed_dim after conv_out projection
        self._static_embed_dim = int(self.conv_out.out_features)

def register_wrap_modules():
    """Compatibility helper for API parity with thinker registration."""
    return None
