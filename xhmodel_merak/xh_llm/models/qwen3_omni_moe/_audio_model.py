import math
from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeAudioAttention,
    Qwen3OmniMoeAudioEncoder,
    Qwen3OmniMoeAudioEncoderLayer,
)

from xhquant.nn import (
    MaskedSoftmax,
)
from xhquant.utils.registry import DynamicModule

from ...register import XHLLM_TRACEABLE_MODULES


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

        seq_length, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states).reshape(seq_length, self.num_heads, -1)
        key_states = self.k_proj(hidden_states).reshape(seq_length, self.num_heads, -1)
        value_states = self.v_proj(hidden_states).reshape(seq_length, self.num_heads, -1)

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
        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        attn_output = self.out_proj(attn_output)

        return attn_output

    def _setup(self, cfg=None):
        self.masked_softmax = MaskedSoftmax(dim=-1)


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
        padded_mask_after_cnn,
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
        chip_len = math.ceil(self.n_window * 2 / 2)
        padded_embed = F.gelu(self.conv2d2(padded_embed))
        chip_len = math.ceil(chip_len / 2)
        padded_embed = F.gelu(self.conv2d3(padded_embed))
        chip_len = math.ceil(chip_len / 2)

        b, c, f, t = padded_embed.size()
        padded_embed = self.conv_out(padded_embed.permute(0, 3, 1, 2).contiguous().view(b, t, c * f))

        positional_embedding = (
            self.positional_embedding.positional_embedding[:chip_len, :].unsqueeze(0).to(padded_embed.dtype)
        )
        padded_embed = padded_embed + positional_embedding
        hidden_states = padded_embed[padded_mask_after_cnn]
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
        pass


def register_wrap_modules():
    """Compatibility helper for API parity with thinker registration."""
    return None
