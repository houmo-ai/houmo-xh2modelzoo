import math
import sys
from copy import deepcopy
from typing import Dict, Optional, Tuple, Union

import accelerate
import torch
import torch.nn as nn
from torch import Tensor
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention,
    Qwen3DecoderLayer,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
)
from xhquant import nn as xhnn
from xhquant.api import ConfigDict
from xhquant.nn import MaskedAdd, MaskedSoftmax, RMSNorm
from xhquant.utils.registry.dynamic_module import DynamicModule


MASK_VALUE = -66504.0


class Flux2KleinWrappedTextEncoderModel(nn.Module):
    def __init__(self, text_encoder: nn.Module, input_sequence_length: int):
        super().__init__()
        self.text_encoder = text_encoder
        self.input_sequence_length = input_sequence_length
        self.register_buffer(
            "cos_cached",
            text_encoder.model.rotary_emb.cos_cached[:, :, :input_sequence_length, :],
            persistent=False,
        )
        self.register_buffer(
            "sin_cached",
            text_encoder.model.rotary_emb.sin_cached[:, :, :input_sequence_length, :],
            persistent=False,
        )

    def forward(self, inputs_embeds: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        hidden_states = inputs_embeds
        cos = self.cos_cached
        sin = self.sin_cached
        position_embeddings = (cos, sin)
        # attention_mask = None if attention_mask is None else attention_mask.to(hidden_states.device, dtype=torch.int32)

        collected = [hidden_states]
        for decoder_layer in self.text_encoder.model.layers:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
            )[0]
            collected.append(hidden_states)

        return tuple(collected)


class _Flux2KleinQwen3RotaryEmbedding(DynamicModule):
    def _setup(self, cfg: Optional[Dict] = None):
        assert "dynamic" not in self.rope_type, f"{self.rope_type} is not supported in dynamic mode"
        self._setup_cos_sin_cache(seq_len=self.max_seq_len_cached, dtype=self.inv_freq.dtype)

    def _setup_cos_sin_cache(self, seq_len, dtype):
        position_ids = torch.arange(0, seq_len, dtype=torch.long, device=self.inv_freq.device).unsqueeze(0)

        inv_freq = self.inv_freq.to(torch.float32)
        device = self.inv_freq.device
        if torch.cuda.is_available() and inv_freq.device.type != "cuda":
            inv_freq = self.inv_freq.cuda()

        cos, sin = self.forward(inv_freq, position_ids)
        cos = cos.to(device)
        sin = sin.to(device)

        sin = sin.squeeze(0).unsqueeze(0).unsqueeze(0)
        cos = cos.squeeze(0).unsqueeze(0).unsqueeze(0)

        self.register_buffer("sin_cached", sin.to(dtype=dtype), persistent=False)
        self.register_buffer("cos_cached", cos.to(dtype=dtype), persistent=False)

    def _set_dtype(self, dtype: torch.dtype) -> None:
        self.inv_freq = self.inv_freq.to(dtype)
        self._setup_cos_sin_cache(seq_len=self.max_seq_len_cached, dtype=self.inv_freq.dtype)

    @torch.no_grad()
    def forward(self, x, position_ids):
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)

        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class _Flux2KleinQwen3Attention(DynamicModule):
    def rotate_half(self, x: Tensor):
        x1 = self.slice_1(x)
        x2 = self.slice_2(x)
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(self, q: Tensor, k: Tensor, cos: Tensor, sin: Tensor):
        if self.enable_rope:
            q_embed = self.rope(q, cos, sin)
            k_embed = self.rope(k, cos, sin)
        else:
            q_embed = (q * cos) + (self.rotate_half(q) * sin)
            k_embed = (k * cos) + (self.rotate_half(k) * sin)
        return q_embed, k_embed

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = self.q_norm(
            query_states.view(bsz, q_len, self.config.num_attention_heads, self.head_dim)
        ).transpose(1, 2)
        key_states = self.k_norm(
            key_states.view(bsz, q_len, self.config.num_key_value_heads, self.head_dim)
        ).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.config.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = self.apply_rotary_pos_emb(query_states, key_states, cos, sin)

        query_states = query_states * self.kv_scale
        key_states = torch.repeat_interleave(key_states.transpose(2, 3), self.num_key_value_groups, dim=1)
        value_states = torch.repeat_interleave(value_states, self.num_key_value_groups, dim=1)

        attn_weights = torch.matmul(query_states, key_states)
        if attention_mask is not None:
            attn_weights = self.masked_add(attn_weights, attention_mask)
        attn_weights = self.masked_softmax(attn_weights, self.past_seq_length)

        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.config.num_attention_heads * self.head_dim)
        attn_output = self.o_proj(attn_output)
        return attn_output, None, None

    def _setup(self, cfg: Union[ConfigDict, Dict[str, object]]):
        if isinstance(cfg, dict):
            cfg = ConfigDict(cfg)
        if not hasattr(self, "num_key_value_heads"):
            self.num_key_value_heads = self.config.num_key_value_heads
        if not hasattr(self, "num_heads"):
            self.num_heads = self.config.num_attention_heads
        if not hasattr(self, "hidden_size"):
            self.hidden_size = self.config.hidden_size

        self.enable_rope = cfg.get("enable_rope", True)
        if self.enable_rope:
            self.rope = xhnn.Rope()

        self.slice_1 = xhnn.Slice([0], [self.head_dim // 2], [3], [1])
        self.slice_2 = xhnn.Slice([self.head_dim // 2], [sys.maxsize], [3], [1])
        self.masked_add = MaskedAdd()
        self.masked_softmax = MaskedSoftmax(dim=-1)
        self.kv_scale = 1 / math.sqrt(self.head_dim)
        self.past_seq_length = torch.tensor(0, dtype=torch.long)
        return self


class _Flux2KleinQwen3DecoderLayer(DynamicModule):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        del kwargs
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return (hidden_states,)

    def _setup(self, cfg: Optional[Dict] = None):
        return self


class _Flux2KleinQwen3RMSNorm(DynamicModule):
    def forward(self, hidden_states):
        return self.norm(hidden_states)

    def _setup(self, cfg: Optional[Dict] = None):
        del cfg
        hidden_size = self.weight.shape[0]
        self.norm = RMSNorm(hidden_size, self.variance_epsilon)
        self.norm.weight = nn.Parameter(deepcopy(self.weight.data))
        return self


def wrap_text_encoder_model(text_encoder: nn.Module, config: Optional[Union[dict, ConfigDict]] = None) -> nn.Module:
    text_encoder = accelerate.hooks.remove_hook_from_module(text_encoder, recurse=True)
    if config is None:
        config = ConfigDict()
    if isinstance(config, dict):
        config = ConfigDict(config)

    wrapped_classes = {
        Qwen3RotaryEmbedding: _Flux2KleinQwen3RotaryEmbedding,
        Qwen3Attention: _Flux2KleinQwen3Attention,
        Qwen3DecoderLayer: _Flux2KleinQwen3DecoderLayer,
        Qwen3RMSNorm: _Flux2KleinQwen3RMSNorm,
    }
    for _, module in list(text_encoder.named_modules()):
        dynamic_cls = wrapped_classes.get(type(module))
        if dynamic_cls is not None and not isinstance(module, DynamicModule):
            dynamic_cls.convert(module, config)
    return text_encoder
