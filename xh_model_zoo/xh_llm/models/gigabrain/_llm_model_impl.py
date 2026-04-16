import math
import sys
import types
from copy import deepcopy
from typing import Any, Dict, Optional, Tuple, Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import xhquant.nn as xhnn
from torch import Tensor
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.models.gemma2.modeling_gemma2 import (
    Gemma2Attention,
    Gemma2DecoderLayer,
    Gemma2ForCausalLM,
    Gemma2RMSNorm,
    Gemma2Model,
)
from xhquant.api import ConfigDict
from xhquant.nn import LLMCache, RMSNorm, Rope
from xhquant.utils.registry import DynamicModule

from ..builder import XHLLM_TRACEABLE_MODULES

# ============================================================
# Gemma2 RMSNorm Wrapper
# ============================================================
@XHLLM_TRACEABLE_MODULES.register_module({Gemma2RMSNorm: "Gemma2RMSNorm"})
class _Gemma2RMSNorm(DynamicModule):
    def forward(self, hidden_states):
        return self.norm(hidden_states)

    def _setup(self, cfg: Optional[Dict] = None):
        hidden_size = self.weight.shape[0]
        self.norm = RMSNorm(hidden_size, self.eps)
        self.norm.weight = nn.Parameter(deepcopy(self.weight.data + 1))
        return self

# ============================================================
# Gemma2 Attention Wrapper
# ============================================================
@XHLLM_TRACEABLE_MODULES.register_module({Gemma2Attention: "Gemma2Attention"})
class _Gemma2Attention(DynamicModule):

    def rotate_half(self, x: Tensor):
        """旋转一半的 hidden dims"""
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
        past_seq_length: Optional[Tensor] = None,
        current_input_length: Optional[Tensor] = None,
        past_k_cache: Optional[Tensor] = None,
        past_v_cache: Optional[Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = self.apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if self.use_cache:
            key_states = self.k_cache(key_states, past_seq_length, current_input_length, past_k_cache)
            value_states = self.v_cache(value_states, past_seq_length, current_input_length, past_v_cache)

        query_states = query_states * self.kv_scale
        key_states = key_states.transpose(2, 3)

        # GQA Repeat
        key_states = torch.repeat_interleave(key_states, self.num_key_value_groups, dim=1)
        value_states = torch.repeat_interleave(value_states, self.num_key_value_groups, dim=1)

        attn_weights = torch.matmul(query_states, key_states)
        attn_weights = attn_weights + attention_mask
        attn_weights = F.softmax(attn_weights, dim=-1)

        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)

        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    def _setup(self, cfg: Union[ConfigDict, Dict[str, Any]]):
        if isinstance(cfg, dict):
            cfg = ConfigDict(cfg)

        self.num_key_value_heads = getattr(self.config, 'num_key_value_heads', 4)
        self.num_heads = getattr(self.config, 'num_attention_heads', 8)
        self.head_dim = getattr(self.config, 'head_dim', getattr(self.config, 'hidden_size', 2304) // self.num_heads)
        self.hidden_size = getattr(self.config, 'hidden_size', 2304)
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        self.enable_rope = cfg.get("enable_rope", True)
        if self.enable_rope:
            self.rope = Rope()

        self.slice_1 = xhnn.Slice([0], [self.head_dim // 2], [3], [1])
        self.slice_2 = xhnn.Slice([self.head_dim // 2], [sys.maxsize], [3], [1])

        self.use_cache = cfg.use_cache
        if self.use_cache:
            cache_axis = cfg.kv_cache.cache_axis
            attention_max_length = -1
            self.k_cache = LLMCache(axis=cache_axis, attention_max_length=attention_max_length)
            self.v_cache = LLMCache(axis=cache_axis, attention_max_length=attention_max_length)

        # Pre-attention scalar: scaling = query_pre_attn_scalar ** -0.5
        self.kv_scale = getattr(self.config, 'query_pre_attn_scalar', 256.0) ** -0.5
        return self

# ============================================================
# Gemma2 Decoder Layer Wrapper
# ============================================================
@XHLLM_TRACEABLE_MODULES.register_module({Gemma2DecoderLayer: "Gemma2DecoderLayer"})
class _Gemma2DecoderLayer(DynamicModule):
    def forward(
        self,
        hidden_states: torch.Tensor,
        past_seq_length: Optional[Tensor] = None,
        current_input_length: Optional[Tensor] = None,
        past_k_cache: Optional[Tensor] = None,
        past_v_cache: Optional[Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:

        residual = hidden_states

        # Pre-attention norm
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            past_seq_length=past_seq_length,
            current_input_length=current_input_length,
            past_k_cache=past_k_cache,
            past_v_cache=past_v_cache,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
        )

        # Post-attention norm
        hidden_states = self.post_attention_layernorm(hidden_states)

        # Residual connection
        hidden_states = residual + hidden_states

        # MLP block
        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, attn_weights

    def _setup(self, cfg: Optional[Dict] = None):
        assert cfg is not None
        return self

# ============================================================
# Gemma2 Model Wrapper
# ============================================================
@XHLLM_TRACEABLE_MODULES.register_module({Gemma2Model: "Gemma2Model"})
class _Gemma2Model(DynamicModule):
    def _setup_position_cached(self):
        rotary_emb = self.rotary_emb
        max_seq_len = getattr(rotary_emb, 'max_seq_len_cached', 8192)
        # Compute rotary embeddings once
        position_ids = torch.arange(0, max_seq_len, dtype=torch.long, device=rotary_emb.inv_freq.device).unsqueeze(0)
        cos, sin = rotary_emb.forward(rotary_emb.inv_freq, position_ids)
        sin = sin.squeeze(0)
        cos = cos.squeeze(0)

        # Reshape to [1, 1, max_seq_len, feature_dim] to match apply_rotary_pos_emb
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

        self.register_buffer("sin_cached", sin.to(dtype=rotary_emb.inv_freq.dtype), persistent=False)
        self.register_buffer("cos_cached", cos.to(dtype=rotary_emb.inv_freq.dtype), persistent=False)

    def _setup(self, cfg: Optional[Dict] = None):
        self._setup_position_cached()
        self.num_logits_to_keep = cfg.num_logits_to_keep  # type: ignore # 1,取最后一个token的输出，0,取所有token的输出
        assert self.num_logits_to_keep in [0, 1]

        self.use_cache = cfg.use_cache
        input_seq_len = cfg.input_sequence_length  # type: ignore
        self.sin_slice = xhnn.DynamicSlice([input_seq_len], [2], [1])
        self.cos_slice = xhnn.DynamicSlice([input_seq_len], [2], [1])

        def _sin_cos_slice_update_cfg(self, cfg: Optional[Dict] = None):
            input_seq_len = cfg.input_sequence_length
            self.valid_length = [input_seq_len]

        self.sin_slice._update_cfg = types.MethodType(_sin_cos_slice_update_cfg, self.sin_slice)
        self.cos_slice._update_cfg = types.MethodType(_sin_cos_slice_update_cfg, self.cos_slice)

        self.llm_gather = xhnn.BatchGather(1)
        self.llm_gather.update_offset_indices(1, input_seq_len)

        def _llm_gather_update_cfg(self: xhnn.BatchGather, cfg: Optional[Dict] = None):
            input_seq_len = cfg.input_sequence_length  # type: ignore
            batch_size = cfg.get("batch_size", 1)  # type: ignore
            self.update_offset_indices(batch_size, input_seq_len)

        self.llm_gather._update_cfg = types.MethodType(_llm_gather_update_cfg, self.llm_gather)

    def forward(
        self,
        inputs_embeds: Optional[torch.Tensor] = None,
        past_seq_length: Optional[torch.Tensor] = None,
        current_input_length: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_cache: Optional[List[torch.Tensor]] = None,
        past_value_cache: Optional[List[torch.Tensor]] = None,
    ):
        hidden_states = inputs_embeds * (self.config.hidden_size ** 0.5)

        cos = self.cos_slice(self.cos_cached, past_seq_length)
        sin = self.sin_slice(self.sin_cached, past_seq_length)
        position_embeddings = (cos, sin)

        for idx, decoder_layer in enumerate(self.layers):
            if past_key_cache is not None:
                _past_k_cache = past_key_cache[idx]
                _past_v_cache = past_value_cache[idx]
            else:
                _past_k_cache = None
                _past_v_cache = None

            layer_outputs = decoder_layer(
                hidden_states,
                past_seq_length=past_seq_length,
                current_input_length=current_input_length,
                past_k_cache=_past_k_cache,
                past_v_cache=_past_v_cache,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
            )

            hidden_states = layer_outputs[0]

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
        )

# ============================================================
# Gemma2 For Causal LM Wrapper
# ============================================================
@XHLLM_TRACEABLE_MODULES.register_module({Gemma2ForCausalLM: "Gemma2ForCausalLM"})
class _Gemma2ForCausalLM(DynamicModule):
    def _setup(self, cfg: Optional[Dict] = None):
        pass

    def forward(
        self,
        inputs_embeds: Optional[Tensor] = None,
        past_seq_length: Optional[Tensor] = None,
        current_input_length: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        past_key_cache: Optional[List[Tensor]] = None,
        past_value_cache: Optional[List[Tensor]] = None,
    ):
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            past_seq_length=past_seq_length,
            current_input_length=current_input_length,
            attention_mask=attention_mask,
            past_key_cache=past_key_cache,
            past_value_cache=past_value_cache,
        )
        
        hidden_states = outputs[0]

        return hidden_states

def register_wrap_cls(hf_model):
    pass