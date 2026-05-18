"""
DynamicModule wrappers for Gemma 4 text LLM (MoE + hybrid attention)
with explicit attention masks.

This file is vendored into xh2modelzoo so Gemma4 MoE with-mask no longer
depends on the sibling xhquant_llm package at runtime.
"""

from __future__ import annotations

import math
import sys
import types
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import xhquant.nn as xhnn
from torch import Tensor
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.gemma4.modeling_gemma4 import (
    Gemma4ForCausalLM,
    Gemma4RMSNorm,
    Gemma4TextAttention,
    Gemma4TextDecoderLayer,
    Gemma4TextModel,
)

from xhquant.api import ConfigDict
from xhquant.nn import LLMCacheV2, MaskedAdd, RMSNorm, Rope, SoftmaxPlus
from xhquant.nn.modules.moeblock import MoeBlock
from xhquant.utils.registry import DynamicModule

from ...register import XHLLM_TRACEABLE_MODULES


@XHLLM_TRACEABLE_MODULES.register_module({Gemma4RMSNorm: "Gemma4RMSNorm"})
class _Gemma4RMSNorm(DynamicModule):
    def forward(self, hidden_states):
        return self.norm(hidden_states)

    def _setup(self, cfg: Optional[Dict] = None):
        if not self.with_scale:
            class _NoScaleRMSNorm(nn.Module):
                def __init__(self, eps):
                    super().__init__()
                    self.eps = eps

                def forward(self, x):
                    mean_sq = x.float().pow(2).mean(-1, keepdim=True) + self.eps
                    return (x.float() * torch.pow(mean_sq, -0.5)).type_as(x)

            self.norm = _NoScaleRMSNorm(self.eps)
            return self

        hidden_size = self.weight.shape[0]
        self.norm = RMSNorm(hidden_size, self.eps)
        self.norm.weight = nn.Parameter(deepcopy(self.weight.data))
        return self


@XHLLM_TRACEABLE_MODULES.register_module({Gemma4TextAttention: "Gemma4TextAttention"})
class _Gemma4TextAttention(DynamicModule):
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
        hidden_states: Tensor,
        past_seq_length: Optional[Tensor] = None,
        current_input_length: Optional[Tensor] = None,
        local_attention_mask: Optional[Tensor] = None,
        global_attention_mask: Optional[Tensor] = None,
        past_k_cache: Optional[Tensor] = None,
        past_v_cache: Optional[Tensor] = None,
        position_embeddings: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        bsz, q_len, _ = hidden_states.size()
        hidden_states = hidden_states.to(self.q_proj.weight.dtype)

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if self._use_alternative_attention:
            value_states = key_states.clone()
        else:
            value_states = self.v_proj(hidden_states)
            value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)
        value_states = self.v_norm(value_states)

        cos, sin = position_embeddings
        query_states, key_states = self.apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if self.use_cache:
            key_states = self.k_cache(key_states, past_seq_length, current_input_length, past_k_cache)
            value_states = self.v_cache(value_states, past_seq_length, current_input_length, past_v_cache)

        if self.use_cache:
            _, _, _, _ = past_k_cache.shape

        attn_dtype = key_states.dtype
        query_states = query_states.to(attn_dtype)
        value_states = value_states.to(attn_dtype)

        query_states = query_states * self.kv_scale
        key_states = key_states.transpose(2, 3)

        key_states = torch.repeat_interleave(key_states, self.num_key_value_groups, dim=1)
        attn_weights = torch.matmul(query_states, key_states)

        causal_mask = local_attention_mask if self.is_sliding else global_attention_mask
        attn_weights = self.masked_add(attn_weights, causal_mask)
        attn_weights = self.softmax(attn_weights).to(attn_dtype)

        value_states = torch.repeat_interleave(value_states, self.num_key_value_groups, dim=1)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)
        attn_output = attn_output.to(self.o_proj.weight.dtype)
        attn_output = self.o_proj(attn_output)

        return attn_output, None

    def _setup(self, cfg: Union[ConfigDict, Dict[str, Any]]):
        if isinstance(cfg, dict):
            cfg = ConfigDict(cfg)

        self._use_alternative_attention = getattr(self, "use_alternative_attention", False)

        if not hasattr(self, "num_key_value_heads"):
            if self._use_alternative_attention:
                self.num_key_value_heads = self.config.num_global_key_value_heads
            else:
                self.num_key_value_heads = self.config.num_key_value_heads
        if not hasattr(self, "num_heads"):
            self.num_heads = self.config.num_attention_heads

        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        self.enable_rope = cfg.get("enable_rope", True)
        self.rope = Rope()

        self.slice_1 = xhnn.Slice([0], [self.head_dim // 2], [3], [1])
        self.slice_2 = xhnn.Slice([self.head_dim // 2], [sys.maxsize], [3], [1])

        attention_max_length = self.sliding_window if self.is_sliding and self.sliding_window is not None else -1

        self.softmax = SoftmaxPlus(dim=-1)
        self.masked_add = MaskedAdd()

        use_cache = cfg.use_cache
        self.use_cache = use_cache
        self.cos_gather = xhnn.Gather(0)
        self.sin_gather = xhnn.Gather(0)

        if use_cache:
            cache_axis = cfg.kv_cache.cache_axis
            self.k_cache = LLMCacheV2(axis=cache_axis, attention_max_length=attention_max_length)
            self.v_cache = LLMCacheV2(axis=cache_axis, attention_max_length=attention_max_length)

        self.kv_scale = 1.0

        return self


@XHLLM_TRACEABLE_MODULES.register_module({Gemma4TextDecoderLayer: "Gemma4TextDecoderLayer"})
class _Gemma4TextDecoderLayer(DynamicModule):
    def _setup(self, cfg: Optional[Dict] = None):
        assert cfg is not None
        self.use_clip = cfg.get("use_clip", False)

        if isinstance(self.layer_scalar, Tensor) and "layer_scalar" not in self._buffers:
            layer_scalar = self.layer_scalar.clone()
            if "layer_scalar" in self.__dict__:
                del self.__dict__["layer_scalar"]
            self.register_buffer("layer_scalar", layer_scalar)

        if self.enable_moe_block:
            router = self.router
            experts = self.experts

            self.moe_router_norm = router.norm
            moe_gate = nn.Linear(
                router.hidden_size,
                router.proj.out_features,
                bias=False,
                device=router.proj.weight.device,
                dtype=router.proj.weight.dtype,
            )
            moe_gate.weight.data.copy_(router.proj.weight.data)
            self.moe_gate = moe_gate

            self.register_buffer("_moe_scale", router.scale.data.clone())
            self.register_buffer("_moe_per_expert_scale", router.per_expert_scale.data.clone())
            self._moe_scalar_root_size = router.scalar_root_size
            self._moe_top_k = router.config.top_k_experts

            intermediate_dim = experts.intermediate_dim
            gate_up = experts.gate_up_proj.data
            gate_proj_weight = gate_up[:, :intermediate_dim, :].contiguous()
            up_proj_weight = gate_up[:, intermediate_dim:, :].contiguous()
            down_proj_weight = experts.down_proj.data.contiguous()

            self.moe_block = MoeBlock("gelu", self._moe_top_k, normalize_routing_weights=True)
            self.moe_block.expert_gate_proj_weight = nn.Parameter(gate_proj_weight)
            self.moe_block.expert_gate_proj_bias = None
            self.moe_block.expert_up_proj_weight = nn.Parameter(up_proj_weight)
            self.moe_block.expert_up_proj_bias = None
            self.moe_block.expert_down_proj_weight = nn.Parameter(down_proj_weight)
            self.moe_block.expert_down_proj_bias = None

            del self.router
            del self.experts
            torch.cuda.empty_cache()

    def forward(
        self,
        hidden_states: Tensor,
        past_seq_length: Optional[Tensor] = None,
        current_input_length: Optional[Tensor] = None,
        local_attention_mask: Optional[Tensor] = None,
        global_attention_mask: Optional[Tensor] = None,
        past_k_cache: Optional[Tensor] = None,
        past_v_cache: Optional[Tensor] = None,
        position_embeddings_global: Optional[Tuple[Tensor, Tensor]] = None,
        position_embeddings_local: Optional[Tuple[Tensor, Tensor]] = None,
        **kwargs,
    ) -> Tuple[Tensor]:
        clip_value = torch.finfo(torch.float16).max - 1000
        layer_scalar = self.layer_scalar
        if isinstance(layer_scalar, Tensor):
            layer_scalar = layer_scalar.to(hidden_states)

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        position_embeddings = position_embeddings_local if self.self_attn.is_sliding else position_embeddings_global

        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            past_seq_length=past_seq_length,
            current_input_length=current_input_length,
            local_attention_mask=local_attention_mask,
            global_attention_mask=global_attention_mask,
            past_k_cache=past_k_cache,
            past_v_cache=past_v_cache,
            position_embeddings=position_embeddings,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        if self.use_clip:
            hidden_states = torch.clamp(hidden_states, min=-clip_value, max=clip_value)
        hidden_states = residual + hidden_states
        if self.use_clip:
            hidden_states = torch.clamp(hidden_states, min=-clip_value, max=clip_value)

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        if self.use_clip:
            hidden_states = torch.clamp(hidden_states, min=-clip_value, max=clip_value)
        hidden_states = self.mlp(hidden_states)
        if self.use_clip:
            hidden_states = torch.clamp(hidden_states, min=-clip_value, max=clip_value)

        if self.enable_moe_block:
            bsz, seq_len, hidden_dim = residual.shape

            hidden_flat = residual.reshape(-1, hidden_dim)
            normed = self.moe_router_norm(hidden_flat)
            moe_scale = self._moe_scale.to(normed)
            scaled = normed * moe_scale * self._moe_scalar_root_size
            expert_scores = self.moe_gate(scaled)
            routing_probs = F.softmax(expert_scores, dim=-1)
            moe_per_expert_scale = self._moe_per_expert_scale.to(routing_probs)
            routing_probs = routing_probs * moe_per_expert_scale
            routing_weights = routing_probs.view(bsz, seq_len, -1)

            expert_input = self.pre_feedforward_layernorm_2(residual)

            moe_output = self.moe_block(expert_input, routing_weights, fast_mode=False)
            moe_output = moe_output.to(residual.dtype)

            hidden_states_2 = self.post_feedforward_layernorm_2(moe_output)
            hidden_states_1 = self.post_feedforward_layernorm_1(hidden_states)
            hidden_states = hidden_states_1 + hidden_states_2

        hidden_states = self.post_feedforward_layernorm(hidden_states)
        if self.use_clip:
            hidden_states = torch.clamp(hidden_states, min=-clip_value, max=clip_value)
        hidden_states = residual + hidden_states
        if self.use_clip:
            hidden_states = torch.clamp(hidden_states, min=-clip_value, max=clip_value)

        hidden_states = hidden_states * layer_scalar

        return (hidden_states,)


@XHLLM_TRACEABLE_MODULES.register_module({Gemma4TextModel: "Gemma4TextModel"})
class _Gemma4TextModel(DynamicModule):
    def _setup_position_cached(self):
        rotary_emb = self.rotary_emb
        max_seq_len = rotary_emb.max_seq_len_cached
        inv_freq = rotary_emb.full_attention_inv_freq
        device = inv_freq.device
        position_ids = torch.arange(0, max_seq_len, dtype=torch.long, device=device).unsqueeze(0)
        cos, sin = rotary_emb.forward(inv_freq, position_ids, layer_type="full_attention")
        sin = sin.squeeze(0).unsqueeze(0).unsqueeze(0)
        cos = cos.squeeze(0).unsqueeze(0).unsqueeze(0)
        self.register_buffer("sin_cached", sin.to(dtype=inv_freq.dtype), persistent=False)
        self.register_buffer("cos_cached", cos.to(dtype=inv_freq.dtype), persistent=False)

    def _setup_position_local_cached(self):
        rotary_emb = self.rotary_emb
        max_seq_len = rotary_emb.max_seq_len_cached
        inv_freq = rotary_emb.sliding_attention_inv_freq
        device = inv_freq.device
        position_ids = torch.arange(0, max_seq_len, dtype=torch.long, device=device).unsqueeze(0)
        cos, sin = rotary_emb.forward(inv_freq, position_ids, layer_type="sliding_attention")
        sin = sin.squeeze(0).unsqueeze(0).unsqueeze(0)
        cos = cos.squeeze(0).unsqueeze(0).unsqueeze(0)
        self.register_buffer("sin_cached_local", sin.to(dtype=inv_freq.dtype), persistent=False)
        self.register_buffer("cos_cached_local", cos.to(dtype=inv_freq.dtype), persistent=False)

    def _setup(self, cfg: Optional[Dict]):
        self.only_first_block = cfg.get("only_first_block", False)
        self._setup_position_cached()
        self._setup_position_local_cached()

        self.num_logits_to_keep = cfg.num_logits_to_keep
        assert self.num_logits_to_keep in [0, 1]
        self.output_hidden_states_for_export = bool(cfg.get("output_hidden_states_for_export", False))
        self.output_hidden_sequence = bool(cfg.get("output_hidden_sequence", False))

        self.use_cache = cfg.use_cache
        input_seq_len = cfg.input_sequence_length
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
            input_seq_len = cfg.input_sequence_length
            batch_size = cfg.get("batch_size", 1)
            self.update_offset_indices(batch_size, input_seq_len)

        self.llm_gather._update_cfg = types.MethodType(_llm_gather_update_cfg, self.llm_gather)

    def forward(
        self,
        inputs_embeds: Optional[Tensor] = None,
        past_seq_length: Optional[Tensor] = None,
        current_input_length: Optional[Tensor] = None,
        local_attention_mask: Optional[Tensor] = None,
        global_attention_mask: Optional[Tensor] = None,
        past_key_cache: Optional[List[Tensor]] = None,
        past_value_cache: Optional[List[Tensor]] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        hidden_states = inputs_embeds

        cos = self.cos_slice(self.cos_cached, past_seq_length)
        sin = self.sin_slice(self.sin_cached, past_seq_length)
        position_embeddings = (cos, sin)

        cos_local = self.cos_slice(self.cos_cached_local, past_seq_length)
        sin_local = self.sin_slice(self.sin_cached_local, past_seq_length)
        position_embeddings_local = (cos_local, sin_local)

        for idx, decoder_layer in enumerate(self.layers):
            if self.use_cache:
                past_k = past_key_cache[idx]
                past_v = past_value_cache[idx]
            else:
                past_k = None
                past_v = None

            layer_outputs = decoder_layer(
                hidden_states,
                past_seq_length=past_seq_length,
                current_input_length=current_input_length,
                local_attention_mask=local_attention_mask,
                global_attention_mask=global_attention_mask,
                past_k_cache=past_k,
                past_v_cache=past_v,
                position_embeddings_global=position_embeddings,
                position_embeddings_local=position_embeddings_local,
            )
            hidden_states = layer_outputs[0]
            if self.only_first_block:
                break

        normalized_sequence = None
        if self.num_logits_to_keep == 0 or self.output_hidden_sequence:
            normalized_sequence = self.norm(hidden_states)

        if self.num_logits_to_keep != 0:
            hidden_states = self.llm_gather(hidden_states, current_input_length - 1)
            hidden_states = self.norm(hidden_states)
        else:
            hidden_states = normalized_sequence

        if self.output_hidden_states_for_export:
            hidden_output = normalized_sequence if self.output_hidden_sequence else hidden_states
            return hidden_states, hidden_output

        return BaseModelOutputWithPast(last_hidden_state=hidden_states)


@XHLLM_TRACEABLE_MODULES.register_module({Gemma4ForCausalLM: "Gemma4ForCausalLM"})
class _Gemma4ForCausalLM(DynamicModule):
    def _setup(self, cfg: Optional[Dict] = None):
        pass

    def forward(
        self,
        inputs_embeds: Optional[Tensor] = None,
        past_seq_length: Optional[Tensor] = None,
        current_input_length: Optional[Tensor] = None,
        local_attention_mask: Optional[Tensor] = None,
        global_attention_mask: Optional[Tensor] = None,
        past_key_cache: Optional[List[Tensor]] = None,
        past_value_cache: Optional[List[Tensor]] = None,
    ) -> Tensor:
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            past_seq_length=past_seq_length,
            current_input_length=current_input_length,
            local_attention_mask=local_attention_mask,
            global_attention_mask=global_attention_mask,
            past_key_cache=past_key_cache,
            past_value_cache=past_value_cache,
        )

        extra_outputs = []
        if isinstance(outputs, (tuple, list)):
            hidden_states = outputs[0]
            extra_outputs = list(outputs[1:])
        else:
            hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        if self.config.final_logit_softcapping is not None:
            logits = logits / self.config.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * self.config.final_logit_softcapping

        if extra_outputs:
            return (logits, *extra_outputs)
        return logits


def register_wrap_cls(hf_model=None):
    return None