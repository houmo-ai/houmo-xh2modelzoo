from __future__ import annotations

import math
import sys
import types
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers.modeling_outputs import BaseModelOutputWithPast

from xhquant import nn as xhnn
from xhquant.api import ConfigDict
from xhquant.nn import LLMCacheV2, MaskedSoftmax, RMSNorm
from xhquant.utils.registry import DynamicModule

from ...register import XHLLM_TRACEABLE_MODULES
from .modeling_deepseekv2 import (
    DeepseekV2DecoderLayer,
    DeepseekV2ForCausalLM,
    DeepseekV2MLP,
    DeepseekV2MoE,
    DeepseekV2Model,
    DeepseekV2RMSNorm,
    MoEGate,
    SlidingWindowLlamaAttention,
)
from .modeling_unlimitedocr import UnlimitedOCRForCausalLM, UnlimitedOCRModel


@XHLLM_TRACEABLE_MODULES.register_module(
    {
        UnlimitedOCRForCausalLM: "UnlimitedOCRForCausalLM",
        DeepseekV2ForCausalLM: "DeepseekV2ForCausalLM",
    }
)
class _UnlimitedOCRForCausalLM(DynamicModule):
    def _setup(self, cfg: Optional[Dict] = None):
        return self

    def forward(
        self,
        inputs_embeds: Tensor,
        past_seq_length: Tensor,
        current_input_length: Tensor,
        past_key_cache: List[Tensor],
        past_value_cache: List[Tensor],
    ) -> Tensor:
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            past_seq_length=past_seq_length,
            current_input_length=current_input_length,
            past_key_cache=past_key_cache,
            past_value_cache=past_value_cache,
        )
        hidden_states = outputs.last_hidden_state
        return self.lm_head(hidden_states)


@XHLLM_TRACEABLE_MODULES.register_module(
    {
        UnlimitedOCRModel: "UnlimitedOCRModel",
        DeepseekV2Model: "DeepseekV2Model",
    }
)
class _UnlimitedOCRModel(DynamicModule):
    def _setup(self, cfg: ConfigDict | Dict[str, Any]):
        if isinstance(cfg, dict):
            cfg = ConfigDict(cfg)
        self.use_cache = bool(cfg.use_cache)
        self.only_first_block = bool(cfg.get("only_first_block", False))
        self.num_logits_to_keep = int(cfg.get("num_logits_to_keep", 1))
        assert self.num_logits_to_keep in (0, 1)
        max_seq_len = int(cfg.get("context_max_length", cfg.get("max_sequence_length", 2048)))
        input_seq_len = int(cfg.get("input_sequence_length", cfg.get("prefill_chunk_length", max_seq_len)))
        self._precompute_rope_cache(max_seq_len)
        self._setup_rope_slices(input_seq_len)
        self.slice = xhnn.Slice([0], [input_seq_len], [1], [1])
        self.llm_gather = xhnn.BatchGather(1)
        self.llm_gather.update_offset_indices(int(cfg.get("batch_size", 1)), input_seq_len)

        def _slice_update_cfg(slice_module, new_cfg=None):
            slice_module.ends = [new_cfg.input_sequence_length]

        def _gather_update_cfg(gather_module, new_cfg=None):
            gather_module.update_offset_indices(new_cfg.get("batch_size", 1), new_cfg.input_sequence_length)

        self.slice._update_cfg = types.MethodType(_slice_update_cfg, self.slice)
        self.llm_gather._update_cfg = types.MethodType(_gather_update_cfg, self.llm_gather)
        return self

    def _precompute_rope_cache(self, max_seq_len: int):
        rotary_emb = self.layers[0].self_attn.rotary_emb
        inv_freq = rotary_emb.inv_freq
        positions = torch.arange(max_seq_len, device=inv_freq.device, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq.float())
        emb = torch.cat((freqs, freqs), dim=-1)
        scaling = getattr(rotary_emb, "attention_scaling", 1.0)
        cos = (emb.cos() * scaling).to(inv_freq.dtype).unsqueeze(0).unsqueeze(2)
        sin = (emb.sin() * scaling).to(inv_freq.dtype).unsqueeze(0).unsqueeze(2)
        self.register_buffer("_cos_cached", cos, persistent=False)
        self.register_buffer("_sin_cached", sin, persistent=False)

    def _setup_rope_slices(self, input_seq_len: int):
        self.cos_slice = xhnn.DynamicSlice([input_seq_len], [1], [1])
        self.sin_slice = xhnn.DynamicSlice([input_seq_len], [1], [1])

        def _update_cfg(slice_module, cfg=None):
            slice_module.valid_length = [cfg.input_sequence_length]

        self.cos_slice._update_cfg = types.MethodType(_update_cfg, self.cos_slice)
        self.sin_slice._update_cfg = types.MethodType(_update_cfg, self.sin_slice)

    def _get_rope(self, past_seq_length: Tensor, dtype: torch.dtype):
        cos = self.cos_slice(self._cos_cached, past_seq_length).to(dtype)
        sin = self.sin_slice(self._sin_cached, past_seq_length).to(dtype)
        return cos, sin

    def forward(
        self,
        inputs_embeds: Tensor,
        past_seq_length: Tensor,
        current_input_length: Tensor,
        past_key_cache: Optional[List[Tensor]] = None,
        past_value_cache: Optional[List[Tensor]] = None,
    ) -> BaseModelOutputWithPast:
        hidden_states = inputs_embeds
        position_embeddings = self._get_rope(past_seq_length, hidden_states.dtype)

        for idx, decoder_layer in enumerate(self.layers):
            past_k = past_key_cache[idx] if past_key_cache is not None else None
            past_v = past_value_cache[idx] if past_value_cache is not None else None
            layer_outputs = decoder_layer(
                hidden_states,
                past_seq_length=past_seq_length,
                current_input_length=current_input_length,
                past_k_cache=past_k,
                past_v_cache=past_v,
                position_embeddings=position_embeddings,
            )
            hidden_states = layer_outputs[0]
            if self.only_first_block:
                break

        if self.num_logits_to_keep == 0:
            hidden_states = self.slice(hidden_states)
        else:
            hidden_states = self.llm_gather(hidden_states, current_input_length - 1)
        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(last_hidden_state=hidden_states)


@XHLLM_TRACEABLE_MODULES.register_module({DeepseekV2RMSNorm: "DeepseekV2RMSNorm"})
class _DeepseekV2RMSNorm(DynamicModule):
    def forward(self, hidden_states):
        return self.norm(hidden_states)

    def _setup(self, cfg: Optional[Dict] = None):
        hidden_size = self.weight.shape[0]
        self.norm = RMSNorm(hidden_size, self.variance_epsilon).to(self.weight.device)
        self.norm.weight = nn.Parameter(deepcopy(self.weight.data))
        return self


@XHLLM_TRACEABLE_MODULES.register_module({SlidingWindowLlamaAttention: "SlidingWindowLlamaAttention"})
class _SlidingWindowLlamaAttention(DynamicModule):
    def _setup(self, cfg: ConfigDict | Dict[str, Any]):
        if isinstance(cfg, dict):
            cfg = ConfigDict(cfg)
        self.use_cache = bool(cfg.use_cache)
        self.num_heads = int(self.config.num_attention_heads)
        self.num_key_value_heads = int(self.config.num_key_value_heads)
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.hidden_size = int(self.config.hidden_size)
        self.head_dim = int(self.head_dim)
        self.slice_1 = xhnn.Slice([0], [self.head_dim // 2], [3], [1])
        self.slice_2 = xhnn.Slice([self.head_dim // 2], [sys.maxsize], [3], [1])
        # Native SlidingWindowLlamaAttention keeps the full prefill cache visible
        # and only rings decoded tokens. XH's single attention_max_length window
        # cannot express that protected-prefix semantics, so keep full KV here.
        self._sliding_window = int(cfg.get("sliding_window_size", getattr(self.config, "sliding_window", -1) or -1))
        self.masked_softmax = MaskedSoftmax(dim=-1, attention_max_length=-1)
        self.rope = xhnn.Rope()
        if self.use_cache:
            cache_axis = cfg.kv_cache.cache_axis
            self.k_cache = LLMCacheV2(axis=cache_axis, attention_max_length=-1)
            self.v_cache = LLMCacheV2(axis=cache_axis, attention_max_length=-1)
        else:
            self.k_cache = None
            self.v_cache = None
        self.register_buffer("kv_scale", torch.tensor(1 / math.sqrt(self.head_dim), dtype=torch.float16), persistent=False)

        # Register _update_cfg to keep prefill/decode export graphs on full KV.
        def _update_cfg(_module, new_cfg=None):
            if new_cfg is None:
                return
            _module.masked_softmax.attention_max_length = -1
            if _module.k_cache is not None:
                _module.k_cache.attention_max_length = -1
            if _module.v_cache is not None:
                _module.v_cache.attention_max_length = -1

        self._update_cfg = types.MethodType(_update_cfg, self)
        return self

    def rotate_half(self, x: Tensor):
        x1 = self.slice_1(x)
        x2 = self.slice_2(x)
        return torch.cat((-x2, x1), dim=-1)

    def _apply_rope(self, q: Tensor, k: Tensor, cos: Tensor, sin: Tensor):
        try:
            return self.rope(q, cos, sin), self.rope(k, cos, sin)
        except Exception:
            return (q * cos) + (self.rotate_half(q) * sin), (k * cos) + (self.rotate_half(k) * sin)

    def forward(
        self,
        hidden_states: Tensor,
        past_seq_length: Optional[Tensor] = None,
        current_input_length: Optional[Tensor] = None,
        past_k_cache: Optional[Tensor] = None,
        past_v_cache: Optional[Tensor] = None,
        position_embeddings: Optional[Tuple[Tensor, Tensor]] = None,
        attention_mask: Optional[Tensor] = None,
        **kwargs,
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tuple[Tensor, Tensor]]]:
        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim)

        if position_embeddings is not None:
            cos, sin = position_embeddings
            query_states, key_states = self._apply_rope(query_states, key_states, cos, sin)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        if self.use_cache:
            key_states = self.k_cache(key_states, past_seq_length, current_input_length, past_k_cache)
            value_states = self.v_cache(value_states, past_seq_length, current_input_length, past_v_cache)

        key_states = torch.repeat_interleave(key_states.transpose(2, 3), self.num_key_value_groups, dim=1)
        value_states = torch.repeat_interleave(value_states, self.num_key_value_groups, dim=1)
        attn_weights = torch.matmul(query_states * self.kv_scale.to(query_states.dtype), key_states)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = self.masked_softmax(attn_weights, past_seq_length)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        return attn_output, None, None



@XHLLM_TRACEABLE_MODULES.register_module({DeepseekV2DecoderLayer: "DeepseekV2DecoderLayer"})
class _DeepseekV2DecoderLayer(DynamicModule):
    def _setup(self, cfg: Optional[Dict] = None):
        return self

    def forward(
        self,
        hidden_states: Tensor,
        past_seq_length: Optional[Tensor] = None,
        current_input_length: Optional[Tensor] = None,
        past_k_cache: Optional[Tensor] = None,
        past_v_cache: Optional[Tensor] = None,
        position_embeddings: Optional[Tuple[Tensor, Tensor]] = None,
        **kwargs,
    ) -> Tuple[Tensor]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _, _ = self.self_attn(
            hidden_states=hidden_states,
            past_seq_length=past_seq_length,
            current_input_length=current_input_length,
            past_k_cache=past_k_cache,
            past_v_cache=past_v_cache,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return (hidden_states,)


@XHLLM_TRACEABLE_MODULES.register_module({DeepseekV2MLP: "DeepseekV2MLP"})
class _DeepseekV2MLP(DynamicModule):
    def _setup(self, cfg: Optional[Dict] = None):
        return self

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


@XHLLM_TRACEABLE_MODULES.register_module({MoEGate: "MoEGate"})
class _MoEGate(DynamicModule):
    def _setup(self, cfg: Optional[Dict] = None):
        self.proj = nn.Linear(self.weight.shape[-1], self.weight.shape[0], bias=False).to(self.weight.device)
        self.proj.weight = self.weight
        return self

    def forward(self, hidden_states: Tensor) -> Tensor:
        bsz, seq_len, h = hidden_states.shape
        logits = self.proj(hidden_states.view(-1, h))
        if self.scoring_func == "softmax":
            scores = F.softmax(logits, dim=-1).to(logits.dtype)
        elif self.scoring_func == "sigmoid":
            scores = torch.sigmoid(logits)
        else:
            raise NotImplementedError(f"Unsupported MoE scoring function: {self.scoring_func}")
        return scores * self.routed_scaling_factor


@XHLLM_TRACEABLE_MODULES.register_module({DeepseekV2MoE: "DeepseekV2MoE"})
class _DeepseekV2MoE(DynamicModule):
    def _setup(self, cfg: ConfigDict | Dict[str, Any]):
        if isinstance(cfg, dict):
            cfg = ConfigDict(cfg)
        self.device = self.gate.weight.device
        first_expert = self.experts[0]
        activation_type = first_expert.act_fn._get_name().lower()
        self.moeblock = xhnn.MoeBlock(activation_type, self.gate.top_k, self.gate.norm_topk_prob)

        def _stack_param(attr_name: str):
            weights = [getattr(expert, attr_name).weight.data.to(self.device).unsqueeze(0) for expert in self.experts]
            return nn.Parameter(torch.cat(weights, dim=0))

        self.moeblock.expert_gate_proj_weight = _stack_param("gate_proj")
        self.moeblock.expert_up_proj_weight = _stack_param("up_proj")
        self.moeblock.expert_down_proj_weight = _stack_param("down_proj")
        self.moeblock.expert_gate_proj_bias = None
        self.moeblock.expert_up_proj_bias = None
        self.moeblock.expert_down_proj_bias = None
        del self.experts
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return self

    def forward(self, hidden_states: Tensor) -> Tensor:
        identity = hidden_states
        routing_weights = self.gate(hidden_states)
        y = self.moeblock(hidden_states, routing_weights)
        if self.config.n_shared_experts is not None:
            y = y + self.shared_experts(identity)
        return y


def register_wrap_cls(hf_model=None):
    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

    from xhquant.nn.builder import FX_LEAF_MODULES
    from xhquant.quantization.xh2a.builder import register_none_quanted_module

    FX_LEAF_MODULES._module_dict["LlamaRotaryEmbedding"] = True
    register_none_quanted_module(LlamaRotaryEmbedding)
    return None
