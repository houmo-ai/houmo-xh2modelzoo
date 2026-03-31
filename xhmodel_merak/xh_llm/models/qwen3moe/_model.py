import math
import sys
import types
from copy import deepcopy
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeAttention,
    Qwen3MoeDecoderLayer,
    Qwen3MoeForCausalLM,
    Qwen3MoeModel,
    Qwen3MoeRMSNorm,
    Qwen3MoeRotaryEmbedding,
    Qwen3MoeSparseMoeBlock,
)

from xhquant import nn as xhnn
from xhquant.api import ConfigDict
from xhquant.nn import LLMCache, MaskedSoftmax, RMSNorm
from xhquant.nn.modules.moeblock import MoeBlock
from xhquant.utils.registry import DynamicModule

from ...register import XHLLM_TRACEABLE_MODULES


@XHLLM_TRACEABLE_MODULES.register_module(
    {
        Qwen3MoeRotaryEmbedding: "Qwen3MoeRotaryEmbedding",
    }
)
class _Qwen3MoeRotaryEmbedding(DynamicModule):
    def _setup(self, cfg: dict | None = None):
        assert "dynamic" not in self.rope_type, f"{self.rope_type} is not supported in dynamic mode"
        max_seq_len = cfg.context_max_length
        max_seq_len = max(max_seq_len, self.config.max_position_embeddings)

        self.config.max_position_embeddings = max_seq_len
        self.max_seq_len_cached = max_seq_len
        self.original_max_seq_len = max_seq_len

        self._setup_cos_sin_cache(seq_len=self.max_seq_len_cached, dtype=self.inv_freq.dtype)
        if hasattr(self, "setup_after_callback"):
            self.setup_after_callback()

    def _setup_cos_sin_cache(self, seq_len, dtype):
        inv_freq = self.inv_freq.to(torch.float32)
        device = self.inv_freq.device
        if torch.cuda.is_available() and inv_freq.device.type != "cuda":
            inv_freq = self.inv_freq.cuda()

        position_ids = torch.arange(0, seq_len, dtype=torch.long, device=inv_freq.device).unsqueeze(0)
        cos, sin = self.forward(inv_freq, position_ids)
        cos = cos.to(device).squeeze(0)
        sin = sin.to(device).squeeze(0)

        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

        sin = sin.contiguous().to(dtype=dtype)
        cos = cos.contiguous().to(dtype=dtype)
        self.register_buffer("sin_cached", sin, persistent=False)
        self.register_buffer("cos_cached", cos, persistent=False)

    def _set_dtype(self, dtype: torch.dtype) -> None:
        self.inv_freq = self.inv_freq.to(dtype)
        self._setup_cos_sin_cache(seq_len=self.max_seq_len_cached, dtype=self.inv_freq.dtype)

    @torch.no_grad()
    def forward(self, x, position_ids):
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)

        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        device_ = x.device
        inv_freq_expanded = inv_freq_expanded.to(device_)
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


@XHLLM_TRACEABLE_MODULES.register_module(
    {
        Qwen3MoeAttention: "Qwen3MoeAttention",
    }
)
class _Qwen3MoeAttention(DynamicModule):
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

    def graph_forward(
        self,
        hidden_states: torch.Tensor,
        past_seq_length: Tensor | None = None,
        current_input_length: Tensor | None = None,
        past_k_cache: Tensor | None = None,
        past_v_cache: Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]:
        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = self.q_norm(query_states.view(bsz, q_len, self.num_heads, self.head_dim)).transpose(1, 2)
        key_states = self.k_norm(key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        assert position_embeddings is not None
        cos, sin = position_embeddings
        query_states, key_states = self.apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if self.use_cache:
            key_states = self.k_cache(key_states, past_seq_length, current_input_length, past_k_cache)
            value_states = self.v_cache(value_states, past_seq_length, current_input_length, past_v_cache)

        query_states = query_states * self.kv_scale
        key_states = key_states.transpose(2, 3)
        key_states = torch.repeat_interleave(key_states, self.num_key_value_groups, dim=1)

        attn_weights = torch.matmul(query_states, key_states)
        attn_weights: Optional[Tensor] = self.masked_softmax(attn_weights, past_seq_length)

        value_states = torch.repeat_interleave(value_states, self.num_key_value_groups, dim=1)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.config.num_attention_heads * self.head_dim)
        attn_output = self.o_proj(attn_output)

        return attn_output, None, None

    def _setup(self, cfg: ConfigDict | dict[str, Any]):
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

        self.masked_softmax = MaskedSoftmax(dim=-1)
        self.use_cache = cfg.use_cache
        self.key_extra_scale = 1.0 if "key_extra_scale" not in cfg else cfg.key_extra_scale
        self.query_extra_scale = 1.0 if "query_extra_scale" not in cfg else cfg.query_extra_scale

        if self.use_cache:
            cache_axis = cfg.kv_cache.cache_axis
            self.k_cache = LLMCache(axis=cache_axis, attention_max_length=-1)
            self.v_cache = LLMCache(axis=cache_axis, attention_max_length=-1)
        else:
            self.k_cache = None
            self.v_cache = None
        self.kv_scale = 1 / math.sqrt(self.head_dim)
        return self


@XHLLM_TRACEABLE_MODULES.register_module(
    {
        Qwen3MoeDecoderLayer: "Qwen3MoeDecoderLayer",
    }
)
class _Qwen3MoeDecoderLayer(DynamicModule):
    def graph_forward(
        self,
        hidden_states: torch.Tensor,
        past_seq_length: Tensor | None = None,
        current_input_length: Tensor | None = None,
        past_k_cache: Tensor | None = None,
        past_v_cache: Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> tuple[torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor] | None]:
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
        if isinstance(hidden_states, tuple):
            hidden_states, _ = hidden_states
        hidden_states = residual + hidden_states
        return (hidden_states,)

    def _setup(self, cfg: dict | None = None):
        return self


@XHLLM_TRACEABLE_MODULES.register_module(
    {
        Qwen3MoeSparseMoeBlock: "Qwen3MoeSparseMoeBlock",
    }
)
class _Qwen3MoeSparseMoeBlock(DynamicModule):
    def forward(self, hidden_states):
        ori_hidden_states = hidden_states
        router_logits = self.gate(hidden_states)
        routing_weights = F.softmax(router_logits, dim=-1)
        return self.moeblock(ori_hidden_states, routing_weights)

    def _setup(self, cfg: dict | None = None):
        self.device = self.gate.weight.device
        self.moeblock = MoeBlock(self.experts[0].act_fn._get_name().lower(), self.top_k, self.norm_topk_prob)

        self.moeblock.expert_gate_proj_weight = torch.nn.Parameter(
            torch.zeros(
                self.num_experts,
                self.experts[0].gate_proj.weight.shape[0],
                self.experts[0].gate_proj.weight.shape[1],
                device=self.device,
                dtype=self.experts[0].gate_proj.weight.dtype,
            )
        )
        if hasattr(self.experts[0].gate_proj, "quant_weight") and self.experts[0].gate_proj.quant_weight is not None:
            self.moeblock.expert_gate_proj_quant_weight = torch.tensor(
                torch.zeros(
                    self.num_experts,
                    self.experts[0].gate_proj.quant_weight.shape[0],
                    self.experts[0].gate_proj.quant_weight.shape[1],
                    device=self.device,
                    dtype=self.experts[0].gate_proj.quant_weight.dtype,
                )
            )

        if self.experts[0].gate_proj.bias is not None:
            self.moeblock.expert_gate_proj_bias = torch.nn.Parameter(
                torch.zeros(self.num_experts, self.experts[0].gate_proj.bias.shape[0], device=self.device)
            )
        else:
            self.moeblock.expert_gate_proj_bias = None

        self.moeblock.expert_up_proj_weight = torch.nn.Parameter(
            torch.zeros(
                self.num_experts,
                self.experts[0].up_proj.weight.shape[0],
                self.experts[0].up_proj.weight.shape[1],
                device=self.device,
                dtype=self.experts[0].up_proj.weight.dtype,
            )
        )
        self.moeblock.expert_down_proj_weight = torch.nn.Parameter(
            torch.zeros(
                self.num_experts,
                self.experts[0].down_proj.weight.shape[0],
                self.experts[0].down_proj.weight.shape[1],
                device=self.device,
                dtype=self.experts[0].down_proj.weight.dtype,
            )
        )

        with torch.no_grad():
            self.moeblock.expert_gate_proj_weight.copy_(
                torch.cat([expert.gate_proj.weight.data.to(self.device).unsqueeze(0) for expert in self.experts], dim=0)
            )
            self.moeblock.expert_up_proj_weight.copy_(
                torch.cat([expert.up_proj.weight.data.to(self.device).unsqueeze(0) for expert in self.experts], dim=0)
            )
            self.moeblock.expert_down_proj_weight.copy_(
                torch.cat([expert.down_proj.weight.data.to(self.device).unsqueeze(0) for expert in self.experts], dim=0)
            )
            if self.moeblock.expert_gate_proj_bias is not None:
                self.moeblock.expert_gate_proj_bias.copy_(
                    torch.cat(
                        [expert.gate_proj.bias.data.to(self.device).unsqueeze(0) for expert in self.experts], dim=0
                    )
                )

        del self.experts
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return self


@XHLLM_TRACEABLE_MODULES.register_module(
    {
        Qwen3MoeRMSNorm: "Qwen3MoeRMSNorm",
    }
)
class _Qwen3MoeRMSNorm(DynamicModule):
    def forward(self, hidden_states):
        return self.norm(hidden_states)

    def _setup(self, cfg: dict | None = None):
        hidden_size = self.weight.shape[0]
        self.norm = RMSNorm(hidden_size, eps=self.variance_epsilon)
        self.norm.weight = nn.Parameter(deepcopy(self.weight.data))
        return self


@XHLLM_TRACEABLE_MODULES.register_module(
    {
        Qwen3MoeModel: "Qwen3MoeModel",
    }
)
class _Qwen3MoeModel(DynamicModule):
    def graph_forward(
        self,
        inputs_embeds: torch.FloatTensor | None = None,
        past_seq_length: Tensor | None = None,
        current_input_length: Tensor | None = None,
        past_key_cache: list[Tensor] | None = None,
        past_value_cache: list[Tensor] | None = None,
    ):
        hidden_states = inputs_embeds

        cos = self.cos_slice(self.rotary_emb.cos_cached, past_seq_length)
        sin = self.sin_slice(self.rotary_emb.sin_cached, past_seq_length)
        position_embeddings = (cos, sin)

        for idx, decoder_layer in enumerate(self.layers):
            if self.use_cache:
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
            )
            hidden_states = layer_outputs[0]
            if self.max_layers > 0 and idx + 1 >= self.max_layers:
                break

        if self.num_logits_to_keep != 0:
            hidden_states = self.llm_gather(hidden_states, current_input_length - 1)
        hidden_states = self.norm(hidden_states)
        return hidden_states

    def _setup_cos_sin_embeding(self):
        self.rotary_emb.cos_cached
        self.rotary_emb.sin_cached

    def _setup(self, cfg: dict | None = None):
        self.only_first_block = cfg.get("only_first_block", False)
        self.max_layers = 1 if self.only_first_block else -1
        if "max_layers" in cfg:
            self.max_layers = cfg.max_layers

        self.num_logits_to_keep = cfg.num_logits_to_keep
        assert self.num_logits_to_keep in [0, 1]

        input_sequence_length = cfg.input_sequence_length
        self.slice = xhnn.Slice([0], [input_sequence_length], [1], [1])
        self.llm_gather = xhnn.BatchGather(1)
        self.llm_gather.update_offset_indices(1, input_sequence_length)

        def _llm_gather_update_cfg(self: xhnn.BatchGather, cfg: Optional[dict] = None):
            input_seq_len = cfg.input_sequence_length
            batch_size = cfg.get("batch_size", 1)
            self.update_offset_indices(batch_size, input_seq_len)

        self.llm_gather._update_cfg = types.MethodType(_llm_gather_update_cfg, self.llm_gather)

        def _slice_update_cfg(self, cfg: Optional[dict] = None):
            input_seq_len = cfg.input_sequence_length
            self.ends = [input_seq_len]

        self.slice._update_cfg = types.MethodType(_slice_update_cfg, self.slice)
        self.use_cache = cfg.use_cache

        self.sin_slice = xhnn.DynamicSlice([input_sequence_length], [2], [1])
        self.cos_slice = xhnn.DynamicSlice([input_sequence_length], [2], [1])

        def _sin_cos_slice_update_cfg(self, cfg: Optional[dict] = None):
            input_seq_len = cfg.input_sequence_length
            self.valid_length = [input_seq_len]

        self.sin_slice._update_cfg = types.MethodType(_sin_cos_slice_update_cfg, self.sin_slice)
        self.cos_slice._update_cfg = types.MethodType(_sin_cos_slice_update_cfg, self.cos_slice)

        if not hasattr(self.rotary_emb, "cos_cached"):
            self.rotary_emb.setup_after_callback = self._setup_cos_sin_embeding
        else:
            self._setup_cos_sin_embeding()
        return self


@XHLLM_TRACEABLE_MODULES.register_module(
    {
        Qwen3MoeForCausalLM: "Qwen3MoeForCausalLM",
    }
)
class _Qwen3MoeForCausalLM(DynamicModule):
    def graph_forward(
        self,
        inputs_embeds: Tensor | None = None,
        past_seq_length: Tensor | None = None,
        current_input_length: Tensor | None = None,
        past_key_cache: list[Tensor] | None = None,
        past_value_cache: list[Tensor] | None = None,
    ):
        hidden_states = self.model(
            inputs_embeds=inputs_embeds,
            past_seq_length=past_seq_length,
            current_input_length=current_input_length,
            past_key_cache=past_key_cache,
            past_value_cache=past_value_cache,
        )
        logits = self.lm_head(hidden_states)
        return logits

    def _setup(self, cfg: dict | None = None):
        assert not self.config.use_sliding_window, f"{self.__class__.__name__} does not support sliding window."
        return self


def register_wrap_modules(hf_model: Qwen3MoeForCausalLM | None = None):
    pass
