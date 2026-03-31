import math
import sys
import types
from copy import deepcopy
from typing import Any, Optional

import torch
import torch.nn as nn
from torch import Tensor

from xhquant import nn as xhnn
from xhquant.api import ConfigDict
from xhquant.nn import LLMCache, MaskedSoftmax, RMSNorm
from xhquant.nn.modules.moeblock import MoeBlock
from xhquant.utils.registry import DynamicModule

from ...register import XHLLM_TRACEABLE_MODULES
from .spark.modeling_ipt import (
    IPTAttention,
    IPTDecoderLayer,
    IPTForCausalLM,
    IPTMLAttention,
    IPTModel,
    IPTMoE,
    IPTRMSNorm,
    IPTRotaryEmbedding,
)


# ---------------------------------------------------------------------------
# IPTRMSNorm → xhquant RMSNorm
# ---------------------------------------------------------------------------
@XHLLM_TRACEABLE_MODULES.register_module({IPTRMSNorm: "IPTRMSNorm"})
class _IPTRMSNorm(DynamicModule):
    def forward(self, hidden_states):
        return self.norm(hidden_states)

    def _setup(self, cfg: dict | None = None):
        hidden_size = self.weight.shape[0]
        self.norm = RMSNorm(hidden_size, eps=self.variance_epsilon)
        self.norm.weight = nn.Parameter(deepcopy(self.weight.data))
        return self


# ---------------------------------------------------------------------------
# IPTRotaryEmbedding → precomputed cos/sin cache
# ---------------------------------------------------------------------------
@XHLLM_TRACEABLE_MODULES.register_module({IPTRotaryEmbedding: "IPTRotaryEmbedding"})
class _IPTRotaryEmbedding(DynamicModule):
    def _setup(self, cfg: dict | None = None):
        assert "dynamic" not in self.rope_type, f"{self.rope_type} is not supported"
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
        # shape: (1, 1, seq_len, rope_dim)
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
        device = x.device
        inv_freq_expanded = inv_freq_expanded.to(device)
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# ---------------------------------------------------------------------------
# IPTAttention (standard MHA with fused QKV projection)
# ---------------------------------------------------------------------------
@XHLLM_TRACEABLE_MODULES.register_module({IPTAttention: "IPTAttention"})
class _IPTAttention(DynamicModule):
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

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = self.apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if self.use_cache:
            key_states = self.k_cache(key_states, past_seq_length, current_input_length, past_k_cache)
            value_states = self.v_cache(value_states, past_seq_length, current_input_length, past_v_cache)

        query_states = query_states * self.kv_scale
        key_states_t = key_states.transpose(2, 3)
        key_states_t = torch.repeat_interleave(key_states_t, self.num_key_value_groups, dim=1)
        attn_weights = torch.matmul(query_states, key_states_t)
        attn_weights = self.masked_softmax(attn_weights, past_seq_length)
        value_states = torch.repeat_interleave(value_states, self.num_key_value_groups, dim=1)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)
        attn_output = self.out_proj(attn_output)
        return attn_output, None, None

    def _setup(self, cfg: ConfigDict | dict[str, Any]):
        if isinstance(cfg, dict):
            cfg = ConfigDict(cfg)

        self.enable_rope = cfg.get("enable_rope", True)
        if self.enable_rope:
            self.rope = xhnn.Rope()

        self.slice_1 = xhnn.Slice([0], [self.head_dim // 2], [3], [1])
        self.slice_2 = xhnn.Slice([self.head_dim // 2], [sys.maxsize], [3], [1])

        self.masked_softmax = MaskedSoftmax(dim=-1)
        self.use_cache = cfg.use_cache

        if self.use_cache:
            cache_axis = cfg.kv_cache.cache_axis
            self.k_cache = LLMCache(axis=cache_axis, attention_max_length=-1)
            self.v_cache = LLMCache(axis=cache_axis, attention_max_length=-1)

        # Split fused q_k_v_proj into separate projections
        q_size = self.num_heads * self.head_dim
        kv_size = self.num_key_value_heads * self.head_dim
        fused_weight = self.q_k_v_proj.weight.data
        fused_bias = self.q_k_v_proj.bias.data if self.q_k_v_proj.bias is not None else None

        self.q_proj = nn.Linear(self.hidden_size, q_size, bias=fused_bias is not None)
        self.k_proj = nn.Linear(self.hidden_size, kv_size, bias=fused_bias is not None)
        self.v_proj = nn.Linear(self.hidden_size, kv_size, bias=fused_bias is not None)

        with torch.no_grad():
            if self.num_key_value_heads != self.num_heads:
                self.q_proj.weight.copy_(fused_weight[:q_size, :])
                kv_weight = fused_weight[q_size:, :]
                self.k_proj.weight.copy_(kv_weight[:kv_size, :])
                self.v_proj.weight.copy_(kv_weight[kv_size:, :])
                if fused_bias is not None:
                    self.q_proj.bias.copy_(fused_bias[:q_size])
                    kv_bias = fused_bias[q_size:]
                    self.k_proj.bias.copy_(kv_bias[:kv_size])
                    self.v_proj.bias.copy_(kv_bias[kv_size:])
            else:
                chunk_size = q_size
                self.q_proj.weight.copy_(fused_weight[:chunk_size, :])
                self.k_proj.weight.copy_(fused_weight[chunk_size : 2 * chunk_size, :])
                self.v_proj.weight.copy_(fused_weight[2 * chunk_size :, :])
                if fused_bias is not None:
                    self.q_proj.bias.copy_(fused_bias[:chunk_size])
                    self.k_proj.bias.copy_(fused_bias[chunk_size : 2 * chunk_size])
                    self.v_proj.bias.copy_(fused_bias[2 * chunk_size :])

        del self.q_k_v_proj

        self.kv_scale = 1 / math.sqrt(self.head_dim)
        return self


# ---------------------------------------------------------------------------
# IPTMLAttention (Multi-head Latent Attention)
# ---------------------------------------------------------------------------
@XHLLM_TRACEABLE_MODULES.register_module({IPTMLAttention: "IPTMLAttention"})
class _IPTMLAttention(DynamicModule):
    def rotate_half(self, x: Tensor):
        x1 = self.slice_rope_1(x)
        x2 = self.slice_rope_2(x)
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(self, q_pe: Tensor, k_pe: Tensor, cos: Tensor, sin: Tensor):
        if self.enable_rope:
            q_embed = self.rope(q_pe, cos, sin)
            k_embed = self.rope(k_pe, cos, sin)
        else:
            q_embed = (q_pe * cos) + (self.rotate_half(q_pe) * sin)
            k_embed = (k_pe * cos) + (self.rotate_half(k_pe) * sin)
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
        batch_size, seq_length, _ = hidden_states.shape

        # Q path
        if self.apply_q_lora:
            q_states = self.q_down_layernorm(self.q_down_proj(hidden_states))
            q_states = self.q_up_proj(q_states)
        else:
            q_states = self.q_up_proj(hidden_states)

        q_states = q_states.view(batch_size, seq_length, self.num_heads, self.qk_head_dim).transpose(1, 2)
        q_nope = self.slice_q_nope(q_states)
        q_pe = self.slice_q_pe(q_states)

        # Compressed KV path
        compressed_kv = self.kv_down_proj_with_mqa(hidden_states)
        kv_a = self.slice_kv_a(compressed_kv)
        k_pe = self.slice_k_pe(compressed_kv)

        kv_a = self.kv_down_layernorm(kv_a)
        kv_full = self.kv_up_proj(kv_a)
        kv_full = kv_full.view(
            batch_size, seq_length, self.num_heads, self.qk_nope_head_dim + self.v_head_dim
        ).transpose(1, 2)
        k_nope = self.slice_k_nope(kv_full)
        value_states = self.slice_v(kv_full)

        k_pe = k_pe.view(batch_size, 1, seq_length, self.qk_rope_head_dim)

        # RoPE on rope-part only (in float32 per original implementation)
        cos, sin = position_embeddings
        origin_dtype = q_pe.dtype
        q_pe = q_pe.float()
        k_pe = k_pe.float()
        q_pe, k_pe = self.apply_rotary_pos_emb(q_pe, k_pe, cos, sin)
        q_pe = q_pe.to(origin_dtype)
        k_pe = k_pe.to(origin_dtype)

        k_pe = k_pe.expand(-1, self.num_heads, -1, -1)

        # Assemble full query / key
        query_states = torch.cat((q_nope, q_pe), dim=-1)
        key_states = torch.cat((k_nope, k_pe), dim=-1)

        # KV cache
        if self.use_cache:
            key_states = self.k_cache(key_states, past_seq_length, current_input_length, past_k_cache)
            value_states = self.v_cache(value_states, past_seq_length, current_input_length, past_v_cache)

        # Attention: Q @ K^T * scale → softmax → @ V
        query_states = query_states * self._scaling
        key_states_t = key_states.transpose(2, 3)
        attn_weights = torch.matmul(query_states, key_states_t)
        attn_weights = self.masked_softmax(attn_weights, past_seq_length)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(batch_size, seq_length, self.num_heads * self.v_head_dim)
        attn_output = self.o_proj(attn_output)
        return attn_output, None, None

    def _setup(self, cfg: ConfigDict | dict[str, Any]):
        if isinstance(cfg, dict):
            cfg = ConfigDict(cfg)

        self.enable_rope = cfg.get("enable_rope", True)
        if self.enable_rope:
            self.rope = xhnn.Rope()

        # Slice ops for rotate_half on the rope dimension
        self.slice_rope_1 = xhnn.Slice([0], [self.qk_rope_head_dim // 2], [3], [1])
        self.slice_rope_2 = xhnn.Slice([self.qk_rope_head_dim // 2], [sys.maxsize], [3], [1])

        # Slices for q → nope / pe split
        self.slice_q_nope = xhnn.Slice([0], [self.qk_nope_head_dim], [3], [1])
        self.slice_q_pe = xhnn.Slice([self.qk_nope_head_dim], [sys.maxsize], [3], [1])

        # Slices for compressed_kv → kv_a / k_pe split
        self.slice_kv_a = xhnn.Slice([0], [self.kv_lora_rank], [2], [1])
        self.slice_k_pe = xhnn.Slice([self.kv_lora_rank], [sys.maxsize], [2], [1])

        # Slices for decompressed kv → k_nope / value split
        self.slice_k_nope = xhnn.Slice([0], [self.qk_nope_head_dim], [3], [1])
        self.slice_v = xhnn.Slice([self.qk_nope_head_dim], [sys.maxsize], [3], [1])

        self.masked_softmax = MaskedSoftmax(dim=-1)
        self.use_cache = cfg.use_cache

        if self.use_cache:
            cache_axis = cfg.kv_cache.cache_axis
            self.k_cache = LLMCache(axis=cache_axis, attention_max_length=-1)
            self.v_cache = LLMCache(axis=cache_axis, attention_max_length=-1)

        self._scaling = self.scaling

        # Alias for compatibility with text_llm_model cache shape detection
        self.head_dim = self.qk_head_dim
        return self


# ---------------------------------------------------------------------------
# IPTMoE → MoeBlock with pre-computed grouped sigmoid routing
# ---------------------------------------------------------------------------
@XHLLM_TRACEABLE_MODULES.register_module({IPTMoE: "IPTMoE"})
class _IPTMoE(DynamicModule):
    def forward(self, hidden_states, routing_map=None):
        batch_size, seq_length, hidden_dim = hidden_states.shape

        # Shared expert (always active)
        shared_output = None
        if self.shared_experts is not None:
            shared_output = self.shared_experts(hidden_states)

        # Routing
        hidden_flat = hidden_states.view(-1, hidden_dim)
        routing_weights = self._compute_routing_weights(hidden_flat)

        # Expert dispatch via MoeBlock
        moe_output = self._moeblock(hidden_flat, routing_weights)
        if isinstance(moe_output, tuple):
            moe_output = moe_output[0]

        hidden_states = moe_output.view(batch_size, seq_length, hidden_dim)

        if shared_output is not None:
            hidden_states = hidden_states + shared_output

        return hidden_states

    def _compute_routing_weights(self, hidden_flat: Tensor) -> Tensor:
        """Compute dense routing weights using IPT's grouped sigmoid routing."""
        num_tokens = hidden_flat.shape[0]
        device = hidden_flat.device

        logits = self._router_gating(hidden_flat.float())
        logits = logits.view(num_tokens, self._num_groups, self._num_experts_per_group)
        scores = logits.sigmoid()

        # Top-k selection per group with score correction bias
        scores_for_choice = scores.view(num_tokens, -1) + self._e_score_correction_bias.to(device)
        scores_for_choice = scores_for_choice.view(num_tokens, self._num_groups, self._num_experts_per_group)
        _, topk_indices = torch.topk(scores_for_choice, k=self._top_k, dim=-1)

        # Gather actual sigmoid scores for selected experts
        topk_probs = scores.gather(-1, topk_indices)

        # Normalize probabilities
        if self._top_k > 1:
            if self._calc_denominator_cross_groups:
                denom = topk_probs.view(num_tokens, -1).sum(dim=-1, keepdim=True).unsqueeze(-1) + 1e-20
            else:
                denom = topk_probs.sum(dim=-1, keepdim=True) + 1e-20
            topk_probs = topk_probs / denom
        topk_probs = topk_probs * self._routed_scaling_factor

        # Flatten indices with group offset and create dense routing weights
        head_incre = self._head_incre.to(device)
        flat_indices = (topk_indices + head_incre.view(1, -1, 1)).view(num_tokens, -1)
        flat_probs = topk_probs.view(num_tokens, -1)

        total_experts = self._num_groups * self._num_experts_per_group
        routing_weights = torch.zeros(num_tokens, total_experts, device=device, dtype=flat_probs.dtype)
        routing_weights.scatter_(1, flat_indices, flat_probs)

        return routing_weights

    def _setup(self, cfg: dict | None = None):
        router = self.router

        self._num_groups = router.num_groups
        self._num_experts_per_group = router.num_experts
        self._top_k = router.top_k
        self._calc_denominator_cross_groups = router.calc_denominator_cross_groups
        self._routed_scaling_factor = router.routed_scaling_factor
        self._e_score_correction_bias = router.e_score_correction_bias
        self._router_gating = router.gating

        self._head_incre = torch.arange(self._num_groups, dtype=torch.long) * self._num_experts_per_group

        total_top_k = self._top_k * self._num_groups

        # Extract and stack expert weights
        if self.grouped_gemm:
            fc1_weights = self.routed_experts.fc1_weights.data
            fc2_weights = self.routed_experts.fc2_weights.data
        else:
            fc1_weights = torch.stack([exp.fc1.weight.data for exp in self.routed_experts])
            fc2_weights = torch.stack([exp.fc2.weight.data for exp in self.routed_experts])

        inter_size = fc1_weights.shape[1] // 2

        # Create MoeBlock: gelu(gate_proj(x)) * up_proj(x) → down_proj
        self._moeblock = MoeBlock("gelu", total_top_k, False)
        self._moeblock.expert_gate_proj_weight = nn.Parameter(fc1_weights[:, :inter_size, :].contiguous())
        self._moeblock.expert_up_proj_weight = nn.Parameter(fc1_weights[:, inter_size:, :].contiguous())
        self._moeblock.expert_down_proj_weight = nn.Parameter(fc2_weights.contiguous())
        self._moeblock.expert_gate_proj_bias = None

        # Clean up original modules
        del self.routed_experts
        del self.router
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return self


# ---------------------------------------------------------------------------
# IPTDecoderLayer
# ---------------------------------------------------------------------------
@XHLLM_TRACEABLE_MODULES.register_module({IPTDecoderLayer: "IPTDecoderLayer"})
class _IPTDecoderLayer(DynamicModule):
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
        orig_dtype = hidden_states.dtype
        residual = hidden_states
        hidden_states = self.layer_norm(hidden_states)

        # Self Attention
        hidden_states, _, _ = self.attention(
            hidden_states=hidden_states,
            past_seq_length=past_seq_length,
            current_input_length=current_input_length,
            past_k_cache=past_k_cache,
            past_v_cache=past_v_cache,
            position_embeddings=position_embeddings,
        )
        hidden_states = (residual.float() + hidden_states.float()).to(orig_dtype)

        # Feed-forward (dense MLP or MoE)
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = (residual.float() + hidden_states.float()).to(orig_dtype)

        return (hidden_states,)

    def _setup(self, cfg: dict | None = None):
        # Create alias expected by text_llm_model._wraped_post
        self.self_attn = self.attention
        return self


# ---------------------------------------------------------------------------
# IPTModel
# ---------------------------------------------------------------------------
@XHLLM_TRACEABLE_MODULES.register_module({IPTModel: "IPTModel"})
class _IPTModel(DynamicModule):
    def graph_forward(
        self,
        inputs_embeds: torch.FloatTensor | None = None,
        past_seq_length: Tensor | None = None,
        current_input_length: Tensor | None = None,
        past_key_cache: list[Tensor] | None = None,
        past_value_cache: list[Tensor] | None = None,
    ) -> Tensor:
        hidden_states = inputs_embeds

        cos = self.cos_slice(self.rotary_emb.cos_cached, past_seq_length)
        sin = self.sin_slice(self.rotary_emb.sin_cached, past_seq_length)
        position_embeddings = (cos, sin)

        for idx, decoder_layer in enumerate(self.transformer.layers):
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

        hidden_states = self.transformer.layernorm(hidden_states)
        return hidden_states

    def _setup_cos_sin_embedding(self):
        cos_cached = self.rotary_emb.cos_cached
        sin_cached = self.rotary_emb.sin_cached
        if cos_cached is None or sin_cached is None:
            from xhquant.utils import get_root_logger

            get_root_logger().warning("Cosine/sine caches are not set. Positional embeddings may be incorrect.")

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

        # Create aliases expected by text_llm_model._wraped_post
        self.layers = self.transformer.layers
        self.norm = self.transformer.layernorm

        if not hasattr(self.rotary_emb, "cos_cached"):
            self.rotary_emb.setup_after_callback = self._setup_cos_sin_embedding
        else:
            self._setup_cos_sin_embedding()

        return self


# ---------------------------------------------------------------------------
# IPTForCausalLM
# ---------------------------------------------------------------------------
@XHLLM_TRACEABLE_MODULES.register_module({IPTForCausalLM: "IPTForCausalLM"})
class _IPTForCausalLM(DynamicModule):
    def graph_forward(
        self,
        inputs_embeds: Tensor | None = None,
        past_seq_length: Tensor | None = None,
        current_input_length: Tensor | None = None,
        past_key_cache: list[Tensor] | None = None,
        past_value_cache: list[Tensor] | None = None,
    ) -> Tensor:
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
        return self


# ---------------------------------------------------------------------------
# Registration entry point (called by spark_moe_model.py)
# ---------------------------------------------------------------------------
def register_wrap_modules(hf_model=None):
    pass
