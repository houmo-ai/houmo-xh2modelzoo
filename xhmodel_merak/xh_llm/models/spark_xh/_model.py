import math
import sys
import types
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Optional

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
    IPTMLP,
    IPTAttention,
    IPTDecoder,
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
if TYPE_CHECKING:

    class _IPTRMSNormBase(DynamicModule, IPTRMSNorm):  # type: ignore[misc]
        ...

else:
    _IPTRMSNormBase = DynamicModule


@XHLLM_TRACEABLE_MODULES.register_module({IPTRMSNorm: "IPTRMSNorm"})
class _IPTRMSNorm(_IPTRMSNormBase):
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
if TYPE_CHECKING:

    class _IPTRotaryEmbeddingBase(DynamicModule, IPTRotaryEmbedding):  # type: ignore[misc]
        ...

else:
    _IPTRotaryEmbeddingBase = DynamicModule


@XHLLM_TRACEABLE_MODULES.register_module({IPTRotaryEmbedding: "IPTRotaryEmbedding"})
class _IPTRotaryEmbedding(_IPTRotaryEmbeddingBase):
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
# IPTMLP (fused gate/up dense MLP)
# ---------------------------------------------------------------------------
if TYPE_CHECKING:

    class _IPTMLPBase(DynamicModule, IPTMLP):  # type: ignore[misc]
        ...

else:
    _IPTMLPBase = DynamicModule


@XHLLM_TRACEABLE_MODULES.register_module({IPTMLP: "IPTMLP"})
class _IPTMLP(_IPTMLPBase):
    def forward(self, x: Tensor):
        # if self.clamp_input_value > 0:
        #     x = torch.clamp_(x, -self.clamp_input_value, self.clamp_input_value)
        # intermediate_parallel = self.fc1(x)

        # intermediate_parallel1, intermediate_parallel2 = torch.chunk(intermediate_parallel, 2, dim=-1)
        # intermediate_parallel1 = intermediate_parallel1.squeeze(-1)
        # intermediate_parallel2 = intermediate_parallel2.squeeze(-1)
        # intermediate_parallel1 = self.act_fn(intermediate_parallel1)
        # intermediate_parallel = intermediate_parallel1 * intermediate_parallel2

        # # if self.clamp_input_value > 0:
        # #     intermediate_parallel = torch.clamp_(intermediate_parallel, -self.clamp_input_value, self.clamp_input_value)
        # output = self.fc2(intermediate_parallel)
        # return output
        if self.grouped_mlp:
            raise AssertionError(
                "Grouped MLP is not supported in the current implementation. Please set `grouped_mlp=False` in the configuration to disable the grouped MLP and use the unfused MLP implementation instead."
            )
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj

    def _setup(self, *args, **kwargs):
        return self


# ---------------------------------------------------------------------------
# IPTAttention (standard MHA with fused QKV projection)
# ---------------------------------------------------------------------------
if TYPE_CHECKING:

    class _IPTAttentionBase(DynamicModule, IPTAttention):  # type: ignore[misc]
        ...

else:
    _IPTAttentionBase = DynamicModule


@XHLLM_TRACEABLE_MODULES.register_module({IPTAttention: "IPTAttention"})
class _IPTAttention(_IPTAttentionBase):
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
if TYPE_CHECKING:

    class _IPTMLAttentionBase(DynamicModule, IPTMLAttention):  # type: ignore[misc]
        ...

else:
    _IPTMLAttentionBase = DynamicModule


@XHLLM_TRACEABLE_MODULES.register_module({IPTMLAttention: "IPTMLAttention"})
class _IPTMLAttention(_IPTMLAttentionBase):
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

        # Q path (absorbed: q_rope_proj for RoPE part, q_absorbed_proj for content in latent space)
        if self.apply_q_lora:
            inp = self.q_down_layernorm(self.q_down_proj(hidden_states))
        else:
            inp = hidden_states

        q_pe = self.q_rope_proj(inp).view(batch_size, seq_length, self.num_heads, self.qk_rope_head_dim)
        q_content = self.q_absorbed_proj(inp).view(batch_size, seq_length, self.num_heads, self.kv_lora_rank)

        # KV path (latent space — no kv_up_proj at runtime)
        k_latent = self.kv_down_layernorm(self.kv_a_proj_latent(hidden_states))
        k_rot = self.kv_a_proj_rope(hidden_states).view(batch_size, 1, seq_length, self.qk_rope_head_dim)

        q_pe = q_pe.transpose(1, 2)
        cos, sin = position_embeddings
        origin_dtype = q_pe.dtype
        q_pe = q_pe.float()
        k_rot = k_rot.float()
        q_pe, k_rot = self.apply_rotary_pos_emb(q_pe, k_rot, cos, sin)
        q_pe = q_pe.to(origin_dtype)
        k_rot = k_rot.to(origin_dtype)
        q_pe = q_pe.transpose(1, 2)

        # KV cache (extremely compact: 1 head, latent/rope dims only)
        if self.use_cache:
            k_rot = self.k_cache(k_rot, past_seq_length, current_input_length, past_k_cache)
            k_rot = k_rot.squeeze(1)

            k_latent = k_latent.unsqueeze(1)
            k_latent = self.v_cache(k_latent, past_seq_length, current_input_length, past_v_cache)
            k_latent = k_latent.squeeze(1)

        # Attention with broadcast MQA (head dim=1 broadcasts against num_heads)
        query_states = torch.cat((q_content, q_pe), dim=-1).transpose(1, 2) * self._scaling
        key_states = torch.cat((k_latent, k_rot), dim=-1).unsqueeze(1).transpose(2, 3)
        attn_weights = torch.matmul(query_states, key_states)
        attn_weights = self.masked_softmax(attn_weights, past_seq_length)

        value_states = k_latent.unsqueeze(1)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
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

        self.masked_softmax = MaskedSoftmax(dim=-1)
        self.use_cache = cfg.use_cache

        if self.use_cache:
            cache_axis = cfg.kv_cache.cache_axis
            self.k_cache = LLMCache(axis=cache_axis, attention_max_length=-1)
            self.v_cache = LLMCache(axis=cache_axis, attention_max_length=-1)

        self._scaling = self.scaling

        # ====================================================================
        # Weight absorption: fuse kv_up_proj (W_UK / W_UV) into Q and O paths
        # ====================================================================
        with torch.no_grad():
            device = self.kv_up_proj.weight.device
            weight_dtype = self.kv_up_proj.weight.dtype
            fusion_dtype = torch.float32

            # ------------------------------------------------------------------
            # Step 1: Split kv_down_proj_with_mqa → kv_a_proj_latent + kv_a_proj_rope
            # ------------------------------------------------------------------
            W_KV_down = self.kv_down_proj_with_mqa.weight
            has_kv_bias = self.kv_down_proj_with_mqa.bias is not None
            hidden_size = W_KV_down.shape[1]
            split_idx = self.kv_lora_rank

            self.kv_a_proj_latent = nn.Linear(hidden_size, self.kv_lora_rank, bias=has_kv_bias)
            self.kv_a_proj_latent.weight = nn.Parameter(W_KV_down[:split_idx, :].contiguous().to(dtype=weight_dtype))
            if has_kv_bias:
                self.kv_a_proj_latent.bias = nn.Parameter(
                    self.kv_down_proj_with_mqa.bias[:split_idx].contiguous().to(dtype=weight_dtype)
                )

            self.kv_a_proj_rope = nn.Linear(hidden_size, self.qk_rope_head_dim, bias=has_kv_bias)
            self.kv_a_proj_rope.weight = nn.Parameter(W_KV_down[split_idx:, :].contiguous().to(dtype=weight_dtype))
            if has_kv_bias:
                self.kv_a_proj_rope.bias = nn.Parameter(
                    self.kv_down_proj_with_mqa.bias[split_idx:].contiguous().to(dtype=weight_dtype)
                )

            # ------------------------------------------------------------------
            # Step 2: Decompose kv_up_proj → W_UK + W_UV
            # ------------------------------------------------------------------
            W_Up = self.kv_up_proj.weight.to(device=device, dtype=fusion_dtype)
            D_latent = W_Up.shape[1]  # kv_lora_rank

            W_Up_view = W_Up.view(self.num_heads, self.qk_nope_head_dim + self.v_head_dim, D_latent)
            W_UK = W_Up_view[:, : self.qk_nope_head_dim, :].clone()
            W_UV = W_Up_view[:, self.qk_nope_head_dim :, :].clone()

            # ------------------------------------------------------------------
            # Step 3: Fuse W_UK into Q → q_rope_proj + q_absorbed_proj
            # ------------------------------------------------------------------
            W_Q_all = self.q_up_proj.weight.to(device=device, dtype=fusion_dtype)
            D_in = W_Q_all.shape[1]  # hidden_size or q_lora_rank

            W_Q_view = W_Q_all.view(self.num_heads, self.qk_nope_head_dim + self.qk_rope_head_dim, D_in)
            W_Q_nope = W_Q_view[:, : self.qk_nope_head_dim, :]
            W_Q_rope = W_Q_view[:, self.qk_nope_head_dim :, :]

            # q_rope_proj: [num_heads * qk_rope_head_dim, D_in]
            W_Q_rope_flat = W_Q_rope.reshape(-1, D_in).contiguous()
            self.q_rope_proj = nn.Linear(D_in, self.num_heads * self.qk_rope_head_dim, bias=False)
            self.q_rope_proj.weight = nn.Parameter(W_Q_rope_flat.to(device=device, dtype=weight_dtype))

            # q_absorbed_proj: einsum("hni, hnc -> hci", W_Q_nope, W_UK)
            W_Q_absorbed = torch.einsum("hni, hnc -> hci", W_Q_nope, W_UK)
            W_Q_absorbed_flat = W_Q_absorbed.reshape(-1, D_in).contiguous()
            self.q_absorbed_proj = nn.Linear(D_in, self.num_heads * self.kv_lora_rank, bias=False)
            self.q_absorbed_proj.weight = nn.Parameter(W_Q_absorbed_flat.to(device=device, dtype=weight_dtype))

            # ------------------------------------------------------------------
            # Step 4: Fuse W_UV into o_proj
            # ------------------------------------------------------------------
            W_O = self.o_proj.weight.to(device=device, dtype=fusion_dtype)
            W_O_view = W_O.view(hidden_size, self.num_heads, self.v_head_dim)

            W_Fused_VO = torch.einsum("xhd, hdc -> xhc", W_O_view, W_UV)
            new_in_features = self.num_heads * self.kv_lora_rank
            W_Fused_flat = W_Fused_VO.reshape(hidden_size, new_in_features).contiguous()

            has_o_bias = self.o_proj.bias is not None
            old_o_bias = self.o_proj.bias
            self.o_proj = nn.Linear(new_in_features, hidden_size, bias=has_o_bias)
            self.o_proj.weight = nn.Parameter(W_Fused_flat.to(device=device, dtype=weight_dtype))
            if has_o_bias:
                self.o_proj.bias = old_o_bias.to(device=device, dtype=weight_dtype) if old_o_bias is not None else None

            # ------------------------------------------------------------------
            # Step 5: Delete old layers
            # ------------------------------------------------------------------
            del self.q_up_proj
            del self.kv_up_proj
            del self.kv_down_proj_with_mqa

        # Alias for compatibility with text_llm_model cache shape detection
        self.head_dim = self.qk_head_dim
        return self


# ---------------------------------------------------------------------------
# IPTMoE → MoeBlock with pre-computed grouped sigmoid routing
# ---------------------------------------------------------------------------
if TYPE_CHECKING:

    class _IPTMoEBase(DynamicModule, IPTMoE):  # type: ignore[misc]
        ...

else:
    _IPTMoEBase = DynamicModule


@XHLLM_TRACEABLE_MODULES.register_module({IPTMoE: "IPTMoE"})
class _IPTMoE(_IPTMoEBase):
    def _moe(self, hidden_states, topk_indices, topk_weights):
        pass

    def _router(self, hidden_states, routing_map):
        hidden_states = hidden_states.float()
        logits = self.router.gating(hidden_states)
        logits = logits.view(-1, self.router.num_groups, self.router.num_experts)

        num_tokens, num_groups, num_experts_per_group = logits.shape
        scores = logits.sigmoid()  # [num_tokens, groups, num_routed_experts_per_group]
        if (self.router.e_score_correction_bias != 0).any():
            raise AssertionError(
                "e_score_correction_bias is not supported in the current implementation. Please set `e_score_correction_bias=0` in the configuration to disable the score correction bias and compute the routing scores directly from the gating logits without any correction."
            )
            scores_for_choice = scores.view(num_tokens, -1) + self.router.e_score_correction_bias.unsqueeze(
                0
            )  # [num_tokens, groups * num_routed_experts_per_group]
            scores_for_choice = scores_for_choice.view_as(scores)  # [num_tokens, groups, num_routed_experts_per_group]

            _, topk_indices = torch.topk(
                scores_for_choice, k=self.router.top_k, dim=-1, sorted=False
            )  # [num_tokens, groups, topk]
            topk_probs = scores.gather(-1, topk_indices)
        else:
            topk_probs, topk_indices = torch.topk(
                scores, k=self.router.top_k, dim=-1, sorted=False
            )  # [num_tokens, groups, topk]
        if self.router.top_k > 1:
            if self.router.calc_denominator_cross_groups:
                raise AssertionError(
                    "Cross-group denominator calculation is not supported in the current implementation. Please set `calc_denominator_cross_groups=False` in the configuration to compute the denominator separately for each group."
                )
                denominator = topk_probs.view(topk_probs.size(0), -1)
                denominator = denominator.sum(dim=-1, keepdim=True) + 1e-20
                denominator = denominator.unsqueeze(-1)
            else:
                denominator = topk_probs.sum(dim=-1, keepdim=True) + 1e-20
            topk_probs = topk_probs / denominator
        # topk_probs = topk_probs * self.router.routed_scaling_factor

        topk_mask = torch.zeros(logits.shape, dtype=torch.int32, device=logits.device).scatter(-1, topk_indices, 1)
        tokens_per_expert = topk_mask.sum(dim=0)
        assert num_groups == 1, (
            "Multiple groups are not supported in the current implementation. Please set `num_groups=1` in the configuration to use a single group of experts and compute the routing weights accordingly."
        )
        head_incre = (
            torch.arange(num_groups, dtype=topk_indices.dtype, device=topk_indices.device) * num_experts_per_group
        ).view(1, -1, 1)
        topk_indices = (topk_indices + head_incre).view(num_tokens, -1)
        topk_probs = topk_probs.view(num_tokens, -1)
        tokens_per_expert = tokens_per_expert.view(-1)
        tokens_per_expert = tokens_per_expert.cpu().to(torch.long)

        return topk_probs.to(torch.float32), topk_indices, tokens_per_expert

    def forward(self, hidden_states, routing_map=None):
        orig_shape = hidden_states.shape
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        # if routing_map is not None:
        #     # print('0000000000000000000000000000')
        #     # 检查一下routing map的尺寸是否相符
        #     rp_batch_size, rp_sequence, rp_expert_num = routing_map.shape

        #     assert rp_batch_size == batch_size, (
        #         f"[spark_routing_model] Shape mismatch: routing_map.shape={routing_map.shape} but hidden_states.shape={hidden_states.shape}"
        #     )
        #     assert rp_sequence == sequence_length, (
        #         f"[spark_routing_model] Shape mismatch: routing_map.shape={routing_map.shape} but hidden_states.shape={hidden_states.shape}"
        #     )
        #     assert rp_expert_num == self.config.num_experts_per_tok, (
        #         f"[spark_routing_model] Expert number mismatch: rp_expert_num={rp_expert_num} but top_k={self.config.num_experts_per_tok}"
        #     )
        assert routing_map is None, (
            "Routing map is not supported for the attention layer and should only be passed to the MLP. Please make sure to only pass the routing map in the `routing_maps` argument of the decoder and not in `position_embeddings` or other arguments."
        )

        if self.shared_experts:
            shared_output = self.shared_experts(hidden_states)

        if self.grouped_gemm:
            raise AssertionError(
                "Grouped GEMM is not supported in the current implementation. Please set `grouped_gemm=False` in the configuration to use the standard MoE forward pass with pre-computed routing weights."
            )
            probs, indices, tokens_per_expert = self.router(hidden_states, routing_map)
            hidden_states = self.grouped_moe(hidden_states, indices, probs, tokens_per_expert).view(*orig_shape)
        else:
            # 1. 计算激活分数
            logits = self.router.gating(hidden_states)  # [num_tokens, num_groups * num_experts_per_group]
            scores = logits.sigmoid()
            # 2. 加入专家修正偏差（防止负载失衡）
            scores_for_choice = scores + self.e_score_correction_bias

            # 3. 选取前 K 个专家
            _, topk_indices = torch.topk(
                scores_for_choice, k=self.config.num_experts_per_tok, dim=-1, sorted=True
            )  # [num_tokens, groups, topk]

            # 4. 获取对应专家的权重
            # routing_weights = scores.gather(-1, topk_indices)
            routing_weights = scores

            # 5. 调整维度以匹配 (batch, seq_len, k)
            # routing_weights = routing_weights.view(batch_size, sequence_length, self.config.num_experts_per_tok)
            # topk_indices = topk_indices.view(batch_size, sequence_length, self.config.num_experts_per_tok)

            # 6. 将 Token 路由至选中的专家进行计算
            hidden_states = self.moeblock(hidden_states, routing_weights, topk_indices)

            # 7. 应用缩放因子
            hidden_states = hidden_states * self.router.routed_scaling_factor

        if self.shared_experts:
            hidden_states = hidden_states + shared_output

        return hidden_states

    def _setup(self, cfg: dict | None = None):
        assert self.grouped_gemm is False, (
            "Grouped GEMM is not supported in the current implementation. Please set `grouped_gemm=False` in the configuration to use the standard MoE forward pass with pre-computed routing weights."
        )
        # has_expert_modules = True
        # has_gate_quant = has_expert_modules and all(
        #     hasattr(expert.fc1, "quant_weight") and expert.fc1.quant_weight is not None
        #     for expert in self.routed_experts
        # )
        # has_up_quant = has_expert_modules and all(
        #     hasattr(expert.fc1, "quant_weight") and expert.fc1.quant_weight is not None
        #     for expert in self.routed_experts
        # )
        # has_down_quant = has_expert_modules and all(
        #     hasattr(expert.fc2, "quant_weight") and expert.fc2.quant_weight is not None
        #     for expert in self.routed_experts
        # )
        self.register_buffer(
            "e_score_correction_bias",
            self.router.e_score_correction_bias.unsqueeze(0).unsqueeze(0).contiguous(),
            persistent=False,
        )
        self.moeblock = MoeBlock(
            self.config.hidden_act, self.config.num_experts_per_tok, normalize_routing_weights=True, topk_outside=True
        )

        expert_modules = [expert for expert in self.routed_experts]
        gate_proj_weights = []
        gate_proj_biases = []
        up_proj_weights = []
        up_proj_biases = []
        down_proj_weights = []
        down_proj_biases = []

        with torch.no_grad():
            for expert in expert_modules:
                if hasattr(expert, "fc1"):
                    gate_proj_weight = expert.fc1.weight.data[: expert.intermediate_size, :].unsqueeze(0)
                    up_proj_weight = expert.fc1.weight.data[expert.intermediate_size :, :].unsqueeze(0)
                    down_proj_weight = expert.fc2.weight.data.unsqueeze(0)
                else:
                    gate_proj_weight = expert.gate_proj.weight.data.unsqueeze(0)
                    up_proj_weight = expert.up_proj.weight.data.unsqueeze(0)
                    down_proj_weight = expert.down_proj.weight.data.unsqueeze(0)

                gate_proj_weights.append(gate_proj_weight)
                up_proj_weights.append(up_proj_weight)
                down_proj_weights.append(down_proj_weight)

                if hasattr(expert, "fc1") and expert.fc1.bias is not None:
                    gate_proj_bias = expert.fc1.bias.data[: expert.intermediate_size].unsqueeze(0)
                    up_proj_bias = expert.fc1.bias.data[expert.intermediate_size :].unsqueeze(0)
                    gate_proj_biases.append(gate_proj_bias)
                    up_proj_biases.append(up_proj_bias)
                else:
                    if hasattr(expert, "gate_proj") and expert.gate_proj.bias is not None:
                        gate_proj_biases.append(expert.gate_proj.bias.data.unsqueeze(0))
                    if hasattr(expert, "up_proj") and expert.up_proj.bias is not None:
                        up_proj_biases.append(expert.up_proj.bias.data.unsqueeze(0))

                if hasattr(expert, "fc2") and expert.fc2.bias is not None:
                    down_proj_bias = expert.fc2.bias.data.unsqueeze(0)
                    down_proj_biases.append(down_proj_bias)

        gate_proj_weight = torch.cat(gate_proj_weights, dim=0).contiguous()
        up_proj_weight = torch.cat(up_proj_weights, dim=0).contiguous()
        down_proj_weight = torch.cat(down_proj_weights, dim=0).contiguous()

        self.moeblock.expert_gate_proj_weight = nn.Parameter(gate_proj_weight)
        self.moeblock.expert_up_proj_weight = nn.Parameter(up_proj_weight)
        self.moeblock.expert_down_proj_weight = nn.Parameter(down_proj_weight)

        if len(gate_proj_biases) > 0:
            self.moeblock.expert_gate_proj_bias = nn.Parameter(torch.cat(gate_proj_biases, dim=0))
        else:
            self.moeblock.expert_gate_proj_bias = None

        if len(up_proj_biases) > 0:
            self.moeblock.expert_up_proj_bias = nn.Parameter(torch.cat(up_proj_biases, dim=0))
        else:
            self.moeblock.expert_up_proj_bias = None

        if len(down_proj_biases) > 0:
            self.moeblock.expert_down_proj_bias = nn.Parameter(torch.cat(down_proj_biases, dim=0))
        else:
            self.moeblock.expert_down_proj_bias = None

        del self.routed_experts
        torch.cuda.empty_cache()

        return self


# ---------------------------------------------------------------------------
# IPTDecoderLayer
# ---------------------------------------------------------------------------
if TYPE_CHECKING:

    class _IPTDecoderLayerBase(DynamicModule, IPTDecoderLayer):  # type: ignore[misc]
        ...

else:
    _IPTDecoderLayerBase = DynamicModule


@XHLLM_TRACEABLE_MODULES.register_module({IPTDecoderLayer: "IPTDecoderLayer"})
class _IPTDecoderLayer(_IPTDecoderLayerBase):
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
        hidden_states = residual + hidden_states

        # Feed-forward (dense MLP or MoE)
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return (hidden_states,)

    def _setup(self, cfg: dict | None = None):
        # Create alias expected by text_llm_model._wraped_post
        self.self_attn = self.attention
        return self


# ---------------------------------------------------------------------------
# _IPTDecoder
# ---------------------------------------------------------------------------
if TYPE_CHECKING:

    class _IPTDecoderBase(DynamicModule, IPTDecoder):  # type: ignore[misc]
        ...

else:
    _IPTDecoderBase = DynamicModule


@XHLLM_TRACEABLE_MODULES.register_module({IPTDecoder: "IPTDecoder"})
class _IPTDecoder(_IPTDecoderBase):
    def graph_forward(
        self,
        hidden_states: torch.FloatTensor | None = None,
        past_seq_length: Tensor | None = None,
        current_input_length: Tensor | None = None,
        past_key_cache: list[Tensor] | None = None,
        past_value_cache: list[Tensor] | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> Tensor:
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

        hidden_states = self.layernorm(hidden_states)
        return hidden_states

    def _setup(self, cfg: dict | None = None):
        self.only_first_block = cfg.get("only_first_block", False)
        self.max_layers = 1 if self.only_first_block else -1
        if "max_layers" in cfg:
            self.max_layers = cfg.max_layers
        self.use_cache = cfg.use_cache
        self.num_logits_to_keep = cfg.num_logits_to_keep
        assert self.num_logits_to_keep in [0, 1]

        input_sequence_length = cfg.input_sequence_length

        self.llm_gather = xhnn.BatchGather(1)
        self.llm_gather.update_offset_indices(1, input_sequence_length)

        def _llm_gather_update_cfg(self: xhnn.BatchGather, cfg: Optional[dict] = None):
            input_seq_len = cfg.input_sequence_length
            batch_size = cfg.get("batch_size", 1)
            self.update_offset_indices(batch_size, input_seq_len)

        self.llm_gather._update_cfg = types.MethodType(_llm_gather_update_cfg, self.llm_gather)

        # self.use_cache = cfg.use_cache


# ---------------------------------------------------------------------------
# IPTModel
# ---------------------------------------------------------------------------
if TYPE_CHECKING:

    class _IPTModelBase(DynamicModule, IPTModel):  # type: ignore[misc]
        ...

else:
    _IPTModelBase = DynamicModule


@XHLLM_TRACEABLE_MODULES.register_module({IPTModel: "IPTModel"})
class _IPTModel(_IPTModelBase):
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

        hidden_states = self.transformer(
            hidden_states,
            current_input_length=current_input_length,
            past_seq_length=past_seq_length,
            past_key_cache=past_key_cache,
            past_value_cache=past_value_cache,
            position_embeddings=position_embeddings,
        )

        return hidden_states

    def _setup_cos_sin_embedding(self):
        cos_cached = self.rotary_emb.cos_cached
        sin_cached = self.rotary_emb.sin_cached
        if cos_cached is None or sin_cached is None:
            from xhquant.utils import get_root_logger

            get_root_logger().warning("Cosine/sine caches are not set. Positional embeddings may be incorrect.")

    def _setup(self, cfg: dict | None = None):
        input_sequence_length = cfg.input_sequence_length

        self.sin_slice = xhnn.DynamicSlice([input_sequence_length], [2], [1])
        self.cos_slice = xhnn.DynamicSlice([input_sequence_length], [2], [1])

        def _sin_cos_slice_update_cfg(self, cfg: Optional[dict] = None):
            input_seq_len = cfg.input_sequence_length
            self.valid_length = [input_seq_len]

        self.sin_slice._update_cfg = types.MethodType(_sin_cos_slice_update_cfg, self.sin_slice)
        self.cos_slice._update_cfg = types.MethodType(_sin_cos_slice_update_cfg, self.cos_slice)

        if not hasattr(self.rotary_emb, "cos_cached"):
            self.rotary_emb.setup_after_callback = self._setup_cos_sin_embedding
        else:
            self._setup_cos_sin_embedding()

        return self


# ---------------------------------------------------------------------------
# IPTForCausalLM
# ---------------------------------------------------------------------------
if TYPE_CHECKING:

    class _IPTForCausalLMBase(DynamicModule, IPTForCausalLM):  # type: ignore[misc]
        ...

else:
    _IPTForCausalLMBase = DynamicModule


@XHLLM_TRACEABLE_MODULES.register_module({IPTForCausalLM: "IPTForCausalLM"})
class _IPTForCausalLM(_IPTForCausalLMBase):
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
