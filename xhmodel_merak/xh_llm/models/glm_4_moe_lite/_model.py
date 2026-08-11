import math
import sys
import types
from copy import deepcopy
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.glm4_moe_lite.modeling_glm4_moe_lite import (
    Glm4MoeLiteAttention,
    Glm4MoeLiteDecoderLayer,
    Glm4MoeLiteForCausalLM,
    Glm4MoeLiteMoE,
    Glm4MoeLiteModel,
    Glm4MoeLiteNaiveMoe,
    Glm4MoeLiteRMSNorm,
    Glm4MoeLiteRotaryEmbedding,
    Glm4MoeLiteTopkRouter,
)
from xhquant import nn as xhnn
from xhquant.api import ConfigDict
from xhquant.nn import LLMCache, MaskedSoftmax, RMSNorm
from xhquant.nn.modules.moeblock import MoeBlock
from xhquant.utils.registry import DynamicModule

from ...register import XHLLM_TRACEABLE_MODULES


def _get_wrap_debug_flag(cfg: dict[str, Any] | ConfigDict | None, key: str, default: bool = False) -> bool:
    """Read optional debug flags from wrap_cfg.debug.<key> with safe fallback."""
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        cfg = ConfigDict(cfg)
    if not hasattr(cfg, "get"):
        return default

    debug_cfg = cfg.get("debug", None)
    if debug_cfg is None:
        return default
    if isinstance(debug_cfg, dict):
        debug_cfg = ConfigDict(debug_cfg)
    if not hasattr(debug_cfg, "get"):
        return default
    return bool(debug_cfg.get(key, default))


@XHLLM_TRACEABLE_MODULES.register_module(
    {
        Glm4MoeLiteRotaryEmbedding: "Glm4MoeLiteRotaryEmbedding",
    }
)
class _Glm4MoeLiteRotaryEmbedding(DynamicModule):
    def _setup(self, cfg: dict | None = None):
        assert "dynamic" not in self.rope_type, f"{self.rope_type} is not supported in dynamic mode"

        # self.max_position_embeddings = max_position_embeddings
        # Build here to make `torch.jit.trace` work.
        self._setup_cos_sin_cache(seq_len=self.max_seq_len_cached, dtype=self.inv_freq.dtype)
        if hasattr(self, "setup_after_callback"):
            self.setup_after_callback()

    def _setup_cos_sin_cache(self, seq_len, dtype):
        """
        4.45 版本实现
        """
        position_ids = torch.arange(0, seq_len, dtype=torch.long, device=self.inv_freq.device).unsqueeze(0)
        # self.inv_freq = self.inv_freq.to(torch.float16)
        inv_freq = self.inv_freq.to(torch.float32)
        device = self.inv_freq.device
        if torch.cuda.is_available() and inv_freq.device.type != "cuda":
            inv_freq = self.inv_freq.cuda()

        cos, sin = self.forward(inv_freq, position_ids)
        cos = cos.to(device)
        sin = sin.to(device)
        sin = sin.squeeze(0)
        cos = cos.squeeze(0)

        # TODO:临时处理
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

        self.register_buffer("sin_cached", sin.to(dtype=dtype), persistent=False)
        self.register_buffer("cos_cached", cos.to(dtype=dtype), persistent=False)
        # self.sin_cached = nn.Parameter(sin.to(device=device, dtype=dtype), requires_grad=False)
        # self.cos_cached = nn.Parameter(cos.to(device=device, dtype=dtype), requires_grad=False)

    def _set_dtype(self, dtype: torch.dtype) -> None:
        self.inv_freq = self.inv_freq.to(dtype)
        self._setup_cos_sin_cache(seq_len=self.max_seq_len_cached, dtype=self.inv_freq.dtype)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        """
        4.37 版本实现
        """
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)

        # Core RoPE block
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        # with torch.autocast(device_type=device_type, enabled=False):
        #     freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        #     emb = torch.cat((freqs, freqs), dim=-1)
        #     cos = emb.cos()
        #     sin = emb.sin()
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


@XHLLM_TRACEABLE_MODULES.register_module(
    {
        Glm4MoeLiteAttention: "Glm4MoeLiteAttention",
    }
)
class _Glm4MoeLiteAttention(DynamicModule):
    def rotate_half(self, x: Tensor):
        """Rotates half the hidden dims of the input."""
        # x1 = x[..., : x.shape[-1] // 2]
        # x2 = x[..., x.shape[-1] // 2 :]
        # x1 = torch_ops_xh2a_slice(x, [0], [self.head_dim // 2], [3], [1])
        # x2 = torch_ops_xh2a_slice(x, [self.head_dim // 2], [sys.maxsize], [3], [1])
        x1 = self.slice_1(x)
        x2 = self.slice_2(x)
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(self, q: Tensor, k: Tensor, cos: Tensor, sin: Tensor, unsqueeze_dim: int = 1):
        # cos = cos.unsqueeze(unsqueeze_dim)
        # sin = sin.unsqueeze(unsqueeze_dim)
        # cos = self.cos_unsqueeze(cos)
        # sin = self.sin_unsqueeze(sin)
        if self.enable_rope:
            q_embed = self.rope(q, cos, sin)
            k_embed = self.rope(k, cos, sin)
        else:
            q_embed = (q * cos) + (self.rotate_half(q) * sin)
            k_embed = (k * cos) + (self.rotate_half(k) * sin)
        return q_embed, k_embed

    def apply_rotary_pos_emb_interleave(self, q: Tensor, k: Tensor, cos: Tensor, sin: Tensor, unsqueeze_dim: int = 1):

        b, h, s, d = q.shape
        q = q.view(b, h, s, -1, 2).transpose(4, 3).reshape(b, h, s, d)

        b, h, s, d = k.shape
        k = k.view(b, h, s, -1, 2).transpose(4, 3).reshape(b, h, s, d)

        if self.enable_rope:
            q_embed = self.rope(q, cos, sin)
            k_embed = self.rope(k, cos, sin)
        else:
            q_embed = (q * cos) + (self.rotate_half(q) * sin)
            k_embed = (k * cos) + (self.rotate_half(k) * sin)
        return q_embed, k_embed
    
    def forward_standard(
        self,
        hidden_states: torch.Tensor,
        past_seq_length: Tensor | None = None,
        current_input_length: Tensor | None = None,
        # position_ids: torch.Tensor = None,
        past_k_cache: Tensor | None = None,
        past_v_cache: Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]:
        batch_size, seq_length, hidden_dim = hidden_states.shape
        # query_shape = (batch_size, seq_length, -1, self.qk_head_dim)
        # key_shape = (batch_size, seq_length, -1, self.qk_nope_head_dim + self.v_head_dim)

        if self.q_lora_rank is None:
            q_states = self.q_proj(hidden_states)
        else:
            q_states = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q_states = q_states.view(batch_size, seq_length, -1, self.qk_head_dim).transpose(1, 2)
        q_pass, q_rot = torch.split(q_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        k_pass, k_rot = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)

        k_pass = (
            self.kv_b_proj(self.kv_a_layernorm(k_pass))
            .view(batch_size, seq_length, -1, self.qk_nope_head_dim + self.v_head_dim)
            .transpose(1, 2)
        )
        k_pass, value_states = torch.split(k_pass, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        k_rot = k_rot.view(batch_size, 1, seq_length, self.qk_rope_head_dim)

        cos, sin = position_embeddings
        if self.config.rope_interleave:  # support using interleaved weights for efficiency
            q_rot, k_rot = self.apply_rotary_pos_emb_interleave(q_rot, k_rot, cos, sin)
        else:
            q_rot, k_rot = self.apply_rotary_pos_emb(q_rot, k_rot, cos, sin)
        k_rot = k_rot.expand(-1, self.num_heads, -1, -1)
        

        query_states = torch.cat((q_pass, q_rot), dim=-1)
        key_states = torch.cat((k_pass, k_rot), dim=-1)

        if self.use_cache:
            key_states = self.k_cache(key_states, past_seq_length, current_input_length, past_k_cache)
            value_states = self.v_cache(value_states, past_seq_length, current_input_length, past_v_cache)


        query_states = query_states * self.kv_scale
        key_states = key_states.transpose(2, 3)
        key_states = torch.repeat_interleave(
            key_states,
            self.num_key_value_groups,
            dim=1,
        )
        attn_weights = torch.matmul(query_states, key_states)  # [4, 28, 256, 128], [4, 28, 128, 32768]
        attn_weights: Tensor | None = self.masked_softmax(attn_weights, past_seq_length)
        value_states = torch.repeat_interleave(
            value_states,
            self.num_key_value_groups,
            dim=1,
        )
        attn_output = torch.matmul(attn_weights, value_states)  # [4, 28, 256, 32768], [4, 28, 32768, 128]
        attn_output = attn_output.transpose(1, 2)


        attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, None, None

    def _setup_standard(self, cfg: dict | None = None):
        if isinstance(cfg, dict):
            cfg = ConfigDict(cfg)
        if not hasattr(self, "num_key_value_heads"):
            self.num_key_value_heads = self.config.num_key_value_heads
        if not hasattr(self, "num_heads"):
            self.num_heads = self.config.num_attention_heads  # 28
        if not hasattr(self, "hidden_size"):
            self.hidden_size = self.config.hidden_size  # 3584

        self.enable_rope = cfg.get("enable_rope", True)
        if self.enable_rope:
            self.rope = xhnn.Rope()

        self.slice_1 = xhnn.Slice([0], [self.qk_head_dim // 2], [3], [1])
        self.slice_2 = xhnn.Slice([self.qk_head_dim // 2], [sys.maxsize], [3], [1])
        self.half_head_dim = self.qk_head_dim // 2

        self.masked_softmax = MaskedSoftmax(dim=-1)
        use_cache = cfg.use_cache
        self.use_cache = use_cache
        self.key_extra_scale = 1.0 if "key_extra_scale" not in cfg else cfg.key_extra_scale
        self.query_extra_scale = 1.0 if "query_extra_scale" not in cfg else cfg.query_extra_scale

        self.cos_gather = xhnn.Gather(0)
        self.sin_gather = xhnn.Gather(0)

        if use_cache:
            cache_axis = cfg.kv_cache.cache_axis
            self.k_cache = LLMCache(
                axis=cache_axis,
                attention_max_length=-1,
            )
            self.v_cache = LLMCache(
                axis=cache_axis,
                attention_max_length=-1,
            )
        else:
            self.k_cache = None
            self.v_cache = None
        _kv_scale = 1 / math.sqrt(self.qk_head_dim)
        self.kv_scale = _kv_scale
        return self

    def graph_forward(
        self,
        hidden_states: torch.Tensor,
        past_seq_length: Tensor | None = None,
        current_input_length: Tensor | None = None,
        # position_ids: torch.Tensor = None,
        past_k_cache: Tensor | None = None,
        past_v_cache: Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]:
        batch_size, seq_length, hidden_dim = hidden_states.shape


        # =================================================================
        # 1. Query 生成：兵分两路 (RoPE vs Content)
        # =================================================================
        # Q-LoRA 的压缩层 (如果存在)
        if self.q_lora_rank is None:
            inp = hidden_states
        else:
            # 先过压缩层和 Norm
            inp = self.q_a_layernorm(self.q_a_proj(hidden_states))

        # 【路径 A】RoPE 部分 Query
        # q_pe: [B, S, H, D_rope]
        q_pe = self.q_rope_proj(inp).view(batch_size, seq_length, self.num_heads, self.qk_rope_head_dim)

        # 【路径 B】Content 部分 Query (已融合 W_UK)
        # q_content: [B, S, H, LatentDim]
        # 注意：这里直接输出了 Latent 维度的 Query，不需要再乘 W_UK
        q_content = self.q_absorbed_proj(inp).view(batch_size, seq_length, self.num_heads, self.kv_lora_rank)

        # =================================================================


        # 2. Key/Value 生成：Latent 空间 (压缩)
        # =================================================================
        # 生成压缩的 KV (Latent Vector)
        # A. 生成 Latent 向量 [B, S, LatentDim]
        k_latent_raw = self.kv_a_proj_latent(hidden_states)

        # 对 Latent 向量做 Layernorm (这是 MLA 的标准操作)
        k_latent = self.kv_a_layernorm(k_latent_raw)

        # B. 生成 RoPE 部分向量 [B, S, RopeDim]
        k_rot = self.kv_a_proj_rope(hidden_states)

        # =================================================================
        # 3. RoPE 位置编码
        # =================================================================
        # k_rot: [B, 1, S, RopeDim] (调整为 RoPE 接口需要的形状)
        k_rot = k_rot.view(batch_size, 1, seq_length, self.qk_rope_head_dim)
        cos, sin = position_embeddings
        # 为了 RoPE 计算，调整 q_pe 维度 -> [B, H, S, D]
        q_pe = q_pe.transpose(1, 2) 
        if self.config.rope_interleave:
            q_pe, k_rot = self.apply_rotary_pos_emb_interleave(q_pe, k_rot, cos, sin)
        else:
            q_pe, k_rot = self.apply_rotary_pos_emb(q_pe, k_rot, cos, sin)

        # q_pe 恢复形状 -> [B, S, H, D]
        q_pe = q_pe.transpose(1, 2)

        query_states = torch.cat((q_content, q_pe), dim=-1).transpose(1,2) * self.kv_scale
        k_rot = k_rot.squeeze(1)
        key_states = torch.cat((k_latent, k_rot), dim=-1).unsqueeze(1)
        value_states = k_latent.unsqueeze(1)

        # =================================================================
        # 4. KV Cache 管理 (极低显存占用)
        # =================================================================
        if self.use_cache:
            # K cache: [B, 1, S_total, LatentDim + RopeDim], 序列轴为 -2。
            key_states = self.k_cache(key_states, past_seq_length, current_input_length, past_k_cache)

            # V cache: [B, 1, S_total, LatentDim], 序列轴为 -2。
            value_states = self.v_cache(value_states, past_seq_length, current_input_length, past_v_cache)

        key_states = key_states.transpose(2, 3)

        # MQA/MLA 优化：因为unsqueeze(1)之后对应的维度大小为 1，直接利用 Matmul 的原生广播（Broadcasting）免去 repeat_interleave

        # xh_pragma_fx(
        #     query_states,
        #     {
        #         "type": "quanted",
        #         "action": "start",
        #         "qconfig": {
        #             "act_schema": {
        #                 "bits": 16,
        #                 "fp_mode": "sefp",
        #             },
        #             "act_schema_2": {
        #                 "bits": 16,
        #                 "fp_mode": "sefp",
        #             },
        #         },
        #     },
        # )
        attn_weights = torch.matmul(query_states, key_states)
        # xh_pragma_fx(
        #     query_states,
        #     {
        #         "type": "quanted",
        #         "action": "end",
        #     },
        # )

        attn_weights: Tensor | None = self.masked_softmax(attn_weights, past_seq_length)

        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
        attn_output = self.o_proj(attn_output)


        # k_rot 扩展到所有头 (MQA 广播) -> [B, S, H, D]
        # # 注意：如果 k_rot 是单头的，这里可以不 expand，利用 matmul 广播，但为了代码清晰保持一致
        # k_rot = k_rot.unsqueeze(1).expand(-1, self.num_heads, -1, -1).transpose(1, 2) # [B, S, H, D]

        # # =================================================================
        # # 5. Attention Score 计算 (融合核心)
        # # =================================================================
        # # 调整 K 的维度以适配 Multi-Head Dot Product
        # # k_latent: [B, S, LatentDim] -> [B, S, 1, LatentDim] (为了广播)
        # k_latent_head = k_latent.unsqueeze(2)

        # # Part A: Content Score (Latent Space)
        # # q_content: [B, S, H, Latent]
        # # k_latent:  [B, S, 1, Latent]
        # # result:    [B, S, H, S_kv] (Implicit Transpose in matmul or einsum)
        # # 这里使用 einsum 更清晰: bshc (q), btkc (k) -> bsht
        # scores_content = torch.einsum("bshc,btkc->bsht", q_content, k_latent_head)

        # # Part B: RoPE Score
        # # q_pe: [B, S, H, RopeDim]
        # # k_rot: [B, S, H, RopeDim]
        # scores_rope = torch.einsum("bshd,bthd->bsht", q_pe, k_rot)

        # # 总分融合
        # attn_weights = (scores_content + scores_rope) * self.kv_scale

        # # Mask & Softmax
        # attn_weights = self.masked_softmax(attn_weights, past_seq_length)

        # # =================================================================
        # # 6. Context 聚合 (Value 聚合)
        # # =================================================================
        # # 注意：MLA 的 Value 就是 k_latent 本身！
        # # Weights: [B, S, H, S_kv]
        # # Values:  [B, S_kv, 1, Latent] (广播)
        # # Output:  [B, S, H, Latent]
        # context_layer = torch.einsum("bsht,btkc->bshc", attn_weights, k_latent_head)
        # # 展平 Head 和 Latent 维度
        # # [B, S, H * Latent]
        # context_layer = context_layer.flatten(2)
        # # 直接过融合后的 Output Layer
        # # 这一步等价于: (Context * W_UV) * W_O
        # attn_output = self.o_proj(context_layer)

        return attn_output, None, None

    def _setup(self, cfg: dict | None = None):
        if isinstance(cfg, dict):
            cfg = ConfigDict(cfg)
        if not hasattr(self, "num_key_value_heads"):
            self.num_key_value_heads = self.config.num_key_value_heads
        if not hasattr(self, "num_heads"):
            self.num_heads = self.config.num_attention_heads  # 28
        if not hasattr(self, "hidden_size"):
            self.hidden_size = self.config.hidden_size  # 3584

        self.enable_rope = cfg.get("enable_rope", True)
        if self.enable_rope:
            self.rope = xhnn.Rope()

        self.slice_1 = xhnn.Slice([0], [self.qk_head_dim // 2], [3], [1])
        self.slice_2 = xhnn.Slice([self.qk_head_dim // 2], [sys.maxsize], [3], [1])
        self.half_head_dim = self.qk_head_dim // 2

        self.masked_softmax = MaskedSoftmax(dim=-1)
        use_cache = cfg.use_cache
        self.use_cache = use_cache
        self.key_extra_scale = 1.0 if "key_extra_scale" not in cfg else cfg.key_extra_scale
        self.query_extra_scale = 1.0 if "query_extra_scale" not in cfg else cfg.query_extra_scale

        self.cos_gather = xhnn.Gather(0)
        self.sin_gather = xhnn.Gather(0)

        if use_cache:
            cache_axis = cfg.kv_cache.cache_axis
            self.k_cache = LLMCache(
                axis=cache_axis,
                attention_max_length=-1,
            )
            self.v_cache = LLMCache(
                axis=cache_axis,
                attention_max_length=-1,
            )
        else:
            self.k_cache = None
            self.v_cache = None
        _kv_scale = 1 / math.sqrt(self.qk_head_dim)
        self.kv_scale = _kv_scale

        def _maybe_register_quant_weight(linear: torch.nn.Linear, quant_weight: torch.Tensor | None):
            if quant_weight is not None:
                assert quant_weight.dtype in (torch.int8, torch.int16)
                linear.register_buffer("quant_weight", quant_weight.contiguous())

        with torch.no_grad(): # 将kv_b_proj融合到q_absorbed_proj 和 o_proj中
            # ========================================================================
            # 第一步：拆分 kv_a_proj_with_mqa 
            # ========================================================================
            if hasattr(self, "kv_a_proj_with_mqa"):
                W_KV_A = self.kv_a_proj_with_mqa.weight # [Latent + Rope, Hidden]
                W_KV_A_quant = getattr(self.kv_a_proj_with_mqa, "quant_weight", None)
                has_bias = self.kv_a_proj_with_mqa.bias is not None
                proj_dtype = W_KV_A.dtype
                
                # 确定切分点
                split_idx = self.kv_lora_rank
                
                # 1. 构造 Latent 投影层
                W_Latent = W_KV_A[:split_idx, :].contiguous()
                self.kv_a_proj_latent = torch.nn.Linear(self.hidden_size, self.kv_lora_rank, bias=has_bias)
                self.kv_a_proj_latent.weight = torch.nn.Parameter(W_Latent.to(dtype=proj_dtype))
                _maybe_register_quant_weight(
                    self.kv_a_proj_latent,
                    None
                    if W_KV_A_quant is None
                    else W_KV_A_quant[:split_idx, :].to(device=self.kv_a_proj_latent.weight.device),
                )
                if has_bias:
                    self.kv_a_proj_latent.bias = torch.nn.Parameter(
                        self.kv_a_proj_with_mqa.bias[:split_idx].contiguous().to(dtype=proj_dtype)
                    )
                    
                # 2. 构造 RoPE 投影层
                W_Rope = W_KV_A[split_idx:, :].contiguous()
                self.kv_a_proj_rope = torch.nn.Linear(self.hidden_size, self.qk_rope_head_dim, bias=has_bias)
                self.kv_a_proj_rope.weight = torch.nn.Parameter(W_Rope.to(dtype=proj_dtype))
                _maybe_register_quant_weight(
                    self.kv_a_proj_rope,
                    None
                    if W_KV_A_quant is None
                    else W_KV_A_quant[split_idx:, :].to(device=self.kv_a_proj_rope.weight.device),
                )
                if has_bias:
                    self.kv_a_proj_rope.bias = torch.nn.Parameter(
                        self.kv_a_proj_with_mqa.bias[split_idx:].contiguous().to(dtype=proj_dtype)
                    )

            # ========================================================================
            # 第二步：处理 kv_b_proj (必须最先做，因为 Q 融合依赖于 W_UK)
            # ========================================================================
            # 原始权重形状: [NumHeads * (NopeDim + VDim), LatentDim]
            # 注意：PyTorch Linear 权重是 [Out, In]
            device = self.kv_b_proj.weight.device
            weight_dtype = self.kv_b_proj.weight.dtype
            fusion_dtype = torch.float32
            W_Up = self.kv_b_proj.weight.to(device=device, dtype=fusion_dtype)
            W_Up_quant = getattr(self.kv_b_proj, "quant_weight", None)
            D_latent = W_Up.shape[1]
            
            # View 成 [NumHeads, NopeDim + VDim, LatentDim]
            W_Up_view = W_Up.view(self.num_heads, self.qk_nope_head_dim + self.v_head_dim, D_latent)
            W_Up_quant_view = (
                W_Up_quant.to(device=device).view(self.num_heads, self.qk_nope_head_dim + self.v_head_dim, D_latent)
                if W_Up_quant is not None
                else None
            )
            
            # 提取 W_UK (用于 Q 融合) -> [NumHeads, NopeDim, LatentDim]
            W_UK = W_Up_view[:, :self.qk_nope_head_dim, :].clone()
            W_UK_quant = W_Up_quant_view[:, :self.qk_nope_head_dim, :].clone() if W_Up_quant_view is not None else None
            
            # 提取 W_UV (用于 O 融合) -> [NumHeads, VDim, LatentDim]
            W_UV = W_Up_view[:, self.qk_nope_head_dim:, :].clone()
            W_UV_quant = W_Up_quant_view[:, self.qk_nope_head_dim:, :].clone() if W_Up_quant_view is not None else None

            # ========================================================================
            # 第三步：处理 q_b_proj (依赖 W_UK)
            # ========================================================================
            W_Q_all = self.q_b_proj.weight.to(
                device=device, dtype=fusion_dtype
            )  # [NumHeads * (NopeDim + RopeDim), Hidden]
            W_Q_all_quant = getattr(self.q_b_proj, "quant_weight", None)
            D_in = W_Q_all.shape[1]
            
            # View 成 [NumHeads, NopeDim + RopeDim, Hidden]
            W_Q_view = W_Q_all.view(self.num_heads, self.qk_nope_head_dim + self.qk_rope_head_dim, D_in)
            W_Q_quant_view = (
                W_Q_all_quant.to(device=device).view(self.num_heads, self.qk_nope_head_dim + self.qk_rope_head_dim, D_in)
                if W_Q_all_quant is not None
                else None
            )
            
            # 拆分 Nope (Content) 和 Rope
            W_Q_nope = W_Q_view[:, :self.qk_nope_head_dim, :] # [H, NopeDim, Hidden]
            W_Q_rope = W_Q_view[:, self.qk_nope_head_dim:, :] # [H, RopeDim, Hidden]
            W_Q_nope_quant = (
                W_Q_quant_view[:, :self.qk_nope_head_dim, :]
                if W_Q_quant_view is not None
                else None
            )
            W_Q_rope_quant = W_Q_quant_view[:, self.qk_nope_head_dim:, :].clone() if W_Q_quant_view is not None else None
            
            # --- 2.1 构造 RoPE 专用层 ---
            # 直接展平 W_Q_rope -> [H * RopeDim, Hidden]
            W_Q_rope_flat = W_Q_rope.reshape(-1, D_in).contiguous()
            self.q_rope_proj = torch.nn.Linear(D_in, self.num_heads * self.qk_rope_head_dim, bias=False)
            self.q_rope_proj.weight = torch.nn.Parameter(W_Q_rope_flat.to(device=device, dtype=weight_dtype))
            _maybe_register_quant_weight(
                self.q_rope_proj,
                None if W_Q_rope_quant is None else W_Q_rope_quant.reshape(-1, D_in).to(device=device),
            )
            
            # --- 2.2 构造 Content (Absorbed) 专用层 ---
            # 融合公式: W_Fused = W_UK^T * W_Q_nope
            # 维度: [H, Latent, Hidden] = [H, Nope, Hidden] * [H, Nope, Latent] (转置后)
            # Einsum: "hni, hnc -> hci" (h=Head, n=NopeDim, i=Hidden, c=Latent)
            W_Q_absorbed = torch.einsum("hni, hnc -> hci", W_Q_nope, W_UK)
            W_Q_absorbed_quant = (
                torch.einsum("hni, hnc -> hci", W_Q_nope_quant, W_UK_quant)
                if W_Q_nope_quant is not None and W_UK_quant is not None
                else None
            )
            
            # 展平 -> [H * LatentDim, Hidden]
            W_Q_absorbed_flat = W_Q_absorbed.reshape(-1, D_in).contiguous()
            self.q_absorbed_proj = torch.nn.Linear(D_in, self.num_heads * self.kv_lora_rank, bias=False)
            self.q_absorbed_proj.weight = torch.nn.Parameter(W_Q_absorbed_flat.to(device=device, dtype=weight_dtype))
            _maybe_register_quant_weight(
                self.q_absorbed_proj,
                None if W_Q_absorbed_quant is None else W_Q_absorbed_quant.reshape(-1, D_in).to(device=device),
            )

            # ========================================================================
            # 第四步：处理 o_proj (依赖 W_UV)
            # ========================================================================
            W_O = self.o_proj.weight.to(device=device, dtype=fusion_dtype) # [Hidden, NumHeads * VDim]
            W_O_quant = getattr(self.o_proj, "quant_weight", None)
            
            # View 成 [Hidden, NumHeads, VDim]
            W_O_view = W_O.view(self.hidden_size, self.num_heads, self.v_head_dim)
            W_O_quant_view = (
                W_O_quant.to(device=device).view(self.hidden_size, self.num_heads, self.v_head_dim)
                if W_O_quant is not None
                else None
            )
            
            # 融合公式: W_Fused = W_O * W_UV
            # Einsum: "xhd, hdc -> xhc" (x=Hidden, h=Head, d=VDim, c=Latent)
            W_Fused_VO = torch.einsum("xhd,hdc->xhc", W_O_view, W_UV)
            W_Fused_VO_quant = (
                torch.einsum("xhd,hdc->xhc", W_O_quant_view, W_UV_quant)
                if W_O_quant_view is not None and W_UV_quant is not None
                else None
            )
            
            # 展平 -> [Hidden, NumHeads * LatentDim]
            new_in_features = self.num_heads * self.kv_lora_rank
            W_Fused_Flat = W_Fused_VO.reshape(self.hidden_size, new_in_features).contiguous()
            
            # 重建 Linear 层
            has_bias = self.o_proj.bias is not None
            old_bias = self.o_proj.bias
            self.o_proj = torch.nn.Linear(new_in_features, self.hidden_size, bias=has_bias)
            self.o_proj.weight = torch.nn.Parameter(W_Fused_Flat.to(device=device, dtype=weight_dtype))
            _maybe_register_quant_weight(
                self.o_proj,
                None if W_Fused_VO_quant is None else W_Fused_VO_quant.reshape(self.hidden_size, new_in_features).to(device=device),
            )
            if has_bias:
                self.o_proj.bias = (
                    torch.nn.Parameter(old_bias.to(device=device, dtype=weight_dtype))
                    if old_bias is not None
                    else None
                )

            # ========================================================================
            # 第五步：清理旧层
            # ========================================================================
            del self.q_b_proj
            del self.kv_b_proj
            del self.kv_a_proj_with_mqa

        return self


@XHLLM_TRACEABLE_MODULES.register_module(
    {
        Glm4MoeLiteDecoderLayer: "Glm4MoeLiteDecoderLayer",
    }
)
class _Glm4MoeLiteDecoderLayer(DynamicModule):
    def graph_forward(
        self,
        hidden_states: torch.Tensor,
        past_seq_length: Tensor | None = None,
        current_input_length: Tensor | None = None,
        # position_ids: Optional[torch.LongTensor] = None,
        past_k_cache: Tensor | None = None,
        past_v_cache: Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> tuple[torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor] | None]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence.
            position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            # position_ids=position_ids,
            past_seq_length=past_seq_length,
            current_input_length=current_input_length,
            past_k_cache=past_k_cache,
            past_v_cache=past_v_cache,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        if isinstance(hidden_states, tuple):
            hidden_states, router_logits = hidden_states
        else:
            router_logits = None
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        return outputs

    def _setup(self, cfg: dict | None = None):
        return self


@XHLLM_TRACEABLE_MODULES.register_module(
    {
        Glm4MoeLiteTopkRouter: "Glm4MoeLiteTopkRouter",
    }
)
class _Glm4MoeLiteTopkRouter(DynamicModule):
    def graph_forward(self, hidden_states):
        hidden_states = hidden_states.view(-1, self.config.hidden_size)
        router_logits = self.proj(hidden_states)
        return router_logits

    def _setup(self, cfg: dict | None = None):
        self.proj = nn.Linear(self.weight.shape[1], self.weight.shape[0], bias=None)
        self.proj.weight.data = self.weight.data
        return self


@XHLLM_TRACEABLE_MODULES.register_module(
    {
        Glm4MoeLiteNaiveMoe: "Glm4MoeLiteNaiveMoe",
    }
)
class _Glm4MoeLiteNaiveMoe(DynamicModule):
    def graph_forward(self, hidden_states, routing_weights=None):
        if routing_weights is None:
            raise ValueError("routing_weights must be provided for MoeBlock routing")
        out = self.moeblock(
            hidden_states,
            routing_weights,
        )
        return out

    def _setup(self, cfg: dict | None = None):
        self.input_seq_len = cfg.input_sequence_length
        self.batch_size = cfg.batch_size
        has_expert_modules = not isinstance(self.down_proj, nn.Parameter)

        def _stack_expert_attr(proj_name: str, attr_name: str) -> torch.Tensor | None:
            values = [getattr(getattr(expert, proj_name), attr_name, None) for expert in self.experts]
            if any(value is None for value in values):
                return None
            return torch.cat([value.detach().to(self.device).unsqueeze(0) for value in values], dim=0)

        if has_expert_modules:
            self.device = self.experts[0].down_proj.weight.device
            gate_weight = _stack_expert_attr("gate_proj", "weight")
            up_weight = _stack_expert_attr("up_proj", "weight")
            down_weight = _stack_expert_attr("down_proj", "weight")
            gate_quant_weight = _stack_expert_attr("gate_proj", "quant_weight")
            up_quant_weight = _stack_expert_attr("up_proj", "quant_weight")
            down_quant_weight = _stack_expert_attr("down_proj", "quant_weight")
        else:
            self.device = self.down_proj.device
            gate_weight, up_weight = self.gate_up_proj.detach().to(self.device).chunk(2, dim=1)
            down_weight = self.down_proj.detach().to(self.device)
            gate_up_quant_weight = getattr(self, "gate_up_proj_quant_weight", None)
            if gate_up_quant_weight is not None:
                gate_quant_weight, up_quant_weight = gate_up_quant_weight.detach().to(self.device).chunk(2, dim=1)
            else:
                gate_quant_weight = None
                up_quant_weight = None
            down_proj_quant_weight = getattr(self, "down_proj_quant_weight", None)
            down_quant_weight = None if down_proj_quant_weight is None else down_proj_quant_weight.detach().to(self.device)

        def _register_expert_linear(name: str, weight: torch.Tensor, quant_weight: torch.Tensor | None = None):
            setattr(
                self.moeblock,
                f"expert_{name}_weight",
                torch.nn.Parameter(weight.contiguous(), requires_grad=False),
            )
            if quant_weight is not None:
                assert quant_weight.dtype in (torch.int8, torch.int16)
                self.moeblock.register_buffer(
                    f"expert_{name}_quant_weight",
                    quant_weight.contiguous(),
                )

        self.moeblock = MoeBlock(
            self.act_fn._get_name().lower(),
            self.config.num_experts_per_tok,
            self.config.norm_topk_prob,
            topk_outside=False,
        )

        _register_expert_linear("gate_proj", gate_weight, gate_quant_weight)
        _register_expert_linear("up_proj", up_weight, up_quant_weight)
        _register_expert_linear("down_proj", down_weight, down_quant_weight)

        if hasattr(self, "gate_up_proj"):
            del self.gate_up_proj
        if hasattr(self, "down_proj"):
            del self.down_proj
        if hasattr(self, "gate_up_proj_quant_weight"):
            del self.gate_up_proj_quant_weight
        if hasattr(self, "down_proj_quant_weight"):
            del self.down_proj_quant_weight
        # The expert module list is only used to pack MoeBlock weights above.
        # Keeping it would retain an extra full copy of expert tensors.
        if hasattr(self, "experts"):
            del self.experts
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return self

    # def create_zeros(self, shape, dtype, device):
    #     return torch.zeros(shape, dtype=dtype, device=device)


@XHLLM_TRACEABLE_MODULES.register_module(
    {
        Glm4MoeLiteMoE: "Glm4MoeLiteMoE",
    }
)
class _Glm4MoeLiteMoE(DynamicModule):
    def graph_forward(self, hidden_states):
        residuals = hidden_states
        batch_size = hidden_states.size(0)
        seq_length = hidden_states.size(1)
        hidden_dim = hidden_states.size(2)

        router_logits = self.gate(hidden_states)
        routing_scores = router_logits.sigmoid().view(batch_size, seq_length, -1)
        hidden_states = self.experts(
            hidden_states,
            routing_weights=routing_scores,
        ).view(
            batch_size,
            seq_length,
            hidden_dim,
        )
        hidden_states = hidden_states * self.routed_scaling_factor
        hidden_states = hidden_states + self.shared_experts(residuals)
        return hidden_states

    def _setup(self, cfg):
        return self


@XHLLM_TRACEABLE_MODULES.register_module(
    {
        Glm4MoeLiteRMSNorm: "Glm4MoeLiteRMSNorm",
    }
)
class _Glm4MoeLiteRMSNorm(DynamicModule):
    def graph_forward(self, hidden_states):
        return self.norm(hidden_states)

    def _setup(self, cfg: dict | None = None):
        hidden_size = self.weight.shape[0]
        self.norm = RMSNorm(hidden_size, self.variance_epsilon)
        self.norm.weight = nn.Parameter(deepcopy(self.weight.data))
        return self


@XHLLM_TRACEABLE_MODULES.register_module(
    {
        Glm4MoeLiteModel: "Glm4MoeLiteModel",
    }
)
class _Glm4MoeLiteModel(DynamicModule):
    def graph_forward(
        self,
        inputs_embeds: torch.FloatTensor | None = None,
        past_seq_length: Tensor | None = None,
        current_input_length: Tensor | None = None,
        # position_ids: Optional[torch.LongTensor] = None,
        past_key_cache: list[Tensor] | None = None,
        past_value_cache: list[Tensor] | None = None,
    ) -> tuple | BaseModelOutputWithPast:

        causal_mask = None  # 在 attention 中处理
        hidden_states = inputs_embeds

        cos = self.cos_slice(self.rotary_emb.cos_cached, past_seq_length)
        sin = self.sin_slice(self.rotary_emb.sin_cached, past_seq_length)

        # cos = self.cos_embeding(position_ids)
        # sin = self.sin_embeding(position_ids)
        # cos = cos.unsqueeze(1)
        # sin = sin.unsqueeze(1)

        position_embeddings = (cos, sin)

        for idx, decoder_layer in enumerate(self.layers):
            # print("processing: ", idx)
            if self.use_cache:
                _past_k_cache = past_key_cache[idx]
                _past_v_cache = past_value_cache[idx]
            else:
                _past_k_cache = None
                _past_v_cache = None

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                past_seq_length=past_seq_length,
                current_input_length=current_input_length,
                # position_ids=position_ids,
                past_k_cache=_past_k_cache,
                past_v_cache=_past_v_cache,
                position_embeddings=position_embeddings,
            )

            hidden_states = layer_outputs[0]
            # break
            if self.only_first_block:
                break

        # hidden_states = hidden_states[
        #     :,
        #     -num_logits_to_keep:,
        # ]
        # hidden_states = hidden_states[:, :current_input_length, :]
        # hidden_states = self.llm_gather(hidden_states, current_input_length, num_logits_to_keep)
        if self.num_logits_to_keep == 0:  # for PPL task
            # hidden_states = torch_ops_xh2a_slice(hidden_states, [0], [current_input_length], [1], [1])
            # hidden_states = self.slice(
            #     hidden_states
            # )  # 此时返回的结果，含有padding,调用者需要根据current_input_length切片
            pass
        else:
            # 取最后一个token的输出
            hidden_states = self.llm_gather(hidden_states, current_input_length - 1)
        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
        )

    def _setup_cos_sin_embeding(self):
        self.rotary_emb.cos_cached
        self.rotary_emb.sin_cached

    def _setup(self, cfg: dict | None = None):
        self.batch_size = cfg.get("batch_size", 1)
        self.only_first_block = cfg.get("only_first_block", False)
        # max_seq_len = cfg.max_sequence_length
        # self.rotary_matrix_cache = RotaryMatrixCache(self.rotary_emb, max_seq_len)

        self.num_logits_to_keep = cfg.num_logits_to_keep  # 1,取最后一个token的输出，0,取所有token的输出
        assert self.num_logits_to_keep in [0, 1]

        input_seq_len = cfg.input_sequence_length
        self.slice = xhnn.Slice([0], [input_seq_len], [1], [1])

        self.llm_gather = xhnn.BatchGather(1)
        self.llm_gather.update_offset_indices(self.batch_size, input_seq_len)

        def _llm_gather_update_cfg(self: xhnn.BatchGather, cfg: dict | None = None):
            self.num_logits_to_keep = cfg.num_logits_to_keep
            input_seq_len = cfg.input_sequence_length
            batch_size = cfg.get("batch_size", 1)
            self.update_offset_indices(batch_size, input_seq_len)

        self.llm_gather._update_cfg = types.MethodType(_llm_gather_update_cfg, self.llm_gather)

        def _slice_update_cfg(self, cfg: dict | None = None):
            input_seq_len = cfg.input_sequence_length
            self.ends = [input_seq_len]

        self.slice._update_cfg = types.MethodType(_slice_update_cfg, self.slice)
        self.use_cache = cfg.use_cache

        self.sin_slice = xhnn.DynamicSlice([input_seq_len], [2], [1])
        self.cos_slice = xhnn.DynamicSlice([input_seq_len], [2], [1])

        def _sin_cos_slice_update_cfg(self, cfg: dict | None = None):
            input_seq_len = cfg.input_sequence_length
            self.valid_length = [input_seq_len]

        self.sin_slice._update_cfg = types.MethodType(_sin_cos_slice_update_cfg, self.sin_slice)
        self.cos_slice._update_cfg = types.MethodType(_sin_cos_slice_update_cfg, self.cos_slice)

        self.cos_unsqueeze = xhnn.Unsqueeze(0)
        self.sin_unsqueeze = xhnn.Unsqueeze(0)

        if not hasattr(self.rotary_emb, "cos_cached"):
            self.rotary_emb.setup_after_callback = self._setup_cos_sin_embeding
        else:
            self._setup_cos_sin_embeding()

        return self


@XHLLM_TRACEABLE_MODULES.register_module(
    {
        Glm4MoeLiteForCausalLM: "Glm4MoeLiteForCausalLM",
    }
)
class _Glm4MoeLiteForCausalLM(DynamicModule):
    def graph_forward(
        self,
        inputs_embeds: Tensor | None = None,
        past_seq_length: Tensor | None = None,
        current_input_length: Tensor | None = None,
        # position_ids: Tensor | None = None,
        past_key_cache: list[Tensor] | None = None,
        past_value_cache: list[Tensor] | None = None,
    ):
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            past_seq_length=past_seq_length,
            current_input_length=current_input_length,
            # position_ids=position_ids,
            past_key_cache=past_key_cache,
            past_value_cache=past_value_cache,
        )

        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)
        return logits

    def _setup(self, cfg: dict | None = None):
        return self



def register_wrap_modules():
    pass
