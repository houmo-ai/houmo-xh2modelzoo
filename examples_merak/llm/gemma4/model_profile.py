#!/usr/bin/env python3
"""
Gemma4-31B-IT 模型 Profile 工具

基于 Gemma4 架构代码和模型 config.json，
计算不同输入 token 数下的：
  - 各组件 FLOPs（矩阵乘法算力需求）
  - 参数量（全量 / 激活）
  - 权重带宽需求
  - KV Cache 内存占用（异构: Sliding Window + Full Attention）
  - Roofline 分析（计算瓶颈 vs 带宽瓶颈）

架构概述:
  - 混合注意力: Sliding Window Attention (50层) + Full Attention (10层)
  - MLP: Dense (GeGLU: gate_proj + up_proj → GELU * mul → down_proj)
  - 双 Head Dim: Sliding(256) / Full(512)
  - 双 KV Heads: Sliding(16 heads) / Full(4 heads)
  - attention_k_eq_v: Full Attention 层 V 复用 K 投影 (无独立 v_proj),
                       Sliding Attention 层仍有独立 v_proj
  - tie_word_embeddings: Embedding 与 LM Head 共享权重
  - 4 层 LayerNorm: input / post_attention / pre_feedforward / post_feedforward
  - Q/K/V Norm: 注意力中的 RMSNorm (v_norm 无 scale)
  - final_logit_softcapping: tanh(logits / 30) * 30

KV Cache 特性:
  - Sliding layers: 仅缓存最近 sliding_window (1024) 个 token
  - Full layers: 缓存全部历史 token
  - 异构 cache 显著降低长上下文显存需求

用法:
  python model_profile.py --config weights/gemma-4-31B-it/config.json \\
                          -A 100 -B 1000 \\
                          --input-tokens 1024 4096 32768 \\
                          --context-lens 1024 4096 32768 131072
"""

import json
import argparse
import sys
import os
from datetime import datetime
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ============================================================================
# Utility
# ============================================================================

def fmt_num(n: float, unit: str = "", si: bool = False) -> str:
    """格式化大数字

    si=False: T/B/M/K (参数量常用)
    si=True:  T/G/M/K (FLOPs/算力 SI 标准前缀)
    """
    if si:
        prefixes = [(1e15, "P"), (1e12, "T"), (1e9, "G"), (1e6, "M"), (1e3, "K")]
    else:
        prefixes = [(1e15, "P"), (1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")]
    for threshold, suffix in prefixes:
        if abs(n) >= threshold:
            return f"{n / threshold:.2f}{suffix}{unit}"
    return f"{n:.2f}{unit}"


def fmt_flops(n: float) -> str:
    """格式化 FLOPs (使用 SI 前缀 T/G/M/K)"""
    return fmt_num(n, " FLOPs", si=True)


def fmt_bytes(n: float) -> str:
    """格式化字节数 (二进制单位, 显存以 GiB 计量)"""
    KiB, MiB, GiB, TiB = 1024, 1024**2, 1024**3, 1024**4
    for threshold, suffix in [(TiB, "TiB"), (GiB, "GiB"), (MiB, "MiB"), (KiB, "KiB")]:
        if abs(n) >= threshold:
            return f"{n / threshold:.2f} {suffix}"
    return f"{n:.0f} B"


def fmt_time(ms: float) -> str:
    if ms >= 1000:
        return f"{ms / 1000:.3f} s"
    return f"{ms:.4f} ms"


# ============================================================================
# Model Config
# ============================================================================

@dataclass
class ModelConfig:
    """从 config.json 解析 Gemma4 模型配置"""
    # 基础维度
    hidden_size: int = 5376
    num_hidden_layers: int = 60
    vocab_size: int = 262144

    # Sliding Attention
    head_dim: int = 256
    num_attention_heads: int = 32       # Q heads (all layers)
    num_key_value_heads: int = 16       # KV heads for sliding attention

    # Full Attention
    global_head_dim: int = 512
    num_global_key_value_heads: int = 4  # KV heads for full attention

    # Sliding Window
    sliding_window: int = 1024

    # MLP
    intermediate_size: int = 21504

    # Structure
    attention_k_eq_v: bool = True       # Full Attention V 复用 K; Sliding 仍有独立 v_proj
    tie_word_embeddings: bool = True    # Embedding 与 LM Head 共享权重
    final_logit_softcapping: float = 30.0

    # Layer type pattern
    layer_types_cfg: Optional[List[str]] = None

    @classmethod
    def from_json(cls, path: str) -> "ModelConfig":
        with open(path) as f:
            raw = json.load(f)
        d = raw.get("text_config", raw)
        valid = {f for f in cls.__dataclass_fields__}
        payload = {k: v for k, v in d.items() if k in valid}
        if "layer_types" in d:
            payload["layer_types_cfg"] = d["layer_types"]
        return cls(**payload)

    # ---- 派生属性 ----

    @property
    def layer_types(self) -> List[str]:
        if self.layer_types_cfg and len(self.layer_types_cfg) == self.num_hidden_layers:
            return list(self.layer_types_cfg)
        # 默认: 5 sliding + 1 full 循环往复
        return [
            "full_attention" if (i + 1) % 6 == 0
            else "sliding_attention"
            for i in range(self.num_hidden_layers)
        ]

    @property
    def num_sliding_attn_layers(self) -> int:
        return sum(1 for t in self.layer_types if t == "sliding_attention")

    @property
    def num_full_attn_layers(self) -> int:
        return sum(1 for t in self.layer_types if t == "full_attention")

    # Sliding Attention 维度
    @property
    def sliding_q_dim(self) -> int:
        """Sliding Q output = num_heads × head_dim"""
        return self.num_attention_heads * self.head_dim  # 32*256 = 8192

    @property
    def sliding_kv_dim(self) -> int:
        """Sliding K/V output = num_kv_heads × head_dim"""
        return self.num_key_value_heads * self.head_dim  # 16*256 = 4096

    # Full Attention 维度
    @property
    def full_q_dim(self) -> int:
        """Full Q output = num_heads × global_head_dim"""
        return self.num_attention_heads * self.global_head_dim  # 32*512 = 16384

    @property
    def full_kv_dim(self) -> int:
        """Full K/V output = num_global_kv_heads × global_head_dim"""
        return self.num_global_key_value_heads * self.global_head_dim  # 4*512 = 2048


# ============================================================================
# Profiler
# ============================================================================

class Gemma4Profiler:
    def __init__(self, config: ModelConfig, bytes_per_param: float = 2,
                 kv_bytes_per_element: int = 1):
        self.c = config
        self.bpp = bytes_per_param    # 权重精度: bf16=2, int8=1, fp32=4
        self.kv_bpp = kv_bytes_per_element  # KV cache 精度: int8=1, bf16=2
        self.h = config.hidden_size

    # ----------------------------------------------------------------
    # 参数量 (parameter count, 非字节)
    # ----------------------------------------------------------------

    def params_embedding(self) -> Dict[str, int]:
        return {"embed_tokens": self.c.vocab_size * self.h}

    def params_lm_head(self) -> Dict[str, int]:
        """LM Head 参数 (tie_word_embeddings=True 时与 Embedding 共享)"""
        if self.c.tie_word_embeddings:
            return {"lm_head(tied)": 0}  # 共享权重, 不额外计数
        return {"lm_head": self.h * self.c.vocab_size}

    def params_final_norm(self) -> Dict[str, int]:
        return {"final_norm": self.h}

    def params_sliding_attn(self) -> Dict[str, int]:
        """单层 Sliding Attention 参数

        注意: use_alternative_attention = attention_k_eq_v AND (NOT is_sliding)
        所以 sliding 层始终有独立 v_proj, 即使 attention_k_eq_v=True
        """
        c, h = self.c, self.h
        return {
            "Q_proj":  h * c.sliding_q_dim,      # 5376 * 8192
            "K_proj":  h * c.sliding_kv_dim,      # 5376 * 4096
            "V_proj":  h * c.sliding_kv_dim,      # 5376 * 4096 (始终存在)
            "O_proj":  c.sliding_q_dim * h,       # 8192 * 5376
            "q_norm":  c.head_dim,                # 256
            "k_norm":  c.head_dim,                # 256
        }

    def params_full_attn(self) -> Dict[str, int]:
        """单层 Full Attention 参数

        attention_k_eq_v=True 时: use_alternative_attention=True,
        v_proj=None, V 复用 K 投影
        """
        c, h = self.c, self.h
        result = {
            "Q_proj":  h * c.full_q_dim,          # 5376 * 16384
            "K_proj":  h * c.full_kv_dim,         # 5376 * 2048
            "O_proj":  c.full_q_dim * h,          # 16384 * 5376
            "q_norm":  c.global_head_dim,         # 512
            "k_norm":  c.global_head_dim,         # 512
        }
        if not c.attention_k_eq_v:
            result["V_proj"] = h * c.full_kv_dim  # 5376 * 2048
        return result

    def params_layer_norms(self) -> Dict[str, int]:
        """每层 4 个 LayerNorm 参数"""
        return {
            "input_layernorm":          self.h,
            "post_attention_layernorm": self.h,
            "pre_feedforward_layernorm":  self.h,
            "post_feedforward_layernorm": self.h,
        }

    def params_dense_ffn(self) -> Dict[str, int]:
        """单层 Dense FFN 参数 (GeGLU)"""
        c, h = self.c, self.h
        return {
            "gate_proj": h * c.intermediate_size,
            "up_proj":   h * c.intermediate_size,
            "down_proj": c.intermediate_size * h,
        }

    def total_params(self) -> int:
        c = self.c
        total = 0
        total += sum(self.params_embedding().values())
        total += sum(self.params_lm_head().values())
        total += sum(self.params_final_norm().values())
        total += (sum(self.params_sliding_attn().values()) +
                  sum(self.params_layer_norms().values())) * c.num_sliding_attn_layers
        total += (sum(self.params_full_attn().values()) +
                  sum(self.params_layer_norms().values())) * c.num_full_attn_layers
        total += sum(self.params_dense_ffn().values()) * c.num_hidden_layers
        return total

    # ----------------------------------------------------------------
    # FLOPs 计算 (multiply-accumulate × 2)
    # ----------------------------------------------------------------

    @staticmethod
    def _linear_flops(in_dim: int, out_dim: int, tokens: int = 1) -> int:
        return 2 * in_dim * out_dim * tokens

    def flops_sliding_attn_proj(self, tokens: int = 1) -> Dict[str, int]:
        """Sliding Attention 线性投影 FLOPs (单层)

        sliding 层始终有独立 v_proj (use_alternative_attention 对 sliding 层为 False)
        """
        c, h = self.c, self.h
        return {
            "Q_proj": self._linear_flops(h, c.sliding_q_dim, tokens),
            "K_proj": self._linear_flops(h, c.sliding_kv_dim, tokens),
            "V_proj": self._linear_flops(h, c.sliding_kv_dim, tokens),
            "O_proj": self._linear_flops(c.sliding_q_dim, h, tokens),
        }

    def flops_sliding_attn_score(self, tokens: int, context_len: int) -> Dict[str, int]:
        """Sliding Attention score FLOPs (QK^T, softmax, score@V) (单层)

        有效上下文长度 = min(context_len, sliding_window)
        """
        c = self.c
        effective_ctx = min(context_len, c.sliding_window)
        n_heads = c.num_attention_heads
        d = c.head_dim
        return {
            "QK^T":    2 * n_heads * d * tokens * effective_ctx,
            "softmax": 5 * n_heads * tokens * effective_ctx,
            "score@V": 2 * n_heads * d * tokens * effective_ctx,
        }

    def flops_full_attn_proj(self, tokens: int = 1) -> Dict[str, int]:
        """Full Attention 线性投影 FLOPs (单层)

        attention_k_eq_v=True 时: V 复用 K 投影, 无 V_proj FLOPs
        """
        c, h = self.c, self.h
        result = {
            "Q_proj": self._linear_flops(h, c.full_q_dim, tokens),
            "K_proj": self._linear_flops(h, c.full_kv_dim, tokens),
            "O_proj": self._linear_flops(c.full_q_dim, h, tokens),
        }
        if not c.attention_k_eq_v:
            result["V_proj"] = self._linear_flops(h, c.full_kv_dim, tokens)
        return result

    def flops_full_attn_score(self, tokens: int, context_len: int) -> Dict[str, int]:
        """Full Attention score FLOPs (QK^T, softmax, score@V) (单层)

        GQA: Q 有 32 heads, KV 有 4 heads, FLOPs 按 Q heads 算
        """
        c = self.c
        n_heads = c.num_attention_heads
        d = c.global_head_dim
        return {
            "QK^T":    2 * n_heads * d * tokens * context_len,
            "softmax": 5 * n_heads * tokens * context_len,
            "score@V": 2 * n_heads * d * tokens * context_len,
        }

    def flops_dense_ffn(self, tokens: int = 1) -> Dict[str, int]:
        """Dense FFN FLOPs (GeGLU: gate+up → gelu*mul → down) (单层)"""
        c, h = self.c, self.h
        mid = c.intermediate_size
        return {
            "gate_proj": self._linear_flops(h, mid, tokens),
            "up_proj":   self._linear_flops(h, mid, tokens),
            "gelu_mul":  2 * mid * tokens,  # GELU + element-wise mul
            "down_proj": self._linear_flops(mid, h, tokens),
        }

    def flops_lm_head(self, tokens: int = 1) -> Dict[str, int]:
        return {"lm_head": self._linear_flops(self.h, self.c.vocab_size, tokens)}

    # ----------------------------------------------------------------
    # KV Cache 内存 (异构: Sliding + Full)
    # ----------------------------------------------------------------

    def kv_cache_per_token_sliding(self) -> int:
        """Sliding Attention: 每层每 token KV cache 字节数

        Sliding 层有独立 v_proj, K/V 为不同投影.
        KV cache = 2 × kv_heads × head_dim × kv_bpp
        """
        c = self.c
        elements = 2 * c.num_key_value_heads * c.head_dim
        return elements * self.kv_bpp

    def kv_cache_per_token_full(self) -> int:
        """Full Attention: 每层每 token KV cache 字节数"""
        c = self.c
        elements = 2 * c.num_global_key_value_heads * c.global_head_dim
        return elements * self.kv_bpp

    def total_cache_memory(self, context_len: int, batch_size: int = 1) -> Dict[str, int]:
        """总 KV cache 内存

        Sliding layers: min(context_len, sliding_window) 个 token 的 cache
        Full layers: 全部 context_len 个 token 的 cache
        """
        c = self.c
        sliding_tokens = min(context_len, c.sliding_window)
        sliding_bytes = (self.kv_cache_per_token_sliding() *
                         sliding_tokens * c.num_sliding_attn_layers)
        full_bytes = (self.kv_cache_per_token_full() *
                      context_len * c.num_full_attn_layers)
        return {
            "sliding_attn_kv_cache": sliding_bytes * batch_size,
            "full_attn_kv_cache":    full_bytes * batch_size,
            "total":                 (sliding_bytes + full_bytes) * batch_size,
        }

    # ----------------------------------------------------------------
    # 带宽分析 (decode 阶段)
    # ----------------------------------------------------------------

    def decode_weight_bytes(self) -> Dict[str, int]:
        """Decode 一次需要从 HBM 加载的权重字节"""
        c = self.c
        bpp = self.bpp

        sliding_attn_bytes = sum(self.params_sliding_attn().values()) * bpp * c.num_sliding_attn_layers
        sliding_norm_bytes = sum(self.params_layer_norms().values()) * bpp * c.num_sliding_attn_layers

        full_attn_bytes = sum(self.params_full_attn().values()) * bpp * c.num_full_attn_layers
        full_norm_bytes = sum(self.params_layer_norms().values()) * bpp * c.num_full_attn_layers

        mlp_bytes = sum(self.params_dense_ffn().values()) * bpp * c.num_hidden_layers

        # LM Head: 即使 tied, 仍需从 HBM 读取做 matmul
        head_bytes = self.h * c.vocab_size * bpp

        final_norm_bytes = sum(self.params_final_norm().values()) * bpp

        return {
            "sliding_attn_weights": sliding_attn_bytes + sliding_norm_bytes,
            "full_attn_weights":    full_attn_bytes + full_norm_bytes,
            "dense_ffn_weights":    mlp_bytes,
            "lm_head":             head_bytes,
            "final_norm":          final_norm_bytes,
        }

    def decode_kv_io_bytes(self, context_len: int, batch_size: int = 1) -> Dict[str, int]:
        """Decode 一次 KV cache I/O 字节"""
        c = self.c
        sliding_tokens = min(context_len, c.sliding_window)

        # Sliding attention: 读 min(ctx, window) tokens + 写 1 token
        sliding_read = self.kv_cache_per_token_sliding() * sliding_tokens * c.num_sliding_attn_layers
        sliding_write = self.kv_cache_per_token_sliding() * c.num_sliding_attn_layers

        # Full attention: 读全部 ctx tokens + 写 1 token
        full_read = self.kv_cache_per_token_full() * context_len * c.num_full_attn_layers
        full_write = self.kv_cache_per_token_full() * c.num_full_attn_layers

        return {
            "sliding_kv_read":  sliding_read * batch_size,
            "sliding_kv_write": sliding_write * batch_size,
            "full_kv_read":     full_read * batch_size,
            "full_kv_write":    full_write * batch_size,
        }

    # ----------------------------------------------------------------
    # Prefill 带宽分析
    # ----------------------------------------------------------------

    def prefill_weight_bytes(self, input_tokens: int) -> Dict[str, int]:
        """Prefill 阶段需要从 HBM 加载的权重字节

        Dense 模型, 所有权重均需加载.
        """
        return self.decode_weight_bytes()  # Dense 模型 prefill/decode 权重相同

    def prefill_kv_io_bytes(self, input_tokens: int, batch_size: int = 1) -> Dict[str, int]:
        """Prefill 阶段 KV cache 和激活 I/O 字节"""
        c = self.c

        # Sliding attention: 写入 min(input_tokens, window) 个 KV
        sliding_tokens = min(input_tokens, c.sliding_window)
        kv_write_sliding = (self.kv_cache_per_token_sliding() *
                            sliding_tokens * c.num_sliding_attn_layers)

        # Full attention: 写入全部 input_tokens 个 KV
        kv_write_full = (self.kv_cache_per_token_full() *
                         input_tokens * c.num_full_attn_layers)

        # 激活值 I/O
        act_io = 2 * self.h * self.bpp * input_tokens * c.num_hidden_layers

        # Sliding attn scratch: num_heads * tokens * min(tokens, window)
        sliding_scratch = (c.num_attention_heads * input_tokens *
                           min(input_tokens, c.sliding_window) *
                           self.bpp * c.num_sliding_attn_layers)

        # Full attn scratch: num_heads * tokens * tokens
        full_scratch = (c.num_attention_heads * input_tokens * input_tokens *
                        self.bpp * c.num_full_attn_layers)

        return {
            "kv_cache_write_sliding": kv_write_sliding * batch_size,
            "kv_cache_write_full":    kv_write_full * batch_size,
            "activation_io":          act_io * batch_size,
            "sliding_attn_scratch":   sliding_scratch * batch_size,
            "full_attn_scratch":      full_scratch * batch_size,
        }

    # ----------------------------------------------------------------
    # 逐算子 Roofline
    # ----------------------------------------------------------------

    def _gemm_bytes(self, M: int, K: int, N: int) -> int:
        """GEMM Y[M,N] = X[M,K] @ W[K,N] 的内存流量"""
        return (K * N + M * K + M * N) * self.bpp

    def build_op_list(self, tokens: int, context_len: int,
                      is_decode: bool = True) -> list:
        """构建逐算子列表: [(name, flops, bytes, num_layers), ...]"""
        ops = []
        c, h, bpp = self.c, self.h, self.bpp
        kv_bpp = self.kv_bpp

        # ===== Sliding Attention 层 (×num_sliding_attn_layers) =====
        n_sliding = c.num_sliding_attn_layers
        effective_ctx = min(context_len, c.sliding_window) if is_decode else min(tokens, c.sliding_window)

        # q_proj
        M, K, N = tokens, h, c.sliding_q_dim
        ops.append(("slide.q_proj", 2*M*K*N, self._gemm_bytes(M, K, N), n_sliding))

        # k_proj
        M, K, N = tokens, h, c.sliding_kv_dim
        ops.append(("slide.k_proj", 2*M*K*N, self._gemm_bytes(M, K, N), n_sliding))

        # v_proj (sliding 层始终有独立 v_proj, 不受 attention_k_eq_v 影响)
        ops.append(("slide.v_proj", 2*M*K*N, self._gemm_bytes(M, K, N), n_sliding))

        # QK^T
        n_h = c.num_attention_heads
        n_kv = c.num_key_value_heads
        d = c.head_dim
        ctx_eff = effective_ctx if is_decode else min(tokens, c.sliding_window)
        flops_qkt = 2 * n_h * d * tokens * ctx_eff
        if is_decode:
            bytes_qkt = ((n_h * d) * bpp +
                         (n_kv * ctx_eff * d) * kv_bpp +
                         (n_h * ctx_eff) * bpp)
        else:
            bytes_qkt = (n_h * tokens * d + n_kv * tokens * d + n_h * tokens * ctx_eff) * bpp
        ops.append(("slide.QK^T", flops_qkt, bytes_qkt, n_sliding))

        # softmax
        flops_sm = 5 * n_h * tokens * ctx_eff
        bytes_sm = 2 * n_h * tokens * ctx_eff * bpp
        ops.append(("slide.softmax", flops_sm, bytes_sm, n_sliding))

        # score@V
        flops_sv = 2 * n_h * d * tokens * ctx_eff
        if is_decode:
            bytes_sv = ((n_h * ctx_eff) * bpp +
                        (n_kv * ctx_eff * d) * kv_bpp +
                        (n_h * d) * bpp)
        else:
            bytes_sv = (n_h * tokens * ctx_eff + n_kv * tokens * d + n_h * tokens * d) * bpp
        ops.append(("slide.score@V", flops_sv, bytes_sv, n_sliding))

        # o_proj
        M, K, N = tokens, c.sliding_q_dim, h
        ops.append(("slide.o_proj", 2*M*K*N, self._gemm_bytes(M, K, N), n_sliding))

        # ===== Full Attention 层 (×num_full_attn_layers) =====
        n_full = c.num_full_attn_layers

        # q_proj
        M, K, N = tokens, h, c.full_q_dim
        ops.append(("full.q_proj", 2*M*K*N, self._gemm_bytes(M, K, N), n_full))

        # k_proj
        M, K, N = tokens, h, c.full_kv_dim
        ops.append(("full.k_proj", 2*M*K*N, self._gemm_bytes(M, K, N), n_full))

        if not c.attention_k_eq_v:
            ops.append(("full.v_proj", 2*M*K*N, self._gemm_bytes(M, K, N), n_full))

        # QK^T (full context)
        n_h_f = c.num_attention_heads
        n_kv_f = c.num_global_key_value_heads
        d_f = c.global_head_dim
        full_ctx = context_len if is_decode else tokens
        flops_qkt_f = 2 * n_h_f * d_f * tokens * full_ctx
        if is_decode:
            bytes_qkt_f = ((n_h_f * d_f) * bpp +
                           (n_kv_f * full_ctx * d_f) * kv_bpp +
                           (n_h_f * full_ctx) * bpp)
        else:
            bytes_qkt_f = (n_h_f * tokens * d_f + n_kv_f * tokens * d_f +
                           n_h_f * tokens * full_ctx) * bpp
        ops.append(("full.QK^T", flops_qkt_f, bytes_qkt_f, n_full))

        # softmax
        flops_sm_f = 5 * n_h_f * tokens * full_ctx
        bytes_sm_f = 2 * n_h_f * tokens * full_ctx * bpp
        ops.append(("full.softmax", flops_sm_f, bytes_sm_f, n_full))

        # score@V
        flops_sv_f = 2 * n_h_f * d_f * tokens * full_ctx
        if is_decode:
            bytes_sv_f = ((n_h_f * full_ctx) * bpp +
                          (n_kv_f * full_ctx * d_f) * kv_bpp +
                          (n_h_f * d_f) * bpp)
        else:
            bytes_sv_f = (n_h_f * tokens * full_ctx + n_kv_f * tokens * d_f +
                          n_h_f * tokens * d_f) * bpp
        ops.append(("full.score@V", flops_sv_f, bytes_sv_f, n_full))

        # o_proj
        M, K, N = tokens, c.full_q_dim, h
        ops.append(("full.o_proj", 2*M*K*N, self._gemm_bytes(M, K, N), n_full))

        # ===== Dense FFN (×num_hidden_layers) =====
        n_all = c.num_hidden_layers
        mid = c.intermediate_size

        # gate_proj
        M, K, N = tokens, h, mid
        ops.append(("dense.gate_proj", 2*M*K*N, self._gemm_bytes(M, K, N), n_all))
        # up_proj
        ops.append(("dense.up_proj", 2*M*K*N, self._gemm_bytes(M, K, N), n_all))
        # gelu_mul
        gelu_f = 2 * mid * tokens
        gelu_b = 2 * mid * tokens * bpp
        ops.append(("dense.gelu_mul", gelu_f, gelu_b, n_all))
        # down_proj
        M, K, N = tokens, mid, h
        ops.append(("dense.down_proj", 2*M*K*N, self._gemm_bytes(M, K, N), n_all))

        # ===== LayerNorms (4 per layer × num_hidden_layers) =====
        ops.append(("layernorm(x4/layer)", 20 * h * tokens, 8 * h * tokens * bpp, n_all))
        ops.append(("final_norm", 5 * h * tokens, 2 * h * tokens * bpp, 1))

        # ===== LM Head =====
        M, K, N = tokens, h, c.vocab_size
        ops.append(("lm_head", 2*M*K*N, self._gemm_bytes(M, K, N), 1))

        return ops

    def per_op_latency_ms(self, ops: list,
                          chip_tops: float, chip_bw_gbs: float) -> list:
        """计算逐算子延迟"""
        results = []
        for name, flops, nbytes, n_layers in ops:
            c_ms = flops / (chip_tops * 1e12) * 1e3 * n_layers
            b_ms = nbytes / (chip_bw_gbs * 1e9) * 1e3 * n_layers
            lat = max(c_ms, b_ms)
            intensity = flops / nbytes if nbytes > 0 else float("inf")
            bound = "C" if c_ms >= b_ms else "M"
            results.append((name, flops * n_layers, nbytes * n_layers,
                            n_layers, c_ms, b_ms, lat, intensity, bound))
        return results

    # ----------------------------------------------------------------
    # 综合 Profile
    # ----------------------------------------------------------------

    def profile_decode(self, context_len: int) -> Dict[str, Dict[str, int]]:
        """单 token decode 各组件 FLOPs (所有层汇总)"""
        c = self.c
        result = {}

        # Sliding Attention
        sp = self.flops_sliding_attn_proj(tokens=1)
        ss = self.flops_sliding_attn_score(tokens=1, context_len=context_len)
        result["sliding_attn_proj"] = {k: v * c.num_sliding_attn_layers for k, v in sp.items()}
        result["sliding_attn_score"] = {k: v * c.num_sliding_attn_layers for k, v in ss.items()}

        # Full Attention
        fp = self.flops_full_attn_proj(tokens=1)
        fs = self.flops_full_attn_score(tokens=1, context_len=context_len)
        result["full_attn_proj"] = {k: v * c.num_full_attn_layers for k, v in fp.items()}
        result["full_attn_score"] = {k: v * c.num_full_attn_layers for k, v in fs.items()}

        # Dense FFN
        ffn = self.flops_dense_ffn(tokens=1)
        result["dense_ffn"] = {k: v * c.num_hidden_layers for k, v in ffn.items()}

        # LM Head
        result["lm_head"] = self.flops_lm_head(tokens=1)

        return result

    def profile_prefill(self, input_tokens: int) -> Dict[str, Dict[str, int]]:
        """Prefill 阶段各组件 FLOPs (所有层汇总)"""
        c = self.c
        result = {}

        # Sliding Attention (context = input_tokens, 但窗口限制为 sliding_window)
        sp = self.flops_sliding_attn_proj(tokens=input_tokens)
        ss = self.flops_sliding_attn_score(tokens=input_tokens, context_len=input_tokens)
        result["sliding_attn_proj"] = {k: v * c.num_sliding_attn_layers for k, v in sp.items()}
        result["sliding_attn_score"] = {k: v * c.num_sliding_attn_layers for k, v in ss.items()}

        # Full Attention (context = input_tokens, 完整自注意力)
        fp = self.flops_full_attn_proj(tokens=input_tokens)
        fs = self.flops_full_attn_score(tokens=input_tokens, context_len=input_tokens)
        result["full_attn_proj"] = {k: v * c.num_full_attn_layers for k, v in fp.items()}
        result["full_attn_score"] = {k: v * c.num_full_attn_layers for k, v in fs.items()}

        # Dense FFN
        ffn = self.flops_dense_ffn(tokens=input_tokens)
        result["dense_ffn"] = {k: v * c.num_hidden_layers for k, v in ffn.items()}

        # LM Head
        result["lm_head"] = self.flops_lm_head(tokens=input_tokens)

        return result

    # ----------------------------------------------------------------
    # 硬件概览汇总表
    # ----------------------------------------------------------------

    def print_hw_summary_table(
        self,
        model_name: str,
        chip_tops: float,
        chip_bw_gbs: float,
        compute_util: float,
        bw_util: float,
        c2c_ratio: float,
        context_lens: list,
        batch_size: int = 1,
    ):
        """输出硬件概览汇总表 (对应 Excel 格式)"""
        GiB = 1024 ** 3
        eff_tops = chip_tops * compute_util
        eff_bw = chip_bw_gbs * bw_util

        # 权重在显存中的总占用 (全量参数, HBM 中实际存储)
        weight_bytes = self.total_params() * self.bpp
        # LM Head 即使 tied 也需要在 HBM 中存在 (与 Embedding 共享同一块)
        weight_gib = weight_bytes / GiB

        W = 130
        sep = "=" * W

        print(f"\n{sep}")
        print(f"{model_name:^{W}s}")
        print(sep)

        print(
            f"  算力（T-FLOPS）：{chip_tops:>8.0f}    "
            f"算力需求          显存占用（GiB）"
            f"                              "
            f"算力利用率   带宽利用率   C2C倍率"
        )
        print(
            f"  带宽（GB/s）：  {chip_bw_gbs:>8.0f}    "
            f"（T-FLOPS）"
            f"                                              "
            f"    {compute_util:<12.2f}{bw_util:<12.2f}{c2c_ratio:.0f}"
        )

        print(
            f"  {'Context Length(K)':>18s}  "
            f"{'Prefill':>12s}  {'KV-Cache':>10s}  {'Weight':>10s}  "
            f"{'Weight+KV':>10s}  {'TTFT (s)':>10s}  "
            f"{'Prefill-TPS':>14s}  {'Decode-TPS':>12s}"
        )
        print(f"  {'─' * (W - 2)}")

        for ctx in context_lens:
            ctx_k = ctx / 1024

            # Prefill 算力需求
            profile = self.profile_prefill(ctx)
            prefill_flops = sum(sum(ops.values()) for ops in profile.values()) * batch_size
            prefill_tflops = prefill_flops / 1e12

            # KV-Cache
            cache = self.total_cache_memory(ctx, batch_size)
            cache_gib = cache["total"] / GiB

            # Weight + KV
            wkv_gib = weight_gib + cache_gib

            # TTFT
            pw_bytes = sum(self.prefill_weight_bytes(ctx).values())
            pk_bytes = sum(self.prefill_kv_io_bytes(ctx, batch_size).values())
            prefill_mem = pw_bytes + pk_bytes
            compute_s = prefill_flops / (eff_tops * 1e12)
            bw_s = prefill_mem / (eff_bw * 1e9)
            ttft = max(compute_s, bw_s) * c2c_ratio
            prefill_tps = ctx * batch_size / ttft if ttft > 0 else 0

            # Decode TPS
            decode_profile = self.profile_decode(ctx)
            decode_flops = (sum(sum(ops.values()) for ops in decode_profile.values())
                           * batch_size)
            dw_bytes = sum(self.decode_weight_bytes().values())
            dk_bytes = sum(self.decode_kv_io_bytes(ctx, batch_size).values())
            decode_mem = dw_bytes + dk_bytes
            d_compute_s = decode_flops / (eff_tops * 1e12)
            d_bw_s = decode_mem / (eff_bw * 1e9)
            decode_lat = max(d_compute_s, d_bw_s) * c2c_ratio
            decode_tps = 1.0 / decode_lat if decode_lat > 0 else 0

            if ctx_k >= 1024:
                ctx_label = f"{ctx_k / 1024:.0f}M"
            elif ctx_k >= 1:
                ctx_label = (f"{ctx_k:.0f}K" if ctx_k == int(ctx_k)
                             else f"{ctx_k:.1f}K")
            else:
                ctx_label = str(ctx)

            print(
                f"  {ctx_label:>18s}  "
                f"{prefill_tflops:>12.2f}  {cache_gib:>10.2f}  {weight_gib:>10.2f}  "
                f"{wkv_gib:>10.2f}  {ttft:>10.3f}  "
                f"{prefill_tps:>14,.0f}  {decode_tps:>12.1f}"
            )

        print(sep)

    # ----------------------------------------------------------------
    # 输出报告
    # ----------------------------------------------------------------

    def print_report(
        self,
        chip_tops: float,
        chip_bw_gbs: float,
        input_tokens_list: list,
        context_lens: list,
        batch_size: int = 1,
        model_name: str = "",
        compute_util: float = 1.0,
        bw_util: float = 1.0,
        c2c_ratio: float = 1.0,
    ):
        c = self.c
        bpp = self.bpp
        sep = "=" * 100

        print(sep)
        print("Gemma4 Model Profiler")
        print(sep)
        print(f"  模型:         Dense (Sliding Window + Full Attention)")
        print(f"  总层数:       {c.num_hidden_layers}  "
              f"(Sliding Attention × {c.num_sliding_attn_layers} + "
              f"Full Attention × {c.num_full_attn_layers})")
        print(f"  Hidden size:  {c.hidden_size}")
        print(f"  Head dim:     Sliding={c.head_dim}, Full={c.global_head_dim}")
        print(f"  KV heads:     Sliding={c.num_key_value_heads}, "
              f"Full={c.num_global_key_value_heads}")
        print(f"  Sliding window: {c.sliding_window}")
        print(f"  Vocab size:   {c.vocab_size}")
        print(f"  attention_k_eq_v: {c.attention_k_eq_v}  (Full Attn: V复用K; Sliding: 独立V)")
        print(f"  tie_word_embeddings: {c.tie_word_embeddings}")
        bpp_label = 'bf16' if bpp == 2 else f'{bpp*8}bit' if bpp >= 0.5 else f'w{bpp*8:.1f}'
        print(f"  精度:         {bpp} Bytes/param ({bpp_label})")
        kv_bits = self.kv_bpp * 8
        kv_bits_str = f"{int(kv_bits)}bit" if kv_bits == int(kv_bits) else f"{kv_bits}bit"
        print(f"  KV Cache:     {self.kv_bpp} Bytes/element ({kv_bits_str})")
        print(f"  Batch size:   {batch_size}")
        print(f"  芯片算力:     {chip_tops} TOPS")
        print(f"  芯片带宽:     {chip_bw_gbs} GB/s")
        print(f"  Ridge point:  {chip_tops * 1e12 / (chip_bw_gbs * 1e9):.1f} FLOPs/Byte")
        print()

        # ======== 1. 参数量 ========
        print(sep)
        print("1. 参数量")
        print(sep)

        total_p = self.total_params()
        print(f"\n  总参数:       {fmt_num(total_p)} ({fmt_bytes(total_p * bpp)} @ {bpp}B/param)")
        if c.tie_word_embeddings:
            print(f"  (LM Head 与 Embedding 共享权重, 不重复计数)")

        def _print_params(title: str, params: Dict[str, int]):
            print(f"\n  --- {title} ---")
            for k, v in params.items():
                print(f"    {k:32s}  {fmt_num(v):>12s} params  ({fmt_bytes(v * bpp):>10s})")
            print(f"    {'SUBTOTAL':32s}  {fmt_num(sum(params.values())):>12s} params  "
                  f"({fmt_bytes(sum(params.values()) * bpp):>10s})")

        _print_params(f"Sliding Attention (单层, ×{c.num_sliding_attn_layers})", self.params_sliding_attn())
        _print_params(f"Full Attention (单层, ×{c.num_full_attn_layers})", self.params_full_attn())
        _print_params("LayerNorm (每层, 4个)", self.params_layer_norms())
        _print_params(f"Dense FFN (单层, ×{c.num_hidden_layers})", self.params_dense_ffn())
        _print_params("Embedding", self.params_embedding())
        _print_params("LM Head", self.params_lm_head())

        # ======== 2. KV Cache 内存 ========
        print(f"\n{sep}")
        print("2. KV Cache 内存 (异构: Sliding Window + Full Attention)")
        print(sep)

        kv_sliding = self.kv_cache_per_token_sliding()
        kv_full = self.kv_cache_per_token_full()

        print(f"\n  Sliding Attention KV Cache (每层):")
        print(f"    每 token:            {fmt_bytes(kv_sliding)}  "
              f"(2 × {c.num_key_value_heads} heads × {c.head_dim} dim × {self.kv_bpp}B)")
        print(f"    最大 per 层:         {fmt_bytes(kv_sliding * c.sliding_window)}  "
              f"(window={c.sliding_window})")

        print(f"\n  Full Attention KV Cache (每层):")
        print(f"    每 token:            {fmt_bytes(kv_full)}  "
              f"(2 × {c.num_global_key_value_heads} heads × {c.global_head_dim} dim × {self.kv_bpp}B)")

        print(f"\n  {'Context Len':>12s} | {'Sliding KV':>14s} | {'Full KV':>14s} | {'Total Cache':>14s}")
        print(f"  {'-' * 12}-+-{'-' * 14}-+-{'-' * 14}-+-{'-' * 14}")
        for ctx in context_lens:
            cache = self.total_cache_memory(ctx, batch_size=1)
            print(f"  {ctx:>12,d} | {fmt_bytes(cache['sliding_attn_kv_cache']):>14s} | "
                  f"{fmt_bytes(cache['full_attn_kv_cache']):>14s} | "
                  f"{fmt_bytes(cache['total']):>14s}")

        # ======== 3. Decode 分析 ========
        print(f"\n{sep}")
        print("3. Decode 分析 (单 token 生成)")
        print(sep)

        for ctx in context_lens:
            print(f"\n  ┌─ Context Length = {ctx:,d}, Batch = {batch_size}")
            print(f"  │  (Sliding window = {c.sliding_window}, "
                  f"effective sliding ctx = {min(ctx, c.sliding_window)})")
            print(f"  │")

            profile = self.profile_decode(ctx)
            total_flops = 0
            for cat, ops in profile.items():
                cat_sum = sum(ops.values())
                total_flops += cat_sum
                print(f"  │  {cat:32s}  {fmt_flops(cat_sum):>18s}")
                for k, v in ops.items():
                    print(f"  │    {k:30s}  {fmt_flops(v):>18s}")

            batch_flops = total_flops * batch_size

            weight_bytes = self.decode_weight_bytes()
            total_weight_bytes = sum(weight_bytes.values())
            kv_io = self.decode_kv_io_bytes(ctx, batch_size)
            total_kv_io = sum(kv_io.values())
            total_bytes = total_weight_bytes + total_kv_io

            compute_ms = batch_flops / (chip_tops * 1e12) * 1e3
            bw_ms = total_bytes / (chip_bw_gbs * 1e9) * 1e3
            intensity = batch_flops / total_bytes if total_bytes > 0 else float("inf")
            ridge = chip_tops * 1e12 / (chip_bw_gbs * 1e9)
            bound = "Compute-bound ⚡" if compute_ms > bw_ms else "Memory-bound 📦"
            latency_ms = max(compute_ms, bw_ms)

            print(f"  │")
            print(f"  │  总 FLOPs:               {fmt_flops(batch_flops)}")
            print(f"  │  权重加载:               {fmt_bytes(total_weight_bytes)}")
            for k, v in weight_bytes.items():
                print(f"  │    {k:30s}  {fmt_bytes(v):>14s}")
            print(f"  │  KV Cache I/O:           {fmt_bytes(total_kv_io)}")
            for k, v in kv_io.items():
                print(f"  │    {k:30s}  {fmt_bytes(v):>14s}")
            print(f"  │  总内存流量:             {fmt_bytes(total_bytes)}")
            print(f"  │  算术强度:               {intensity:.2f} FLOPs/Byte  (ridge={ridge:.1f})")
            print(f"  │  计算耗时:               {fmt_time(compute_ms)}")
            print(f"  │  带宽耗时:               {fmt_time(bw_ms)}")
            print(f"  │  瓶颈:                   {bound}")
            print(f"  │  预估延迟:               {fmt_time(latency_ms)}  "
                  f"({1000 / latency_ms:.1f} tokens/s)")
            print(f"  └{'─' * 70}")

        # ======== 4. Prefill 分析 ========
        print(f"\n{sep}")
        print("4. Prefill 分析")
        print(sep)

        ridge = chip_tops * 1e12 / (chip_bw_gbs * 1e9)

        for n_tok in input_tokens_list:
            print(f"\n  ┌─ Input Tokens = {n_tok:,d}, Batch = {batch_size}")
            print(f"  │  (Sliding attn: {c.num_sliding_attn_layers}层 × window={c.sliding_window}, "
                  f"Full attn: {c.num_full_attn_layers}层 × ctx={n_tok})")
            print(f"  │")

            profile = self.profile_prefill(n_tok)
            total_flops = 0
            grand_total = sum(sum(ops.values()) for ops in profile.values())
            for cat, ops in profile.items():
                cat_sum = sum(ops.values())
                total_flops += cat_sum
                pct = cat_sum / grand_total * 100 if grand_total > 0 else 0
                print(f"  │  {cat:32s}  {fmt_flops(cat_sum):>18s}  ({pct:5.1f}%)")
                for k, v in ops.items():
                    print(f"  │    {k:30s}  {fmt_flops(v):>18s}")

            batch_flops = total_flops * batch_size

            weight_bytes = self.prefill_weight_bytes(n_tok)
            total_weight_bytes = sum(weight_bytes.values())
            kv_io = self.prefill_kv_io_bytes(n_tok, batch_size)
            total_kv_io = sum(kv_io.values())
            total_bytes = total_weight_bytes + total_kv_io

            compute_ms = batch_flops / (chip_tops * 1e12) * 1e3
            bw_ms = total_bytes / (chip_bw_gbs * 1e9) * 1e3
            intensity = batch_flops / total_bytes if total_bytes > 0 else float("inf")
            bound = "Compute-bound ⚡" if compute_ms > bw_ms else "Memory-bound 📦"
            latency_ms = max(compute_ms, bw_ms)
            throughput = n_tok * batch_size / (latency_ms / 1e3)

            print(f"  │")
            print(f"  │  总 FLOPs:               {fmt_flops(batch_flops)}")
            print(f"  │  权重加载:               {fmt_bytes(total_weight_bytes)}")
            for k, v in weight_bytes.items():
                print(f"  │    {k:30s}  {fmt_bytes(v):>14s}")
            print(f"  │  KV/Act I/O:             {fmt_bytes(total_kv_io)}")
            for k, v in kv_io.items():
                print(f"  │    {k:30s}  {fmt_bytes(v):>14s}")
            print(f"  │  总内存流量:             {fmt_bytes(total_bytes)}")
            print(f"  │  算术强度:               {intensity:.2f} FLOPs/Byte  (ridge={ridge:.1f})")
            print(f"  │  计算耗时:               {fmt_time(compute_ms)}")
            print(f"  │  带宽耗时:               {fmt_time(bw_ms)}")
            print(f"  │  瓶颈:                   {bound}")
            print(f"  │  预估延迟:               {fmt_time(latency_ms)}  "
                  f"({throughput:,.0f} tok/s)")
            print(f"  └{'─' * 70}")

        # Prefill 汇总表
        print(f"\n  --- Prefill 汇总表 ---")
        print(f"  {'Tokens':>10s} | {'FLOPs':>14s} | {'MemTraffic':>12s} | {'Intensity':>12s} | "
              f"{'Compute':>12s} | {'BW':>12s} | {'Latency':>12s} | {'tok/s':>10s} | {'瓶颈':>8s}")
        print(f"  {'-' * 10}-+-{'-' * 14}-+-{'-' * 12}-+-{'-' * 12}-+-"
              f"{'-' * 12}-+-{'-' * 12}-+-{'-' * 12}-+-{'-' * 10}-+-{'-' * 16}")
        for n_tok in input_tokens_list:
            profile = self.profile_prefill(n_tok)
            bf = sum(sum(ops.values()) for ops in profile.values()) * batch_size
            wb = sum(self.prefill_weight_bytes(n_tok).values())
            kio = sum(self.prefill_kv_io_bytes(n_tok, batch_size).values())
            tm = wb + kio
            c_ms = bf / (chip_tops * 1e12) * 1e3
            b_ms = tm / (chip_bw_gbs * 1e9) * 1e3
            lat = max(c_ms, b_ms)
            tps = n_tok * batch_size / (lat / 1e3)
            ai = bf / tm if tm > 0 else float("inf")
            bd = "Compute" if c_ms > b_ms else "MemBound"
            print(f"  {n_tok:>10,d} | {fmt_flops(bf):>14s} | {fmt_bytes(tm):>12s} | "
                  f"{ai:>9.1f} F/B | {fmt_time(c_ms):>12s} | {fmt_time(b_ms):>12s} | "
                  f"{fmt_time(lat):>12s} | {tps:>10,.0f} | {bd:>8s}")

        # ======== 5. 逐算子 Roofline ========
        print(f"\n{sep}")
        print("5. 逐算子 Roofline 分析 (聚合 vs 逐算子, 无算子间重叠)")
        print(sep)
        print(f"\n  说明: 聚合方式把所有 FLOPs 和所有内存流量混合后取 max(计算, 带宽)")
        print(f"        逐算子方式对每个算子独立取 max，再求和 (更贴近实际)")
        print(f"        差异 = (逐算子 - 聚合) / 聚合 × 100%\n")

        # Decode
        print(f"  --- Decode (单 token) ---")
        print(f"  {'Context':>10s} | {'聚合延迟':>14s} | {'逐算子延迟':>14s} | {'差异':>8s} | {'最大瓶颈算子':>24s}")
        print(f"  {'-' * 10}-+-{'-' * 14}-+-{'-' * 14}-+-{'-' * 8}-+-{'-' * 24}")
        for ctx in context_lens:
            profile = self.profile_decode(ctx)
            bf = sum(sum(o.values()) for o in profile.values()) * batch_size
            wb = self.decode_weight_bytes()
            total_wb = sum(wb.values())
            kv_io = self.decode_kv_io_bytes(ctx, batch_size)
            total_mem = total_wb + sum(kv_io.values())
            agg_ms = max(bf / (chip_tops * 1e12) * 1e3, total_mem / (chip_bw_gbs * 1e9) * 1e3)

            op_list = self.build_op_list(tokens=1, context_len=ctx, is_decode=True)
            op_results = self.per_op_latency_ms(op_list, chip_tops, chip_bw_gbs)
            perop_ms = sum(r[6] for r in op_results)

            bottleneck = max(op_results, key=lambda r: r[6])
            diff_pct = (perop_ms - agg_ms) / agg_ms * 100 if agg_ms > 0 else 0

            print(f"  {ctx:>10,d} | {fmt_time(agg_ms):>14s} | {fmt_time(perop_ms):>14s} | "
                  f"{diff_pct:>+6.1f}% | {bottleneck[0]:>24s}")

        # 逐算子明细 (取第一个 context_len)
        ctx0 = context_lens[0]
        op_list = self.build_op_list(tokens=1, context_len=ctx0, is_decode=True)
        op_results = self.per_op_latency_ms(op_list, chip_tops, chip_bw_gbs)
        print(f"\n  --- Decode 逐算子明细 (context={ctx0:,d}) ---")
        print(f"  {'算子':>22s} | {'FLOPs':>14s} | {'MemBytes':>12s} | {'强度':>10s} | "
              f"{'Compute':>12s} | {'BW':>12s} | {'Latency':>12s} | {'B':>1s}")
        print(f"  {'-' * 22}-+-{'-' * 14}-+-{'-' * 12}-+-{'-' * 10}-+-"
              f"{'-' * 12}-+-{'-' * 12}-+-{'-' * 12}-+---")
        for name, flops, nbytes, n_layers, c_ms, b_ms, lat, ai, bd in op_results:
            print(f"  {name:>22s} | {fmt_flops(flops):>14s} | {fmt_bytes(nbytes):>12s} | "
                  f"{ai:>7.1f}F/B | {fmt_time(c_ms):>12s} | {fmt_time(b_ms):>12s} | "
                  f"{fmt_time(lat):>12s} | {bd}")
        total_perop = sum(r[6] for r in op_results)
        print(f"  {'TOTAL':>22s} |                |              |            | "
              f"             |              | {fmt_time(total_perop):>12s} |")

        # Prefill
        print(f"\n  --- Prefill ---")
        print(f"  {'Tokens':>10s} | {'聚合延迟':>14s} | {'逐算子延迟':>14s} | {'差异':>8s} | "
              f"{'聚合tok/s':>12s} | {'逐算子tok/s':>12s} | {'最大瓶颈算子':>24s}")
        print(f"  {'-' * 10}-+-{'-' * 14}-+-{'-' * 14}-+-{'-' * 8}-+-"
              f"{'-' * 12}-+-{'-' * 12}-+-{'-' * 24}")
        for n_tok in input_tokens_list:
            profile = self.profile_prefill(n_tok)
            bf = sum(sum(o.values()) for o in profile.values()) * batch_size
            wb = sum(self.prefill_weight_bytes(n_tok).values())
            kio = sum(self.prefill_kv_io_bytes(n_tok, batch_size).values())
            tm = wb + kio
            agg_ms = max(bf / (chip_tops * 1e12) * 1e3, tm / (chip_bw_gbs * 1e9) * 1e3)
            agg_tps = n_tok * batch_size / (agg_ms / 1e3) if agg_ms > 0 else 0

            op_list = self.build_op_list(tokens=n_tok, context_len=n_tok, is_decode=False)
            op_results = self.per_op_latency_ms(op_list, chip_tops, chip_bw_gbs)
            perop_ms = sum(r[6] for r in op_results)
            perop_tps = n_tok * batch_size / (perop_ms / 1e3) if perop_ms > 0 else 0

            bottleneck = max(op_results, key=lambda r: r[6])
            diff_pct = (perop_ms - agg_ms) / agg_ms * 100 if agg_ms > 0 else 0

            print(f"  {n_tok:>10,d} | {fmt_time(agg_ms):>14s} | {fmt_time(perop_ms):>14s} | "
                  f"{diff_pct:>+6.1f}% | {agg_tps:>10,.0f}  | {perop_tps:>10,.0f}  | "
                  f"{bottleneck[0]:>24s}")

        # Prefill 逐算子明细
        mid_tok = input_tokens_list[len(input_tokens_list) // 2]
        op_list = self.build_op_list(tokens=mid_tok, context_len=mid_tok, is_decode=False)
        op_results = self.per_op_latency_ms(op_list, chip_tops, chip_bw_gbs)
        print(f"\n  --- Prefill 逐算子明细 (tokens={mid_tok:,d}) ---")
        print(f"  {'算子':>22s} | {'FLOPs':>14s} | {'MemBytes':>12s} | {'强度':>10s} | "
              f"{'Compute':>12s} | {'BW':>12s} | {'Latency':>12s} | {'B':>1s}")
        print(f"  {'-' * 22}-+-{'-' * 14}-+-{'-' * 12}-+-{'-' * 10}-+-"
              f"{'-' * 12}-+-{'-' * 12}-+-{'-' * 12}-+---")
        for name, flops, nbytes, n_layers, c_ms, b_ms, lat, ai, bd in op_results:
            print(f"  {name:>22s} | {fmt_flops(flops):>14s} | {fmt_bytes(nbytes):>12s} | "
                  f"{ai:>7.1f}F/B | {fmt_time(c_ms):>12s} | {fmt_time(b_ms):>12s} | "
                  f"{fmt_time(lat):>12s} | {bd}")
        total_perop = sum(r[6] for r in op_results)
        print(f"  {'TOTAL':>22s} |                |              |            | "
              f"             |              | {fmt_time(total_perop):>12s} |")

        # ======== 6. Decode 延迟汇总表 ========
        print(f"\n{sep}")
        print("6. Decode 延迟汇总表 (ms)")
        print(sep)

        header = f"  {'Context':>10s}"
        for bs in [1, batch_size] if batch_size > 1 else [1]:
            header += f" | {'BS=' + str(bs) + ' latency':>14s} {'tok/s':>8s}"
        print(f"\n{header}")
        print(f"  {'-' * 10}" + ("-+-" + "-" * 14 + " " + "-" * 8) * (2 if batch_size > 1 else 1))

        for ctx in context_lens:
            row = f"  {ctx:>10,d}"
            for bs in [1, batch_size] if batch_size > 1 else [1]:
                profile = self.profile_decode(ctx)
                bf = sum(sum(ops.values()) for ops in profile.values()) * bs
                wb = self.decode_weight_bytes()
                total_wb = sum(wb.values())
                kv_io = self.decode_kv_io_bytes(ctx, bs)
                total_mem = total_wb + sum(kv_io.values())
                c_ms = bf / (chip_tops * 1e12) * 1e3
                b_ms = total_mem / (chip_bw_gbs * 1e9) * 1e3
                lat = max(c_ms, b_ms)
                row += f" | {lat:>11.4f} ms {1000 / lat:>7.1f}"
            print(row)

        # ======== 7. 硬件概览汇总表 ========
        self.print_hw_summary_table(
            model_name=model_name or "Gemma4-31B",
            chip_tops=chip_tops,
            chip_bw_gbs=chip_bw_gbs,
            compute_util=compute_util,
            bw_util=bw_util,
            c2c_ratio=c2c_ratio,
            context_lens=context_lens,
            batch_size=batch_size,
        )

        print(f"\n{sep}")
        print("Done.")
        print(sep)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Gemma4 模型 Profile 工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config", type=str,
        default="weights/gemma-4-31B-it/config.json",
        help="模型 config.json 路径",
    )
    parser.add_argument(
        "-A", "--compute", type=float, default=100.0,
        help="芯片算力 (TOPS), 默认 100",
    )
    parser.add_argument(
        "-B", "--bandwidth", type=float, default=1000.0,
        help="芯片带宽 (GB/s), 默认 1000",
    )
    parser.add_argument(
        "--batch-size", type=int, default=1,
        help="Batch size, 默认 1",
    )
    parser.add_argument(
        "--bytes-per-param", type=float, default=2,
        help="每参数字节数 (2=bf16, 1=int8, 0.5625=w4.5), 默认 2",
    )
    parser.add_argument(
        "--kv-bits", type=int, default=8,
        help="KV cache 每元素 bit 数 (8=int8, 16=bf16), 默认 8",
    )
    parser.add_argument(
        "--input-tokens", type=int, nargs="+",
        default=[2048, 8192, 16384, 32768, 65536, 131072, 262144, 262144*2, 262144*4, 262144*8, 262144*16],
        help="Prefill 输入 token 数列表",
    )
    parser.add_argument(
        "--context-lens", type=int, nargs="+",
        default=[2048, 8192, 16384, 32768, 65536, 131072, 262144, 262144*2, 262144*4, 262144*8, 262144*16],
        help="Decode 上下文长度列表",
    )
    parser.add_argument(
        "--log-file", type=str, default=None,
        help="输出日志文件路径 (默认自动生成带时间戳的文件名)",
    )
    parser.add_argument(
        "--no-log", action="store_true",
        help="不写日志文件，直接打印到终端",
    )
    parser.add_argument(
        "--model-name", type=str, default=None,
        help="模型名称 (默认从 config 路径推断)",
    )
    parser.add_argument(
        "--compute-util", type=float, default=1.0,
        help="算力利用率 (0-1), 默认 1.0",
    )
    parser.add_argument(
        "--bw-util", type=float, default=1.0,
        help="带宽利用率 (0-1), 默认 1.0",
    )
    parser.add_argument(
        "--c2c-ratio", type=float, default=1.0,
        help="C2C 倍率, 默认 1.0",
    )

    args = parser.parse_args()

    config = ModelConfig.from_json(args.config)
    kv_bytes = args.kv_bits / 8
    profiler = Gemma4Profiler(config, bytes_per_param=args.bytes_per_param,
                              kv_bytes_per_element=kv_bytes)
    model_name = (args.model_name
                  or os.path.basename(os.path.dirname(args.config)))

    def _run_report():
        profiler.print_report(
            chip_tops=args.compute,
            chip_bw_gbs=args.bandwidth,
            input_tokens_list=args.input_tokens,
            context_lens=args.context_lens,
            batch_size=args.batch_size,
            model_name=model_name,
            compute_util=args.compute_util,
            bw_util=args.bw_util,
            c2c_ratio=args.c2c_ratio,
        )

    if args.no_log:
        _run_report()
    else:
        if args.log_file:
            log_path = args.log_file
        else:
            os.makedirs("output", exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path = f"output/profile_{model_name}_A{args.compute}_B{args.bandwidth}_{ts}.log"

        with open(log_path, "w", encoding="utf-8") as f:
            with redirect_stdout(f):
                _run_report()

        print(f"Profile 报告已写入: {log_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
