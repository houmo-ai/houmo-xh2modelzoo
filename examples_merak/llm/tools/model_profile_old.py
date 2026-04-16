#!/usr/bin/env python3
"""
Qwen3-Next-80B-A3B-Instruct 模型 Profile 工具

基于 Qwen3.5/Qwen3-Next 架构代码和模型 config.json，
计算不同输入 token 数下的：
  - 各组件 FLOPs（矩阵乘法算力需求）
  - 激活参数量
  - 权重带宽需求
  - KV Cache / 递推状态内存占用
  - Roofline 分析（计算瓶颈 vs 带宽瓶颈）

架构概述:
  - 混合层结构：Linear Attention (Gated Delta Net) + Full Attention (GQA)
  - MLP 兼容两种模式：MoE 或 Dense
  - Linear Attention: Q/K/V/Z 投影 + A/G(beta) 投影 + depthwise Conv1d + 递推状态
  - Full Attention:   Q(+Z gate)/K/V/O 投影 + GQA attention score

用法:
  python profile.py --config weights/Qwen3-Next-80B-A3B-Instruct/config.json \\
                    -A 100 -B 1000 \\
                    --input-tokens 128 1024 4096 \\
                    --context-lens 1024 4096 32768 131072
"""

import argparse
import json
import os
import sys
from contextlib import redirect_stdout
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple


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
    """从 config.json 解析模型配置"""

    # 基础维度
    hidden_size: int = 2048
    num_hidden_layers: int = 48
    head_dim: int = 256
    vocab_size: int = 151936

    # Full Attention
    num_attention_heads: int = 16  # Q heads
    num_key_value_heads: int = 2  # KV heads (GQA)

    # Linear Attention (Gated Delta Net)
    linear_key_head_dim: int = 128
    linear_num_key_heads: int = 16
    linear_value_head_dim: int = 128
    linear_num_value_heads: int = 32
    linear_conv_kernel_dim: int = 4

    # MoE
    num_experts: int = 0
    num_experts_per_tok: int = 0
    moe_intermediate_size: int = 0
    shared_expert_intermediate_size: int = 0
    intermediate_size: int = 5120
    decoder_sparse_step: int = 1

    # 结构
    full_attention_interval: int = 4
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
        cfg = cls(**payload)
        if cfg.moe_intermediate_size <= 0:
            cfg.moe_intermediate_size = cfg.intermediate_size
        return cfg

    # ---- 派生属性 ----

    @property
    def is_moe(self) -> bool:
        return self.num_experts > 0 and self.num_experts_per_tok > 0

    @property
    def num_full_attn_layers(self) -> int:
        if self.layer_types_cfg and len(self.layer_types_cfg) == self.num_hidden_layers:
            return sum(1 for t in self.layer_types_cfg if t == "full_attention")
        return sum(1 for i in range(self.num_hidden_layers) if (i + 1) % self.full_attention_interval == 0)

    @property
    def num_linear_attn_layers(self) -> int:
        return self.num_hidden_layers - self.num_full_attn_layers

    @property
    def layer_types(self) -> List[str]:
        if self.layer_types_cfg and len(self.layer_types_cfg) == self.num_hidden_layers:
            return list(self.layer_types_cfg)
        return [
            "full_attention" if (i + 1) % self.full_attention_interval == 0 else "linear_attention"
            for i in range(self.num_hidden_layers)
        ]

    # Linear Attention 维度
    @property
    def lin_key_dim(self) -> int:
        """Q/K 总维度 = num_key_heads * key_head_dim"""
        return self.linear_num_key_heads * self.linear_key_head_dim  # 16*128 = 2048

    @property
    def lin_value_dim(self) -> int:
        """V/Z 总维度 = num_value_heads * value_head_dim"""
        return self.linear_num_value_heads * self.linear_value_head_dim  # 32*128 = 4096

    @property
    def lin_conv_dim(self) -> int:
        """Conv1d 通道数 = Q + K + V"""
        return self.lin_key_dim + self.lin_key_dim + self.lin_value_dim  # 2048+2048+4096 = 8192

    # Full Attention 维度
    @property
    def full_q_out_dim(self) -> int:
        """Q 输出 (不含 gate Z) = num_heads * head_dim"""
        return self.num_attention_heads * self.head_dim  # 16*256 = 4096

    @property
    def full_qz_out_dim(self) -> int:
        """q_proj 实际输出 (Q + Z gate) = num_heads * head_dim * 2"""
        return self.num_attention_heads * self.head_dim * 2  # 8192

    @property
    def full_kv_out_dim(self) -> int:
        """K/V 输出 = num_kv_heads * head_dim"""
        return self.num_key_value_heads * self.head_dim  # 2*256 = 512


# ============================================================================
# Profiler
# ============================================================================


class Qwen3NextProfiler:
    def __init__(self, config: ModelConfig, bytes_per_param: int = 2, kv_bytes_per_element: int = 1):
        self.c = config
        self.bpp = bytes_per_param  # 权重精度: bf16=2, int8=1, fp32=4
        self.kv_bpp = kv_bytes_per_element  # KV cache / 递推状态精度: int8=1, bf16=2
        self.h = config.hidden_size

    # ----------------------------------------------------------------
    # 参数量 (parameter count, 非字节)
    # ----------------------------------------------------------------

    def params_embedding(self) -> Dict[str, int]:
        return {"embed_tokens": self.c.vocab_size * self.h}

    def params_lm_head(self) -> Dict[str, int]:
        return {"lm_head": self.h * self.c.vocab_size}

    def params_final_norm(self) -> Dict[str, int]:
        return {"final_norm": self.h}

    def params_full_attn(self) -> Dict[str, int]:
        """单层 Full Attention 参数 (q k v z o + norms)"""
        c, h = self.c, self.h
        return {
            "Q_proj": h * c.full_q_out_dim,  # 2048 * 4096
            "Z_proj": h * c.full_q_out_dim,  # 2048 * 4096 (gate, 与 Q 合并在 q_proj 中)
            "K_proj": h * c.full_kv_out_dim,  # 2048 * 512
            "V_proj": h * c.full_kv_out_dim,  # 2048 * 512
            "O_proj": c.full_q_out_dim * h,  # 4096 * 2048
            "q_norm": c.head_dim,  # 256
            "k_norm": c.head_dim,  # 256
        }

    def params_linear_attn(self) -> Dict[str, int]:
        """单层 Linear Attention (Gated Delta Net) 参数"""
        c, h = self.c, self.h
        return {
            "Q_proj": h * c.lin_key_dim,  # 2048 * 2048
            "K_proj": h * c.lin_key_dim,  # 2048 * 2048
            "V_proj": h * c.lin_value_dim,  # 2048 * 4096
            "Z_proj": h * c.lin_value_dim,  # 2048 * 4096 (output gate)
            "A_proj": h * c.linear_num_value_heads,  # 2048 * 32   (alpha)
            "G_proj": h * c.linear_num_value_heads,  # 2048 * 32   (beta)
            "O_proj": c.lin_value_dim * h,  # 4096 * 2048
            "conv1d": c.lin_conv_dim * c.linear_conv_kernel_dim,  # 8192 * 4 (depthwise)
            "dt_bias": c.linear_num_value_heads,  # 32
            "A_log": c.linear_num_value_heads,  # 32
            "gated_norm": c.linear_value_head_dim,  # 128
        }

    def params_layer_norms(self) -> Dict[str, int]:
        """每层的 layernorm 参数"""
        return {
            "input_layernorm": self.h,
            "post_attn_layernorm": self.h,
        }

    def params_dense_ffn(self) -> Dict[str, int]:
        """单层 Dense FFN 参数"""
        c, h = self.c, self.h
        return {
            "gate_proj": h * c.intermediate_size,
            "up_proj": h * c.intermediate_size,
            "down_proj": c.intermediate_size * h,
        }

    def params_moe(self, activated_only: bool = False) -> Dict[str, int]:
        """单层 MoE MLP 参数"""
        c, h = self.c, self.h
        n_exp = c.num_experts_per_tok if activated_only else c.num_experts

        expert_params = (
            h * c.moe_intermediate_size  # gate_proj
            + h * c.moe_intermediate_size  # up_proj
            + c.moe_intermediate_size * h
        )  # down_proj
        shared_params = (
            h * c.shared_expert_intermediate_size
            + h * c.shared_expert_intermediate_size
            + c.shared_expert_intermediate_size * h
        )

        label = f"experts(x{n_exp})"
        return {
            "router": h * c.num_experts,  # 始终全量计算
            "shared_expert_gate": h,
            label: expert_params * n_exp,
            "shared_expert": shared_params,
        }

    def params_mlp(self, activated_only: bool = False) -> Dict[str, int]:
        if self.c.is_moe:
            return self.params_moe(activated_only=activated_only)
        return self.params_dense_ffn()

    def total_params(self) -> int:
        c = self.c
        total = 0
        total += sum(self.params_embedding().values())
        total += sum(self.params_lm_head().values())
        total += sum(self.params_final_norm().values())

        total += (
            sum(self.params_full_attn().values()) + sum(self.params_layer_norms().values())
        ) * c.num_full_attn_layers
        total += (
            sum(self.params_linear_attn().values()) + sum(self.params_layer_norms().values())
        ) * c.num_linear_attn_layers
        total += sum(self.params_mlp(activated_only=False).values()) * c.num_hidden_layers
        return total

    def activated_params(self) -> int:
        c = self.c
        if not c.is_moe:
            return self.total_params()
        total = 0
        total += sum(self.params_embedding().values())
        total += sum(self.params_lm_head().values())
        total += sum(self.params_final_norm().values())

        total += (
            sum(self.params_full_attn().values()) + sum(self.params_layer_norms().values())
        ) * c.num_full_attn_layers
        total += (
            sum(self.params_linear_attn().values()) + sum(self.params_layer_norms().values())
        ) * c.num_linear_attn_layers
        total += sum(self.params_mlp(activated_only=True).values()) * c.num_hidden_layers
        return total

    # ----------------------------------------------------------------
    # FLOPs 计算 (multiply-accumulate × 2)
    # ----------------------------------------------------------------

    @staticmethod
    def _linear_flops(in_dim: int, out_dim: int, tokens: int = 1) -> int:
        return 2 * in_dim * out_dim * tokens

    def flops_full_attn_proj(self, tokens: int = 1) -> Dict[str, int]:
        """Full Attention 线性投影 FLOPs (单层)"""
        c, h = self.c, self.h
        return {
            "Q_proj": self._linear_flops(h, c.full_q_out_dim, tokens),
            "Z_proj": self._linear_flops(h, c.full_q_out_dim, tokens),
            "K_proj": self._linear_flops(h, c.full_kv_out_dim, tokens),
            "V_proj": self._linear_flops(h, c.full_kv_out_dim, tokens),
            "O_proj": self._linear_flops(c.full_q_out_dim, h, tokens),
        }

    def flops_full_attn_score(self, tokens: int, context_len: int) -> Dict[str, int]:
        """Full Attention score 计算 FLOPs (QK^T, softmax, score@V) (单层)

        GQA: Q 有 16 heads, KV 有 2 heads, 实际计算量以 Q heads 为准
        """
        c = self.c
        n_heads = c.num_attention_heads
        d = c.head_dim
        return {
            "QK^T": 2 * n_heads * d * tokens * context_len,
            "softmax": 5 * n_heads * tokens * context_len,  # 近似
            "score@V": 2 * n_heads * d * tokens * context_len,
        }

    def flops_linear_attn_proj(self, tokens: int = 1) -> Dict[str, int]:
        """Linear Attention 线性投影 FLOPs (单层)"""
        c, h = self.c, self.h
        return {
            "Q_proj": self._linear_flops(h, c.lin_key_dim, tokens),
            "K_proj": self._linear_flops(h, c.lin_key_dim, tokens),
            "V_proj": self._linear_flops(h, c.lin_value_dim, tokens),
            "Z_proj": self._linear_flops(h, c.lin_value_dim, tokens),
            "A_proj": self._linear_flops(h, c.linear_num_value_heads, tokens),
            "G_proj": self._linear_flops(h, c.linear_num_value_heads, tokens),
            "O_proj": self._linear_flops(c.lin_value_dim, h, tokens),
        }

    def flops_linear_attn_conv(self, tokens: int = 1) -> Dict[str, int]:
        """Depthwise Conv1d FLOPs (单层)

        conv_dim=8192 channels, kernel_size=4, depthwise (groups=conv_dim)
        每个输出位置: 1 个 channel × kernel_size 次 MAC
        """
        c = self.c
        return {
            "conv1d": 2 * c.lin_conv_dim * c.linear_conv_kernel_dim * tokens,
            "silu": 4 * c.lin_conv_dim * tokens,  # 近似
        }

    def flops_linear_attn_recurrent(self, tokens: int = 1) -> Dict[str, int]:
        """Gated Delta Rule 递推计算 FLOPs (单层, decode 阶段)

        状态形状: (num_v_heads, k_head_dim, v_head_dim) = (32, 128, 128)
        每 token:
          - gate: α * S (逐元素) → n_v * k_d * v_d
          - update: β * (k ⊗ v) + add → 2 * n_v * k_d * v_d
          - output: o = q @ S     → 2 * n_v * k_d * v_d
        """
        c = self.c
        n_v = c.linear_num_value_heads
        k_d = c.linear_key_head_dim
        v_d = c.linear_value_head_dim
        state_size = n_v * k_d * v_d  # 32*128*128 = 524288

        return {
            "state_gate": state_size * tokens,  # α * S
            "state_update": 2 * state_size * tokens,  # β * (k⊗v) + add
            "state_output": 2 * state_size * tokens,  # q @ S
        }

    def flops_dense_ffn(self, tokens: int = 1) -> Dict[str, int]:
        """Dense FFN FLOPs (单层)"""
        c, h = self.c, self.h
        mid = c.intermediate_size
        return {
            "gate_proj": self._linear_flops(h, mid, tokens),
            "up_proj": self._linear_flops(h, mid, tokens),
            "silu_mul": 2 * mid * tokens,
            "down_proj": self._linear_flops(mid, h, tokens),
        }

    def flops_moe(self, tokens: int = 1, activated_only: bool = True) -> Dict[str, int]:
        """MoE MLP FLOPs (单层)"""
        c, h = self.c, self.h
        n_exp = c.num_experts_per_tok if activated_only else c.num_experts

        # 每个 expert: SiLU(gate_proj(x)) * up_proj(x) → down_proj
        gate_up = self._linear_flops(h, c.moe_intermediate_size, tokens) * 2  # gate + up
        silu_mul = 2 * c.moe_intermediate_size * tokens  # SiLU(g)*u element-wise
        down = self._linear_flops(c.moe_intermediate_size, h, tokens)
        expert_total = (gate_up + silu_mul + down) * n_exp

        # Shared expert
        s_gate_up = self._linear_flops(h, c.shared_expert_intermediate_size, tokens) * 2
        s_silu = 2 * c.shared_expert_intermediate_size * tokens
        s_down = self._linear_flops(c.shared_expert_intermediate_size, h, tokens)

        return {
            "router": self._linear_flops(h, c.num_experts, tokens),
            "shared_expert_gate": self._linear_flops(h, 1, tokens),
            f"experts(x{n_exp})": expert_total,
            "shared_expert": s_gate_up + s_silu + s_down,
        }

    def flops_mlp(self, tokens: int = 1, activated_only: bool = True) -> Dict[str, int]:
        if self.c.is_moe:
            return self.flops_moe(tokens=tokens, activated_only=activated_only)
        return self.flops_dense_ffn(tokens=tokens)

    def flops_lm_head(self, tokens: int = 1) -> Dict[str, int]:
        return {"lm_head": self._linear_flops(self.h, self.c.vocab_size, tokens)}

    # ----------------------------------------------------------------
    # KV Cache / 递推状态内存
    # ----------------------------------------------------------------

    def kv_cache_per_token_per_layer(self) -> int:
        """Full Attention: 每层每 token KV cache 字节数"""
        c = self.c
        # K: (num_kv_heads, head_dim), V: (num_kv_heads, head_dim)
        elements = 2 * c.num_key_value_heads * c.head_dim  # 2*2*256 = 1024
        return elements * self.kv_bpp

    def linear_state_per_layer(self) -> Dict[str, int]:
        """Linear Attention: 每层固定状态字节数 (与上下文长度无关)"""
        c = self.c
        conv_elements = c.lin_conv_dim * c.linear_conv_kernel_dim  # 8192*4 = 32768
        recur_elements = (
            c.linear_num_value_heads * c.linear_key_head_dim * c.linear_value_head_dim
        )  # 32*128*128 = 524288
        return {
            "conv_cache": conv_elements * self.kv_bpp,
            "recurrent_state": recur_elements * self.kv_bpp,
            "total": (conv_elements + recur_elements) * self.kv_bpp,
        }

    def total_cache_memory(self, context_len: int, batch_size: int = 1) -> Dict[str, int]:
        """总 cache/state 内存"""
        c = self.c
        kv_bytes = self.kv_cache_per_token_per_layer() * context_len * c.num_full_attn_layers
        lin_bytes = self.linear_state_per_layer()["total"] * c.num_linear_attn_layers
        return {
            "full_attn_kv_cache": kv_bytes * batch_size,
            "linear_attn_state": lin_bytes * batch_size,
            "total": (kv_bytes + lin_bytes) * batch_size,
        }

    # ----------------------------------------------------------------
    # 带宽分析 (decode 阶段需要从 HBM 读取的字节数)
    # ----------------------------------------------------------------

    def decode_weight_bytes(self) -> Dict[str, int]:
        """Decode 一次需要从 HBM 加载的权重字节 (激活的部分)"""
        c = self.c
        bpp = self.bpp

        # Full attention 权重
        full_attn_bytes = sum(self.params_full_attn().values()) * bpp * c.num_full_attn_layers
        full_norm_bytes = sum(self.params_layer_norms().values()) * bpp * c.num_full_attn_layers

        # Linear attention 权重
        lin_attn_bytes = sum(self.params_linear_attn().values()) * bpp * c.num_linear_attn_layers
        lin_norm_bytes = sum(self.params_layer_norms().values()) * bpp * c.num_linear_attn_layers

        # MLP 权重 (MoE 激活或 Dense 全量)
        mlp_bytes = sum(self.params_mlp(activated_only=True).values()) * bpp * c.num_hidden_layers
        mlp_label = "moe_weights(activated)" if c.is_moe else "dense_ffn_weights"

        # LM head
        head_bytes = sum(self.params_lm_head().values()) * bpp

        # Final norm
        final_norm_bytes = sum(self.params_final_norm().values()) * bpp

        return {
            "full_attn_weights": full_attn_bytes + full_norm_bytes,
            "linear_attn_weights": lin_attn_bytes + lin_norm_bytes,
            mlp_label: mlp_bytes,
            "lm_head": head_bytes,
            "final_norm": final_norm_bytes,
        }

    def decode_kv_io_bytes(self, context_len: int, batch_size: int = 1) -> Dict[str, int]:
        """Decode 一次 KV cache 和 state 的 I/O 字节"""
        c = self.c

        # Full attention: 读取全部 KV cache + 写入 1 token
        kv_read = self.kv_cache_per_token_per_layer() * context_len * c.num_full_attn_layers
        kv_write = self.kv_cache_per_token_per_layer() * c.num_full_attn_layers

        # Linear attention: 读 + 写 recurrent_state 和 conv_cache
        lin_state = self.linear_state_per_layer()["total"] * c.num_linear_attn_layers
        lin_rw = lin_state * 2  # 读 + 写

        return {
            "kv_cache_read": kv_read * batch_size,
            "kv_cache_write": kv_write * batch_size,
            "linear_state_rw": lin_rw * batch_size,
        }

    # ----------------------------------------------------------------
    # 逐算子 Roofline (per-operator, 无算子间重叠)
    # ----------------------------------------------------------------

    def _gemm_bytes(self, M: int, K: int, N: int) -> int:
        """GEMM Y[M,N] = X[M,K] @ W[K,N] 的内存流量 (读权重+读输入+写输出)"""
        return (K * N + M * K + M * N) * self.bpp

    def build_op_list(self, tokens: int, context_len: int, is_decode: bool = True) -> List[tuple]:
        """构建逐算子列表: [(name, flops, bytes, num_layers), ...]

        每个 tuple 表示一个算子/融合算子组:
          - flops: 单次/单层 FLOPs
          - bytes: 单次/单层内存流量 (权重 + 激活读写)
          - num_layers: 该算子重复的层数

        is_decode=True:  tokens=1, 从 KV cache 读历史
        is_decode=False: prefill, tokens=S, 自注意力 S×S
        """
        ops = []
        c, h, bpp = self.c, self.h, self.bpp
        kv_bpp = self.kv_bpp

        # ===== Full Attention 层 (×num_full_attn_layers) =====
        n_full = c.num_full_attn_layers

        # q_proj (实际包含 Q+Z gate): GEMM(tokens, h, full_qz_out_dim)
        M, K, N = tokens, h, c.full_qz_out_dim
        ops.append(("full.q_proj(+Z)", 2 * M * K * N, self._gemm_bytes(M, K, N), n_full))

        # k_proj
        M, K, N = tokens, h, c.full_kv_out_dim
        ops.append(("full.k_proj", 2 * M * K * N, self._gemm_bytes(M, K, N), n_full))

        # v_proj
        ops.append(("full.v_proj", 2 * M * K * N, self._gemm_bytes(M, K, N), n_full))

        # QK^T: (n_heads, tokens, d) @ (n_kv, d, ctx) → (n_heads, tokens, ctx)
        n_h, n_kv, d = c.num_attention_heads, c.num_key_value_heads, c.head_dim
        flops_qkt = 2 * n_h * d * tokens * context_len
        if is_decode:
            # Q: n_h*d (计算精度), K cache: n_kv*ctx*d (KV精度从HBM读), scores: n_h*ctx
            bytes_qkt = (n_h * d) * bpp + (n_kv * context_len * d) * kv_bpp + (n_h * context_len) * bpp
        else:
            # prefill: Q/K/scores 都是计算精度
            bytes_qkt = (n_h * tokens * d + n_kv * tokens * d + n_h * tokens * context_len) * bpp
        ops.append(("full.QK^T", flops_qkt, bytes_qkt, n_full))

        # softmax
        flops_sm = 5 * n_h * tokens * context_len
        bytes_sm = 2 * n_h * tokens * context_len * bpp
        ops.append(("full.softmax", flops_sm, bytes_sm, n_full))

        # score@V: (n_heads, tokens, ctx) @ (n_kv, ctx, d) → (n_heads, tokens, d)
        flops_sv = 2 * n_h * d * tokens * context_len
        if is_decode:
            # scores: n_h*ctx (计算精度), V cache: n_kv*ctx*d (KV精度), output: n_h*d
            bytes_sv = (n_h * context_len) * bpp + (n_kv * context_len * d) * kv_bpp + (n_h * d) * bpp
        else:
            bytes_sv = (n_h * tokens * context_len + n_kv * tokens * d + n_h * tokens * d) * bpp
        ops.append(("full.score@V", flops_sv, bytes_sv, n_full))

        # o_proj
        M, K, N = tokens, c.full_q_out_dim, h
        ops.append(("full.o_proj", 2 * M * K * N, self._gemm_bytes(M, K, N), n_full))

        # ===== Linear Attention 层 (×num_linear_attn_layers) =====
        n_lin = c.num_linear_attn_layers

        # in_proj_qkvz: GEMM(tokens, h, key_dim*2 + value_dim*2)
        out_qkvz = c.lin_key_dim * 2 + c.lin_value_dim * 2
        M, K, N = tokens, h, out_qkvz
        ops.append(("lin.qkvz_proj", 2 * M * K * N, self._gemm_bytes(M, K, N), n_lin))

        # in_proj_ba: GEMM(tokens, h, num_v_heads*2)
        out_ba = c.linear_num_value_heads * 2
        M, K, N = tokens, h, out_ba
        ops.append(("lin.ba_proj", 2 * M * K * N, self._gemm_bytes(M, K, N), n_lin))

        # conv1d (depthwise): channels=conv_dim, kernel=4
        conv_f = 2 * c.lin_conv_dim * c.linear_conv_kernel_dim * tokens
        conv_b = (
            c.lin_conv_dim * c.linear_conv_kernel_dim  # 权重
            + c.lin_conv_dim * tokens  # 输入
            + c.lin_conv_dim * tokens
        ) * bpp  # 输出
        ops.append(("lin.conv1d+silu", conv_f + 4 * c.lin_conv_dim * tokens, conv_b, n_lin))

        # recurrent state: state_size = n_v * k_d * v_d
        ss = c.linear_num_value_heads * c.linear_key_head_dim * c.linear_value_head_dim
        rec_f = 5 * ss * tokens  # gate + update + output
        # state read+write 使用 kv_bpp, q/k/v/z 激活使用 bpp
        rec_b = 2 * ss * kv_bpp + (c.lin_key_dim + c.lin_key_dim + c.lin_value_dim + c.lin_value_dim) * tokens * bpp
        ops.append(("lin.recurrent", rec_f, rec_b, n_lin))

        # out_proj
        M, K, N = tokens, c.lin_value_dim, h
        ops.append(("lin.o_proj", 2 * M * K * N, self._gemm_bytes(M, K, N), n_lin))

        # ===== MLP (×num_hidden_layers) =====
        n_all = c.num_hidden_layers
        if c.is_moe:
            # router
            M, K, N = tokens, h, c.num_experts
            ops.append(("moe.router", 2 * M * K * N, self._gemm_bytes(M, K, N), n_all))

            # activated experts: top-k fused (gate_proj + up_proj → silu*mul → down_proj)
            k = c.num_experts_per_tok
            mid = c.moe_intermediate_size
            # FLOPs: 每 token 只经过 top-k expert 计算
            exp_flops = tokens * k * (2 * h * mid + 2 * h * mid + 2 * mid + 2 * mid * h)  # gate+up+silu+down
            # 权重加载: decode 仅加载 top-k expert; prefill batch tokens 激活全部 expert
            n_loaded = k if is_decode else c.num_experts
            exp_bytes = n_loaded * 3 * h * mid * bpp + tokens * (h + h + mid + h) * bpp
            label = f"moe.experts(x{k})" if is_decode else f"moe.experts(all{c.num_experts})"
            ops.append((label, exp_flops, exp_bytes, n_all))

            # shared expert
            smid = c.shared_expert_intermediate_size
            se_flops = tokens * (2 * h * smid + 2 * h * smid + 2 * smid + 2 * smid * h)
            se_bytes = 3 * h * smid * bpp + tokens * (h + h + smid + h) * bpp
            ops.append(("moe.shared_expert", se_flops, se_bytes, n_all))

            # shared_expert_gate
            M, K, N = tokens, h, 1
            ops.append(("moe.shared_gate", 2 * M * K * N, self._gemm_bytes(M, K, N), n_all))
        else:
            mid = c.intermediate_size
            # gate_proj
            M, K, N = tokens, h, mid
            ops.append(("dense.gate_proj", 2 * M * K * N, self._gemm_bytes(M, K, N), n_all))
            # up_proj
            ops.append(("dense.up_proj", 2 * M * K * N, self._gemm_bytes(M, K, N), n_all))
            # silu_mul
            silu_f = 2 * mid * tokens
            silu_b = (2 * mid * tokens) * bpp
            ops.append(("dense.silu_mul", silu_f, silu_b, n_all))
            # down_proj
            M, K, N = tokens, mid, h
            ops.append(("dense.down_proj", 2 * M * K * N, self._gemm_bytes(M, K, N), n_all))

        # ===== LayerNorms =====
        ops.append(("layernorm(x2/layer)", 10 * h * tokens, 4 * h * tokens * bpp, n_all))
        ops.append(("final_norm", 5 * h * tokens, 2 * h * tokens * bpp, 1))

        # ===== LM Head =====
        M, K, N = tokens, h, c.vocab_size
        ops.append(("lm_head", 2 * M * K * N, self._gemm_bytes(M, K, N), 1))

        return ops

    def per_op_latency_ms(self, ops: List[tuple], chip_tops: float, chip_bw_gbs: float) -> List[tuple]:
        """计算逐算子延迟: [(name, flops, bytes, layers, compute_ms, bw_ms, latency_ms, bound), ...]"""
        results = []
        for name, flops, nbytes, n_layers in ops:
            c_ms = flops / (chip_tops * 1e12) * 1e3 * n_layers
            b_ms = nbytes / (chip_bw_gbs * 1e9) * 1e3 * n_layers
            lat = max(c_ms, b_ms)
            intensity = flops / nbytes if nbytes > 0 else float("inf")
            bound = "C" if c_ms >= b_ms else "M"
            results.append((name, flops * n_layers, nbytes * n_layers, n_layers, c_ms, b_ms, lat, intensity, bound))
        return results

    # ----------------------------------------------------------------
    # Prefill 带宽分析
    # ----------------------------------------------------------------

    def prefill_weight_bytes(self, input_tokens: int) -> Dict[str, int]:
        """Prefill 阶段需要从 HBM 加载的权重字节

        与 decode 的区别:
          - MoE 权重: Prefill 的 batch tokens 会激活所有 expert,
            因此必须从 HBM 加载全量 expert 权重 (不是仅 top-k).
          - 其余权重 (attention / lm_head 等) 与 decode 相同.
        """
        c = self.c
        bpp = self.bpp

        full_attn_bytes = sum(self.params_full_attn().values()) * bpp * c.num_full_attn_layers
        full_norm_bytes = sum(self.params_layer_norms().values()) * bpp * c.num_full_attn_layers

        lin_attn_bytes = sum(self.params_linear_attn().values()) * bpp * c.num_linear_attn_layers
        lin_norm_bytes = sum(self.params_layer_norms().values()) * bpp * c.num_linear_attn_layers

        # MoE: Prefill 加载全量 expert 权重
        mlp_bytes = sum(self.params_mlp(activated_only=False).values()) * bpp * c.num_hidden_layers
        mlp_label = "moe_weights(all_experts)" if c.is_moe else "dense_ffn_weights"

        head_bytes = sum(self.params_lm_head().values()) * bpp
        final_norm_bytes = sum(self.params_final_norm().values()) * bpp

        return {
            "full_attn_weights": full_attn_bytes + full_norm_bytes,
            "linear_attn_weights": lin_attn_bytes + lin_norm_bytes,
            mlp_label: mlp_bytes,
            "lm_head": head_bytes,
            "final_norm": final_norm_bytes,
        }

    def prefill_kv_io_bytes(self, input_tokens: int, batch_size: int = 1) -> Dict[str, int]:
        """Prefill 阶段 KV cache 和 state 的 I/O 字节

        - Full Attention: 写入 input_tokens 个 token 的 KV cache
        - Linear Attention: 写入 recurrent_state 和 conv_cache (固定大小)
        - 激活值 I/O: 每层每 token 读写 hidden_size (简化估算)
        """
        c = self.c

        # Full attention: 写入 KV cache
        kv_write = self.kv_cache_per_token_per_layer() * input_tokens * c.num_full_attn_layers

        # Linear attention: 写 recurrent_state 和 conv_cache
        lin_state = self.linear_state_per_layer()["total"] * c.num_linear_attn_layers

        # 激活值 I/O: 每层读入 + 写出 hidden states (2 * hidden_size * tokens * bpp)
        # Full attn score 中间结果: QK^T 矩阵 (num_heads * tokens * tokens * bpp)
        act_io = 2 * self.h * self.bpp * input_tokens * c.num_hidden_layers
        attn_scratch = c.num_attention_heads * input_tokens * input_tokens * self.bpp * c.num_full_attn_layers

        return {
            "kv_cache_write": kv_write * batch_size,
            "linear_state_write": lin_state * batch_size,
            "activation_io": act_io * batch_size,
            "attn_scratch": attn_scratch * batch_size,
        }

    # ----------------------------------------------------------------
    # 综合 Profile
    # ----------------------------------------------------------------

    def profile_decode(self, context_len: int) -> Dict[str, Dict[str, int]]:
        """单 token decode 各组件 FLOPs (所有层汇总)"""
        c = self.c
        result = {}

        # Full Attention
        fp = self.flops_full_attn_proj(tokens=1)
        fs = self.flops_full_attn_score(tokens=1, context_len=context_len)
        result["full_attn_proj"] = {k: v * c.num_full_attn_layers for k, v in fp.items()}
        result["full_attn_score"] = {k: v * c.num_full_attn_layers for k, v in fs.items()}

        # Linear Attention
        lp = self.flops_linear_attn_proj(tokens=1)
        lc = self.flops_linear_attn_conv(tokens=1)
        lr = self.flops_linear_attn_recurrent(tokens=1)
        result["linear_attn_proj"] = {k: v * c.num_linear_attn_layers for k, v in lp.items()}
        result["linear_attn_conv"] = {k: v * c.num_linear_attn_layers for k, v in lc.items()}
        result["linear_attn_recurrent"] = {k: v * c.num_linear_attn_layers for k, v in lr.items()}

        # MLP
        mlp = self.flops_mlp(tokens=1, activated_only=True)
        mlp_key = "moe_mlp" if c.is_moe else "dense_ffn"
        result[mlp_key] = {k: v * c.num_hidden_layers for k, v in mlp.items()}

        # LM Head
        result["lm_head"] = self.flops_lm_head(tokens=1)

        return result

    def profile_prefill(self, input_tokens: int) -> Dict[str, Dict[str, int]]:
        """Prefill 阶段各组件 FLOPs (所有层汇总)"""
        c = self.c
        result = {}

        # Full Attention
        fp = self.flops_full_attn_proj(tokens=input_tokens)
        # Prefill: 自注意力 QK^T 是 S×S (causal mask 不省 FLOPs)
        fs = self.flops_full_attn_score(tokens=input_tokens, context_len=input_tokens)
        result["full_attn_proj"] = {k: v * c.num_full_attn_layers for k, v in fp.items()}
        result["full_attn_score"] = {k: v * c.num_full_attn_layers for k, v in fs.items()}

        # Linear Attention (chunk-based, 与序列长度线性相关)
        lp = self.flops_linear_attn_proj(tokens=input_tokens)
        lc = self.flops_linear_attn_conv(tokens=input_tokens)
        lr = self.flops_linear_attn_recurrent(tokens=input_tokens)
        result["linear_attn_proj"] = {k: v * c.num_linear_attn_layers for k, v in lp.items()}
        result["linear_attn_conv"] = {k: v * c.num_linear_attn_layers for k, v in lc.items()}
        result["linear_attn_recurrent"] = {k: v * c.num_linear_attn_layers for k, v in lr.items()}

        # MLP
        mlp = self.flops_mlp(tokens=input_tokens, activated_only=True)
        mlp_key = "moe_mlp" if c.is_moe else "dense_ffn"
        result[mlp_key] = {k: v * c.num_hidden_layers for k, v in mlp.items()}

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
        context_lens: List[int],
        batch_size: int = 1,
    ):
        """输出硬件概览汇总表 (对应 Excel 格式)

        - 权重/KV-Cache 使用 GiB (1024 进位, 匹配显存规格)
        - 带宽使用 GB/s (1000 进位, 匹配内存规格)
        - TTFT / TPS 考虑算力利用率、带宽利用率和 C2C 倍率
        """
        GiB = 1024**3
        eff_tops = chip_tops * compute_util
        eff_bw = chip_bw_gbs * bw_util

        # 权重在显存中的总占用 (全量, 包含所有 expert)
        weight_bytes = self.total_params() * self.bpp
        weight_gib = weight_bytes / GiB

        W = 130
        sep = "=" * W

        print(f"\n{sep}")
        # 居中标题 (model_name 全 ASCII, 不需要处理 CJK 宽度)
        print(f"{model_name:^{W}s}")
        print(sep)

        # --- 硬件参数行 (含 CJK, 手动拼接保证对齐) ---
        # 行 1: 算力 + 列组标题 + 利用率标题
        print(
            f"  算力（T-FLOPS）：{chip_tops:>8.0f}    "
            f"算力需求          显存占用（GiB）"
            f"                              "
            f"算力利用率   带宽利用率   C2C倍率"
        )
        # 行 2: 带宽 + 列组标题续 + 利用率数值
        print(
            f"  带宽（GB/s）：  {chip_bw_gbs:>8.0f}    "
            f"（T-FLOPS）"
            f"                                              "
            f"    {compute_util:<12.2f}{bw_util:<12.2f}{c2c_ratio:.0f}"
        )

        # --- 列标题行 ---
        print(
            f"  {'Context Length(K)':>18s}  "
            f"{'Prefill':>12s}  {'KV-Cache':>10s}  {'Weight':>10s}  "
            f"{'Weight+KV':>10s}  {'TTFT (s)':>10s}  "
            f"{'Prefill-TPS':>14s}  {'Decode-TPS':>12s}"
        )
        print(f"  {'─' * (W - 2)}")

        for ctx in context_lens:
            ctx_k = ctx / 1024

            # --- Prefill 算力需求 (T-FLOPS) ---
            profile = self.profile_prefill(ctx)
            prefill_flops = sum(sum(ops.values()) for ops in profile.values()) * batch_size
            prefill_tflops = prefill_flops / 1e12

            # --- KV-Cache (GiB) ---
            cache = self.total_cache_memory(ctx, batch_size)
            cache_gib = cache["total"] / GiB

            # --- Weight + KV ---
            wkv_gib = weight_gib + cache_gib

            # --- TTFT: max(计算耗时, 带宽耗时) × C2C ---
            pw_bytes = sum(self.prefill_weight_bytes(ctx).values())
            pk_bytes = sum(self.prefill_kv_io_bytes(ctx, batch_size).values())
            prefill_mem = pw_bytes + pk_bytes
            compute_s = prefill_flops / (eff_tops * 1e12)
            bw_s = prefill_mem / (eff_bw * 1e9)
            ttft = max(compute_s, bw_s) * c2c_ratio
            prefill_tps = ctx * batch_size / ttft if ttft > 0 else 0

            # --- Decode TPS ---
            decode_profile = self.profile_decode(ctx)
            decode_flops = sum(sum(ops.values()) for ops in decode_profile.values()) * batch_size
            dw_bytes = sum(self.decode_weight_bytes().values())
            dk_bytes = sum(self.decode_kv_io_bytes(ctx, batch_size).values())
            decode_mem = dw_bytes + dk_bytes
            d_compute_s = decode_flops / (eff_tops * 1e12)
            d_bw_s = decode_mem / (eff_bw * 1e9)
            decode_lat = max(d_compute_s, d_bw_s) * c2c_ratio
            decode_tps = 1.0 / decode_lat if decode_lat > 0 else 0

            # --- 格式化 Context Length ---
            if ctx_k >= 1024:
                ctx_label = f"{ctx_k / 1024:.0f}M"
            elif ctx_k >= 1:
                ctx_label = f"{ctx_k:.0f}K" if ctx_k == int(ctx_k) else f"{ctx_k:.1f}K"
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
        input_tokens_list: List[int],
        context_lens: List[int],
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
        print("Qwen3-Next Model Profiler")
        print(sep)
        print(f"  模型:         {'MoE-Hybrid' if c.is_moe else 'Dense-Hybrid'}")
        print(
            f"  总层数:       {c.num_hidden_layers}  "
            f"(Linear Attention × {c.num_linear_attn_layers} + Full Attention × {c.num_full_attn_layers})"
        )
        print(f"  Hidden size:  {c.hidden_size}")
        print(f"  Vocab size:   {c.vocab_size}")
        print(f"  精度:         {bpp} Bytes/param ({'bf16' if bpp == 2 else f'{bpp * 8}bit'})")
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
        print("1. 参数量 & 激活参数")
        print(sep)

        total_p = self.total_params()
        act_p = self.activated_params()
        print(f"\n  总参数:       {fmt_num(total_p)} ({fmt_bytes(total_p * bpp)} @ {bpp}B/param)")
        print(f"  激活参数:     {fmt_num(act_p)} ({fmt_bytes(act_p * bpp)} @ {bpp}B/param)")
        print(f"  激活比例:     {act_p / total_p * 100:.2f}%")

        def _print_params(title: str, params: Dict[str, int]):
            print(f"\n  --- {title} ---")
            for k, v in params.items():
                print(f"    {k:30s}  {fmt_num(v):>12s} params  ({fmt_bytes(v * bpp):>10s})")
            print(
                f"    {'SUBTOTAL':30s}  {fmt_num(sum(params.values())):>12s} params  "
                f"({fmt_bytes(sum(params.values()) * bpp):>10s})"
            )

        _print_params(f"Full Attention (单层, ×{c.num_full_attn_layers})", self.params_full_attn())
        _print_params(f"Linear Attention (单层, ×{c.num_linear_attn_layers})", self.params_linear_attn())
        _print_params("LayerNorm (每层)", self.params_layer_norms())
        if c.is_moe:
            _print_params(f"MoE MLP 总量 (单层, ×{c.num_hidden_layers})", self.params_moe(False))
            _print_params(f"MoE MLP 激活 (单层, top-{c.num_experts_per_tok}+shared)", self.params_moe(True))
        else:
            _print_params(f"Dense FFN (单层, ×{c.num_hidden_layers})", self.params_dense_ffn())
        _print_params("Embedding", self.params_embedding())
        _print_params("LM Head", self.params_lm_head())

        # ======== 2. KV Cache / State 内存 ========
        print(f"\n{sep}")
        print("2. KV Cache / 递推状态内存 (per batch)")
        print(sep)

        kv_per_tok = self.kv_cache_per_token_per_layer()
        lin_state = self.linear_state_per_layer()

        print(f"\n  Full Attention KV Cache:")
        print(f"    每 token 每层:       {fmt_bytes(kv_per_tok)}")
        print(f"    每 token 全部 {c.num_full_attn_layers} 层:  {fmt_bytes(kv_per_tok * c.num_full_attn_layers)}")

        print(f"\n  Linear Attention 递推状态 (固定, 与上下文无关):")
        print(
            f"    conv_cache 每层:     {fmt_bytes(lin_state['conv_cache'])}  "
            f"(dim={c.lin_conv_dim}, kernel={c.linear_conv_kernel_dim})"
        )
        print(
            f"    recurrent_state 每层:{fmt_bytes(lin_state['recurrent_state'])}  "
            f"(heads={c.linear_num_value_heads}, k_dim={c.linear_key_head_dim}, v_dim={c.linear_value_head_dim})"
        )
        print(
            f"    全部 {c.num_linear_attn_layers} 层:         {fmt_bytes(lin_state['total'] * c.num_linear_attn_layers)}"
        )

        lin_total = lin_state["total"] * c.num_linear_attn_layers

        print(f"\n  {'Context Len':>12s} | {'Full Attn KV':>14s} | {'Linear State':>14s} | {'Total Cache':>14s}")
        print(f"  {'-' * 12}-+-{'-' * 14}-+-{'-' * 14}-+-{'-' * 14}")
        for ctx in context_lens:
            cache = self.total_cache_memory(ctx, batch_size=1)
            print(
                f"  {ctx:>12,d} | {fmt_bytes(cache['full_attn_kv_cache']):>14s} | "
                f"{fmt_bytes(cache['linear_attn_state']):>14s} | "
                f"{fmt_bytes(cache['total']):>14s}"
            )

        # ======== 3. Decode 分析 ========
        print(f"\n{sep}")
        print("3. Decode 分析 (单 token 生成)")
        print(sep)

        for ctx in context_lens:
            print(f"\n  ┌─ Context Length = {ctx:,d}, Batch = {batch_size}")
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

            # 带宽
            weight_bytes = self.decode_weight_bytes()
            total_weight_bytes = sum(weight_bytes.values())
            kv_io = self.decode_kv_io_bytes(ctx, batch_size)
            total_kv_io = sum(kv_io.values())
            total_bytes = total_weight_bytes + total_kv_io

            # Roofline
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
            print(f"  │  KV/State I/O:           {fmt_bytes(total_kv_io)}")
            for k, v in kv_io.items():
                print(f"  │    {k:30s}  {fmt_bytes(v):>14s}")
            print(f"  │  总内存流量:             {fmt_bytes(total_bytes)}")
            print(f"  │  算术强度:               {intensity:.2f} FLOPs/Byte  (ridge={ridge:.1f})")
            print(f"  │  计算耗时:               {fmt_time(compute_ms)}")
            print(f"  │  带宽耗时:               {fmt_time(bw_ms)}")
            print(f"  │  瓶颈:                   {bound}")
            print(f"  │  预估延迟:               {fmt_time(latency_ms)}  ({1000 / latency_ms:.1f} tokens/s)")
            print(f"  └{'─' * 70}")

        # ======== 4. Prefill 分析 (带 Roofline) ========
        print(f"\n{sep}")
        print("4. Prefill 分析")
        print(sep)

        ridge = chip_tops * 1e12 / (chip_bw_gbs * 1e9)

        for n_tok in input_tokens_list:
            print(f"\n  ┌─ Input Tokens = {n_tok:,d}, Batch = {batch_size}")
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

            # 带宽
            weight_bytes = self.prefill_weight_bytes(n_tok)
            total_weight_bytes = sum(weight_bytes.values())
            kv_io = self.prefill_kv_io_bytes(n_tok, batch_size)
            total_kv_io = sum(kv_io.values())
            total_bytes = total_weight_bytes + total_kv_io

            # Roofline
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
            print(f"  │  KV/State/Act I/O:       {fmt_bytes(total_kv_io)}")
            for k, v in kv_io.items():
                print(f"  │    {k:30s}  {fmt_bytes(v):>14s}")
            print(f"  │  总内存流量:             {fmt_bytes(total_bytes)}")
            print(f"  │  算术强度:               {intensity:.2f} FLOPs/Byte  (ridge={ridge:.1f})")
            print(f"  │  计算耗时:               {fmt_time(compute_ms)}")
            print(f"  │  带宽耗时:               {fmt_time(bw_ms)}")
            print(f"  │  瓶颈:                   {bound}")
            print(f"  │  预估延迟:               {fmt_time(latency_ms)}  ({throughput:,.0f} tok/s)")
            print(f"  └{'─' * 70}")

        # Prefill 汇总表
        print(f"\n  --- Prefill 汇总表 ---")
        print(
            f"  {'Tokens':>10s} | {'FLOPs':>14s} | {'MemTraffic':>12s} | {'Intensity':>12s} | "
            f"{'Compute':>12s} | {'BW':>12s} | {'Latency':>12s} | {'tok/s':>10s} | {'瓶颈':>8s}"
        )
        print(
            f"  {'-' * 10}-+-{'-' * 14}-+-{'-' * 12}-+-{'-' * 12}-+-"
            f"{'-' * 12}-+-{'-' * 12}-+-{'-' * 12}-+-{'-' * 10}-+-{'-' * 16}"
        )
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
            print(
                f"  {n_tok:>10,d} | {fmt_flops(bf):>14s} | {fmt_bytes(tm):>12s} | "
                f"{ai:>9.1f} F/B | {fmt_time(c_ms):>12s} | {fmt_time(b_ms):>12s} | "
                f"{fmt_time(lat):>12s} | {tps:>10,.0f} | {bd:>8s}"
            )

        # ======== 5. 逐算子 Roofline 对比 ========
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
            # 聚合
            profile = self.profile_decode(ctx)
            bf = sum(sum(o.values()) for o in profile.values()) * batch_size
            wb = self.decode_weight_bytes()
            total_wb = sum(wb.values())
            kv_io = self.decode_kv_io_bytes(ctx, batch_size)
            total_mem = total_wb + sum(kv_io.values())
            agg_ms = max(bf / (chip_tops * 1e12) * 1e3, total_mem / (chip_bw_gbs * 1e9) * 1e3)

            # 逐算子
            op_list = self.build_op_list(tokens=1, context_len=ctx, is_decode=True)
            op_results = self.per_op_latency_ms(op_list, chip_tops, chip_bw_gbs)
            perop_ms = sum(r[6] for r in op_results)

            # 最大瓶颈算子
            bottleneck = max(op_results, key=lambda r: r[6])
            diff_pct = (perop_ms - agg_ms) / agg_ms * 100 if agg_ms > 0 else 0

            print(
                f"  {ctx:>10,d} | {fmt_time(agg_ms):>14s} | {fmt_time(perop_ms):>14s} | "
                f"{diff_pct:>+6.1f}% | {bottleneck[0]:>24s}"
            )

        # 逐算子明细 (取第一个 context_len 展示)
        ctx0 = context_lens[0]
        op_list = self.build_op_list(tokens=1, context_len=ctx0, is_decode=True)
        op_results = self.per_op_latency_ms(op_list, chip_tops, chip_bw_gbs)
        print(f"\n  --- Decode 逐算子明细 (context={ctx0:,d}) ---")
        print(
            f"  {'算子':>22s} | {'FLOPs':>14s} | {'MemBytes':>12s} | {'强度':>10s} | "
            f"{'Compute':>12s} | {'BW':>12s} | {'Latency':>12s} | {'B':>1s}"
        )
        print(f"  {'-' * 22}-+-{'-' * 14}-+-{'-' * 12}-+-{'-' * 10}-+-{'-' * 12}-+-{'-' * 12}-+-{'-' * 12}-+---")
        for name, flops, nbytes, n_layers, c_ms, b_ms, lat, ai, bd in op_results:
            print(
                f"  {name:>22s} | {fmt_flops(flops):>14s} | {fmt_bytes(nbytes):>12s} | "
                f"{ai:>7.1f}F/B | {fmt_time(c_ms):>12s} | {fmt_time(b_ms):>12s} | "
                f"{fmt_time(lat):>12s} | {bd}"
            )
        total_perop = sum(r[6] for r in op_results)
        print(
            f"  {'TOTAL':>22s} |                |              |            | "
            f"             |              | {fmt_time(total_perop):>12s} |"
        )

        # Prefill
        print(f"\n  --- Prefill ---")
        print(
            f"  {'Tokens':>10s} | {'聚合延迟':>14s} | {'逐算子延迟':>14s} | {'差异':>8s} | "
            f"{'聚合tok/s':>12s} | {'逐算子tok/s':>12s} | {'最大瓶颈算子':>24s}"
        )
        print(f"  {'-' * 10}-+-{'-' * 14}-+-{'-' * 14}-+-{'-' * 8}-+-{'-' * 12}-+-{'-' * 12}-+-{'-' * 24}")
        for n_tok in input_tokens_list:
            # 聚合
            profile = self.profile_prefill(n_tok)
            bf = sum(sum(o.values()) for o in profile.values()) * batch_size
            wb = sum(self.prefill_weight_bytes(n_tok).values())
            kio = sum(self.prefill_kv_io_bytes(n_tok, batch_size).values())
            tm = wb + kio
            agg_ms = max(bf / (chip_tops * 1e12) * 1e3, tm / (chip_bw_gbs * 1e9) * 1e3)
            agg_tps = n_tok * batch_size / (agg_ms / 1e3) if agg_ms > 0 else 0

            # 逐算子
            op_list = self.build_op_list(tokens=n_tok, context_len=n_tok, is_decode=False)
            op_results = self.per_op_latency_ms(op_list, chip_tops, chip_bw_gbs)
            perop_ms = sum(r[6] for r in op_results)
            perop_tps = n_tok * batch_size / (perop_ms / 1e3) if perop_ms > 0 else 0

            bottleneck = max(op_results, key=lambda r: r[6])
            diff_pct = (perop_ms - agg_ms) / agg_ms * 100 if agg_ms > 0 else 0

            print(
                f"  {n_tok:>10,d} | {fmt_time(agg_ms):>14s} | {fmt_time(perop_ms):>14s} | "
                f"{diff_pct:>+6.1f}% | {agg_tps:>10,.0f}  | {perop_tps:>10,.0f}  | "
                f"{bottleneck[0]:>24s}"
            )

        # Prefill 逐算子明细 (取中间的 token 长度展示)
        mid_tok = input_tokens_list[len(input_tokens_list) // 2]
        op_list = self.build_op_list(tokens=mid_tok, context_len=mid_tok, is_decode=False)
        op_results = self.per_op_latency_ms(op_list, chip_tops, chip_bw_gbs)
        print(f"\n  --- Prefill 逐算子明细 (tokens={mid_tok:,d}) ---")
        print(
            f"  {'算子':>22s} | {'FLOPs':>14s} | {'MemBytes':>12s} | {'强度':>10s} | "
            f"{'Compute':>12s} | {'BW':>12s} | {'Latency':>12s} | {'B':>1s}"
        )
        print(f"  {'-' * 22}-+-{'-' * 14}-+-{'-' * 12}-+-{'-' * 10}-+-{'-' * 12}-+-{'-' * 12}-+-{'-' * 12}-+---")
        for name, flops, nbytes, n_layers, c_ms, b_ms, lat, ai, bd in op_results:
            print(
                f"  {name:>22s} | {fmt_flops(flops):>14s} | {fmt_bytes(nbytes):>12s} | "
                f"{ai:>7.1f}F/B | {fmt_time(c_ms):>12s} | {fmt_time(b_ms):>12s} | "
                f"{fmt_time(lat):>12s} | {bd}"
            )
        total_perop = sum(r[6] for r in op_results)
        print(
            f"  {'TOTAL':>22s} |                |              |            | "
            f"             |              | {fmt_time(total_perop):>12s} |"
        )

        # ======== 6. 汇总对比表 ========
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
            model_name=model_name or "Model",
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
        description="Qwen3-Next 模型 Profile 工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="weights/Qwen3-Next-80B-A3B-Instruct/config.json",
        help="模型 config.json 路径",
    )
    parser.add_argument(
        "-A",
        "--compute",
        type=float,
        default=100.0,
        help="芯片算力 (TOPS), 默认 100",
    )
    parser.add_argument(
        "-B",
        "--bandwidth",
        type=float,
        default=1000.0,
        help="芯片带宽 (GB/s), 默认 1000",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size, 默认 1",
    )
    parser.add_argument(
        "--bytes-per-param",
        type=float,
        default=2,
        help="每参数字节数 (2=bf16, 1=int8, 0.5625=w4.5), 默认 2",
    )
    parser.add_argument(
        "--kv-bits",
        type=int,
        default=8,
        help="KV cache 每元素 bit 数 (8=int8, 16=bf16), 默认 8",
    )
    parser.add_argument(
        "--input-tokens",
        type=int,
        nargs="+",
        default=[2048, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304],
        help="Prefill 输入 token 数列表 ",
    )
    parser.add_argument(
        "--context-lens",
        type=int,
        nargs="+",
        default=[2048, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304],
        help="Decode 上下文长度列表",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="输出日志文件路径 (默认自动生成带时间戳的文件名)",
    )
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="不写日志文件，直接打印到终端 (旧行为)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="模型名称 (默认从 config 路径推断)",
    )
    parser.add_argument(
        "--compute-util",
        type=float,
        default=1.0,
        help="算力利用率 (0-1), 默认 1.0",
    )
    parser.add_argument(
        "--bw-util",
        type=float,
        default=1.0,
        help="带宽利用率 (0-1), 默认 1.0",
    )
    parser.add_argument(
        "--c2c-ratio",
        type=float,
        default=1.0,
        help="C2C 倍率, 默认 1.0",
    )

    args = parser.parse_args()

    config = ModelConfig.from_json(args.config)
    kv_bytes = args.kv_bits / 8
    profiler = Qwen3NextProfiler(config, bytes_per_param=args.bytes_per_param, kv_bytes_per_element=kv_bytes)
    model_name = args.model_name or os.path.basename(os.path.dirname(args.config))

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
        # 直接打印到终端
        _run_report()
    else:
        # 写入日志文件
        if args.log_file:
            log_path = args.log_file
        else:
            os.makedirs("output", exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = os.path.basename(os.path.dirname(args.config))
            log_path = f"output/profile_{model_name}_A{args.compute}_B{args.bandwidth}_{ts}.log"

        with open(log_path, "w", encoding="utf-8") as f:
            with redirect_stdout(f):
                _run_report()

        print(f"Profile 报告已写入: {log_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
