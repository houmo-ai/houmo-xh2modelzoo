# ================================================================== #
#  File: _compressor_nl.py                                            #
#  Description:                                                       #
#    DeepSeek-V4 Compressor 非线性逻辑（stateless prefill 模式）。    #
#                                                                     #
#    线性投影（kv_proj, gate_proj, q_b_proj 等）已提取为独立         #
#    nn.Linear，由标准量化流水线处理。本文件仅保留 softmax gating、  #
#    Ca/Cb overlap、RMSNorm、RoPE、indexer scoring、TopK 等非 GEMM  #
#    逻辑。                                                           #
#                                                                     #
#    所有模块为普通 nn.Module，但使用 concrete seq_len（由 _setup     #
#    从 wrap_cfg 传入）替代 tensor.shape 推导，确保 TorchFX          #
#    symbolic trace 时不产生 Proxy 类型。                             #
# ================================================================== #

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from xhquant.nn.modules.one_hot import OneHot
from xhquant.nn.modules.onnx_style_modules import (
    GreaterOrEqual,
    ReduceSum,
    Where,
)


# ================================================================== #
#  Interleaved RoPE（free function，FX trace 直接看到原子 op）       #
# ================================================================== #


def _apply_interleaved_rope(
    x: Tensor,
    cos: Tensor,
    sin: Tensor,
    rope_dim: int,
    unsqueeze_dim: int = 1,
    even_idx: Optional[Tensor] = None,
    odd_idx: Optional[Tensor] = None,
) -> Tensor:
    """Apply interleaved RoPE on the trailing rope_dim of x.

    纯 tensor op：repeat_interleave, stack, flatten, cat。
    FX trace 直接分解为原子 op，不需要 FX leaf。

    Args:
        rope_dim: rope 维度（concrete int，从 _setup 传入，避免 FX trace 时
                  从 cos.shape 推导触发 quant graph 错误）
        unsqueeze_dim: cos/sin 广播维度（1=BHSD, 2=BSHD）
    """
    # cos/sin 保持 [.., rope_dim/2]（不 repeat_interleave，避免 xh2a 量化 Tile 失真）
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    nope, rope = x[..., :-rope_dim], x[..., -rope_dim:]
    # xh2a 量化 runtime 对 step=2 strided Slice 支持不稳定，致 interleaved RoPE
    # 的 x1/x2 拆分错、rope 维度失真。提供 even_idx/odd_idx 时改用 index_select
    # （连续 gather，xh2a 原生支持）。
    if even_idx is not None:
        x1 = rope.index_select(-1, even_idx)
        x2 = rope.index_select(-1, odd_idx)
    else:
        x1, x2 = rope[..., 0::2], rope[..., 1::2]
    # interleaved RoPE per pair: out_x=x1*cos-x2*sin, out_y=x2*cos+x1*sin
    ox = x1.float() * cos - x2.float() * sin
    oy = x2.float() * cos + x1.float() * sin
    rotated = torch.stack([ox, oy], dim=-1).flatten(-2).to(x.dtype)
    return torch.cat([nope, rotated], dim=-1)


# ================================================================== #
#  RMSNorm（inline：x * rsqrt(var + eps) * weight）                   #
# ================================================================== #


def _rms_norm(x: Tensor, weight: Tensor, eps: float = 1e-6) -> Tensor:
    """Inline RMSNorm — 纯 tensor op，FX trace 可见。

    用 reciprocal(sqrt) 替代 rsqrt：xhquanttool 的 rsqrt_to_sqrt transform 会把
    rsqrt 转成 Div(1.0, sqrt)，Div 收到 python float 触发 ShapeProp AttributeError。
    """
    variance = (x * x).mean(-1, keepdim=True)
    return x * torch.reciprocal(torch.sqrt(variance + eps)) * weight


# ================================================================== #
#  Ca/Cb overlap — 统一的无分支实现                                   #
# ================================================================== #


def _ca_cb_overlap(
    chunk_kv: Tensor,
    chunk_gate: Tensor,
    head_dim: int,
    n_windows: int,
) -> tuple[Tensor, Tensor]:
    """Ca/Cb overlap via torch.cat + zero padding（无 if 分支）。

    用 cat([zeros, kv[:-1]]) 替代 if n_windows > 1 的 shift 操作。
    第一个窗口的 Ca gate 填 -inf，确保 softmax 后权重为 0。

    decode (n_windows=0) 时 padding 窗口数 = 0，cat 产生空 tensor，
    不触发维度不匹配。

    Args:
        chunk_kv:   [B, W, ratio, 2*head_dim]
        chunk_gate: [B, W, ratio, 2*head_dim]（已加 position_bias）
        n_windows:  concrete int（由调用者从 seq_len // ratio 推导，
                    不从 tensor.shape 提取，避免 FX Proxy 问题）

    Returns:
        new_kv:   [B, W, 2*ratio, head_dim]
        new_gate: [B, W, 2*ratio, head_dim]
    """

    # -- Cb (current window) --
    cb_kv = chunk_kv[..., head_dim:]  # [B, W, ratio, head_dim]
    cb_gate = chunk_gate[..., head_dim:]

    # -- Ca (prior window, shifted) --
    # n_pad = 1 (n_windows>0) 或 0 (n_windows=0, decode)
    # 避免空 tensor cat 时 dim 不匹配
    n_pad = min(1, n_windows)
    # FX symbolic trace 时 shape 返回 Proxy，torch.zeros/full/full_like 均不被
    # 量化器支持（call_function 残留会 raise）。用算术构造：x*0=0（Mul 被量化），
    # 0+(-inf)=-inf（Add 被量化）。0*(-inf)=nan 仅当 x 含 inf，此处 x 为有限激活。
    pad_ref = chunk_kv[:, :n_pad, :, :head_dim]
    zero_kv = pad_ref * 0
    neg_inf_gate = pad_ref * 0 + float("-inf")

    ca_kv = torch.cat([zero_kv, chunk_kv[:, :-1, :, :head_dim]], dim=1)
    ca_gate = torch.cat([neg_inf_gate, chunk_gate[:, :-1, :, :head_dim]], dim=1)

    # -- 合并 Ca + Cb --
    new_kv = torch.cat([ca_kv, cb_kv], dim=2)  # [B, W, 2*ratio, head_dim]
    new_gate = torch.cat([ca_gate, cb_gate], dim=2)

    return new_kv, new_gate


# ================================================================== #
#  CSA 非线性逻辑                                                     #
# ================================================================== #


class _CSANonLinear(nn.Module):
    """CSA compressor 非线性逻辑（stateless prefill 模式）。

    使用 concrete seq_len（由 Attention._setup 传入），确保 TorchFX
    symbolic trace 时 n_windows / T_idx 为具体 int，不产生 Proxy。

    输入: comp_kv, comp_gate, idx_kv, idx_gate, idx_q, idx_w, seq_len
    输出: (compressed_kv, q, idx_kv_t, idx_w_scaled)
    """

    def __init__(
        self,
        head_dim: int,
        compress_rate: int,
        position_bias: Tensor,
        kv_norm_weight: Tensor,
        idx_head_dim: int,
        idx_n_heads: int,
        idx_compress_rate: int,
        idx_index_topk: int,
        idx_position_bias: Tensor,
        idx_kv_norm_weight: Tensor,
        compress_cos_cached: Tensor,
        compress_sin_cached: Tensor,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.compress_rate = compress_rate
        self.idx_head_dim = idx_head_dim
        self.idx_n_heads = idx_n_heads
        self.idx_compress_rate = idx_compress_rate
        self.idx_index_topk = idx_index_topk
        self.softmax_scale = idx_head_dim**-0.5
        self.weights_scaling = idx_n_heads**-0.5

        # -- CSA parameters --
        self.register_buffer("position_bias", position_bias, persistent=True)
        self.register_buffer("kv_norm_weight", kv_norm_weight, persistent=True)
        self.kv_norm_eps = eps

        # -- Indexer parameters --
        self.register_buffer("idx_position_bias", idx_position_bias, persistent=True)
        self.register_buffer("idx_kv_norm_weight", idx_kv_norm_weight, persistent=True)
        self.idx_kv_norm_eps = eps

        # -- Compress RoPE caches --
        self.register_buffer("compress_cos_cached", compress_cos_cached, persistent=True)
        self.register_buffer("compress_sin_cached", compress_sin_cached, persistent=True)
        # concrete int — 避免 FX trace 时 cos.shape[-1] * 2 触发 quant graph 错误
        self.compress_rope_dim = compress_cos_cached.shape[-1] * 2

    def forward(
        self,
        comp_kv: Tensor,
        comp_gate: Tensor,
        idx_kv: Tensor,
        idx_gate: Tensor,
        idx_q: Tensor,
        idx_w: Tensor,
        seq_len: int,
        position_ids: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        batch = comp_kv.shape[0]
        head_dim = self.head_dim
        ratio = self.compress_rate

        # -- Part 1: CSA compression --
        compressed_kv = self._compress_csa(
            comp_kv,
            comp_gate,
            batch,
            seq_len,
            head_dim,
            ratio,
        )

        # -- Part 2: Indexer compression --
        compressed_idx_kv = self._compress_indexer(idx_kv, idx_gate, batch, seq_len)

        # -- Part 3: Indexer query with RoPE --
        # HF indexer 对 query 应用 compress RoPE（与 compressed KV 共享 theta）
        cos_q = self.compress_cos_cached[position_ids]  # [B, S, D/2]
        sin_q = self.compress_sin_cached[position_ids]
        q = idx_q.view(batch, seq_len, self.idx_n_heads, self.idx_head_dim).transpose(1, 2)
        q = _apply_interleaved_rope(q, cos_q, sin_q, self.compress_rope_dim, unsqueeze_dim=1)

        # -- Part 4: Prepare intermediates for scoring matmul --
        idx_kv_t = compressed_idx_kv.transpose(-1, -2).unsqueeze(1)
        idx_w_scaled = (idx_w.float() * self.weights_scaling).transpose(1, 2).to(idx_w.dtype)

        return compressed_kv, q, idx_kv_t, idx_w_scaled

    def _compress_csa(self, kv, gate, batch, seq_len, head_dim, ratio):
        """CSA compression: Ca/Cb overlap + softmax gating + RMSNorm + RoPE。"""
        usable = (seq_len // ratio) * ratio
        n_windows = usable // ratio
        # concrete dim — 避免 decode (n_windows=0) 时 -1 对空 tensor 歧义
        kv_dim = 2 * head_dim

        # 切片 + reshape（n_windows=0 时产生合法空 tensor）
        chunk_kv = kv[:, :usable].view(batch, n_windows, ratio, kv_dim)
        chunk_gate = gate[:, :usable].view(batch, n_windows, ratio, kv_dim)
        chunk_gate = chunk_gate + self.position_bias.to(chunk_gate.dtype)

        # Ca/Cb overlap（无分支，n_windows 为 concrete int）
        new_kv, new_gate = _ca_cb_overlap(chunk_kv, chunk_gate, head_dim, n_windows)

        # Softmax gating + weighted sum + RMSNorm
        weights = new_gate.softmax(dim=2, dtype=torch.float32).to(new_kv.dtype)
        compressed = _rms_norm(
            (new_kv * weights).sum(dim=2),
            self.kv_norm_weight,
            self.kv_norm_eps,
        )

        # RoPE — 用 buffer 的 device（FX trace 安全）
        positions = torch.arange(n_windows, device=self.compress_cos_cached.device).unsqueeze(0) * ratio
        cos = self.compress_cos_cached[positions]
        sin = self.compress_sin_cached[positions]
        compressed = _apply_interleaved_rope(compressed.unsqueeze(1), cos, sin, self.compress_rope_dim).squeeze(1)

        return compressed.unsqueeze(1)

    def _compress_indexer(self, kv, gate, batch, seq_len):
        """Indexer compression: same Ca/Cb pattern at idx_head_dim scale。"""
        ratio = self.idx_compress_rate
        head_dim = self.idx_head_dim
        usable = (seq_len // ratio) * ratio
        n_windows = usable // ratio
        # concrete dim — 避免 decode (n_windows=0) 时 -1 对空 tensor 歧义
        kv_dim = 2 * head_dim

        chunk_kv = kv[:, :usable].view(batch, n_windows, ratio, kv_dim)
        chunk_gate = gate[:, :usable].view(batch, n_windows, ratio, kv_dim)
        chunk_gate = chunk_gate + self.idx_position_bias.to(chunk_gate.dtype)

        new_kv, new_gate = _ca_cb_overlap(chunk_kv, chunk_gate, head_dim, n_windows)

        weights = new_gate.softmax(dim=2, dtype=torch.float32).to(new_kv.dtype)
        compressed = _rms_norm(
            (new_kv * weights).sum(dim=2),
            self.idx_kv_norm_weight,
            self.idx_kv_norm_eps,
        )

        positions = torch.arange(n_windows, device=self.compress_cos_cached.device).unsqueeze(0) * ratio
        cos = self.compress_cos_cached[positions]
        sin = self.compress_sin_cached[positions]
        compressed = _apply_interleaved_rope(compressed.unsqueeze(1), cos, sin, self.compress_rope_dim).squeeze(1)

        return compressed


# ================================================================== #
#  CSA Indexer Scoring                                                #
# ================================================================== #


class _CSAScoring(nn.Module):
    """CSA indexer scoring: ReLU + ReduceSum + TopK + block_bias。

    使用 concrete seq_len（由 _setup 传入），确保 TorchFX trace 时
    T_idx / top_k 为具体 int。

    Inputs:  scores (from quantized matmul), idx_w_scaled, compressed_kv, seq_len
    Output:  block_bias [B, 1, S, compressed_len]
    """

    def __init__(
        self,
        idx_index_topk: int,
        softmax_scale: float,
        idx_compress_rate: int = 1,
        compress_rate: int = 1,
        seq_len: int = 0,
        device: torch.device = None,
    ):
        super().__init__()
        self.idx_index_topk = idx_index_topk
        self.softmax_scale = softmax_scale
        self.idx_compress_rate = idx_compress_rate
        # concrete int — 避免 compressed_kv.shape[2] 触发量化模块包装
        self.compress_rate = compress_rate
        # dummy buffer 锚定 device，FX trace 时 tensor.device 是真正的 torch.device
        self.register_buffer("_device_anchor", torch.zeros(1, device=device), persistent=False)

        # -- onnx_style 算子（量化友好的 call_module，替代会被 _validate/dump
        #    拒绝的 call_function/method：ge/where/masked_fill/scatter/.sum）--
        compressed_len = seq_len // compress_rate
        self._where = Where()
        self._ge = GreaterOrEqual()
        self._reduce_sum_heads = ReduceSum(dim=1)
        self._reduce_sum_k = ReduceSum(dim=-2)
        # OneHot 的 num_classes 需 concrete；decode(compressed_len=0) 退化为 1
        self._one_hot = OneHot(num_classes=max(compressed_len + 1, 1))
        # sentinel: invalid 索引填 compressed_len（整数）。用 buffer + Where，
        # 避免运行时 `idx * 0 + compressed_len`——后者是 call_function Mul/Add，
        # 会被量化器误当可量化 op（int input 触发 fp16 assert；导出时量化整数
        # Add 污染 block_bias）。
        self.register_buffer(
            "_sentinel",
            torch.tensor(max(compressed_len, 0), dtype=torch.long),
            persistent=False,
        )
        # block_bias 的 fp16 zero/neg_inf reference（预计算 buffer）。
        # 避免 hit(int) 的 `* 0`/`+ 标量` 被 FX trace 成 call_function Mul/Add
        # 进而被量化（.to(fp16) 是 call_method 量化器不支持，hit 停留 int；
        # Where.expand_as 要求 condition 与 Y 同 shape，故 buffer 取 [1,1,S,C+1]）。
        if compressed_len > 0:
            _bb_shape = (1, 1, seq_len, compressed_len + 1)
            self.register_buffer(
                "_bb_zero",
                torch.zeros(_bb_shape, dtype=torch.float16),
                persistent=False,
            )
            self.register_buffer(
                "_bb_neg_inf",
                torch.full(_bb_shape, float("-inf"), dtype=torch.float16),
                persistent=False,
            )
        else:
            self.register_buffer("_bb_zero", torch.zeros(1, dtype=torch.float16), persistent=False)
            self.register_buffer("_bb_neg_inf", torch.zeros(1, dtype=torch.float16), persistent=False)

        # -- 预计算 causal mask（prefill 时 position_ids 固定） --
        # 避免运行时 // 或 torch.floor（xh2a 量化均不支持）
        if compressed_len > 0:
            position_ids = torch.arange(seq_len, device=device)
            causal_threshold = (position_ids + 1) // compress_rate
            entry_indices = torch.arange(compressed_len, device=device)
            # future_mask[s, c]: True if entry c is in the future of position s
            future_mask = entry_indices.view(1, -1) >= causal_threshold.unsqueeze(-1)
            # causal_invalid_mask: 用于 invalid index detection
            # invalid = top_k_indices >= causal_threshold
            causal_threshold_row = causal_threshold  # [S]
            self.register_buffer("_csa_future_mask", future_mask.unsqueeze(0), persistent=True)
            self.register_buffer("_csa_causal_threshold", causal_threshold_row.unsqueeze(0), persistent=True)
        else:
            self.register_buffer(
                "_csa_future_mask",
                torch.zeros(1, 1, 1, dtype=torch.bool, device=device),
                persistent=True,
            )
            self.register_buffer(
                "_csa_causal_threshold",
                torch.zeros(1, 1, dtype=torch.long, device=device),
                persistent=True,
            )

    def forward(
        self,
        scores: Tensor,
        idx_w_scaled: Tensor,
        compressed_kv: Tensor,
        seq_len: int,
        position_ids: Tensor,
    ) -> Tensor:
        # -- ReLU scoring（F.relu + 标量 Mul 被 PTQ 量化支持）--
        scores = F.relu(scores) * self.softmax_scale
        weighted = scores * idx_w_scaled.unsqueeze(-1)  # [B, H, S, T_idx]
        index_scores = self._reduce_sum_heads(weighted)  # [B, S, T_idx]

        # concrete int — TorchFX trace 安全
        T_idx = seq_len // self.idx_compress_rate
        compressed_len = seq_len // self.compress_rate
        top_k = min(self.idx_index_topk, T_idx)

        # -- Causal mask：future 位置 → -inf（Where 替代 masked_fill）--
        if compressed_len > 0:
            fut = self._csa_future_mask[:, :seq_len, :compressed_len]  # [1, S, C]
            neg = index_scores * 0 + float("-inf")  # [B, S, T_idx]
            index_scores = self._where(fut, neg, index_scores)

        top_k_indices = index_scores.topk(top_k, dim=-1)[1]  # [B, S, k]

        # -- 无效索引（>= causal_threshold，含 future）→ 哨兵 C，截断时丢弃 --
        # sentinel 用 buffer（long scalar）+ Where 广播，杜绝 `* 0 +` 的整数
        # Mul/Add（会被量化器误当可量化，污染整数 op 链）。
        if compressed_len > 0:
            ct = self._csa_causal_threshold[:, :seq_len]  # [1, S]
            invalid = self._ge(top_k_indices, ct.unsqueeze(-1))  # [B, S, k]
            safe_indices = self._where(invalid, self._sentinel, top_k_indices)
        else:
            safe_indices = top_k_indices

        # -- block_bias：命中 safe_indices 的 entry 可见(0)，否则 -inf --
        # OneHot+ReduceSum 给命中计数 hit（int）；hit 仅用于 GE(>=1)→bool。
        # zero/neg_inf 的 fp16 reference 用 compressed_kv broadcast —— 避免
        # hit(int) 的 `* 0`/`+ 标量` 被 FX trace 成 call_function Mul/Add，进而
        # 被量化器当可量化（.to(fp16) 是 call_method，量化器不支持，hit 停留 int；
        # int input 触发 fp16 assert，导出时量化整数 op 污染 block_bias）。
        oh = self._one_hot(safe_indices)  # [B, S, k, C+1]
        hit = self._reduce_sum_k(oh)  # int [B, S, C+1]
        hit_bool = self._ge(hit, 1).unsqueeze(1)  # bool [B, 1, S, C+1]
        # zero/neg_inf 用预计算 fp16 buffer（hit 仅用于 GE→bool，杜绝 int Mul/Add
        # 被量化；buffer shape [1,1,S,C+1] 匹配 hit_bool，Where.expand_as 安全）。
        block_bias = self._where(hit_bool, self._bb_zero, self._bb_neg_inf)[..., :compressed_len]

        return block_bias


# ================================================================== #
#  HCA 非线性逻辑                                                     #
# ================================================================== #


class _HCANonLinear(nn.Module):
    """HCA compressor 非线性逻辑（stateless prefill 模式）。

    直接窗口压缩 + softmax gating + causal block_bias。
    无 Ca/Cb overlap，无 indexer，无 TopK。

    使用 concrete seq_len，确保 TorchFX trace 兼容。

    输入: comp_kv[B,S,head_dim], comp_gate[B,S,head_dim], seq_len(int),
          position_ids[B,S]
    输出: (compressed_kv[B,1,T,head_dim], block_bias[B,1,S,T])
    """

    def __init__(
        self,
        head_dim: int,
        compress_rate: int,
        position_bias: Tensor,
        kv_norm_weight: Tensor,
        compress_cos_cached: Tensor,
        compress_sin_cached: Tensor,
        eps: float = 1e-6,
        seq_len: int = 0,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.compress_rate = compress_rate

        self.register_buffer("position_bias", position_bias, persistent=True)
        self.register_buffer("kv_norm_weight", kv_norm_weight, persistent=True)
        self.kv_norm_eps = eps

        self.register_buffer("compress_cos_cached", compress_cos_cached, persistent=True)
        self.register_buffer("compress_sin_cached", compress_sin_cached, persistent=True)
        # concrete int — 避免 FX trace 时 cos.shape[-1] * 2 触发 quant graph 错误
        self.compress_rope_dim = compress_cos_cached.shape[-1] * 2

        # -- 预计算 causal block_bias（prefill 模式） --
        # position_ids 在 prefill 时固定为 [0, 1, ..., seq_len-1]，
        # causal_threshold 也固定，所以可以在 __init__ 时预计算。
        # 避免运行时 // 或 torch.floor（xh2a 量化均不支持）。
        ratio = compress_rate
        n_windows = seq_len // ratio
        if seq_len > 1 and n_windows > 0:
            position_ids = torch.arange(seq_len, device=compress_cos_cached.device)
            causal_threshold = (position_ids + 1) // ratio
            entry_indices = torch.arange(n_windows, device=compress_cos_cached.device)
            bias = torch.zeros(1, 1, seq_len, n_windows, device=compress_cos_cached.device)
            bias = bias.masked_fill(
                entry_indices.view(1, 1, 1, -1) >= causal_threshold.view(1, 1, -1, 1),
                float("-inf"),
            )
            self.register_buffer("_hca_block_bias", bias, persistent=True)
        else:
            self.register_buffer(
                "_hca_block_bias",
                torch.zeros(1, 1, 1, 1, device=compress_cos_cached.device),
                persistent=True,
            )

    def forward(
        self,
        comp_kv: Tensor,
        comp_gate: Tensor,
        seq_len: int,
        position_ids: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        batch = comp_kv.shape[0]
        head_dim = self.head_dim
        ratio = self.compress_rate

        usable = (seq_len // ratio) * ratio
        n_windows = usable // ratio

        # 切片 + reshape（n_windows=0 时产生合法空 tensor）
        # HCA 无 Ca/Cb overlap，comp_kv_proj 输出维度 = head_dim
        chunk_kv = comp_kv[:, :usable].view(batch, n_windows, ratio, head_dim)
        chunk_gate = comp_gate[:, :usable].view(batch, n_windows, ratio, head_dim)
        chunk_gate = chunk_gate + self.position_bias.to(chunk_gate.dtype)

        # Softmax gating + weighted sum + RMSNorm
        weights = chunk_gate.softmax(dim=2, dtype=torch.float32).to(chunk_kv.dtype)
        compressed = _rms_norm(
            (chunk_kv * weights).sum(dim=2),
            self.kv_norm_weight,
            self.kv_norm_eps,
        )

        # RoPE — 用 buffer 的 device（FX trace 安全）
        positions = torch.arange(n_windows, device=self.compress_cos_cached.device).unsqueeze(0) * ratio
        cos = self.compress_cos_cached[positions]
        sin = self.compress_sin_cached[positions]
        compressed = _apply_interleaved_rope(compressed.unsqueeze(1), cos, sin, self.compress_rope_dim).squeeze(1)

        compressed_kv = compressed.unsqueeze(1)  # [B, 1, n_windows, head_dim]

        # -- HCA causal block_bias --
        # query t 只能看到 compressed entry w 其中 t > w * compress_rate。
        # entry_indices[w] >= (t + 1) // ratio 的位置填 -inf。
        # decode (seq_len == 1) 时无 compressed entry 或由 attention_mask 处理。
        # 用 n_windows（concrete int）而非 compressed_kv.shape[2]
        # （FX trace 时 .shape[2] 是 Proxy，不能用于控制流）
        compressed_len = n_windows
        if seq_len > 1 and compressed_len > 0:
            # causal block_bias 预计算为 buffer（__init__ 时完成）
            # 避免运行时 // 或 torch.floor（xh2a 量化均不支持）
            block_bias = self._hca_block_bias[:, :, :seq_len, :compressed_len]
        else:
            block_bias = comp_kv[:, None, :, :compressed_len] * 0
        return compressed_kv, block_bias
