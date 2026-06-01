"""MiniCPM4 模块的 xh2modelzoo wrap 层。

VoxCPM2 里有多个 MiniCPMModel 实例(base_lm / residual_lm / locenc 内部 /
locdit 内部),本文件统一对 MiniCPM4 的核心模块进行 wrap 注册:

- `_MiniCPMRMSNorm`        → xhnn.RMSNorm
- `_MiniCPMLongRoPE`       → 预计算 cos/sin cache + DynamicSlice
- `_MiniCPMAttention`      → GQA attention,支持 prefill/decode 两种图
- `_MiniCPMDecoderLayer`   → 保持 residual 结构(VoxCPM2 use_mup=False)
- `_MiniCPMModel`          → 顶层,负责 embed_tokens (Identity or lookup)、
                              循环调 layers、norm、可选 gather 最后 token

注意这一套 wrap 只给 **带 KV cache 的 causal LM**(base_lm / residual_lm) 用。
locenc / locdit 内部的 MiniCPMModel 走的是 non-causal + no-cache,单独处理
(见 _voxcpm2_llm_model.py 的 LocEnc/LocDiT wrap)。
"""

from __future__ import annotations

import math
import sys
import types
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

from xhquant import nn as xhnn
from xhquant.api import ConfigDict
from xhquant.nn import LLMCache, MaskedSoftmax, RMSNorm, Rope
from xhquant.utils.registry import DynamicModule

# VoxCPM2 的 MiniCPM4 原始实现
from voxcpm.modules.minicpm4.model import (
    MiniCPMAttention,
    MiniCPMDecoderLayer,
    MiniCPMLongRoPE,
    MiniCPMMLP,
    MiniCPMModel,
    MiniCPMRMSNorm,
)

# xh_model_zoo 的注册表
from ..builder import XHLLM_TRACEABLE_MODULES


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

@XHLLM_TRACEABLE_MODULES.register_module(
    {MiniCPMRMSNorm: "MiniCPMRMSNorm"}
)
class _MiniCPMRMSNorm(DynamicModule):
    """MiniCPM 的 RMSNorm wrap,用 xhquant 的等价实现替换。"""

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.norm(hidden_states)

    def _setup(self, cfg: Optional[Dict] = None):
        hidden_size = self.weight.shape[0]
        self.norm = RMSNorm(hidden_size, self.variance_epsilon)
        self.norm.weight = nn.Parameter(deepcopy(self.weight.data))
        return self


# ---------------------------------------------------------------------------
# LongRoPE
# ---------------------------------------------------------------------------

@XHLLM_TRACEABLE_MODULES.register_module(
    {MiniCPMLongRoPE: "MiniCPMLongRoPE"}
)
class _MiniCPMLongRoPE(DynamicModule):
    """MiniCPMLongRoPE wrap:导出阶段把 cos/sin cache 固化为 [1,1,seq,dim] buffer。

    原生实现按 position_ids 索引一张 [max_pos, dim] 的表。导出时我们把这张表
    在 `_setup` 里预先生成好,运行时通过 `xhnn.DynamicSlice` 按 past_seq_length
    / current_input_length 切片,让 HMONNX 后端看到静态 shape。
    """

    def _setup(self, cfg: Optional[Dict] = None):
        # 原生的 _set_cos_sin_cache 已经在 HF 加载时执行过一次,这里重新按目标 dtype 做。
        # 注意:VoxCPM2 使用了全长 max_position_embeddings=32768,但实际部署的
        # prefill + cache_length 远小于此,导出 cache 只需保留到
        # cfg.max_sequence_length 即可。
        target_max_len = cfg.get("max_sequence_length", self.max_seq_len_cached) if cfg else self.max_seq_len_cached
        target_max_len = min(int(target_max_len), int(self.max_seq_len_cached))

        # 复用原生 _set_cos_sin_cache 在同 dtype 下重算一次,保证数值一致
        dtype = self.inv_freq.dtype
        self._set_cos_sin_cache(seq_len=target_max_len, device=self.inv_freq.device, dtype=dtype)

        # 把 cos/sin cache reshape 成 [1, 1, seq, head_dim] 方便切片广播
        cos_cached = self.cos_cached.to(dtype).unsqueeze(0).unsqueeze(0)  # [1,1,seq,head_dim]
        sin_cached = self.sin_cached.to(dtype).unsqueeze(0).unsqueeze(0)
        # 重新注册为 buffer,避免和原生属性冲突
        self.register_buffer("cos_cached_4d", cos_cached, persistent=False)
        self.register_buffer("sin_cached_4d", sin_cached, persistent=False)
        return self

    def _set_dtype(self, dtype: torch.dtype) -> None:
        self.inv_freq = self.inv_freq.to(dtype)
        # 重算 cache
        self._set_cos_sin_cache(
            seq_len=self.max_seq_len_cached, device=self.inv_freq.device, dtype=dtype
        )
        cos_cached = self.cos_cached.to(dtype).unsqueeze(0).unsqueeze(0)
        sin_cached = self.sin_cached.to(dtype).unsqueeze(0).unsqueeze(0)
        self.register_buffer("cos_cached_4d", cos_cached, persistent=False)
        self.register_buffer("sin_cached_4d", sin_cached, persistent=False)


# ---------------------------------------------------------------------------
# Attention (GQA, with KV cache)
# ---------------------------------------------------------------------------

@XHLLM_TRACEABLE_MODULES.register_module(
    {MiniCPMAttention: "MiniCPMAttention"}
)
class _MiniCPMAttention(DynamicModule):
    """MiniCPM GQA attention 的导出友好实现。

    区别于原生:
    - 用 xhnn.Rope / xhnn.LLMCache / MaskedSoftmax 替换 SDPA
    - 统一 prefill 和 decode 两种输入长度(靠 cfg.input_sequence_length)
    - 不走 is_causal / attn_mask 运行时分支,而是依赖 MaskedSoftmax 的静态 mask
    """

    # --------- 核心 forward (prefill / decode 共用) ---------

    def rotate_half(self, x: Tensor) -> Tensor:
        x1 = self.slice_1(x)
        x2 = self.slice_2(x)
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary(self, q: Tensor, k: Tensor, cos: Tensor, sin: Tensor):
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
        past_k_cache: Optional[Tensor] = None,
        past_v_cache: Optional[Tensor] = None,
        position_embeddings: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> Tuple[Tensor, None, None]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is not None:
            cos, sin = position_embeddings
            query_states, key_states = self.apply_rotary(query_states, key_states, cos, sin)

        if self.use_cache:
            key_states = self.k_cache(key_states, past_seq_length, current_input_length, past_k_cache)
            value_states = self.v_cache(value_states, past_seq_length, current_input_length, past_v_cache)

        # GQA: 把 kv head 扩展到 num_heads
        query_states = query_states * self.kv_scale
        key_states = key_states.transpose(2, 3)
        key_states = torch.repeat_interleave(key_states, self.num_key_value_groups, dim=1)

        attn_weights: Optional[Tensor] = self.masked_softmax(
            torch.matmul(query_states, key_states),
            past_seq_length,
        )

        value_states = torch.repeat_interleave(value_states, self.num_key_value_groups, dim=1)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)
        attn_output = self.o_proj(attn_output)
        return attn_output, None, None

    def _setup(self, cfg: Union[ConfigDict, Dict[str, Any]]):
        if isinstance(cfg, dict):
            cfg = ConfigDict(cfg)

        # MiniCPMAttention 已经在 __init__ 里挂了 num_heads / num_key_value_heads /
        # head_dim / num_key_value_groups,这里不需要重建,只补 wrap 需要的组件。
        self.enable_rope = cfg.get("enable_rope", True)
        if self.enable_rope:
            self.rope = xhnn.Rope()

        self.slice_1 = xhnn.Slice([0], [self.head_dim // 2], [3], [1])
        self.slice_2 = xhnn.Slice([self.head_dim // 2], [sys.maxsize], [3], [1])
        self.masked_softmax = MaskedSoftmax(dim=-1)

        use_cache = cfg.use_cache
        self.use_cache = use_cache
        self.max_sequence_length = cfg.max_sequence_length

        if use_cache:
            cache_axis = cfg.kv_cache.cache_axis
            self.k_cache = LLMCache(axis=cache_axis)
            self.v_cache = LLMCache(axis=cache_axis)
        else:
            self.k_cache = None
            self.v_cache = None

        self.kv_scale = 1.0 / math.sqrt(self.head_dim)
        return self


# ---------------------------------------------------------------------------
# MLP (MiniCPMMLP 不需要 wrap,结构就是三个 Linear + SiLU,FX 能直接 trace)
# 但为了确保量化命中,我们注册一个透传 wrap。
# ---------------------------------------------------------------------------

@XHLLM_TRACEABLE_MODULES.register_module(
    {MiniCPMMLP: "MiniCPMMLP"}
)
class _MiniCPMMLP(DynamicModule):
    """透传 wrap:保留原 forward,只是让 xhquant 能识别该模块。"""

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

    def _setup(self, cfg: Optional[Dict] = None):
        return self


# ---------------------------------------------------------------------------
# DecoderLayer
# ---------------------------------------------------------------------------

@XHLLM_TRACEABLE_MODULES.register_module(
    {MiniCPMDecoderLayer: "MiniCPMDecoderLayer"}
)
class _MiniCPMDecoderLayer(DynamicModule):
    """decoder 层 wrap。VoxCPM2 use_mup=False,residual 不需要 scale_depth 缩放。"""

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
        if self.use_mup:
            hidden_states = residual + hidden_states * self._scale
        else:
            hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        if self.use_mup:
            hidden_states = residual + hidden_states * self._scale
        else:
            hidden_states = residual + hidden_states
        return (hidden_states,)

    def _setup(self, cfg: Optional[Dict] = None):
        # use_mup / scale_depth / num_hidden_layers 都是原生 __init__ 里挂好的属性
        if self.use_mup:
            self._scale = self.scale_depth / math.sqrt(self.num_hidden_layers)
        else:
            self._scale = 1.0
        return self


# ---------------------------------------------------------------------------
# MiniCPMModel (顶层)
# ---------------------------------------------------------------------------

@XHLLM_TRACEABLE_MODULES.register_module(
    {MiniCPMModel: "MiniCPMModel"}
)
class _MiniCPMModel(DynamicModule):
    """MiniCPMModel 的 wrap,签名与 Qwen3-ASR 的 text model wrap 对齐。

    输入:
        inputs_embeds          [1, N, H]
        past_seq_length        [1]   int32
        current_input_length   [1]   int32
        past_key_cache         [L]   list of [1, kv_heads, cache_len, head_dim]
        past_value_cache       [L]   list of 同上
    输出:
        hidden_states          [1, N, H]  或 [1, 1, H](当 num_logits_to_keep=1)

    注意:本 wrap **不带 lm_head**。base_lm 和 residual_lm 都是 "只出 hidden" 的,
    后续的 fsq / stop_head / residual_lm 串联都在 host 侧或其他图里完成。
    """

    def forward(
        self,
        inputs_embeds: Tensor,
        past_seq_length: Tensor,
        current_input_length: Tensor,
        past_key_cache: Optional[List[Tensor]] = None,
        past_value_cache: Optional[List[Tensor]] = None,
    ) -> Tensor:
        hidden_states = inputs_embeds

        if self.rope_emb is not None:
            # 从全长 cos/sin cache 切出当前分块对应的部分。
            # 切片起点 = past_seq_length,长度 = input_sequence_length(静态)
            cos = self.cos_slice(self.rope_emb.cos_cached_4d, past_seq_length)
            sin = self.sin_slice(self.rope_emb.sin_cached_4d, past_seq_length)
            position_embeddings = (cos, sin)
        else:
            position_embeddings = None

        for idx, decoder_layer in enumerate(self.layers):
            if self.use_cache:
                _past_k = past_key_cache[idx]
                _past_v = past_value_cache[idx]
            else:
                _past_k = None
                _past_v = None

            layer_outputs = decoder_layer(
                hidden_states,
                past_seq_length=past_seq_length,
                current_input_length=current_input_length,
                past_k_cache=_past_k,
                past_v_cache=_past_v,
                position_embeddings=position_embeddings,
            )
            hidden_states = layer_outputs[0]

        if self.num_logits_to_keep == 1:
            # 只保留最后一个有效 token(current_input_length-1 位置)
            hidden_states = self.llm_gather(hidden_states, current_input_length - 1)
            hidden_states = hidden_states[-1]
            # 形状变为 [1, H] 或 [1, 1, H],最后再过 norm
            if hidden_states.dim() == 2:
                hidden_states = hidden_states.unsqueeze(1)

        hidden_states = self.norm(hidden_states)
        return hidden_states

    def _setup(self, cfg: Optional[Dict] = None):
        # num_logits_to_keep: 0 → 返回全部 N 个 hidden(prefill 给 residual_lm 用)
        #                     1 → 只返回最后一个(decode 用)
        self.num_logits_to_keep = cfg.get("num_logits_to_keep", 0)
        assert self.num_logits_to_keep in (0, 1)

        input_seq_len = cfg.input_sequence_length

        # gather 工具:按 indices 抽取特定位置的 hidden
        self.llm_gather = xhnn.BatchGather(1)
        self.llm_gather.update_offset_indices(1, input_seq_len)

        def _llm_gather_update_cfg(self: xhnn.BatchGather, cfg: Optional[Dict] = None):
            batch_size = cfg.get("batch_size", 1)
            self.update_offset_indices(batch_size, cfg.input_sequence_length)

        self.llm_gather._update_cfg = types.MethodType(_llm_gather_update_cfg, self.llm_gather)

        # cos/sin cache 切片:起点 past_seq_length,长度 input_seq_len
        self.sin_slice = xhnn.DynamicSlice([input_seq_len], [2], [1])
        self.cos_slice = xhnn.DynamicSlice([input_seq_len], [2], [1])

        def _sin_cos_slice_update_cfg(self, cfg: Optional[Dict] = None):
            self.valid_length = [cfg.input_sequence_length]

        self.sin_slice._update_cfg = types.MethodType(_sin_cos_slice_update_cfg, self.sin_slice)
        self.cos_slice._update_cfg = types.MethodType(_sin_cos_slice_update_cfg, self.cos_slice)

        self.use_cache = cfg.use_cache
        return self

    def _update_cfg(self, cfg: Optional[Dict] = None):
        """prefill → decode 切换时,把新的 cfg 推到本层状态上。

        xhquant 的 set_input_sequence_length(N) 会遍历所有 DynamicModule 触发
        `_update_cfg`,默认实现只处理 input_sequence_length。但 VoxCPM2 导出
        需要同时刷新 `num_logits_to_keep`(prefill=0, decode=1),因此这里显式
        重写一份,优先保险。

        注意:子模块(llm_gather, sin_slice, cos_slice)的 _update_cfg 已经在
        _setup 里各自注册,由 xhquant 框架在遍历过程中自动调用,这里只管顶层
        自己的状态。
        """
        if cfg is None:
            return
        if hasattr(cfg, "num_logits_to_keep"):
            new_val = cfg.num_logits_to_keep
        elif isinstance(cfg, dict) and "num_logits_to_keep" in cfg:
            new_val = cfg["num_logits_to_keep"]
        else:
            new_val = None
        if new_val is not None:
            assert new_val in (0, 1), f"num_logits_to_keep must be 0 or 1, got {new_val}"
            self.num_logits_to_keep = int(new_val)


# ---------------------------------------------------------------------------
# register hook(给外部脚本调用,确认一次注册已完成)
# ---------------------------------------------------------------------------

def register_wrap_cls(_hf_model=None):
    """VoxCPM2 的 MiniCPM 模块通过 @register_module 已在 import 时完成注册。

    这里提供一个空函数,和 qwen3_asr 的接口对齐;调用它也能顺带保证本模块被
    import(避免因为 lazy import 导致 wrap class 未注册的尴尬情况)。
    """
    return None