# ================================================================== #
#  File: _model.py                                                    #
#  Description:                                                       #
#    DeepSeek-V4 core DynamicModule wrappers for FX tracing.         #
#                                                                     #
#    Covers: RMSNorm, UnweightedRMSNorm, RotaryEmbedding,           #
#    Attention (MLA + sliding window + compressor), DecoderLayer,    #
#    Model, ForCausalLM.                                              #
#                                                                     #
#    MLA (Multi-head Latent Attention) features:                     #
#    - LoRA-style Q projection (q_a -> q_b)                         #
#    - Shared KV projection (K==V, single head)                     #
#    - Grouped output projection (o_a -> o_b)                       #
#    - Per-head attention sinks                                      #
#    - Conjugate RoPE on output                                      #
#    - Optional CSA/HCA KV cache compressor                         #
# ================================================================== #

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from xhquant.nn import LLMCacheV2
from xhquant.nn.modules.masked_softmax import SinksMaskedSoftmax
from xhquant.utils.registry import DynamicModule

from ..builder import XHLLM_TRACEABLE_MODULES
from ._compressor_nl import _apply_interleaved_rope


try:
    from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
        DeepseekV4Attention,
        DeepseekV4RMSNorm,
        DeepseekV4RotaryEmbedding,
        DeepseekV4UnweightedRMSNorm,
    )
except ImportError:
    DeepseekV4Attention = None
    DeepseekV4RMSNorm = DeepseekV4RotaryEmbedding = DeepseekV4UnweightedRMSNorm = None


# ================================================================== #
#  RMSNorm                                                            #
# ================================================================== #


if DeepseekV4RMSNorm is not None:
    _rms_registry = {DeepseekV4RMSNorm: "DeepseekV4RMSNorm"}
else:
    _rms_registry = {}


@XHLLM_TRACEABLE_MODULES.register_module(_rms_registry)
class _DeepseekV4RMSNorm(DynamicModule):
    """inline RMSNorm：x * rsqrt(mean(x²) + eps) * weight。

    不用 xhquant.nn.RMSNorm —— 后者内部 Div(1.0, sqrt) 在 ShapeProp 收到
    python float 1.0 时触发 AttributeError（float 无 is_floating_point），
    而量化工具链把 ShapeProp 错误降级为 warning 吞掉，会掩盖真实问题。
    改用 inline torch.rsqrt（与 _DeepseekV4UnweightedRMSNorm / _rms_norm 一致），
    从模型导出侧规避该报错，无需改 xhquanttool。
    """

    def forward(self, hidden_states):
        xf = hidden_states.float()
        var = (xf * xf).mean(-1, keepdim=True)
        # 用 reciprocal(sqrt) 替代 rsqrt：xhquanttool 的 rsqrt_to_sqrt 会把 rsqrt
        # 转成 Div(1.0, sqrt)，Div 收到 python float 触发 ShapeProp AttributeError
        # （被工具链吞成 warning，shape meta 缺失）。reciprocal/sqrt 不被该
        # transform 处理，规避报错。
        inv = torch.reciprocal(torch.sqrt(var + self.eps)).to(hidden_states.dtype)
        return hidden_states * inv * self.weight.to(hidden_states.dtype)

    def _setup(self, cfg: Optional[Dict] = None):
        self.eps = getattr(self, "variance_epsilon", getattr(self, "eps", 1e-6))
        return self


# ================================================================== #
#  UnweightedRMSNorm                                                  #
# ================================================================== #


if DeepseekV4UnweightedRMSNorm is not None:
    _unorm_registry = {DeepseekV4UnweightedRMSNorm: "DeepseekV4UnweightedRMSNorm"}
else:
    _unorm_registry = {}


@XHLLM_TRACEABLE_MODULES.register_module(_unorm_registry)
class _DeepseekV4UnweightedRMSNorm(DynamicModule):
    """UnweightedRMSNorm: x * rsqrt(mean(x^2, -1) + eps).

    No learnable weight — equivalent to RMSNorm(weight=ones) but avoids
    creating an xh2a::RMSNorm op with weight shape [1], which crashes the
    Triton kernel when the actual hidden dim is 16384+.
    """

    def forward(self, x: Tensor) -> Tensor:
        xf = x.float()
        var = (xf * xf).mean(-1, keepdim=True) + self.eps
        # reciprocal(sqrt) 替代 rsqrt，规避 rsqrt_to_sqrt 的 Div(1.0) ShapeProp 报错
        return x * torch.reciprocal(torch.sqrt(var)).to(x.dtype)

    def _setup(self, cfg: Optional[Dict] = None):
        self.eps = getattr(self, "eps", 1e-6)
        return self


# ================================================================== #
#  RotaryEmbedding (dual-frequency: main + compress)                 #
# ================================================================== #


if DeepseekV4RotaryEmbedding is not None:
    _rope_registry = {DeepseekV4RotaryEmbedding: "DeepseekV4RotaryEmbedding"}
else:
    _rope_registry = {}


@XHLLM_TRACEABLE_MODULES.register_module(_rope_registry)
class _DeepseekV4RotaryEmbedding(DynamicModule):
    """Pre-compute cos/sin caches for both 'main' and 'compress' rope types."""

    def _setup(self, cfg):
        max_pe_length = getattr(cfg, "max_pe_length", None)
        if not max_pe_length:
            max_pe_length = getattr(cfg, "max_sequence_length", 2048)
        self._setup_cos_sin_cache(max_pe_length)

    def _compute_cos_sin(self, layer_type, max_seq_len):
        inv_freq = getattr(self, f"{layer_type}_inv_freq")
        scaling = getattr(self, f"{layer_type}_attention_scaling")
        inv_freq_exp = inv_freq[None, :, None].float()  # [1, rope_dim/2, 1]
        positions = torch.arange(max_seq_len, device=inv_freq.device).float()[None, None, :]  # [1, 1, max_seq_len]
        device_type = inv_freq.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_exp @ positions).transpose(1, 2).squeeze(0)  # [max_seq_len, rope_dim/2]
            cos = freqs.cos() * scaling
            sin = freqs.sin() * scaling
        return cos.to(inv_freq.dtype), sin.to(inv_freq.dtype)

    def _setup_cos_sin_cache(self, max_seq_len):
        for lt in self.layer_types:
            cos, sin = self._compute_cos_sin(lt, max_seq_len)
            self.register_buffer(f"{lt}_cos_cached", cos, persistent=True)
            self.register_buffer(f"{lt}_sin_cached", sin, persistent=True)

    # NOTE: 死代码，待删除。_layers.py 直接访问 main_cos_cached[pid]，不经过此方法。
    # def get_embeddings(self, position_ids, layer_type="main"):
    #     """Return (cos, sin) for given position_ids and layer_type."""
    #     cos = getattr(self, f"{layer_type}_cos_cached")[position_ids]
    #     sin = getattr(self, f"{layer_type}_sin_cached")[position_ids]
    #     return cos, sin


# ================================================================== #
#  Attention (MLA + sliding window + optional compressor)            #
# ================================================================== #


if DeepseekV4Attention is not None:
    _attn_registry = {DeepseekV4Attention: "DeepseekV4Attention"}
else:
    _attn_registry = {}


@XHLLM_TRACEABLE_MODULES.register_module(_attn_registry)
class _DeepseekV4Attention(DynamicModule):
    """MLA attention with sliding window KV cache + optional compressor.

    Key architectural differences from standard attention:
    - LoRA Q: q_a_proj -> q_a_norm -> q_b_proj -> q_b_norm
    - Shared KV: kv_proj -> kv_norm (K==V, single head)
    - Grouped output: o_a_proj (grouped) -> o_b_proj
    - Per-head attention sinks
    - Conjugate RoPE on output
    """

    def forward(
        self,
        hidden_states: Tensor,
        cos: Optional[Tensor] = None,
        sin: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_seq_length: Optional[Tensor] = None,
        current_input_length: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        input_ids: Optional[Tensor] = None,
        past_k_cache: Optional[Tensor] = None,
        compressed_kv_cache: Optional[Tensor] = None,
        **kwargs,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        bsz, seq_len, _ = hidden_states.shape
        head_dim = self.head_dim
        num_heads = self.num_heads
        dtype = hidden_states.dtype

        # -- Q: LoRA projection --
        q_residual = self.q_a_norm(self.q_a_proj(hidden_states))
        q = self.q_b_proj(q_residual).view(bsz, seq_len, num_heads, head_dim)
        q = q.transpose(1, 2)
        q = self.q_b_norm(q)

        # -- KV: shared projection --
        kv = self.kv_norm(self.kv_proj(hidden_states))
        kv = kv.view(bsz, seq_len, 1, head_dim).transpose(1, 2)

        # -- RoPE（free function，FX trace 看到原子 op） --
        q = _apply_interleaved_rope(
            q, cos, sin, self.rope_dim, unsqueeze_dim=1, even_idx=self._rope_even_idx, odd_idx=self._rope_odd_idx
        )
        kv = _apply_interleaved_rope(
            kv, cos, sin, self.rope_dim, unsqueeze_dim=1, even_idx=self._rope_even_idx, odd_idx=self._rope_odd_idx
        )

        # -- Sliding window KV cache --
        if self.use_cache:
            kv_full = self.kv_cache(kv, past_seq_length, current_input_length, past_k_cache)
        else:
            kv_full = kv

        # -- Compressor branch (CSA/HCA) --
        # 三路分支，FX trace 时由 concrete 值决定路径：
        #   prefill: attention_mask 非空 → 计算 compressed_kv，拼在 kv_full 前面
        #   decode:  compressed_kv_cache 非空 → 拼在 kv_full 前面
        #   decode:  无 cache → 跳过
        #
        # KV 布局（compressed 在前）:
        #   [compressed_kv | regular_kv | zero_padding]
        # 例如 CSA(compress_rate=4, seq_len=256):
        #   位置 0-63:  compressed entries (block_bias 控制)
        #   位置 64-319: regular KV (causal mask 控制)
        #   位置 320-2111: zero padding (hybrid mask 控制)
        compressed_kv_out = None
        if self.has_compressor:
            if attention_mask is not None:
                # Prefill: 正常计算 compressor，compressed 拼在前面
                compressed_kv, block_bias = self._compressor_forward(
                    hidden_states,
                    q_residual,
                    position_ids,
                    current_input_length,
                    past_seq_length,
                )
                kv_full = torch.cat([compressed_kv, kv_full], dim=2)
                attention_mask = torch.cat([block_bias.to(attention_mask.dtype), attention_mask], dim=-1)
                compressed_kv_out = compressed_kv
            elif compressed_kv_cache is not None:
                # Decode: 复用 prefill 缓存，compressed 拼在前面（与 prefill 布局一致）
                kv_full = torch.cat([compressed_kv_cache, kv_full], dim=2)

        # -- Attention: broadcast KV to all heads --
        kv_expanded = kv_full.expand(-1, num_heads, -1, -1)

        # -- Eager attention with sinks --
        q_scaled = q * self.scaling
        attn_weights = torch.matmul(q_scaled, kv_expanded.transpose(-1, -2))

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # -- Fused softmax + sinks via SinksMaskedSoftmax --
        # compressed_kv 拼在 kv_full 前面时，需扩展有效长度以避免被 mask。
        # 用 buffer 做加法（tensor + tensor），避免 QAdd(tensor, int) 崩溃。
        #
        # KV 布局（compressor 在前）:
        #   [compressed_kv | regular_kv | zero_padding]
        # hybrid mask 条件: j > (i + effective_psl)
        #   - decode: effective_psl = past_seq_length + ckv_offset
        #   - prefill: effective_psl = compressed_len + past_seq_length
        effective_psl = past_seq_length
        if self.has_compressor:
            if self._is_decode:
                effective_psl = past_seq_length + self._ckv_seq_offset
            elif compressed_kv_out is not None:
                # Prefill: effective_psl = compressed_len + seq_len
                # 用预计算的 buffer（避免 new_tensor 或 int + tensor 崩溃）
                effective_psl = self._prefill_psl_offset + past_seq_length

        sinks = self.sinks.reshape(1, num_heads, 1, 1)
        probs = self.sinks_softmax(
            attn_weights.float(),
            effective_psl,
            sinks=sinks,
        )
        attn_to_kv = probs.to(dtype)

        # -- Attention output --
        attn_output = torch.matmul(attn_to_kv, kv_expanded)

        # -- Conjugate RoPE on output: transpose BHSD→BSHD, apply, keep BSHD for grouped proj --
        attn_output = _apply_interleaved_rope(
            attn_output.transpose(1, 2),
            cos,
            -sin,
            self.rope_dim,
            unsqueeze_dim=2,
            even_idx=self._rope_even_idx,
            odd_idx=self._rope_odd_idx,
        )

        # -- Grouped output projection: per-group quantized Linear --
        grouped = attn_output.reshape(bsz, seq_len, self.o_groups, -1)
        parts = [self.o_a_projs[g](grouped[:, :, g, :]) for g in range(self.o_groups)]
        grouped = torch.stack(parts, dim=2).flatten(2)
        output = self.o_b_proj(grouped)
        return output, compressed_kv_out

    def _compressor_forward(
        self,
        hidden_states,
        q_residual,
        position_ids,
        current_input_length,
        past_seq_length,
    ):
        """Forward through decomposed compressor: quantized linears + non-linear logic."""
        if not hasattr(self, "compressor_nl"):
            return None, None

        # -- concrete seq_len，TorchFX trace 时不产生 Proxy --
        seq_len = self.input_sequence_length

        comp_kv = self.comp_kv_proj(hidden_states)
        comp_gate = self.comp_gate_proj(hidden_states)

        if self.has_indexer:
            idx_kv = self.idx_kv_proj(hidden_states)
            idx_gate = self.idx_gate_proj(hidden_states)
            idx_q = self.idx_q_proj(q_residual)
            idx_w = self.idx_w_proj(hidden_states)

            compressed_kv, q, idx_kv_t, idx_w_scaled = self.compressor_nl(
                comp_kv,
                comp_gate,
                idx_kv,
                idx_gate,
                idx_q,
                idx_w,
                seq_len,
                position_ids,
            )

            scores = torch.matmul(q, idx_kv_t)

            block_bias = self.csa_scoring(scores, idx_w_scaled, compressed_kv, seq_len, position_ids)

            return compressed_kv, block_bias
        else:
            return self.compressor_nl(comp_kv, comp_gate, seq_len, position_ids)

    def _setup(self, cfg: Optional[Dict] = None):
        self.num_heads = self.config.num_attention_heads
        self.head_dim = self.config.head_dim
        self.o_groups = self.config.o_groups
        self.scaling = self.head_dim**-0.5
        self.rope_dim = self.config.qk_rope_head_dim
        # xh2a 量化 runtime 对 step=2 strided Slice 支持不稳定（实测 RoPE rope 维
        # cosine≈0）。预计算 even/odd idx buffer，_apply_interleaved_rope 用
        # index_select（连续 gather）替代 strided slice。
        self.register_buffer(
            "_rope_even_idx",
            torch.arange(0, self.rope_dim, 2, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "_rope_odd_idx",
            torch.arange(1, self.rope_dim, 2, dtype=torch.long),
            persistent=False,
        )
        self._wrap_cfg = cfg

        # -- concrete seq_len for TorchFX tracing compatibility --
        self.input_sequence_length = cfg.input_sequence_length
        self._is_decode = cfg.input_sequence_length == 1

        # -- RoPE: rope_dim stored as attribute, apply via free function --
        # 不再使用 FX leaf module，直接用 _apply_interleaved_rope free function
        # FX trace 看到原子 op（repeat_interleave, stack, flatten, cat）

        layer_type = self.config.layer_types[self.layer_idx]
        self.rope_layer_type = "main" if layer_type == "sliding_attention" else "compress"
        self.has_compressor = layer_type != "sliding_attention"

        # -- Compressed KV 序列长度偏移量（decode 模式用 buffer 而非 int） --
        # 直接用 compressed_kv_cache.shape[2]（Python int）做 + 操作，
        # 量化后变成 QAdd(tensor, int)，QAdd.forward 调 x2.device 会崩。
        # 存为 buffer 保证 FakeTensorProp 阶段两侧都是 fake tensor。
        # 必须在 has_compressor 之后注册。
        ckv_lens = getattr(cfg, "compressed_kv_seq_lens", None)
        ckv_offset = 0
        if ckv_lens is not None and self.has_compressor:
            ckv_offset = ckv_lens.get(self.layer_idx, 0)
        self.register_buffer(
            "_ckv_seq_offset",
            torch.tensor([ckv_offset], dtype=torch.int32),
            persistent=False,
        )

        # -- Prefill effective_psl 偏移（= compressed_len） --
        # KV 布局: [compressed(0..ckv_len-1) | regular(ckv_len..ckv_len+seq-1) | padding]
        # hybrid mask: j > (i + effective_psl)
        # effective_psl = compressed_len + past_seq_length (= ckv_len + 0 for prefill)
        # → query i 可见范围: compressed 全部 + regular 0..i
        # → padding 从 ckv_len+i+1 起被 mask
        pf_psl_offset = 0
        if self.has_compressor and not self._is_decode:
            if ckv_lens is not None:
                pf_psl_offset = ckv_lens.get(self.layer_idx, 0)
        self.register_buffer(
            "_prefill_psl_offset",
            torch.tensor([pf_psl_offset], dtype=torch.int32),
            persistent=False,
        )

        # -- KV cache --
        self.use_cache = cfg.use_cache
        if self.use_cache:
            cache_axis = cfg.kv_cache.cache_axis
            # -- Sliding window: decode 时 LLMCacheV2 只返回最近 window 个位置 --
            # 输出 shape 固定为 [B, H, aligned(sliding_window, 16), D],
            # register_fake 已正确处理, HMONNX runtime parser 已支持 amax。
            sw = self.config.sliding_window if layer_type == "sliding_attention" else 0
            amax = sw if (self.input_sequence_length == 1 and sw > 0) else -1
            self.kv_cache = LLMCacheV2(axis=cache_axis, attention_max_length=amax)

        # -- Attention sinks as buffer --
        sinks_data = self.sinks.data.clone()
        del self._parameters["sinks"]
        self.register_buffer("sinks", sinks_data, persistent=True)

        # -- SinksMaskedSoftmax (fused xh2a op) --
        # causal + boundary masking 由 past_seq_length 动态完成;
        # sliding window 由 LLMCacheV2 的 attention_max_length 在数据层处理,
        # 此处无需额外 mask。
        self.sinks_softmax = SinksMaskedSoftmax(dim=-1)

        # -- o_a_proj 分解为独立 nn.Linear（可量化） --
        self._decompose_o_a_proj()

        # -- Compressor 线性投影提取（可量化） --
        if self.has_compressor and hasattr(self, "compressor"):
            self._decompose_compressor()

        return self

    def _decompose_o_a_proj(self):
        """将 o_a_proj GroupedLinear 分解为独立 nn.Linear 模块。"""
        o_a = self.o_a_proj
        if o_a is None:
            return

        w = o_a.weight.data  # [n_groups * out_per_group, in_features]
        n_groups = getattr(o_a, "n_groups", self.o_groups)
        out_features = w.shape[0] // n_groups  # 每组输出维度
        in_features = w.shape[-1]  # 每组输入维度

        self.o_a_projs = nn.ModuleList([nn.Linear(in_features, out_features, bias=False) for _ in range(n_groups)])
        for i, lin in enumerate(self.o_a_projs):
            lin.weight.data.copy_(w[i * out_features : (i + 1) * out_features, :])
        del self.o_a_proj

    def _decompose_compressor(self):
        """从 compressor 提取线性投影为独立 nn.Linear（可量化）。"""

        comp = self.compressor

        eps = getattr(self.config, "rms_norm_eps", 1e-6)
        head_dim = self.config.head_dim
        max_seq_len = self._wrap_cfg.max_sequence_length

        # -- 从 compressor 自己的 rotary_emb 计算 compress RoPE cos/sin --
        compress_cos, compress_sin = self._compute_compress_rope(comp.rotary_emb, max_seq_len)
        if compress_cos is None:
            return

        # 判断 compressor 类型
        comp_cls_name = type(comp).__name__

        if "CSA" in comp_cls_name:
            self._decompose_csa(comp, head_dim, eps, compress_cos, compress_sin)
        elif "HCA" in comp_cls_name:
            self._decompose_hca(comp, head_dim, eps, compress_cos, compress_sin)

    def _compute_compress_rope(self, rotary_emb, max_seq_len):
        """从 compressor 的 HF rotary_emb 计算 compress RoPE cos/sin 缓存。"""
        inv_freq = getattr(rotary_emb, "compress_inv_freq", None)
        if inv_freq is None:
            return None, None
        scaling = getattr(rotary_emb, "compress_attention_scaling", 1.0)
        inv_freq_exp = inv_freq[None, :, None].float()  # [1, rope_dim/2, 1]
        positions = torch.arange(max_seq_len, device=inv_freq.device).float()[None, None, :]
        with torch.autocast(device_type=inv_freq.device.type, enabled=False):
            freqs = (inv_freq_exp @ positions).transpose(1, 2).squeeze(0)
            cos = freqs.cos() * scaling
            sin = freqs.sin() * scaling
        return cos.to(inv_freq.dtype), sin.to(inv_freq.dtype)

    def _decompose_csa(self, comp, head_dim, eps, compress_cos, compress_sin):
        """提取 CSA compressor 的线性投影。"""
        from ._compressor_nl import _CSANonLinear

        # -- CSA 自身的 2 个 Linear --
        self.comp_kv_proj = nn.Linear(
            comp.kv_proj.in_features,
            comp.kv_proj.out_features,
            bias=False,
        )
        self.comp_kv_proj.weight.data.copy_(comp.kv_proj.weight.data)
        self.comp_gate_proj = nn.Linear(
            comp.gate_proj.in_features,
            comp.gate_proj.out_features,
            bias=False,
        )
        self.comp_gate_proj.weight.data.copy_(comp.gate_proj.weight.data)

        # -- Indexer 的 4 个 Linear --
        idx = comp.indexer
        self.idx_kv_proj = nn.Linear(
            idx.kv_proj.in_features,
            idx.kv_proj.out_features,
            bias=False,
        )
        self.idx_kv_proj.weight.data.copy_(idx.kv_proj.weight.data)
        self.idx_gate_proj = nn.Linear(
            idx.gate_proj.in_features,
            idx.gate_proj.out_features,
            bias=False,
        )
        self.idx_gate_proj.weight.data.copy_(idx.gate_proj.weight.data)
        self.idx_q_proj = nn.Linear(
            idx.q_b_proj.in_features,
            idx.q_b_proj.out_features,
            bias=False,
        )
        self.idx_q_proj.weight.data.copy_(idx.q_b_proj.weight.data)
        self.idx_w_proj = nn.Linear(
            idx.weights_proj.in_features,
            idx.weights_proj.out_features,
            bias=False,
        )
        self.idx_w_proj.weight.data.copy_(idx.weights_proj.weight.data)

        # -- 构建非线性模块 --
        self.has_indexer = True
        self.compressor_nl = _CSANonLinear(
            head_dim=head_dim,
            compress_rate=comp.compress_rate,
            position_bias=comp.position_bias.data,
            kv_norm_weight=comp.kv_norm.weight.data,
            idx_head_dim=idx.head_dim,
            idx_n_heads=idx.num_heads,
            idx_compress_rate=idx.compress_rate,
            idx_index_topk=idx.index_topk,
            idx_position_bias=idx.position_bias.data,
            idx_kv_norm_weight=idx.kv_norm.weight.data,
            compress_cos_cached=compress_cos,
            compress_sin_cached=compress_sin,
            eps=eps,
        )

        # -- Indexer scoring（普通 nn.Module，FX trace 可见） --
        from ._compressor_nl import _CSAScoring

        self.csa_scoring = _CSAScoring(
            idx_index_topk=idx.index_topk,
            softmax_scale=idx.head_dim**-0.5,
            idx_compress_rate=idx.compress_rate,
            compress_rate=comp.compress_rate,
            seq_len=self.input_sequence_length,
            device=compress_cos.device,
        )

        del self.compressor

    def _decompose_hca(self, comp, head_dim, eps, compress_cos, compress_sin):
        """提取 HCA compressor 的线性投影。"""
        from ._compressor_nl import _HCANonLinear

        self.comp_kv_proj = nn.Linear(
            comp.kv_proj.in_features,
            comp.kv_proj.out_features,
            bias=False,
        )
        self.comp_kv_proj.weight.data.copy_(comp.kv_proj.weight.data)
        self.comp_gate_proj = nn.Linear(
            comp.gate_proj.in_features,
            comp.gate_proj.out_features,
            bias=False,
        )
        self.comp_gate_proj.weight.data.copy_(comp.gate_proj.weight.data)

        self.has_indexer = False
        self.compressor_nl = _HCANonLinear(
            head_dim=head_dim,
            compress_rate=comp.compress_rate,
            position_bias=comp.position_bias.data,
            kv_norm_weight=comp.kv_norm.weight.data,
            compress_cos_cached=compress_cos,
            compress_sin_cached=compress_sin,
            eps=eps,
            seq_len=self.input_sequence_length,
        )

        del self.compressor
