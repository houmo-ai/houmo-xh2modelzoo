"""Merak dynamic wrappers for the checkpoint-side Ling-3 implementation."""

from __future__ import annotations

import math
import sys
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from xhquant import nn as xhnn
from xhquant.nn import FlashAttention, LLMCacheV2, MaskedSoftmax, RMSNorm
from xhquant.nn.modules import GDRChunkScan
from xhquant.nn.modules.moeblock import MoeBlock
from xhquant.utils.registry import DynamicModule

from ...register import XHLLM_TRACEABLE_MODULES
from ..qwen3_5._gdr_ops import GDRBlockTriInverse, GDRRecurrentScan
from ..qwen3_5._hybrid_moe import _pack_defused_expert_linear_to_moeblock
from ..qwen3_5.split_conv_cache_utils import _select_linear_attn_conv_cache
from ._kda_rule import chunk_kda, kda_log_decay, recurrent_kda_with_gdr_scan


class _LingRMSNorm(DynamicModule):
    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.norm(hidden_states)

    def _setup(self, cfg=None):
        del cfg
        eps = getattr(self, "variance_epsilon", getattr(self, "eps", 1e-6))
        self.norm = RMSNorm(self.weight.shape[0], eps).to(
            device=self.weight.device,
            dtype=self.weight.dtype,
        )
        with torch.no_grad():
            self.norm.weight.copy_(self.weight)
        return self


class _LingRotaryEmbedding(DynamicModule):
    def _setup(self, cfg=None):
        del cfg
        if "dynamic" in self.rope_type:
            raise ValueError("Ling-3 dynamic RoPE is not supported for static export")
        self._setup_cos_sin_cache(self.max_seq_len_cached, self.inv_freq.dtype)
        if hasattr(self, "setup_after_callback"):
            self.setup_after_callback()
        return self

    def _setup_cos_sin_cache(self, seq_len: int, dtype: torch.dtype) -> None:
        positions = torch.arange(seq_len, dtype=torch.long, device=self.inv_freq.device).unsqueeze(0)
        inv_freq = self.inv_freq.float()
        cos, sin = self.forward(inv_freq, positions)
        self.register_buffer("cos_cached", cos.squeeze(0).unsqueeze(0).unsqueeze(0).to(dtype), persistent=False)
        self.register_buffer("sin_cached", sin.squeeze(0).unsqueeze(0).unsqueeze(0).to(dtype), persistent=False)

    @torch.no_grad()
    def forward(self, x: Tensor, position_ids: Tensor) -> tuple[Tensor, Tensor]:
        inv = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        positions = position_ids[:, None, :].float()
        freqs = (inv @ positions).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        return (emb.cos() * self.attention_scaling).to(x.dtype), (emb.sin() * self.attention_scaling).to(x.dtype)


class _LingMLAAttention(DynamicModule):
    """Absorbed MLA with independent K/V cache quantization domains.

    K stores ``[normalized latent, RoPE]`` and V stores another copy of the
    normalized latent.  The duplicate is intentional: K SEFP shares exponents
    across head-dim, while V SEFP shares them across tokens.

    Both the explicit attention path and XH2 FlashAttention consume the same
    fully absorbed representation: Q/K use 576 coordinates (512 latent + 64
    RoPE) while V/output use 512 latent coordinates.  The kernel choice must
    never change the persistent cache representation.
    """

    def _interleaved_rope(self, q: Tensor, k: Tensor, cos: Tensor, sin: Tensor):
        q = q.view(
            self.batch_size,
            self.num_heads,
            self.input_sequence_length,
            self.qk_rope_head_dim // 2,
            2,
        ).transpose(4, 3).reshape(
            self.batch_size,
            self.num_heads,
            self.input_sequence_length,
            self.qk_rope_head_dim,
        )
        k = k.view(
            self.batch_size,
            1,
            self.input_sequence_length,
            self.qk_rope_head_dim // 2,
            2,
        ).transpose(4, 3).reshape(
            self.batch_size,
            1,
            self.input_sequence_length,
            self.qk_rope_head_dim,
        )
        if self.enable_rope:
            return self.rope(q, cos, sin), self.rope(k, cos, sin)
        q1, q2 = q.chunk(2, dim=-1)
        k1, k2 = k.chunk(2, dim=-1)
        return q * cos + torch.cat((-q2, q1), dim=-1) * sin, k * cos + torch.cat((-k2, k1), dim=-1) * sin

    def forward(
        self,
        hidden_states: Tensor,
        past_seq_length: Tensor | None = None,
        current_input_length: Tensor | None = None,
        position_embeddings: tuple[Tensor, Tensor] | None = None,
        past_k_cache: Tensor | None = None,
        past_v_cache: Tensor | None = None,
        **kwargs,
    ) -> Tensor:
        del kwargs
        batch = self.batch_size
        sequence = self.input_sequence_length
        q_input = (
            hidden_states
            if self.q_lora_rank is None
            else self.q_a_layernorm(self.q_a_proj(hidden_states))
        )
        q_content = self.q_absorbed_proj(q_input).view(
            batch,
            sequence,
            self.num_heads,
            self.kv_lora_rank,
        )
        q_rope = self.q_rope_proj(q_input).view(
            batch,
            sequence,
            self.num_heads,
            self.qk_rope_head_dim,
        )

        compressed = self.kv_a_proj_with_mqa(hidden_states)
        latent_raw, k_rope = torch.split(compressed, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        latent = self.kv_a_layernorm(latent_raw)
        q_rope = q_rope.transpose(1, 2)
        k_rope = k_rope.view(batch, 1, sequence, self.qk_rope_head_dim)
        cos, sin = position_embeddings
        q_rope, k_rope = self._interleaved_rope(q_rope, k_rope, cos, sin)

        query = torch.cat((q_content.transpose(1, 2), q_rope), dim=-1)
        key = torch.cat((latent.unsqueeze(1), k_rope), dim=-1)
        # K and V deliberately receive separate tensors/caches even though
        # both contain the same normalized latent rows. Their SEFP exponent
        # sharing axes differ, so sharing physical storage would be invalid.
        value = latent.unsqueeze(1)
        if self.use_cache:
            key = self.k_cache(key, past_seq_length, current_input_length, past_k_cache)
            value = self.v_cache(value, past_seq_length, current_input_length, past_v_cache)

        if self.use_flash_attention:
            context = self.flash_attn(
                query,
                key,
                value,
                past_seq_length=past_seq_length,
                current_input_length=current_input_length,
            )
        else:
            scores = torch.matmul(query * self.attention_scale, key.transpose(-1, -2))
            scores = self.masked_softmax(scores, past_seq_length)
            context = torch.matmul(scores, value)
        context = context.transpose(1, 2)
        if self.use_elementwise_value_projection:
            context = context.permute(0, 2, 3, 1).reshape(
                batch,
                self.num_heads * self.kv_lora_rank,
                sequence,
            )
            context = self.value_up_conv(context).view(
                batch,
                self.num_heads,
                self.v_head_dim,
                sequence,
            ).permute(0, 3, 1, 2)
        if self.g_proj is not None:
            gate = torch.sigmoid(self.g_proj(hidden_states))
            if self.gated_attention_proj_granularity_type == "head_wise":
                gate = gate.view(batch, sequence, self.num_heads, 1)
            elif self.gated_attention_proj_granularity_type == "element_wise":
                gate = gate.view(batch, sequence, self.num_heads, self.v_head_dim)
            else:
                raise ValueError(
                    "Unsupported Ling MLA gate granularity: "
                    f"{self.gated_attention_proj_granularity_type!r}"
                )
            context = context * gate
        output = context.reshape(batch, sequence, -1)
        if self.use_elementwise_value_projection:
            return self.dense(output)
        return self.absorbed_dense(output)

    def _setup(self, cfg: Optional[Dict] = None):
        self.hidden_size = int(getattr(self, "hidden_size", self.config.hidden_size))
        self.use_cache = bool(cfg.use_cache)
        self.batch_size = int(cfg.get("batch_size", 1))
        self.input_sequence_length = int(cfg.input_sequence_length)
        self.enable_rope = bool(cfg.get("enable_rope", True))
        if self.enable_rope:
            self.rope = xhnn.Rope()
        self.masked_softmax = MaskedSoftmax(dim=-1)

        flash_cfg = cfg.get("flash_attention", None)
        cfg_get = flash_cfg.get if hasattr(flash_cfg, "get") else None
        self.use_flash_attention = bool(cfg_get("enable", False)) if cfg_get else False
        if self.use_flash_attention:
            bits = {
                name: int(cfg_get(name, 8))
                for name in ("q_bits", "k_bits", "v_bits", "s_bits", "p_bits")
            }
            if any(value not in (8, 16) for value in bits.values()):
                raise ValueError(f"Ling FlashAttention bits must be 8 or 16, got {bits}")
            self.flash_attn = FlashAttention(
                num_heads=self.num_heads,
                num_kv_heads=1,
                scale=float(self.scaling),
                is_causal=True,
                **bits,
            )
        if self.use_cache:
            axis = cfg.kv_cache.cache_axis
            self.k_cache = LLMCacheV2(axis=axis)
            self.v_cache = LLMCacheV2(axis=axis)

        device = self.kv_b_proj.weight.device
        dtype = self.kv_b_proj.weight.dtype
        fusion_dtype = torch.float32 if device.type != "meta" else dtype
        w_kv = self.kv_b_proj.weight.to(fusion_dtype).view(
            self.num_heads,
            self.qk_nope_head_dim + self.v_head_dim,
            self.kv_lora_rank,
        )
        w_uk = w_kv[:, : self.qk_nope_head_dim]
        w_uv = w_kv[:, self.qk_nope_head_dim :]
        q_source = self.q_proj if self.q_lora_rank is None else self.q_b_proj
        q_weight = q_source.weight.to(fusion_dtype).view(
            self.num_heads,
            self.qk_nope_head_dim + self.qk_rope_head_dim,
            -1,
        )
        q_nope = q_weight[:, : self.qk_nope_head_dim]
        q_rope = q_weight[:, self.qk_nope_head_dim :]
        q_absorbed = torch.einsum("hni,hnc->hci", q_nope, w_uk)
        q_in = q_source.in_features
        self.q_absorbed_proj = nn.Linear(
            q_in,
            self.num_heads * self.kv_lora_rank,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.q_rope_proj = nn.Linear(
            q_in,
            self.num_heads * self.qk_rope_head_dim,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.q_absorbed_proj.weight.data.copy_(q_absorbed.reshape(-1, q_in).to(dtype))
        self.q_rope_proj.weight.data.copy_(q_rope.reshape(-1, q_in).to(dtype))
        self.use_elementwise_value_projection = (
            self.g_proj is not None
            and self.gated_attention_proj_granularity_type == "element_wise"
        )
        if self.use_elementwise_value_projection:
            # Element-wise V gates do not commute with W_uv, so the usual
            # W_uv+dense absorption would change the model.  A compact grouped
            # 1x1 projection reconstructs each head's V coordinates before the
            # gate while preserving the latent-only V cache.
            self.value_up_conv = nn.Conv1d(
                self.num_heads * self.kv_lora_rank,
                self.num_heads * self.v_head_dim,
                kernel_size=1,
                groups=self.num_heads,
                bias=False,
                device=device,
                dtype=dtype,
            )
            self.value_up_conv.weight.data.copy_(
                w_uv.reshape(self.num_heads * self.v_head_dim, self.kv_lora_rank, 1).to(dtype)
            )
        else:
            dense_weight = self.dense.weight.to(fusion_dtype).view(
                self.hidden_size,
                self.num_heads,
                self.v_head_dim,
            )
            absorbed = torch.einsum("xhv,hvc->xhc", dense_weight, w_uv)
            self.absorbed_dense = nn.Linear(
                self.num_heads * self.kv_lora_rank,
                self.hidden_size,
                bias=self.dense.bias is not None,
                device=device,
                dtype=dtype,
            )
            self.absorbed_dense.weight.data.copy_(absorbed.reshape(self.hidden_size, -1).to(dtype))
            if self.dense.bias is not None:
                self.absorbed_dense.bias.data.copy_(self.dense.bias)
            del self.dense
        # The absorbed matrices are new quantization boundaries.  They are
        # quantized by Merak's W8 export scheme rather than pretending that a
        # product of packed integer weights is itself a valid packed weight.
        if self.q_lora_rank is None:
            del self.q_proj
        else:
            del self.q_b_proj
        del self.kv_b_proj
        self.register_buffer(
            "attention_scale",
            torch.tensor(float(self.scaling), device=device, dtype=dtype),
            persistent=False,
        )
        return self

    def _update_cfg(self, cfg: Optional[Dict] = None):
        if cfg is None:
            return
        self.use_cache = bool(cfg.use_cache)
        self.batch_size = int(cfg.get("batch_size", self.batch_size))
        self.input_sequence_length = int(
            cfg.get("input_sequence_length", self.input_sequence_length)
        )


def _mask_views(mask: Tensor | tuple[Tensor, Tensor, Tensor], dtype: torch.dtype):
    if isinstance(mask, tuple):
        return mask
    mask = mask.to(dtype)
    return mask.unsqueeze(-1), mask.unsqueeze(1), mask.unsqueeze(-1).unsqueeze(-1)


class _LingKDAAttention(DynamicModule):
    def _project_gate(self, hidden_states: Tensor, prefix: str) -> Tensor:
        direct = getattr(self, f"{prefix}_proj", None)
        if direct is not None:
            return direct(hidden_states)
        return getattr(self, f"{prefix}_b_proj")(getattr(self, f"{prefix}_a_proj")(hidden_states))

    def _conv_branch(self, x: Tensor, cache: Tensor, conv: nn.Conv1d, current_input_length: Tensor):
        merged = torch.cat((cache, x.transpose(1, 2)), dim=-1).to(conv.weight.dtype)
        cache_out = self.conv_cache_slice(merged, current_input_length)
        start = self.conv_size
        output = conv(merged)[:, :, start : start + self.input_sequence_length]
        return F.silu(output).transpose(1, 2), cache_out

    def forward(
        self,
        hidden_states: Tensor,
        conv_cache=None,
        recurrent_state: Tensor | None = None,
        linear_attn_mask: Tensor | tuple[Tensor, Tensor, Tensor] | None = None,
        current_input_length: Tensor | None = None,
    ):
        mask, _, state_mask = _mask_views(linear_attn_mask, hidden_states.dtype)
        if not isinstance(conv_cache, (tuple, list)) or len(conv_cache) != 3:
            raise ValueError("Ling KDA requires split q/k/v convolution caches")
        q, q_cache = self._conv_branch(self.q_proj(hidden_states), conv_cache[0], self.q_conv, current_input_length)
        k, k_cache = self._conv_branch(self.k_proj(hidden_states), conv_cache[1], self.k_conv, current_input_length)
        v, v_cache = self._conv_branch(self.v_proj(hidden_states), conv_cache[2], self.v_conv, current_input_length)
        q, k, v = (branch * mask for branch in (q, k, v))
        batch, sequence, _ = q.shape
        q = q.view(batch, sequence, self.num_heads, self.head_k_dim)
        k = k.view(batch, sequence, self.num_heads, self.head_k_dim)
        v = v.view(batch, sequence, self.num_heads, self.head_dim)
        raw_gate = self._project_gate(hidden_states, "f").view(
            batch,
            sequence,
            self.num_heads,
            self.head_k_dim,
        )
        log_decay = kda_log_decay(
            raw_gate,
            self.A_log,
            self.dt_bias,
            safe_gate=self.safe_gate,
            lower_bound=self.lower_bound,
            num_heads=self.num_heads,
            key_dim=self.head_k_dim,
            decay_rate=self.kda_decay_rate,
            gate_bias=self.kda_gate_bias,
        ) * state_mask
        beta = torch.sigmoid(self.b_proj(hidden_states)) * mask
        if self.linear_attention_mode == "recurrent":
            core, state = recurrent_kda_with_gdr_scan(
                q,
                k,
                v,
                log_decay,
                beta,
                recurrent_state,
                scale=self.chunk_scale,
                recurrent_scan_op=self.recurrent_scan_op,
                batch_size=self.batch_size,
                sequence_length=self.input_sequence_length,
                num_heads=self.num_heads,
                key_dim=self.head_k_dim,
                value_dim=self.head_dim,
            )
        else:
            core, state = chunk_kda(
                q,
                k,
                v,
                log_decay,
                beta,
                recurrent_state,
                chunk_size=self.linear_chunk_size,
                block_tri_inverse_op=self.block_tri_inverse_op,
                chunk_scan_op=self.chunk_scan_op,
                eye_block_batched=self.chunk_eye_8_batched,
                scale=self.chunk_scale,
                batch_size=self.batch_size,
                sequence_length=self.input_sequence_length,
                num_heads=self.num_heads,
                key_dim=self.head_k_dim,
                value_dim=self.head_dim,
                causal_pair_mask=self.chunk_causal_pair_mask,
                strict_pair_mask=self.chunk_strict_pair_mask,
            )
            if bool(getattr(self.chunk_scan_op, "state_is_cache", False)):
                # The unified GDRChunkScan publishes the KDA recurrent state
                # into its CacheTensor input. Do not duplicate that cache as
                # an explicit graph output.
                state = None
        output_gate = self._project_gate(hidden_states, "g").view(batch, sequence, self.num_heads, self.head_dim)
        core = self.output_norm(core) * torch.sigmoid(output_gate)
        output = self.o_proj(core.reshape(batch, sequence, -1).to(self.o_proj.weight.dtype)) * mask
        return output, (q_cache, k_cache, v_cache), state

    def _make_conv(self, source, channels: int) -> nn.Conv1d:
        conv = nn.Conv1d(
            channels,
            channels,
            self.conv_size,
            groups=channels,
            padding=self.conv_size - 1,
            bias=source.bias is not None,
            device=source.weight.device,
            dtype=source.weight.dtype,
        )
        conv.weight.data.copy_(source.weight)
        if source.bias is not None:
            conv.bias.data.copy_(source.bias)
        return conv

    def _setup(self, cfg: Optional[Dict] = None):
        self.use_cache = bool(cfg.use_cache)
        self.input_sequence_length = int(cfg.input_sequence_length)
        self.batch_size = int(cfg.get("batch_size", 1))
        self.linear_attention_mode = cfg.get("linear_attention_mode", "chunk")
        self.linear_chunk_size = int(cfg.get("linear_chunk_size", 64))
        if self.linear_attention_mode == "chunk" and self.input_sequence_length % self.linear_chunk_size:
            raise ValueError("Ling KDA chunk prefill length must be divisible by linear_chunk_size")
        key_projection = self.num_heads * self.head_k_dim
        value_projection = self.num_heads * self.head_dim
        expected_projections = {
            "q_proj": key_projection,
            "k_proj": key_projection,
            "v_proj": value_projection,
        }
        for name, expected in expected_projections.items():
            actual = int(getattr(self, name).out_features)
            if actual != expected:
                raise ValueError(
                    f"Ling KDA {name} must project to {expected} features, got {actual}"
                )
        decay_projection = getattr(self, "f_proj", None)
        if decay_projection is None:
            decay_projection = self.f_b_proj
        if int(decay_projection.out_features) != key_projection:
            raise ValueError(
                "Ling KDA decay gate must have [B,T,H,K] layout: "
                f"expected {key_projection} features, got {decay_projection.out_features}"
            )
        if self.dt_bias.numel() != key_projection:
            raise ValueError(
                "Ling KDA dt_bias must align with the per-key decay gate: "
                f"expected {key_projection} values, got {self.dt_bias.numel()}"
            )
        self.q_conv = self._make_conv(self.q_conv1d, key_projection)
        self.k_conv = self._make_conv(self.k_conv1d, key_projection)
        self.v_conv = self._make_conv(self.v_conv1d, value_projection)
        del self.q_conv1d, self.k_conv1d, self.v_conv1d
        self.conv_cache_slice = xhnn.DynamicSlice([self.conv_size], [2], [1])
        self.output_norm = RMSNorm(self.head_dim, self.o_norm.eps).to(
            device=self.o_norm.weight.device,
            dtype=self.o_norm.weight.dtype,
        )
        self.output_norm.weight.data.copy_(self.o_norm.weight)
        del self.o_norm
        for name in ("A_log", "dt_bias"):
            if name in self._parameters:
                data = self._parameters.pop(name).detach().clone()
                self.register_buffer(name, data, persistent=False)
        compute_dtype = self.q_proj.weight.dtype
        self.register_buffer(
            "kda_decay_rate",
            self.A_log.float().exp().to(compute_dtype).reshape(1, 1, self.num_heads, 1),
            persistent=False,
        )
        self.register_buffer(
            "kda_gate_bias",
            self.dt_bias.to(compute_dtype).reshape(1, 1, self.num_heads, self.head_k_dim),
            persistent=False,
        )
        self.register_buffer(
            "chunk_scale",
            torch.tensor(self.head_k_dim ** -0.5, device=self.A_log.device, dtype=compute_dtype),
            persistent=False,
        )
        causal_pair_mask = torch.tril(
            torch.ones(
                self.linear_chunk_size,
                self.linear_chunk_size,
                device=self.A_log.device,
                dtype=compute_dtype,
            )
        )
        self.register_buffer("chunk_causal_pair_mask", causal_pair_mask, persistent=False)
        self.register_buffer(
            "chunk_strict_pair_mask",
            torch.tril(causal_pair_mask, diagonal=-1),
            persistent=False,
        )
        num_chunks = max(1, math.ceil(self.input_sequence_length / self.linear_chunk_size))
        blocks = self.linear_chunk_size // 8
        eye = torch.eye(8, device=self.A_log.device, dtype=self.q_proj.weight.dtype).unsqueeze(0)
        eye = eye.expand(self.batch_size * self.num_heads * num_chunks * blocks, 8, 8).contiguous()
        self.register_buffer("chunk_eye_8_batched", eye, persistent=False)
        self.fuse_gdr_block_recurrent_ops = bool(cfg.get("fuse_gdr_block_recurrent_ops", True))
        self.fuse_gdr_ops = bool(cfg.get("fuse_gdr_ops", True))
        self.block_tri_inverse_op = (
            GDRBlockTriInverse(self.linear_chunk_size, 8)
            if self.fuse_gdr_block_recurrent_ops
            else None
        )
        self.recurrent_scan_op = (
            GDRRecurrentScan(1, False)
            if self.fuse_gdr_block_recurrent_ops
            else None
        )
        self.chunk_scan_op = (
            GDRChunkScan(num_chunks, self.num_heads, self.head_k_dim, self.head_dim, self.linear_chunk_size)
            if self.fuse_gdr_ops
            else None
        )
        return self

    def _update_cfg(self, cfg: Optional[Dict] = None):
        if cfg is None:
            return
        immutable_fields = {
            "linear_chunk_size": self.linear_chunk_size,
            "fuse_gdr_block_recurrent_ops": self.fuse_gdr_block_recurrent_ops,
            "fuse_gdr_ops": self.fuse_gdr_ops,
            "split_conv_cache": True,
        }
        for field, current_value in immutable_fields.items():
            requested = cfg.get(field, current_value)
            if requested != current_value:
                raise ValueError(
                    f"{field} is setup-time immutable for Ling KDA: "
                    f"wrapped value={current_value!r}, requested value={requested!r}. "
                    "Re-wrap the model with the requested structural config."
                )

        self.use_cache = bool(cfg.use_cache)
        self.input_sequence_length = int(cfg.get("input_sequence_length", self.input_sequence_length))
        self.batch_size = int(cfg.get("batch_size", self.batch_size))
        self.linear_attention_mode = cfg.get("linear_attention_mode", self.linear_attention_mode)
        if self.linear_attention_mode == "chunk" and self.input_sequence_length % self.linear_chunk_size:
            raise ValueError("Ling KDA chunk prefill length must be divisible by linear_chunk_size")

        num_chunks = max(1, math.ceil(self.input_sequence_length / self.linear_chunk_size))
        blocks = self.linear_chunk_size // 8
        eye = torch.eye(8, device=self.A_log.device, dtype=self.chunk_scale.dtype).unsqueeze(0)
        eye = eye.expand(self.batch_size * self.num_heads * num_chunks * blocks, 8, 8).contiguous()
        self.register_buffer("chunk_eye_8_batched", eye, persistent=False)
        if self.chunk_scan_op is not None:
            self.chunk_scan_op.num_chunks = num_chunks
        self.conv_cache_slice = xhnn.DynamicSlice([self.conv_size], [2], [1])


class _LingSparseMoeBlock(DynamicModule):
    def _selection_scores(self, logits: Tensor) -> Tensor:
        # Ling's reference router selects experts from
        # ``sigmoid(logits) + expert_bias`` in FP32.  The deployment graph is
        # FP16, where materializing sigmoid near one discards enough low bits
        # to change group/expert TopK decisions.  Use the exactly equivalent
        # score shifted by -1:
        #
        #   sigmoid(x) + bias - 1 == -sigmoid(-x) + bias
        #
        # Subtracting the same constant from every expert preserves both the
        # per-group top-2 sum ordering and the final expert ordering, while
        # the negative-sigmoid representation retains FP16 precision for the
        # high-scoring experts.  Routing weights still use the unshifted
        # sigmoid values below.
        return -torch.sigmoid(-logits) + self.expert_bias

    def _mask_excluded_experts(self, choice: Tensor, expert_mask: Tensor) -> Tensor:
        # Keep the tensor as the left operand of Sub.  The XH frontend lowers
        # Python's reflected ``1.0 - expert_mask`` as ``expert_mask - 1.0``;
        # retaining the negative floor in that expression would turn every
        # excluded expert into +32768 and make TopK select the mask complement.
        # This algebraically equivalent form has an unambiguous exported
        # direction: selected -> 0, excluded -> router_mask_floor.
        return choice * expert_mask + (expert_mask - 1.0) * (-self.router_mask_floor)

    def forward(self, hidden_states: Tensor) -> Tensor:
        batch = self.batch_size
        sequence = self.input_sequence_length
        logits = self.router(hidden_states)
        scores = torch.sigmoid(logits)
        choice = self._selection_scores(logits)
        grouped = choice.view(batch * sequence, self.n_group, self.experts_per_group)
        group_scores = self.group_top2(grouped)[0].sum(dim=-1)
        group_indices = self.group_topk(group_scores)[1]
        group_mask = group_scores * 0
        group_updates = group_scores * 0 + 1
        group_mask = self.scatter_group_mask(group_mask, group_indices, group_updates)
        expert_mask = group_mask.unsqueeze(-1).expand(
            batch * sequence,
            self.n_group,
            self.experts_per_group,
        ).reshape(batch, sequence, self.num_experts)
        # Keep the masking sentinel finite in the graph dtype.  Writing this
        # as ``(expert_mask - 1) * 1e9`` looks harmless in eager PyTorch, but
        # the FP16 exporter materializes ``1e9`` as ``+inf``.  Selected groups
        # then evaluate ``0 * inf`` to NaN and TopK chooses arbitrary experts.
        masked = self._mask_excluded_experts(choice, expert_mask)
        selected = self.expert_topk(masked)[1]
        selected_weights = self.gather_weights(scores, selected)
        selected_weights = selected_weights / (selected_weights.sum(dim=-1, keepdim=True) + 1e-20)
        selected_weights = selected_weights * self.routed_scaling_factor
        routing = self.scatter_routing(scores * 0, selected, selected_weights)
        routed = self.moeblock(hidden_states, routing, selected)
        return routed + self.shared_experts(hidden_states)

    def _setup(self, cfg: Optional[Dict] = None):
        gate = self.gate
        self.n_group = int(gate.n_group)
        self.num_experts = int(gate.weight.shape[0])
        self.experts_per_group = self.num_experts // self.n_group
        self.routed_scaling_factor = float(gate.routed_scaling_factor)
        self.router = nn.Linear(
            gate.weight.shape[1],
            gate.weight.shape[0],
            bias=False,
            device=gate.weight.device,
            dtype=gate.weight.dtype,
        )
        self.router.weight.data.copy_(gate.weight)
        self.register_buffer(
            "expert_bias",
            gate.expert_bias.detach().clone().to(self.router.weight.dtype),
            persistent=False,
        )
        self.register_buffer(
            "router_mask_floor",
            torch.tensor(
                # The exported graph is executed in FP16 even when the source
                # checkpoint was loaded as BF16.  -32768 is exactly
                # representable in BF16/FP16 and remains overwhelmingly below
                # the sigmoid router score after every supported dtype cast.
                # FP16's -65504 is not safe here: BF16 rounds it to -65536,
                # which becomes -inf when converted back to FP16.
                -32768.0,
                device=self.router.weight.device,
                dtype=self.router.weight.dtype,
            ),
            persistent=False,
        )
        self.group_top2 = xhnn.TopK(2, axis=-1)
        self.group_topk = xhnn.TopK(gate.topk_group, axis=-1)
        self.expert_topk = xhnn.TopK(gate.top_k, axis=-1)
        self.scatter_group_mask = xhnn.ScatterElements(axis=1)
        self.gather_weights = xhnn.GatherElements(axis=2)
        self.scatter_routing = xhnn.ScatterElements(axis=2)
        self.batch_size = int(cfg.get("batch_size", 1)) if cfg is not None else 1
        self.input_sequence_length = (
            int(cfg.input_sequence_length) if cfg is not None else 1
        )
        experts = self.experts
        self.moeblock = MoeBlock(
            experts[0].act_fn._get_name().lower(),
            gate.top_k,
            False,
            topk_outside=True,
        )
        device = experts[0].gate_proj.weight.device
        for name in ("gate_proj", "up_proj", "down_proj"):
            _pack_defused_expert_linear_to_moeblock(self.moeblock, experts, name, device)
        self.experts = nn.ModuleList()
        del self.gate
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return self

    def _update_cfg(self, cfg: Optional[Dict] = None):
        if cfg is None:
            return
        self.batch_size = int(cfg.get("batch_size", self.batch_size))
        self.input_sequence_length = int(
            cfg.get("input_sequence_length", self.input_sequence_length)
        )


class _LingDecoderLayer(DynamicModule):
    def forward(
        self,
        hidden_states: Tensor,
        past_seq_length: Tensor | None = None,
        current_input_length: Tensor | None = None,
        position_embeddings: tuple[Tensor, Tensor] | None = None,
        linear_attn_mask: Tensor | None = None,
        past_k_cache: Tensor | None = None,
        past_v_cache: Tensor | None = None,
        past_conv_cache=None,
        past_recurrent_state: Tensor | None = None,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        if self.layer_type == "linear_attention":
            hidden_states, conv_out, state_out = self.attention(
                hidden_states,
                conv_cache=past_conv_cache,
                recurrent_state=past_recurrent_state,
                linear_attn_mask=linear_attn_mask,
                current_input_length=current_input_length,
            )
        else:
            hidden_states = self.attention(
                hidden_states,
                past_seq_length=past_seq_length,
                current_input_length=current_input_length,
                position_embeddings=position_embeddings,
                past_k_cache=past_k_cache,
                past_v_cache=past_v_cache,
            )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.mlp(self.post_attention_layernorm(hidden_states))
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]
        hidden_states = residual + hidden_states
        if self.layer_type == "linear_attention":
            return hidden_states, conv_out, state_out
        return hidden_states

    def _setup(self, cfg=None):
        del cfg
        self.layer_type = "full_attention" if self.attention_layer_type == "attention" else "linear_attention"
        return self


class _LingTextModel(DynamicModule):
    def _setup_cos_sin(self):
        _ = self.rotary_emb.cos_cached, self.rotary_emb.sin_cached

    def _setup(self, cfg):
        self.batch_size = int(cfg.get("batch_size", 1))
        self.input_sequence_length = int(cfg.input_sequence_length)
        self.use_cache = bool(cfg.use_cache)
        self.num_logits_to_keep = int(cfg.num_logits_to_keep)
        self.max_layers = int(cfg.get("max_layers", -1) or -1)
        self.layer_types = [
            "full_attention" if layer.attention_layer_type == "attention" else "linear_attention"
            for layer in self.layers
        ]
        # PageAttention uses these identity nodes as the authoritative decoder
        # layer boundaries. Ling's sparse-MoE placeholders do not retain the
        # ``layers.N.mlp.down_proj`` names used by the legacy fallback, so emit
        # the same explicit contract as the common hybrid text model instead
        # of asking the ONNX converter to guess optimized graph topology.
        self.enable_layer_tag = bool(cfg.get("enable_layer_tag", True))
        if self.enable_layer_tag:
            self.tags = nn.ModuleList(
                [
                    xhnn.XHTag(
                        f"layer_{layer_idx}",
                        "LLM",
                        f"layer_{layer_idx}",
                    )
                    for layer_idx in range(len(self.layers))
                ]
            )
        self.llm_gather = xhnn.BatchGather(1)
        self.llm_gather.update_offset_indices(self.batch_size, self.input_sequence_length)
        self.sin_slice = xhnn.DynamicSlice([self.input_sequence_length], [2], [1])
        self.cos_slice = xhnn.DynamicSlice([self.input_sequence_length], [2], [1])
        if not hasattr(self.rotary_emb, "cos_cached"):
            self.rotary_emb.setup_after_callback = self._setup_cos_sin
        else:
            self._setup_cos_sin()
        return self

    def _update_cfg(self, cfg=None):
        if cfg is None:
            return
        self.batch_size = int(cfg.get("batch_size", self.batch_size))
        self.input_sequence_length = int(cfg.get("input_sequence_length", self.input_sequence_length))
        self.use_cache = bool(cfg.use_cache)
        self.num_logits_to_keep = int(cfg.get("num_logits_to_keep", self.num_logits_to_keep))
        self.max_layers = int(cfg.get("max_layers", self.max_layers) or -1)
        self.llm_gather.update_offset_indices(self.batch_size, self.input_sequence_length)
        self.sin_slice = xhnn.DynamicSlice([self.input_sequence_length], [2], [1])
        self.cos_slice = xhnn.DynamicSlice([self.input_sequence_length], [2], [1])

    def forward(
        self,
        inputs_embeds: Tensor,
        past_seq_length: Tensor,
        current_input_length: Tensor,
        linear_attn_mask: Tensor,
        past_key_cache=None,
        past_value_cache=None,
        past_conv_cache=None,
        past_recurrent_state=None,
    ):
        cos = self.cos_slice(self.rotary_emb.cos_cached, past_seq_length)
        sin = self.sin_slice(self.rotary_emb.sin_cached, past_seq_length)
        position_embeddings = (cos, sin)
        hidden_states = inputs_embeds
        conv_outputs, state_outputs = [], []
        full_idx = linear_idx = 0
        for layer_idx, layer in enumerate(self.layers):
            layer_type = self.layer_types[layer_idx]
            if layer_type == "full_attention":
                hidden_states = layer(
                    hidden_states,
                    past_seq_length=past_seq_length,
                    current_input_length=current_input_length,
                    position_embeddings=position_embeddings,
                    past_k_cache=None if past_key_cache is None else past_key_cache[full_idx],
                    past_v_cache=None if past_value_cache is None else past_value_cache[full_idx],
                )
                full_idx += 1
            else:
                caches = _select_linear_attn_conv_cache(
                    past_conv_cache,
                    linear_idx,
                    split_conv_cache=True,
                )
                hidden_states, conv_out, state_out = layer(
                    hidden_states,
                    current_input_length=current_input_length,
                    linear_attn_mask=linear_attn_mask,
                    past_conv_cache=caches,
                    past_recurrent_state=None if past_recurrent_state is None else past_recurrent_state[linear_idx],
                )
                conv_outputs.extend(conv_out)
                if state_out is not None:
                    state_outputs.append(state_out)
                linear_idx += 1
            if self.enable_layer_tag:
                hidden_states = self.tags[layer_idx](hidden_states)
            if self.max_layers > 0 and layer_idx + 1 >= self.max_layers:
                break
        if self.num_logits_to_keep == 1:
            hidden_states = self.llm_gather(hidden_states, current_input_length - 1)
        return self.norm(hidden_states), conv_outputs, state_outputs


class _LingForCausalLM(DynamicModule):
    def forward(
        self,
        inputs_embeds: Tensor,
        past_seq_length: Tensor,
        current_input_length: Tensor,
        linear_attn_mask: Tensor,
        past_key_cache=None,
        past_value_cache=None,
        past_conv_cache=None,
        past_recurrent_state=None,
    ):
        hidden, conv_out, state_out = self.model(
            inputs_embeds,
            past_seq_length,
            current_input_length,
            linear_attn_mask,
            past_key_cache,
            past_value_cache,
            past_conv_cache,
            past_recurrent_state,
        )
        return self.lm_head(hidden), conv_out, state_out

    def _setup(self, cfg=None):
        del cfg
        return self


_WRAPPERS = {
    "BailingMoeV3RMSNorm": _LingRMSNorm,
    "BailingMoeV3RotaryEmbedding": _LingRotaryEmbedding,
    "BailingMoeV3MultiLatentAttention": _LingMLAAttention,
    "BailingMoeV3KimiDeltaAttention": _LingKDAAttention,
    "BailingMoeV3SparseMoeBlock": _LingSparseMoeBlock,
    "BailingMoeV3DecoderLayer": _LingDecoderLayer,
    "BailingMoeV3Model": _LingTextModel,
    "BailingMoeV3ForCausalLM": _LingForCausalLM,
}


def _runtime_classes(hf_model=None):
    classes = {}
    if hf_model is not None:
        for module in hf_model.modules():
            classes.setdefault(type(module).__name__, type(module))
    for module_name, module in tuple(sys.modules.items()):
        if "modeling_bailing_moe_v3" not in module_name or module is None:
            continue
        for name in _WRAPPERS:
            cls = getattr(module, name, None)
            if isinstance(cls, type):
                classes.setdefault(name, cls)
    return classes


def register_wrap_modules(hf_model=None) -> None:
    classes = _runtime_classes(hf_model)
    for name, wrapper in _WRAPPERS.items():
        hf_cls = classes.get(name)
        if hf_cls is None or hf_cls in XHLLM_TRACEABLE_MODULES:
            continue
        XHLLM_TRACEABLE_MODULES.register_module({hf_cls: name}, wrapper)


__all__ = ["register_wrap_modules"]
