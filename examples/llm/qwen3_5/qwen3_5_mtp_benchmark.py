"""Qwen3.5 MTP (Multi-Token Prediction) Benchmark

Tests the built-in MTP capability of Qwen3.5 for speculative decoding.
HF Transformers ignores MTP weights (`_keys_to_ignore_on_load_unexpected = [r"^mtp.*"]`);
this script loads them manually from safetensors and implements the MTP head.

Measures:
  1. Acceptance rate  — batch evaluation (no KV-cache complexity)
  2. Latency overhead — MTP-step vs main-model-step wall-clock
  3. Speculative decode — actual end-to-end speedup with MTP-1 drafting

Reference architecture from vLLM Qwen3NextMultiTokenPredictor:
  embeds = pre_fc_norm_embedding(embed_tokens(next_token))
  hidden = pre_fc_norm_hidden(main_model_final_norm_hidden)
  x = fc(cat([embeds, hidden], dim=-1))
  x = decoder_layer(x)   # single full-attention layer
  x = norm(x)
  logits = lm_head(x)    # shared with main model
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

# ════════════════════════════════════════════════════════════════
# Constants
# ════════════════════════════════════════════════════════════════

DTYPE_MAP = {
    "fp16": torch.float16,
    "float16": torch.float16,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp32": torch.float32,
    "float32": torch.float32,
    "auto": "auto",
}

DEDICATED_MTP_HEAD_KEY = "mtp.lm_head_weight"
LM_NORM_SCALE_KEY = "mtp.language_model_norm_scale"
MTP_NORM_SCALE_KEY = "mtp.mtp_norm_scale"
LM_NORM_ROTATED_MATRIX_KEY = "mtp.language_model_norm_rotated_matrix"
MTP_NORM_ROTATED_MATRIX_KEY = "mtp.mtp_norm_rotated_matrix"


# ════════════════════════════════════════════════════════════════
# Rotary helpers  (must match transformers Qwen3_5 exactly)
# ════════════════════════════════════════════════════════════════


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)
    return torch.cat([q_embed, q_pass], dim=-1), torch.cat([k_embed, k_pass], dim=-1)


# ════════════════════════════════════════════════════════════════
# MTP Head sub-modules
# ════════════════════════════════════════════════════════════════


class RMSNorm(nn.Module):
    """Qwen3.5-style RMSNorm: output = rms_norm(x) * (1 + weight).  weight inits to zeros."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (normed * (1.0 + self.weight.float())).type_as(x)


class MTPAttention(nn.Module):
    """Gated full-attention with GQA and partial rotary (matches Qwen3_5Attention)."""

    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int, head_dim: int, eps: float):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_kv_groups = num_heads // num_kv_heads
        self.scaling = head_dim**-0.5

        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim * 2, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
        self.q_norm = RMSNorm(head_dim, eps=eps)
        self.k_norm = RMSNorm(head_dim, eps=eps)

    def forward(
        self,
        x: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        kv_cache: dict | None = None,
    ) -> torch.Tensor:
        bsz, seq_len, _ = x.shape

        # Q with gating — q_proj outputs 2× for (query, gate)
        qg = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim * 2)
        query, gate = qg.chunk(2, dim=-1)  # each [B, S, heads, D]
        gate = gate.reshape(bsz, seq_len, -1)  # [B, S, heads*D]

        query = self.q_norm(query).transpose(1, 2)  # [B, heads, S, D]
        key = self.k_norm(self.k_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim)).transpose(1, 2)
        value = self.v_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query, key = apply_rotary_pos_emb(query, key, cos, sin)

        # KV cache (stores un-expanded kv_heads)
        if kv_cache is not None:
            if "key" in kv_cache:
                key = torch.cat([kv_cache["key"], key], dim=2)
                value = torch.cat([kv_cache["value"], value], dim=2)
            kv_cache["key"] = key
            kv_cache["value"] = value

        # GQA expand
        if self.num_kv_groups > 1:
            key = key.repeat_interleave(self.num_kv_groups, dim=1)
            value = value.repeat_interleave(self.num_kv_groups, dim=1)

        # SDPA — causal only when q_len == kv_len (batch mode)
        is_causal = seq_len == key.size(2) and seq_len > 1
        attn_out = F.scaled_dot_product_attention(query, key, value, is_causal=is_causal, scale=self.scaling)

        attn_out = attn_out.transpose(1, 2).reshape(bsz, seq_len, -1)
        attn_out = attn_out * torch.sigmoid(gate)
        return self.o_proj(attn_out)


class MTPMLP(nn.Module):
    """SwiGLU MLP: down(silu(gate(x)) * up(x))."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ── MoE MTP MLP (for 35B-A3B) ──────────────────────────────────


class MTPMoEExpert(nn.Module):
    """Single SwiGLU expert for MoE MTP MLP."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MTPSparseMoeBlock(nn.Module):
    """Sparse MoE MLP matching Qwen3_5MoeSparseMoeBlock weight structure.

    State-dict keys:
      gate.weight, experts.{i}.{gate,up,down}_proj.weight,
      shared_expert.{gate,up,down}_proj.weight, shared_expert_gate.weight
    """

    def __init__(
        self,
        hidden_size: int,
        moe_intermediate_size: int,
        shared_expert_intermediate_size: int,
        num_experts: int,
        num_experts_per_tok: int,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = num_experts_per_tok
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        self.experts = nn.ModuleList(
            [MTPMoEExpert(hidden_size, moe_intermediate_size) for _ in range(num_experts)]
        )
        self.shared_expert = MTPMLP(hidden_size, shared_expert_intermediate_size)
        self.shared_expert_gate = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x_flat = x.view(-1, orig_shape[-1])

        # Shared expert
        shared_out = self.shared_expert(x_flat)
        shared_out = torch.sigmoid(self.shared_expert_gate(x_flat)) * shared_out

        # Router
        logits = self.gate(x_flat)
        probs = F.softmax(logits, dim=-1, dtype=torch.float)
        top_vals, top_idx = torch.topk(probs, self.top_k, dim=-1)
        top_vals = (top_vals / top_vals.sum(dim=-1, keepdim=True)).to(x.dtype)

        # Expert dispatch
        out = torch.zeros_like(x_flat)
        expert_mask = F.one_hot(top_idx, num_classes=self.num_experts).permute(2, 1, 0)
        expert_hit = expert_mask.sum(dim=(-1, -2)).nonzero(as_tuple=True)[0]

        for eidx in expert_hit:
            top_k_pos, token_idx = torch.where(expert_mask[eidx])
            expert_out = self.experts[eidx](x_flat[token_idx])
            weights = top_vals[token_idx, top_k_pos].unsqueeze(-1)
            out.index_add_(0, token_idx, (expert_out * weights).to(out.dtype))

        out = out + shared_out
        return out.view(orig_shape)


class MTPDecoderLayer(nn.Module):
    """Pre-norm transformer decoder layer (attention + MLP with residuals)."""

    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int, head_dim: int, intermediate_size: int, eps: float):
        super().__init__()
        self.input_layernorm = RMSNorm(hidden_size, eps=eps)
        self.self_attn = MTPAttention(hidden_size, num_heads, num_kv_heads, head_dim, eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=eps)
        self.mlp = MTPMLP(hidden_size, intermediate_size)

    def forward(self, x: torch.Tensor, position_embeddings: tuple, kv_cache: dict | None = None) -> torch.Tensor:
        residual = x
        x = self.self_attn(self.input_layernorm(x), position_embeddings, kv_cache)
        x = residual + x

        residual = x
        x = self.mlp(self.post_attention_layernorm(x))
        x = residual + x
        return x


class MTPMoEDecoderLayer(nn.Module):
    """Pre-norm decoder layer with MoE MLP (for 35B-A3B)."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        moe_intermediate_size: int,
        shared_expert_intermediate_size: int,
        num_experts: int,
        num_experts_per_tok: int,
        eps: float,
    ):
        super().__init__()
        self.input_layernorm = RMSNorm(hidden_size, eps=eps)
        self.self_attn = MTPAttention(hidden_size, num_heads, num_kv_heads, head_dim, eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=eps)
        self.mlp = MTPSparseMoeBlock(
            hidden_size, moe_intermediate_size, shared_expert_intermediate_size,
            num_experts, num_experts_per_tok,
        )

    def forward(self, x: torch.Tensor, position_embeddings: tuple, kv_cache: dict | None = None) -> torch.Tensor:
        residual = x
        x = self.self_attn(self.input_layernorm(x), position_embeddings, kv_cache)
        x = residual + x

        residual = x
        x = self.mlp(self.post_attention_layernorm(x))
        x = residual + x
        return x


# ════════════════════════════════════════════════════════════════
# Qwen3_5MTPHead — complete MTP module
# ════════════════════════════════════════════════════════════════


class Qwen3_5MTPHead(nn.Module):
    """
    State-dict keys (after stripping ``mtp.`` prefix) match exactly:
      fc.weight, pre_fc_norm_hidden.weight, pre_fc_norm_embedding.weight,
      layers.0.{input_layernorm,self_attn.*,post_attention_layernorm,mlp.*}.weight,
      norm.weight   (15 tensors for dense; 785 for MoE)

    Shared modules (embed_tokens, lm_head, rotary_emb) are set AFTER
    ``load_state_dict(strict=…)`` to avoid missing-key errors.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        intermediate_size: int,
        eps: float,
        *,
        moe_config: dict | None = None,
    ):
        super().__init__()
        self.pre_fc_norm_embedding = RMSNorm(hidden_size, eps=eps)
        self.pre_fc_norm_hidden = RMSNorm(hidden_size, eps=eps)
        self.fc = nn.Linear(hidden_size * 2, hidden_size, bias=False)

        if moe_config is not None:
            layer = MTPMoEDecoderLayer(
                hidden_size, num_heads, num_kv_heads, head_dim,
                moe_intermediate_size=moe_config["moe_intermediate_size"],
                shared_expert_intermediate_size=moe_config["shared_expert_intermediate_size"],
                num_experts=moe_config["num_experts"],
                num_experts_per_tok=moe_config["num_experts_per_tok"],
                eps=eps,
            )
        else:
            layer = MTPDecoderLayer(hidden_size, num_heads, num_kv_heads, head_dim, intermediate_size, eps)

        self.layers = nn.ModuleList([layer])
        self.norm = RMSNorm(hidden_size, eps=eps)
        self._shared: dict = {}
        self._source_norm_transforms: dict[str, torch.Tensor | None] = {"llm": None, "mtp": None}

    # ── shared-module plumbing ──────────────────────────────────

    def set_shared_modules(self, embed_tokens: nn.Embedding, lm_head: nn.Linear, rotary_emb: nn.Module):
        self._shared = {"embed_tokens": embed_tokens, "lm_head": lm_head, "rotary_emb": rotary_emb}

    def set_source_norm_transforms(
        self,
        *,
        llm_norm_transform: torch.Tensor | None = None,
        mtp_norm_transform: torch.Tensor | None = None,
    ):
        self._source_norm_transforms = {"llm": llm_norm_transform, "mtp": mtp_norm_transform}

    @property
    def _mtp_device(self) -> torch.device:
        return self.fc.weight.device

    def _embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        et = self._shared["embed_tokens"]
        return et(token_ids.to(et.weight.device)).to(self._mtp_device)

    def _lm_head_forward(self, x: torch.Tensor) -> torch.Tensor:
        lm = self._shared["lm_head"]
        x = x.to(lm.weight.device)
        if x.dtype != lm.weight.dtype:
            x = x.to(lm.weight.dtype)
        return lm(x)

    def _position_embeddings(self, x: torch.Tensor, position_ids: torch.Tensor):
        re = self._shared["rotary_emb"]
        cos, sin = re(x, position_ids)
        d = self._mtp_device
        return cos.to(device=d, dtype=x.dtype), sin.to(device=d, dtype=x.dtype)

    def _apply_source_norm_transform(self, x: torch.Tensor, source: str) -> torch.Tensor:
        transform = self._source_norm_transforms.get(source)
        if transform is None:
            return x
        transform = transform.to(device=x.device)
        if transform.ndim == 1:
            return x.float() * transform.float().view(1, 1, -1)
        return torch.matmul(x.float(), transform.float())

    # ── forward paths ───────────────────────────────────────────

    def forward_batch(
        self,
        final_norm_hidden: torch.Tensor,
        next_token_ids: torch.Tensor,
        positions: torch.Tensor | None = None,
        kv_cache: dict | None = None,
        hidden_source: str = "llm",
    ) -> torch.Tensor:
        """Batch forward (optionally populates *kv_cache* for later step calls).

        Args:
            final_norm_hidden: ``[B, L, H]``  hidden states after the main model final RMSNorm
            next_token_ids:  ``[B, L]``      shifted token ids  (token[i+1] at position i)
            positions:       ``[L]``         absolute position indices (default ``0..L-1``)
            kv_cache:        if provided, MTP attention KV will be stored here for
                             subsequent ``forward_step`` calls
        Returns:
            logits ``[B, L, V]``
        """
        d = self._mtp_device
        bsz, seq_len = next_token_ids.shape
        compute_dtype = self.fc.weight.dtype

        embeds = self.pre_fc_norm_embedding(self._embed(next_token_ids).to(compute_dtype))
        hidden = self.pre_fc_norm_hidden(
            self._apply_source_norm_transform(final_norm_hidden.to(d), hidden_source).to(compute_dtype)
        )
        x = self.fc(torch.cat([embeds, hidden], dim=-1))

        if positions is None:
            positions = torch.arange(seq_len, device=d)
        pos_ids = positions.to(d).view(1, 1, -1).expand(3, bsz, -1)
        pos_emb = self._position_embeddings(x, pos_ids)

        x = self.layers[0](x, pos_emb, kv_cache)
        x = self.norm(x)
        return self._lm_head_forward(x)

    def forward_step(
        self,
        final_norm_hidden: torch.Tensor,
        next_token_id: torch.Tensor,
        position: int,
        kv_cache: dict | None = None,
        hidden_source: str = "llm",
        return_hidden: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Single-step forward for speculative decoding.

        Args:
            final_norm_hidden: ``[1, 1, H]``
            next_token_id:   ``[1, 1]``
            position:        absolute position index of the hidden state
            kv_cache:        mutable dict (``{"key": …, "value": …}`` or empty ``{}``)
            return_hidden:   if True, also return the MTP hidden after final norm
        Returns:
            logits ``[1, 1, V]``  or  (logits, hidden ``[1, 1, H]``)
        """
        d = self._mtp_device
        compute_dtype = self.fc.weight.dtype

        embeds = self.pre_fc_norm_embedding(self._embed(next_token_id).to(compute_dtype))
        hidden = self.pre_fc_norm_hidden(
            self._apply_source_norm_transform(final_norm_hidden.to(d), hidden_source).to(compute_dtype)
        )
        x = self.fc(torch.cat([embeds, hidden], dim=-1))

        pos = torch.tensor([position], device=d)
        pos_ids = pos.view(1, 1, -1).expand(3, 1, -1)
        pos_emb = self._position_embeddings(x, pos_ids)

        x = self.layers[0](x, pos_emb, kv_cache)
        x = self.norm(x)
        final_hidden = x  # [1, 1, H] — MTP hidden after final norm
        logits = self._lm_head_forward(x)
        if return_hidden:
            return logits, final_hidden
        return logits


# ════════════════════════════════════════════════════════════════
# Weight loading
# ════════════════════════════════════════════════════════════════


def _get_text_model(model):
    """Return the Qwen3_5TextModel regardless of wrapper."""
    if hasattr(model.model, "embed_tokens"):
        return model.model
    if hasattr(model.model, "language_model"):
        return model.model.language_model
    raise ValueError(f"Cannot locate TextModel inside {type(model)}")


def _get_text_config(model):
    cfg = model.config
    return getattr(cfg, "text_config", cfg)


def load_mtp_weights(model_path: str | Path) -> tuple[
    dict[str, torch.Tensor],
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    model_path = Path(model_path)
    state: dict[str, torch.Tensor] = {}
    dedicated_lm_head: torch.Tensor | None = None
    llm_norm_scale: torch.Tensor | None = None
    mtp_norm_scale: torch.Tensor | None = None
    llm_norm_rotated_matrix: torch.Tensor | None = None
    mtp_norm_rotated_matrix: torch.Tensor | None = None
    for sf in sorted(model_path.glob("*.safetensors")):
        with safe_open(str(sf), framework="pt", device="cpu") as f:
            for key in f.keys():
                if key == DEDICATED_MTP_HEAD_KEY:
                    dedicated_lm_head = f.get_tensor(key)
                elif key == LM_NORM_SCALE_KEY:
                    llm_norm_scale = f.get_tensor(key)
                elif key == MTP_NORM_SCALE_KEY:
                    mtp_norm_scale = f.get_tensor(key)
                elif key == LM_NORM_ROTATED_MATRIX_KEY:
                    llm_norm_rotated_matrix = f.get_tensor(key)
                elif key == MTP_NORM_ROTATED_MATRIX_KEY:
                    mtp_norm_rotated_matrix = f.get_tensor(key)
                elif key.startswith("mtp."):
                    state[key[4:]] = f.get_tensor(key)
    if not state:
        raise RuntimeError(f"No MTP weights found in {model_path}/*.safetensors")
    print(f"  Loaded {len(state)} MTP tensors from safetensors")
    if dedicated_lm_head is not None:
        print("  Loaded dedicated rotated MTP lm_head_weight")
    if llm_norm_rotated_matrix is not None and mtp_norm_rotated_matrix is not None:
        print("  Loaded rotated MTP source norm transforms (LLM/MTP)")
    elif llm_norm_scale is not None and mtp_norm_scale is not None:
        print("  Loaded rotated MTP source norm scales (LLM/MTP)")
    return state, dedicated_lm_head, llm_norm_scale, mtp_norm_scale, llm_norm_rotated_matrix, mtp_norm_rotated_matrix


def build_mtp_head(model, model_path: str, dtype: str) -> Qwen3_5MTPHead:
    cfg = _get_text_config(model)
    hs = cfg.hidden_size
    nh = cfg.num_attention_heads
    nkv = cfg.num_key_value_heads
    hd = getattr(cfg, "head_dim", hs // nh)
    inter = getattr(cfg, "intermediate_size", None)
    eps = cfg.rms_norm_eps

    # Detect MoE
    num_experts = getattr(cfg, "num_experts", None)
    moe_config = None
    if num_experts is not None and num_experts > 1:
        moe_config = {
            "moe_intermediate_size": cfg.moe_intermediate_size,
            "shared_expert_intermediate_size": cfg.shared_expert_intermediate_size,
            "num_experts": num_experts,
            "num_experts_per_tok": cfg.num_experts_per_tok,
        }
        print(f"  Config: hidden={hs} heads={nh} kv_heads={nkv} head_dim={hd} MoE={num_experts}×{cfg.moe_intermediate_size}")
    else:
        print(f"  Config: hidden={hs} heads={nh} kv_heads={nkv} head_dim={hd} inter={inter}")

    head = Qwen3_5MTPHead(hs, nh, nkv, hd, inter or hs * 4, eps, moe_config=moe_config)
    (
        sd,
        dedicated_lm_head_weight,
        llm_norm_scale,
        mtp_norm_scale,
        llm_norm_rotated_matrix,
        mtp_norm_rotated_matrix,
    ) = load_mtp_weights(model_path)
    head.load_state_dict(sd, strict=True)

    tm = _get_text_model(model)
    lm_head = model.lm_head
    if dedicated_lm_head_weight is not None:
        lm_head = nn.Linear(dedicated_lm_head_weight.shape[1], dedicated_lm_head_weight.shape[0], bias=False)
        lm_head.weight = nn.Parameter(
            dedicated_lm_head_weight.to(device=model.lm_head.weight.device, dtype=model.lm_head.weight.dtype)
        )
    head.set_shared_modules(tm.embed_tokens, lm_head, tm.rotary_emb)
    head.set_source_norm_transforms(
        llm_norm_transform=llm_norm_rotated_matrix if llm_norm_rotated_matrix is not None else llm_norm_scale,
        mtp_norm_transform=mtp_norm_rotated_matrix if mtp_norm_rotated_matrix is not None else mtp_norm_scale,
    )

    target_dev = lm_head.weight.device
    target_dtype = lm_head.weight.dtype if dtype == "auto" else DTYPE_MAP.get(dtype, torch.bfloat16)
    head = head.to(device=target_dev, dtype=target_dtype)
    head.eval()

    n_params = sum(p.numel() for p in head.parameters())
    print(f"  MTP head: {n_params:,} params ({n_params / 1e6:.1f}M) on {target_dev} ({target_dtype})")
    return head


# ════════════════════════════════════════════════════════════════
# Final-norm hidden-state capture
# ════════════════════════════════════════════════════════════════


class AfterNormCapture:
    """``register_forward_hook`` on ``TextModel.norm`` to grab hidden states
    *after* the final RMSNorm — exactly what MTP needs."""

    def __init__(self, model):
        self.hidden_states: list[torch.Tensor] = []
        self._hook = _get_text_model(model).norm.register_forward_hook(self._fn)

    def _fn(self, _module, _args, output):
        self.hidden_states.append(output.detach())

    def reset(self):
        self.hidden_states.clear()

    def remove(self):
        self._hook.remove()

    def get_all(self) -> torch.Tensor:
        if len(self.hidden_states) == 1:
            return self.hidden_states[0]
        return torch.cat(self.hidden_states, dim=1)


# ════════════════════════════════════════════════════════════════
# 1.  Batch acceptance-rate measurement
# ════════════════════════════════════════════════════════════════


@torch.no_grad()
def measure_acceptance_rate(
    model,
    mtp_head: Qwen3_5MTPHead,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 128,
    system_prompt: str = "",
    enable_thinking: bool | None = None,
) -> dict:
    """Generate → single forward pass to capture all final-norm hidden → MTP batch → compare."""
    device = model.device

    # Tokenise
    msgs = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
    text = _apply_chat_template(tokenizer, msgs, enable_thinking=enable_thinking)
    inputs = tokenizer([text], return_tensors="pt").to(device)
    prompt_len = inputs.input_ids.shape[1]

    # Greedy generate
    gen_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    total_len = gen_ids.shape[1]
    gen_len = total_len - prompt_len
    if gen_len < 3:
        return {"error": "Generated < 3 tokens — too short for MTP eval", "gen_len": gen_len}

    # Full forward pass to get final-norm hidden states
    cap = AfterNormCapture(model)
    cap.reset()
    model(input_ids=gen_ids, use_cache=False)
    final_hidden = cap.get_all()  # [1, T, H]
    cap.remove()

    # MTP: position i → (hidden[i], embed(token[i+1])) → predicts token[i+2]
    mtp_hidden = final_hidden[:, :-2, :]  # [1, T-2, H]
    mtp_tokens = gen_ids[:, 1:-1]      # [1, T-2]
    targets = gen_ids[0, 2:]           # [T-2]

    logits = mtp_head.forward_batch(mtp_hidden, mtp_tokens)  # [1, T-2, V]
    preds = logits[0].argmax(dim=-1)                         # [T-2]
    targets_dev = targets.to(preds.device)

    correct = preds == targets_dev

    # Regions
    gen_start = max(0, prompt_len - 2)  # first position whose target is in the generated part
    overall_acc = correct.float().mean().item()
    prompt_acc = correct[:gen_start].float().mean().item() if gen_start > 0 else float("nan")
    gen_acc = correct[gen_start:].float().mean().item() if gen_start < len(correct) else float("nan")

    # Top-5
    top5 = logits[0].topk(5, dim=-1).indices
    top5_hit = (top5 == targets_dev.unsqueeze(-1)).any(dim=-1)
    top5_overall = top5_hit.float().mean().item()
    top5_gen = top5_hit[gen_start:].float().mean().item() if gen_start < len(correct) else float("nan")

    # Sample predictions from the generated region
    samples: list[str] = []
    for i in range(gen_start, min(gen_start + 10, len(correct))):
        t_tok = tokenizer.decode([targets[i].item()])
        p_tok = tokenizer.decode([preds[i].item()])
        mark = "✓" if correct[i].item() else "✗"
        samples.append(f"    pos {i + 2:4d}: target={t_tok!r:12s}  pred={p_tok!r:12s}  {mark}")

    output_text = tokenizer.decode(gen_ids[0, prompt_len:].tolist(), skip_special_tokens=True)

    return {
        "prompt_len": prompt_len,
        "gen_len": gen_len,
        "total_len": total_len,
        "overall_acceptance_rate": overall_acc,
        "prompt_acceptance_rate": prompt_acc,
        "generation_acceptance_rate": gen_acc,
        "top5_acceptance_rate": top5_overall,
        "top5_generation_acceptance_rate": top5_gen,
        "output_text": output_text[:300],
        "samples": samples,
    }


# ════════════════════════════════════════════════════════════════
# 2.  Latency benchmark
# ════════════════════════════════════════════════════════════════


@torch.no_grad()
def benchmark_timing(
    model,
    mtp_head: Qwen3_5MTPHead,
    tokenizer,
    prompt: str,
    system_prompt: str = "",
    warmup: int = 5,
    num_steps: int = 50,
    enable_thinking: bool | None = None,
) -> dict:
    device = model.device
    msgs = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
    text = _apply_chat_template(tokenizer, msgs, enable_thinking=enable_thinking)
    inputs = tokenizer([text], return_tensors="pt").to(device)

    # ── prefill ─────────────────────────────────────────────────
    cap = AfterNormCapture(model)
    cap.reset()
    outputs = model(**inputs, use_cache=True)
    past_kv = outputs.past_key_values
    next_tok = outputs.logits[:, -1:, :].argmax(dim=-1)
    final_h = cap.hidden_states[0][:, -1:, :]

    # ── warm-up main-model decode ───────────────────────────────
    for _ in range(warmup):
        cap.reset()
        kv_len = past_kv.get_seq_length()
        mask = torch.ones(1, kv_len + 1, device=device, dtype=torch.long)
        outputs = model(input_ids=next_tok, attention_mask=mask, past_key_values=past_kv, use_cache=True)
        past_kv = outputs.past_key_values
        next_tok = outputs.logits[:, -1:, :].argmax(dim=-1)
        final_h = cap.hidden_states[0][:, -1:, :]

    # ── timed main-model decode ─────────────────────────────────
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(num_steps):
        cap.reset()
        kv_len = past_kv.get_seq_length()
        mask = torch.ones(1, kv_len + 1, device=device, dtype=torch.long)
        outputs = model(input_ids=next_tok, attention_mask=mask, past_key_values=past_kv, use_cache=True)
        past_kv = outputs.past_key_values
        next_tok = outputs.logits[:, -1:, :].argmax(dim=-1)
        final_h = cap.hidden_states[0][:, -1:, :]
    torch.cuda.synchronize()
    main_ms = (time.perf_counter() - t0) / num_steps * 1000

    cap.remove()

    # ── warm-up MTP decode ──────────────────────────────────────
    mtp_d = mtp_head._mtp_device
    dummy_h = final_h.to(mtp_d)
    dummy_t = next_tok.to(mtp_d)
    for i in range(warmup):
        mtp_head.forward_step(dummy_h, dummy_t, position=i, kv_cache=None)

    # ── timed MTP decode (with growing KV cache) ────────────────
    mtp_kv: dict = {}
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(num_steps):
        mtp_head.forward_step(dummy_h, dummy_t, position=i, kv_cache=mtp_kv)
    torch.cuda.synchronize()
    mtp_ms = (time.perf_counter() - t0) / num_steps * 1000

    overhead = mtp_ms / main_ms if main_ms > 0 else float("inf")
    return {
        "main_model_ms": round(main_ms, 2),
        "mtp_head_ms": round(mtp_ms, 2),
        "overhead_ratio": round(overhead, 4),
        "num_steps": num_steps,
    }


# ════════════════════════════════════════════════════════════════
# 3.  Speculative decoding  (MTP-1)
# ════════════════════════════════════════════════════════════════


@torch.no_grad()
def baseline_decode(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 128,
    system_prompt: str = "",
    enable_thinking: bool | None = None,
) -> dict:
    """Standard autoregressive greedy decode (for comparison)."""
    device = model.device
    msgs = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
    text = _apply_chat_template(tokenizer, msgs, enable_thinking=enable_thinking)
    inputs = tokenizer([text], return_tensors="pt").to(device)
    eos = tokenizer.eos_token_id

    outputs = model(**inputs, use_cache=True)
    past_kv = outputs.past_key_values
    next_tok = outputs.logits[:, -1:, :].argmax(dim=-1)
    tokens = [next_tok.item()]

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(max_new_tokens - 1):
        if tokens[-1] == eos:
            break
        kv_len = past_kv.get_seq_length()
        mask = torch.ones(1, kv_len + 1, device=device, dtype=torch.long)
        outputs = model(input_ids=next_tok, attention_mask=mask, past_key_values=past_kv, use_cache=True)
        past_kv = outputs.past_key_values
        next_tok = outputs.logits[:, -1:, :].argmax(dim=-1)
        tokens.append(next_tok.item())
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    return {
        "tokens": tokens,
        "text": tokenizer.decode(tokens, skip_special_tokens=True),
        "num_tokens": len(tokens),
        "elapsed_s": round(elapsed, 3),
        "tok_per_s": round(len(tokens) / elapsed, 2) if elapsed > 0 else 0,
    }


@torch.no_grad()
def speculative_decode(
    model,
    mtp_head: Qwen3_5MTPHead,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 128,
    system_prompt: str = "",
    enable_thinking: bool | None = None,
) -> dict:
    """MTP-1 speculative decoding with sequential verification.

    **Why sequential?**  Qwen3.5 uses 48 DeltaNet linear-attention layers whose
    recurrent states are order-dependent.  Processing ``[next_tok, draft_tok]``
    in a single forward pass produces *different* logits than processing them
    one-at-a-time (verified empirically: batch-verify gives wrong argmax).
    Therefore we verify the draft by processing ``next_tok`` alone, comparing
    ``argmax(logits)`` against ``draft_tok``.

    Algorithm per round:
      1. Process ``next_tok`` through the main model → logits → ``verified_tok``.
      2. If ``verified_tok == draft_tok`` (accept):
         process ``draft_tok`` → logits → ``bonus_tok``, commit 2 tokens.
      3. If ``verified_tok != draft_tok`` (reject):
         commit 1 token (``next_tok``), set ``next_tok = verified_tok``.

    Because each verify is a single-token forward, no state rollback is needed.
    The trade-off: on accept we need 2 main forwards instead of 1, so the
    theoretical throughput gain from speculation is modest (tokens generated /
    main-model forward ≈ 1.0× at 95% acceptance).  The benchmark accurately
    measures what the HF DeltaNet path achieves; a fused kernel (vLLM) can
    batch-verify correctly and would yield higher speedups.
    """
    device = model.device
    eos = tokenizer.eos_token_id

    msgs = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
    text = _apply_chat_template(tokenizer, msgs, enable_thinking=enable_thinking)
    inputs = tokenizer([text], return_tensors="pt").to(device)
    prompt_ids = inputs.input_ids  # [1, prompt_len]
    prompt_len = prompt_ids.shape[1]

    # ── prefill (main model) ────────────────────────────────────
    cap = AfterNormCapture(model)
    cap.reset()
    outputs = model(**inputs, use_cache=True)
    past_kv = outputs.past_key_values
    next_tok = outputs.logits[:, -1:, :].argmax(dim=-1)
    prefill_final_hidden = cap.hidden_states[0]  # [1, prompt_len, H]
    committed_len = prompt_len

    # ── prefill (MTP) — populate MTP KV cache with prompt context
    mtp_kv: dict = {}
    if prompt_len >= 2:
        mtp_head.forward_batch(
            prefill_final_hidden[:, :-1, :],
            prompt_ids[:, 1:],
            positions=torch.arange(prompt_len - 1, device=mtp_head._mtp_device),
            kv_cache=mtp_kv,
        )

    # First MTP step
    mtp_logits = mtp_head.forward_step(
        prefill_final_hidden[:, -1:, :], next_tok,
        position=committed_len - 1, kv_cache=mtp_kv,
    )
    draft_tok = mtp_logits[:, -1:, :].argmax(dim=-1)

    tokens: list[int] = []
    num_accepts = 0
    num_rounds = 0
    num_main_fwd = 0  # main-model forward calls (excludes prefill)

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    while len(tokens) < max_new_tokens:
        num_rounds += 1

        # ── step 1: process next_tok (always needed) ────────────
        num_main_fwd += 1
        kv_len = past_kv.get_seq_length()
        mask = torch.ones(1, kv_len + 1, device=device, dtype=torch.long)
        cap.reset()
        out1 = model(input_ids=next_tok, attention_mask=mask,
                      past_key_values=past_kv, use_cache=True)
        past_kv = out1.past_key_values
        hidden1 = cap.hidden_states[0]  # [1, 1, H]
        verified_tok = out1.logits[:, -1:, :].argmax(dim=-1)

        if verified_tok.item() == draft_tok.item():
            # ── accept ──────────────────────────────────────────
            num_accepts += 1
            tokens.append(next_tok.item())
            tokens.append(draft_tok.item())
            old_committed = committed_len
            committed_len += 2

            if draft_tok.item() == eos or len(tokens) >= max_new_tokens:
                break

            # step 2a: also process draft_tok to get bonus
            num_main_fwd += 1
            kv_len2 = past_kv.get_seq_length()
            mask2 = torch.ones(1, kv_len2 + 1, device=device, dtype=torch.long)
            cap.reset()
            out2 = model(input_ids=draft_tok.to(device), attention_mask=mask2,
                         past_key_values=past_kv, use_cache=True)
            past_kv = out2.past_key_values
            hidden2 = cap.hidden_states[0]  # [1, 1, H]
            bonus = out2.logits[:, -1:, :].argmax(dim=-1)

            # Advance MTP KV for both committed positions
            mtp_head.forward_step(
                hidden1, draft_tok,
                position=old_committed, kv_cache=mtp_kv,
            )
            next_tok = bonus
            mtp_logits = mtp_head.forward_step(
                hidden2, bonus,
                position=old_committed + 1, kv_cache=mtp_kv,
            )
            draft_tok = mtp_logits[:, -1:, :].argmax(dim=-1)
        else:
            # ── reject ──────────────────────────────────────────
            tokens.append(next_tok.item())
            committed_len += 1

            if next_tok.item() == eos or len(tokens) >= max_new_tokens:
                break

            # next_tok is already processed; use verified_tok as new next
            next_tok = verified_tok
            mtp_logits = mtp_head.forward_step(
                hidden1, verified_tok,
                position=committed_len - 1, kv_cache=mtp_kv,
            )
            draft_tok = mtp_logits[:, -1:, :].argmax(dim=-1)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    cap.remove()

    # Trim to exact limit (accept commits 2 tokens and may overshoot by 1)
    tokens = tokens[:max_new_tokens]

    acc_rate = num_accepts / num_rounds if num_rounds else 0
    return {
        "tokens": tokens,
        "text": tokenizer.decode(tokens, skip_special_tokens=True),
        "num_tokens": len(tokens),
        "elapsed_s": round(elapsed, 3),
        "tok_per_s": round(len(tokens) / elapsed, 2) if elapsed > 0 else 0,
        "num_rounds": num_rounds,
        "num_accepts": num_accepts,
        "acceptance_rate": round(acc_rate, 4),
        "avg_tok_per_round": round(len(tokens) / num_rounds, 2) if num_rounds else 0,
        "main_fwd_calls": num_main_fwd,
    }


def benchmark_speculative_decode(
    model,
    mtp_head: Qwen3_5MTPHead,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 128,
    system_prompt: str = "",
    enable_thinking: bool | None = None,
) -> dict:
    """Run both baseline and speculative decode, report comparative results."""
    print("  [baseline] autoregressive …")
    base = baseline_decode(
        model,
        tokenizer,
        prompt,
        max_new_tokens,
        system_prompt,
        enable_thinking=enable_thinking,
    )
    print(f"    {base['num_tokens']} tokens in {base['elapsed_s']}s  ({base['tok_per_s']} tok/s)")

    print("  [spec-dec] MTP-1 speculative …")
    spec = speculative_decode(
        model,
        mtp_head,
        tokenizer,
        prompt,
        max_new_tokens,
        system_prompt,
        enable_thinking=enable_thinking,
    )
    print(f"    {spec['num_tokens']} tokens in {spec['elapsed_s']}s  ({spec['tok_per_s']} tok/s)")
    print(f"    accept rate: {spec['acceptance_rate']:.2%}  avg tok/round: {spec['avg_tok_per_round']}")

    speedup = spec["tok_per_s"] / base["tok_per_s"] if base["tok_per_s"] > 0 else 0
    text_match = base["text"] == spec["text"]

    return {
        "baseline": base,
        "speculative": spec,
        "speedup": round(speedup, 3),
        "text_match": text_match,
    }


# ════════════════════════════════════════════════════════════════
# 4.  Forced-decode DeltaNet monkey-patch
# ════════════════════════════════════════════════════════════════
#
# Problem: HF Qwen3_5GatedDeltaNet uses chunk_gated_delta_rule (parallel)
# when seq_len > 1, which gives DIFFERENT results from the recurrent
# (sequential) path used for single-token decode.  The chunk path also
# ignores the existing recurrent state (initial_state=None).
#
# Solution: Monkey-patch the DeltaNet forward so that when a special flag
# is active and seq_len > 1, it processes tokens one-at-a-time through
# the single-token decode path (conv1d_update + recurrent_gated_delta_rule).
# After processing token 0, we save the intermediate DeltaNet states
# (conv + recurrent) for each layer; on reject we roll back to these states.

_forced_decode_active = False
_intermediate_deltanet_states: dict[int, list[dict[str, torch.Tensor]]] = {}
_original_deltanet_forwards: dict[type, callable] = {}
_original_deltanet_instance_forwards: dict[int, tuple[object, str, callable]] = {}
_DELTANET_CLASS_NAMES = {"Qwen3_5GatedDeltaNet", "Qwen3_5MoeGatedDeltaNet"}


def _patched_deltanet_forward(self, hidden_states, cache_params=None, cache_position=None, attention_mask=None):
    """Drop-in replacement for Qwen3_5GatedDeltaNet.forward.

    When ``_forced_decode_active`` is True, ``seq_len > 1``, and the cache
    already has previous states, this processes each token sequentially
    through the recurrent path instead of the chunk path.  Intermediate
    DeltaNet states after each token (except the last) are saved in
    ``_intermediate_deltanet_states`` for rollback on reject.
    """
    global _forced_decode_active, _intermediate_deltanet_states

    batch_size, seq_len, _ = hidden_states.shape
    should_force = (
        _forced_decode_active
        and cache_params is not None
        and cache_params.has_previous_state
        and seq_len > 1
        and cache_position is not None
    )

    if not should_force:
        return _original_deltanet_forwards[type(self)](self, hidden_states, cache_params, cache_position, attention_mask)

    # Process tokens one-at-a-time through the decode path
    outputs = []
    for t in range(seq_len):
        tok_h = hidden_states[:, t : t + 1, :]
        tok_cp = cache_position[t : t + 1]
        # Call original forward with seq_len=1 → triggers the recurrent decode path
        tok_out = _original_deltanet_forwards[type(self)](self, tok_h, cache_params, tok_cp, None)
        outputs.append(tok_out)

        # Save state after each token except the last (for rollback)
        if t < seq_len - 1:
            if self.layer_idx not in _intermediate_deltanet_states:
                _intermediate_deltanet_states[self.layer_idx] = []
            _intermediate_deltanet_states[self.layer_idx].append({
                "conv": cache_params.conv_states[self.layer_idx].clone(),
                "recurrent": cache_params.recurrent_states[self.layer_idx].clone(),
            })

    return torch.cat(outputs, dim=1)


def _patch_deltanet_class(cls: type):
    if cls not in _original_deltanet_forwards:
        _original_deltanet_forwards[cls] = cls.forward
        cls.forward = _patched_deltanet_forward


def _patch_deltanet_instance(module):
    _patch_deltanet_class(type(module))
    attr = "_old_forward" if hasattr(module, "_old_forward") else None
    if attr is None:
        return
    key = id(module)
    if key not in _original_deltanet_instance_forwards:
        _original_deltanet_instance_forwards[key] = (module, attr, getattr(module, attr))
        setattr(module, attr, _patched_deltanet_forward.__get__(module, type(module)))


def install_forced_decode_patch(model=None):
    """Monkey-patch Qwen3_5GatedDeltaNet.forward with forced-decode version.

    Patches both the canonical Transformers classes and any trust_remote_code
    classes that were actually instantiated by ``AutoModel``.
    """
    try:
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet
        _patch_deltanet_class(Qwen3_5GatedDeltaNet)
    except ImportError:
        pass

    try:
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeGatedDeltaNet
        _patch_deltanet_class(Qwen3_5MoeGatedDeltaNet)
    except ImportError:
        pass

    if model is not None:
        for module in model.modules():
            cls = type(module)
            if cls.__name__ in _DELTANET_CLASS_NAMES:
                _patch_deltanet_instance(module)


def remove_forced_decode_patch():
    """Restore original Qwen3_5GatedDeltaNet.forward."""
    for _key, (module, attr, original_forward) in list(_original_deltanet_instance_forwards.items()):
        setattr(module, attr, original_forward)
    _original_deltanet_instance_forwards.clear()

    for cls, original_forward in list(_original_deltanet_forwards.items()):
        cls.forward = original_forward
    _original_deltanet_forwards.clear()


class ForcedDecodeContext:
    """Context manager that activates forced-decode mode for DeltaNet layers."""

    def __enter__(self):
        global _forced_decode_active, _intermediate_deltanet_states
        _forced_decode_active = True
        _intermediate_deltanet_states.clear()
        return self

    def __exit__(self, *args):
        global _forced_decode_active
        _forced_decode_active = False


def rollback_deltanet_states(past_kv, rollback_idx: int = 0):
    """Restore DeltaNet conv/recurrent states to the intermediate point at ``rollback_idx``.

    With K+1 token forward, states are saved after each token 0..K-1.
    ``rollback_idx=0`` restores state after the first token (next_tok);
    ``rollback_idx=j`` restores state after the (j+1)-th token.
    """
    for layer_idx, states_list in _intermediate_deltanet_states.items():
        if rollback_idx < len(states_list):
            past_kv.conv_states[layer_idx] = states_list[rollback_idx]["conv"]
            past_kv.recurrent_states[layer_idx] = states_list[rollback_idx]["recurrent"]


def trim_full_attention_kv(past_kv, n_trim: int = 1):
    """Remove the last n_trim entries from full-attention KV cache layers."""
    for layer_idx in past_kv.transformer_layers:
        if past_kv.key_cache[layer_idx] is not None:
            past_kv.key_cache[layer_idx] = past_kv.key_cache[layer_idx][:, :, :-n_trim, :]
            past_kv.value_cache[layer_idx] = past_kv.value_cache[layer_idx][:, :, :-n_trim, :]


def _is_moe_model(model) -> bool:
    config = getattr(model, "config", None)
    model_type = str(getattr(config, "model_type", "")).lower()
    return "moe" in model_type or "moe" in type(model).__name__.lower() or "moe" in type(_get_text_model(model)).__name__.lower()


# ════════════════════════════════════════════════════════════════
# 5.  Forced-decode speculative decode  (1 forward per round, K drafts)
# ════════════════════════════════════════════════════════════════


@torch.no_grad()
def generate_drafts(
    mtp_head: Qwen3_5MTPHead,
    main_hidden: torch.Tensor,
    next_tok: torch.Tensor,
    num_drafts: int,
    position: int,
    mtp_kv: dict,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Generate K draft tokens via auto-regressive MTP.

    Step 0 uses the real main-model hidden and appends to ``mtp_kv``.
    Steps 1..K-1 use speculative MTP hidden states and a *copy* of mtp_kv
    to avoid contaminating the real cache.

    Returns:
        drafts:  list of K draft token tensors [1,1]
        mtp_hiddens: list of K MTP hidden states [1,1,H] (for KV update on accept)
    """
    drafts: list[torch.Tensor] = []
    mtp_hiddens: list[torch.Tensor] = []

    # Step 0: real MTP step
    logits_0, mtp_h0 = mtp_head.forward_step(
        main_hidden, next_tok, position=position, kv_cache=mtp_kv, return_hidden=True,
    )
    draft_0 = logits_0[:, -1:, :].argmax(dim=-1)
    drafts.append(draft_0)
    mtp_hiddens.append(mtp_h0)

    if num_drafts <= 1:
        return drafts, mtp_hiddens

    # Steps 1..K-1: speculative auto-regressive MTP (separate KV)
    spec_kv = {"key": mtp_kv["key"].clone(), "value": mtp_kv["value"].clone()} if mtp_kv else {}
    h = mtp_h0
    for j in range(1, num_drafts):
        logits_j, h = mtp_head.forward_step(
            h, drafts[-1], position=position + j, kv_cache=spec_kv, hidden_source="mtp", return_hidden=True,
        )
        draft_j = logits_j[:, -1:, :].argmax(dim=-1)
        drafts.append(draft_j)
        mtp_hiddens.append(h)

    return drafts, mtp_hiddens


@torch.no_grad()
def speculative_decode_forced(
    model,
    mtp_head: Qwen3_5MTPHead,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 128,
    system_prompt: str = "",
    num_draft_tokens: int = 1,
    enable_thinking: bool | None = None,
) -> dict:
    """Forced-decode speculative decoding with K draft tokens.

    Algorithm per round:
      1. MTP head auto-regressively drafts K tokens: [d0, d1, ..., d_{K-1}].
      2. Enable forced-decode; process [next_tok, d0, ..., d_{K-1}] in 1 forward.
      3. Check logits sequentially: logits[j] top1 == d_j?
      4. First rejection at position j:
         - Commit j+1 tokens (next_tok, d0..d_{j-1}).
         - Rollback DeltaNet to state after token j, trim KV by K-j.
         - next_tok = argmax(logits[j]).
      5. All accepted:
         - Commit K+1 tokens (next_tok, d0..d_{K-1}).
         - bonus = argmax(logits[K]).

    Cost: 1 main-model forward per round (accept or reject).
    """
    K = num_draft_tokens
    device = model.device
    eos = tokenizer.eos_token_id

    msgs = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
    text = _apply_chat_template(tokenizer, msgs, enable_thinking=enable_thinking)
    inputs = tokenizer([text], return_tensors="pt").to(device)
    prompt_ids = inputs.input_ids
    prompt_len = prompt_ids.shape[1]
    exact_verify = _is_moe_model(model)

    if not exact_verify:
        install_forced_decode_patch(model)

    # ── prefill (main model) ────────────────────────────────────
    cap = AfterNormCapture(model)
    cap.reset()
    outputs = model(**inputs, use_cache=True)
    past_kv = outputs.past_key_values
    next_tok = outputs.logits[:, -1:, :].argmax(dim=-1)
    prefill_final_hidden = cap.hidden_states[0]
    committed_len = prompt_len

    # ── prefill (MTP) ──────────────────────────────────────────
    mtp_kv: dict = {}
    if prompt_len >= 2:
        mtp_head.forward_batch(
            prefill_final_hidden[:, :-1, :],
            prompt_ids[:, 1:],
            positions=torch.arange(prompt_len - 1, device=mtp_head._mtp_device),
            kv_cache=mtp_kv,
        )

    # First draft batch
    drafts, _ = generate_drafts(
        mtp_head, prefill_final_hidden[:, -1:, :], next_tok, K,
        position=committed_len - 1, mtp_kv=mtp_kv,
    )

    tokens: list[int] = []
    num_accepts = 0
    num_rounds = 0
    num_main_fwd = 0
    total_accepted_drafts = 0
    mtp_time_acc = 0.0
    main_model_time_acc = 0.0

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    while len(tokens) < max_new_tokens:
        num_rounds += 1
        num_main_fwd += 1

        torch.cuda.synchronize()
        _main_t0 = time.perf_counter()
        replacement_tok = None
        bonus_tok = None
        if exact_verify:
            # MoE under device_map uses Accelerate hooks and batched matmuls that
            # are not bitwise-equivalent to single-token greedy decode.  Verify
            # drafts with the exact recurrent path to preserve output identity.
            accepted_count = 0
            exact_fwd_calls = 0
            hiddens = []
            for j in range(K):
                tok_j = next_tok if j == 0 else drafts[j - 1].to(device)
                kv_len = past_kv.get_seq_length()
                mask = torch.ones(1, kv_len + 1, device=device, dtype=torch.long)
                cap.reset()
                out_j = model(
                    input_ids=tok_j,
                    attention_mask=mask,
                    past_key_values=past_kv,
                    use_cache=True,
                )
                exact_fwd_calls += 1
                past_kv = out_j.past_key_values
                hiddens.append(cap.hidden_states[0])
                verified_j = out_j.logits[:, -1:, :].argmax(dim=-1)
                if verified_j.item() == drafts[j].item():
                    accepted_count += 1
                else:
                    replacement_tok = verified_j
                    break

            if accepted_count == K:
                kv_len = past_kv.get_seq_length()
                mask = torch.ones(1, kv_len + 1, device=device, dtype=torch.long)
                cap.reset()
                out_bonus = model(
                    input_ids=drafts[-1].to(device),
                    attention_mask=mask,
                    past_key_values=past_kv,
                    use_cache=True,
                )
                exact_fwd_calls += 1
                past_kv = out_bonus.past_key_values
                hiddens.append(cap.hidden_states[0])
                bonus_tok = out_bonus.logits[:, -1:, :].argmax(dim=-1)

            num_main_fwd += exact_fwd_calls - 1
            hidden_all = torch.cat(hiddens, dim=1)
        else:
            # ── batch verify: [next_tok, d0, ..., d_{K-1}] in 1 forward ──
            all_toks = torch.cat([next_tok] + [d.to(device) for d in drafts], dim=1)  # [1, K+1]
            kv_len = past_kv.get_seq_length()
            mask = torch.ones(1, kv_len + K + 1, device=device, dtype=torch.long)

            with ForcedDecodeContext():
                cap.reset()
                out = model(
                    input_ids=all_toks,
                    attention_mask=mask,
                    past_key_values=past_kv,
                    use_cache=True,
                )
            past_kv = out.past_key_values
            hidden_all = cap.hidden_states[0]  # [1, K+1, H]

            # Sequential acceptance check
            accepted_count = 0
            for j in range(K):
                verified_j = out.logits[:, j, :].argmax(dim=-1, keepdim=True)  # [1, 1]
                if verified_j.item() == drafts[j].item():
                    accepted_count += 1
                else:
                    break
        torch.cuda.synchronize()
        main_model_time_acc += time.perf_counter() - _main_t0
            
        print(f"Round {num_rounds}: accepted {accepted_count}/{K} drafts")

        if accepted_count == K:
            # ── ALL ACCEPTED ────────────────────────────────────
            num_accepts += 1
            total_accepted_drafts += K
            tokens.append(next_tok.item())
            for d in drafts:
                tokens.append(d.item())
            old_committed = committed_len
            committed_len += K + 1

            if any(d.item() == eos for d in drafts) or len(tokens) >= max_new_tokens:
                break

            # Update MTP KV with real main-model hidden states for accepted positions.
            # At position old_committed+j, MTP needs (hidden[j], token_at[old_committed+j+1]).
            # token_at[old_committed+j+1] = drafts[j] for j=0..K-1.
            torch.cuda.synchronize()
            _mtp_t0 = time.perf_counter()
            for j in range(K):
                mtp_head.forward_step(
                    hidden_all[:, j:j+1, :], drafts[j],
                    position=old_committed + j, kv_cache=mtp_kv,
                )

            if bonus_tok is None:
                bonus_tok = out.logits[:, K, :].argmax(dim=-1, keepdim=True)
            next_tok = bonus_tok

            # generate_drafts adds MTP KV at position old_committed+K with bonus_tok
            drafts, _ = generate_drafts(
                mtp_head, hidden_all[:, K:K+1, :], bonus_tok, K,
                position=committed_len - 1, mtp_kv=mtp_kv,
            )
            torch.cuda.synchronize()
            mtp_time_acc += time.perf_counter() - _mtp_t0
        else:
            # ── PARTIAL/FULL REJECT ─────────────────────────────
            # Commit next_tok + accepted drafts (accepted_count of them)
            tokens.append(next_tok.item())
            for j in range(accepted_count):
                total_accepted_drafts += 1
                tokens.append(drafts[j].item())
            committed_len += accepted_count + 1

            if next_tok.item() == eos or len(tokens) >= max_new_tokens:
                break
            if any(drafts[j].item() == eos for j in range(accepted_count)):
                break

            if not exact_verify:
                # Rollback: restore DeltaNet to state after token[accepted_count]
                rollback_deltanet_states(past_kv, rollback_idx=accepted_count)
                trim_full_attention_kv(past_kv, n_trim=K - accepted_count)

            # Update MTP KV for accepted positions
            torch.cuda.synchronize()
            _mtp_t0 = time.perf_counter()
            for j in range(accepted_count):
                mtp_head.forward_step(
                    hidden_all[:, j:j+1, :], drafts[j],
                    position=committed_len - accepted_count - 1 + j,
                    kv_cache=mtp_kv,
                )

            # Replacement token from rejected position
            replacement = replacement_tok if replacement_tok is not None else out.logits[:, accepted_count, :].argmax(dim=-1, keepdim=True)
            next_tok = replacement

            # Generate next K drafts
            drafts, _ = generate_drafts(
                mtp_head, hidden_all[:, accepted_count:accepted_count+1, :], replacement, K,
                position=committed_len - 1, mtp_kv=mtp_kv,
            )
            torch.cuda.synchronize()
            mtp_time_acc += time.perf_counter() - _mtp_t0

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    cap.remove()
    if not exact_verify:
        remove_forced_decode_patch()

    tokens = tokens[:max_new_tokens]

    draft_accept_rate = total_accepted_drafts / (num_rounds * K) if num_rounds else 0
    total_decode_time = main_model_time_acc + mtp_time_acc
    mtp_ratio = mtp_time_acc / total_decode_time if total_decode_time > 0 else 0
    return {
        "tokens": tokens,
        "text": tokenizer.decode(tokens, skip_special_tokens=True),
        "num_tokens": len(tokens),
        "elapsed_s": round(elapsed, 3),
        "tok_per_s": round(len(tokens) / elapsed, 2) if elapsed > 0 else 0,
        "num_rounds": num_rounds,
        "num_accepts": num_accepts,
        "total_accepted_drafts": total_accepted_drafts,
        "acceptance_rate": round(draft_accept_rate, 4),
        "avg_tok_per_round": round(len(tokens) / num_rounds, 2) if num_rounds else 0,
        "main_fwd_calls": num_main_fwd,
        "tok_per_fwd": round(len(tokens) / num_main_fwd, 2) if num_main_fwd else 0,
        "num_draft_tokens": K,
        "main_model_time_s": round(main_model_time_acc, 4),
        "mtp_time_s": round(mtp_time_acc, 4),
        "mtp_ratio": round(mtp_ratio, 4),
    }


def benchmark_speculative_decode_forced(
    model,
    mtp_head: Qwen3_5MTPHead,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 128,
    system_prompt: str = "",
    num_draft_tokens: int = 1,
    enable_thinking: bool | None = None,
) -> dict:
    """Run baseline and forced-decode spec-dec comparison."""
    print("  [baseline] autoregressive …")
    base = baseline_decode(
        model,
        tokenizer,
        prompt,
        max_new_tokens,
        system_prompt,
        enable_thinking=enable_thinking,
    )
    print(f"    {base['num_tokens']} tokens in {base['elapsed_s']}s  ({base['tok_per_s']} tok/s)")

    print(f"  [forced-decode K={num_draft_tokens}] speculative …")
    spec = speculative_decode_forced(
        model, mtp_head, tokenizer, prompt, max_new_tokens, system_prompt,
        num_draft_tokens=num_draft_tokens,
        enable_thinking=enable_thinking,
    )
    print(f"    {spec['num_tokens']} tokens in {spec['elapsed_s']}s  ({spec['tok_per_s']} tok/s)")
    print(f"    accept rate: {spec['acceptance_rate']:.2%}  avg tok/round: {spec['avg_tok_per_round']}")
    print(f"    tok/fwd: {spec['tok_per_fwd']}  main fwds: {spec['main_fwd_calls']}")

    speedup = spec["tok_per_s"] / base["tok_per_s"] if base["tok_per_s"] > 0 else 0
    text_match = base["text"] == spec["text"]

    return {
        "baseline": base,
        "speculative": spec,
        "speedup": round(speedup, 3),
        "text_match": text_match,
    }


def benchmark_multi_k(
    model,
    mtp_head: Qwen3_5MTPHead,
    tokenizer,
    prompts: list[dict],
    max_new_tokens: int = 128,
    system_prompt: str = "",
    k_values: list[int] | None = None,
    enable_thinking: bool | None = None,
) -> dict:
    """Run forced-decode benchmark for multiple K values across all prompts.

    Returns:
        {
            "baseline_results": {prompt_name: baseline_result},
            "k_results": {K: {prompt_name: spec_result}},
            "comparison": {K: {prompt_name: {speedup, text_match, ...}}},
        }
    """
    if k_values is None:
        k_values = [2, 3, 4]

    # Run baseline once per prompt
    baseline_results: dict[str, dict] = {}
    for p in prompts:
        print(f"\n[baseline] {p['name']} …")
        base = baseline_decode(
            model,
            tokenizer,
            p["prompt"],
            max_new_tokens,
            system_prompt,
            enable_thinking=enable_thinking,
        )
        print(f"  {base['num_tokens']} tokens in {base['elapsed_s']}s  ({base['tok_per_s']} tok/s)")
        baseline_results[p["name"]] = base

    # Run each K value
    k_results: dict[int, dict[str, dict]] = {}
    comparison: dict[int, dict[str, dict]] = {}

    for K in k_values:
        print(f"\n{'═' * 60}")
        print(f"  K={K} (num_draft_tokens={K})")
        print(f"{'═' * 60}")
        k_results[K] = {}
        comparison[K] = {}

        for p in prompts:
            print(f"\n  [{K}-draft] {p['name']} …")
            spec = speculative_decode_forced(
                model, mtp_head, tokenizer, p["prompt"],
                max_new_tokens, system_prompt, num_draft_tokens=K,
                enable_thinking=enable_thinking,
            )
            k_results[K][p["name"]] = spec
            base = baseline_results[p["name"]]
            speedup = spec["tok_per_s"] / base["tok_per_s"] if base["tok_per_s"] > 0 else 0
            text_match = base["text"] == spec["text"]
            comparison[K][p["name"]] = {
                "speedup": round(speedup, 3),
                "text_match": text_match,
                "acceptance_rate": spec["acceptance_rate"],
                "tok_per_fwd": spec["tok_per_fwd"],
                "avg_tok_per_round": spec["avg_tok_per_round"],
            }
            print(f"    accept: {spec['acceptance_rate']:.2%}  tok/fwd: {spec['tok_per_fwd']}"
                  f"  speedup: {speedup:.3f}×  match: {text_match}")

    return {
        "baseline_results": baseline_results,
        "k_results": k_results,
        "comparison": comparison,
    }


def print_multi_k_summary(results: dict, model_name: str = ""):
    """Print a formatted summary table for multi-K benchmark results."""
    comparison = results["comparison"]
    baseline_results = results["baseline_results"]
    prompt_names = list(baseline_results.keys())

    print(f"\n{'═' * 80}")
    print(f"  Multi-K Summary{f' — {model_name}' if model_name else ''}")
    print(f"{'═' * 80}")

    # Per-K summary table
    for K in sorted(comparison.keys()):
        print(f"\n  K={K} (draft tokens per round):")
        print(f"  {'Prompt':<25s} {'Accept%':>8s} {'Tok/fwd':>8s} {'Speedup':>8s} {'Match':>6s}")
        print(f"  {'─' * 55}")
        accs, spds, matches = [], [], []
        for name in prompt_names:
            c = comparison[K].get(name, {})
            acc = c.get("acceptance_rate", 0)
            tpf = c.get("tok_per_fwd", 0)
            spd = c.get("speedup", 0)
            mtch = c.get("text_match", False)
            print(f"  {name:<25s} {acc:>7.1%} {tpf:>8.2f} {spd:>7.3f}× {'✅' if mtch else '❌':>5s}")
            accs.append(acc)
            spds.append(spd)
            matches.append(mtch)
        if accs:
            print(f"  {'─' * 55}")
            avg_acc = sum(accs) / len(accs)
            avg_spd = sum(spds) / len(spds)
            match_ct = sum(matches)
            print(f"  {'AVERAGE':<25s} {avg_acc:>7.1%} {'':>8s} {avg_spd:>7.3f}× {match_ct}/{len(matches)}")

    # Cross-K comparison
    print(f"\n  Cross-K Comparison:")
    print(f"  {'Prompt':<25s}", end="")
    for K in sorted(comparison.keys()):
        print(f"  K={K:>2d}", end="")
    print()
    print(f"  {'─' * (25 + 7 * len(comparison))}")
    for name in prompt_names:
        print(f"  {name:<25s}", end="")
        for K in sorted(comparison.keys()):
            spd = comparison[K].get(name, {}).get("speedup", 0)
            print(f" {spd:.3f}×", end="")
        print()


# ════════════════════════════════════════════════════════════════
# Reporting helpers
# ════════════════════════════════════════════════════════════════


def _separator(title: str = "", width: int = 72):
    if title:
        print(f"\n{'─' * 4} {title} {'─' * (width - 6 - len(title))}")
    else:
        print("─" * width)


def print_acceptance_result(name: str, r: dict):
    _separator(f"Acceptance: {name}")
    if "error" in r:
        print(f"  ERROR: {r['error']}")
        return
    print(f"  Prompt {r['prompt_len']} tok → generated {r['gen_len']} tok  (total {r['total_len']})")
    print(f"  Overall  acceptance: {r['overall_acceptance_rate']:.2%}")
    print(f"  Prompt   acceptance: {r['prompt_acceptance_rate']:.2%}")
    print(f"  Generate acceptance: {r['generation_acceptance_rate']:.2%}")
    print(f"  Top-5    acceptance: {r['top5_acceptance_rate']:.2%}")
    print(f"  Top-5 gen accept  : {r['top5_generation_acceptance_rate']:.2%}")
    if r.get("samples"):
        print("  Sample predictions (generated region):")
        for s in r["samples"]:
            print(s)
    print(f"  Output: {r['output_text'][:120]}…")


def print_timing_result(r: dict):
    _separator("Timing")
    print(f"  Main model : {r['main_model_ms']:.2f} ms/step")
    print(f"  MTP head   : {r['mtp_head_ms']:.2f} ms/step")
    print(f"  Overhead   : {r['overhead_ratio']:.4f}×  ({r['num_steps']} steps)")

    if r["overhead_ratio"] < 1:
        # theoretical speedup with different acceptance rates
        print("  Theoretical speedup = (1+α) / (1 + overhead):")
        for alpha in [0.3, 0.4, 0.5, 0.6, 0.7]:
            su = (1 + alpha) / (1 + r["overhead_ratio"])
            print(f"    α={alpha:.0%} → {su:.3f}×")


def print_spec_result(name: str, r: dict, mode: str = "sequential"):
    _separator(f"Speculative decode ({mode}): {name}")
    b, s = r["baseline"], r["speculative"]
    print(f"  Baseline  : {b['num_tokens']} tok, {b['elapsed_s']}s, {b['tok_per_s']} tok/s")
    print(f"  Spec-dec  : {s['num_tokens']} tok, {s['elapsed_s']}s, {s['tok_per_s']} tok/s")
    print(f"  Accept    : {s['acceptance_rate']:.2%}  ({s['num_accepts']}/{s['num_rounds']} rounds)")
    print(f"  Tok/round : {s['avg_tok_per_round']}")
    print(f"  Main fwds : {s['main_fwd_calls']}  (baseline ≈ {b['num_tokens']})")
    if "tok_per_fwd" in s:
        print(f"  Tok/fwd   : {s['tok_per_fwd']}")
    print(f"  Speedup   : {r['speedup']:.3f}×")
    print(f"  Text match: {r['text_match']}")
    if mode == "sequential" and r["speedup"] < 1.1:
        print("  ⚠ DeltaNet limitation: batch-verify gives wrong logits, so tokens")
        print("    are verified sequentially.  Accept costs 2 fwd, reject costs 1.")
        print("    Effective speedup ≈ 1.0×.  Use --forced-decode for 1-fwd verification.")


def print_summary(all_results: list[dict]):
    _separator("Summary")
    print(f"  {'Name':<25s} {'Gen%':>7s} {'Top5%':>7s} {'Prompt':>6s} {'Gen':>5s} {'Total':>5s}")
    for r in all_results:
        name = r["name"]
        d = r.get("acceptance", {})
        if "error" in d:
            print(f"  {name:<25s}  (error)")
            continue
        gen_a = d.get("generation_acceptance_rate", 0)
        t5a = d.get("top5_generation_acceptance_rate", 0)
        print(
            f"  {name:<25s} {gen_a:>6.1%} {t5a:>6.1%} "
            f"{d.get('prompt_len', 0):>6d} {d.get('gen_len', 0):>5d} {d.get('total_len', 0):>5d}"
        )
    gen_rates = [
        r["acceptance"]["generation_acceptance_rate"]
        for r in all_results
        if "generation_acceptance_rate" in r.get("acceptance", {})
    ]
    if gen_rates:
        avg = sum(gen_rates) / len(gen_rates)
        print(f"\n  Average generation acceptance rate: {avg:.2%}  (across {len(gen_rates)} prompts)")


# ════════════════════════════════════════════════════════════════
# Test prompts
# ════════════════════════════════════════════════════════════════

TEST_PROMPTS = [
    {
        "name": "chinese_science",
        "prompt": "请详细解释量子纠缠的原理，以及它在量子计算中的应用。",
    },
    {
        "name": "chinese_essay",
        "prompt": "请写一篇关于人工智能对教育行业影响的议论文，500字左右。",
    },
    {
        "name": "english_explanation",
        "prompt": "Explain the difference between TCP and UDP protocols. When would you choose one over the other?",
    },
    {
        "name": "python_tutorial",
        "prompt": "Write a Python tutorial on implementing a binary search tree with insert, search, and delete operations. Include complete code with type hints.",
    },
    {
        "name": "code_rbtree",
        "prompt": "Implement a Red-Black Tree in C++ with insert and left-right rotation. Include the color-flip logic.",
    },
    {
        "name": "code_go_http",
        "prompt": "Write a simple HTTP server in Go that handles GET and POST requests with JSON parsing. Include error handling and middleware.",
    },
    {
        "name": "math_proof",
        "prompt": "Prove that the square root of 2 is irrational using proof by contradiction. Be rigorous.",
    },
    {
        "name": "math_word_problem",
        "prompt": "A train leaves station A at 60 km/h. Another train leaves station B (300 km away) 30 minutes later at 90 km/h heading toward A. When and where do they meet? Show all steps.",
    },
    {
        "name": "translation",
        "prompt": (
            "Translate the following to Chinese:\n"
            '"The development of large language models has fundamentally transformed how we '
            "interact with artificial intelligence. These models understand context, generate "
            'creative content, and assist with complex reasoning."'
        ),
    },
    {
        "name": "creative_story",
        "prompt": "Write a short sci-fi story (≈300 words) about a programmer who discovers the AI they created has developed genuine emotions.",
    },
]


def _apply_chat_template(tokenizer, messages: list[dict[str, str]], enable_thinking: bool | None = None) -> str:
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": False,
    }
    if enable_thinking is not None:
        kwargs["enable_thinking"] = True
    return tokenizer.apply_chat_template(messages, **kwargs)


# ════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════


def parse_args():
    p = argparse.ArgumentParser(
        description="Qwen3.5 MTP benchmark — acceptance rate, timing, speculative decode",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", type=str, default="weights/Qwen3.5-27B")
    p.add_argument("--prompt", type=str, default=None, help="Single prompt (overrides --run-all)")
    p.add_argument("--run-all", action="store_true", help="Run all TEST_PROMPTS")
    p.add_argument("--timing", action="store_true", help="Run latency benchmark")
    p.add_argument("--spec-decode", action="store_true", help="Run speculative-decode comparison (sequential)")
    p.add_argument("--forced-decode", action="store_true", help="Run forced-decode speculative-decode (1 fwd/round)")
    p.add_argument("--num-draft-tokens", type=int, default=1, help="Number of MTP draft tokens per round (K)")
    p.add_argument("--multi-k", action="store_true", help="Run multi-K experiment (K=2,3,4)")
    p.add_argument("--max-new-tokens", type=int, default=4096)
    p.add_argument("--timing-steps", type=int, default=50)
    p.add_argument("--dtype", type=str, default="bf16", choices=sorted(DTYPE_MAP.keys()))
    p.add_argument("--system-prompt", type=str, default="")
    p.add_argument(
        "--save-json", type=str, default=None,
        help="If set, save acceptance-rate results as JSON to this path.",
    )
    thinking_group = p.add_mutually_exclusive_group()
    thinking_group.add_argument("--enable-thinking", action="store_true", help="Enable thinking mode in chat template.")
    thinking_group.add_argument("--disable-thinking", action="store_true", help="Disable thinking mode in chat template.")
    return p.parse_args()


def main():
    args = parse_args()
    dtype = args.dtype
    enable_thinking = True if args.enable_thinking else False if args.disable_thinking else None

    print("═" * 72)
    print("  Qwen3.5 MTP Benchmark")
    print("═" * 72)

    # ── load model ──────────────────────────────────────────────
    print(f"\nLoading model from {args.model} …")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=DTYPE_MAP[dtype] if dtype != "auto" else "auto",
        device_map="auto",
    )
    model.eval()
    print(f"  Model type: {type(model).__name__}")
    print(f"  Device: {model.device}")

    # ── build MTP head ──────────────────────────────────────────
    print("\nBuilding MTP head …")
    mtp_head = build_mtp_head(model, args.model, dtype)

    # ── determine prompts ───────────────────────────────────────
    if args.prompt:
        prompts = [{"name": "custom", "prompt": args.prompt}]
    elif args.run_all:
        prompts = TEST_PROMPTS
    else:
        prompts = [TEST_PROMPTS[0]]  # default: first prompt

    # ── timing benchmark ────────────────────────────────────────
    if args.timing:
        print("\nRunning timing benchmark …")
        timing = benchmark_timing(
            model, mtp_head, tokenizer,
            prompts[0]["prompt"],
            system_prompt=args.system_prompt,
            num_steps=args.timing_steps,
            enable_thinking=enable_thinking,
        )
        print_timing_result(timing)

    # ── acceptance rate ─────────────────────────────────────────
    all_results: list[dict] = []
    for p in prompts:
        print(f"\nMeasuring acceptance rate: {p['name']} …")
        r = measure_acceptance_rate(
            model, mtp_head, tokenizer,
            p["prompt"],
            max_new_tokens=args.max_new_tokens,
            system_prompt=args.system_prompt,
            enable_thinking=enable_thinking,
        )
        print_acceptance_result(p["name"], r)
        all_results.append({"name": p["name"], "acceptance": r})

    if len(all_results) > 1:
        print_summary(all_results)

    # ── save JSON results ───────────────────────────────────────
    if args.save_json:
        out_path = Path(args.save_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        json_data = {
            "model": args.model,
            "dtype": dtype,
            "max_new_tokens": args.max_new_tokens,
            "results": all_results,
        }
        out_path.write_text(json.dumps(json_data, ensure_ascii=False, indent=2))
        print(f"\n  Saved acceptance results → {out_path}")

    # ── speculative decode (sequential) ───────────────────────────
    if args.spec_decode:
        for p in prompts:
            print(f"\nSpeculative decode benchmark (sequential): {p['name']} …")
            sr = benchmark_speculative_decode(
                model, mtp_head, tokenizer,
                p["prompt"],
                max_new_tokens=args.max_new_tokens,
                system_prompt=args.system_prompt,
                enable_thinking=enable_thinking,
            )
            print_spec_result(p["name"], sr, mode="sequential")

    # ── speculative decode (forced-decode) ──────────────────────
    if args.forced_decode:
        K = args.num_draft_tokens
        for p in prompts:
            print(f"\nSpeculative decode benchmark (forced-decode K={K}): {p['name']} …")
            sr = benchmark_speculative_decode_forced(
                model, mtp_head, tokenizer,
                p["prompt"],
                max_new_tokens=args.max_new_tokens,
                system_prompt=args.system_prompt,
                num_draft_tokens=K,
                enable_thinking=enable_thinking,
            )
            print_spec_result(p["name"], sr, mode="forced-decode")

    # ── multi-K experiment ──────────────────────────────────────
    if args.multi_k:
        print(f"\n{'═' * 72}")
        print(f"  Multi-K Experiment")
        print(f"{'═' * 72}")
        mk_results = benchmark_multi_k(
            model, mtp_head, tokenizer, prompts,
            max_new_tokens=args.max_new_tokens,
            system_prompt=args.system_prompt,
            k_values=[2, 3, 4],
            enable_thinking=enable_thinking,
        )
        model_name = args.model.split("/")[-1]
        print_multi_k_summary(mk_results, model_name)

    print("\n" + "═" * 72)
    print("  Done.")
    print("═" * 72)


if __name__ == "__main__":
    main()
