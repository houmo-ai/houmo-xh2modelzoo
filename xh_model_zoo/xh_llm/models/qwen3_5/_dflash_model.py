"""DFlash draft models for xh2a export."""

import math
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from xhquant import nn as xhnn
from xhquant.core import CacheTensor
from xhquant.nn import LLMCacheV2, RMSNorm, MaskedAdd


def _build_rope_cache(
    *,
    head_dim: int,
    rope_theta: float,
    max_pe_length: int,
) -> tuple[Tensor, Tensor]:
    inv_freq = 1.0 / (
        rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    )
    positions = torch.arange(max_pe_length, dtype=torch.float32).reshape(-1, 1)
    freqs = positions * inv_freq.reshape(1, -1)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos().unsqueeze(0).unsqueeze(0), emb.sin().unsqueeze(0).unsqueeze(0)


def _ensure_cache_tensor(cache: Tensor) -> CacheTensor:
    if isinstance(cache, CacheTensor):
        return cache
    if isinstance(cache, torch.Tensor):
        return CacheTensor(cache)
    return cache


class DFlashCrossAttention(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        rms_norm_eps: float,
        input_sequence_length: int,
        max_pe_length: int,
        rope_theta: float,
        use_cache: bool,
    ):
        super().__init__()
        self.num_heads = num_attention_heads
        self.num_kv_heads = num_key_value_heads
        self.head_dim = head_dim
        self.num_kv_groups = num_attention_heads // num_key_value_heads
        self.use_cache = use_cache

        self.q_proj = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_attention_heads * head_dim, hidden_size, bias=False)
        self.q_norm = RMSNorm(head_dim, rms_norm_eps)
        self.k_norm = RMSNorm(head_dim, rms_norm_eps)
        self.q_rope = xhnn.Rope()
        self.k_rope = xhnn.Rope()
        self.cos_slice = xhnn.DynamicSlice([input_sequence_length], [2], [1])
        self.sin_slice = xhnn.DynamicSlice([input_sequence_length], [2], [1])
        self.k_cache = LLMCacheV2(axis=2) if use_cache else None
        self.v_cache = LLMCacheV2(axis=2) if use_cache else None
        self.masked_add = MaskedAdd() 

        cos_cached, sin_cached = _build_rope_cache(
            head_dim=head_dim,
            rope_theta=rope_theta,
            max_pe_length=max_pe_length,
        )
        self.register_buffer("cos_cached", cos_cached, persistent=False)
        self.register_buffer("sin_cached", sin_cached, persistent=False)
        self.register_buffer(
            "scale",
            torch.tensor(head_dim**-0.5, dtype=torch.float32),
            persistent=False,
        )

    def build_target_kv(
        self,
        target_hidden: Tensor,
        past_seq_length: Tensor,
        current_input_length: Tensor,
        past_key_cache: Tensor | None = None,
        past_value_cache: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        bsz, seq_len, _ = target_hidden.shape
        key_states = self.k_norm(
            self.k_proj(target_hidden).view(bsz, seq_len, self.num_kv_heads, self.head_dim)
        ).transpose(1, 2)
        value_states = self.v_proj(target_hidden).view(
            bsz, seq_len, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)
        cos = self.cos_slice(self.cos_cached, past_seq_length)
        sin = self.sin_slice(self.sin_cached, past_seq_length)
        key_states = self.k_rope(key_states, cos, sin)
        if not self.use_cache:
            return key_states, value_states
        return (
            self.k_cache(key_states, past_seq_length, current_input_length, past_key_cache),
            self.v_cache(value_states, past_seq_length, current_input_length, past_value_cache),
        )

    def forward_decode(
        self,
        hidden_states: Tensor,
        past_seq_length: Tensor,
        current_input_length: Tensor,
        target_key_cache: Tensor,
        target_value_cache: Tensor,
        attn_mask: Tensor,
    ) -> Tensor:
        bsz, q_len, _ = hidden_states.shape
        query_states = self.q_norm(
            self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim)
        ).transpose(1, 2)
        noise_key_states = self.k_norm(
            self.k_proj(hidden_states).view(
                bsz, q_len, self.num_kv_heads, self.head_dim
            )
        ).transpose(1, 2)
        noise_value_states = self.v_proj(hidden_states).view(
            bsz, q_len, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)
        q_cos = self.cos_slice(self.cos_cached, past_seq_length)
        q_sin = self.sin_slice(self.sin_cached, past_seq_length)
        query_states = self.q_rope(query_states, q_cos, q_sin)
        noise_key_states = self.k_rope(noise_key_states, q_cos, q_sin)
        query_states = query_states * self.scale.to(query_states.dtype)
        target_key_cache = _ensure_cache_tensor(target_key_cache)
        target_value_cache = _ensure_cache_tensor(target_value_cache)

        combined_key_states = self.k_cache(
            noise_key_states,
            past_seq_length,
            current_input_length,
            target_key_cache,
        )
        combined_value_states = self.v_cache(
            noise_value_states,
            past_seq_length,
            current_input_length,
            target_value_cache,
        )
        key_states = torch.repeat_interleave(
            combined_key_states.transpose(2, 3), self.num_kv_groups, dim=1
        )
        value_states = torch.repeat_interleave(
            combined_value_states, self.num_kv_groups, dim=1
        )
        attn_weights = torch.matmul(query_states, key_states)
        attn_weights = self.masked_add(attn_weights, attn_mask.unsqueeze(1).unsqueeze(1))
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            query_states.dtype
        )
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, -1)
        return self.o_proj(attn_output)


class DFlashMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DFlashDecoderLayer(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        intermediate_size: int,
        rms_norm_eps: float,
        input_sequence_length: int,
        max_pe_length: int,
        rope_theta: float,
        use_cache: bool,
    ):
        super().__init__()
        self.self_attn = DFlashCrossAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            rms_norm_eps=rms_norm_eps,
            input_sequence_length=input_sequence_length,
            max_pe_length=max_pe_length,
            rope_theta=rope_theta,
            use_cache=use_cache,
        )
        self.input_layernorm = RMSNorm(hidden_size, rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, rms_norm_eps)
        self.mlp = DFlashMLP(hidden_size, intermediate_size)

    def forward_decode(
        self,
        hidden_states: Tensor,
        past_seq_length: Tensor,
        current_input_length: Tensor,
        target_key_cache: Tensor,
        target_value_cache: Tensor,
        attn_mask: Tensor,
    ) -> Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn.forward_decode(
            hidden_states,
            past_seq_length=past_seq_length,
            current_input_length=current_input_length,
            target_key_cache=target_key_cache,
            target_value_cache=target_value_cache,
            attn_mask=attn_mask,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.mlp(self.post_attention_layernorm(hidden_states))
        return residual + hidden_states


class DFlashModelXH2a(nn.Module):
    def __init__(
        self,
        *,
        mode: str,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        intermediate_size: int,
        num_hidden_layers: int,
        rms_norm_eps: float,
        num_target_layers: int,
        target_layer_ids: List[int],
        vocab_size: int,
        input_sequence_length: int,
        max_pe_length: int,
        max_sequence_length: int,
        rope_theta: float = 10_000_000.0,
    ):
        super().__init__()
        if mode not in {"context", "decode"}:
            raise ValueError(f"Unsupported DFlash mode: {mode}")
        self.mode = mode
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_target_layers = num_target_layers
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.vocab_size = vocab_size
        self.input_sequence_length = input_sequence_length
        self.max_sequence_length = max_sequence_length
        self.target_layer_ids = target_layer_ids

        self.fc = nn.Linear(len(target_layer_ids) * hidden_size, hidden_size, bias=False)
        self.hidden_norm = RMSNorm(hidden_size, rms_norm_eps)
        self.layers = nn.ModuleList(
            [
                DFlashDecoderLayer(
                    hidden_size=hidden_size,
                    num_attention_heads=num_attention_heads,
                    num_key_value_heads=num_key_value_heads,
                    head_dim=head_dim,
                    intermediate_size=intermediate_size,
                    rms_norm_eps=rms_norm_eps,
                    input_sequence_length=input_sequence_length,
                    max_pe_length=max_pe_length,
                    rope_theta=rope_theta,
                    use_cache=True,
                )
                for _ in range(num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(hidden_size, rms_norm_eps)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def _split_cache_tensors(
        self, cache_tensors: tuple[Tensor, ...]
    ) -> tuple[list[Tensor], list[Tensor]]:
        if len(cache_tensors) != self.num_hidden_layers * 2:
            raise ValueError(
                f"Expected {self.num_hidden_layers * 2} cache tensors, got {len(cache_tensors)}"
            )
        key_caches = list(cache_tensors[: self.num_hidden_layers])
        value_caches = list(cache_tensors[self.num_hidden_layers :])
        return key_caches, value_caches

    def forward_context(
        self,
        target_hidden: Tensor,
        past_seq_length: Tensor,
        current_input_length: Tensor,
        *cache_tensors: Tensor,
    ) -> tuple[Tensor, ...]:
        target_proj = self.hidden_norm(self.fc(target_hidden))
        past_key_caches, past_value_caches = self._split_cache_tensors(cache_tensors)
        present_key_caches: list[Tensor] = []
        present_value_caches: list[Tensor] = []
        for idx, layer in enumerate(self.layers):
            present_k, present_v = layer.self_attn.build_target_kv(
                target_proj,
                past_seq_length=past_seq_length,
                current_input_length=current_input_length,
                past_key_cache=past_key_caches[idx],
                past_value_cache=past_value_caches[idx],
            )
            present_key_caches.append(present_k)
            present_value_caches.append(present_v)
        return tuple(present_key_caches + present_value_caches)

    def forward_decode(
        self,
        noise_embedding: Tensor,
        past_seq_length: Tensor,
        current_input_length: Tensor,
        attn_mask: Tensor,
        *cache_tensors: Tensor,
    ) -> Tensor:
        # attn_mask = attn_mask + current_input_length.to(attn_mask.dtype).unsqueeze(-1) * 0
        past_key_caches, past_value_caches = self._split_cache_tensors(cache_tensors)
        hidden_states = noise_embedding
        for idx, layer in enumerate(self.layers):
            hidden_states = layer.forward_decode(
                hidden_states,
                past_seq_length=past_seq_length,
                current_input_length=current_input_length,
                target_key_cache=past_key_caches[idx],
                target_value_cache=past_value_caches[idx],
                attn_mask=attn_mask,
            )
        return self.lm_head(self.norm(hidden_states))

    def forward(
        self,
        input0: Tensor,
        input1: Tensor,
        input2: Tensor,
        input3: Tensor | None = None,
        input4: Tensor | None = None,
        input5: Tensor | None = None,
        input6: Tensor | None = None,
        input7: Tensor | None = None,
        input8: Tensor | None = None,
        input9: Tensor | None = None,
        input10: Tensor | None = None,
        input11: Tensor | None = None,
        input12: Tensor | None = None,
        input13: Tensor | None = None,
        input14: Tensor | None = None,
        input15: Tensor | None = None,
        input16: Tensor | None = None,
        input17: Tensor | None = None,
        input18: Tensor | None = None,
        input19: Tensor | None = None,
    ):
        if self.mode == "context":
            cache_inputs = (
                input3,
                input4,
                input5,
                input6,
                input7,
                input8,
                input9,
                input10,
                input11,
                input12,
                input13,
                input14,
                input15,
                input16,
                input17,
                input18,
                input19,
            )[: self.num_hidden_layers * 2]
            return self.forward_context(input0, input1, input2, *cache_inputs)
        if input3 is None:
            raise ValueError("DFlash decode requires attn_mask input.")
        cache_inputs = (
            input4,
            input5,
            input6,
            input7,
            input8,
            input9,
            input10,
            input11,
            input12,
            input13,
            input14,
            input15,
            input16,
            input17,
            input18,
            input19,
        )[: self.num_hidden_layers * 2]
        return self.forward_decode(input0, input1, input2, input3, *cache_inputs)

    @staticmethod
    def from_pretrained(
        dflash_model_dir: str,
        target_model_dir: str,
        *,
        mode: str,
        dtype: torch.dtype = torch.float16,
        input_sequence_length: int,
        max_pe_length: int,
        max_sequence_length: int,
    ) -> "DFlashModelXH2a":
        import json
        from safetensors import safe_open

        with open(Path(dflash_model_dir) / "config.json", encoding="utf-8") as f:
            cfg = json.load(f)

        target_layer_ids = cfg["dflash_config"]["target_layer_ids"]
        hidden_size = cfg["hidden_size"]
        head_dim = cfg.get("head_dim", hidden_size // cfg["num_attention_heads"])
        model = DFlashModelXH2a(
            mode=mode,
            hidden_size=hidden_size,
            num_attention_heads=cfg["num_attention_heads"],
            num_key_value_heads=cfg["num_key_value_heads"],
            head_dim=head_dim,
            intermediate_size=cfg["intermediate_size"],
            num_hidden_layers=cfg["num_hidden_layers"],
            rms_norm_eps=cfg.get("rms_norm_eps", 1e-6),
            num_target_layers=len(target_layer_ids),
            target_layer_ids=target_layer_ids,
            vocab_size=cfg["vocab_size"],
            input_sequence_length=input_sequence_length,
            max_pe_length=max_pe_length,
            max_sequence_length=max_sequence_length,
            rope_theta=cfg.get("rope_theta", 10_000_000.0),
        )

        with safe_open(str(Path(dflash_model_dir) / "model.safetensors"), framework="pt") as f:
            dflash_sd = {k: f.get_tensor(k).to(dtype) for k in f.keys()}
        missing, unexpected = model.load_state_dict(dflash_sd, strict=False)
        unexpected = [name for name in unexpected if not name.endswith(("cos_cached", "sin_cached"))]
        missing = [name for name in missing if not name.endswith(("cos_cached", "sin_cached")) and name != "lm_head.weight"]
        if unexpected:
            raise RuntimeError(f"Unexpected DFlash keys: {unexpected}")
        if missing:
            raise RuntimeError(f"Missing DFlash keys: {missing}")

        lm_head_loaded = False
        for sf_path in sorted(Path(target_model_dir).glob("*.safetensors")):
            with safe_open(str(sf_path), framework="pt") as f:
                if "lm_head.weight" in f.keys():
                    model.lm_head.weight.data.copy_(f.get_tensor("lm_head.weight").to(dtype))
                    lm_head_loaded = True
                    break
        if not lm_head_loaded:
            for sf_path in sorted(Path(target_model_dir).glob("*.safetensors")):
                with safe_open(str(sf_path), framework="pt") as f:
                    for key in f.keys():
                        if "embed_tokens.weight" in key:
                            model.lm_head.weight.data.copy_(f.get_tensor(key).to(dtype))
                            lm_head_loaded = True
                            break
                if lm_head_loaded:
                    break
        if not lm_head_loaded:
            raise RuntimeError("Could not load lm_head.weight from target model")

        model.to(dtype)
        return model
