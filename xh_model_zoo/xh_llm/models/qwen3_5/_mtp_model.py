"""Cache-aware MTP draft model for xh2a export."""

import math
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from xhquant import nn as xhnn
from xhquant.nn import LLMCacheV2, MaskedSoftmax, RMSNorm
from xhquant.nn.modules.moeblock import MoeBlock


def _build_rotary_cache(
    *,
    rotary_dim: int,
    rope_theta: float,
    max_pe_length: int,
) -> tuple[Tensor, Tensor, Tensor]:
    inv_freq = 1.0 / (
        rope_theta
        ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim)
    )
    positions = torch.arange(max_pe_length, dtype=torch.float32).reshape(-1, 1)
    freqs = positions * inv_freq.reshape(1, -1)
    emb = torch.cat([freqs, freqs], dim=-1)
    cos = emb.cos().unsqueeze(0).unsqueeze(0)
    sin = emb.sin().unsqueeze(0).unsqueeze(0)
    return inv_freq, cos, sin


class MTPGatedAttention(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        rotary_dim: int,
        rms_norm_eps: float,
        input_sequence_length: int,
        rope_theta: float,
        max_pe_length: int,
        use_cache: bool,
    ):
        super().__init__()
        self.num_heads = num_attention_heads
        self.num_kv_heads = num_key_value_heads
        self.head_dim = head_dim
        self.rotary_dim = rotary_dim
        self.num_kv_groups = num_attention_heads // num_key_value_heads
        self.use_cache = use_cache

        self.q_proj = nn.Linear(
            hidden_size, num_attention_heads * head_dim * 2, bias=False
        )
        self.k_proj = nn.Linear(
            hidden_size, num_key_value_heads * head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            hidden_size, num_key_value_heads * head_dim, bias=False
        )
        self.o_proj = nn.Linear(num_attention_heads * head_dim, hidden_size, bias=False)
        self.q_norm = RMSNorm(head_dim, rms_norm_eps)
        self.k_norm = RMSNorm(head_dim, rms_norm_eps)
        self.masked_softmax = MaskedSoftmax(dim=-1)
        self.rope = xhnn.Rope()

        self.q_rot_slice = xhnn.Slice([0], [rotary_dim], [3], [1])
        self.q_pass_slice = xhnn.Slice([rotary_dim], [head_dim], [3], [1])
        self.k_rot_slice = xhnn.Slice([0], [rotary_dim], [3], [1])
        self.k_pass_slice = xhnn.Slice([rotary_dim], [head_dim], [3], [1])
        self.cos_slice = xhnn.DynamicSlice([input_sequence_length], [2], [1])
        self.sin_slice = xhnn.DynamicSlice([input_sequence_length], [2], [1])

        _, cos_cached, sin_cached = _build_rotary_cache(
            rotary_dim=rotary_dim,
            rope_theta=rope_theta,
            max_pe_length=max_pe_length,
        )
        self.register_buffer("cos_cached", cos_cached, persistent=False)
        self.register_buffer("sin_cached", sin_cached, persistent=False)
        self.register_buffer(
            "kv_scale",
            torch.tensor(head_dim**-0.5, dtype=torch.float32),
            persistent=False,
        )

        self.k_cache = LLMCacheV2(axis=2) if use_cache else None
        self.v_cache = LLMCacheV2(axis=2) if use_cache else None

    def _apply_rotary_pos_emb(
        self, q: Tensor, k: Tensor, cos: Tensor, sin: Tensor
    ) -> tuple[Tensor, Tensor]:
        q_rot = self.q_rot_slice(q)
        q_pass = self.q_pass_slice(q)
        k_rot = self.k_rot_slice(k)
        k_pass = self.k_pass_slice(k)
        q_rot = self.rope(q_rot, cos, sin)
        k_rot = self.rope(k_rot, cos, sin)
        return (
            torch.cat([q_rot, q_pass], dim=-1),
            torch.cat([k_rot, k_pass], dim=-1),
        )

    def forward(
        self,
        hidden_states: Tensor,
        past_seq_length: Tensor,
        current_input_length: Tensor,
        past_k_cache: Optional[Tensor] = None,
        past_v_cache: Optional[Tensor] = None,
    ) -> tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        bsz, q_len, _ = hidden_states.shape
        qg = self.q_proj(hidden_states).view(
            bsz, q_len, self.num_heads, self.head_dim * 2
        )
        query_states, gate = torch.split(qg, self.head_dim, dim=-1)
        gate = gate.reshape(bsz, q_len, -1)

        query_states = self.q_norm(query_states).transpose(1, 2)
        key_states = self.k_norm(
            self.k_proj(hidden_states).view(
                bsz, q_len, self.num_kv_heads, self.head_dim
            )
        ).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(
            bsz, q_len, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)

        cos = self.cos_slice(self.cos_cached, past_seq_length)
        sin = self.sin_slice(self.sin_cached, past_seq_length)
        query_states, key_states = self._apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        present_k_cache = key_states
        present_v_cache = value_states
        if self.use_cache:
            present_k_cache = self.k_cache(
                key_states, past_seq_length, current_input_length, past_k_cache
            )
            present_v_cache = self.v_cache(
                value_states, past_seq_length, current_input_length, past_v_cache
            )

        query_states = query_states * self.kv_scale.to(query_states.dtype)
        key_states_t = torch.repeat_interleave(
            present_k_cache.transpose(2, 3), self.num_kv_groups, dim=1
        )
        attn_weights = torch.matmul(query_states, key_states_t)
        attn_weights = self.masked_softmax(attn_weights, past_seq_length)
        value_states = torch.repeat_interleave(
            present_v_cache, self.num_kv_groups, dim=1
        )
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, -1)
        attn_output = attn_output * torch.sigmoid(gate)
        return self.o_proj(attn_output)


class MTPMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MTPSparseMoEBlock(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        num_experts: int,
        top_k: int,
        expert_intermediate_size: int,
        shared_expert_intermediate_size: int,
    ):
        super().__init__()
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        self.moeblock = MoeBlock("silu", top_k, True)
        self.moeblock.expert_gate_proj_weight = nn.Parameter(
            torch.zeros(num_experts, expert_intermediate_size, hidden_size)
        )
        self.moeblock.expert_gate_proj_bias = None
        self.moeblock.expert_up_proj_weight = nn.Parameter(
            torch.zeros(num_experts, expert_intermediate_size, hidden_size)
        )
        self.moeblock.expert_up_proj_bias = None
        self.moeblock.expert_down_proj_weight = nn.Parameter(
            torch.zeros(num_experts, hidden_size, expert_intermediate_size)
        )
        self.moeblock.expert_down_proj_bias = None
        self.shared_expert = MTPMLP(hidden_size, shared_expert_intermediate_size)
        self.shared_expert_gate = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden_states: Tensor) -> Tensor:
        routing_weights = F.softmax(self.gate(hidden_states), dim=-1)
        moe_out = self.moeblock(hidden_states, routing_weights)
        shared_out = self.shared_expert(hidden_states)
        shared_out = torch.sigmoid(self.shared_expert_gate(hidden_states)) * shared_out
        return moe_out + shared_out


class MTPDecoderLayerXH2a(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        rotary_dim: int,
        intermediate_size: int,
        rms_norm_eps: float,
        input_sequence_length: int,
        rope_theta: float,
        max_pe_length: int,
        use_cache: bool,
    ):
        super().__init__()
        self.self_attn = MTPGatedAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            rotary_dim=rotary_dim,
            rms_norm_eps=rms_norm_eps,
            input_sequence_length=input_sequence_length,
            rope_theta=rope_theta,
            max_pe_length=max_pe_length,
            use_cache=use_cache,
        )
        self.mlp = MTPMLP(hidden_size, intermediate_size)
        self.input_layernorm = RMSNorm(hidden_size, rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, rms_norm_eps)

    def forward(
        self,
        hidden_states: Tensor,
        past_seq_length: Tensor,
        current_input_length: Tensor,
        past_k_cache: Optional[Tensor] = None,
        past_v_cache: Optional[Tensor] = None,
    ) -> tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        residual = hidden_states
        attn_out = self.self_attn(
            self.input_layernorm(hidden_states),
            past_seq_length=past_seq_length,
            current_input_length=current_input_length,
            past_k_cache=past_k_cache,
            past_v_cache=past_v_cache,
        )
        hidden_states = residual + attn_out
        residual = hidden_states
        hidden_states = self.mlp(self.post_attention_layernorm(hidden_states))
        hidden_states = residual + hidden_states
        return hidden_states


class MTPModelXH2a(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        intermediate_size: int,
        rms_norm_eps: float,
        vocab_size: int,
        input_sequence_length: int,
        rope_theta: float = 10_000_000.0,
        partial_rotary_factor: float = 0.25,
        max_pe_length: int = 262144,
        use_cache: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.input_sequence_length = input_sequence_length
        self.use_cache = use_cache
        rotary_dim = int(head_dim * partial_rotary_factor)

        self.pre_fc_norm_embedding = RMSNorm(hidden_size, rms_norm_eps)
        self.pre_fc_norm_hidden = RMSNorm(hidden_size, rms_norm_eps)
        self.fc = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        self.layer = MTPDecoderLayerXH2a(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            rotary_dim=rotary_dim,
            intermediate_size=intermediate_size,
            rms_norm_eps=rms_norm_eps,
            input_sequence_length=input_sequence_length,
            rope_theta=rope_theta,
            max_pe_length=max_pe_length,
            use_cache=use_cache,
        )
        self.norm = RMSNorm(hidden_size, rms_norm_eps)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(
        self,
        next_token_embedding: Tensor,
        post_norm_hidden: Tensor,
        past_seq_length: Tensor,
        current_input_length: Tensor,
        past_key_cache: Optional[Tensor] = None,
        past_value_cache: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        embeds = self.pre_fc_norm_embedding(next_token_embedding)
        hidden = self.pre_fc_norm_hidden(post_norm_hidden)
        hidden_states = self.fc(torch.cat([embeds, hidden], dim=-1))
        hidden_states = self.layer(
            hidden_states,
            past_seq_length=past_seq_length,
            current_input_length=current_input_length,
            past_k_cache=past_key_cache,
            past_v_cache=past_value_cache,
        )
        hidden_states = self.norm(hidden_states)
        post_norm_out = hidden_states
        logits = self.lm_head(hidden_states)
        return logits, post_norm_out

    @staticmethod
    def from_pretrained(
        target_model_dir: str,
        *,
        dtype: torch.dtype = torch.float16,
        input_sequence_length: int = 1,
        max_pe_length: int = 262144,
        use_cache: bool = True,
    ) -> "MTPModelXH2a":
        import json
        from safetensors import safe_open

        with open(Path(target_model_dir) / "config.json", encoding="utf-8") as f:
            cfg = json.load(f)

        text_cfg = cfg.get("text_config", cfg)
        rope_params = text_cfg.get("rope_parameters", {})
        is_moe = "num_experts" in text_cfg and "moe_intermediate_size" in text_cfg

        def _cfg_value(name: str):
            if name in text_cfg:
                return text_cfg[name]
            if name in cfg:
                return cfg[name]
            raise KeyError(name)

        intermediate_size = text_cfg.get("intermediate_size", cfg.get("intermediate_size"))
        if intermediate_size is None and is_moe:
            intermediate_size = text_cfg.get(
                "shared_expert_intermediate_size",
                text_cfg.get("moe_intermediate_size"),
            )
        if intermediate_size is None:
            raise KeyError("intermediate_size")

        model = MTPModelXH2a(
            hidden_size=_cfg_value("hidden_size"),
            num_attention_heads=_cfg_value("num_attention_heads"),
            num_key_value_heads=_cfg_value("num_key_value_heads"),
            head_dim=_cfg_value("head_dim"),
            intermediate_size=intermediate_size,
            rms_norm_eps=text_cfg.get("rms_norm_eps", cfg.get("rms_norm_eps", 1e-6)),
            vocab_size=_cfg_value("vocab_size"),
            input_sequence_length=input_sequence_length,
            rope_theta=rope_params.get("rope_theta", 10_000_000.0),
            partial_rotary_factor=rope_params.get("partial_rotary_factor", 0.25),
            max_pe_length=max_pe_length,
            use_cache=use_cache,
        )
        if is_moe:
            num_experts = _cfg_value("num_experts")
            top_k = _cfg_value("num_experts_per_tok")
            expert_intermediate_size = _cfg_value("moe_intermediate_size")
            shared_expert_intermediate_size = text_cfg.get(
                "shared_expert_intermediate_size",
                cfg.get("shared_expert_intermediate_size", expert_intermediate_size),
            )
            model.layer.mlp = MTPSparseMoEBlock(
                hidden_size=_cfg_value("hidden_size"),
                num_experts=num_experts,
                top_k=top_k,
                expert_intermediate_size=expert_intermediate_size,
                shared_expert_intermediate_size=shared_expert_intermediate_size,
            ).to(dtype=dtype)
        rms_norm_weight_keys = {
            f"{name}.weight"
            for name, module in model.named_modules()
            if isinstance(module, RMSNorm) and name
        }

        mtp_sd = {}
        lm_head_weight = None
        embed_weight = None
        packed_gate_proj_weight = None
        packed_up_proj_weight = None
        packed_down_proj_weight = None
        if is_moe:
            packed_gate_proj_weight = torch.zeros(
                _cfg_value("num_experts"),
                _cfg_value("moe_intermediate_size"),
                _cfg_value("hidden_size"),
                dtype=dtype,
            )
            packed_up_proj_weight = torch.zeros_like(packed_gate_proj_weight)
            packed_down_proj_weight = torch.zeros(
                _cfg_value("num_experts"),
                _cfg_value("hidden_size"),
                _cfg_value("moe_intermediate_size"),
                dtype=dtype,
            )
        for sf_path in sorted(Path(target_model_dir).glob("*.safetensors")):
            with safe_open(str(sf_path), framework="pt") as f:
                for key in f.keys():
                    if key.startswith("mtp."):
                        model_key = key[4:]
                        if model_key.startswith("layers.0."):
                            model_key = "layer." + model_key[len("layers.0.") :]
                        tensor = f.get_tensor(key).to(dtype)
                        if is_moe and model_key == "layer.mlp.experts.gate_up_proj":
                            expert_intermediate_size = _cfg_value("moe_intermediate_size")
                            gate_proj, up_proj = tensor.split(expert_intermediate_size, dim=1)
                            packed_gate_proj_weight.copy_(gate_proj)
                            packed_up_proj_weight.copy_(up_proj)
                            continue
                        if is_moe and model_key == "layer.mlp.experts.down_proj":
                            packed_down_proj_weight.copy_(tensor)
                            continue
                        if is_moe and model_key.startswith("layer.mlp.experts."):
                            parts = model_key.split(".")
                            expert_idx = int(parts[3])
                            proj_name = parts[4]
                            if proj_name == "gate_proj":
                                packed_gate_proj_weight[expert_idx].copy_(tensor)
                            elif proj_name == "up_proj":
                                packed_up_proj_weight[expert_idx].copy_(tensor)
                            elif proj_name == "down_proj":
                                packed_down_proj_weight[expert_idx].copy_(tensor)
                            else:
                                raise RuntimeError(f"Unsupported MoE MTP expert key: {model_key}")
                            continue
                        if model_key in rms_norm_weight_keys:
                            tensor = tensor + 1.0
                        mtp_sd[model_key] = tensor
                    elif key == "lm_head.weight":
                        lm_head_weight = f.get_tensor(key).to(dtype)
                    elif "embed_tokens.weight" in key:
                        embed_weight = f.get_tensor(key).to(dtype)

        if is_moe:
            mtp_sd["layer.mlp.moeblock.expert_gate_proj_weight"] = packed_gate_proj_weight
            mtp_sd["layer.mlp.moeblock.expert_up_proj_weight"] = packed_up_proj_weight
            mtp_sd["layer.mlp.moeblock.expert_down_proj_weight"] = packed_down_proj_weight

        missing, unexpected = model.load_state_dict(mtp_sd, strict=False)
        unexpected = [name for name in unexpected if not name.endswith("cos_cached") and not name.endswith("sin_cached")]
        missing = [
            name
            for name in missing
            if not name.endswith("cos_cached")
            and not name.endswith("sin_cached")
            and name != "lm_head.weight"
        ]
        if unexpected:
            raise RuntimeError(f"Unexpected MTP keys: {unexpected}")
        if missing:
            raise RuntimeError(f"Missing MTP keys: {missing}")

        if lm_head_weight is not None:
            model.lm_head.weight.data.copy_(lm_head_weight)
        elif embed_weight is not None:
            model.lm_head.weight.data.copy_(embed_weight)
        else:
            raise RuntimeError("Could not load lm_head.weight from target model")

        model.to(dtype)
        return model
