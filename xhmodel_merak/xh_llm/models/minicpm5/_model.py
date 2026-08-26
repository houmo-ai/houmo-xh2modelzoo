"""DynamicModule adapters for the remote-code MiniCPM5-MoE checkpoint."""

from __future__ import annotations

import math
import sys
import types
from copy import deepcopy
from typing import Any, Optional

import torch
import torch.nn as nn
from torch import Tensor

from xhquant import nn as xhnn
from xhquant.api import ConfigDict
from xhquant.nn import LLMCache, MaskedSoftmax, RMSNorm
from xhquant.nn.modules.moeblock import MoeBlock
from xhquant.utils.registry import DynamicModule

from ...register import XHLLM_TRACEABLE_MODULES


def _cfg_get(cfg: Any, name: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(name, default)
    value = getattr(cfg, name, default)
    return default if value is None else value


def _register_or_replace_traceable(mapping: dict[type, str]):
    def decorator(dynamic_cls):
        for hf_cls, key in mapping.items():
            # Remote-code classes are loaded from the checkpoint directory and
            # therefore cannot be imported at module import time.
            XHLLM_TRACEABLE_MODULES._registry[hf_cls] = dynamic_cls
            XHLLM_TRACEABLE_MODULES._key_registry[hf_cls] = key
            XHLLM_TRACEABLE_MODULES._dynamic_classes.pop(hf_cls, None)
        return dynamic_cls

    return decorator


def _find_remote_classes(hf_model: nn.Module) -> dict[str, type[nn.Module]]:
    classes: dict[str, type[nn.Module]] = {}
    expected_prefix = "MiniCPM5MoE"
    for module in hf_model.modules():
        name = type(module).__name__
        if name.startswith(expected_prefix):
            classes.setdefault(name, type(module))
    return classes


def _pack_expert_projection(
    experts: nn.ModuleList,
    projection_name: str,
) -> tuple[Tensor, Tensor | None]:
    """Pack one routed-expert projection and preserve optional GPTQ codes."""
    linears = [getattr(expert, projection_name) for expert in experts]
    packed_weight = torch.stack([linear.weight.data for linear in linears], dim=0)
    quant_weights = [getattr(linear, "quant_weight", None) for linear in linears]
    has_quant_weights = [torch.is_tensor(quant_weight) for quant_weight in quant_weights]
    if any(has_quant_weights) and not all(has_quant_weights):
        missing = [str(index) for index, present in enumerate(has_quant_weights) if not present]
        raise RuntimeError(
            f"MiniCPM5 MoE {projection_name} quant_weight must exist for every expert; "
            f"missing experts: {', '.join(missing)}"
        )
    if not all(has_quant_weights):
        return packed_weight, None

    for expert_index, (linear, quant_weight) in enumerate(zip(linears, quant_weights, strict=True)):
        if tuple(quant_weight.shape) != tuple(linear.weight.shape):
            raise RuntimeError(
                f"MiniCPM5 MoE expert {expert_index} {projection_name}.quant_weight shape "
                f"{tuple(quant_weight.shape)} does not match weight shape {tuple(linear.weight.shape)}"
            )
    return packed_weight, torch.stack([quant_weight.data for quant_weight in quant_weights], dim=0)


class _MiniCPM5RMSNorm(DynamicModule):
    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.norm(hidden_states)

    def _setup(self, cfg: Optional[dict] = None):
        self.norm = RMSNorm(self.weight.shape[0], eps=self.variance_epsilon)
        self.norm.weight = nn.Parameter(deepcopy(self.weight.data))
        return self


class _MiniCPM5RotaryEmbedding(DynamicModule):
    def _setup(self, cfg: Optional[dict] = None):
        max_seq_len = int(
            _cfg_get(
                cfg,
                "max_pe_length",
                _cfg_get(cfg, "context_max_length", getattr(self, "max_seq_len_cached", 2048)),
            )
        )
        max_seq_len = max(1, max_seq_len)
        self.max_seq_len_cached = max_seq_len
        inv_freq = self.inv_freq.float()
        positions = torch.arange(max_seq_len, device=inv_freq.device, dtype=torch.float32).view(max_seq_len, 1)
        freqs = positions * inv_freq.view(1, -1)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = (emb.cos() * self.attention_scaling).to(dtype=self.inv_freq.dtype)
        sin = (emb.sin() * self.attention_scaling).to(dtype=self.inv_freq.dtype)
        self.register_buffer("cos_cached", cos.unsqueeze(0).unsqueeze(0).contiguous(), persistent=False)
        self.register_buffer("sin_cached", sin.unsqueeze(0).unsqueeze(0).contiguous(), persistent=False)
        if hasattr(self, "setup_after_callback"):
            self.setup_after_callback()
        return self

    def _set_dtype(self, dtype: torch.dtype) -> None:
        self.cos_cached = self.cos_cached.to(dtype=dtype)
        self.sin_cached = self.sin_cached.to(dtype=dtype)

    @torch.no_grad()
    def forward(self, x: Tensor, position_ids: Tensor):
        del x
        return self.cos_cached[:, :, position_ids], self.sin_cached[:, :, position_ids]


class _MiniCPM5Attention(DynamicModule):
    def _setup(self, cfg: ConfigDict | dict[str, Any] | None = None):
        if isinstance(cfg, dict):
            cfg = ConfigDict(cfg)
        self.num_heads = int(getattr(self, "num_attention_heads", self.config.num_attention_heads))
        self.num_key_value_heads = int(getattr(self, "num_key_value_heads", self.config.num_key_value_heads))
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.attn_hidden_dim = self.num_heads * self.head_dim
        self.use_gated_attention = bool(getattr(self, "use_gated_attention", self.config.use_gated_attention))
        self.enable_rope = bool(_cfg_get(cfg, "enable_rope", True))
        if self.enable_rope:
            self.rope = xhnn.Rope()
        self.slice_1 = xhnn.Slice([0], [self.head_dim // 2], [3], [1])
        self.slice_2 = xhnn.Slice([self.head_dim // 2], [sys.maxsize], [3], [1])
        self.query_slice = xhnn.Slice([0], [self.head_dim], [3], [1])
        self.gate_slice = xhnn.Slice([self.head_dim], [self.head_dim * 2], [3], [1])
        self.masked_softmax = MaskedSoftmax(dim=-1)
        self.use_cache = bool(_cfg_get(cfg, "use_cache", True))
        if self.use_cache:
            cache_axis = int(_cfg_get(_cfg_get(cfg, "kv_cache", {}), "cache_axis", 2))
            self.k_cache = LLMCache(axis=cache_axis, attention_max_length=-1)
            self.v_cache = LLMCache(axis=cache_axis, attention_max_length=-1)
        self.kv_scale = 1.0 / math.sqrt(self.head_dim)
        return self

    def _rotate_half(self, x: Tensor) -> Tensor:
        return torch.cat((-self.slice_2(x), self.slice_1(x)), dim=-1)

    def _apply_rope(self, q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor]:
        if self.enable_rope:
            return self.rope(q, cos, sin), self.rope(k, cos, sin)
        return q * cos + self._rotate_half(q) * sin, k * cos + self._rotate_half(k) * sin

    def forward(
        self,
        hidden_states: Tensor,
        past_seq_length: Tensor | None = None,
        current_input_length: Tensor | None = None,
        past_k_cache: Tensor | None = None,
        past_v_cache: Tensor | None = None,
        position_embeddings: tuple[Tensor, Tensor] | None = None,
        **kwargs,
    ) -> tuple[Tensor, None, None]:
        del kwargs
        batch_size, seq_length, _ = hidden_states.shape
        input_shape = (batch_size, seq_length)
        if self.use_gated_attention:
            q_and_gate = self.q_proj(hidden_states).view(batch_size, seq_length, self.num_heads, self.head_dim * 2)
            query_states = self.query_slice(q_and_gate)
            gate_score = self.gate_slice(q_and_gate)
            gate_score = gate_score.reshape(*input_shape, -1)
        else:
            query_states = self.q_proj(hidden_states).view(batch_size, seq_length, self.num_heads, self.head_dim)
        key_states = self.k_proj(hidden_states).view(batch_size, seq_length, self.num_key_value_heads, self.head_dim)
        value_states = self.v_proj(hidden_states).view(batch_size, seq_length, self.num_key_value_heads, self.head_dim)
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        cos, sin = position_embeddings
        query_states, key_states = self._apply_rope(query_states, key_states, cos, sin)
        if self.use_cache:
            key_states = self.k_cache(key_states, past_seq_length, current_input_length, past_k_cache)
            value_states = self.v_cache(value_states, past_seq_length, current_input_length, past_v_cache)
        query_states = query_states * self.kv_scale
        key_states = torch.repeat_interleave(key_states.transpose(2, 3), self.num_key_value_groups, dim=1)
        attn_weights = self.masked_softmax(torch.matmul(query_states, key_states), past_seq_length)
        value_states = torch.repeat_interleave(value_states, self.num_key_value_groups, dim=1)
        attn_output = (
            torch.matmul(attn_weights, value_states)
            .transpose(1, 2)
            .reshape(batch_size, seq_length, self.attn_hidden_dim)
        )
        if self.use_gated_attention:
            attn_output = attn_output * torch.sigmoid(gate_score)
        return self.o_proj(attn_output), None, None


class _MiniCPM5MoE(DynamicModule):
    def _setup(self, cfg: Optional[dict] = None):
        experts = self.experts
        self.num_experts = len(experts)
        self.top_k = int(getattr(self.gate, "top_k", self.config.num_experts_per_tok))
        self.n_group = int(getattr(self.gate, "n_group", getattr(self.config, "n_group", 1)))
        self.topk_group = int(getattr(self.gate, "topk_group", getattr(self.config, "topk_group", 1)))
        self.norm_topk_prob = bool(getattr(self.gate, "norm_topk_prob", getattr(self.config, "norm_topk_prob", True)))
        self.routed_scaling_factor = float(
            getattr(self.gate, "routed_scaling_factor", getattr(self.config, "routed_scaling_factor", 1.0))
        )
        self.router = nn.Linear(
            self.gate.weight.shape[1],
            self.gate.weight.shape[0],
            bias=False,
            device=self.gate.weight.device,
            dtype=torch.float32,
        )
        self.router.weight = nn.Parameter(self.gate.weight.detach().float())
        self.register_buffer(
            "e_score_correction_bias",
            self.gate.e_score_correction_bias.detach().float(),
            persistent=False,
        )
        self.router_input_cast = xhnn.Cast(torch.float32).to(dtype=torch.float32)
        self.router_topk = xhnn.TopK(self.top_k, axis=-1)
        self.routing_gather = xhnn.GatherElements(axis=1)
        self.routing_sum = xhnn.ReduceSum(dim=-1, keepdim=True)
        self.routing_add = xhnn.Add()
        self.routing_div = xhnn.Div()
        self.routing_mul = xhnn.Mul()
        self.register_buffer(
            "routing_epsilon",
            torch.tensor(1e-20, dtype=torch.float32, device=self.gate.weight.device),
            persistent=False,
        )
        self.register_buffer(
            "routing_scale",
            torch.tensor(self.routed_scaling_factor, dtype=torch.float32, device=self.gate.weight.device),
            persistent=False,
        )
        routing_dtype = experts[0].gate_proj.weight.dtype
        self.routing_cast = xhnn.Cast(routing_dtype).to(dtype=routing_dtype)
        self.moeblock = MoeBlock("silu", self.top_k, normalize_routing_weights=False, topk_outside=True)
        for projection_name in ("gate_proj", "up_proj", "down_proj"):
            packed_weight, packed_quant_weight = _pack_expert_projection(experts, projection_name)
            setattr(self.moeblock, f"expert_{projection_name}_weight", nn.Parameter(packed_weight))
            setattr(self.moeblock, f"expert_{projection_name}_bias", None)
            if packed_quant_weight is not None:
                self.moeblock.register_buffer(f"expert_{projection_name}_quant_weight", packed_quant_weight)
        del self.experts
        del self.gate
        return self

    def forward(self, hidden_states: Tensor) -> Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        flat_hidden_states = hidden_states.reshape(-1, hidden_dim)
        router_logits = self.router(self.router_input_cast(flat_hidden_states))
        scores = torch.sigmoid(router_logits)
        scores_for_choice = scores + self.e_score_correction_bias.unsqueeze(0)
        if self.n_group > 1:
            grouped = scores_for_choice.view(-1, self.n_group, self.num_experts // self.n_group)
            group_scores = grouped.topk(2, dim=-1)[0].sum(dim=-1)
            group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
            group_mask = torch.zeros_like(group_scores).scatter_(1, group_idx, 1)
            scores_for_choice = scores_for_choice.masked_fill(
                ~group_mask.unsqueeze(-1).expand_as(grouped).reshape_as(scores_for_choice).bool(), 0.0
            )
        selected_experts = self.router_topk(scores_for_choice)[1]
        routing_weights = self.routing_gather(scores, selected_experts)
        if self.norm_topk_prob:
            routing_sum = self.routing_add(self.routing_sum(routing_weights), self.routing_epsilon)
            routing_weights = self.routing_div(routing_weights, routing_sum)
        if self.routed_scaling_factor != 1.0:
            routing_weights = self.routing_mul(routing_weights, self.routing_scale)
        routing_weights = self.routing_cast(routing_weights)
        output = self.moeblock(
            hidden_states,
            routing_weights.view(batch_size, sequence_length, self.top_k),
            selected_experts=selected_experts.view(batch_size, sequence_length, self.top_k),
        )
        return output + self.shared_experts(hidden_states)


class _MiniCPM5DecoderLayer(DynamicModule):
    def _setup(self, cfg: Optional[dict] = None):
        return self

    def forward(
        self,
        hidden_states: Tensor,
        past_seq_length: Tensor | None = None,
        current_input_length: Tensor | None = None,
        past_k_cache: Tensor | None = None,
        past_v_cache: Tensor | None = None,
        position_embeddings: tuple[Tensor, Tensor] | None = None,
        **kwargs,
    ) -> tuple[Tensor]:
        del kwargs
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _, _ = self.self_attn(
            hidden_states,
            past_seq_length=past_seq_length,
            current_input_length=current_input_length,
            past_k_cache=past_k_cache,
            past_v_cache=past_v_cache,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return (residual + hidden_states,)


class _MiniCPM5Model(DynamicModule):
    def _setup(self, cfg: Optional[dict] = None):
        self.only_first_block = bool(_cfg_get(cfg, "only_first_block", False))
        max_layers = _cfg_get(cfg, "max_layers", -1)
        self.max_layers = 1 if self.only_first_block else (-1 if max_layers is None else int(max_layers))
        self.num_logits_to_keep = int(_cfg_get(cfg, "num_logits_to_keep", 0) or 0)
        self.input_sequence_length = int(_cfg_get(cfg, "input_sequence_length", 1))
        self.use_cache = bool(_cfg_get(cfg, "use_cache", True))
        self.llm_gather = xhnn.BatchGather(1)
        self.llm_gather.update_offset_indices(1, self.input_sequence_length)

        def _update(gather, new_cfg=None):
            gather.update_offset_indices(
                int(_cfg_get(new_cfg, "batch_size", 1)),
                int(_cfg_get(new_cfg, "input_sequence_length", 1)),
            )

        self.llm_gather._update_cfg = types.MethodType(_update, self.llm_gather)
        if not hasattr(self.rotary_emb, "cos_cached"):
            self.rotary_emb.setup_after_callback = self._setup_rope_cache
        else:
            self._setup_rope_cache()
        self.cos_slice = xhnn.DynamicSlice([self.input_sequence_length], [2], [1])
        self.sin_slice = xhnn.DynamicSlice([self.input_sequence_length], [2], [1])

        def _slice_update(slice_module, new_cfg=None):
            slice_module.valid_length = [int(_cfg_get(new_cfg, "input_sequence_length", 1))]

        self.cos_slice._update_cfg = types.MethodType(_slice_update, self.cos_slice)
        self.sin_slice._update_cfg = types.MethodType(_slice_update, self.sin_slice)
        return self

    def _setup_rope_cache(self):
        if hasattr(self.rotary_emb, "_set_dtype"):
            self.rotary_emb._set_dtype(self.embed_tokens.weight.dtype)

    def forward(
        self,
        inputs_embeds: Tensor,
        past_seq_length: Tensor | None = None,
        current_input_length: Tensor | None = None,
        past_key_cache: list[Tensor] | None = None,
        past_value_cache: list[Tensor] | None = None,
    ) -> Tensor:
        hidden_states = inputs_embeds
        cos = self.cos_slice(self.rotary_emb.cos_cached, past_seq_length)
        sin = self.sin_slice(self.rotary_emb.sin_cached, past_seq_length)
        position_embeddings = (cos, sin)
        for idx, decoder_layer in enumerate(self.layers):
            if self.max_layers > 0 and idx >= self.max_layers:
                break
            hidden_states = decoder_layer(
                hidden_states,
                past_seq_length=past_seq_length,
                current_input_length=current_input_length,
                past_k_cache=past_key_cache[idx] if self.use_cache else None,
                past_v_cache=past_value_cache[idx] if self.use_cache else None,
                position_embeddings=position_embeddings,
            )[0]
        if self.num_logits_to_keep != 0:
            hidden_states = self.llm_gather(hidden_states, current_input_length - 1)
        return self.norm(hidden_states)


class _MiniCPM5ForCausalLM(DynamicModule):
    def _setup(self, cfg: Optional[dict] = None):
        return self

    def forward(
        self,
        inputs_embeds: Tensor,
        past_seq_length: Tensor | None = None,
        current_input_length: Tensor | None = None,
        past_key_cache: list[Tensor] | None = None,
        past_value_cache: list[Tensor] | None = None,
    ) -> Tensor:
        hidden_states = self.model(
            inputs_embeds=inputs_embeds,
            past_seq_length=past_seq_length,
            current_input_length=current_input_length,
            past_key_cache=past_key_cache,
            past_value_cache=past_value_cache,
        )
        return self.lm_head(hidden_states)


_WRAPPERS_BY_REMOTE_NAME = {
    "MiniCPM5MoERMSNorm": _MiniCPM5RMSNorm,
    "MiniCPM5MoERotaryEmbedding": _MiniCPM5RotaryEmbedding,
    "MiniCPM5MoEAttention": _MiniCPM5Attention,
    "MiniCPM5MoEMoE": _MiniCPM5MoE,
    "MiniCPM5MoEDecoderLayer": _MiniCPM5DecoderLayer,
    "MiniCPM5MoEModel": _MiniCPM5Model,
    "MiniCPM5MoEForCausalLM": _MiniCPM5ForCausalLM,
}


def register_wrap_modules(hf_model: nn.Module | None = None):
    if hf_model is None:
        return None
    remote_classes = _find_remote_classes(hf_model)
    missing = [name for name in _WRAPPERS_BY_REMOTE_NAME if name not in remote_classes]
    if missing:
        raise RuntimeError(f"MiniCPM5 remote model is missing expected classes: {missing}")
    for remote_name, dynamic_cls in _WRAPPERS_BY_REMOTE_NAME.items():
        _register_or_replace_traceable({remote_classes[remote_name]: remote_name})(dynamic_cls)
    from xhquant.nn.builder import FX_LEAF_MODULES
    from xhquant.quantization.xh2a.builder import register_none_quanted_module

    FX_LEAF_MODULES._module_dict["MiniCPM5MoERotaryEmbedding"] = True
    register_none_quanted_module(remote_classes["MiniCPM5MoERotaryEmbedding"])
    return None
