from __future__ import annotations

import types
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers.models.gemma4.modeling_gemma4 import (
    Gemma4ForConditionalGeneration,
    Gemma4RMSNorm,
    Gemma4TextAttention,
    Gemma4TextDecoderLayer,
    Gemma4TextModel,
    Gemma4TextRotaryEmbedding,
)

from xhquant import nn as xhnn
from xhquant.nn import LLMCacheV2, MaskedAdd, MaskedSoftmax, RMSNorm, SoftmaxPlus
from xhquant.nn.modules.moeblock import MoeBlock
from xhquant.utils.registry import DynamicModule

from ...register import XHLLM_TRACEABLE_MODULES


def _register_or_replace_traceable(cls_to_key):
    """Register Gemma4 traceable modules idempotently across legacy/unified imports.

    The legacy ``gemma4e`` module and the unified ``gemma4`` module wrap the
    same upstream Transformers classes.  Test and workflow processes may import
    both modules, so plain registry insertion can assert on the second import.
    Last import wins intentionally because both wrappers target the same public
    HF classes and are selected by the active model path.
    """

    def decorator(dm_class):
        for nn_cls, key in cls_to_key.items():
            XHLLM_TRACEABLE_MODULES._registry[nn_cls] = dm_class
            XHLLM_TRACEABLE_MODULES._key_registry[nn_cls] = key
            XHLLM_TRACEABLE_MODULES._dynamic_classes.pop(nn_cls, None)
        return dm_class

    return decorator


def _move_parameter_to_meta(module: nn.Module, parameter_name: str) -> None:
    parameter = getattr(module, parameter_name, None)
    if parameter is None:
        return
    setattr(
        module,
        parameter_name,
        nn.Parameter(
            torch.empty_like(parameter.data, device="meta"),
            requires_grad=parameter.requires_grad,
        ),
    )


def _move_buffer_to_meta(module: nn.Module, buffer_name: str) -> None:
    buffer = getattr(module, buffer_name, None)
    if not torch.is_tensor(buffer):
        return
    meta_buffer = torch.empty_like(buffer, device="meta")
    if buffer_name in module._buffers:
        module._buffers[buffer_name] = meta_buffer
    else:
        setattr(module, buffer_name, meta_buffer)


def _register_or_replace_buffer(module: nn.Module, name: str, value: torch.Tensor) -> None:
    if name in module._buffers:
        module._buffers[name] = value
    else:
        module.register_buffer(name, value)


def _iter_defused_experts(experts: nn.Module):
    if hasattr(experts, "experts"):
        experts = experts.experts
    try:
        return list(experts)
    except TypeError:
        return []


def _pack_defused_experts_to_moeblock(moe_block: MoeBlock, experts: nn.Module) -> bool:
    expert_list = _iter_defused_experts(experts)
    if not expert_list:
        return False
    required = ("gate_proj", "up_proj", "down_proj")
    if not all(all(hasattr(expert, name) for name in required) for expert in expert_list):
        return False

    for linear_name in required:
        first_linear = getattr(expert_list[0], linear_name)
        weight = first_linear.weight
        first_quant_weight = getattr(first_linear, "quant_weight", None)
        has_quant_weight = torch.is_tensor(first_quant_weight)
        packed = torch.empty(len(expert_list), *weight.shape, device=weight.device, dtype=weight.dtype)
        packed_quant: torch.Tensor | None = None
        if has_quant_weight:
            if tuple(first_quant_weight.shape) != tuple(weight.shape):
                raise RuntimeError(
                    f"Gemma4 MoE {linear_name}.quant_weight shape {tuple(first_quant_weight.shape)} "
                    f"does not match weight shape {tuple(weight.shape)}"
                )
            packed_quant = torch.empty(
                len(expert_list),
                *first_quant_weight.shape,
                device=first_quant_weight.device,
                dtype=first_quant_weight.dtype,
            )
        with torch.no_grad():
            for expert_idx, expert in enumerate(expert_list):
                linear = getattr(expert, linear_name)
                packed[expert_idx].copy_(linear.weight.data.to(device=packed.device, dtype=packed.dtype))
                if has_quant_weight:
                    quant_weight = getattr(linear, "quant_weight", None)
                    if not torch.is_tensor(quant_weight):
                        raise RuntimeError(f"Gemma4 MoE expert {expert_idx} {linear_name} is missing quant_weight")
                    if tuple(quant_weight.shape) != tuple(weight.shape):
                        raise RuntimeError(
                            f"Gemma4 MoE expert {expert_idx} {linear_name}.quant_weight shape "
                            f"{tuple(quant_weight.shape)} does not match weight shape {tuple(weight.shape)}"
                        )
                    packed_quant[expert_idx].copy_(
                        quant_weight.data.to(device=packed_quant.device, dtype=packed_quant.dtype)
                    )
                _move_parameter_to_meta(linear, "weight")
                if getattr(linear, "bias", None) is not None:
                    _move_parameter_to_meta(linear, "bias")
                if has_quant_weight:
                    _move_buffer_to_meta(linear, "quant_weight")
        setattr(moe_block, f"expert_{linear_name}_weight", nn.Parameter(packed))
        setattr(moe_block, f"expert_{linear_name}_bias", None)
        if packed_quant is not None:
            _register_or_replace_buffer(moe_block, f"expert_{linear_name}_quant_weight", packed_quant)
    return True


def _cfg_get(cfg, name: str, default=None):
    if hasattr(cfg, name):
        value = getattr(cfg, name)
        return default if value is None else value
    if hasattr(cfg, "get"):
        value = cfg.get(name, default)
        return default if value is None else value
    return default


def _compute_gemma4_rotary_cache(
    inv_freq: Tensor,
    attention_scaling: float,
    max_seq_len: int,
) -> tuple[Tensor, Tensor]:
    positions = torch.arange(max_seq_len, device=inv_freq.device, dtype=torch.float32).view(max_seq_len, 1)
    freqs = positions * inv_freq.float().view(1, -1)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos() * attention_scaling
    sin = emb.sin() * attention_scaling
    return cos.to(dtype=inv_freq.dtype), sin.to(dtype=inv_freq.dtype)


@_register_or_replace_traceable({Gemma4RMSNorm: "Gemma4RMSNorm"})
class _Gemma4RMSNorm(DynamicModule):
    def forward(self, hidden_states):
        return self.norm(hidden_states)

    def _setup(self, cfg: Optional[Dict] = None):
        if hasattr(self, "weight") and self.weight is not None:
            hidden_size = self.weight.shape[0]
            device = self.weight.device
        elif hasattr(self, "_cached_hidden_size"):
            hidden_size = self._cached_hidden_size
            device = None
        elif hasattr(self, "dim"):
            hidden_size = self.dim
            device = None
        elif hasattr(self, "_head_dim_hint"):
            hidden_size = self._head_dim_hint
            device = None
        else:
            # Fallback for wrapped modules that can't determine hidden_size
            hidden_size = 8192  # Default for Gemma4
            device = None

        # Cache the hidden_size for later
        self._cached_hidden_size = hidden_size

        if device is None:
            try:
                device = next(self.parameters()).device
            except StopIteration:
                device = torch.device("cuda:0")  # Default to GPU for Gemma4
        self.norm = RMSNorm(hidden_size, self.eps).to(device)
        if getattr(self, "with_scale", True) and hasattr(self, "weight") and self.weight is not None:
            self.norm.weight = nn.Parameter(self.weight.data.clone().to(device))
        else:
            self.norm.weight = nn.Parameter(
                torch.ones(hidden_size, dtype=torch.float32, device=device), requires_grad=False
            )
        return self


@_register_or_replace_traceable({Gemma4TextRotaryEmbedding: "Gemma4TextRotaryEmbedding"})
class _Gemma4TextRotaryEmbedding(DynamicModule):
    def _setup(self, cfg=None):
        max_seq_len = int(
            _cfg_get(
                cfg,
                "context_max_length",
                getattr(self, "max_seq_len_cached", getattr(self, "original_max_seq_len", 2048)),
            )
        )
        for layer_type in ("full_attention", "sliding_attention"):
            inv_freq = getattr(self, f"{layer_type}_inv_freq", None)
            attention_scaling = getattr(self, f"{layer_type}_attention_scaling", None)
            if inv_freq is None or attention_scaling is None:
                continue
            cos, sin = _compute_gemma4_rotary_cache(inv_freq, attention_scaling, max_seq_len)
            self.register_buffer(f"{layer_type}_cos_cached", cos.unsqueeze(1), persistent=False)
            self.register_buffer(f"{layer_type}_sin_cached", sin.unsqueeze(1), persistent=False)
        return self

    def set_target_dtype(self, target_dtype: torch.dtype):
        for layer_type in ("full_attention", "sliding_attention"):
            cos_cache = getattr(self, f"{layer_type}_cos_cached", None)
            sin_cache = getattr(self, f"{layer_type}_sin_cached", None)
            if cos_cache is not None:
                setattr(self, f"{layer_type}_cos_cached", cos_cache.to(dtype=target_dtype))
            if sin_cache is not None:
                setattr(self, f"{layer_type}_sin_cached", sin_cache.to(dtype=target_dtype))
        return self

    def forward(self, x, position_ids, layer_type=None):
        del x
        cos_cache = getattr(self, f"{layer_type}_cos_cached")
        sin_cache = getattr(self, f"{layer_type}_sin_cached")
        return cos_cache[position_ids], sin_cache[position_ids]


@_register_or_replace_traceable({Gemma4TextAttention: "Gemma4TextAttention"})
class _Gemma4TextAttention(DynamicModule):
    def _setup(self, cfg=None):
        self.use_cache = bool(_cfg_get(cfg, "use_cache", True))
        self.use_flash_attention_v2 = int(_cfg_get(cfg, "attention_contract_version", 1)) >= 2
        self.layer_type = getattr(self, "layer_type", None)
        if self.layer_type not in ("full_attention", "sliding_attention"):
            raise ValueError(
                f"Gemma4 attention layer_type must be 'full_attention' or 'sliding_attention', got {self.layer_type!r}"
            )
        self.rotary_half_dim = self.head_dim // 2
        self.num_attention_heads = self.q_proj.weight.shape[0] // self.head_dim
        k_proj = getattr(self, "k_proj", None)
        if k_proj is not None:
            self.num_key_value_heads = k_proj.weight.shape[0] // self.head_dim
        else:
            # Gemma4 assistant layers and target KV-shared suffix layers are
            # Q-only: Transformers does not construct k_proj/v_proj/k_norm/v_norm.
            # They still need the target KV-head count for repeat_interleave
            # after reading shared KV by attention type.
            layer_type = getattr(self, "layer_type", None)
            config = getattr(self, "config", None)
            use_global_kv = layer_type == "full_attention" and bool(getattr(config, "attention_k_eq_v", False))
            if use_global_kv and getattr(config, "num_global_key_value_heads", None):
                self.num_key_value_heads = int(config.num_global_key_value_heads)
            else:
                self.num_key_value_heads = int(config.num_key_value_heads)
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        for norm_name in ("q_norm", "k_norm", "v_norm"):
            norm = getattr(self, norm_name, None)
            if norm is not None and not getattr(norm, "with_scale", True):
                norm._head_dim_hint = self.head_dim
        self.rope = xhnn.Rope()
        self.attn_compute_cast = xhnn.Cast(torch.float16).to(dtype=torch.float16)
        self.attn_output_cast = xhnn.Cast(self.o_proj.weight.dtype).to(dtype=self.o_proj.weight.dtype)
        raw_sliding_window = getattr(self, "sliding_window", None)
        if self.use_flash_attention_v2:
            if self.layer_type == "sliding_attention":
                if type(raw_sliding_window) is not int or raw_sliding_window <= 0:
                    raise ValueError(
                        "Gemma4 contract-v2 sliding_attention requires sliding_window "
                        f"to be a positive integer, got {raw_sliding_window!r}"
                    )
                attention_max_length = raw_sliding_window
                self.is_sliding_attention = True
            else:
                if raw_sliding_window is not None:
                    raise ValueError(
                        f"Gemma4 contract-v2 full_attention requires sliding_window=None, got {raw_sliding_window!r}"
                    )
                attention_max_length = -1
                self.is_sliding_attention = False
        else:
            attention_max_length = int(raw_sliding_window) if raw_sliding_window is not None else -1
            self.is_sliding_attention = attention_max_length > 0
        self.enable_accepted_count_input = False
        if self.use_flash_attention_v2:
            self.flash_attn = xhnn.FlashAttention(
                embed_dim=self.num_attention_heads * self.head_dim,
                num_heads=self.num_attention_heads,
                batch_first=True,
                # Gemma4's Q/K RMSNorm contract already uses the model's
                # explicit attention scaling (1.0).  Applying the generic
                # 1/sqrt(head_dim) factor here changes every v2 layer.
                scale=1.0,
                num_kv_heads=self.num_key_value_heads,
                is_causal=True,
                sliding_window=attention_max_length if self.is_sliding_attention else None,
            )
        else:
            self.k_repeat_interleave = xhnn.RepeatInterleave()
            self.v_repeat_interleave = xhnn.RepeatInterleave()
            self.qk_matmul = xhnn.MatMul()
            self.pv_matmul = xhnn.MatMul()
            # Target full-attention decode/prefill intentionally uses the standard
            # causal MaskedSoftmax path by passing attention_mask=None.  MTP draft
            # full layers can still receive an explicit assistant-side mask to hide
            # invalid padded cache slots.
            self.masked_add = MaskedAdd()
            self.softmax = SoftmaxPlus(dim=-1)
            self.masked_softmax = MaskedSoftmax(dim=-1, attention_max_length=-1)
        if self.use_cache:
            cache_axis = cfg.kv_cache.cache_axis
            self.k_cache = LLMCacheV2(axis=cache_axis, attention_max_length=attention_max_length)
            self.v_cache = LLMCacheV2(axis=cache_axis, attention_max_length=attention_max_length)
        else:
            self.k_cache = None
            self.v_cache = None
        return self

    def _update_cfg(self, cfg: Optional[Dict] = None):
        if cfg is None:
            return self
        requested_accepted_count = bool(
            _cfg_get(cfg, "enable_accepted_count_input", getattr(self, "enable_accepted_count_input", False))
        )
        # accepted_count is part of the target verify decode contract only.
        # The wrapper requests it for MTP slice-window decode graphs, and the
        # attention layer further narrows consumption to local/sliding layers.
        self.enable_accepted_count_input = requested_accepted_count and self.is_sliding_attention
        if not self.use_cache or self.k_cache is None or self.v_cache is None:
            return self
        if self.is_sliding_attention:
            # LLMCache's ``attention_max_length`` is the model's local-attention
            # window.  The exported sliding output width is therefore naturally
            # aligned(sliding_window + current_seq_len - 1, 16): prefill with
            # q=320 yields 1344/832, while MTP verify decode with q=5 yields
            # 1040/528.  The physical cache input can still be the larger
            # slice_window + prefill_input_length tensor.
            attention_max_length = int(getattr(self, "sliding_window", -1) or -1)
            self.k_cache.attention_max_length = attention_max_length
            self.v_cache.attention_max_length = attention_max_length
        return self

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor,
        attention_mask: torch.Tensor | None,
        past_seq_length: Optional[Tensor] = None,
        current_input_length: Optional[Tensor] = None,
        past_k_cache: Optional[Tensor] = None,
        past_v_cache: Optional[Tensor] = None,
        accepted_count: Optional[Tensor] = None,
        shared_kv: Optional[dict[int | str, tuple[torch.Tensor, torch.Tensor]]] = None,
        mm_prefix_ranges: Optional[Tensor] = None,
        kv_window_start_abs: Optional[Tensor] = None,
        kv_valid_length: Optional[Tensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        del kwargs
        batch_size = hidden_states.shape[0]
        seq_length = hidden_states.shape[1]
        cos, sin = position_embeddings

        query_states = self.q_proj(hidden_states).view(batch_size, seq_length, -1, self.head_dim)
        query_states = self.q_norm(query_states).transpose(1, 2)
        query_states = self.attn_compute_cast(query_states)
        query_states = self.attn_compute_cast(self.rope(query_states, cos, sin))

        if getattr(self, "is_kv_shared_layer", False):
            assert shared_kv is not None
            layer_type = getattr(self, "layer_type", None)
            if layer_type in shared_kv:
                key_states, value_states = shared_kv[layer_type]
            else:
                # Compatibility for older local experiments that keyed by an
                # explicit source layer index.  New Gemma4 Series MTP must use
                # layer type because vLLM/Transformers map draft S/S/S/F layers
                # to the last non-shared target layer of the same type.
                key_states, value_states = shared_kv[self.kv_shared_layer_index]
        else:
            key_states = self.k_proj(hidden_states).view(batch_size, seq_length, -1, self.head_dim)
            # Gemma4 dense global layers can use attention_k_eq_v=True: v_proj is absent and V reuses K input.
            value_states = (
                self.v_proj(hidden_states).view(batch_size, seq_length, -1, self.head_dim)
                if self.v_proj is not None
                else key_states
            )

            key_states = self.k_norm(key_states).transpose(1, 2)
            key_states = self.attn_compute_cast(key_states)
            key_states = self.attn_compute_cast(self.rope(key_states, cos, sin))
            value_states = self.attn_compute_cast(self.v_norm(value_states)).transpose(1, 2)

            if self.use_cache and past_k_cache is not None and past_v_cache is not None:
                cache_accepted_count = accepted_count if getattr(self, "enable_accepted_count_input", False) else None
                key_states = self.k_cache(
                    key_states,
                    past_seq_length,
                    current_input_length,
                    past_k_cache,
                    cache_accepted_count,
                )
                value_states = self.v_cache(
                    value_states,
                    past_seq_length,
                    current_input_length,
                    past_v_cache,
                    cache_accepted_count,
                )

        if getattr(self, "store_full_length_kv", False):
            assert shared_kv is not None
            shared_kv[self.layer_idx] = (key_states, value_states)
            layer_type = getattr(self, "layer_type", None)
            if layer_type is not None:
                shared_kv[layer_type] = (key_states, value_states)

        if getattr(self, "use_flash_attention_v2", False):
            attn_output = self.flash_attn(
                query_states,
                key_states,
                value_states,
                past_seq_length=past_seq_length,
                current_input_length=current_input_length,
                mm_prefix_range=mm_prefix_ranges,
                kv_window_start_abs=kv_window_start_abs if self.is_sliding_attention else None,
                kv_valid_length=kv_valid_length if self.is_sliding_attention else None,
            )
            attn_weights = None
        else:
            key_states = self.k_repeat_interleave(key_states.transpose(2, 3), self.num_key_value_groups, 1)
            value_states = self.v_repeat_interleave(value_states, self.num_key_value_groups, 1)

            attn_weights = self.qk_matmul(query_states, key_states)
            if attention_mask is not None:
                attn_weights = self.masked_add(attn_weights, attention_mask)
                attn_weights = self.softmax(attn_weights).to(query_states.dtype)
            else:
                attn_weights = self.masked_softmax(attn_weights, past_seq_length).to(query_states.dtype)

            attn_output = self.pv_matmul(attn_weights, value_states).transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
        attn_output = self.attn_output_cast(attn_output)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


@_register_or_replace_traceable({Gemma4TextDecoderLayer: "Gemma4TextDecoderLayer"})
class _Gemma4TextDecoderLayer(DynamicModule):
    def _setup(self, cfg: Optional[Dict] = None):
        layer_scalar = getattr(self, "layer_scalar", None)
        if layer_scalar is not None and torch.all(layer_scalar == 1):
            del self.layer_scalar

        if getattr(self, "enable_moe_block", False):
            router = self.router
            experts = self.experts

            moe_scale = router.scale.data.clone() * torch.as_tensor(
                router.scalar_root_size,
                dtype=router.scale.dtype,
                device=router.scale.device,
            )
            router_norm = router.norm
            router_norm_weight = getattr(router_norm, "weight", None)
            if router_norm_weight is not None and getattr(router_norm, "with_scale", True):
                moe_scale = moe_scale * router_norm_weight.data.to(moe_scale)
            self.moe_router_norm = RMSNorm(router.hidden_size, router_norm.eps).to(
                device=moe_scale.device,
                dtype=moe_scale.dtype,
            )
            self.moe_router_norm.weight = nn.Parameter(moe_scale)

            moe_gate = nn.Linear(
                router.hidden_size,
                router.proj.out_features,
                bias=False,
                device=router.proj.weight.device,
                dtype=router.proj.weight.dtype,
            )
            moe_gate.weight.data.copy_(router.proj.weight.data)
            self.moe_gate = moe_gate

            self.register_buffer("_moe_per_expert_scale", router.per_expert_scale.data.clone())
            self._moe_top_k = router.config.top_k_experts

            self.moe_block = MoeBlock("gelu", self._moe_top_k, normalize_routing_weights=True)
            if hasattr(experts, "gate_up_proj") and hasattr(experts, "down_proj"):
                intermediate_dim = experts.intermediate_dim
                gate_up = experts.gate_up_proj.data
                gate_proj_weight = gate_up[:, :intermediate_dim, :].contiguous()
                up_proj_weight = gate_up[:, intermediate_dim:, :].contiguous()
                down_proj_weight = experts.down_proj.data.contiguous()
                gate_up_quant_weight = getattr(experts, "gate_up_proj_quant_weight", None)
                down_proj_quant_weight = getattr(experts, "down_proj_quant_weight", None)

                self.moe_block.expert_gate_proj_weight = nn.Parameter(gate_proj_weight)
                self.moe_block.expert_gate_proj_bias = None
                self.moe_block.expert_up_proj_weight = nn.Parameter(up_proj_weight)
                self.moe_block.expert_up_proj_bias = None
                self.moe_block.expert_down_proj_weight = nn.Parameter(down_proj_weight)
                self.moe_block.expert_down_proj_bias = None
                if torch.is_tensor(gate_up_quant_weight):
                    gate_quant_weight = gate_up_quant_weight[:, :intermediate_dim, :].contiguous()
                    up_quant_weight = gate_up_quant_weight[:, intermediate_dim:, :].contiguous()
                    _register_or_replace_buffer(self.moe_block, "expert_gate_proj_quant_weight", gate_quant_weight)
                    _register_or_replace_buffer(self.moe_block, "expert_up_proj_quant_weight", up_quant_weight)
                if torch.is_tensor(down_proj_quant_weight):
                    _register_or_replace_buffer(
                        self.moe_block,
                        "expert_down_proj_quant_weight",
                        down_proj_quant_weight.contiguous(),
                    )
            elif not _pack_defused_experts_to_moeblock(self.moe_block, experts):
                raise RuntimeError(f"Unsupported Gemma4 MoE experts structure: {type(experts)}")

            del self.router
            del self.experts
        return self

    def forward(
        self,
        hidden_states: torch.Tensor,
        per_layer_input: torch.Tensor = None,
        position_embeddings: torch.Tensor = None,
        attention_mask: torch.Tensor | None = None,
        past_seq_length: Optional[Tensor] = None,
        current_input_length: Optional[Tensor] = None,
        past_k_cache: Optional[Tensor] = None,
        past_v_cache: Optional[Tensor] = None,
        accepted_count: Optional[Tensor] = None,
        shared_kv: Optional[dict[int, tuple[torch.Tensor, torch.Tensor]]] = None,
        mm_prefix_ranges: Optional[Tensor] = None,
        kv_window_start_abs: Optional[Tensor] = None,
        kv_valid_length: Optional[Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        del kwargs

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_seq_length=past_seq_length,
            current_input_length=current_input_length,
            past_k_cache=past_k_cache,
            past_v_cache=past_v_cache,
            accepted_count=accepted_count,
            shared_kv=shared_kv,
            mm_prefix_ranges=mm_prefix_ranges,
            kv_window_start_abs=kv_window_start_abs,
            kv_valid_length=kv_valid_length,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        if getattr(self, "enable_moe_block", False):
            hidden_states_1 = self.post_feedforward_layernorm_1(hidden_states)

            batch_size, seq_length, hidden_dim = residual.shape
            hidden_states_flat = residual.reshape(-1, hidden_dim)
            scaled = self.moe_router_norm(hidden_states_flat)
            expert_scores = self.moe_gate(scaled)
            routing_probs = F.softmax(expert_scores, dim=-1)
            moe_per_expert_scale = self._moe_per_expert_scale.to(routing_probs)
            routing_probs = routing_probs * moe_per_expert_scale
            routing_weights = routing_probs.view(batch_size, seq_length, -1)

            hidden_states_2 = self.pre_feedforward_layernorm_2(residual)
            hidden_states_2 = self.moe_block(hidden_states_2, routing_weights, fast_mode=False)
            hidden_states_2 = hidden_states_2.to(residual.dtype)
            hidden_states_2 = self.post_feedforward_layernorm_2(hidden_states_2)
            hidden_states = hidden_states_1 + hidden_states_2

        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        if getattr(self, "hidden_size_per_layer_input", 0):
            residual = hidden_states
            hidden_states = self.per_layer_input_gate(hidden_states)
            hidden_states = self.act_fn(hidden_states)
            hidden_states = hidden_states * per_layer_input
            hidden_states = self.per_layer_projection(hidden_states)
            hidden_states = self.post_per_layer_input_norm(hidden_states)
            hidden_states = residual + hidden_states

        if hasattr(self, "layer_scalar"):
            hidden_states = hidden_states * self.layer_scalar
        return hidden_states


@_register_or_replace_traceable({Gemma4TextModel: "Gemma4TextModel"})
class _Gemma4TextModel(DynamicModule):
    def _setup(self, cfg: Optional[Dict] = None):
        self.only_first_block = bool(_cfg_get(cfg, "only_first_block", False))
        self.enable_mtp_outputs = bool(
            _cfg_get(cfg, "enable_mtp_outputs", False) or _cfg_get(cfg, "spec_decode_mode", None) == "mtp"
        )
        self.num_logits_to_keep = int(_cfg_get(cfg, "num_logits_to_keep", 1))
        self.input_sequence_length = int(_cfg_get(cfg, "input_sequence_length", 1))
        self.image_token_id = int(_cfg_get(cfg, "image_token_id", -1))
        self.audio_token_id = int(_cfg_get(cfg, "audio_token_id", -1))
        self.video_token_id = int(_cfg_get(cfg, "video_token_id", -1))
        self.per_layer_embed_scale = float(getattr(self, "hidden_size_per_layer_input", 0) or 0) ** 0.5

        self.llm_gather = xhnn.BatchGather(1)
        self.llm_gather.update_offset_indices(1, self.input_sequence_length)

        def _update_gather_cfg(batch_gather: xhnn.BatchGather, new_cfg=None):
            input_sequence_length = int(_cfg_get(new_cfg, "input_sequence_length", self.input_sequence_length))
            batch_size = int(_cfg_get(new_cfg, "batch_size", 1))
            batch_gather.update_offset_indices(batch_size, input_sequence_length)

        self.llm_gather._update_cfg = types.MethodType(_update_gather_cfg, self.llm_gather)
        if hasattr(self.rotary_emb, "set_target_dtype"):
            self.rotary_emb.set_target_dtype(self.embed_tokens.weight.dtype)
        if self.hidden_size_per_layer_input and not getattr(self, "_ple_scales_fused", False):
            projection_scale = float(self.per_layer_model_projection_scale)
            input_scale = float(self.per_layer_input_scale)
            self.per_layer_model_projection.weight.data.mul_(projection_scale)
            if self.per_layer_model_projection.bias is not None:
                self.per_layer_model_projection.bias.data.mul_(projection_scale)
            norm_weight = getattr(self.per_layer_projection_norm, "weight", None)
            if norm_weight is not None:
                norm_weight.data.mul_(input_scale)
            elif hasattr(self.per_layer_projection_norm, "norm") and hasattr(
                self.per_layer_projection_norm.norm, "weight"
            ):
                self.per_layer_projection_norm.norm.weight.data.mul_(input_scale)
            self.per_layer_model_projection_scale = 1.0
            self.per_layer_embed_scale = 1.0
            self.per_layer_input_scale = 1.0
            self._ple_scales_fused = True
        self._setup_rope_cache(cfg)
        return self

    def _update_cfg(self, cfg: Optional[Dict] = None):
        if cfg is None:
            return self
        self.enable_mtp_outputs = bool(
            _cfg_get(cfg, "enable_mtp_outputs", self.enable_mtp_outputs)
            or _cfg_get(cfg, "spec_decode_mode", None) == "mtp"
        )
        self.num_logits_to_keep = int(_cfg_get(cfg, "num_logits_to_keep", self.num_logits_to_keep) or 0)
        self.input_sequence_length = int(_cfg_get(cfg, "input_sequence_length", self.input_sequence_length))
        if hasattr(self, "llm_gather"):
            self.llm_gather._update_cfg(cfg)
        return self

    def _setup_rope_cache(self, cfg: Optional[Dict] = None):
        max_seq_len = getattr(self.config, "max_position_embeddings", None)
        if max_seq_len is None:
            max_seq_len = int(
                _cfg_get(
                    cfg,
                    "context_max_length",
                    getattr(
                        self.rotary_emb, "max_seq_len_cached", getattr(self.rotary_emb, "original_max_seq_len", 2048)
                    ),
                )
            )
        max_seq_len = int(max_seq_len)
        target_dtype = self.embed_tokens.weight.dtype
        for layer_type in set(self.config.layer_types):
            inv_freq = getattr(self.rotary_emb, f"{layer_type}_inv_freq", None)
            attention_scaling = getattr(self.rotary_emb, f"{layer_type}_attention_scaling", None)
            if inv_freq is None or attention_scaling is None:
                continue
            cos, sin = _compute_gemma4_rotary_cache(inv_freq, attention_scaling, max_seq_len)
            cos_buf = cos.unsqueeze(0).unsqueeze(0).to(dtype=target_dtype).contiguous()
            sin_buf = sin.unsqueeze(0).unsqueeze(0).to(dtype=target_dtype).contiguous()
            self.register_buffer(f"_{layer_type}_cos_cache", cos_buf, persistent=False)
            self.register_buffer(f"_{layer_type}_sin_cache", sin_buf, persistent=False)
            cos_slice = xhnn.DynamicSlice([self.input_sequence_length], [2], [1])
            sin_slice = xhnn.DynamicSlice([self.input_sequence_length], [2], [1])

            def _slice_update_cfg(slice_self, new_cfg=None):
                slice_self.valid_length = [int(_cfg_get(new_cfg, "input_sequence_length", self.input_sequence_length))]

            cos_slice._update_cfg = types.MethodType(_slice_update_cfg, cos_slice)
            sin_slice._update_cfg = types.MethodType(_slice_update_cfg, sin_slice)
            setattr(self, f"_{layer_type}_cos_slice", cos_slice)
            setattr(self, f"_{layer_type}_sin_slice", sin_slice)

    def _get_per_layer_inputs(self, input_ids: Tensor) -> Tensor:
        batch_size = input_ids.shape[0]
        seq_length = input_ids.shape[1]
        per_layer_inputs = self.embed_tokens_per_layer(input_ids).reshape(
            batch_size,
            seq_length,
            self.config.num_hidden_layers,
            self.hidden_size_per_layer_input,
        )
        return per_layer_inputs

    def _project_per_layer_inputs(self, inputs_embeds: Tensor, per_layer_inputs: Optional[Tensor] = None) -> Tensor:
        batch_size = inputs_embeds.shape[0]
        seq_length = inputs_embeds.shape[1]
        per_layer_projection = self.per_layer_model_projection(inputs_embeds)
        per_layer_projection = per_layer_projection.reshape(
            batch_size,
            seq_length,
            self.config.num_hidden_layers,
            self.hidden_size_per_layer_input,
        )
        per_layer_projection = per_layer_projection.permute(0, 2, 1, 3).contiguous()
        per_layer_projection = self.per_layer_projection_norm(per_layer_projection)
        if per_layer_inputs is None:
            return per_layer_projection
        # Unified preprocessing exports raw embed_tokens_per_layer output as [B, S, L, H].
        # Convert once to the legacy/Transformers math layout [B, L, S, H] without
        # Python shape branching so torch.fx can trace the full graph.
        per_layer_inputs = per_layer_inputs.permute(0, 2, 1, 3).contiguous()
        per_layer_inputs = per_layer_inputs.to(per_layer_projection.dtype)
        return per_layer_projection + per_layer_inputs

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        per_layer_inputs: Optional[Tensor] = None,
        past_seq_length: Optional[Tensor] = None,
        current_input_length: Optional[Tensor] = None,
        local_attention_mask: Optional[Tensor] = None,
        full_attention_mask: Optional[Tensor] = None,
        sliding_attention_mask: Optional[Tensor] = None,
        past_key_cache: Optional[list[Tensor]] = None,
        past_value_cache: Optional[list[Tensor]] = None,
        accepted_count: Optional[Tensor] = None,
        mm_prefix_ranges: Optional[Tensor] = None,
        kv_window_start_abs: Optional[Tensor] = None,
        kv_valid_length: Optional[Tensor] = None,
    ):
        if inputs_embeds is None:
            raise ValueError("Gemma4 text graph requires inputs_embeds.")
        if past_seq_length is None:
            raise ValueError("Gemma4 text graph requires past_seq_length.")
        if local_attention_mask is None:
            local_attention_mask = sliding_attention_mask
        if self.hidden_size_per_layer_input:
            if per_layer_inputs is None:
                if input_ids is None:
                    raise ValueError("Gemma4 text graph requires explicit per_layer_inputs.")
                input_ids = input_ids.to(torch.long)
                per_layer_inputs = self._project_per_layer_inputs(inputs_embeds, self._get_per_layer_inputs(input_ids))
            else:
                per_layer_inputs = self._project_per_layer_inputs(inputs_embeds, per_layer_inputs)
        else:
            per_layer_inputs = None

        position_embeddings_by_type: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for layer_type in set(self.config.layer_types):
            cos_cache = getattr(self, f"_{layer_type}_cos_cache")
            sin_cache = getattr(self, f"_{layer_type}_sin_cache")
            cos_slice = getattr(self, f"_{layer_type}_cos_slice")
            sin_slice = getattr(self, f"_{layer_type}_sin_slice")
            position_embeddings_by_type[layer_type] = (
                cos_slice(cos_cache, past_seq_length),
                sin_slice(sin_cache, past_seq_length),
            )

        hidden_states = inputs_embeds
        shared_kv: dict[int | str, tuple[torch.Tensor, torch.Tensor]] = {}
        cache_idx = 0
        for layer_idx, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            if getattr(decoder_layer.self_attn, "is_kv_shared_layer", False):
                layer_past_key_cache = None
                layer_past_value_cache = None
            else:
                layer_past_key_cache = past_key_cache[cache_idx] if past_key_cache is not None else None
                layer_past_value_cache = past_value_cache[cache_idx] if past_value_cache is not None else None
                cache_idx += 1

            layer_type = self.config.layer_types[layer_idx]
            layer_position_embeddings = position_embeddings_by_type[layer_type]
            layer_attention_mask = local_attention_mask if layer_type == "sliding_attention" else full_attention_mask
            layer_per_input = per_layer_inputs[:, layer_idx, :, :] if per_layer_inputs is not None else None

            hidden_states = decoder_layer(
                hidden_states,
                per_layer_input=layer_per_input,
                position_embeddings=layer_position_embeddings,
                attention_mask=layer_attention_mask,
                past_seq_length=past_seq_length,
                current_input_length=current_input_length,
                past_k_cache=layer_past_key_cache,
                past_v_cache=layer_past_value_cache,
                accepted_count=accepted_count,
                shared_kv=shared_kv,
                mm_prefix_ranges=mm_prefix_ranges,
                kv_window_start_abs=kv_window_start_abs,
                kv_valid_length=kv_valid_length,
            )
            if self.only_first_block:
                break

        hidden_states = self.norm(hidden_states)
        if self.num_logits_to_keep == 0:
            if self.enable_mtp_outputs:
                return hidden_states
            return hidden_states
        hidden_states = self.llm_gather(hidden_states, current_input_length - 1)
        if self.enable_mtp_outputs:
            return hidden_states
        return hidden_states


@_register_or_replace_traceable({Gemma4ForConditionalGeneration: "Gemma4ForConditionalGeneration"})
class _Gemma4ForConditionalGeneration(DynamicModule):
    def _setup(self, cfg: Optional[Dict] = None):
        if cfg is None:
            self.num_logits_to_keep = 0
            self.enable_mtp_outputs = False
        elif hasattr(cfg, "get"):
            self.num_logits_to_keep = int(cfg.get("num_logits_to_keep", 0) or 0)
            self.enable_mtp_outputs = bool(cfg.get("enable_mtp_outputs", False) or cfg.get("spec_decode_mode") == "mtp")
        else:
            self.num_logits_to_keep = int(getattr(cfg, "num_logits_to_keep", 0) or 0)
            self.enable_mtp_outputs = bool(
                getattr(cfg, "enable_mtp_outputs", False) or getattr(cfg, "spec_decode_mode", None) == "mtp"
            )
        # _Gemma4TextModel already applies num_logits_to_keep with BatchGather.
        # Keep the outer HF conditional wrapper branch-free for torch.fx.
        self.valid_logits_slice = None
        return self

    def forward(
        self,
        inputs_embeds: Tensor,
        past_seq_length: Tensor,
        current_input_length: Tensor,
        full_attention_mask: Tensor,
        sliding_attention_mask: Tensor,
        past_key_cache: List[Tensor],
        past_value_cache: List[Tensor],
    ) -> Tensor:
        outputs = self.model.language_model(
            inputs_embeds=inputs_embeds,
            past_seq_length=past_seq_length,
            current_input_length=current_input_length,
            full_attention_mask=full_attention_mask,
            sliding_attention_mask=sliding_attention_mask,
            past_key_cache=past_key_cache,
            past_value_cache=past_value_cache,
        )
        # The text model owns the actual MTP-output switch during export.
        # Keep this branch fully static so FX/torch.export sees tensor getitems,
        # never a runtime isinstance check or a tuple flowing into lm_head.
        mtp_outputs_enabled = self.enable_mtp_outputs or bool(
            getattr(self.model.language_model, "enable_mtp_outputs", False)
        )
        hidden_states = outputs
        if self.valid_logits_slice is not None:
            hidden_states = self.valid_logits_slice(hidden_states, current_input_length - 1)
        logits = self.lm_head(hidden_states)
        final_logit_softcapping = getattr(self.config.text_config, "final_logit_softcapping", None)
        if final_logit_softcapping is not None:
            logits = logits / final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * final_logit_softcapping
        if mtp_outputs_enabled:
            return logits, hidden_states
        return logits


def register_wrap_modules():
    from xhquant.nn.builder import FX_LEAF_MODULES
    from xhquant.nn.modules.normalized_modules import FloorDiv
    from xhquant.quantization.xh2a.builder import register_none_quanted_module

    FX_LEAF_MODULES._module_dict["Gemma4TextRotaryEmbedding"] = True

    register_none_quanted_module(Gemma4TextRotaryEmbedding)
    register_none_quanted_module(FloorDiv)
    return None
