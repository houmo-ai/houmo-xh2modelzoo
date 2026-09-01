import math
import sys
import types
from copy import deepcopy
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

from xhquant import nn as xhnn
from xhquant.api import ConfigDict
from xhquant.nn import FlashAttention, LLMCacheV2, MaskedAdd, MaskedSoftmax, RMSNorm, SoftmaxPlus
from xhquant.nn.modules.moeblock import MoeBlock
from xhquant.utils.registry import DynamicModule

from ...register import XHLLM_TRACEABLE_MODULES


def _cfg_get(cfg, name: str, default=None):
    if cfg is None:
        return default
    if hasattr(cfg, name):
        value = getattr(cfg, name)
        return default if value is None else value
    if hasattr(cfg, "get"):
        value = cfg.get(name, default)
        return default if value is None else value
    return default


def _module_floating_dtype_device(module: nn.Module) -> tuple[torch.dtype, torch.device]:
    """Resolve the activation dtype/device for dense and packed linear modules."""

    for name in ("weight", "scales", "bias"):
        value = getattr(module, name, None)
        if torch.is_tensor(value) and value.dtype.is_floating_point:
            return value.dtype, value.device

    for value in (*module.parameters(recurse=False), *module.buffers(recurse=False)):
        if value.dtype.is_floating_point:
            return value.dtype, value.device

    raise TypeError(f"{type(module).__name__} has no floating-point parameter or buffer")


def _register_or_replace_traceable(cls_to_key):
    def decorator(dm_class):
        for nn_cls, key in cls_to_key.items():
            XHLLM_TRACEABLE_MODULES._registry[nn_cls] = dm_class
            XHLLM_TRACEABLE_MODULES._key_registry[nn_cls] = key
            XHLLM_TRACEABLE_MODULES._dynamic_classes.pop(nn_cls, None)
        return dm_class

    return decorator


def _class_name(module_or_cls) -> str:
    cls = module_or_cls if isinstance(module_or_cls, type) else module_or_cls.__class__
    return cls.__name__


def _find_remote_classes(hf_model: nn.Module) -> dict[str, type[nn.Module]]:
    classes: dict[str, type[nn.Module]] = {}
    for module in hf_model.modules():
        name = _class_name(module)
        if name.startswith("Laguna"):
            classes.setdefault(name, module.__class__)
    return classes


def _compute_rotary_cache(inv_freq: Tensor, attention_scaling: float, max_seq_len: int) -> tuple[Tensor, Tensor]:
    positions = torch.arange(max_seq_len, device=inv_freq.device, dtype=torch.float32).view(max_seq_len, 1)
    freqs = positions * inv_freq.float().view(1, -1)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos() * attention_scaling
    sin = emb.sin() * attention_scaling
    return cos.to(dtype=inv_freq.dtype), sin.to(dtype=inv_freq.dtype)


def _compute_default_inv_freq(config, device: torch.device) -> tuple[Tensor, float]:
    rope_parameters = config.rope_parameters
    base = rope_parameters["rope_theta"]
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    rotary_dim = int(head_dim * rope_parameters.get("partial_rotary_factor", 1.0))
    exponent = torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=device) / rotary_dim
    return 1.0 / (base**exponent), 1.0


def _activation_name(act_fn, config) -> str:
    hidden_act = getattr(config, "hidden_act", None) if config is not None else None
    if hidden_act:
        return str(hidden_act)
    if hasattr(act_fn, "_get_name"):
        return act_fn._get_name().lower()
    return act_fn.__class__.__name__.lower()


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


def _register_or_replace_buffer(module: nn.Module, name: str, value: torch.Tensor) -> None:
    if name in module._buffers:
        module._buffers[name] = value
    else:
        module.register_buffer(name, value, persistent=True)


class _LagunaRMSNorm(DynamicModule):
    def forward(self, hidden_states):
        return self.norm(hidden_states)

    def _setup(self, cfg: Optional[dict] = None):
        hidden_size = self.weight.shape[0]
        self.norm = RMSNorm(hidden_size, self.variance_epsilon).to(
            device=self.weight.device,
            dtype=self.weight.dtype,
        )
        self.norm.weight = nn.Parameter(deepcopy(self.weight.data))
        return self


class _LagunaRotaryEmbedding(DynamicModule):
    def _setup(self, cfg: Optional[dict] = None):
        if self.rope_type == "default":
            inv_freq, attention_scaling = _compute_default_inv_freq(self.config, self.inv_freq.device)
        else:
            rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
            inv_freq, attention_scaling = rope_init_fn(self.config, device=self.inv_freq.device)
        self.inv_freq = inv_freq.float()
        self.original_inv_freq = inv_freq.float().clone()
        self.attention_scaling = attention_scaling
        max_seq_len = int(
            _cfg_get(
                cfg,
                "context_max_length",
                getattr(self, "max_seq_len_cached", getattr(self, "original_max_seq_len", 2048)),
            )
        )
        max_seq_len = max(max_seq_len, int(getattr(self, "max_seq_len_cached", max_seq_len)))
        self.max_seq_len_cached = max_seq_len
        cos, sin = _compute_rotary_cache(self.inv_freq, self.attention_scaling, max_seq_len)
        self.register_buffer("cos_cached", cos.unsqueeze(0).unsqueeze(0).contiguous(), persistent=False)
        self.register_buffer("sin_cached", sin.unsqueeze(0).unsqueeze(0).contiguous(), persistent=False)
        if hasattr(self, "setup_after_callback"):
            self.setup_after_callback()
        return self

    def _set_dtype(self, dtype: torch.dtype) -> None:
        cos, sin = _compute_rotary_cache(self.inv_freq, self.attention_scaling, self.max_seq_len_cached)
        self.cos_cached = cos.unsqueeze(0).unsqueeze(0).contiguous().to(dtype=dtype)
        self.sin_cached = sin.unsqueeze(0).unsqueeze(0).contiguous().to(dtype=dtype)

    @torch.no_grad()
    def forward(self, x, position_ids):
        del x
        return self.cos_cached[:, :, position_ids], self.sin_cached[:, :, position_ids]


class _LagunaAttention(DynamicModule):
    def rotate_half(self, x: Tensor):
        x1 = self.slice_1(x)
        x2 = self.slice_2(x)
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(self, q: Tensor, k: Tensor, cos: Tensor, sin: Tensor):
        rotary_dim = cos.shape[-1]
        q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
        k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
        if self.enable_rope:
            q_rot = self.rope(q_rot, cos, sin)
            k_rot = self.rope(k_rot, cos, sin)
        else:
            q_rot = (q_rot * cos) + (self.rotate_half(q_rot) * sin)
            k_rot = (k_rot * cos) + (self.rotate_half(k_rot) * sin)
        return torch.cat([q_rot, q_pass], dim=-1), torch.cat([k_rot, k_pass], dim=-1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_seq_length: Optional[Tensor] = None,
        current_input_length: Optional[Tensor] = None,
        past_k_cache: Optional[Tensor] = None,
        past_v_cache: Optional[Tensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, None, None]:
        del kwargs
        batch_size, seq_length, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states).view(batch_size, seq_length, self.num_heads, self.head_dim)
        key_states = self.k_proj(hidden_states).view(
            batch_size,
            seq_length,
            self.num_key_value_heads,
            self.head_dim,
        )
        value_states = self.v_proj(hidden_states).view(
            batch_size,
            seq_length,
            self.num_key_value_heads,
            self.head_dim,
        )

        query_states = self.q_norm(query_states).transpose(1, 2)
        key_states = self.k_norm(key_states).transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = self.apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if self.use_cache:
            key_states = self.k_cache(key_states, past_seq_length, current_input_length, past_k_cache)
            value_states = self.v_cache(value_states, past_seq_length, current_input_length, past_v_cache)

        if self.use_flash_attention:
            attn_output = self.flash_attn(
                query_states,
                key_states,
                value_states,
                past_seq_length=past_seq_length,
                current_input_length=current_input_length,
            )
        else:
            query_states = query_states * self.kv_scale
            key_states = key_states.transpose(2, 3)
            key_states = self.k_repeat_interleave(key_states, self.num_key_value_groups, 1)
            attn_weights = torch.matmul(query_states, key_states)
            if attention_mask is None:
                attn_weights = self.masked_softmax(attn_weights, past_seq_length)
            else:
                attn_weights = self.masked_add(attn_weights, attention_mask)
                attn_weights = self.softmax(attn_weights).to(query_states.dtype)

            value_states = self.v_repeat_interleave(value_states, self.num_key_value_groups, 1)
            attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_length, self.attn_hidden_dim)

        if self.gating:
            gate = F.softplus(self.gate_compute_cast(self.g_proj(hidden_states)))
            gate = self.gate_output_cast(gate)
            if self.gate_per_head:
                gate = gate.unsqueeze(-1)
                attn_output = attn_output.view(batch_size, seq_length, self.num_heads, self.head_dim) * gate
                attn_output = attn_output.reshape(batch_size, seq_length, self.attn_hidden_dim)
            else:
                attn_output = attn_output * gate

        attn_output = attn_output.to(self.o_proj_dtype)
        attn_output = self.o_proj(attn_output)
        return attn_output, None, None

    def _setup(self, cfg: Optional[dict] = None):
        if isinstance(cfg, dict):
            cfg = ConfigDict(cfg)
        self.num_key_value_heads = int(self.config.num_key_value_heads)
        if not hasattr(self, "num_heads"):
            self.num_heads = int(self.config.num_attention_heads)
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.attn_hidden_dim = self.num_heads * self.head_dim
        self.gating = bool(getattr(self, "gating", getattr(self.config, "gating", True)))
        self.gate_per_head = getattr(self, "gate_per_head", getattr(self.config, "gating", True) == "per-head")
        self.o_proj_dtype, _ = _module_floating_dtype_device(self.o_proj)
        if self.gating:
            self.gate_compute_cast = xhnn.Cast(torch.float32)
            self.gate_output_cast = xhnn.Cast(self.o_proj_dtype)

        self.enable_rope = bool(_cfg_get(cfg, "enable_rope", True))
        if self.enable_rope:
            self.rope = xhnn.Rope()
        self.slice_1 = xhnn.Slice([0], [self.head_dim // 2], [3], [1])
        self.slice_2 = xhnn.Slice([self.head_dim // 2], [sys.maxsize], [3], [1])
        self.k_repeat_interleave = xhnn.RepeatInterleave()
        self.v_repeat_interleave = xhnn.RepeatInterleave()
        attention_max_length = self.sliding_window if getattr(self, "is_sliding", False) else -1
        attention_max_length = int(attention_max_length) if attention_max_length is not None else -1
        self.masked_add = MaskedAdd()
        self.softmax = SoftmaxPlus(dim=-1)
        self.masked_softmax = MaskedSoftmax(dim=-1, attention_max_length=attention_max_length)
        flash_attention_cfg = _cfg_get(cfg, "flash_attention", None)
        self.use_flash_attention = bool(
            flash_attention_cfg and _cfg_get(flash_attention_cfg, "enable", False)
        )
        if self.use_flash_attention:
            q_bits = int(_cfg_get(flash_attention_cfg, "q_bits", 8))
            k_bits = int(_cfg_get(flash_attention_cfg, "k_bits", 8))
            v_bits = int(_cfg_get(flash_attention_cfg, "v_bits", 8))
            s_bits = int(_cfg_get(flash_attention_cfg, "s_bits", q_bits))
            p_bits = int(_cfg_get(flash_attention_cfg, "p_bits", s_bits))
            bits = {
                "q_bits": q_bits,
                "k_bits": k_bits,
                "v_bits": v_bits,
                "s_bits": s_bits,
                "p_bits": p_bits,
            }
            invalid = {name: value for name, value in bits.items() if value not in (8, 16)}
            if invalid:
                raise ValueError(f"Laguna flash_attention bits must be 8 or 16, got {invalid}")
            self.flash_attn = FlashAttention(
                self.num_heads,
                scale=1.0 / math.sqrt(self.head_dim),
                num_kv_heads=self.num_key_value_heads,
                is_causal=True,
                sliding_window=attention_max_length if attention_max_length > 0 else None,
                **bits,
            )
        self.use_cache = bool(_cfg_get(cfg, "use_cache", True))
        if self.use_cache:
            cache_axis = _cfg_get(_cfg_get(cfg, "kv_cache", {}), "cache_axis", 2)
            self.k_cache = LLMCacheV2(axis=cache_axis, attention_max_length=attention_max_length)
            self.v_cache = LLMCacheV2(axis=cache_axis, attention_max_length=attention_max_length)
        else:
            self.k_cache = None
            self.v_cache = None
        q_proj_dtype, q_proj_device = _module_floating_dtype_device(self.q_proj)
        self.register_buffer(
            "kv_scale",
            torch.tensor(
                1.0 / math.sqrt(self.head_dim),
                dtype=q_proj_dtype,
                device=q_proj_device,
            ),
            persistent=True,
        )
        return self


class _LagunaSparseMoeBlock(DynamicModule):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        flat_hidden_states = hidden_states.reshape(-1, hidden_dim)
        shared_expert_output = self.shared_expert(flat_hidden_states)
        router_logits = self.router_compute_cast(self.router(flat_hidden_states))
        if self.router_logit_softcapping > 0.0:
            router_logits = torch.tanh(router_logits / self.router_logit_softcapping) * self.router_logit_softcapping
        routing_scores = torch.sigmoid(router_logits)
        scores_for_selection = routing_scores + self.e_score_correction_bias
        _, selected_experts = self.router_topk(scores_for_selection)

        expert_output = self.moeblock(
            hidden_states,
            routing_scores.to(dtype=hidden_states.dtype).reshape(batch_size, sequence_length, -1),
            selected_experts=selected_experts.reshape(batch_size, sequence_length, self.top_k),
        )
        if self.routed_scaling_factor != 1.0:
            expert_output = expert_output * self.routed_scaling_factor

        shared_expert_output = shared_expert_output.reshape(batch_size, sequence_length, hidden_dim)
        return expert_output + shared_expert_output

    def _setup(self, cfg: Optional[dict] = None):
        experts = self.experts
        gate_up_proj = experts.gate_up_proj
        down_proj = experts.down_proj
        intermediate_dim = int(getattr(experts, "intermediate_dim", gate_up_proj.shape[1] // 2))
        gate_weight = gate_up_proj.data[:, :intermediate_dim, :].contiguous()
        up_weight = gate_up_proj.data[:, intermediate_dim:, :].contiguous()
        down_weight = down_proj.data.contiguous()

        shared_config = getattr(self.shared_expert, "config", None)
        self.top_k = int(getattr(self.gate, "top_k", getattr(shared_config, "num_experts_per_tok", 1)))
        self.norm_topk_prob = bool(getattr(self.gate, "norm_topk_prob", getattr(shared_config, "norm_topk_prob", True)))
        self.routed_scaling_factor = float(
            getattr(self, "routed_scaling_factor", getattr(shared_config, "moe_routed_scaling_factor", 1.0))
        )
        self.router_logit_softcapping = float(getattr(self.gate, "router_logit_softcapping", 0.0) or 0.0)
        self.router = nn.Linear(
            self.gate.weight.shape[1],
            self.gate.weight.shape[0],
            bias=False,
            device=self.gate.weight.device,
            dtype=self.gate.weight.dtype,
        )
        self.router.weight = self.gate.weight
        self.e_score_correction_bias = nn.Parameter(
            deepcopy(self.gate.e_score_correction_bias.data).float(),
            requires_grad=False,
        )
        self.router_compute_cast = xhnn.Cast(torch.float32)
        self.router_topk = xhnn.TopK(self.top_k, axis=-1)
        self.moeblock = MoeBlock(
            _activation_name(getattr(experts, "act_fn", None), shared_config),
            self.top_k,
            normalize_routing_weights=self.norm_topk_prob,
            topk_outside=True,
        )
        self.moeblock.expert_gate_proj_weight = nn.Parameter(gate_weight)
        self.moeblock.expert_gate_proj_bias = None
        self.moeblock.expert_up_proj_weight = nn.Parameter(up_weight)
        self.moeblock.expert_up_proj_bias = None
        self.moeblock.expert_down_proj_weight = nn.Parameter(down_weight)
        self.moeblock.expert_down_proj_bias = None

        gate_up_quant_weight = getattr(experts, "gate_up_proj_quant_weight", None)
        if torch.is_tensor(gate_up_quant_weight):
            _register_or_replace_buffer(
                self.moeblock,
                "expert_gate_proj_quant_weight",
                gate_up_quant_weight[:, :intermediate_dim, :].contiguous(),
            )
            _register_or_replace_buffer(
                self.moeblock,
                "expert_up_proj_quant_weight",
                gate_up_quant_weight[:, intermediate_dim:, :].contiguous(),
            )
        down_quant_weight = getattr(experts, "down_proj_quant_weight", None)
        if torch.is_tensor(down_quant_weight):
            _register_or_replace_buffer(self.moeblock, "expert_down_proj_quant_weight", down_quant_weight.contiguous())

        _move_parameter_to_meta(experts, "gate_up_proj")
        _move_parameter_to_meta(experts, "down_proj")
        del self.experts
        del self.gate
        return self


class _LagunaDecoderLayer(DynamicModule):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        past_seq_length: Optional[Tensor] = None,
        current_input_length: Optional[Tensor] = None,
        past_k_cache: Optional[Tensor] = None,
        past_v_cache: Optional[Tensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor]:
        del kwargs
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            past_seq_length=past_seq_length,
            current_input_length=current_input_length,
            past_k_cache=past_k_cache,
            past_v_cache=past_v_cache,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return (hidden_states,)

    def _setup(self, cfg: Optional[dict] = None):
        return self


class _LagunaModel(DynamicModule):
    def _setup(self, cfg: Optional[dict] = None):
        self.only_first_block = bool(_cfg_get(cfg, "only_first_block", False))
        self.max_layers = 1 if self.only_first_block else int(_cfg_get(cfg, "max_layers", -1))
        self.num_logits_to_keep = int(_cfg_get(cfg, "num_logits_to_keep", 0) or 0)
        self.input_sequence_length = int(_cfg_get(cfg, "input_sequence_length", 1))
        self.use_cache = bool(_cfg_get(cfg, "use_cache", True))
        self.llm_gather = xhnn.BatchGather(1)
        self.llm_gather.update_offset_indices(1, self.input_sequence_length)

        def _llm_gather_update_cfg(batch_gather: xhnn.BatchGather, new_cfg=None):
            input_seq_len = int(_cfg_get(new_cfg, "input_sequence_length", self.input_sequence_length))
            batch_size = int(_cfg_get(new_cfg, "batch_size", 1))
            batch_gather.update_offset_indices(batch_size, input_seq_len)

        self.llm_gather._update_cfg = types.MethodType(_llm_gather_update_cfg, self.llm_gather)
        self._setup_rope_cache(cfg)
        return self

    def _setup_rope_cache(self, cfg: Optional[dict] = None):
        target_dtype = self.embed_tokens.weight.dtype
        self._defer_or_setup_rotary_cache(self.rotary_emb, target_dtype)
        self._full_cos_slice = xhnn.DynamicSlice([self.input_sequence_length], [2], [1])
        self._full_sin_slice = xhnn.DynamicSlice([self.input_sequence_length], [2], [1])
        if getattr(self, "swa_rotary_emb", None) is not None:
            self._defer_or_setup_rotary_cache(self.swa_rotary_emb, target_dtype)
            self._sliding_cos_slice = xhnn.DynamicSlice([self.input_sequence_length], [2], [1])
            self._sliding_sin_slice = xhnn.DynamicSlice([self.input_sequence_length], [2], [1])
        else:
            self._sliding_cos_slice = self._full_cos_slice
            self._sliding_sin_slice = self._full_sin_slice

        def _slice_update_cfg(slice_self, new_cfg=None):
            slice_self.valid_length = [int(_cfg_get(new_cfg, "input_sequence_length", self.input_sequence_length))]

        for slice_module in (
            self._full_cos_slice,
            self._full_sin_slice,
            self._sliding_cos_slice,
            self._sliding_sin_slice,
        ):
            slice_module._update_cfg = types.MethodType(_slice_update_cfg, slice_module)

    @staticmethod
    def _defer_or_setup_rotary_cache(rotary_emb: nn.Module, target_dtype: torch.dtype):
        if hasattr(rotary_emb, "_set_dtype"):
            rotary_emb._set_dtype(target_dtype)
            return

        def _setup_after_callback(rotary_module=rotary_emb, dtype=target_dtype):
            rotary_module._set_dtype(dtype)

        rotary_emb.setup_after_callback = _setup_after_callback

    def _position_embeddings_by_type(self, past_seq_length: Optional[Tensor]):
        full = (
            self._full_cos_slice(self.rotary_emb.cos_cached, past_seq_length),
            self._full_sin_slice(self.rotary_emb.sin_cached, past_seq_length),
        )
        if getattr(self, "swa_rotary_emb", None) is None:
            sliding = full
        else:
            sliding = (
                self._sliding_cos_slice(self.swa_rotary_emb.cos_cached, past_seq_length),
                self._sliding_sin_slice(self.swa_rotary_emb.sin_cached, past_seq_length),
            )
        return {"full_attention": full, "sliding_attention": sliding}

    def forward(
        self,
        inputs_embeds: torch.FloatTensor,
        past_seq_length: Optional[Tensor] = None,
        current_input_length: Optional[Tensor] = None,
        sliding_attention_mask: Optional[Tensor] = None,
        past_key_cache: Optional[list[Tensor]] = None,
        past_value_cache: Optional[list[Tensor]] = None,
    ):
        hidden_states = inputs_embeds
        position_embeddings_by_type = self._position_embeddings_by_type(past_seq_length)
        layer_types = getattr(self.config, "layer_types", ["full_attention"] * len(self.layers))

        for idx, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            if self.max_layers > 0 and idx >= self.max_layers:
                break
            layer_type = layer_types[idx]
            attention_mask = sliding_attention_mask if layer_type == "sliding_attention" else None
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                past_seq_length=past_seq_length,
                current_input_length=current_input_length,
                past_k_cache=past_key_cache[idx] if past_key_cache is not None else None,
                past_v_cache=past_value_cache[idx] if past_value_cache is not None else None,
                position_embeddings=position_embeddings_by_type[layer_type],
            )
            hidden_states = layer_outputs[0]

        if self.num_logits_to_keep != 0:
            hidden_states = self.llm_gather(hidden_states, current_input_length - 1)
        hidden_states = self.norm(hidden_states)
        return hidden_states


class _LagunaForCausalLM(DynamicModule):
    def _setup(self, cfg: Optional[dict] = None):
        return self

    def forward(
        self,
        inputs_embeds: Tensor,
        past_seq_length: Optional[Tensor] = None,
        current_input_length: Optional[Tensor] = None,
        sliding_attention_mask: Optional[Tensor] = None,
        past_key_cache: Optional[list[Tensor]] = None,
        past_value_cache: Optional[list[Tensor]] = None,
    ) -> Tensor:
        hidden_states = self.model(
            inputs_embeds=inputs_embeds,
            past_seq_length=past_seq_length,
            current_input_length=current_input_length,
            sliding_attention_mask=sliding_attention_mask,
            past_key_cache=past_key_cache,
            past_value_cache=past_value_cache,
        )
        return self.lm_head(hidden_states)


_WRAPPERS_BY_REMOTE_NAME = {
    "LagunaRMSNorm": _LagunaRMSNorm,
    "LagunaRotaryEmbedding": _LagunaRotaryEmbedding,
    "LagunaAttention": _LagunaAttention,
    "LagunaSparseMoeBlock": _LagunaSparseMoeBlock,
    "LagunaDecoderLayer": _LagunaDecoderLayer,
    "LagunaModel": _LagunaModel,
    "LagunaForCausalLM": _LagunaForCausalLM,
}


def register_wrap_modules(hf_model: nn.Module | None = None):
    if hf_model is None:
        return None

    remote_classes = _find_remote_classes(hf_model)
    missing = [name for name in _WRAPPERS_BY_REMOTE_NAME if name not in remote_classes]
    if missing:
        raise RuntimeError(f"Laguna remote model is missing expected classes: {missing}")

    for remote_name, dm_class in _WRAPPERS_BY_REMOTE_NAME.items():
        _register_or_replace_traceable({remote_classes[remote_name]: remote_name})(dm_class)

    from xhquant.nn.builder import FX_LEAF_MODULES
    from xhquant.quantization.xh2a.builder import register_none_quanted_module

    FX_LEAF_MODULES._module_dict["LagunaRotaryEmbedding"] = True
    register_none_quanted_module(remote_classes["LagunaRotaryEmbedding"])
    return None
