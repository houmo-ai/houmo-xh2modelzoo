from __future__ import annotations

import types
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.gemma4.modeling_gemma4 import (
    Gemma4ForConditionalGeneration,
    Gemma4RMSNorm,
    Gemma4TextAttention,
    Gemma4TextDecoderLayer,
    Gemma4TextModel,
)

from xhquant import nn as xhnn
from xhquant.api import ConfigDict
from xhquant.nn import LLMCache, MaskedAdd, RMSNorm, SoftmaxPlus
from xhquant.utils.registry import DynamicModule

from ...register import XHLLM_TRACEABLE_MODULES


@XHLLM_TRACEABLE_MODULES.register_module({Gemma4RMSNorm: "Gemma4RMSNorm"})
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


@XHLLM_TRACEABLE_MODULES.register_module({Gemma4TextAttention: "Gemma4TextAttention"})
class _Gemma4TextAttention(DynamicModule):
    def _setup(self, cfg: ConfigDict | dict[str, Any]):
        if isinstance(cfg, dict):
            cfg = ConfigDict(cfg)
        self.use_cache = bool(cfg.use_cache)
        self.head_dim = int(getattr(self, "head_dim"))
        self.num_heads = int(getattr(self.config, "num_attention_heads"))
        self.num_key_value_heads = self.k_proj.out_features // self.head_dim
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        # Gemma4 uses q_norm/k_norm before attention, so scaling=1.0 (no extra scaling needed)
        self.kv_scale = getattr(self, "scaling", 1.0)
        # Provide head_dim hint to scale-free child norms (v_norm) before they get _setup'd
        for norm_name in ("q_norm", "k_norm", "v_norm"):
            norm = getattr(self, norm_name, None)
            if norm is not None and not getattr(norm, "with_scale", True):
                norm._head_dim_hint = self.head_dim
        # Per-layer sliding window: sliding layers get attention_max_length=sliding_window,
        # full attention layers get -1 (no truncation). The compiler uses this marker to
        # recognize sliding-window KV caches and optimize accordingly.
        attention_max_length = self.sliding_window if getattr(self, "sliding_window", None) is not None else -1
        self.masked_add = MaskedAdd()
        self.softmax = SoftmaxPlus(dim=-1)
        if self.use_cache:
            self.k_cache = LLMCache(axis=cfg.kv_cache.cache_axis, attention_max_length=attention_max_length)
            self.v_cache = LLMCache(axis=cfg.kv_cache.cache_axis, attention_max_length=attention_max_length)
        else:
            self.k_cache = None
            self.v_cache = None
        self.rope = xhnn.Rope()
        return self

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_seq_length: Optional[Tensor] = None,
        current_input_length: Optional[Tensor] = None,
        past_k_cache: Optional[Tensor] = None,
        past_v_cache: Optional[Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim)
        key_states_linear = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        value_states_linear = (
            self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim)
            if self.v_proj is not None
            else key_states_linear
        )

        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states_linear)
        value_states = self.v_norm(value_states_linear)

        if position_embeddings is not None:
            cos, sin = position_embeddings
            query_states = self.rope(query_states, cos, sin)
            key_states = self.rope(key_states, cos, sin)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        if self.use_cache:
            key_states = self.k_cache(key_states, past_seq_length, current_input_length, past_k_cache)
            value_states = self.v_cache(value_states, past_seq_length, current_input_length, past_v_cache)
        key_states = key_states.transpose(2, 3)
        key_states = torch.repeat_interleave(key_states, self.num_key_value_groups, dim=1)
        value_states = torch.repeat_interleave(value_states, self.num_key_value_groups, dim=1)
        if self.kv_scale == 1.0:
            attn_weights = torch.matmul(query_states, key_states)
        else:
            attn_weights = torch.matmul(query_states, key_states) * self.kv_scale
        if attention_mask is not None:
            attn_weights = self.masked_add(attn_weights, attention_mask)
        attn_weights = self.softmax(attn_weights).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, self.num_heads * self.head_dim)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


@XHLLM_TRACEABLE_MODULES.register_module({Gemma4TextDecoderLayer: "Gemma4TextDecoderLayer"})
class _Gemma4TextDecoderLayer(DynamicModule):
    def _setup(self, cfg: Optional[Dict] = None):
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
        **kwargs,
    ) -> torch.Tensor:
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
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        if getattr(self, "hidden_size_per_layer_input", 0) and per_layer_input is not None:
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


@XHLLM_TRACEABLE_MODULES.register_module({Gemma4TextModel: "Gemma4TextModel"})
class _Gemma4TextModel(DynamicModule):
    def _setup(self, cfg: Optional[Dict]):
        self.use_cache = cfg.get("use_cache", True)
        # Use config.max_position_embeddings for the full cos/sin buffer length,
        # not config.context_max_length. This ensures DynamicSlice can properly
        # truncate from the full position range (parity with gemma4_moe).
        context_len = (
            cfg.get("context_max_length", 2048) if hasattr(cfg, "get") else getattr(cfg, "context_max_length", 2048)
        )
        # Read max_seq_len directly from the text model config, same as how
        # rotary_emb.max_seq_len_cached is initialized (= config.max_position_embeddings).
        max_seq_len = getattr(self.config, "max_position_embeddings", None)
        if max_seq_len is None:
            max_seq_len = getattr(self.rotary_emb, "max_seq_len_cached", context_len)
        input_seq_len = (
            cfg.get("input_sequence_length", context_len)
            if hasattr(cfg, "get")
            else getattr(cfg, "input_sequence_length", context_len)
        )
        self._precompute_rope_cache(max_seq_len)
        self._setup_rope_slices(input_seq_len)
        return self

    def _precompute_rope_cache(self, max_seq_len: int):
        """Pre-compute cos/sin tables. Shape: [1, 1, max_seq_len, head_dim] for DynamicSlice dim=2 + xhnn.Rope."""
        rope = self.rotary_emb
        for layer_type in set(self.config.layer_types):
            inv_freq = getattr(rope, f"{layer_type}_inv_freq")
            attn_scaling = getattr(rope, f"{layer_type}_attention_scaling")
            positions = torch.arange(max_seq_len, device=inv_freq.device, dtype=torch.float32)
            freqs = torch.outer(positions, inv_freq.float())
            emb = torch.cat((freqs, freqs), dim=-1)
            cos_table = (emb.cos() * attn_scaling).to(inv_freq.dtype)
            sin_table = (emb.sin() * attn_scaling).to(inv_freq.dtype)
            # [max_seq_len, head_dim] → [1, 1, max_seq_len, head_dim]
            cos_table = cos_table.unsqueeze(0).unsqueeze(0)
            sin_table = sin_table.unsqueeze(0).unsqueeze(0)
            self.register_buffer(f"_{layer_type}_cos", cos_table, persistent=False)
            self.register_buffer(f"_{layer_type}_sin", sin_table, persistent=False)

    def _setup_rope_slices(self, input_seq_len: int):
        """Create DynamicSlice modules for cos/sin lookup by past_seq_length."""
        for layer_type in set(self.config.layer_types):
            cos_slice = xhnn.DynamicSlice([input_seq_len], [2], [1])
            sin_slice = xhnn.DynamicSlice([input_seq_len], [2], [1])

            def _slice_update_cfg(self, cfg=None):
                self.valid_length = [cfg.input_sequence_length]

            cos_slice._update_cfg = types.MethodType(_slice_update_cfg, cos_slice)
            sin_slice._update_cfg = types.MethodType(_slice_update_cfg, sin_slice)
            setattr(self, f"_{layer_type}_cos_slice", cos_slice)
            setattr(self, f"_{layer_type}_sin_slice", sin_slice)

    def _get_rope(self, past_seq_length: Tensor, layer_type: str, dtype: torch.dtype):
        cos_table = getattr(self, f"_{layer_type}_cos")
        sin_table = getattr(self, f"_{layer_type}_sin")
        cos_slice = getattr(self, f"_{layer_type}_cos_slice")
        sin_slice = getattr(self, f"_{layer_type}_sin_slice")
        cos = cos_slice(cos_table, past_seq_length).to(dtype)  # [1, 1, seq_len, head_dim]
        sin = sin_slice(sin_table, past_seq_length).to(dtype)
        return cos, sin

    def forward(
        self,
        inputs_embeds: Tensor,
        past_seq_length: Tensor,
        current_input_length: Tensor,
        full_attention_mask: Tensor,
        sliding_attention_mask: Tensor,
        past_key_cache: List[Tensor],
        past_value_cache: List[Tensor],
    ) -> BaseModelOutputWithPast:
        hidden_states = inputs_embeds
        position_embeddings = {}
        for layer_type in set(self.config.layer_types):
            position_embeddings[layer_type] = self._get_rope(past_seq_length, layer_type, hidden_states.dtype)

        for idx, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            layer_type = self.config.layer_types[idx]
            layer_mask = sliding_attention_mask if layer_type == "sliding_attention" else full_attention_mask
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings[layer_type],
                attention_mask=layer_mask,
                past_seq_length=past_seq_length,
                current_input_length=current_input_length,
                past_k_cache=past_key_cache[idx] if past_key_cache is not None else None,
                past_v_cache=past_value_cache[idx] if past_value_cache is not None else None,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(last_hidden_state=hidden_states)


@XHLLM_TRACEABLE_MODULES.register_module({Gemma4ForConditionalGeneration: "Gemma4ForConditionalGeneration"})
class _Gemma4ForConditionalGeneration(DynamicModule):
    def _setup(self, cfg: Optional[Dict] = None):
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
        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)
        final_logit_softcapping = getattr(self.config.text_config, "final_logit_softcapping", None)
        if final_logit_softcapping is not None:
            logits = logits / final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * final_logit_softcapping
        return logits


def register_wrap_modules():
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding

    from xhquant.nn.builder import FX_LEAF_MODULES
    from xhquant.nn.modules.normalized_modules import FloorDiv
    from xhquant.quantization.xh2a.builder import register_none_quanted_module

    # Register Gemma4TextRotaryEmbedding as FX leaf so the tracer doesn't trace into it
    # (its forward uses getattr with f-strings and x.device which are not proxy-safe)
    FX_LEAF_MODULES._module_dict["Gemma4TextRotaryEmbedding"] = True

    # Register modules as none-quanted so the quantization pipeline passes them through
    register_none_quanted_module(Gemma4TextRotaryEmbedding)
    register_none_quanted_module(FloorDiv)
    return None
