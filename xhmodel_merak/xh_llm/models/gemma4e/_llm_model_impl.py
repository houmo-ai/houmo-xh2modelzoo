import types
from typing import Optional

import torch
import torch.nn as nn
import xhquant.nn as xhnn
from torch import Tensor
from transformers.models.gemma4.modeling_gemma4 import (
    Gemma4RMSNorm,
    Gemma4TextAttention,
    Gemma4TextDecoderLayer,
    Gemma4TextModel,
    Gemma4TextRotaryEmbedding,
)
from xhquant.nn import LLMCacheV2, MaskedAdd, RMSNorm, SoftmaxPlus
from xhquant.utils.registry import DynamicModule

from ...register import XHLLM_TRACEABLE_MODULES


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


def _rotate_half_with_constant_dim(x: Tensor, half_dim: int) -> Tensor:
    x1 = x[..., :half_dim]
    x2 = x[..., half_dim : half_dim * 2]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb_with_constant_dim(
    x: Tensor,
    cos: Tensor,
    sin: Tensor,
    unsqueeze_dim: int,
    half_dim: int,
) -> Tensor:
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (x * cos) + (_rotate_half_with_constant_dim(x, half_dim) * sin)


@XHLLM_TRACEABLE_MODULES.register_module({Gemma4RMSNorm: "Gemma4RMSNorm"})
class _Gemma4RMSNorm(DynamicModule):
    def _setup(self, cfg=None):
        if getattr(self, "with_scale", True) and hasattr(self, "weight"):
            hidden_size = self.weight.shape[0]
            device = self.weight.device
        elif hasattr(self, "_head_dim_hint"):
            hidden_size = self._head_dim_hint
            device = None
        elif hasattr(self, "dim"):
            hidden_size = self.dim
            device = None
        else:
            if cfg is not None:
                # cfg is a BaseAttrDict (addict.Dict) whose __getattr__
                # never raises AttributeError for missing keys; use
                # dict-style .get() instead of getattr().
                hidden_size = cfg.get("hidden_size", 1)
            else:
                hidden_size = 1
            device = None
        self.norm = RMSNorm(hidden_size, self.eps)
        if device is not None:
            self.norm = self.norm.to(device)
        if getattr(self, "with_scale", True) and hasattr(self, "weight"):
            self.norm.weight = nn.Parameter(self.weight.detach().clone())
        else:
            self.norm.weight = nn.Parameter(
                torch.ones(hidden_size, dtype=torch.float32, device=device),
                requires_grad=False,
            )
        return self

    def forward(self, hidden_states: torch.Tensor):
        return self.norm(hidden_states)


@XHLLM_TRACEABLE_MODULES.register_module({Gemma4TextRotaryEmbedding: "Gemma4TextRotaryEmbedding"})
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


@XHLLM_TRACEABLE_MODULES.register_module({Gemma4TextAttention: "Gemma4TextAttention"})
class _Gemma4TextAttention(DynamicModule):
    def _setup(self, cfg=None):
        self.use_cache = bool(_cfg_get(cfg, "use_cache", True))
        self.rotary_half_dim = self.head_dim // 2
        self.num_attention_heads = self.q_proj.weight.shape[0] // self.head_dim
        self.num_key_value_heads = self.k_proj.weight.shape[0] // self.head_dim
        for norm_name in ("q_norm", "k_norm", "v_norm"):
            norm = getattr(self, norm_name, None)
            if norm is not None and not getattr(norm, "with_scale", True):
                norm._head_dim_hint = self.head_dim
        self.rope = xhnn.Rope()
        self.k_repeat_interleave = xhnn.RepeatInterleave()
        self.v_repeat_interleave = xhnn.RepeatInterleave()
        self.qk_matmul = xhnn.MatMul()
        self.pv_matmul = xhnn.MatMul()
        self.attn_compute_cast = xhnn.Cast(torch.float16).to(dtype=torch.float16)
        self.attn_output_cast = xhnn.Cast(self.o_proj.weight.dtype).to(dtype=self.o_proj.weight.dtype)
        attention_max_length = getattr(self, "sliding_window", None)
        attention_max_length = int(attention_max_length) if attention_max_length is not None else -1
        self.masked_add = MaskedAdd()
        self.masked_add_2 = MaskedAdd()
        self.softmax = SoftmaxPlus(dim=-1)
        if self.use_cache:
            cache_axis = cfg.kv_cache.cache_axis
            self.k_cache = LLMCacheV2(axis=cache_axis, attention_max_length=attention_max_length)
            self.v_cache = LLMCacheV2(axis=cache_axis, attention_max_length=attention_max_length)
        else:
            self.k_cache = None
            self.v_cache = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor,
        attention_mask: torch.Tensor | None,
        past_seq_length: Optional[Tensor] = None,
        current_input_length: Optional[Tensor] = None,
        past_k_cache: Optional[Tensor] = None,
        past_v_cache: Optional[Tensor] = None,
        shared_kv: Optional[dict[int, tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size = hidden_states.shape[0]
        seq_length = hidden_states.shape[1]
        cos, sin = position_embeddings

        query_states = self.q_proj(hidden_states).view(batch_size, seq_length, -1, self.head_dim)
        query_states = self.q_norm(query_states)
        query_states = self.attn_compute_cast(query_states)
        query_states = self.attn_compute_cast(self.rope(query_states, cos, sin)).transpose(1, 2)

        if self.is_kv_shared_layer:
            assert shared_kv is not None
            key_states, value_states = shared_kv[self.kv_shared_layer_index]
        else:
            key_states = self.k_proj(hidden_states).view(batch_size, seq_length, -1, self.head_dim)
            # When attention_k_eq_v=True for global (non-sliding) layers, v_proj is None and K=V
            value_states = (
                self.v_proj(hidden_states).view(batch_size, seq_length, -1, self.head_dim)
                if self.v_proj is not None
                else key_states
            )

            key_states = self.k_norm(key_states)
            key_states = self.attn_compute_cast(key_states)
            key_states = self.attn_compute_cast(self.rope(key_states, cos, sin)).transpose(1, 2)
            value_states = self.attn_compute_cast(self.v_norm(value_states)).transpose(1, 2)

            if self.use_cache and past_k_cache is not None and past_v_cache is not None:
                key_states = self.k_cache(
                    key_states,
                    past_seq_length,
                    current_input_length,
                    past_k_cache,
                )
                value_states = self.v_cache(
                    value_states,
                    past_seq_length,
                    current_input_length,
                    past_v_cache,
                )

        if self.store_full_length_kv:
            assert shared_kv is not None
            shared_kv[self.layer_idx] = (key_states, value_states)

        key_states = self.k_repeat_interleave(key_states.transpose(2, 3), self.num_key_value_groups, 1)
        value_states = self.v_repeat_interleave(value_states, self.num_key_value_groups, 1)

        attn_weights = self.qk_matmul(query_states, key_states)
        if attention_mask is not None:
            attn_weights = self.masked_add(attn_weights, attention_mask)
            attn_weights = self.masked_add_2(attn_weights, attention_mask)
        attn_weights = self.softmax(attn_weights).to(query_states.dtype)

        attn_output = self.pv_matmul(attn_weights, value_states).transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
        attn_output = self.attn_output_cast(attn_output)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


@XHLLM_TRACEABLE_MODULES.register_module({Gemma4TextDecoderLayer: "Gemma4TextDecoderLayer"})
class _Gemma4TextDecoderLayer(DynamicModule):
    def _setup(self, cfg=None):
        return None

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
        shared_kv: Optional[dict[int, tuple[torch.Tensor, torch.Tensor]]] = None,
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
            shared_kv=shared_kv,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        if self.enable_moe_block:
            hidden_states_1 = self.post_feedforward_layernorm_1(hidden_states)

            hidden_states_flat = residual.reshape(-1, residual.shape[-1])
            _, top_k_weights, top_k_index = self.router(hidden_states_flat)
            hidden_states_2 = self.pre_feedforward_layernorm_2(hidden_states_flat)
            hidden_states_2 = self.experts(hidden_states_2, top_k_index, top_k_weights)
            hidden_states_2 = hidden_states_2.reshape(residual.shape[0], residual.shape[1], residual.shape[2])
            hidden_states_2 = self.post_feedforward_layernorm_2(hidden_states_2)
            hidden_states = hidden_states_1 + hidden_states_2

        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        if self.hidden_size_per_layer_input:
            residual = hidden_states
            hidden_states = self.per_layer_input_gate(hidden_states)
            hidden_states = self.act_fn(hidden_states)
            hidden_states = hidden_states * per_layer_input
            hidden_states = self.per_layer_projection(hidden_states)
            hidden_states = self.post_per_layer_input_norm(hidden_states)
            hidden_states = residual + hidden_states

        hidden_states *= self.layer_scalar
        return hidden_states


@XHLLM_TRACEABLE_MODULES.register_module({Gemma4TextModel: "Gemma4TextModel"})
class _Gemma4TextModel(DynamicModule):
    def _setup(self, cfg=None):
        self.only_first_block = bool(_cfg_get(cfg, "only_first_block", False))
        self.num_logits_to_keep = int(_cfg_get(cfg, "num_logits_to_keep", 1))
        self.input_sequence_length = int(_cfg_get(cfg, "input_sequence_length", 1))
        self.image_token_id = int(_cfg_get(cfg, "image_token_id", -1))
        self.audio_token_id = int(_cfg_get(cfg, "audio_token_id", -1))
        self.video_token_id = int(_cfg_get(cfg, "video_token_id", -1))

        self.llm_gather = xhnn.BatchGather(1)
        self.llm_gather.update_offset_indices(1, self.input_sequence_length)

        def _update_gather_cfg(batch_gather: xhnn.BatchGather, new_cfg=None):
            input_sequence_length = int(_cfg_get(new_cfg, "input_sequence_length", self.input_sequence_length))
            batch_size = int(_cfg_get(new_cfg, "batch_size", 1))
            batch_gather.update_offset_indices(batch_size, input_sequence_length)

        self.llm_gather._update_cfg = types.MethodType(_update_gather_cfg, self.llm_gather)
        if hasattr(self.rotary_emb, "set_target_dtype"):
            self.rotary_emb.set_target_dtype(self.embed_tokens.weight.dtype)
        self._setup_rope_cache(cfg)
        return self

    def _setup_rope_cache(self, cfg=None):
        # Pre-compute cos/sin per layer_type and register as buffers with shape
        # (1, max_seq_len, 1, head_dim) so a DynamicSlice on dim=1 can pick the
        # current chunk indexed by past_seq_length. This removes the need for an
        # explicit position_ids graph input (parity with gemma4_moe).
        max_seq_len = int(
            _cfg_get(
                cfg,
                "context_max_length",
                getattr(self.rotary_emb, "max_seq_len_cached", getattr(self.rotary_emb, "original_max_seq_len", 2048)),
            )
        )
        target_dtype = self.embed_tokens.weight.dtype
        for layer_type in set(self.config.layer_types):
            inv_freq = getattr(self.rotary_emb, f"{layer_type}_inv_freq", None)
            attention_scaling = getattr(self.rotary_emb, f"{layer_type}_attention_scaling", None)
            if inv_freq is None or attention_scaling is None:
                continue
            cos, sin = _compute_gemma4_rotary_cache(inv_freq, attention_scaling, max_seq_len)
            # (max_seq_len, head_dim) -> (1, max_seq_len, 1, head_dim)
            cos_buf = cos.unsqueeze(0).unsqueeze(2).to(dtype=target_dtype).contiguous()
            sin_buf = sin.unsqueeze(0).unsqueeze(2).to(dtype=target_dtype).contiguous()
            self.register_buffer(f"_{layer_type}_cos_cache", cos_buf, persistent=False)
            self.register_buffer(f"_{layer_type}_sin_cache", sin_buf, persistent=False)
            cos_slice = xhnn.DynamicSlice([self.input_sequence_length], [1], [1])
            sin_slice = xhnn.DynamicSlice([self.input_sequence_length], [1], [1])

            def _slice_update_cfg(slice_self, new_cfg=None):
                slice_self.valid_length = [
                    int(_cfg_get(new_cfg, "input_sequence_length", self.input_sequence_length))
                ]

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
        return per_layer_inputs.permute(0, 2, 1, 3).contiguous()

    def _project_per_layer_inputs(self, inputs_embeds: Tensor, per_layer_inputs: Optional[Tensor] = None) -> Tensor:
        batch_size = inputs_embeds.shape[0]
        seq_length = inputs_embeds.shape[1]
        per_layer_projection = self.per_layer_model_projection(inputs_embeds) * self.per_layer_model_projection_scale
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
        return (per_layer_projection + per_layer_inputs) * self.per_layer_input_scale

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        per_layer_inputs: Optional[Tensor] = None,
        past_seq_length: Optional[Tensor] = None,
        current_input_length: Optional[Tensor] = None,
        local_attention_mask: Optional[Tensor] = None,
        global_attention_mask: Optional[Tensor] = None,
        past_key_cache: Optional[list[Tensor]] = None,
        past_value_cache: Optional[list[Tensor]] = None,
    ):
        if inputs_embeds is None:
            raise ValueError("Gemma4 text graph requires inputs_embeds.")
        if past_seq_length is None:
            raise ValueError("Gemma4 text graph requires past_seq_length.")
        if self.hidden_size_per_layer_input:
            if per_layer_inputs is None:
                if input_ids is None:
                    raise ValueError("Gemma4 text graph requires explicit per_layer_inputs.")
                input_ids = input_ids.to(torch.long)
                per_layer_inputs = self._project_per_layer_inputs(inputs_embeds, self._get_per_layer_inputs(input_ids))
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
        shared_kv: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        cache_idx = 0
        for layer_idx, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            if decoder_layer.self_attn.is_kv_shared_layer:
                layer_past_key_cache = None
                layer_past_value_cache = None
            else:
                layer_past_key_cache = past_key_cache[cache_idx] if past_key_cache is not None else None
                layer_past_value_cache = past_value_cache[cache_idx] if past_value_cache is not None else None
                cache_idx += 1

            layer_type = self.config.layer_types[layer_idx]
            layer_position_embeddings = position_embeddings_by_type[layer_type]
            layer_attention_mask = global_attention_mask if layer_type == "full_attention" else local_attention_mask
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
                shared_kv=shared_kv,
            )
            if self.only_first_block:
                break

        hidden_states = self.norm(hidden_states)
        if self.num_logits_to_keep == 0:
            return hidden_states
        return self.llm_gather(hidden_states, current_input_length - 1)


def register_wrap_modules(hf_model=None):
    return None


register_wrap_cls = register_wrap_modules
