import math
import sys
import types
from copy import deepcopy
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.qwen2_vl.modeling_qwen2_vl import (
    Qwen2VLAttention,
    Qwen2VLDecoderLayer,
    Qwen2VLForConditionalGeneration,
    Qwen2VLModel,
    Qwen2RMSNorm,
    Qwen2VLRotaryEmbedding,
    Qwen2VLTextModel,
)
from xhquant import nn as xhnn
from xhquant.api import ConfigDict
from xhquant.nn import LLMCacheV2, MaskedSoftmax, RMSNorm
from xhquant.utils.registry import DynamicModule

from ...register import XHLLM_TRACEABLE_MODULES


@XHLLM_TRACEABLE_MODULES.register_module({Qwen2RMSNorm: "Qwen2RMSNorm"})
class _Qwen2RMSNorm(DynamicModule):
    def forward(self, hidden_states):
        return self.norm(hidden_states)

    def _setup(self, cfg: Optional[Dict] = None):
        hidden_size = self.weight.shape[0]
        self.norm = RMSNorm(hidden_size, self.variance_epsilon)
        self.norm.weight = nn.Parameter(deepcopy(self.weight.data))
        return self


@XHLLM_TRACEABLE_MODULES.register_module({Qwen2VLRotaryEmbedding: "Qwen2VLRotaryEmbedding"})
class _Qwen2VLRotaryEmbedding(DynamicModule):
    def _setup(self, cfg: ConfigDict):
        assert "dynamic" not in self.rope_type, f"{self.rope_type} is not supported in dynamic mode"
        max_sequence_length = cfg.get("max_pe_length", cfg.max_sequence_length)
        self._setup_cos_sin_cache(seq_len=max_sequence_length, dtype=self.inv_freq.dtype)
        if hasattr(self, "setup_after_callback"):
            self.setup_after_callback()

    def _setup_cos_sin_cache(self, seq_len, dtype):
        device = self.inv_freq.device
        position_ids = torch.arange(0, seq_len, dtype=torch.long, device=device).view(1, 1, -1).expand(3, 1, -1)
        x = torch.empty((1, 1, 1), dtype=dtype, device=device)
        cos, sin = self.forward(x, position_ids)
        self.register_buffer("sin_cached", sin.to(dtype=dtype), persistent=False)
        self.register_buffer("cos_cached", cos.to(dtype=dtype), persistent=False)

    def _set_dtype(self, dtype: torch.dtype) -> None:
        self.inv_freq = self.inv_freq.to(dtype)
        self._setup_cos_sin_cache(seq_len=self.max_seq_len_cached, dtype=self.inv_freq.dtype)


@XHLLM_TRACEABLE_MODULES.register_module({Qwen2VLAttention: "Qwen2VLAttention"})
class _Qwen2VLAttention(DynamicModule):
    def rotate_half(self, x: Tensor):
        x1 = self.slice_1(x)
        x2 = self.slice_2(x)
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(self, q: Tensor, k: Tensor, cos: Tensor, sin: Tensor, unsqueeze_dim: int = 1):
        if self.enable_rope:
            q_embed = self.rope(q, cos, sin)
            k_embed = self.rope(k, cos, sin)
        else:
            q_embed = (q * cos) + (self.rotate_half(q) * sin)
            k_embed = (k * cos) + (self.rotate_half(k) * sin)
        return q_embed, k_embed

    def _setup(self, cfg: ConfigDict):
        self.enable_rope = cfg.get("enable_rope", True)
        if self.enable_rope:
            self.rope = xhnn.Rope()

        self.num_heads = getattr(self, "num_heads", self.config.num_attention_heads)
        self.num_key_value_heads = getattr(self, "num_key_value_heads", self.config.num_key_value_heads)
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.slice_1 = xhnn.Slice([0], [self.head_dim // 2], [3], [1])
        self.slice_2 = xhnn.Slice([self.head_dim // 2], [sys.maxsize], [3], [1])
        self.masked_softmax = MaskedSoftmax(dim=-1)
        use_cache = cfg.use_cache
        self.use_cache = use_cache
        if use_cache:
            cache_axis = cfg.kv_cache.cache_axis
            self.k_cache = LLMCacheV2(axis=cache_axis)
            self.v_cache = LLMCacheV2(axis=cache_axis)
        else:
            self.k_cache = None
            self.v_cache = None
        self.register_buffer("kv_scale", torch.tensor(1 / math.sqrt(self.head_dim), dtype=torch.float16), persistent=False)

    def graph_forward(
        self,
        hidden_states: torch.Tensor,
        past_seq_length: Optional[Tensor] = None,
        current_input_length: Optional[Tensor] = None,
        position_ids: torch.Tensor = None,
        past_k_cache: Optional[Tensor] = None,
        past_v_cache: Optional[Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = self.apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if self.use_cache:
            key_states = self.k_cache(key_states, past_seq_length, current_input_length, past_k_cache)
            value_states = self.v_cache(value_states, past_seq_length, current_input_length, past_v_cache)

        query_states = query_states * self.kv_scale
        key_states = key_states.transpose(2, 3)
        key_states = torch.repeat_interleave(key_states, self.num_key_value_groups, dim=1)
        attn_weights = torch.matmul(query_states, key_states)
        attn_weights = self.masked_softmax(attn_weights, past_seq_length)

        value_states = torch.repeat_interleave(value_states, self.num_key_value_groups, dim=1)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        return attn_output, None, None


@XHLLM_TRACEABLE_MODULES.register_module({Qwen2VLDecoderLayer: "Qwen2VLDecoderLayer"})
class _Qwen2VLDecoderLayer(DynamicModule):
    def _setup(self, cfg: ConfigDict):
        pass

    def graph_forward(
        self,
        hidden_states: torch.Tensor,
        past_seq_length: Optional[Tensor] = None,
        current_input_length: Optional[Tensor] = None,
        past_k_cache: Optional[Tensor] = None,
        past_v_cache: Optional[Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _, _ = self.self_attn(
            hidden_states=hidden_states,
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
        hidden_states = residual + hidden_states
        return (hidden_states,)


@XHLLM_TRACEABLE_MODULES.register_module({Qwen2VLTextModel: "Qwen2VLTextModel"})
class _Qwen2VLTextModel(DynamicModule):
    def _setup_cos_sin_embeding(self):
        cos = self.rotary_emb.cos_cached.squeeze(1)
        sin = self.rotary_emb.sin_cached.squeeze(1)
        t, num_embeddings, embedding_dim = cos.shape
        assert t == 3
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

    def _setup(self, cfg: ConfigDict):
        self.batch_size = cfg.get("batch_size", 1)
        self.only_first_block = cfg.get("only_first_block", False)
        self.num_logits_to_keep = cfg.num_logits_to_keep
        assert self.num_logits_to_keep in [0, 1]

        input_seq_len = cfg.input_sequence_length
        self.slice = xhnn.Slice([0], [input_seq_len], [1], [1])
        self.llm_gather = xhnn.BatchGather(1)
        self.llm_gather.update_offset_indices(self.batch_size, input_seq_len)

        def _llm_gather_update_cfg(self: xhnn.BatchGather, cfg: Optional[Dict] = None):
            input_seq_len = cfg.input_sequence_length
            batch_size = cfg.get("batch_size", 1)
            self.update_offset_indices(batch_size, input_seq_len)

        self.llm_gather._update_cfg = types.MethodType(_llm_gather_update_cfg, self.llm_gather)

        def _slice_update_cfg(self, cfg: Optional[Dict] = None):
            input_seq_len = cfg.input_sequence_length
            self.ends = [input_seq_len]

        self.slice._update_cfg = types.MethodType(_slice_update_cfg, self.slice)
        self.use_cache = cfg.use_cache

        if not hasattr(self.rotary_emb, "cos_cached"):
            self.rotary_emb.setup_after_callback = self._setup_cos_sin_embeding
        else:
            self._setup_cos_sin_embeding()

    def graph_forward(
        self,
        inputs_embeds: Optional[Tensor] = None,
        time_position_ids: Optional[Tensor] = None,
        height_position_ids: Optional[Tensor] = None,
        width_position_ids: Optional[Tensor] = None,
        past_seq_length: Optional[Tensor] = None,
        current_input_length: Optional[Tensor] = None,
        past_key_cache: Optional[List[Tensor]] = None,
        past_value_cache: Optional[List[Tensor]] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        hidden_states = inputs_embeds

        time_cos = self.cos_cached[0][time_position_ids]
        time_sin = self.sin_cached[0][time_position_ids]
        height_cos = self.cos_cached[1][height_position_ids]
        height_sin = self.sin_cached[1][height_position_ids]
        width_cos = self.cos_cached[2][width_position_ids]
        width_sin = self.sin_cached[2][width_position_ids]
        cos = torch.stack([time_cos, height_cos, width_cos], dim=0)
        sin = torch.stack([time_sin, height_sin, width_sin], dim=0)

        mrope_section = self.layers[0].self_attn.rope_scaling["mrope_section"]
        mrope_section = mrope_section * 2
        cos_splits = cos.split(mrope_section, dim=-1)
        sin_splits = sin.split(mrope_section, dim=-1)

        cos_list = []
        sin_list = []
        for i in range(len(mrope_section)):
            cos_list.append(cos_splits[i][i % 3])
            sin_list.append(sin_splits[i][i % 3])

        cos = torch.cat(cos_list, dim=-1).unsqueeze(0).unsqueeze(0)
        sin = torch.cat(sin_list, dim=-1).unsqueeze(0).unsqueeze(0)
        position_embeddings = (cos, sin)

        for idx, decoder_layer in enumerate(self.layers):
            if self.use_cache:
                _past_k_cache = past_key_cache[idx]
                _past_v_cache = past_value_cache[idx]
            else:
                _past_k_cache = None
                _past_v_cache = None

            layer_outputs = decoder_layer(
                hidden_states,
                past_seq_length=past_seq_length,
                current_input_length=current_input_length,
                past_k_cache=_past_k_cache,
                past_v_cache=_past_v_cache,
                position_embeddings=position_embeddings,
            )
            hidden_states = layer_outputs[0]
            if self.only_first_block:
                break

        if self.num_logits_to_keep == 0:
            hidden_states = self.slice(hidden_states)
        else:
            hidden_states = self.llm_gather(hidden_states, current_input_length - 1)
        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(last_hidden_state=hidden_states)


@XHLLM_TRACEABLE_MODULES.register_module({Qwen2VLModel: "Qwen2VLModel"})
class _Qwen2VLModel(DynamicModule):
    def _setup(self, cfg: ConfigDict):
        self.cfg = cfg
        if "visual" in self._modules:
            del self._modules["visual"]

    def graph_forward(
        self,
        inputs_embeds: Optional[Tensor] = None,
        time_position_ids: Optional[Tensor] = None,
        height_position_ids: Optional[Tensor] = None,
        width_position_ids: Optional[Tensor] = None,
        past_seq_length: Optional[Tensor] = None,
        current_input_length: Optional[Tensor] = None,
        past_key_cache: Optional[List[Tensor]] = None,
        past_value_cache: Optional[List[Tensor]] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        return self.language_model(
            inputs_embeds=inputs_embeds,
            time_position_ids=time_position_ids,
            height_position_ids=height_position_ids,
            width_position_ids=width_position_ids,
            past_seq_length=past_seq_length,
            current_input_length=current_input_length,
            past_key_cache=past_key_cache,
            past_value_cache=past_value_cache,
        )


@XHLLM_TRACEABLE_MODULES.register_module({Qwen2VLForConditionalGeneration: "Qwen2VLForConditionalGeneration"})
class _Qwen2VLForConditionalGeneration(DynamicModule):
    def _setup(self, cfg: ConfigDict):
        self.cfg = cfg
        if "visual" in self._modules:
            del self._modules["visual"]
        if hasattr(self, "model") and "visual" in self.model._modules:
            del self.model._modules["visual"]

    def graph_forward(
        self,
        inputs_embeds: Optional[Tensor] = None,
        time_position_ids: Optional[Tensor] = None,
        height_position_ids: Optional[Tensor] = None,
        width_position_ids: Optional[Tensor] = None,
        past_seq_length: Optional[Tensor] = None,
        current_input_length: Optional[Tensor] = None,
        past_key_cache: Optional[List[Tensor]] = None,
        past_value_cache: Optional[List[Tensor]] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            time_position_ids=time_position_ids,
            height_position_ids=height_position_ids,
            width_position_ids=width_position_ids,
            past_seq_length=past_seq_length,
            current_input_length=current_input_length,
            past_key_cache=past_key_cache,
            past_value_cache=past_value_cache,
        )
        hidden_states = outputs[0]
        return self.lm_head(hidden_states)


def register_wrap_cls(hf_model):
    pass
