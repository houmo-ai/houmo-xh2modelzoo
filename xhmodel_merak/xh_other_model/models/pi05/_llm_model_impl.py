import math
import sys
import types
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.gemma.modeling_gemma import (
    GemmaAttention,
    GemmaDecoderLayer,
    GemmaForCausalLM,
    GemmaModel,
    GemmaRMSNorm,
)

import xhquant.nn as xhnn
from xhquant.api import ConfigDict
from xhquant.nn import LLMCache, MaskedSoftmax, RMSNorm, Rope
from xhquant.utils.registry import DynamicModule

from ...builder import XHLLM_TRACEABLE_MODULES


step_i = 0


@XHLLM_TRACEABLE_MODULES.register_module({GemmaRMSNorm: "GemmaRMSNorm"})
class _GemmaRMSNorm(DynamicModule):
    def forward(self, hidden_states):
        return self.norm(hidden_states)

    def _setup(self, cfg: Optional[Dict] = None):
        hidden_size = self.weight.shape[0]
        self.norm = RMSNorm(hidden_size, self.eps)
        self.norm.weight = nn.Parameter(deepcopy(self.weight.data + 1))
        return self


@XHLLM_TRACEABLE_MODULES.register_module(
    {
        GemmaAttention: "GemmaAttention",
    }
)
class _GemmaAttention(DynamicModule):
    def rotate_half(self, x: Tensor):
        """Rotates half the hidden dims of the input."""
        # x1 = x[..., : x.shape[-1] // 2]
        # x2 = x[..., x.shape[-1] // 2 :]
        # x1 = torch_ops_xh2a_slice(x, [0], [self.head_dim // 2], [3], [1])
        # x2 = torch_ops_xh2a_slice(x, [self.head_dim // 2], [sys.maxsize], [3], [1])
        x1 = self.slice_1(x)
        x2 = self.slice_2(x)
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(self, q: Tensor, k: Tensor, cos: Tensor, sin: Tensor, unsqueeze_dim: int = 1):
        # cos = cos.unsqueeze(unsqueeze_dim)
        # sin = sin.unsqueeze(unsqueeze_dim)
        # cos = self.cos_unsqueeze(cos)
        # sin = self.sin_unsqueeze(sin)
        if self.enable_rope:
            q_embed = self.rope(q, cos, sin)
            k_embed = self.rope(k, cos, sin)
        else:
            q_embed = (q * cos) + (self.rotate_half(q) * sin)
            k_embed = (k * cos) + (self.rotate_half(k) * sin)
        return q_embed, k_embed

    def forward(
        self,
        hidden_states: torch.Tensor,
        # position_ids: torch.Tensor,
        past_seq_length: Optional[Tensor] = None,
        current_input_length: Optional[Tensor] = None,
        past_k_cache: Optional[Tensor] = None,
        past_v_cache: Optional[Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
        bsz, q_len, _ = hidden_states.size()
        # causal_mask = None
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = self.apply_rotary_pos_emb(query_states, key_states, cos, sin, unsqueeze_dim=0)

        if self.use_cache:
            key_states = self.k_cache(key_states, past_seq_length, current_input_length, past_k_cache)
            value_states = self.v_cache(value_states, past_seq_length, current_input_length, past_v_cache)

        if self.use_cache:
            bz, num_heads, seq_len, head_dim = past_k_cache.shape
        else:
            bz, num_heads, seq_len, head_dim = key_states.shape

        if True:
            query_states = query_states * self.kv_scale  # [bsz, self.num_key_value_heads, seq_len, self.head_dim]
            # key_states = key_states * self.key_extra_scale  # [bsz, self.num_key_value_heads, seq_len, self.head_dim]
            key_states = key_states.transpose(2, 3)
            # TODO: HMMatMul broadcast
            # key_states = self.key_unsqueeze(key_states)
            # key_states = self.key_expand(key_states)
            # key_states = key_states.reshape(bz, self.num_heads, self.head_dim, -1)

            key_states = torch.repeat_interleave(
                key_states,
                self.num_key_value_groups,
                dim=1,
            )

            attn_weights = torch.matmul(query_states, key_states)  # [4, 28, 256, 128], [4, 28, 128, 32768]
            # attn_weights = attn_weights / self.key_extra_scale
            # attn_weights = self.key_group_broadcast_matmul(query_states, key_states)
            # attn_weights = torch.matmul(query_states, key_states) / math.sqrt(self.head_dim) #fp16下会出现nan
            # attn_weights: Optional[Tensor] = self.masked_softmax(attn_weights, past_seq_length)
            attn_weights = self.masked_add(attn_weights, attention_mask)
            attn_weights = nn.functional.softmax(attn_weights, dim=-1)

            # TODO: HMMatMul broadcast
            # value_states = self.value_unsqueeze(value_states)
            # value_states = self.value_expand(value_states)
            # value_states = value_states.reshape(bz, self.num_heads, -1, self.head_dim)
            value_states = torch.repeat_interleave(
                value_states,
                self.num_key_value_groups,
                dim=1,
            )
            attn_output = torch.matmul(attn_weights, value_states)  # [4, 28, 256, 32768], [4, 28, 32768, 128]
            # attn_output = self.value_group_broadcast_matmul(attn_weights, value_states)

            # attn_weights = nn.functional.dropout(
            #     attn_weights, p=self.attention_dropout, training=self.training
            # )

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.config.num_attention_heads * self.head_dim)

        attn_output = self.o_proj(attn_output)

        # return attn_output, attn_weights
        return attn_output, attn_weights

    def _setup(self, cfg: Union[ConfigDict, Dict[str, Any]]):
        if isinstance(cfg, dict):
            cfg = ConfigDict(cfg)
        if not hasattr(self, "num_key_value_heads"):
            self.num_key_value_heads = self.config.num_key_value_heads
        if not hasattr(self, "num_heads"):
            self.num_heads = self.config.num_attention_heads  # 28
        if not hasattr(self, "hidden_size"):
            self.hidden_size = self.config.hidden_size  # 3584

        self.enable_rope = cfg.get("enable_rope", True)
        if self.enable_rope:
            self.rope = xhnn.Rope()

        self.slice_1 = xhnn.Slice([0], [self.head_dim // 2], [3], [1])
        self.slice_2 = xhnn.Slice([self.head_dim // 2], [sys.maxsize], [3], [1])

        attention_max_length = -1

        self.masked_softmax = MaskedSoftmax(dim=-1, attention_max_length=attention_max_length)
        self.masked_add = xhnn.MaskedAdd()
        use_cache = cfg.use_cache
        self.use_cache = use_cache
        self.key_extra_scale = 1.0 if "key_extra_scale" not in cfg else cfg.key_extra_scale
        self.query_extra_scale = 1.0 if "query_extra_scale" not in cfg else cfg.query_extra_scale
        # max_sequence_length = cfg.max_sequence_length
        # self.max_sequence_length = max_sequence_length
        # input_seq_len = cfg.input_sequence_length
        self.cos_gather = xhnn.Gather(0)
        self.sin_gather = xhnn.Gather(0)

        self.rope = Rope()

        self.key_unsqueeze = xhnn.Unsqueeze(2)
        self.value_unsqueeze = xhnn.Unsqueeze(2)

        if use_cache:
            cache_axis = cfg.kv_cache.cache_axis
            self.k_cache = LLMCache(
                axis=cache_axis,
                attention_max_length=attention_max_length,
            )
            self.v_cache = LLMCache(
                axis=cache_axis,
                attention_max_length=attention_max_length,
            )
        else:
            self.k_cache = None
            self.v_cache = None
        _kv_scale = 1 / math.sqrt(self.head_dim)
        self.kv_scale = _kv_scale
        # self.register_buffer("kv_scale", torch.tensor(_kv_scale, dtype=torch.float16), persistent=False)
        # self.register_parameter(
        #     "kv_scale", nn.Parameter(torch.tensor([_kv_scale], dtype=torch.float16), requires_grad=False)
        # )
        return self


@XHLLM_TRACEABLE_MODULES.register_module(
    {
        GemmaDecoderLayer: "GemmaDecoderLayer",
    }
)
class _GemmaDecoderLayer(DynamicModule):
    def _setup(self, cfg: Optional[Dict] = None):
        assert cfg is not None
        self.use_clip = cfg.get("use_clip", False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        # position_ids: Optional[torch.LongTensor] = None,
        past_seq_length: Optional[Tensor] = None,
        current_input_length: Optional[Tensor] = None,
        past_k_cache: Optional[Tensor] = None,
        past_v_cache: Optional[Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        clip_value = torch.finfo(torch.float16).max - 1000
        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            # position_ids=position_ids,
            past_seq_length=past_seq_length,
            current_input_length=current_input_length,
            past_k_cache=past_k_cache,
            past_v_cache=past_v_cache,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
        )

        hidden_states = residual + hidden_states
        if self.use_clip:
            hidden_states = torch.clamp(hidden_states, min=-clip_value, max=clip_value)

        residual = hidden_states
        
        hidden_states = self.post_attention_layernorm(hidden_states)
        if self.use_clip:
            hidden_states = torch.clamp(hidden_states, min=-clip_value, max=clip_value)

        hidden_states = self.mlp(hidden_states)
        if self.use_clip:
            hidden_states = torch.clamp(hidden_states, min=-clip_value, max=clip_value)

        hidden_states = residual + hidden_states
        if self.use_clip:
            hidden_states = torch.clamp(hidden_states, min=-clip_value, max=clip_value)

        outputs = (hidden_states,)

        return outputs


@XHLLM_TRACEABLE_MODULES.register_module(
    {
        GemmaModel: "GemmaModel",
    }
)
class _GemmaModel(DynamicModule):
    def _setup_position_cached(self):
        rotary_emb = self.rotary_emb
        max_seq_len = rotary_emb.max_seq_len_cached
        position_ids = torch.arange(0, max_seq_len, dtype=torch.long, device=rotary_emb.inv_freq.device).unsqueeze(
            0
        )  # [1, max_seq_len]
        cos, sin = rotary_emb.forward(rotary_emb.inv_freq, position_ids)  # [1, max_seq_len, feature_dim]
        sin = sin.squeeze(0)
        cos = cos.squeeze(0)

        # TODO:临时处理
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

        self.register_buffer("sin_cached", sin.to(dtype=rotary_emb.inv_freq.dtype), persistent=False)
        self.register_buffer("cos_cached", cos.to(dtype=rotary_emb.inv_freq.dtype), persistent=False)

        # num_embeddings, embedding_dim = cos.shape
        # cos_embeding = nn.Embedding(num_embeddings, embedding_dim)
        # cos_embeding.weight.data = cos
        # self.cos_embeding = cos_embeding

        # sin_embeding = nn.Embedding(num_embeddings, embedding_dim)
        # sin_embeding.weight.data = sin
        # self.sin_embeding = sin_embeding

    def _setup(self, cfg: Optional[Dict]):
        self.only_first_block = cfg.get("only_first_block", False)
        self._setup_position_cached()

        self.num_logits_to_keep = cfg.num_logits_to_keep  # type: ignore # 1,取最后一个token的输出，0,取所有token的输出
        assert self.num_logits_to_keep in [0, 1]

        self.use_cache = cfg.use_cache
        input_seq_len = cfg.input_sequence_length  # type: ignore
        self.sin_slice = xhnn.DynamicSlice([input_seq_len], [2], [1])
        self.cos_slice = xhnn.DynamicSlice([input_seq_len], [2], [1])

        def _sin_cos_slice_update_cfg(self, cfg: Optional[Dict] = None):
            self.valid_length = [cfg.input_sequence_length]

        self.sin_slice._update_cfg = types.MethodType(_sin_cos_slice_update_cfg, self.sin_slice)
        self.cos_slice._update_cfg = types.MethodType(_sin_cos_slice_update_cfg, self.cos_slice)

        self.llm_gather = xhnn.BatchGather(1)
        self.llm_gather.update_offset_indices(1, input_seq_len)

        def _llm_gather_update_cfg(self: xhnn.BatchGather, cfg: Optional[Dict] = None):
            input_seq_len = cfg.input_sequence_length  # type: ignore
            batch_size = cfg.get("batch_size", 1)  # type: ignore
            self.update_offset_indices(batch_size, input_seq_len)

        self.llm_gather._update_cfg = types.MethodType(_llm_gather_update_cfg, self.llm_gather)

    def forward(
        self,
        inputs_embeds: Optional[Tensor] = None,
        past_seq_length: Optional[Tensor] = None,
        current_input_length: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        past_key_cache: Optional[List[Tensor]] = None,
        past_value_cache: Optional[List[Tensor]] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        # embed positions
        hidden_states = inputs_embeds
        cos = self.cos_slice(self.cos_cached, past_seq_length)
        sin = self.sin_slice(self.sin_cached, past_seq_length)
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
                attention_mask=attention_mask,
                past_seq_length=past_seq_length,
                current_input_length=current_input_length,
                past_k_cache=_past_k_cache,
                past_v_cache=_past_v_cache,
                position_embeddings=position_embeddings,
            )

            hidden_states = layer_outputs[0]
            if self.only_first_block:
                break

        # if self.num_logits_to_keep == 0:
        #     # hidden_states = torch_ops_xh2a_slice(hidden_states, [0], [current_input_length], [1], [1])
        #     # hidden_states = self.slice(
        #     #     hidden_states
        #     # )  # 此时返回的结果，含有padding,调用者需要根据current_input_length切片
        #     pass
        # else:
        #     # 取最后一个token的输出
        #     hidden_states = self.llm_gather(hidden_states, current_input_length - 1)
        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
        )


@XHLLM_TRACEABLE_MODULES.register_module(
    {
        GemmaForCausalLM: "GemmaForCausalLM",
    }
)
class _GemmaForCausalLM(DynamicModule):
    def _setup(self, cfg: Optional[Dict] = None):
        pass

    def forward(
        self,
        inputs_embeds: Optional[Tensor] = None,
        past_seq_length: Optional[Tensor] = None,
        current_input_length: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        past_key_cache: Optional[List[Tensor]] = None,
        past_value_cache: Optional[List[Tensor]] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            past_seq_length=past_seq_length,
            current_input_length=current_input_length,
            attention_mask=attention_mask,
            past_key_cache=past_key_cache,
            past_value_cache=past_value_cache,
        )

        hidden_states = outputs[0]
        return hidden_states


def register_wrap_cls(hf_model):
    if hf_model is None:
        return
    registrations = (
        (type(hf_model), _GemmaModel),
        (type(hf_model.layers[0]), _GemmaDecoderLayer),
        (type(hf_model.norm), _GemmaRMSNorm),
    )
    for hf_cls, wrapper_cls in registrations:
        if hf_cls not in XHLLM_TRACEABLE_MODULES:
            XHLLM_TRACEABLE_MODULES.register_module({hf_cls: hf_cls.__name__}, wrapper_cls)
