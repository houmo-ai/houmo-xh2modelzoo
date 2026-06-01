import math
import sys
import types
from copy import deepcopy
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from xhquant import nn as xhnn
from xhquant.api import ConfigDict
from xhquant.nn import BfpFlashAttention, LLMCacheV2, MaskedSoftmax, RMSNorm
from xhquant.utils.registry import DynamicModule

from ..builder import XHLLM_TRACEABLE_MODULES
from .modeling_paddleocr_vl import (
    Ernie4_5Attention,
    Ernie4_5DecoderLayer,
    Ernie4_5Model,
    Ernie4_5RMSNorm,
    Ernie4_5RotaryEmbedding,
    PaddleOCRVLForConditionalGeneration,
)


@XHLLM_TRACEABLE_MODULES.register_module({Ernie4_5RMSNorm: "Ernie4_5RMSNorm"})
class _Ernie4_5RMSNorm(DynamicModule):
    """FX-friendly Ernie4_5RMSNorm"""

    def _setup(self, cfg: Optional[Dict] = None):
        hidden_size = self.weight.shape[0]
        self.norm = RMSNorm(hidden_size, self.variance_epsilon)
        self.norm.weight = nn.Parameter(deepcopy(self.weight.data))

    def forward(self, hidden_states):
        return self.norm(hidden_states)


@XHLLM_TRACEABLE_MODULES.register_module(
    {Ernie4_5RotaryEmbedding: "Ernie4_5RotaryEmbedding"}
)
class _Ernie4_5RotaryEmbedding(DynamicModule):
    """FX-friendly Ernie4_5RotaryEmbedding"""

    def _setup(self, cfg: ConfigDict):
        # No-op: rotary is only used to precompute cache in _Ernie4_5Model._setup
        pass


@XHLLM_TRACEABLE_MODULES.register_module(
    {
        Ernie4_5Model: "Ernie4_5Model",
    }
)
class _Ernie4_5Model(DynamicModule):
    """FX-friendly Ernie4_5Model"""

    def _setup(self, cfg: ConfigDict):
        self.cfg = cfg
        self.batch_size = cfg.get("batch_size", 1)
        self.num_logits_to_keep = cfg.get("num_logits_to_keep", 0)
        assert self.num_logits_to_keep in [0, 1]

        input_seq_len = cfg.input_sequence_length
        self.slice = xhnn.Slice([0], [input_seq_len], [1], [1])

        self.llm_gather = xhnn.BatchGather(1)
        self.llm_gather.update_offset_indices(self.batch_size, input_seq_len)

        def _llm_gather_update_cfg(self: xhnn.BatchGather, cfg: Optional[Dict] = None):
            input_seq_len = cfg.input_sequence_length
            batch_size = cfg.get("batch_size", 1)
            self.update_offset_indices(batch_size, input_seq_len)

        self.llm_gather._update_cfg = types.MethodType(
            _llm_gather_update_cfg, self.llm_gather
        )

        def _slice_update_cfg(self, cfg: Optional[Dict] = None):
            input_seq_len = cfg.input_sequence_length
            self.ends = [input_seq_len]

        self.slice._update_cfg = types.MethodType(_slice_update_cfg, self.slice)
        self.use_cache = cfg.use_cache

        max_seq_len = cfg.get("max_sequence_length", None)
        if max_seq_len is None:
            max_seq_len = cfg.get("max_position_embeddings", None)
        if max_seq_len is None:
            max_seq_len = cfg.input_sequence_length
        position_ids = torch.arange(max_seq_len, dtype=torch.long).view(1, 1, -1)
        position_ids = position_ids.expand(3, self.batch_size, -1)
        dummy = torch.zeros(1, 1)
        cos, sin = self.rotary_emb(dummy, position_ids)
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

    def forward(
        self,
        input_embeds: Optional[Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_seq_length: Optional[Tensor] = None,
        current_input_length: Optional[Tensor] = None,
        past_key_cache: Optional[List[Tensor]] = None,
        past_value_cache: Optional[List[Tensor]] = None,
        time_position_ids: Optional[Tensor] = None,
        hight_position_ids: Optional[Tensor] = None,
        width_position_ids: Optional[Tensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        if input_embeds is None:
            raise ValueError("input_embeds must be provided")

        if past_seq_length is not None:
            past_seq_length = past_seq_length.to(torch.int32)
        if current_input_length is not None:
            current_input_length = current_input_length.to(torch.int32)
        if time_position_ids is not None:
            time_position_ids = time_position_ids.to(torch.float16)
        if hight_position_ids is not None:
            hight_position_ids = hight_position_ids.to(torch.float16)
        if width_position_ids is not None:
            width_position_ids = width_position_ids.to(torch.float16)

        fixed_bsz = self.batch_size
        time_position_ids = time_position_ids.reshape(1, -1).expand(fixed_bsz, -1)
        hight_position_ids = hight_position_ids.reshape(1, -1).expand(fixed_bsz, -1)
        width_position_ids = width_position_ids.reshape(1, -1).expand(fixed_bsz, -1)
        position_ids = torch.stack(
            [time_position_ids, hight_position_ids, width_position_ids], dim=0
        )

        position_ids = position_ids.to(torch.float16)
        inv_freq = self.rotary_emb.inv_freq
        inv_freq_expanded = (
            inv_freq[None, None, :, None].float().expand(3, fixed_bsz, -1, 1)
        )
        position_ids_expanded = position_ids[:, :, None, :].to(inv_freq_expanded.dtype)
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(2, 3)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = torch.cos(emb) * self.rotary_emb.attention_scaling
        sin = torch.sin(emb) * self.rotary_emb.attention_scaling
        position_embeddings = (cos, sin)

        hidden_states = input_embeds
        for layer_idx, decoder_layer in enumerate(self.layers):
            if self.use_cache:
                _past_k_cache = (
                    past_key_cache[layer_idx] if past_key_cache is not None else None
                )
                _past_v_cache = (
                    past_value_cache[layer_idx]
                    if past_value_cache is not None
                    else None
                )
            else:
                _past_k_cache = None
                _past_v_cache = None

            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_seq_length=past_seq_length,
                current_input_length=current_input_length,
                past_k_cache=_past_k_cache,
                past_v_cache=_past_v_cache,
            )

        if self.num_logits_to_keep == 0:
            hidden_states = self.slice(hidden_states)
        else:
            hidden_states = self.llm_gather(hidden_states, current_input_length - 1)

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(last_hidden_state=hidden_states)


@XHLLM_TRACEABLE_MODULES.register_module({Ernie4_5Attention: "Ernie4_5Attention"})
class _Ernie4_5Attention(DynamicModule):
    """FX-friendly Ernie4_5Attention"""

    def _setup(self, cfg: ConfigDict):
        if not hasattr(self, "num_key_value_heads"):
            self.num_key_value_heads = self.config.num_key_value_heads
        if not hasattr(self, "num_heads"):
            self.num_heads = self.config.num_attention_heads
        if not hasattr(self, "num_key_value_groups"):
            self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.half_head_dim = self.head_dim // 2
        self.batch_size = cfg.get("batch_size", 1)
        self.input_seq_len = cfg.input_sequence_length

        self.masked_softmax = MaskedSoftmax(dim=-1)
        self.attn_hidden_dim = self.head_dim * self.num_heads

        bfp_flash_attention_cfg = cfg.get("bfp_flash_attention", None)
        self.use_bfp_flash_attention = False
        if bfp_flash_attention_cfg is not None:
            self.use_bfp_flash_attention = bfp_flash_attention_cfg.enable
            self.sefp_manbit = bfp_flash_attention_cfg.sefp_manbit
            self.out_fp_manbit = bfp_flash_attention_cfg.out_fp_manbit
            self.out_fp_expbit = bfp_flash_attention_cfg.out_fp_expbit
            if self.use_bfp_flash_attention:
                self.bfp_attn = BfpFlashAttention(
                    self.attn_hidden_dim,
                    self.num_heads,
                    True,
                    self.sefp_manbit,
                    self.out_fp_expbit,
                    self.out_fp_manbit,
                )

        self.use_cache = cfg.use_cache
        if self.use_cache:
            cache_axis = cfg.kv_cache.cache_axis
            self.k_cache = LLMCacheV2(axis=cache_axis)
            self.v_cache = LLMCacheV2(axis=cache_axis)
        else:
            self.k_cache = None
            self.v_cache = None

        _kv_scale = 1 / math.sqrt(self.head_dim)
        self.register_buffer(
            "kv_scale", torch.tensor(_kv_scale, dtype=torch.float32), persistent=False
        )

        self.use_mrope = bool(
            getattr(self, "rope_scaling", None) is not None
            and "mrope_section" in self.rope_scaling
        )
        if self.use_mrope:
            mrope_section = self.rope_scaling["mrope_section"]
            if isinstance(mrope_section, int):
                mrope_section = [mrope_section]
            mrope_section = list(mrope_section) * 2
            mask0 = []
            mask1 = []
            mask2 = []
            for i, seg_len in enumerate(mrope_section):
                seg_len = int(seg_len)
                if i % 3 == 0:
                    mask0.extend([1.0] * seg_len)
                    mask1.extend([0.0] * seg_len)
                    mask2.extend([0.0] * seg_len)
                elif i % 3 == 1:
                    mask0.extend([0.0] * seg_len)
                    mask1.extend([1.0] * seg_len)
                    mask2.extend([0.0] * seg_len)
                else:
                    mask0.extend([0.0] * seg_len)
                    mask1.extend([0.0] * seg_len)
                    mask2.extend([1.0] * seg_len)
            if len(mask0) == 0:
                mask0 = [1.0] * self.head_dim
                mask1 = [0.0] * self.head_dim
                mask2 = [0.0] * self.head_dim
            if len(mask0) > self.head_dim:
                mask0 = mask0[: self.head_dim]
                mask1 = mask1[: self.head_dim]
                mask2 = mask2[: self.head_dim]
            if len(mask0) < self.head_dim:
                pad_len = self.head_dim - len(mask0)
                mask0.extend([mask0[-1]] * pad_len)
                mask1.extend([mask1[-1]] * pad_len)
                mask2.extend([mask2[-1]] * pad_len)
            time_mask = torch.tensor(mask0, dtype=torch.float16).view(1, 1, -1)
            hight_mask = torch.tensor(mask1, dtype=torch.float16).view(1, 1, -1)
            width_mask = torch.tensor(mask2, dtype=torch.float16).view(1, 1, -1)
            self.register_buffer("time_mask", time_mask, persistent=False)
            self.register_buffer("hight_mask", hight_mask, persistent=False)
            self.register_buffer("width_mask", width_mask, persistent=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_seq_length: Optional[Tensor] = None,
        current_input_length: Optional[Tensor] = None,
        past_k_cache: Optional[Tensor] = None,
        past_v_cache: Optional[Tensor] = None,
    ) -> Tuple[torch.FloatTensor, Optional[torch.FloatTensor]]:
        """Forward pass for attention layer"""
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = (
            query_states.view(bsz, q_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )
        key_states = (
            key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )
        value_states = (
            value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

        if position_embeddings is not None:
            cos, sin = position_embeddings
            if self.use_mrope:
                query_states, key_states = self._apply_multimodal_rotary_pos_emb(
                    query_states, key_states, cos, sin
                )
            else:
                cos = cos[0].unsqueeze(1)
                sin = sin[0].unsqueeze(1)
                query_states = (query_states * cos) + (
                    self._rotate_half(query_states) * sin
                )
                key_states = (key_states * cos) + (self._rotate_half(key_states) * sin)

        if self.use_cache and current_input_length is not None:
            key_states = self.k_cache(
                key_states, past_seq_length, current_input_length, past_k_cache
            )
            value_states = self.v_cache(
                value_states, past_seq_length, current_input_length, past_v_cache
            )

        if self.use_bfp_flash_attention:
            attn_output = self.bfp_attn(query_states, key_states, value_states)
        else:
            query_states = query_states * self.kv_scale
            key_states = key_states.transpose(2, 3)

            key_states = torch.repeat_interleave(
                key_states, self.num_key_value_groups, dim=1
            ).contiguous()
            attn_weights = torch.matmul(query_states.float(), key_states.float()).to(
                query_states.dtype
            )

            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
                attn_weights = nn.functional.softmax(
                    attn_weights, dim=-1, dtype=torch.float32
                ).to(query_states.dtype)
            else:
                attn_weights = self.masked_softmax(attn_weights, past_seq_length)
            attn_weights = nn.functional.dropout(
                attn_weights, p=0.0, training=self.training
            )

            value_states = torch.repeat_interleave(
                value_states, self.num_key_value_groups, dim=1
            ).contiguous()
            value_states = value_states.to(attn_weights.dtype)
            attn_output = torch.matmul(attn_weights, value_states)

            attn_output = attn_output.transpose(1, 2)
            attn_output = attn_output.reshape(
                bsz, q_len, self.num_heads * self.head_dim
            )

        attn_output = attn_output.to(self.o_proj.weight.dtype)
        attn_output = self.o_proj(attn_output)
        return attn_output, None

    def _rotate_half(self, x):
        x1 = x[..., : self.half_head_dim]
        x2 = x[..., self.half_head_dim :]
        return torch.cat((-x2, x1), dim=-1)

    def _apply_multimodal_rotary_pos_emb(self, q, k, cos, sin, unsqueeze_dim=1):
        cos0, cos1, cos2 = cos[0], cos[1], cos[2]
        sin0, sin1, sin2 = sin[0], sin[1], sin[2]
        cos = cos0 * self.time_mask + cos1 * self.hight_mask + cos2 * self.width_mask
        sin = sin0 * self.time_mask + sin1 * self.hight_mask + sin2 * self.width_mask
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)

        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)
        return q_embed, k_embed


@XHLLM_TRACEABLE_MODULES.register_module({Ernie4_5DecoderLayer: "Ernie4_5DecoderLayer"})
class _Ernie4_5DecoderLayer(DynamicModule):
    """FX-friendly Ernie4_5DecoderLayer"""

    def _setup(self, cfg: ConfigDict):
        pass

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_seq_length: Optional[Tensor] = None,
        current_input_length: Optional[Tensor] = None,
        past_k_cache: Optional[Tensor] = None,
        past_v_cache: Optional[Tensor] = None,
    ) -> torch.FloatTensor:
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
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


@XHLLM_TRACEABLE_MODULES.register_module(
    {
        PaddleOCRVLForConditionalGeneration: "PaddleOCRVLForConditionalGeneration",
    }
)
class _PaddleOCRVLForConditionalGeneration(DynamicModule):
    """FX-friendly PaddleOCRVLForConditionalGeneration"""

    def _setup(self, cfg: ConfigDict):
        self.cfg = cfg

    def forward(
        self,
        inputs_embeds: Optional[Tensor] = None,
        time_position_ids: Optional[Tensor] = None,
        hight_position_ids: Optional[Tensor] = None,
        width_position_ids: Optional[Tensor] = None,
        past_seq_length: Optional[Tensor] = None,
        current_input_length: Optional[Tensor] = None,
        past_key_cache: Optional[List[Tensor]] = None,
        past_value_cache: Optional[List[Tensor]] = None,
    ) -> torch.Tensor:
        outputs = self.model(
            input_embeds=inputs_embeds,
            past_seq_length=past_seq_length,
            current_input_length=current_input_length,
            past_key_cache=past_key_cache,
            past_value_cache=past_value_cache,
            time_position_ids=time_position_ids,
            hight_position_ids=hight_position_ids,
            width_position_ids=width_position_ids,
        )

        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)
        return logits


def register_wrap_cls(hf_model):
    return
