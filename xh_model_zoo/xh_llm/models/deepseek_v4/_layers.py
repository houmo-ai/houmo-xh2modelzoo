# ================================================================== #
#  File: _layers.py                                                   #
#  Description:                                                       #
#    DeepSeek-V4 DecoderLayer, Model, ForCausalLM wrappers.          #
#                                                                     #
#    DecoderLayer: orchestrates HC + attention + MoE + HC            #
#    Model: embed + layers + hc_head + norm                           #
#    ForCausalLM: model + lm_head                                     #
#                                                                     #
#    Key difference: hidden_states flow as [B, S, hc_mult, D]        #
#    throughout all layers (hc_mult=4 parallel streams).              #
#                                                                     #
#    Compressed KV cache (CSA/HCA) is packed into past_key_caches    #
#    for decode: past_key_caches = [kv_0..kv_N, ckv_0..ckv_N].      #
#    FX trace uses self._is_decode (concrete bool) to resolve        #
#    prefill/decode branches at trace time.                           #
# ================================================================== #

from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor

from xhquant import nn as xhnn
from xhquant.utils.registry import DynamicModule

from ..builder import XHLLM_TRACEABLE_MODULES


try:
    from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
        DeepseekV4DecoderLayer,
        DeepseekV4ForCausalLM,
        DeepseekV4Model,
    )
except ImportError:
    DeepseekV4DecoderLayer = DeepseekV4ForCausalLM = DeepseekV4Model = None


# ================================================================== #
#  DecoderLayer                                                       #
# ================================================================== #


if DeepseekV4DecoderLayer is not None:
    _layer_registry = {DeepseekV4DecoderLayer: "DeepseekV4DecoderLayer"}
else:
    _layer_registry = {}


@XHLLM_TRACEABLE_MODULES.register_module(_layer_registry)
class _DeepseekV4DecoderLayer(DynamicModule):
    """V4 decoder block with Hyper-Connections.

    hidden_states: [B, S, hc_mult, D] throughout.
    Each HC produces (post, comb, collapsed) to mix the parallel streams.
    """

    def forward(
        self,
        hidden_states: Tensor,
        cos: Optional[Tensor] = None,
        sin: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_seq_length: Optional[Tensor] = None,
        current_input_length: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        input_ids: Optional[Tensor] = None,
        past_k_cache: Optional[Tensor] = None,
        compressed_kv_cache: Optional[Tensor] = None,
        **kwargs,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        dtype = hidden_states.dtype

        # -- Attention site --
        post, comb, collapsed = self.attn_hc(hidden_states)
        attn_output, compressed_kv = self.self_attn(
            self.input_layernorm(collapsed),
            cos=cos,
            sin=sin,
            position_ids=position_ids,
            past_seq_length=past_seq_length,
            current_input_length=current_input_length,
            attention_mask=attention_mask,
            input_ids=input_ids,
            past_k_cache=past_k_cache,
            compressed_kv_cache=compressed_kv_cache,
        )
        hidden_states = post.to(dtype).unsqueeze(-1) * attn_output.unsqueeze(-2) + torch.matmul(
            comb.to(dtype).transpose(-1, -2), hidden_states
        )

        # -- FFN site --
        post, comb, collapsed = self.ffn_hc(hidden_states)
        mlp_output = self.mlp(
            self.post_attention_layernorm(collapsed),
            input_ids=input_ids,
        )
        hidden_states = post.to(dtype).unsqueeze(-1) * mlp_output.unsqueeze(-2) + torch.matmul(
            comb.to(dtype).transpose(-1, -2), hidden_states
        )
        return hidden_states, compressed_kv

    def _setup(self, cfg: Optional[Dict] = None):
        return self


# ================================================================== #
#  Model                                                              #
# ================================================================== #


if DeepseekV4Model is not None:
    _model_registry = {DeepseekV4Model: "DeepseekV4Model"}
else:
    _model_registry = {}


@XHLLM_TRACEABLE_MODULES.register_module(_model_registry)
class _DeepseekV4Model(DynamicModule):
    """V4 model body: embed -> layers -> hc_head -> norm.

    Expands embeddings to [B, S, hc_mult, D] at entry,
    collapses back to [B, S, D] at exit via hc_head.

    For decode, compressed_kv_cache is packed into past_key_caches:
        past_key_caches = [kv_0, ..., kv_N-1, ckv_0, ..., ckv_N-1]
    Extracted via index offset `num_layers + idx`.
    """

    def _setup(self, cfg):
        self.batch_size = cfg.get("batch_size", 1)
        self.num_logits_to_keep = cfg.num_logits_to_keep
        self.use_cache = cfg.use_cache
        self.hc_mult = self.config.hc_mult
        self._num_layers = self.config.num_hidden_layers
        self._is_decode = cfg.input_sequence_length == 1

        input_seq_len = cfg.input_sequence_length
        self.slice = xhnn.Slice([0], [input_seq_len], [1], [1])
        self.llm_gather = xhnn.BatchGather(1)
        self.llm_gather.update_offset_indices(self.batch_size, input_seq_len)

        # -- RoPE cos/sin caches are on self.rotary_emb --
        # -- Build causal mask for sliding window attention --
        sliding_window = self.config.sliding_window
        context_length = cfg.kv_cache.context_length
        self._build_attention_mask(sliding_window, input_seq_len, context_length)

    def _build_attention_mask(self, sliding_window, input_seq_len, context_length):
        """Pre-build sliding window causal mask.

        Prefill (input_seq_len > 1): standard causal mask.
        Decode  (input_seq_len == 1): no explicit mask — SinksMaskedSoftmax
        handles boundary masking via past_seq_length.
        """
        if input_seq_len == 1:
            # Decode: no explicit mask — SinksMaskedSoftmax handles boundary
            # masking via past_seq_length. 注册空 tensor 避免 AttributeError。
            self.register_buffer(
                "attention_mask_cached",
                torch.zeros(0, dtype=torch.float16),
                persistent=True,
            )
        else:
            mask = torch.full(
                (input_seq_len, context_length),
                float("-inf"),
                dtype=torch.float16,
            )
            for i in range(input_seq_len):
                for j in range(i + 1):
                    if sliding_window and sliding_window > 0 and j < i - sliding_window + 1:
                        continue
                    mask[i, j] = 0.0
            self.register_buffer("attention_mask_cached", mask.unsqueeze(0).unsqueeze(0), persistent=True)

    # NOTE: 死代码，待删除。converter 为 prefill/decode 各自创建独立模型，无需运行时重配。
    # def _update_cfg(self, cfg=None):
    #     if cfg is None:
    #         return
    #     self.batch_size = cfg.get("batch_size", self.batch_size)
    #     input_seq_len = cfg.input_sequence_length
    #     self._is_decode = input_seq_len == 1
    #     self.slice.ends = [input_seq_len]
    #     self.llm_gather.update_offset_indices(self.batch_size, input_seq_len)
    #     context_length = cfg.kv_cache.context_length
    #     self._build_attention_mask(self.config.sliding_window, input_seq_len, context_length)

    def forward(
        self,
        inputs_embeds: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_seq_length: Optional[Tensor] = None,
        current_input_length: Optional[Tensor] = None,
        input_ids: Optional[Tensor] = None,
        past_key_caches: Optional[List[Tensor]] = None,
        **kwargs,
    ):
        # -- Expand to hc_mult streams --
        hidden_states = inputs_embeds.unsqueeze(2).expand(-1, -1, self.hc_mult, -1).contiguous()

        # -- cast_data converts int64→int32, but aten.index needs int64 --
        pid = position_ids.to(torch.int64)
        iid = input_ids.to(torch.int64)

        # -- Compute RoPE cos/sin indexed by position_ids --
        rotary = self.rotary_emb
        main_cos = rotary.main_cos_cached[pid]
        main_sin = rotary.main_sin_cached[pid]
        compress_cos = getattr(rotary, "compress_cos_cached", None)
        if compress_cos is not None:
            compress_cos = compress_cos[pid]
            compress_sin = rotary.compress_sin_cached[pid]

        # -- Run decoder layers --
        # Decode 时 compressed_kv 打包在 past_key_caches[num_layers:]
        # self._is_decode 是 concrete bool, FX trace 时直接解析分支
        #
        # 关键：decode 必须传 attention_mask=None，否则 attention 误走
        # prefill 路径（从头计算 compressor 而非复用 cached compressed_kv），
        # 导致 past_key_caches[5+idx] 从不被访问 → ExpandInputListProp 只展开
        # 5 个 placeholder → input_names:10 vs input_args:15 断言失败。
        _is_dec = self._is_decode
        _mask = None if _is_dec else self.attention_mask_cached

        compressed_kv_outputs: List[Optional[Tensor]] = []
        for idx, decoder_layer in enumerate(self.layers):
            _past_k = past_key_caches[idx] if past_key_caches is not None and self.use_cache else None
            _ckv = None
            if _is_dec and past_key_caches is not None:
                _ckv = past_key_caches[self._num_layers + idx]

            rt = decoder_layer.self_attn.rope_layer_type
            _cos = main_cos if rt == "main" else compress_cos
            _sin = main_sin if rt == "main" else compress_sin
            hidden_states, ckv = decoder_layer(
                hidden_states,
                cos=_cos,
                sin=_sin,
                position_ids=position_ids,
                past_seq_length=past_seq_length,
                current_input_length=current_input_length,
                attention_mask=_mask,
                input_ids=iid,
                past_k_cache=_past_k,
                compressed_kv_cache=_ckv,
            )
            compressed_kv_outputs.append(ckv)

        # -- Collapse HC streams + norm --
        hidden_states = self.norm(self.hc_head(hidden_states))

        if self.num_logits_to_keep != 0:
            hidden_states = self.llm_gather(hidden_states, current_input_length - 1)

        return hidden_states, compressed_kv_outputs


# ================================================================== #
#  ForCausalLM                                                        #
# ================================================================== #


if DeepseekV4ForCausalLM is not None:
    _causal_registry = {DeepseekV4ForCausalLM: "DeepseekV4ForCausalLM"}
else:
    _causal_registry = {}


@XHLLM_TRACEABLE_MODULES.register_module(_causal_registry)
class _DeepseekV4ForCausalLM(DynamicModule):
    """Top-level: model -> lm_head.

    Forward signature: 6 positional args (no compressed_kv_caches param).
    Prefill/decode branching via self._is_decode (concrete bool).

    Prefill: returns tuple (logits, compressed_kv_2, compressed_kv_3, ...)
    Decode:  returns single Tensor logits.
    """

    def _setup(self, cfg):
        self._is_decode = cfg.input_sequence_length == 1

    def forward(
        self,
        inputs_embeds: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_seq_length: Optional[Tensor] = None,
        current_input_length: Optional[Tensor] = None,
        input_ids: Optional[Tensor] = None,
        past_key_caches: Optional[List[Tensor]] = None,
    ):
        hidden_states, compressed_kv_outputs = self.model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            past_seq_length=past_seq_length,
            current_input_length=current_input_length,
            input_ids=input_ids,
            past_key_caches=past_key_caches,
        )
        logits = self.lm_head(hidden_states)

        # -- self._is_decode 是 concrete bool, FX trace 时直接解析 --
        if not self._is_decode:
            outs = [logits] + [ckv for ckv in compressed_kv_outputs if ckv is not None]
            return tuple(outs)
        return logits


# ================================================================== #
#  Module registration helper                                         #
# ================================================================== #


def register_wrap_modules():
    """Import all wrappers to trigger XHLLM_TRACEABLE_MODULES registration."""
    from . import _hc, _model, _moe  # noqa: F401
