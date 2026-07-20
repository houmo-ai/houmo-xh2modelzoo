# -*- coding: utf-8 -*-
"""Registry-free text-model cache loop shared by hybrid Qwen MoE models."""

import types
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch import Tensor

from xhquant import nn as xhnn

from .split_conv_cache_utils import (
    _flatten_merged_conv_cache_outputs,
    _flatten_split_conv_cache_outputs,
    _is_nested_split_conv_cache,
    _layers_use_split_conv_cache,
    _looks_like_flat_split_conv_cache,
    _regroup_flat_split_conv_cache,
    _select_linear_attn_conv_cache,
)


class HybridTextModelMixin:
    """Shared text-model cache loop for hybrid Qwen decoder families."""

    def _setup(self, cfg):
        self.batch_size = cfg.get("batch_size", 1)
        self.only_first_block = cfg.get("only_first_block", False)
        self.max_layers = -1
        if self.only_first_block:
            self.max_layers = 1
        else:
            if "max_layers" in cfg and cfg.max_layers is not None:
                self.max_layers = cfg.max_layers
        self.num_logits_to_keep = cfg.num_logits_to_keep
        assert self.num_logits_to_keep in [0, 1]
        self.output_hidden_state_indices = cfg.get("output_hidden_state_indices", None)
        if self.output_hidden_state_indices is not None:
            self._output_hidden_set = set(self.output_hidden_state_indices)
        self.output_post_norm_hidden = cfg.get("output_post_norm_hidden", False)
        input_seq_len = cfg.input_sequence_length
        self.slice = xhnn.Slice([0], [input_seq_len], [1], [1])
        self.llm_gather = xhnn.BatchGather(1)
        self.llm_gather.update_offset_indices(self.batch_size, input_seq_len)

        def _llm_gather_update_cfg(self_g: xhnn.BatchGather, cfg_inner: Optional[Dict] = None):
            input_seq_len_inner = cfg_inner.input_sequence_length
            batch_size_inner = cfg_inner.get("batch_size", 1)
            self_g.update_offset_indices(batch_size_inner, input_seq_len_inner)

        self.llm_gather._update_cfg = types.MethodType(_llm_gather_update_cfg, self.llm_gather)

        def _slice_update_cfg(self_s, cfg_inner: Optional[Dict] = None):
            input_seq_len_inner = cfg_inner.input_sequence_length
            self_s.ends = [input_seq_len_inner]

        self.slice._update_cfg = types.MethodType(_slice_update_cfg, self.slice)
        self.use_cache = cfg.use_cache
        self.split_conv_cache = cfg.get("split_conv_cache", True)
        # Layer type tracking
        self.layer_types = self.config.layer_types
        self.num_full_attention_layers = sum(1 for t in self.layer_types if t == "full_attention")
        self.num_linear_attention_layers = sum(1 for t in self.layer_types if t == "linear_attention")
        # MoE conversion can enter TextModel tracing before child linear-attn
        # modules have independently consumed the full wrap cfg. Make the split
        # cache contract explicit at the TextModel boundary so a flat external
        # q/k/v signature is never threaded into a merged qkv GatedDeltaNet.
        if self.split_conv_cache:
            for idx_layer, decoder_layer in enumerate(self.layers):
                if self.layer_types[idx_layer] != "linear_attention":
                    continue
                linear_attn = getattr(decoder_layer, "linear_attn", None)
                if linear_attn is None:
                    continue
                linear_attn.split_conv_cache = True
                if hasattr(linear_attn, "_setup") and not hasattr(linear_attn, "conv1d_q"):
                    linear_attn._setup(cfg)
        # Mark specific linear_attention layers for alpha scaling
        alpha_scaling_layers = cfg.get("alpha_scaling_layers", [8, 20])
        chunk_inverse_alpha = cfg.get("chunk_inverse_alpha", 0.5)
        for idx_layer, decoder_layer in enumerate(self.layers):
            if self.layer_types[idx_layer] == "linear_attention" and idx_layer in alpha_scaling_layers:
                gdn = decoder_layer.linear_attn
                gdn._alpha_scaling_config = {"alpha": chunk_inverse_alpha}
        self._setup_position_embeddings(cfg)
        self.enable_layer_tag = cfg.get("enable_layer_tag", False)
        if self.enable_layer_tag:
            self.tags = nn.ModuleList(
                [xhnn.XHTag(f"layer_{layer_idx}", "LLM", f"layer_{layer_idx}") for layer_idx in range(len(self.layers))]
            )
        return self

    def _build_position_embeddings(
        self,
        past_seq_length,
        time_position_ids,
        hight_position_ids,
        width_position_ids,
    ):
        del past_seq_length
        return self.rotary_emb(time_position_ids, hight_position_ids, width_position_ids)

    def _setup_position_embeddings(self, cfg):
        del cfg
        if not hasattr(self.rotary_emb, "cos_cached"):
            self.rotary_emb.setup_after_callback = self._setup_cos_sin_embeding
        else:
            self._setup_cos_sin_embeding()

    def _setup_cos_sin_embeding(self):
        if hasattr(self.rotary_emb, "cos_cached"):
            _ = self.rotary_emb.cos_cached
        if hasattr(self.rotary_emb, "sin_cached"):
            _ = self.rotary_emb.sin_cached

    def forward(
        self,
        inputs_embeds: Optional[Tensor] = None,
        time_position_ids: Optional[Tensor] = None,
        hight_position_ids: Optional[Tensor] = None,
        width_position_ids: Optional[Tensor] = None,
        past_seq_length: Optional[Tensor] = None,
        current_input_length: Optional[Tensor] = None,
        linear_attn_mask: Optional[Tensor] = None,
        past_key_cache: Optional[List[Tensor]] = None,
        past_value_cache: Optional[List[Tensor]] = None,
        past_conv_cache: Optional[List[Tensor]] = None,
        past_recurrent_state: Optional[List[Tensor]] = None,
    ):
        position_embeddings = self._build_position_embeddings(
            past_seq_length,
            time_position_ids,
            hight_position_ids,
            width_position_ids,
        )
        hidden_states = inputs_embeds
        conv_cache_out_list = []
        recurrent_state_out_list = []
        collected_hidden_states = []
        full_attn_cache_idx = 0
        linear_attn_cache_idx = 0
        split_conv_cache = self.split_conv_cache
        if not split_conv_cache and (
            _looks_like_flat_split_conv_cache(past_conv_cache) or _layers_use_split_conv_cache(self.layers)
        ):
            split_conv_cache = True
        if split_conv_cache and _is_nested_split_conv_cache(past_conv_cache):
            past_conv_cache = _regroup_flat_split_conv_cache(past_conv_cache)
        for idx_layer, decoder_layer in enumerate(self.layers):
            layer_type = self.layer_types[idx_layer]
            if self.use_cache:
                if layer_type == "full_attention":
                    _past_k_cache = past_key_cache[full_attn_cache_idx] if past_key_cache is not None else None
                    _past_v_cache = past_value_cache[full_attn_cache_idx] if past_value_cache is not None else None
                    _past_conv_cache = None
                    _past_recurrent_state = None
                    full_attn_cache_idx += 1
                else:
                    _past_k_cache = None
                    _past_v_cache = None
                    _past_conv_cache = _select_linear_attn_conv_cache(
                        past_conv_cache,
                        linear_attn_cache_idx,
                        split_conv_cache,
                    )
                    _past_recurrent_state = (
                        past_recurrent_state[linear_attn_cache_idx] if past_recurrent_state is not None else None
                    )
                    linear_attn_cache_idx += 1
            else:
                _past_k_cache = None
                _past_v_cache = None
                # Linear attention layers need conv_cache/recurrent_state even
                # when use_cache=False (as initial state for the computation)
                if layer_type == "linear_attention":
                    _past_conv_cache = _select_linear_attn_conv_cache(
                        past_conv_cache,
                        linear_attn_cache_idx,
                        split_conv_cache,
                    )
                    _past_recurrent_state = (
                        past_recurrent_state[linear_attn_cache_idx] if past_recurrent_state is not None else None
                    )
                    linear_attn_cache_idx += 1
                else:
                    _past_conv_cache = None
                    _past_recurrent_state = None
            if layer_type == "linear_attention":
                hidden_states, conv_cache_out, recurrent_state_out = decoder_layer(
                    hidden_states,
                    past_seq_length=past_seq_length,
                    current_input_length=current_input_length,
                    position_embeddings=position_embeddings,
                    linear_attn_mask=linear_attn_mask,
                    past_k_cache=_past_k_cache,
                    past_v_cache=_past_v_cache,
                    past_conv_cache=_past_conv_cache,
                    past_recurrent_state=_past_recurrent_state,
                )
                if isinstance(conv_cache_out, (list, tuple)):
                    conv_cache_out_list.extend(conv_cache_out)
                else:
                    conv_cache_out_list.append(conv_cache_out)
                if isinstance(recurrent_state_out, (list, tuple)):
                    recurrent_state_out_list.extend(recurrent_state_out)
                elif recurrent_state_out is not None:
                    recurrent_state_out_list.append(recurrent_state_out)
            else:
                hidden_states = decoder_layer(
                    hidden_states,
                    past_seq_length=past_seq_length,
                    current_input_length=current_input_length,
                    position_embeddings=position_embeddings,
                    linear_attn_mask=linear_attn_mask,
                    past_k_cache=_past_k_cache,
                    past_v_cache=_past_v_cache,
                    past_conv_cache=_past_conv_cache,
                    past_recurrent_state=_past_recurrent_state,
                )
            if self.output_hidden_state_indices is not None and idx_layer in self._output_hidden_set:
                collected_hidden_states.append(hidden_states)
            if self.enable_layer_tag:
                hidden_states = self.tags[idx_layer](hidden_states)
            if self.max_layers > 0 and idx_layer + 1 >= self.max_layers:
                break
            # if True:
            #     break
            # if idx_layer == 3:
            #     break
        if self.num_logits_to_keep == 0:
            pass
        else:
            hidden_states = self.llm_gather(hidden_states, current_input_length - 1)
        if self.output_hidden_state_indices is not None:
            target_hidden = torch.cat(collected_hidden_states, dim=-1)
            if self.num_logits_to_keep != 0:
                target_hidden = self.llm_gather(target_hidden, current_input_length - 1)
        hidden_states = self.norm(hidden_states)
        post_norm_out = hidden_states
        if split_conv_cache:
            conv_cache_out_list = _flatten_split_conv_cache_outputs(conv_cache_out_list)
        else:
            conv_cache_out_list = _flatten_merged_conv_cache_outputs(conv_cache_out_list)
        if self.output_hidden_state_indices is not None:
            return hidden_states, conv_cache_out_list, recurrent_state_out_list, target_hidden
        if self.output_post_norm_hidden:
            return hidden_states, conv_cache_out_list, recurrent_state_out_list, post_norm_out
        return hidden_states, conv_cache_out_list, recurrent_state_out_list
