# Copyright 2025 HOUMO AI
#
# File: _model_opt.py
# Description:
#   Model Opt model implementation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

from calendar import c
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
from torch import nn

# from transformers.masking_utils import create_causal_mask
from transformers.cache_utils import EncoderDecoderCache
from transformers.modeling_outputs import BaseModelOutput
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.whisper.modeling_whisper import WhisperAttention
from xhquant.frontend.dynamo_fx.dynamo_symbolic_trace import (
    dynamo_fx_trace,
    wrap_module,
)

# from transformers.models.whisper.modeling_whisper import eager_attention_forward
from xhquant.nn import BfpFlashAttention, LLMCacheV2, MaskedAdd, MaskedSoftmax, RMSNorm
from xhquant.ops.xh import kv_cache
from xhquant.patch.core.rewriters import FUNCTION_REWRITER, MODULE_REWRITER

from ..builder import XHLLM_TRACEABLE_MODULES, DynamicRegister


def eager_attention_forward_cus(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
):
    attn_weights = torch.matmul(query, key.transpose(2, 3))
    # attn_weights = torch.matmul(query, key.transpose(2, 3)) / math.sqrt(query.size(-1))
    if attention_mask is not None and attention_mask.ndim == 4:
        attn_weights = attn_weights + attention_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights



@FUNCTION_REWRITER.register_rewriter(
    "transformers.models.whisper.modeling_whisper.WhisperEncoder.forward"
)
def whisper_encoder_forward_v2(
    self,
    input_features,
    attention_mask=None,
    head_mask=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
):
    expected_seq_length = (
        self.config.max_source_positions * self.conv1.stride[0] * self.conv2.stride[0]
    )
    if input_features.shape[-1] != expected_seq_length:
        raise ValueError(
            f"Whisper expects the mel input features to be of length {expected_seq_length}, but found {input_features.shape[-1]}. Make sure to pad the input mel features to {expected_seq_length}."
        )

    output_attentions = (
        output_attentions
        if output_attentions is not None
        else self.config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.config.output_hidden_states
    )
    return_dict = (
        return_dict if return_dict is not None else self.config.use_return_dict
    )
    inputs_embeds = nn.functional.gelu(self.conv1(input_features))
    inputs_embeds = nn.functional.gelu(self.conv2(inputs_embeds))

    inputs_embeds = inputs_embeds.permute(0, 2, 1)
    all_positions = torch.arange(
        self.embed_positions.num_embeddings, device=inputs_embeds.device
    )

    hidden_states = inputs_embeds + self.embed_positions(all_positions)
    hidden_states = nn.functional.dropout(
        hidden_states, p=self.dropout, training=self.training
    )

    encoder_states = () if output_hidden_states else None
    all_attentions = () if output_attentions else None

    # check if head_mask has a correct number of layers specified if desired
    if head_mask is not None:
        assert head_mask.size()[0] == (len(self.layers)), (
            f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
        )

    for idx, encoder_layer in enumerate(self.layers):
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)
        # add LayerDrop (see https://huggingface.co/papers/1909.11556 for description)
        to_drop = False
        if self.training:
            dropout_probability = torch.rand([])
            if dropout_probability < self.layerdrop:  # skip the layer
                to_drop = True

        if to_drop:
            layer_outputs = (None, None)
        else:
            layer_outputs = encoder_layer(
                hidden_states,
                None,
                layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                output_attentions=output_attentions,
            )

            hidden_states = layer_outputs[0]

        if output_attentions:
            all_attentions = all_attentions + (layer_outputs[1],)

    hidden_states = self.layer_norm(hidden_states)
    # if output_hidden_states:
    #     encoder_states = encoder_states + (hidden_states,)

    output_kv_output = []
    # print(hidden_states.sum())
    bsz, tgt_len = hidden_states.shape[:-1]
    for layer in self.decoder_m.layers:
        # k_state = layer.encoder_attn.k_proj(hidden_states).view(1, -1, 16, 64).transpose(1, 2).contiguous()
        # v_state = layer.encoder_attn.v_proj(hidden_states).view(1, -1, 16, 64).transpose(1, 2).contiguous()
        k_state = (
            layer.encoder_attn.k_proj(hidden_states)
            .view(bsz, -1, layer.encoder_attn.num_heads, layer.encoder_attn.head_dim)
            .transpose(1, 2)
            .contiguous()
        )
        v_state = (
            layer.encoder_attn.v_proj(hidden_states)
            .view(bsz, -1, layer.encoder_attn.num_heads, layer.encoder_attn.head_dim)
            .transpose(1, 2)
            .contiguous()
        )
        output_kv_output.append((k_state, v_state))

    return output_kv_output


@FUNCTION_REWRITER.register_rewriter(
    "transformers.models.whisper.modeling_whisper.WhisperDecoder.forward"
)
def whisper_decoder_forward_v2(
    self,
    input_ids=None,
    attention_mask=None,
    encoder_hidden_states=None,
    head_mask=None,
    cross_attn_head_mask=None,
    past_key_values=None,
    inputs_embeds=None,
    position_ids=None,
    use_cache=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
    cache_position=None,
    past_key_values_length=0,
    k_cache=None,
    v_cache=None,
    current_len=None,
    past_len=None,
    k_list=None,
    v_list=None,
    mask_atten=None,
    encoder_attention_mask=None
):
    output_attentions = (
        output_attentions
        if output_attentions is not None
        else self.config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = (
        return_dict if return_dict is not None else self.config.use_return_dict
    )

    # retrieve input_ids and inputs_embeds
    if input_ids is not None and inputs_embeds is not None:
        raise ValueError(
            "You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time"
        )
    elif input_ids is not None:
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
    elif inputs_embeds is not None:
        input_shape = inputs_embeds.size()[:-1]
    else:
        raise ValueError(
            "You have to specify either decoder_input_ids or decoder_inputs_embeds"
        )

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    # if use_cache and past_key_values is None:
    #     if self.config.is_encoder_decoder:
    #         past_key_values = EncoderDecoderCache(
    #             DynamicCache(config=self.config), DynamicCache(config=self.config)
    #         )
    #     else:
    #         past_key_values = DynamicCache(config=self.config)

    # past_key_values_length = 0
    if cache_position is not None:
        past_key_values_length = cache_position[0]  # 0
    elif past_key_values is not None:
        past_key_values_length = past_key_values.get_seq_length()

    # if cache_position is None:
    #     cache_position = torch.arange(
    #         past_key_values_length, past_key_values_length + input_shape[1], device=inputs_embeds.device
    #     )

    if position_ids is None:
        position_ids = cache_position.unsqueeze(0).repeat(input_shape[0], 1)

    # embed positions
    if input_ids is not None:
        positions = self.embed_positions(
            input_ids,
            past_key_values_length=past_key_values_length,
            position_ids=position_ids,
        )
    else:
        positions = self.embed_positions(
            inputs_embeds,
            past_key_values_length=past_key_values_length,
            position_ids=position_ids,
        )

    hidden_states = inputs_embeds + positions  # .to(inputs_embeds.device)
    hidden_states = nn.functional.dropout(
        hidden_states, p=self.dropout, training=self.training
    )

    # causal_mask = create_causal_mask(
    #     config=self.config,
    #     input_embeds=inputs_embeds,
    #     attention_mask=attention_mask,
    #     cache_position=cache_position,
    #     past_key_values=past_key_values,
    #     position_ids=position_ids,
    # )
    causal_mask = None

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    all_cross_attentions = (
        () if (output_attentions and encoder_hidden_states is not None) else None
    )

    # check if head_mask/cross_attn_head_mask has a correct number of layers specified if desired
    for attn_mask, mask_name in zip(
        [head_mask, cross_attn_head_mask], ["head_mask", "cross_attn_head_mask"]
    ):
        if attn_mask is not None:
            assert attn_mask.size()[0] == (len(self.layers)), (
                f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for"
                f" {head_mask.size()[0]}."
            )

    k_cache_list = []
    v_cache_list = []

    for idx, decoder_layer in enumerate(self.layers):
        # add LayerDrop (see https://huggingface.co/papers/1909.11556 for description)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        if self.training:
            dropout_probability = torch.rand([])
            if dropout_probability < self.layerdrop:
                continue

        layer_outputs, k_cache_n, v_cache_n = decoder_layer(
            hidden_states,
            attention_mask=causal_mask,
            # encoder_hidden_states=encoder_hidden_states[idx], # encoder kv list
            k_list=k_list[idx],
            v_list=v_list[idx],
            layer_head_mask=(head_mask[idx] if head_mask is not None else None),
            cross_attn_layer_head_mask=(
                cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None
            ),
            past_key_values=past_key_values if use_cache else None,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            k_cache=k_cache[idx],
            v_cache=v_cache[idx],
            current_len=current_len,
            past_len=past_len,
            mask_atten=mask_atten,
            encoder_attention_mask=encoder_attention_mask,
        )
        hidden_states = layer_outputs[0]

        k_cache_list.append(k_cache_n)
        v_cache_list.append(v_cache_n)

    hidden_states = self.layer_norm(hidden_states)

    return hidden_states, k_cache_list, v_cache_list


@FUNCTION_REWRITER.register_rewriter(
    "transformers.models.whisper.modeling_whisper.WhisperDecoderLayer.forward"
)
def whisper_decoder_layer_forward_v2(
    self,
    hidden_states,
    attention_mask=None,
    encoder_hidden_states=None,
    encoder_attention_mask=None,
    layer_head_mask=None,
    cross_attn_layer_head_mask=None,
    past_key_values=None,
    output_attentions=False,
    use_cache=False,
    cache_position=None,
    k_cache=None,
    v_cache=None,
    current_len=None,
    k_list=None,
    v_list=None,
    mask_atten=None,
    past_len=None,
):
    residual = hidden_states
    hidden_states = self.self_attn_layer_norm(hidden_states)

    # Self Attention
    hidden_states, self_attn_weights, k_cache, v_cache = self.self_attn(
        hidden_states=hidden_states,
        past_key_values=past_key_values,
        attention_mask=attention_mask,
        layer_head_mask=layer_head_mask,
        output_attentions=output_attentions,
        cache_position=cache_position,
        k_cache=k_cache,
        v_cache=v_cache,
        current_len=current_len,
        past_len=past_len,
        mask_atten=mask_atten,
    )
    # self.key_cache_data = k_cache
    # self.value_cache_data = v_cache

    hidden_states = nn.functional.dropout(
        hidden_states, p=self.dropout, training=self.training
    )
    hidden_states = residual + hidden_states

    # Cross-Attention Block
    cross_attn_weights = None
    if k_list is not None:
        residual = hidden_states
        hidden_states = self.encoder_attn_layer_norm(hidden_states)
        hidden_states, cross_attn_weights, _, _ = self.encoder_attn(
            hidden_states=hidden_states,
            key_value_states=encoder_hidden_states,  # encoder
            attention_mask=encoder_attention_mask,
            layer_head_mask=cross_attn_layer_head_mask,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            k_list=k_list,
            v_list=v_list,
            past_len=past_len,
            current_len=current_len,
        )
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )
        hidden_states = residual + hidden_states

    # Fully Connected
    residual = hidden_states
    hidden_states = self.final_layer_norm(hidden_states)
    hidden_states = self.activation_fn(self.fc1(hidden_states))
    hidden_states = nn.functional.dropout(
        hidden_states, p=self.activation_dropout, training=self.training
    )
    hidden_states = self.fc2(hidden_states)
    hidden_states = nn.functional.dropout(
        hidden_states, p=self.dropout, training=self.training
    )
    hidden_states = residual + hidden_states

    outputs = (hidden_states,)  # 1.9854

    if output_attentions:
        outputs += (self_attn_weights, cross_attn_weights)

    return outputs, k_cache, v_cache


@XHLLM_TRACEABLE_MODULES.register_module(
    {
        WhisperAttention: "WhisperAttention",
    }
)
class _Whisper_attention(DynamicRegister):
    def _setup(self, cfg: Optional[Dict] = None):
        self.k_cache = LLMCacheV2(
            axis=2,
        )
        self.v_cache = LLMCacheV2(
            axis=2,
        )

        return self

    def forward(
        self,
        hidden_states,
        key_value_states=None,
        past_key_values=None,
        attention_mask=None,
        layer_head_mask=None,
        output_attentions=False,
        cache_position=None,
        k_cache=None,
        v_cache=None,
        current_len=None,
        past_len=None,
        k_list=None,
        v_list=None,
        mask_atten=None,
        **kwargs,
    ):
        is_cross_attention = k_list is not None

        # determine input shapes
        bsz, tgt_len = hidden_states.shape[:-1]
        q_input_shape = (bsz, tgt_len, -1, self.head_dim)

        # Scaling is susceptible to floating point arithmetics' inprecisions
        # which can lead to different results (this is dependent from model
        # to model, e.g. whisper is one such case). We therefore keep the
        # original order of scaling to follow the original implementation
        # and enforce no scaling (1.0) in the attention call below.
        query_states = self.q_proj(hidden_states) * self.scaling  # 4.4932
        query_states = query_states.view(*q_input_shape)
        query_states = query_states.transpose(1, 2).contiguous()

        # Check is encoder-decoder model is being used. Otherwise we'll get `DynamicCache`
        if past_key_values is not None and isinstance(
            past_key_values, EncoderDecoderCache
        ):
            is_updated = past_key_values.is_updated.get(self.layer_idx)
            if is_cross_attention:
                key_states = k_list
                value_states = v_list
                # past_key_values.is_updated[self.layer_idx] = True
                # past_key_values = past_key_values.cross_attention_cache
            else:
                past_key_values = past_key_values.self_attention_cache

        # use key_value_states if cross attention
        current_states = (
            key_value_states if key_value_states is not None else hidden_states
        )  # -3.8923
        if is_cross_attention:
            key_states = k_list
            value_states = v_list
        else:
            key_states = self.k_proj(current_states).view(
                bsz, -1, self.num_heads, self.head_dim
            )  # 60.5748
            value_states = self.v_proj(current_states).view(
                bsz, -1, self.num_heads, self.head_dim
            )  # 11.0236
            key_states = key_states.transpose(1, 2).contiguous()
            value_states = value_states.transpose(1, 2).contiguous()

            # cache_len = torch.tensor(key_states.shape[2], device=key_states.device)

            key_states = self.k_cache(
                key_states, past_len, current_len, k_cache
            )  # 62.2250
            value_states = self.v_cache(
                value_states, past_len, current_len, v_cache
            )  # 10.9369

            if past_key_values is not None:
                # save all key/value_states to cache to be re-used for fast auto-regressive generation
                cache_position = cache_position if not is_cross_attention else None
                key_states, value_states = past_key_values.update(
                    key_states,
                    value_states,
                    self.layer_idx,
                    {"cache_position": cache_position},
                )

        # attention_interface: Callable = eager_attention_forward
        # if self.config._attn_implementation != "eager":
        #     attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        # if self.is_encoder_layer:
        #     # if self.layer_idx == 0:
        #     # print(key_states.sum(), "**********************")
        #     print(key_states.shape, value_states.shape, self.layer_idx)

        # if not is_cross_attention:
        #     attention_mask = torch.ones(
        #         (query_states.shape[0], query_states.shape[1], query_states.shape[2], key_states.shape[2]),
        #         dtype=torch.float16, # [1,16, 1, 1024]
        #     )
        #     attention_mask[:,:,:,  current_len+cache_len: ] *= -65504

        # attn_output, attn_weights = eager_attention_forward_cus(
        #     query_states,  # [1, 16, 1, 64]  4.4932
        #     key_states,  # [1, 16, 1024, 64] 62.2250
        #     value_states,  # [1, 16, 1024, 64] 10.9369
        #     mask_atten,  # none
        # )
        
        real_mask = attention_mask if is_cross_attention else mask_atten

        attn_output, attn_weights = eager_attention_forward_cus(
            query_states,  
            key_states,  
            value_states,  
            real_mask,  # 传入根据情况选择的 mask
        )

        attn_output = attn_output.reshape(bsz, tgt_len, -1).contiguous()  # 6.1019
        attn_output = self.out_proj(attn_output)  # -1.2885

        return attn_output, attn_weights, key_states, value_states


"""
@FUNCTION_REWRITER.register_rewriter("transformers.models.whisper.modeling_whisper.WhisperAttention.forward")
def whisper_attention_forward_v2(
    self,
    hidden_states,
    key_value_states=None,
    past_key_values=None,
    attention_mask=None,
    layer_head_mask=None,
    output_attentions=False,
    cache_position=None,
    k_cache =None,
    v_cache =None,
    current_len=None,
    **kwargs,
):
    is_cross_attention = key_value_states is not None

    # determine input shapes
    bsz, tgt_len = hidden_states.shape[:-1]
    q_input_shape = (bsz, tgt_len, -1, self.head_dim)

    # Scaling is susceptible to floating point arithmetics' inprecisions
    # which can lead to different results (this is dependent from model
    # to model, e.g. whisper is one such case). We therefore keep the
    # original order of scaling to follow the original implementation
    # and enforce no scaling (1.0) in the attention call below.
    query_states = self.q_proj(hidden_states) * self.scaling # 4.4932
    query_states = query_states.view(*q_input_shape)
    query_states = query_states.transpose(1, 2).contiguous()

    # Check is encoder-decoder model is being used. Otherwise we'll get `DynamicCache`
    if past_key_values is not None and isinstance(past_key_values, EncoderDecoderCache):
        is_updated = past_key_values.is_updated.get(self.layer_idx) 
        if is_cross_attention:
            key_states = past_key_values[0]
            value_states = past_key_values[1]
            # past_key_values.is_updated[self.layer_idx] = True
            # past_key_values = past_key_values.cross_attention_cache
        else:
            past_key_values = past_key_values.self_attention_cache

    # use key_value_states if cross attention
    current_states = key_value_states if key_value_states is not None else hidden_states # -3.8923
    if is_cross_attention and key_value_states:
        key_states = key_value_states[0]
        value_states = key_value_states[1]
    else:
        key_states = self.k_proj(current_states).view(bsz, -1, self.num_heads, self.head_dim) # 60.5748
        value_states = self.v_proj(current_states).view(bsz, -1, self.num_heads, self.head_dim) # 11.0236
        key_states = key_states.transpose(1, 2).contiguous()
        value_states = value_states.transpose(1, 2).contiguous()

        cache_len = key_states.shape[2]
        key_states = torch.cat([k_cache[:,:, :current_len, :], key_states], dim=2)
        # seq_indices = torch.arange(current_len, current_len + cache_len, device=k_cache.device)
        # seq_indices = seq_indices[None, None, :, None].expand(key_states.shape[0], key_states.shape[1], -1, key_states.shape[3])
        # k_cache.scatter_(dim=2, index=seq_indices, src=key_states)

        
        k_cache[:,:, current_len:current_len+cache_len, :] = key_states
        # key_states = torch.cat([
        #     k_cache[:,:, :current_len, :]
        #     key_states,
        #     k_cache[:,:, current_len+cache_len:, :]
        # ], dim=2)

        # value_states = torch.cat([v_cache[:,:, :current_len, :], value_states], dim=2)
        # v_cache.scatter_(dim=2, index=seq_indices, src=value_states)
        v_cache[:,:, current_len:current_len+cache_len, :] = value_states

        if past_key_values is not None:
            # save all key/value_states to cache to be re-used for fast auto-regressive generation
            cache_position = cache_position if not is_cross_attention else None
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx, {"cache_position": cache_position}
            )

    attention_interface: Callable = eager_attention_forward
    if self.config._attn_implementation != "eager":
        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
    
    # if self.is_encoder_layer:
    #     # if self.layer_idx == 0:
    #     # print(key_states.sum(), "**********************")
    #     print(key_states.shape, value_states.shape, self.layer_idx)
    if k_cache is not None:
        attn_output, attn_weights = attention_interface(
            self,
            query_states, # [1, 16, 4, 64]
            k_cache, # [1, 1500, 16, 64]
            v_cache, # [1, 1500, 16, 64]
            attention_mask,
            dropout=0.0 if not self.training else self.dropout,
            scaling=1.0,
            output_attentions=output_attentions,
            head_mask=layer_head_mask,
            **kwargs,
        )
    else:
        attn_output, attn_weights = attention_interface(
            self,
            query_states, # [1, 16, 4, 64]
            key_states, # [1, 1500, 16, 64]
            value_states, # [1, 1500, 16, 64]
            attention_mask,
            dropout=0.0 if not self.training else self.dropout,
            scaling=1.0,
            output_attentions=output_attentions,
            head_mask=layer_head_mask,
            **kwargs,
        )

    attn_output = attn_output.reshape(bsz, tgt_len, -1).contiguous() # 11.0236
    attn_output = self.out_proj(attn_output) # 0.6203

    return attn_output, attn_weights, k_cache, v_cache

"""


def register_wrap_modules():
    pass
