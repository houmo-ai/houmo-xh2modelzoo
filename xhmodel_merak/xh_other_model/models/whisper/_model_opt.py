from typing import Dict, Optional

import torch
from torch import nn
from transformers.cache_utils import EncoderDecoderCache
from transformers.models.whisper.modeling_whisper import WhisperAttention

from xhquant.nn import LLMCacheV2, MaskedSoftmax
from xhquant.patch.core.rewriters import FUNCTION_REWRITER

from ...builder import XHLLM_TRACEABLE_MODULES, DynamicRegister


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
            f"Whisper expects the mel input features to be of length {expected_seq_length}, "
            f"but found {input_features.shape[-1]}. Make sure to pad the input mel features "
            f"to {expected_seq_length}."
        )

    output_attentions = (
        output_attentions if output_attentions is not None else self.config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    inputs_embeds = nn.functional.gelu(self.conv1(input_features))
    inputs_embeds = nn.functional.gelu(self.conv2(inputs_embeds))
    inputs_embeds = inputs_embeds.permute(0, 2, 1)

    all_positions = torch.arange(self.embed_positions.num_embeddings, device=inputs_embeds.device)
    hidden_states = inputs_embeds + self.embed_positions(all_positions)
    hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

    encoder_states = () if output_hidden_states else None
    all_attentions = () if output_attentions else None

    if head_mask is not None:
        assert head_mask.size()[0] == (len(self.layers)), (
            f"The head_mask should be specified for {len(self.layers)} layers, "
            f"but it is for {head_mask.size()[0]}."
        )

    for idx, encoder_layer in enumerate(self.layers):
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)
        to_drop = False
        if self.training:
            dropout_probability = torch.rand([])
            if dropout_probability < self.layerdrop:
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

    bsz, _ = hidden_states.shape[:-1]
    k_states = []
    v_states = []
    for layer in self.decoder_m.layers:
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
        k_states.append(k_state)
        v_states.append(v_state)

    # Grouped return: [k0, k1, ..., kN, v0, v1, ..., vN]
    return k_states + v_states


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
    encoder_attention_mask=None,
):
    output_attentions = (
        output_attentions if output_attentions is not None else self.config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

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

    if cache_position is not None:
        past_key_values_length = cache_position[0]
    elif past_key_values is not None:
        past_key_values_length = past_key_values.get_seq_length()

    if position_ids is None:
        position_ids = cache_position.unsqueeze(0).repeat(input_shape[0], 1)

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

    hidden_states = inputs_embeds + positions
    hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

    causal_mask = None

    all_hidden_states = () if output_hidden_states else None

    for attn_mask, mask_name in zip(
        [head_mask, cross_attn_head_mask], ["head_mask", "cross_attn_head_mask"],
        strict=True,
    ):
        if attn_mask is not None:
            assert attn_mask.size()[0] == (len(self.layers)), (
                f"The `{mask_name}` should be specified for {len(self.layers)} layers, "
                f"but it is for {head_mask.size()[0]}."
            )

    k_cache_list = []
    v_cache_list = []

    for idx, decoder_layer in enumerate(self.layers):
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        if self.training:
            dropout_probability = torch.rand([])
            if dropout_probability < self.layerdrop:
                continue

        layer_outputs, k_cache_n, v_cache_n = decoder_layer(
            hidden_states,
            attention_mask=causal_mask,
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

    hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states = residual + hidden_states

    # Cross-Attention Block
    cross_attn_weights = None
    if k_list is not None:
        residual = hidden_states
        hidden_states = self.encoder_attn_layer_norm(hidden_states)
        hidden_states, cross_attn_weights, _, _ = self.encoder_attn(
            hidden_states=hidden_states,
            key_value_states=encoder_hidden_states,
            attention_mask=encoder_attention_mask,
            layer_head_mask=cross_attn_layer_head_mask,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            k_list=k_list,
            v_list=v_list,
            past_len=past_len,
            current_len=current_len,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

    # Fully Connected
    residual = hidden_states
    hidden_states = self.final_layer_norm(hidden_states)
    hidden_states = self.activation_fn(self.fc1(hidden_states))
    hidden_states = nn.functional.dropout(
        hidden_states, p=self.activation_dropout, training=self.training
    )
    hidden_states = self.fc2(hidden_states)
    hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states = residual + hidden_states

    outputs = (hidden_states,)

    if output_attentions:
        outputs += (self_attn_weights, cross_attn_weights)

    return outputs, k_cache, v_cache


@XHLLM_TRACEABLE_MODULES.register_module({WhisperAttention: "WhisperAttention"})
class _Whisper_attention(DynamicRegister):  # noqa: N801
    def _setup(self, cfg: Optional[Dict] = None):
        self.k_cache = LLMCacheV2(axis=2)
        self.v_cache = LLMCacheV2(axis=2)
        self.masked_softmax = MaskedSoftmax(dim=-1, attention_max_length=-1)
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

        bsz, tgt_len = hidden_states.shape[:-1]
        q_input_shape = (bsz, tgt_len, -1, self.head_dim)

        query_states = self.q_proj(hidden_states) * self.scaling
        query_states = query_states.view(*q_input_shape)
        query_states = query_states.transpose(1, 2).contiguous()

        if past_key_values is not None and isinstance(past_key_values, EncoderDecoderCache):
            if is_cross_attention:
                key_states = k_list
                value_states = v_list
            else:
                past_key_values = past_key_values.self_attention_cache

        current_states = key_value_states if key_value_states is not None else hidden_states
        if is_cross_attention:
            key_states = k_list
            value_states = v_list
        else:
            key_states = self.k_proj(current_states).view(bsz, -1, self.num_heads, self.head_dim)
            value_states = self.v_proj(current_states).view(bsz, -1, self.num_heads, self.head_dim)
            key_states = key_states.transpose(1, 2).contiguous()
            value_states = value_states.transpose(1, 2).contiguous()

            key_states = self.k_cache(key_states, past_len, current_len, k_cache)
            value_states = self.v_cache(value_states, past_len, current_len, v_cache)

            if past_key_values is not None:
                cache_position = cache_position if not is_cross_attention else None
                key_states, value_states = past_key_values.update(
                    key_states,
                    value_states,
                    self.layer_idx,
                    {"cache_position": cache_position},
                )

        real_mask = attention_mask if is_cross_attention else mask_atten

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3))
        if is_cross_attention:
            if real_mask is not None and real_mask.ndim == 4:
                attn_weights = attn_weights + real_mask
            attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
            attn_output = torch.matmul(attn_weights, value_states)
            attn_output = attn_output.transpose(1, 2).contiguous()
        else:
            if real_mask is not None and real_mask.ndim == 4:
                attn_weights = attn_weights + real_mask
            attn_weights = self.masked_softmax(attn_weights, past_len)
            attn_output = torch.matmul(attn_weights, value_states)
            attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, tgt_len, -1).contiguous()
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights, key_states, value_states


def register_wrap_modules():
    pass
