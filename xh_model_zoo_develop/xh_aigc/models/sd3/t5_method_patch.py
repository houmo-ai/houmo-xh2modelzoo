import torch
import torch.nn as nn
from transformers import T5EncoderModel
from transformers.cache_utils import Cache, DynamicCache, EncoderDecoderCache
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers.modeling_utils import is_torchdynamo_compiling, logger
from transformers.models.t5.modeling_t5 import T5Stack

from .hadamard_utils import random_hadamard_matrix


def T5_forward(
    self,
    input_ids=None,
    attention_mask=None,
    encoder_hidden_states=None,
    encoder_attention_mask=None,
    inputs_embeds=None,
    head_mask=None,
    cross_attn_head_mask=None,
    past_key_values=None,
    use_cache=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
    cache_position=None,
):
    # Model parallel
    if self.model_parallel:
        torch.cuda.set_device(self.first_device)
        self.embed_tokens = self.embed_tokens.to(self.first_device)
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if input_ids is not None and inputs_embeds is not None:
        err_msg_prefix = "decoder_" if self.is_decoder else ""
        raise ValueError(
            f"You cannot specify both {err_msg_prefix}input_ids and {err_msg_prefix}inputs_embeds at the same time"
        )
    elif input_ids is not None:
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
    elif inputs_embeds is not None:
        input_shape = inputs_embeds.size()[:-1]
    else:
        err_msg_prefix = "decoder_" if self.is_decoder else ""
        raise ValueError(f"You have to specify either {err_msg_prefix}input_ids or {err_msg_prefix}inputs_embeds")

    if self.gradient_checkpointing and self.training:
        if use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

    if inputs_embeds is None:
        if self.embed_tokens is None:
            raise ValueError("You have to initialize the model with valid token embeddings")
        inputs_embeds = self.embed_tokens(input_ids)

    batch_size, seq_length = input_shape

    if use_cache is True:
        if not self.is_decoder:
            raise ValueError(f"`use_cache` can only be set to `True` if {self} is used as a decoder")

    # initialize past_key_values
    return_legacy_cache = False
    return_self_attention_cache = False
    if self.is_decoder and (use_cache or past_key_values is not None):
        if isinstance(past_key_values, Cache) and not isinstance(past_key_values, EncoderDecoderCache):
            return_self_attention_cache = True
            past_key_values = EncoderDecoderCache(past_key_values, DynamicCache())
        elif not isinstance(past_key_values, EncoderDecoderCache):
            return_legacy_cache = True
            logger.warning_once(
                "Passing a tuple of `past_key_values` is deprecated and will be removed in Transformers v4.48.0. "
                "You should pass an instance of `EncoderDecoderCache` instead, e.g. "
                "`past_key_values=EncoderDecoderCache.from_legacy_cache(past_key_values)`."
            )
            past_key_values = EncoderDecoderCache.from_legacy_cache(past_key_values)
        elif past_key_values is None:
            past_key_values = EncoderDecoderCache(DynamicCache(), DynamicCache())
    elif not self.is_decoder:
        # do not pass cache object down the line for encoder stack
        # it messes indexing later in decoder-stack because cache object is modified in-place
        past_key_values = None

    past_key_values_length = past_key_values.get_seq_length() if past_key_values is not None else 0
    if cache_position is None:
        cache_position = torch.arange(
            past_key_values_length, past_key_values_length + seq_length, device=inputs_embeds.device
        )

    if attention_mask is None and not is_torchdynamo_compiling():
        # required mask seq length can be calculated via length of past cache
        mask_seq_length = past_key_values_length + seq_length
        attention_mask = torch.ones(batch_size, mask_seq_length, device=inputs_embeds.device)

    if self.config.is_decoder:
        causal_mask = self._update_causal_mask(
            attention_mask,
            inputs_embeds,
            cache_position,
            past_key_values.self_attention_cache if past_key_values is not None else None,
            output_attentions,
        )
    elif attention_mask is not None:
        causal_mask = attention_mask[:, None, None, :]
        causal_mask = causal_mask.to(dtype=inputs_embeds.dtype)
        causal_mask = (1.0 - causal_mask) * torch.finfo(inputs_embeds.dtype).min
    else:
        causal_mask = None

    # If a 2D or 3D attention mask is provided for the cross-attention
    # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
    if self.is_decoder and encoder_hidden_states is not None:
        encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
        encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
        if encoder_attention_mask is None:
            encoder_attention_mask = torch.ones(encoder_hidden_shape, device=inputs_embeds.device, dtype=torch.long)
        encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
    else:
        encoder_extended_attention_mask = None

    # Prepare head mask if needed
    head_mask = self.get_head_mask(head_mask, self.config.num_layers)
    cross_attn_head_mask = self.get_head_mask(cross_attn_head_mask, self.config.num_layers)
    all_hidden_states = () if output_hidden_states else None
    all_attentions = () if output_attentions else None
    all_cross_attentions = () if (output_attentions and self.is_decoder) else None
    position_bias = None
    encoder_decoder_position_bias = None

    hidden_states = self.dropout(inputs_embeds)
    debug = False

    for i, layer_module in enumerate(self.block):
        layer_head_mask = head_mask[i]
        cross_attn_layer_head_mask = cross_attn_head_mask[i]
        # Model parallel
        if self.model_parallel:
            torch.cuda.set_device(hidden_states.device)
            # Ensure that attention_mask is always on the same device as hidden_states
            if causal_mask is not None:
                causal_mask = causal_mask.to(hidden_states.device)
            if position_bias is not None:
                position_bias = position_bias.to(hidden_states.device)
            if encoder_hidden_states is not None:
                encoder_hidden_states = encoder_hidden_states.to(hidden_states.device)
            if encoder_extended_attention_mask is not None:
                encoder_extended_attention_mask = encoder_extended_attention_mask.to(hidden_states.device)
            if encoder_decoder_position_bias is not None:
                encoder_decoder_position_bias = encoder_decoder_position_bias.to(hidden_states.device)
            if layer_head_mask is not None:
                layer_head_mask = layer_head_mask.to(hidden_states.device)
            if cross_attn_layer_head_mask is not None:
                cross_attn_layer_head_mask = cross_attn_layer_head_mask.to(hidden_states.device)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                layer_module.forward,
                hidden_states,
                causal_mask,
                position_bias,
                encoder_hidden_states,
                encoder_extended_attention_mask,
                encoder_decoder_position_bias,
                layer_head_mask,
                cross_attn_layer_head_mask,
                None,  # past_key_value is always None with gradient checkpointing
                use_cache,
                output_attentions,
                return_dict,
                cache_position,
            )
        else:
            layer_outputs = layer_module(
                hidden_states,
                attention_mask=causal_mask,
                position_bias=position_bias,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
                encoder_decoder_position_bias=encoder_decoder_position_bias,
                layer_head_mask=layer_head_mask,
                cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                return_dict=return_dict,
                cache_position=cache_position,
                # indice= i if i >= len(self.block) - 1 else None,
            )

        # layer_outputs is a tuple with:
        # hidden-states, key-value-states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)
        if use_cache is False:
            layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]

        hidden_states, next_decoder_cache = layer_outputs[:2]

        # We share the position biases between the layers - the first layer store them
        # layer_outputs = hidden-states, key-value-states (self-attention position bias), (self-attention weights),
        # (cross-attention position bias), (cross-attention weights)
        position_bias = layer_outputs[2]
        if self.is_decoder and encoder_hidden_states is not None:
            encoder_decoder_position_bias = layer_outputs[4 if output_attentions else 3]

        if output_attentions:
            all_attentions = all_attentions + (layer_outputs[3],)
            if self.is_decoder:
                all_cross_attentions = all_cross_attentions + (layer_outputs[5],)

        # Model Parallel: If it's the last layer for that device, put things on the next device
        if self.model_parallel:
            for k, v in self.device_map.items():
                if i == v[-1] and "cuda:" + str(k) != self.last_device:
                    hidden_states = hidden_states.to("cuda:" + str(k + 1))

    if hasattr(self, "online_rotation"):
        hidden_states = hidden_states.to(torch.float32)
        self.online_rotation = self.online_rotation.to(hidden_states.dtype)
        hidden_states = self.online_rotation(hidden_states)
        # hidden_states = hidden_states.half()

    if debug:
        torch.save(hidden_states, "final_layer_norm-input.pth")

    hidden_states = self.final_layer_norm(hidden_states)
    hidden_states = self.dropout(hidden_states)

    if hasattr(self, "online_rotation_after_norm"):
        self.online_rotation_after_norm = self.online_rotation_after_norm.to(hidden_states.dtype)
        hidden_states = self.online_rotation_after_norm(hidden_states)

    # Add last layer
    if output_hidden_states:
        all_hidden_states = all_hidden_states + (hidden_states,)

    next_cache = next_decoder_cache if use_cache else None
    if return_self_attention_cache:
        next_cache = past_key_values.self_attention_cache
    if return_legacy_cache:
        next_cache = past_key_values.to_legacy_cache()

    if not return_dict:
        return tuple(
            v
            for v in [
                hidden_states,
                next_cache,
                all_hidden_states,
                all_attentions,
                all_cross_attentions,
            ]
            if v is not None
        )
    return BaseModelOutputWithPastAndCrossAttentions(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_attentions,
        cross_attentions=all_cross_attentions,
    )


def get_hadmard_matrix(hidden_size, quarot_matrix_size=None, rotate_mode="hadamard", device=None):
    if quarot_matrix_size is None:
        Q = random_hadamard_matrix(hidden_size, device=device)
    else:
        Q = torch.zeros(
            [hidden_size, hidden_size],
            dtype=torch.float64,
            device=device,
            requires_grad=False,
        )
        assert int(hidden_size / quarot_matrix_size) * quarot_matrix_size == hidden_size, "not fully dividen"
        for i in range(0, hidden_size, quarot_matrix_size):
            local_Q = random_hadamard_matrix(quarot_matrix_size, device=device)
            Q[i : i + quarot_matrix_size, i : i + quarot_matrix_size] = local_Q

    return Q


def rotation_weight(layer, Q, with_bias=False, transpose=False):
    dtype = layer.weight.data.dtype
    W = layer.weight.data.to(dtype=torch.float64)
    if transpose:
        layer.weight.data = torch.matmul(Q.T, W).to(dtype=dtype)
    else:
        layer.weight.data = torch.matmul(W, Q).to(dtype=dtype)

    if with_bias and layer.bias is not None:
        b = layer.bias.data.to(dtype=torch.float64)
        if transpose:
            layer.bias.data = torch.matmul(Q.T, b).to(dtype=dtype)
        else:
            layer.bias.data = torch.matmul(b, Q).to(dtype=dtype)


def fuse_layernorm_weight(linear, ln):
    assert hasattr(ln, "weight")
    assert hasattr(linear, "weight")
    linear_dtype = linear.weight.dtype
    W = linear.weight.data.double()
    linear.weight.data = (W * ln.weight.double()).to(linear_dtype)
    # ln.weight.fill_(1.)

    if hasattr(ln, "bias") and ln.bias is not None:
        if linear.bias is None:
            linear.bias = torch.nn.Parameter(torch.zeros(linear.out_features, dtype=torch.float64))
        linear.bias.data = linear.bias.data.double() + torch.matmul(W, ln.bias.double())
        linear.bias.data = linear.bias.data.to(linear_dtype)
        # ln.bias.fill_(0.)


@torch.no_grad()
def hadmard_t5(text_encoder_3: T5EncoderModel):
    t5 = text_encoder_3
    if t5 is None:
        return

    # t5 = t5.to(torch.float32)
    for name, module in t5.named_modules():
        if isinstance(module, T5Stack):
            hidden_size = module.embed_tokens.weight.shape[-1]
            device = module.embed_tokens.weight.device
            Q = get_hadmard_matrix(hidden_size, device=device)

            # online_rotation = "pre_norm"
            online_rotation = "after_norm"
            if online_rotation == "pre_norm":
                module.add_module(
                    "online_rotation",
                    nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=False, device=device),
                )
                module.online_rotation.weight.data = Q.to(torch.float32)
            else:
                module.add_module(
                    "online_rotation_after_norm",
                    nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=False, device=device),
                )
                module.online_rotation_after_norm.weight.fill_(0.0)
                for i in range(hidden_size):
                    module.online_rotation_after_norm.weight[i][i].fill_(1.0)
                # module.online_rotation_after_norm.weight.data = Q.to(torch.float32)

                fuse_layernorm_weight(module.online_rotation_after_norm, module.final_layer_norm)
                if hasattr(module.final_layer_norm, "weight"):
                    module.final_layer_norm.weight.fill_(1.0)
                if hasattr(module.final_layer_norm, "bias") and module.final_layer_norm.bias is not None:
                    module.final_layer_norm.bias.fill_(0.0)

                rotation_weight(module.online_rotation_after_norm, Q)

            rotation_last_ffn_wo = False
            if rotation_last_ffn_wo:
                rotation_weight(module.block[-1].layer[1].DenseReluDense.wo, Q, with_bias=True, transpose=True)

                module.block[-1].layer[-1].add_module(
                    "online_rotation",
                    nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=False, device=device),
                )
                module.block[-1].layer[-1].online_rotation.weight.data = Q.t().to(torch.float32)
                continue

            rotation_last_attn_wo = False
            if rotation_last_attn_wo:
                module.block[-1].layer[0].add_module(
                    "online_rotation",
                    nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=False, device=device),
                )
                module.block[-1].layer[0].online_rotation.weight.data = Q.t().to(torch.float32)

                # module.block[-1].layer[0].add_module(
                #     "online_rotation_debug",
                #     nn.Linear(
                #         in_features=hidden_size,
                #         out_features=hidden_size,
                #         bias=False,
                #         device=device)
                # )
                # module.block[-1].layer[0].online_rotation_debug.weight.data = Q.to(torch.float32)

                # module.block[-1].layer[1].add_module(
                #     "online_rotation_debug",
                #     nn.Linear(
                #         in_features=hidden_size,
                #         out_features=hidden_size,
                #         bias=False,
                #         device=device)
                # )
                # module.block[-1].layer[1].online_rotation_debug.weight.data = Q.to(torch.float32)

                module.block[-1].layer[0].SelfAttention.o = module.block[-1].layer[0].SelfAttention.o.to(torch.float32)
                rotation_weight(module.block[-1].layer[0].SelfAttention.o, Q, with_bias=True, transpose=True)

                module.block[-1].layer[1].DenseReluDense.wi_0 = (
                    module.block[-1].layer[1].DenseReluDense.wi_0.to(torch.float32)
                )
                module.block[-1].layer[1].DenseReluDense.wi_1 = (
                    module.block[-1].layer[1].DenseReluDense.wi_1.to(torch.float32)
                )
                # fuse RMSnorm weight
                fuse_layernorm_weight(
                    module.block[-1].layer[1].DenseReluDense.wi_0, module.block[-1].layer[1].layer_norm
                )
                fuse_layernorm_weight(
                    module.block[-1].layer[1].DenseReluDense.wi_1, module.block[-1].layer[1].layer_norm
                )
                if hasattr(module.block[-1].layer[1].layer_norm, "weight"):
                    module.block[-1].layer[1].layer_norm.weight.fill_(1.0)
                if hasattr(module.block[-1].layer[1].layer_norm, "bias"):
                    module.block[-1].layer[1].layer_norm.bias.fill_(0.0)

                rotation_weight(module.block[-1].layer[1].DenseReluDense.wi_0, Q)
                rotation_weight(module.block[-1].layer[1].DenseReluDense.wi_1, Q)

                rotation_weight(module.block[-1].layer[1].DenseReluDense.wo, Q, with_bias=True, transpose=True)

            rotation_all = True
            if rotation_all:
                rotation_weight(module.embed_tokens, Q)
                for i, target in enumerate(module.block):
                    fuse_layernorm_weight(target.layer[0].SelfAttention.q, target.layer[0].layer_norm)
                    fuse_layernorm_weight(target.layer[0].SelfAttention.k, target.layer[0].layer_norm)
                    fuse_layernorm_weight(target.layer[0].SelfAttention.v, target.layer[0].layer_norm)
                    if hasattr(target.layer[0].layer_norm, "weight"):
                        target.layer[0].layer_norm.weight.fill_(1.0)
                    if hasattr(target.layer[0].layer_norm, "bias") and target.layer[0].layer_norm.bias is not None:
                        target.layer[0].layer_norm.bias.fill_(0.0)

                    rotation_weight(target.layer[0].SelfAttention.q, Q)
                    rotation_weight(target.layer[0].SelfAttention.k, Q)
                    rotation_weight(target.layer[0].SelfAttention.v, Q)
                    rotation_weight(target.layer[0].SelfAttention.o, Q, with_bias=True, transpose=True)

                    fuse_layernorm_weight(target.layer[1].DenseReluDense.wi_0, target.layer[1].layer_norm)
                    fuse_layernorm_weight(target.layer[1].DenseReluDense.wi_1, target.layer[1].layer_norm)
                    if hasattr(target.layer[1].layer_norm, "weight"):
                        target.layer[1].layer_norm.weight.fill_(1.0)
                    if hasattr(target.layer[1].layer_norm, "bias") and target.layer[1].layer_norm.bias is not None:
                        target.layer[1].layer_norm.bias.fill_(0.0)

                    rotation_weight(target.layer[1].DenseReluDense.wi_0, Q)
                    rotation_weight(target.layer[1].DenseReluDense.wi_1, Q)
                    rotation_weight(target.layer[1].DenseReluDense.wo, Q, with_bias=True, transpose=True)

            force_fp16 = True
            if force_fp16:
                pass
                # if hasattr(module, "online_rotation"):
                #     module.online_rotation = module.online_rotation.to(torch.float16)
                #     module.online_rotation = module.online_rotation.to(torch.float32)

                # target = module.block[-1]
                # target.layer[1].DenseReluDense.wi_0 = target.layer[1].DenseReluDense.wi_0.to(torch.float16)
                # target.layer[1].DenseReluDense.wi_1 = target.layer[1].DenseReluDense.wi_1.to(torch.float16)
                # target.layer[1].DenseReluDense.wo = target.layer[1].DenseReluDense.wo.to(torch.float16)

                # target.layer[1].DenseReluDense.wi_0 = target.layer[1].DenseReluDense.wi_0.to(torch.float32)
                # target.layer[1].DenseReluDense.wi_1 = target.layer[1].DenseReluDense.wi_1.to(torch.float32)
                # target.layer[1].DenseReluDense.wo = target.layer[1].DenseReluDense.wo.to(torch.float32)

                # target.layer[0].SelfAttention.q = target.layer[0].SelfAttention.q.to(torch.float16)
                # target.layer[0].SelfAttention.k = target.layer[0].SelfAttention.k.to(torch.float16)
                # target.layer[0].SelfAttention.v = target.layer[0].SelfAttention.v.to(torch.float16)
                # target.layer[0].SelfAttention.o = target.layer[0].SelfAttention.o.to(torch.float16)

                for i, target in enumerate(module.block):
                    # target = target.to(torch.float16)
                    target.layer[1].DenseReluDense.wo = target.layer[1].DenseReluDense.wo.to(torch.float16)
                    # target.layer[1].DenseReluDense.wo = target.layer[1].DenseReluDense.wo.to(torch.float32)

            # patch the forward method
            import types

            module.forward = types.MethodType(T5_forward, module)
    return t5
