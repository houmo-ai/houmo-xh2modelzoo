import json
import math
import tempfile
import time
import types
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Union

import bitsandbytes.functional as BNBF
import onnx
import onnxsim
import torch
import transformers
from bitsandbytes.nn.modules import Linear4bit
from bitsandbytes.nn.modules import bnb
from diffusers import SD3Transformer2DModel
from diffusers import StableDiffusion3Pipeline
from torch import Tensor
from transformers.models.clip.modeling_clip import BaseModelOutputWithPooling
from transformers.models.clip.modeling_clip import CLIPSdpaAttention
from transformers.models.clip.modeling_clip import CLIPTextTransformer
from transformers.models.clip.modeling_clip import _create_4d_causal_attention_mask
from transformers.models.clip.modeling_clip import _prepare_4d_attention_mask
from transformers.models.clip.modeling_clip import is_torch_greater_or_equal_than_2_2
from transformers.models.t5.modeling_t5 import T5Block
from transformers.models.t5.modeling_t5 import T5Stack
from xhquant.api import QuantScheme
from xhquant.api import convert_onnx_to_hmonnx
from xhquant.nn.modules import Clip
from xhquant.utils import digit_version
from xhquant.utils import get_root_logger

from .t5_method_patch import hadmard_t5


@dataclass
class SD3ConvertConfig:
    quant_scheme: QuantScheme = field(default_factory=QuantScheme)
    guidance_scale: float = 7.0
    num_inference_steps: int = 28
    width: int = 512
    height: int = 512
    hadmard_t5: bool = False

    # 内部配置参数，不建议外部修改
    no_clip_fp16_t5: bool = False
    custom_patch_t5: bool = False


def _gelu_tanh(self, input):
    return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))


def update_submodule(model, name, module):
    splits = name.split(".")
    parent = model
    module_name = splits[-1]
    if len(splits) > 1 and parent is not None:
        for path in splits[:-1]:
            parent = getattr(parent, path)
            if parent is None:
                break
    if parent is not None:
        parent.add_module(module_name, module)


def CLIPTextTransformer_forward(
    self,
    input_ids: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
) -> Union[Tuple, BaseModelOutputWithPooling]:
    r"""
    Returns:

    """
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if input_ids is None:
        raise ValueError("You have to specify input_ids")

    input_shape = input_ids.size()
    input_ids = input_ids.view(-1, input_shape[-1])

    hidden_states = self.embeddings(input_ids=input_ids, position_ids=position_ids)

    # CLIP's text model uses causal mask, prepare it here.
    # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
    causal_attention_mask = _create_4d_causal_attention_mask(
        input_shape, hidden_states.dtype, device=hidden_states.device
    )

    # expand attention_mask
    if attention_mask is not None and not self._use_flash_attention_2:
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        attention_mask = _prepare_4d_attention_mask(attention_mask, hidden_states.dtype)

    encoder_outputs = self.encoder(
        inputs_embeds=hidden_states,
        attention_mask=attention_mask,
        causal_attention_mask=causal_attention_mask,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    last_hidden_state = encoder_outputs[0]
    last_hidden_state = self.final_layer_norm(last_hidden_state)

    if self.eos_token_id == 2:
        # The `eos_token_id` was incorrect before PR #24773: Let's keep what have been done here.
        # A CLIP model with such `eos_token_id` in the config can't work correctly with extra new tokens added
        # ------------------------------------------------------------
        # text_embeds.shape = [batch_size, sequence_length, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # casting to torch.int for onnx compatibility: argmax doesn't support int64 inputs with opset 14

        indice = input_ids.to(dtype=torch.int, device=last_hidden_state.device).argmax(dim=-1)
        pooled_output = last_hidden_state[:, indice]
    else:
        # The config gets updated `eos_token_id` from PR #24773 (so the use of exta new tokens is possible)
        pooled_output = last_hidden_state[
            torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
            # We need to get the first position of `eos_token_id` value (`pad_token_ids` might equal to `eos_token_id`)
            # Note: we assume each sequence (along batch dim.) contains an  `eos_token_id` (e.g. prepared by the tokenizer)
            (input_ids.to(dtype=torch.int, device=last_hidden_state.device) == self.eos_token_id).int().argmax(dim=-1),
        ]

    if not return_dict:
        return (last_hidden_state, pooled_output) + encoder_outputs[1:]

    return BaseModelOutputWithPooling(
        last_hidden_state=last_hidden_state,
        pooler_output=pooled_output,
        hidden_states=encoder_outputs.hidden_states,
        attentions=encoder_outputs.attentions,
    )


def scaled_dot_product_attention(
    query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False
) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias = attn_mask + attn_bias

    if enable_gqa:
        key = key.repeat_interleave(query.size(-3) // key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3) // value.size(-3), -3)

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    # attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value


def linear4bit_forward(self, x: torch.Tensor):
    if self.weight.shape[0] == self.out_features and self.weight.shape[1] == self.in_features:
        return torch.nn.functional.linear(x, self.weight, self.bias)

    from bitsandbytes.nn.modules import fix_4bit_weight_quant_state_from_module

    fix_4bit_weight_quant_state_from_module(self)

    # weights are cast automatically as Int8Params, but the bias has to be cast manually
    if self.bias is not None and self.bias.dtype != x.dtype:
        self.bias.data = self.bias.data.to(x.dtype)

    if not self.compute_type_is_set:
        self.set_compute_type(x)
        self.compute_type_is_set = True

    inp_dtype = x.dtype
    if self.compute_dtype is not None:
        x = x.to(self.compute_dtype)

    bias = None if self.bias is None else self.bias.to(self.compute_dtype)

    out1 = bnb.matmul_4bit(x, self.weight.t(), bias=bias, quant_state=self.weight.quant_state).to(inp_dtype)
    weight = BNBF.dequantize_4bit(self.weight.t(), self.weight.quant_state).to(x.dtype).t()
    self.weight = torch.nn.Parameter(weight, requires_grad=False)
    out2 = torch.nn.functional.linear(x, weight, bias)
    # diff = (out1 - out2).abs().sum().item()
    # print(diff)
    return out1


def CLIPSdpaAttention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    causal_attention_mask: Optional[torch.Tensor] = None,
    output_attentions: Optional[bool] = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if output_attentions:
        # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
        # logger.warning_once(
        #     "CLIPModel is using CLIPSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not "
        #     "support `output_attentions=True`. Falling back to the manual attention implementation, but specifying "
        #     "the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can "
        #     'be removed using the argument `attn_implementation="eager"` when loading the model.'
        # )
        return super().forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
        )

    # CLIP text model uses both `causal_attention_mask` and `attention_mask`
    if attention_mask is not None and causal_attention_mask is not None:
        attn_mask = attention_mask + causal_attention_mask
    elif causal_attention_mask is not None:
        attn_mask = causal_attention_mask
    else:
        attn_mask = attention_mask

    bsz, tgt_len, embed_dim = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)

    # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
    # Reference: https://github.com/pytorch/pytorch/issues/112577.
    if not is_torch_greater_or_equal_than_2_2 and query_states.device.type == "cuda" and attn_mask is not None:
        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()

    # CLIP text model uses both `causal_attention_mask` and `attention_mask` sequentially.
    # attn_output = torch.nn.functional.scaled_dot_product_attention(
    attn_output = scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=attn_mask,
        dropout_p=self.dropout if self.training else 0.0,
        scale=self.scale,
    )

    attn_output = attn_output.transpose(1, 2)
    attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

    attn_output = self.out_proj(attn_output)

    return attn_output, None


def T5Block_forward(
    self,
    hidden_states,
    attention_mask=None,
    position_bias=None,
    encoder_hidden_states=None,
    encoder_attention_mask=None,
    encoder_decoder_position_bias=None,
    layer_head_mask=None,
    cross_attn_layer_head_mask=None,
    past_key_value=None,
    use_cache=False,
    output_attentions=False,
    return_dict=True,
    cache_position=None,
):

    self_attention_outputs = self.layer[0](
        hidden_states,
        attention_mask=attention_mask,
        position_bias=position_bias,
        layer_head_mask=layer_head_mask,
        past_key_value=past_key_value,
        use_cache=use_cache,
        output_attentions=output_attentions,
        cache_position=cache_position,
    )
    hidden_states, past_key_value = self_attention_outputs[:2]
    attention_outputs = self_attention_outputs[2:]  # Keep self-attention outputs and relative position weights

    clamp_value = torch.finfo(torch.float16).max - 1000
    # clamp inf values to enable fp16 training
    # if hidden_states.dtype == torch.float16:
    #     clamp_value = torch.where(
    #         torch.isinf(hidden_states).any(),
    #         torch.finfo(hidden_states.dtype).max - 1000,
    #         torch.finfo(hidden_states.dtype).max,
    #     )
    #     hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

    hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

    do_cross_attention = self.is_decoder and encoder_hidden_states is not None
    if do_cross_attention:
        cross_attention_outputs = self.layer[1](
            hidden_states,
            key_value_states=encoder_hidden_states,
            attention_mask=encoder_attention_mask,
            position_bias=encoder_decoder_position_bias,
            layer_head_mask=cross_attn_layer_head_mask,
            past_key_value=past_key_value,
            query_length=cache_position[-1] + 1,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        hidden_states, past_key_value = cross_attention_outputs[:2]

        # clamp inf values to enable fp16 training
        # if hidden_states.dtype == torch.float16:
        #     clamp_value = torch.where(
        #         torch.isinf(hidden_states).any(),
        #         torch.finfo(hidden_states.dtype).max - 1000,
        #         torch.finfo(hidden_states.dtype).max,
        #     )
        #     hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        # Keep cross-attention outputs and relative position weights
        attention_outputs = attention_outputs + cross_attention_outputs[2:]

    # Apply Feed Forward layer
    hidden_states = self.layer[-1](hidden_states)

    # clamp inf values to enable fp16 training
    # if hidden_states.dtype == torch.float16:
    #     clamp_value = torch.where(
    #         torch.isinf(hidden_states).any(),
    #         torch.finfo(hidden_states.dtype).max - 1000,
    #         torch.finfo(hidden_states.dtype).max,
    #     )
    #     hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
    hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
    outputs = (hidden_states,)

    if use_cache:
        outputs = outputs + (past_key_value,) + attention_outputs
    else:
        outputs = outputs + attention_outputs

    return outputs  # hidden-states, past_key_value, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)


def T5Block_forward_1_1(
    self,
    hidden_states,
    attention_mask=None,
    position_bias=None,
    encoder_hidden_states=None,
    encoder_attention_mask=None,
    encoder_decoder_position_bias=None,
    layer_head_mask=None,
    cross_attn_layer_head_mask=None,
    past_key_value=None,
    use_cache=False,
    output_attentions=False,
    return_dict=True,
    cache_position=None,
    clamp_value=65504.0,
):
    self_attention_outputs = self.layer[0](
        hidden_states,
        attention_mask=attention_mask,
        position_bias=position_bias,
        layer_head_mask=layer_head_mask,
        past_key_value=past_key_value,
        use_cache=use_cache,
        output_attentions=output_attentions,
        cache_position=cache_position,
    )
    hidden_states, past_key_value = self_attention_outputs[:2]
    hidden_states = self.clip_1(hidden_states, min=-65504, max=65504)

    attention_outputs = self_attention_outputs[2:]  # Keep self-attention outputs and relative position weights

    do_cross_attention = self.is_decoder and encoder_hidden_states is not None
    assert not do_cross_attention, "do_cross_attention should be False"
    if do_cross_attention:
        cross_attention_outputs = self.layer[1](
            hidden_states,
            key_value_states=encoder_hidden_states,
            attention_mask=encoder_attention_mask,
            position_bias=encoder_decoder_position_bias,
            layer_head_mask=cross_attn_layer_head_mask,
            past_key_value=past_key_value,
            query_length=cache_position[-1] + 1,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        hidden_states, past_key_value = cross_attention_outputs[:2]
        attention_outputs = attention_outputs + cross_attention_outputs[2:]

    # Apply Feed Forward layer
    hidden_states = self.layer[-1](hidden_states)
    hidden_states = self.clip_2(hidden_states, min=-clamp_value, max=clamp_value)
    outputs = (hidden_states,)

    if use_cache:
        outputs = outputs + (past_key_value,) + attention_outputs
    else:
        outputs = outputs + attention_outputs

    return outputs  # hidden-states, past_key_value, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)


def T5Block_forward_1_2(
    self,
    hidden_states,
    attention_mask=None,
    position_bias=None,
    encoder_hidden_states=None,
    encoder_attention_mask=None,
    encoder_decoder_position_bias=None,
    layer_head_mask=None,
    cross_attn_layer_head_mask=None,
    past_key_value=None,
    use_cache=False,
    output_attentions=False,
    return_dict=True,
    cache_position=None,
    clamp_value=64504.0,
):
    self_attention_outputs = self.layer[0](
        hidden_states,
        attention_mask=attention_mask,
        position_bias=position_bias,
        layer_head_mask=layer_head_mask,
        past_key_value=past_key_value,
        use_cache=use_cache,
        output_attentions=output_attentions,
        cache_position=cache_position,
    )
    hidden_states, past_key_value = self_attention_outputs[:2]
    hidden_states = self.clip_1(hidden_states, min=-65504, max=65504)

    attention_outputs = self_attention_outputs[2:]  # Keep self-attention outputs and relative position weights

    do_cross_attention = self.is_decoder and encoder_hidden_states is not None
    assert not do_cross_attention, "do_cross_attention should be False"
    if do_cross_attention:
        cross_attention_outputs = self.layer[1](
            hidden_states,
            key_value_states=encoder_hidden_states,
            attention_mask=encoder_attention_mask,
            position_bias=encoder_decoder_position_bias,
            layer_head_mask=cross_attn_layer_head_mask,
            past_key_value=past_key_value,
            query_length=cache_position[-1] + 1,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        hidden_states, past_key_value = cross_attention_outputs[:2]
        attention_outputs = attention_outputs + cross_attention_outputs[2:]

    # Apply Feed Forward layer
    hidden_states = self.layer[-1](hidden_states)
    hidden_states = self.clip_2(hidden_states, min=-clamp_value, max=clamp_value)
    outputs = (hidden_states,)

    if use_cache:
        outputs = outputs + (past_key_value,) + attention_outputs
    else:
        outputs = outputs + attention_outputs

    return outputs  # hidden-states, past_key_value, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)


class SD3Converter:
    def __init__(self, pretrained_model_path: str, convert_config: SD3ConvertConfig):
        self.pretrained_model_path = pretrained_model_path
        self.convert_config = convert_config

    @classmethod
    def get_mmdit_dummy_inputs(cls, convert_config: SD3ConvertConfig) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        do_classifier_free_guidance = convert_config.guidance_scale > 1
        if do_classifier_free_guidance:
            batch_size = 2
        else:
            batch_size = 1
        shape = [batch_size, 16, int(convert_config.height // 8), int(convert_config.width // 8)]
        hidden_states = torch.ones(shape, dtype=torch.float32)
        encoder_hidden_states = torch.ones([batch_size, 333, 4096], dtype=torch.float32)
        pooled_projections = torch.ones([batch_size, 2048], dtype=torch.float32)
        timestep = torch.ones([batch_size], dtype=torch.float32)
        return (hidden_states, encoder_hidden_states, pooled_projections, timestep)

    @classmethod
    def export_onnx_mmdit(cls, hf_model: StableDiffusion3Pipeline, convert_config: SD3ConvertConfig, output_onnx_file):
        logger = get_root_logger()
        export_model: SD3Transformer2DModel = getattr(hf_model, "transformer")

        """
        hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)等价于:
            output_permute = input_tensor.permute(0, 5, 1, 3, 2, 4)
        """

        do_classifier_free_guidance = convert_config.guidance_scale > 1
        if do_classifier_free_guidance:
            batch_size = 2
        else:
            batch_size = 1
        shape = [batch_size, 16, int(convert_config.height // 8), int(convert_config.width // 8)]
        onnx_name = Path(output_onnx_file).stem
        # inputs = [("hidden_states", torch.float32, shape)]
        # inputs.append(("encoder_hidden_states", torch.float32, [batch_size, 333, 4096]))
        # inputs.append(("pooled_projections", torch.float32, [batch_size, 2048]))
        # inputs.append(("timestep", torch.float32, [batch_size]))
        # inputs.append(("joint_attention_kwargs", None, None))
        # inputs.append(("return_dict", bool, False))

        hidden_states = torch.ones(shape, dtype=torch.float32)
        encoder_hidden_states = torch.ones([batch_size, 333, 4096], dtype=torch.float32)
        pooled_projections = torch.ones([batch_size, 2048], dtype=torch.float32)
        timestep = torch.ones([batch_size], dtype=torch.float32)

        export_model.eval()
        export_model = export_model.to(torch.float32)

        onnx_input_names = [
            "hidden_states",
            "encoder_hidden_states",
            "pooled_projections",
            "timestep",
        ]
        onnx_output_names = [
            "sample_hidden_states",
        ]

        logger.info(f"exporting model {onnx_name} to onnx")
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_onnx_file = str(Path(tmp_dir) / Path(output_onnx_file).name)
            torch.onnx.export(
                export_model,
                (hidden_states, encoder_hidden_states, pooled_projections, timestep, None, None, False),
                tmp_onnx_file,
                input_names=onnx_input_names,
                export_params=True,
                output_names=onnx_output_names,
                do_constant_folding=True,
                opset_version=18,
                # keep_initializers_as_inputs = True,
                verbose=False,
            )
            onnx_model = onnx.load(tmp_onnx_file)
        from xhquant.utils.onnxsim_large_model.simplify_large_onnx import simplify_large_onnx

        model_opt, check_ok = simplify_large_onnx(
            onnx_model,
            skipped_optimizers=[
                "fuse_pad_into_conv",
                "fuse_consecutive_slices",
                "eliminate_common_subexpression",
                "fuse_qkv",
            ],
        )
        if check_ok:
            onnx_model = model_opt
        else:
            logger.error(f"onnx model simplify failed: {output_onnx_file}")

        onnx.save(
            onnx_model,
            output_onnx_file,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=f"{Path(output_onnx_file).stem}_external_data",
        )
        logger.info(f"save onnx to file {output_onnx_file}")

    @classmethod
    def export_onnx_clip_l(cls, hf_model: StableDiffusion3Pipeline, convert_config: SD3ConvertConfig, output_onnx_file):
        logger = get_root_logger()
        export_model = hf_model.text_encoder_2
        export_model.to(device="cpu", dtype=torch.float32)
        fuse_clip_l = True
        legacy_onnx = True
        simplify_onnx = True
        onnx_name = "clip_l"
        if fuse_clip_l:
            onnx_name = onnx_name + "_fused"

        def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
        ):
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            text_outputs = self.text_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            # nn.GELU()
            pooled_output = text_outputs[1]

            text_embeds = self.text_projection(pooled_output)

            if not return_dict:
                outputs = (text_embeds, text_outputs[0]) + text_outputs[2:]
                return tuple(output for output in outputs if output is not None)

            return text_embeds, text_outputs.hidden_states[-2]

        export_model.forward = types.MethodType(forward, export_model)

        for name, module in export_model.named_modules():
            if isinstance(module, CLIPTextTransformer):
                module.forward = types.MethodType(CLIPTextTransformer_forward, module)
            if isinstance(module, CLIPSdpaAttention):
                module.forward = types.MethodType(CLIPSdpaAttention_forward, module)
            if fuse_clip_l:
                if isinstance(module, Linear4bit):
                    module.forward = types.MethodType(linear4bit_forward, module)

        with torch.no_grad():
            export_model.eval()
            onnx_inputs_name = ["input_ids", "output_hidden_states"]
            onnx_args = (torch.randint(1000, (1, 77), dtype=torch.int32),)
            onnx_kwargs = dict(output_hidden_states=True)

            if Path(output_onnx_file).exists():
                pass
            else:
                import math

                from transformers.activations import GELUActivation

                def _gelu_python(self, input):
                    return input * (1.0 + torch.erf(input / math.sqrt(2.0))) * 0.5
                    # return input * 0.5 * (1.0 + torch.erf(input / math.sqrt(2.0)))

                for name, module in export_model.named_modules():
                    if isinstance(module, GELUActivation):
                        module.act = types.MethodType(_gelu_python, module)

                logger.info(f"exporting model {onnx_name} to onnx")

                with tempfile.TemporaryDirectory() as tmpdir:
                    tmp_onnx_file = str(Path(tmpdir) / Path(output_onnx_file).name)

                    if legacy_onnx:
                        torch.onnx.export(
                            export_model,
                            onnx_args + (onnx_kwargs,),
                            tmp_onnx_file,
                            input_names=onnx_inputs_name,
                            export_params=True,
                            output_names=["prompt_embeds", "pooled_prompt_embeds"],
                            do_constant_folding=True,
                            opset_version=18,
                            # keep_initializers_as_inputs = True,
                            verbose=False,
                        )
                    else:
                        export_program = torch.export.export(
                            export_model,
                            onnx_args,
                            onnx_kwargs,
                        )
                        torch.onnx.dynamo_export(
                            export_program.module(),
                            *onnx_args,
                            **onnx_kwargs,
                        ).save(tmp_onnx_file)
                    onnx_model = onnx.load(tmp_onnx_file, load_external_data=True)

            for i, input in enumerate(onnx_model.graph.input):
                print(f"input[{i}]: {input.name}")
            for i, output in enumerate(onnx_model.graph.output):
                print(f"output[{i}]: {output.name}")

            if simplify_onnx:
                from xhquant.utils.onnxsim_large_model.simplify_large_onnx import simplify_large_onnx

                model_opt, check_ok = simplify_large_onnx(
                    onnx_model,
                    skipped_optimizers=[
                        "fuse_pad_into_conv",
                        "fuse_consecutive_slices",
                        "eliminate_common_subexpression",
                        "fuse_qkv",
                    ],
                    const2init=False,
                )
                if check_ok:
                    onnx_model = model_opt
                # model_opt, check_ok = onnxsim.simplify(
                #     onnx_model,
                #     skipped_optimizers=[
                #         "fuse_pad_into_conv",
                #         "fuse_consecutive_slices",
                #         "eliminate_common_subexpression",
                #         "fuse_qkv",
                #     ],
                # )
                if check_ok:
                    onnx_model = model_opt

            onnx.save(
                onnx_model,
                output_onnx_file,
                save_as_external_data=True,
                all_tensors_to_one_file=True,
                location=f"{Path(output_onnx_file).stem}_external_data",
            )
            logger.info(f"save onnx to file {output_onnx_file}")

    @classmethod
    def export_onnx_clip(cls, hf_model: StableDiffusion3Pipeline, convert_config: SD3ConvertConfig, output_onnx_file):
        logger = get_root_logger()
        export_model = hf_model.text_encoder
        export_model.to(device="cpu", dtype=torch.float32)

        onnx_name = Path(output_onnx_file).stem
        legacy_onnx = True
        fuse_clip = True
        simplify_onnx = True

        def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
        ):
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            text_outputs = self.text_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            pooled_output = text_outputs[1]

            text_embeds = self.text_projection(pooled_output)

            if not return_dict:
                outputs = (text_embeds, text_outputs[0]) + text_outputs[2:]
                return tuple(output for output in outputs if output is not None)

            return text_embeds, text_outputs.hidden_states[-2]

        export_model.forward = types.MethodType(forward, export_model)

        for name, module in export_model.named_modules():
            if isinstance(module, CLIPTextTransformer):
                module.forward = types.MethodType(CLIPTextTransformer_forward, module)
            if isinstance(module, CLIPSdpaAttention):
                module.forward = types.MethodType(CLIPSdpaAttention_forward, module)
            if fuse_clip:
                if isinstance(module, Linear4bit):
                    module.forward = types.MethodType(linear4bit_forward, module)

        with torch.no_grad():
            export_model.eval()
            onnx_inputs_name = ["input_ids", "output_hidden_states"]
            onnx_args = (torch.randint(1000, (1, 77), dtype=torch.int32),)
            onnx_kwargs = dict(output_hidden_states=True)

            if Path(output_onnx_file).exists():
                pass
            else:
                logger.info(f"exporting model {onnx_name} to onnx")

                with tempfile.TemporaryDirectory() as tmpdirname:
                    tmp_onnx_file = str(Path(tmpdirname) / Path(output_onnx_file).name)
                    if legacy_onnx:
                        torch.onnx.export(
                            export_model,
                            onnx_args + (onnx_kwargs,),
                            tmp_onnx_file,
                            input_names=onnx_inputs_name,
                            export_params=True,
                            output_names=["prompt_embeds", "pooled_prompt_embeds"],
                            do_constant_folding=True,
                            opset_version=18,
                            # keep_initializers_as_inputs = True,
                            verbose=False,
                        )
                    else:
                        export_program = torch.export.export(
                            export_model,
                            onnx_args,
                            onnx_kwargs,
                        )
                        torch.onnx.dynamo_export(
                            export_program.module(),
                            *onnx_args,
                            **onnx_kwargs,
                        ).save(tmp_onnx_file)
                    onnx_model = onnx.load(tmp_onnx_file, load_external_data=True)

            for i, input in enumerate(onnx_model.graph.input):
                logger.info(f"input[{i}]: {input.name}")
            for i, output in enumerate(onnx_model.graph.output):
                logger.info(f"output[{i}]: {output.name}")

            if simplify_onnx:
                model_opt, check_ok = onnxsim.simplify(
                    onnx_model,
                    skipped_optimizers=[
                        "fuse_pad_into_conv",
                        "fuse_consecutive_slices",
                        "eliminate_common_subexpression",
                        "fuse_qkv",
                    ],
                )

                if check_ok:
                    onnx_model = model_opt
            onnx.save(
                onnx_model,
                output_onnx_file,
                save_as_external_data=True,
                all_tensors_to_one_file=True,
                location=f"{Path(output_onnx_file).stem}_external_data",
            )
            logger.info(f"save onnx to file {output_onnx_file}")

    @classmethod
    def export_onnx_vae(cls, hf_model, convert_config: SD3ConvertConfig, output_onnx_file):
        logger = get_root_logger()
        legacy_onnx = True
        simplify_onnx = True
        export_model = hf_model.vae.decoder
        export_model.to(device="cpu", dtype=torch.float32)
        shape = [1, 16, int(convert_config.height / 8), int(convert_config.width / 8)]
        onnx_name = Path(output_onnx_file).stem

        with torch.no_grad():
            export_model.eval()
            onnx_input = torch.rand(shape, dtype=torch.float32)
            if Path(output_onnx_file).exists():
                pass
            else:
                logger.info(f"exporting model {onnx_name} to onnx")
                if legacy_onnx:
                    torch.onnx.export(
                        export_model,
                        onnx_input,
                        output_onnx_file,
                        input_names=["input"],
                        export_params=True,
                        # output_names=["pooled_prompt_embeds", "prompt_embeds"],
                        do_constant_folding=True,
                        opset_version=18,
                        # keep_initializers_as_inputs = True,
                        verbose=False,
                    )
                else:
                    export_program = torch.export.export(
                        export_model,
                        (onnx_input,),
                    )

                    torch.onnx.dynamo_export(
                        export_program.module(),
                        onnx_input,
                    ).save(output_onnx_file)

                logger.info(f"export onnx to file {output_onnx_file}")

            onnx_model = onnx.load(output_onnx_file)

            if simplify_onnx:
                model_opt, check_ok = onnxsim.simplify(
                    onnx_model,
                    skipped_optimizers=[
                        "fuse_pad_into_conv",
                        "fuse_consecutive_slices",
                        "eliminate_common_subexpression",
                        "fuse_qkv",
                    ],
                )
                if check_ok:
                    onnx_model = model_opt
            onnx.save(
                onnx_model,
                output_onnx_file,
                save_as_external_data=True,
                all_tensors_to_one_file=True,
                location=f"{Path(output_onnx_file).stem}_external_data",
            )
            logger.info(f"save onnx to file {output_onnx_file}")

    @classmethod
    def export_onnx_t5(cls, hf_model: StableDiffusion3Pipeline, convert_config: SD3ConvertConfig, output_onnx_file):
        logger = get_root_logger()
        no_clip_fp16_t5 = convert_config.no_clip_fp16_t5
        custom_patch_t5 = convert_config.custom_patch_t5
        legacy_onnx = True
        if convert_config.hadmard_t5:
            logger.info("hadmard t5................")
            hf_model.text_encoder_3 = hadmard_t5(hf_model.text_encoder_3)
            no_clip_fp16_t5 = True
        export_model = hf_model.text_encoder_3
        # mode_dtype = export_model.dtype
        if export_model is None:
            return

        onnx_name = Path(output_onnx_file).stem
        # if args.t5_quant:
        #     if args.te_mode_t5 in [0, "sefp"]:
        #         onnx_name = "t5-dequant_0"
        #     else:
        #         onnx_name = "t5-dequant_2"

        # if args.no_clip_fp16_t5:
        #     onnx_name = onnx_name + "_no_clip_fp16"

        # if args.custom_patch_t5:
        #     onnx_name = onnx_name + "_custom_patch"

        def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
        ):
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            return (encoder_outputs[0].half(),)

        export_model.forward = types.MethodType(forward, export_model)

        if not no_clip_fp16_t5:
            for name, module in export_model.named_modules():
                if isinstance(module, T5Block):
                    module.forward = types.MethodType(T5Block_forward, module)

        if custom_patch_t5:
            for name, module in export_model.named_modules():
                if isinstance(module, T5Stack):
                    for i, target in enumerate(module.block):
                        target.add_module("clip_1", Clip())
                        target.add_module("clip_2", Clip())
                        if i >= 10:
                            target.forward = types.MethodType(T5Block_forward_1_2, target)
                        else:
                            target.forward = types.MethodType(T5Block_forward_1_1, target)

        inputs = [("input_ids", torch.int32, [1, 256])]

        with torch.no_grad():
            export_model.eval()
            export_model = export_model.to(torch.float32)
            onnx_inputs_name = list()
            onnx_args: Tuple[Any] = tuple()
            onnx_kwargs = dict()
            for i, ip in enumerate(inputs):
                name, dtype, value = ip
                onnx_inputs_name.append(name)
                if isinstance(value, list):
                    value = torch.ones(value, dtype=dtype)
                    onnx_args = onnx_args + (value,)
                else:
                    keywords = dict()
                    keywords[name] = value
                    onnx_kwargs[name] = value

            from transformers.activations import NewGELUActivation
            from transformers.models.t5.modeling_t5 import T5LayerNorm
            from xhquant.nn.modules import RMSNorm

            # if args.t5_quant:
            #     from qlinear_cuda_old_non_zero import QuantLinear_non_zero
            #     for name, module in export_model.named_modules():
            #         if isinstance(module, QuantLinear_non_zero):
            #             module.save_weight = True
            #             if args.te_mode_t5 in [0, "sefp"]:
            #                 module.quant_type = 0
            #             else:
            #                 module.quant_type = 2
            # onnx_outputs = export_model(*onnx_args, **onnx_kwargs)
            # if args.t5_quant:
            #     from qlinear_cuda_old_non_zero import QuantLinear_non_zero
            #     def qlinear_forward(self, x):
            #         val = torch.nn.functional.linear(x, self.dequant_weight.to(x.device), self.bias)
            #         return val
            #     for name, module in export_model.named_modules():
            #         if isinstance(module, QuantLinear_non_zero):
            #             module.forward = types.MethodType(qlinear_forward, module)
            #             module.qweight = None
            #             module.qzeros = None
            #             module.scales = None
            #             module.g_idx = None
            #             module.wf = None
            #             update = nn.Linear(
            #                 in_features=module.infeatures,
            #                 out_features=module.outfeatures,
            #                 bias=module.bias is not None,
            #                 device=device,
            #                 dtype=torch.float32,
            #             )
            #             update.weight.data = module.dequant_weight
            #             if module.bias is not None:
            #                 update.bias.data = module.bias
            #             if hasattr(module, "quant_weight") and module.quant_weight is not None:
            #                 setattr(update, "quant_weight", module.quant_weight.reshape(module.dequant_weight.shape))
            #             update_submodule(export_model, name, update)
            # for name, module in export_model.named_modules():
            #     if isinstance(module, NewGELUActivation):
            #         update = nn.GELU(approximate="tanh")
            #         update.forward = types.MethodType(_gelu_tanh, update)
            #         update_submodule(export_model, name, update)
            #     if args.fx_trace_t5:
            #         if isinstance(module, T5LayerNorm):
            #             hidden_size = module.weight.shape[-1]
            #             eps = module.variance_epsilon
            #             update = RMSNorm(hidden_size=hidden_size, eps=eps)
            #             update = update.to(device)
            #             if module.weight is not None:
            #                 update.weight.data = module.weight
            #             update_submodule(export_model, name, update)
            # onnx_outputs_ = export_model(*onnx_args, **onnx_kwargs)
            # diff = (onnx_outputs[0] - onnx_outputs_[0]).sum().item()
            # print("diff:", diff)
            # onnx_outputs_ = None

            if Path(output_onnx_file).exists():
                pass
            else:
                logger.info(f"exporting model {onnx_name} to onnx")
                onnx_args = (onnx_args[0].to("cpu"),)
                with tempfile.TemporaryDirectory() as tmp_dir:
                    tmp_out_onnx_file = str(Path(tmp_dir) / f"{onnx_name}.onnx")
                    if legacy_onnx:
                        torch.onnx.export(
                            export_model.to("cpu"),
                            onnx_args + (onnx_kwargs,),
                            tmp_out_onnx_file,
                            input_names=onnx_inputs_name,
                            export_params=True,
                            output_names=["prompt_embeds"],
                            do_constant_folding=True,
                            opset_version=18,
                            # keep_initializers_as_inputs = True,
                            verbose=False,
                        )
                    else:
                        export_program = torch.export.export(
                            export_model,
                            onnx_args,
                            onnx_kwargs,
                        )

                        torch.onnx.dynamo_export(
                            export_program.module(),
                            *onnx_args,
                            **onnx_kwargs,
                        ).save(tmp_out_onnx_file)
                    onnx_model = onnx.load(tmp_out_onnx_file)

                from xhquant.utils.onnxsim_large_model.simplify_large_onnx import simplify_large_onnx

                model_opt, check_ok = simplify_large_onnx(
                    onnx_model,
                    skipped_optimizers=[
                        "fuse_pad_into_conv",
                        "fuse_consecutive_slices",
                        "eliminate_common_subexpression",
                        "fuse_qkv",
                    ],
                )

                if check_ok:
                    onnx_model = model_opt
                onnx.save(
                    onnx_model,
                    output_onnx_file,
                    save_as_external_data=True,
                    all_tensors_to_one_file=True,
                    location=f"{Path(output_onnx_file).stem}_external_data",
                )
                logger.info(f"Save onnx to file {output_onnx_file}")

                del onnx_model
                del model_opt

    def load_model(self, pretrained_model_path, convert_config: SD3ConvertConfig):
        pipe = StableDiffusion3Pipeline.from_pretrained(
            pretrained_model_path,
            torch_dtype=torch.float16,
        )
        return pipe

    def _convert(self, work_dir: str):
        logger = get_root_logger()
        pretrained_model_path = self.pretrained_model_path
        convert_config = self.convert_config
        model_name = Path(pretrained_model_path).name
        pipe = self.load_model(pretrained_model_path, convert_config)
        onnx_dir = str(Path(work_dir) / "onnx")
        Path(onnx_dir).mkdir(exist_ok=True, parents=True)

        hmonnx_dir = str(Path(work_dir) / "hmonnx")
        Path(hmonnx_dir).mkdir(exist_ok=True, parents=True)

        ## export mmdit
        mmdit_onnx_file = str(Path(onnx_dir) / f"mmdit_{convert_config.width}x{convert_config.height}.onnx")

        if Path(mmdit_onnx_file).exists():
            logger.warning(f"{mmdit_onnx_file} already exists")
        else:
            SD3Converter.export_onnx_mmdit(pipe, convert_config, mmdit_onnx_file)
        target_device = convert_config.quant_scheme.target_device

        mmdit_input = SD3Converter.get_mmdit_dummy_inputs(convert_config)
        mmdit_hmonnx_file = str(Path(hmonnx_dir) / f"{Path(mmdit_onnx_file).stem}_{target_device.name}.onnx")
        if Path(mmdit_hmonnx_file).exists():
            logger.warning(f"{mmdit_hmonnx_file} already exists")
        else:
            convert_onnx_to_hmonnx(
                mmdit_onnx_file,
                mmdit_input,
                target_device,
                mmdit_hmonnx_file,
                input_names=["hidden_states", "encoder_hidden_states", "pooled_projections", "timestep"],
                output_names=["sample_hidden_states"],
            )

        ## export clip_l
        clip_l_onnx_file = str(Path(onnx_dir) / f"clip_l.onnx")
        if Path(clip_l_onnx_file).exists():
            logger.warning(f"{clip_l_onnx_file} already exists")
        else:
            SD3Converter.export_onnx_clip_l(pipe, convert_config, clip_l_onnx_file)

        clip_l_inputs = [torch.randint(0, 10000, (1, 77), dtype=torch.int32)]
        clip_l_hmonnx_file = str(Path(hmonnx_dir) / f"clip_l_{target_device.name}.onnx")
        if Path(clip_l_hmonnx_file).exists():
            logger.warning(f"{clip_l_hmonnx_file} already exists")
        else:
            convert_onnx_to_hmonnx(
                clip_l_onnx_file,
                clip_l_inputs,
                target_device,
                clip_l_hmonnx_file,
                input_names=["input_ids"],
                output_names=["prompt_embeds", "pooled_prompt_embeds"],
            )

        ## export clip
        clip_onnx_file = str(Path(onnx_dir) / f"clip.onnx")
        if Path(clip_onnx_file).exists():
            logger.warning(f"{clip_onnx_file} already exists")
        else:
            SD3Converter.export_onnx_clip(pipe, convert_config, clip_onnx_file)

        clip_inputs = [torch.randint(0, 10000, (1, 77), dtype=torch.int32)]
        clip_hmonnx_file = str(Path(hmonnx_dir) / f"clip_{target_device.name}.onnx")
        if Path(clip_hmonnx_file).exists():
            logger.warning(f"{clip_hmonnx_file} already exists")
        else:
            convert_onnx_to_hmonnx(
                clip_onnx_file,
                clip_inputs,
                target_device,
                clip_hmonnx_file,
                input_names=["input_ids"],
                output_names=["prompt_embeds", "pooled_prompt_embeds"],
            )

        ## export vae
        vae_onnx_file = str(Path(onnx_dir) / f"vae.onnx")
        if Path(vae_onnx_file).exists():
            logger.warning(f"{vae_onnx_file} already exists")
        else:
            SD3Converter.export_onnx_vae(pipe, convert_config, vae_onnx_file)

        vae_input = [
            torch.randn((1, 16, int(convert_config.height / 8), int(convert_config.width / 8)), dtype=torch.float32),
        ]
        vae_hmonnx_file = str(Path(hmonnx_dir) / f"vae_{target_device.name}.onnx")
        if Path(vae_hmonnx_file).exists():
            logger.warning(f"{vae_hmonnx_file} already exists")
        else:
            convert_onnx_to_hmonnx(
                vae_onnx_file,
                vae_input,
                target_device,
                vae_hmonnx_file,
                input_names=["latent_sample"],
                output_names=["sample"],
            )

        ## export t5
        t5_onnx_file = str(Path(onnx_dir) / f"t5.onnx")
        if Path(t5_onnx_file).exists():
            logger.warning(f"{t5_onnx_file} already exists")
        else:
            SD3Converter.export_onnx_t5(pipe, convert_config, t5_onnx_file)

        t5_inputs = [torch.ones((1, 256), dtype=torch.int32)]

        t5_hmonnx_file = str(Path(hmonnx_dir) / f"t5_{target_device.name}.onnx")
        if Path(t5_hmonnx_file).exists():
            logger.warning(f"{t5_hmonnx_file} already exists")
        else:
            convert_onnx_to_hmonnx(
                t5_onnx_file,
                t5_inputs,
                target_device,
                t5_hmonnx_file,
                input_names=["input_ids"],
                output_names=["prompt_embeds"],
            )

        meta_info: Dict[str, Any] = {
            "create_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "hf_model_path": pretrained_model_path,
            "model_name": model_name,
            "target_device": target_device.name,
            "height": convert_config.height,
            "width": convert_config.width,
            "guidance_scale": convert_config.guidance_scale,
            "num_inference_steps": convert_config.num_inference_steps,
            "mmdit_hmonnx": str(Path(mmdit_hmonnx_file).relative_to(work_dir)),
            "clip_l_hmonnx": str(Path(clip_l_hmonnx_file).relative_to(work_dir)),
            "clip_hmonnx": str(Path(clip_hmonnx_file).relative_to(work_dir)),
            "vae_hmonnx": str(Path(vae_hmonnx_file).relative_to(work_dir)),
            "t5_hmonnx": str(Path(t5_hmonnx_file).relative_to(work_dir)),
        }

        with open(Path(work_dir) / "meta.json", "w") as f:
            json.dump(meta_info, f, indent=4)

        logger.info(f"save meta info to {Path(work_dir) / 'meta.json'}")

    @classmethod
    def from_pretrained(cls, pretrained_model_path: str, convert_config: SD3ConvertConfig, work_dir: str, **kwargs):
        assert digit_version(transformers.__version__) == digit_version("4.47.0"), "transformers version must be 4.47.0"
        converter = SD3Converter(pretrained_model_path, convert_config)
        converter._convert(work_dir)
