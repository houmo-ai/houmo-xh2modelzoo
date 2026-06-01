import math
import sys
import types
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import xhquant.nn as xhnn
from torch import Tensor
from transformers import Cache
from transformers.modeling_outputs import BaseModelOutputWithPast,BaseModelOutput
# from transformers.models.gemma.modeling_gemma import (
#     GemmaAttention,
#     GemmaDecoderLayer,
#     GemmaForCausalLM,
#     GemmaRMSNorm,
#     GemmaModel,
# )
from lerobot.policies.xvla.modeling_florence2 import (
    Florence2Encoder,
    Florence2Attention,
    Florence2EncoderLayer,
    Florence2LearnedPositionalEmbedding,
)
from xhquant.api import ConfigDict
from xhquant.nn import LLMCache, MaskedSoftmax, RMSNorm, Rope
from xhquant.utils.registry import DynamicModule
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask
from ..builder import XHLLM_TRACEABLE_MODULES

@XHLLM_TRACEABLE_MODULES.register_module(
    {
        Florence2LearnedPositionalEmbedding: "Florence2LearnedPositionalEmbedding",
    }
)
class _Florence2LearnedPositionalEmbedding(DynamicModule):
    def _setup(self, cfg):
        # self.init_pos = torch.zeros((1, self.offset))-1
        input_seq_len = cfg.input_sequence_length  # type: ignore
        self.weight_slice = xhnn.DynamicSlice([input_seq_len], [0], [1])

        def _sin_cos_slice_update_cfg(self, cfg: Optional[Dict] = None):
            input_seq_len = cfg.input_sequence_length
            self.valid_length = [input_seq_len]

        self.weight_slice._update_cfg = types.MethodType(_sin_cos_slice_update_cfg, self.weight_slice)
        return self

    def forward(self, input_ids: torch.Tensor, past_key_values_length: int = 0):
        # bsz, seq_len = input_ids.shape[:2]
        # max_pos = self.weight.shape[0]
        # positions = self.init_pos + past_key_values_length + self.offset
        # positions = torch.arange(max_pos, device=self.weight.device)
        # positions = positions[past_key_values_length : past_key_values_length + seq_len]
        valid_len = torch.tensor(past_key_values_length+self.offset).cuda().unsqueeze(-1)
        return self.weight_slice(self.weight,valid_len)

@XHLLM_TRACEABLE_MODULES.register_module(
    {
        Florence2Attention: "Florence2Attention",
    }
)
class _Florence2Attention(DynamicModule):
    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()


    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: torch.Tensor | None = None,
        past_key_value: tuple[torch.Tensor] | None = None,
        # attention_mask: torch.Tensor | None = None,
        layer_head_mask: torch.Tensor | None = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
        bsz, tgt_len, _ = hidden_states.size()
        bsz = 1
        query_states = self.q_proj(hidden_states) * self.scaling
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.reshape(*proj_shape)
        value_states = value_states.reshape(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.matmul(query_states, key_states.transpose(1, 2))
        # attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
        attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        # attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned across GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, None, past_key_value
        

    def _setup(self, cfg: Union[ConfigDict, Dict[str, Any]]):
        if isinstance(cfg, dict):
            cfg = ConfigDict(cfg)
        # if not hasattr(self, "num_key_value_heads"):
        #     self.num_key_value_heads = self.config.encoder_attention_heads
        # if not hasattr(self, "num_heads"):
        #     self.num_heads = self.config.num_attention_heads  # 28
        # if not hasattr(self, "hidden_size"):
        #     self.hidden_size = self.config.hidden_size  # 3584
        self.num_heads = self.config.encoder_attention_heads
        self.embed_dim = self.config.d_model
        self.head_dim = self.embed_dim // self.num_heads
        # self.enable_rope = cfg.get("enable_rope", True)
        # if self.enable_rope:
        #     self.rope = xhnn.Rope()

        self.slice_1 = xhnn.Slice([0], [self.head_dim // 2], [3], [1])
        # self.slice_2 = xhnn.Slice([self.head_dim // 2], [sys.maxsize], [3], [1])

        attention_max_length = -1

        # self.masked_softmax = MaskedSoftmax(dim=-1, attention_max_length=attention_max_length)
        use_cache = cfg.use_cache
        self.use_cache = use_cache
        self.key_extra_scale = 1.0 if "key_extra_scale" not in cfg else cfg.key_extra_scale
        self.query_extra_scale = 1.0 if "query_extra_scale" not in cfg else cfg.query_extra_scale
        # max_sequence_length = cfg.max_sequence_length
        # self.max_sequence_length = max_sequence_length
        # input_seq_len = cfg.input_sequence_length
        # self.cos_gather = xhnn.Gather(0)
        # self.sin_gather = xhnn.Gather(0)

        # self.rope = Rope()

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
        Florence2EncoderLayer: "Florence2EncoderLayer",
    }
)
class _Florence2EncoderLayer(DynamicModule):
    def _setup(self, cfg: Optional[Dict] = None):
        if cfg is None:
            cfg = ConfigDict()
        # assert cfg is not None
        # self.use_clip = cfg.get("use_clip", False)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        # attention_mask: torch.FloatTensor,
        layer_head_mask: torch.FloatTensor,
        output_attentions: bool | None = False,
    ) -> tuple[torch.FloatTensor, torch.FloatTensor | None]:
        
        residual = hidden_states
        hidden_states, attn_weights, _ = self.self_attn(
            hidden_states=hidden_states,
            #attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        # hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        # hidden_states = nn.functional.dropout(
        #     hidden_states, p=self.activation_dropout, training=self.training
        # )
        hidden_states = self.fc2(hidden_states)
        # hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        # if hidden_states.dtype == torch.float16 and (
        #     torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        # ):
        clamp_value = torch.finfo(torch.float16).max - 1000
        hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


@XHLLM_TRACEABLE_MODULES.register_module(
    {
        Florence2Encoder: "Florence2Encoder",
    }
)
class _Florence2Encoder(DynamicModule):
    def _setup(self, cfg: Optional[Dict]):
        self.only_first_block = cfg.get("only_first_block", False)
        # self._setup_position_cached()

        self.num_logits_to_keep = cfg.num_logits_to_keep  # type: ignore # 1,取最后一个token的输出，0,取所有token的输出
        assert self.num_logits_to_keep in [0, 1]

        self.use_cache = cfg.use_cache
        # input_seq_len = cfg.input_sequence_length  # type: ignore
        # self.sin_slice = xhnn.DynamicSlice([input_seq_len], [2], [1])
        # self.cos_slice = xhnn.DynamicSlice([input_seq_len], [2], [1])

        # def _sin_cos_slice_update_cfg(self, cfg: Optional[Dict] = None):
        #     input_seq_len = cfg.input_sequence_length
        #     self.valid_length = [input_seq_len]

        # self.sin_slice._update_cfg = types.MethodType(_sin_cos_slice_update_cfg, self.sin_slice)
        # self.cos_slice._update_cfg = types.MethodType(_sin_cos_slice_update_cfg, self.cos_slice)

        # self.llm_gather = xhnn.BatchGather(1)
        # self.llm_gather.update_offset_indices(1, input_seq_len)

        # def _llm_gather_update_cfg(self: xhnn.BatchGather, cfg: Optional[Dict] = None):
        #     input_seq_len = cfg.input_sequence_length  # type: ignore
        #     batch_size = cfg.get("batch_size", 1)  # type: ignore
        #     self.update_offset_indices(batch_size, input_seq_len)

        # self.llm_gather._update_cfg = types.MethodType(_llm_gather_update_cfg, self.llm_gather)

    def forward(
        self,
        # input_ids: torch.LongTensor = None,
        # attention_mask: torch.Tensor | None = None,
        # head_mask: torch.Tensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        # output_attentions: bool | None = None,
        # output_hidden_states: bool | None = None,
        # return_dict: bool | None = None,
    ) -> tuple | BaseModelOutput:

        output_attentions = False
        # output_hidden_states =False
        # return_dict = True

        # retrieve input_ids and inputs_embeds
        # if input_ids is not None and inputs_embeds is not None:
        #     raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        # elif input_ids is not None:
        #     input = input_ids
        #     input_ids = input_ids.view(-1, input_ids.shape[-1])
        # elif inputs_embeds is not None:
        # input = inputs_embeds[:, :, -1]
        # else:
        #     raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
            input_tensor = input_ids
        else:
            input_tensor = inputs_embeds

        embed_pos = self.embed_positions(input_ids=input_tensor)
        # embed_pos = embed_pos.to(inputs_embeds.device)

        hidden_states = inputs_embeds + embed_pos
        hidden_states = self.layernorm_embedding(hidden_states)
        # hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        # attention_mask = _prepare_4d_attention_mask(attention_mask, inputs_embeds.dtype)
        # attention_mask = 0
        encoder_states =  None
        all_attentions = None

        for idx, encoder_layer in enumerate(self.layers):
            layer_outputs = encoder_layer(
                hidden_states,
                # attention_mask,
                layer_head_mask= None,
                output_attentions=output_attentions,
            )

            hidden_states = layer_outputs[0]

        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )



def register_wrap_cls(hf_model):
    pass
