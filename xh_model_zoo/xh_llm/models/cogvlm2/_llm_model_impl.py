import math
import sys
import types
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from transformers.modeling_outputs import BaseModelOutputWithPast
from xhquant import nn as xhnn
from xhquant.nn import LLMCache, RMSNorm

from ..builder import XHLLM_TRACEABLE_MODULES, DynamicRegister

DType = torch.dtype

LANGUAGE_TOKEN_TYPE = 0
VISION_TOKEN_TYPE = 1


@XHLLM_TRACEABLE_MODULES.register_module(
    {
        "transformers_modules.cogvlm2-llama3-chat-19B.modeling_cogvlm.VisionExpertMLP": "cogvlm2.VisionExpertMLP",
    }
)
class _VisionExpertMLP(DynamicRegister):
    def _setup(self, cfg: Optional[Dict] = None):
        self.vision_token_num = 2306
        self.is_vision_prefill = True
        # self.slice_bos = xhnn.Slice([0], [1], [1], [1])
        # self.slice_vision = xhnn.Slice([1], [self.vision_token_num], [1], [1])
        # self.slice_prompt = xhnn.Slice([self.vision_token_num], [sys.maxsize], [1], [1])

        # self.slice_vision_reorder = xhnn.Slice([0], [self.vision_token_num - 1], [1], [1])
        # self.slice_language = xhnn.Slice([self.vision_token_num - 1], [sys.maxsize], [1], [1])

        return self

    def forward(self, hidden_states: "torch.Tensor(B, L, D)"):
        if self.is_vision_prefill:
            output = self.vision_mlp(hidden_states)
        else:
            output = self.language_mlp(hidden_states)

        return output


class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = self._compute_inv_freq(device)
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len_cached = 0

    def _compute_inv_freq(self, device=None):
        return 1.0 / (self.base ** (torch.arange(0, self.dim, 2, device=device) / self.dim))

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[:, None, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[:, None, :].to(dtype), persistent=False)

    def forward(self, x, seq_len):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:seq_len, ...].to(dtype=x.dtype),
        )


def _make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


@XHLLM_TRACEABLE_MODULES.register_module(
    {
        "transformers_modules.cogvlm2-llama3-chat-19B.modeling_cogvlm.VisionExpertAttention": "cogvlm2.VisionExpertAttention",
    }
)
class _VisionExpertAttention(DynamicRegister):
    def _update_cfg(self, cfg):
        self.input_sequence_length = cfg.input_sequence_length

    def rotate_half(self, x: Tensor):
        """Rotates half the hidden dims of the input."""
        # x1 = x[..., : x.shape[-1] // 2]
        # x2 = x[..., x.shape[-1] // 2 :]
        # x1 = torch_ops_xh2a_slice(x, [0], [self.head_dim // 2], [3], [1])
        # x2 = torch_ops_xh2a_slice(x, [self.head_dim // 2], [sys.maxsize], [3], [1])
        x1 = self.slice_1(x)
        x2 = self.slice_2(x)
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb_index_bhs(self, q, k, cos, sin, position_id):
        # batch_size, num_head, seq_len, hidden_size
        # cos, sin = F.embedding(position_id, cos.squeeze(1)).unsqueeze(1), F.embedding(
        #     position_id, sin.squeeze(1)
        # ).unsqueeze(1)
        # q, k = (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)

        # cos = F.embedding(position_id, cos.squeeze(1)).unsqueeze(1)
        # sin = F.embedding(position_id, sin.squeeze(1)).unsqueeze(1)
        cos = self.cos_embeding(position_id).unsqueeze(1)
        sin = self.sin_embeding(position_id).unsqueeze(1)

        q = (q * cos) + (self.rotate_half(q) * sin)
        k = (k * cos) + (self.rotate_half(k) * sin)
        return q, k

    def _setup(self, cfg: Optional[Dict] = None):
        # self.is_prefill = False
        self.is_vision_prefill = False
        self.vision_token_num = 2306
        self.use_cache = cfg.use_cache
        self.input_sequence_length = cfg.input_sequence_length
        max_sequence_length = cfg.max_sequence_length
        self.max_sequence_length = cfg.max_sequence_length
        self.rotary_emb = RotaryEmbedding(
            dim=self.head_dim, base=500000, max_position_embeddings=cfg.max_sequence_length
        )
        self.rotary_emb._set_cos_sin_cache(seq_len=max_sequence_length, device=torch.device("cpu"), dtype=torch.float16)

        cos = self.rotary_emb.cos_cached.squeeze(1)
        sin = self.rotary_emb.sin_cached.squeeze(1)
        num_embeddings, embedding_dim = cos.shape

        self.cos_embeding = nn.Embedding(num_embeddings, embedding_dim)
        self.cos_embeding.weight.data = cos

        self.sin_embeding = nn.Embedding(num_embeddings, embedding_dim)
        self.sin_embeding.weight.data = sin

        # self.slice_bos = xhnn.Slice([0], [1], [1], [1])
        # self.slice_vision = xhnn.Slice([1], [self.vision_token_num], [1], [1])
        # self.slice_prompt = xhnn.Slice([self.vision_token_num], [sys.maxsize], [1], [1])

        # self.slice_vision_reorder = xhnn.Slice([0], [self.vision_token_num - 1], [1], [1])
        # self.slice_language = xhnn.Slice([self.vision_token_num - 1], [sys.maxsize], [1], [1])

        # self.slice_feature_prompt = xhnn.Slice([1], [sys.maxsize], [1], [1])

        # attention_mask = _make_causal_mask(input_ids_shape=(1, max_sequence_length), dtype=torch.float16, device="cpu")
        # # hidden_states重排成[vision, bos, prompt]，需要调整attention_mask
        # attention_mask = torch.cat(
        #     [
        #         attention_mask[:, :, :, 1 : self.vision_token_num],
        #         attention_mask[:, :, :, :1],
        #         attention_mask[:, :, :, self.vision_token_num :],
        #     ],
        #     dim=3,
        # )
        # attention_mask = torch.cat(
        #     [
        #         attention_mask[:, :, 1 : self.vision_token_num, :],
        #         attention_mask[:, :, :1, :],
        #         attention_mask[:, :, self.vision_token_num :, :],
        #     ],
        #     dim=2,
        # )

        # self.register_buffer("attention_mask", attention_mask, persistent=False)

        self.slice_query = xhnn.Slice([0], [4096], [2], [1])
        self.slice_key = xhnn.Slice([4096], [4096 + 1024], [2], [1])
        self.slice_value = xhnn.Slice([4096 + 1024], [sys.maxsize], [2], [1])
        self.key_unsqueeze = xhnn.Unsqueeze(2)
        self.value_unsqueeze = xhnn.Unsqueeze(2)

        self.slice_1 = xhnn.Slice([0], [self.head_dim // 2], [3], [1])
        self.slice_2 = xhnn.Slice([self.head_dim // 2], [sys.maxsize], [3], [1])

        if self.use_cache:
            self.key_expand = xhnn.Expand(
                [
                    1,
                    self.stride[1],
                    self.num_attention_heads // self.num_multi_query_heads,
                    max_sequence_length,
                    self.head_dim,
                ]
            )
            self.value_expand = xhnn.Expand(
                [
                    1,
                    self.stride[2],
                    self.num_attention_heads // self.num_multi_query_heads,
                    max_sequence_length,
                    self.head_dim,
                ]
            )
        else:
            self.key_expand = xhnn.Expand(
                [
                    1,
                    self.stride[1],
                    self.num_attention_heads // self.num_multi_query_heads,
                    self.input_sequence_length,
                    self.head_dim,
                ]
            )
            self.value_expand = xhnn.Expand(
                [
                    1,
                    self.stride[2],
                    self.num_attention_heads // self.num_multi_query_heads,
                    self.input_sequence_length,
                    self.head_dim,
                ]
            )

        if self.use_cache:
            cache_axis = cfg.kv_cache.cache_axis
            self.k_cache = LLMCache(
                axis=cache_axis,
            )
            self.v_cache = LLMCache(
                axis=cache_axis,
            )
        else:
            self.k_cache = None
            self.v_cache = None

        # self.register_buffer("kv_scale", 1 / torch.tensor(math.sqrt(self.head_dim)))
        _kv_scale = 1 / math.sqrt(self.head_dim)
        self.kv_scale = _kv_scale
        # self.softmax = xhnn.SoftmaxPlus(-1)
        self.masked_softmax = xhnn.MaskedSoftmax(-1)
        self.prefill_softmax = xhnn.SoftmaxPlus(-1)
        return self

    def _transpose_for_scores(self, tensor):
        """Transpose a 3D tensor [B, L, H*HD] into a 4D tensor with size [B H L HD]."""
        new_tensor_shape = tensor.size()[:-1] + (-1, self.hidden_size_per_attention_head)  # flexible for multi-query
        tensor = tensor.view(*new_tensor_shape)
        return tensor.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_seq_length: int,
        current_input_length: int,
        position_ids: torch.Tensor,
        past_k_cache: Optional[Tensor] = None,
        past_v_cache: Optional[Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        bsz, q_len, _ = hidden_states.size()

        if self.is_vision_prefill:
            mixed_raw_layer = self.vision_expert_query_key_value(hidden_states)
        else:
            mixed_raw_layer = self.language_expert_query_key_value(hidden_states)

        query_states = self.slice_query(mixed_raw_layer)
        key_states = self.slice_key(mixed_raw_layer)
        value_states = self.slice_value(mixed_raw_layer)

        # query_states = self._transpose_for_scores(query_states)  # B, H, L, HD
        # key_states = self._transpose_for_scores(key_states)  # B, H, L, HD
        # value_states = self._transpose_for_scores(value_states)  # B, H, L, HD

        query_states = query_states.view(bsz, q_len, -1, self.hidden_size_per_attention_head).permute(0, 2, 1, 3)
        key_states = key_states.view(bsz, q_len, -1, self.hidden_size_per_attention_head).permute(0, 2, 1, 3)
        value_states = value_states.view(bsz, q_len, -1, self.hidden_size_per_attention_head).permute(0, 2, 1, 3)

        cos = self.rotary_emb.cos_cached
        sin = self.rotary_emb.sin_cached
        query_states, key_states = self.apply_rotary_pos_emb_index_bhs(query_states, key_states, cos, sin, position_ids)

        query_states = query_states
        key_states = key_states

        if self.use_cache:
            key_states = self.k_cache(key_states, past_seq_length, current_input_length, past_k_cache)
            value_states = self.v_cache(value_states, past_seq_length, current_input_length, past_v_cache)

        if self.use_cache:
            bsz, num_heads, max_seq_len, head_dim = past_k_cache.shape
        else:
            bsz, num_heads, max_seq_len, head_dim = key_states.shape

        key_states = self.key_unsqueeze(key_states)
        key_states = self.key_expand(key_states)
        key_states = key_states.reshape(bsz, self.num_attention_heads, max_seq_len, self.head_dim)

        value_states = self.value_unsqueeze(value_states)
        value_states = self.value_expand(value_states)
        value_states = value_states.reshape(bsz, self.num_attention_heads, max_seq_len, self.head_dim)

        query_states = query_states * self.kv_scale
        attention_scores = torch.matmul(query_states, key_states.transpose(-1, -2))

        attention_scores = self.masked_softmax(attention_scores, past_seq_length)
        context_layer = torch.matmul(attention_scores, value_states)
        context_layer = context_layer.transpose(1, 2).reshape(bsz, q_len, self.hidden_size)

        if self.is_vision_prefill:
            attn_output = self.vision_expert_dense(context_layer)
        else:
            attn_output = self.language_expert_dense(context_layer)
        return attn_output, None, None


@XHLLM_TRACEABLE_MODULES.register_module(
    {
        "transformers_modules.cogvlm2-llama3-chat-19B.modeling_cogvlm.RMSNorm": "cogvlm2.RMSNorm",
    }
)
class _RMSNorm(DynamicRegister):
    def forward(self, hidden_states):
        return self.norm(hidden_states)

    def _setup(self, cfg: Optional[Dict] = None):
        hidden_size = self.weight.shape[0]
        self.norm = RMSNorm(hidden_size, self.variance_epsilon)
        # self.norm.weight.data = self.weight.data.clone()
        self.norm.weight = self.weight
        return self


@XHLLM_TRACEABLE_MODULES.register_module(
    {
        "transformers_modules.cogvlm2-llama3-chat-19B.modeling_cogvlm.CogVLMDecoderLayer": "CogVLMDecoderLayer",
    }
)
class _CogVLMDecoderLayer(DynamicRegister):
    def forward(
        self,
        hidden_states: torch.Tensor,
        past_seq_length: int = 0,
        current_input_length: int = 0,
        position_ids: Optional[torch.LongTensor] = None,
        past_k_cache: Optional[Tensor] = None,
        past_v_cache: Optional[Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_seq_length=past_seq_length,
            current_input_length=current_input_length,
            position_ids=position_ids,
            past_k_cache=past_k_cache,
            past_v_cache=past_v_cache,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        # bsz, seq_len, _ = hidden_states.size()
        # token_type_ids = torch.zeros(bsz, seq_len, dtype=torch.long, device=hidden_states.device)
        # token_type_ids[:, 1 : 1 + self.vision_token_num] = VISION_TOKEN_TYPE
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        return outputs  # type: ignore

    def _setup(self, cfg: Optional[Dict] = None):
        self.vision_token_num = 2306
        return self


@XHLLM_TRACEABLE_MODULES.register_module(
    {
        "transformers_modules.cogvlm2-llama3-chat-19B.modeling_cogvlm.CogVLMModel": "CogVLMModel",
    }
)
class _CogVLMModel(DynamicRegister):
    def forward(
        self,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_seq_length: int = 0,
        current_input_length: int = 0,
        position_ids: Optional[Tensor] = None,
        past_key_cache: Optional[List[Tensor]] = None,
        past_value_cache: Optional[List[Tensor]] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        causal_mask = None
        hidden_states = inputs_embeds

        for idx, decoder_layer in enumerate(self.layers):
            past_k_cache = past_key_cache[idx]
            past_v_cache = past_value_cache[idx]

            layer_outputs = decoder_layer(
                hidden_states,
                past_seq_length=past_seq_length,
                current_input_length=current_input_length,
                position_ids=position_ids,
                past_k_cache=past_k_cache,
                past_v_cache=past_v_cache,
                attention_mask=causal_mask,
            )
            hidden_states = layer_outputs[0]
            if self.only_first_block:
                break

        if self.num_logits_to_keep == 0:
            # hidden_states = torch_ops_xh2a_slice(hidden_states, [0], [current_input_length], [1], [1])
            hidden_states = self.slice(
                hidden_states
            )  # 此时返回的结果，含有padding,调用者需要根据current_input_length切片
        else:
            # 取最后一个token的输出
            hidden_states = self.llm_gather(hidden_states, current_input_length - 1)
        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
        )

    def _setup(self, cfg: Optional[Dict] = None):
        self.llm_gather = xhnn.Gather(1)
        self.num_logits_to_keep = cfg.num_logits_to_keep  # 1,取最后一个token的输出，0,取所有token的输出
        assert self.num_logits_to_keep in [0, 1]

        input_seq_len = cfg.input_sequence_length
        self.slice = xhnn.Slice([0], [input_seq_len], [1], [1])

        def _update_cfg(self, cfg: Optional[Dict] = None):
            input_seq_len = cfg.input_sequence_length
            self.ends = [input_seq_len]

        self.slice._update_cfg = types.MethodType(_update_cfg, self.slice)

        self.is_vision_prefill = False
        # self.is_prefill = False
        self.vision_token_num = 2306

        self.slice_bos = xhnn.Slice([0], [1], [1], [1])
        self.slice_vision = xhnn.Slice([1], [self.vision_token_num], [1], [1])
        self.slice_prompt = xhnn.Slice([self.vision_token_num], [sys.maxsize], [1], [1])

        self.only_first_block = False
        return self


@XHLLM_TRACEABLE_MODULES.register_module(
    {
        "transformers_modules.cogvlm2-llama3-chat-19B.modeling_cogvlm.CogVLMForCausalLM": "CogVLMForCausalLM",
    }
)
class _CogVLMForCausalLM(DynamicRegister):
    def forward(
        self,
        inputs_embeds: Optional[Tensor] = None,
        past_seq_length: Optional[Tensor] = None,
        current_input_length: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_key_cache: Optional[List[Tensor]] = None,
        past_value_cache: Optional[List[Tensor]] = None,
    ):

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            past_seq_length=past_seq_length,
            current_input_length=current_input_length,
            position_ids=position_ids,
            past_key_cache=past_key_cache,
            past_value_cache=past_value_cache,
        )

        # hidden_states = outputs[0]
        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)
        return logits

    def _setup(self, cfg: Optional[Dict] = None):
        return self


def register_wrap_cls(hf_model):
    # print(type(hf_model))
    # print(type(hf_model.model))
    # print(type(hf_model.model.layers[0]))
    # assert False
    # _CogVLMForCausalLM.register(type(hf_model))
    # _CogVLMModel.register(type(hf_model.model))
    # _CogVLMDecoderLayer.register(type(hf_model.model.layers[0]))
    pass
