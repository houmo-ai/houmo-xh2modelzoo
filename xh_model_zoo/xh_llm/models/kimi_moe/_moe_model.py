import math
import sys
import types
from copy import deepcopy
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from torch import Tensor
import sys
import importlib
# /data01/home/xuchen/.cache/huggingface/modules/transformers_modules/kimi/modeling_deepseek.py
# sys.path.append("/data01/home/xuchen/.cache/huggingface/modules/transformers_modules/kimi/")

# full_module_name = "transformers_modules.kimi.modeling_deepseek"
# modeling_deepseek = importlib.import_module(full_module_name)
# DeepseekV3Attention = modeling_deepseek.DeepseekV3Attention
# DeepseekV3DecoderLayer = modeling_deepseek.DeepseekV3DecoderLayer
# DeepseekV3RMSNorm = modeling_deepseek.DeepseekV3RMSNorm
# DeepseekV3RotaryEmbedding = modeling_deepseek.DeepseekV3RotaryEmbedding
# DeepseekV3MoE = modeling_deepseek.DeepseekV3MoE
# DeepseekV3Model = modeling_deepseek.DeepseekV3Model
# DeepseekV3ForCausalLM = modeling_deepseek.DeepseekV3ForCausalLM

from xh_model_zoo.xh_llm.models.kimi_moe.modeling_deepseek import \
    DeepseekV3Attention, DeepseekV3DecoderLayer, DeepseekV3RMSNorm, DeepseekV3RotaryEmbedding, \
    DeepseekV3MoE, DeepseekV3Model, DeepseekV3ForCausalLM

from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeAttention,
    Qwen3MoeDecoderLayer,
    Qwen3MoeForCausalLM,
    Qwen3MoeModel,
    Qwen3MoeRMSNorm,
    Qwen3MoeRotaryEmbedding,
    Qwen3MoeSparseMoeBlock,
)
from xhquant import nn as xhnn
from xhquant.nn import LLMCacheV2, MaskedSoftmax, RMSNorm
from xhquant.nn.modules.moeblock import MoeBlock
from xhquant.utils import digit_version
from xhquant.utils.registry import DynamicModule

from ..builder import XHLLM_TRACEABLE_MODULES


@XHLLM_TRACEABLE_MODULES.register_module(
    {
        DeepseekV3RotaryEmbedding: "DeepseekV3RotaryEmbedding",
    }
)
class _DeepseekV3RotaryEmbedding(DynamicModule):
    def _setup(self, cfg: Optional[Dict] = None):
        # assert "dynamic" not in self.rope_type, f"{self.rope_type} is not supported in dynamic mode"

        # self.max_position_embeddings = max_position_embeddings
        # Build here to make `torch.jit.trace` work.
        self.max_seq_len_cached = 40960
        self._setup_cos_sin_cache(seq_len=self.max_seq_len_cached, dtype=self.inv_freq.dtype)
        if hasattr(self, "setup_after_callback"):
            self.setup_after_callback()

    def _setup_cos_sin_cache(self, seq_len, dtype):
        """
        4.45 版本实现
        """
        position_ids = torch.arange(0, seq_len, dtype=torch.long, device=self.inv_freq.device).unsqueeze(0)
        # self.inv_freq = self.inv_freq.to(torch.float16)
        # cos, sin = self.forward(self.inv_freq, position_ids)

        inv_freq = self.inv_freq.to(torch.float32)
        device = self.inv_freq.device
        if torch.cuda.is_available() and inv_freq.device.type != "cuda":
            inv_freq = self.inv_freq.cuda()
        cos, sin = self.forward(self.inv_freq, position_ids)
        cos = cos.to(device)
        sin = sin.to(device)

        sin = sin.squeeze(0)
        cos = cos.squeeze(0)
        self.register_buffer("sin_cached", sin.to(dtype=dtype), persistent=False)
        self.register_buffer("cos_cached", cos.to(dtype=dtype), persistent=False)
        # self.sin_cached = nn.Parameter(sin.to(device=device, dtype=dtype), requires_grad=False)
        # self.cos_cached = nn.Parameter(cos.to(device=device, dtype=dtype), requires_grad=False)

    def _set_dtype(self, dtype: torch.dtype) -> None:
        self.inv_freq = self.inv_freq.to(dtype)
        self._setup_cos_sin_cache(seq_len=self.max_seq_len_cached, dtype=self.inv_freq.dtype)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        """
        4.37 版本实现
        """
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        # if "dynamic" in self.rope_type:
        #     self._dynamic_frequency_update(position_ids, device=x.device)

        # Core RoPE block
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        # with torch.autocast(device_type=device_type, enabled=False):
        #     freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        #     emb = torch.cat((freqs, freqs), dim=-1)
        #     cos = emb.cos()
        #     sin = emb.sin()
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos #* self.attention_scaling
        sin = sin #* self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


@XHLLM_TRACEABLE_MODULES.register_module(
    {
        DeepseekV3Attention: "DeepseekV3Attention",
    }
)
class _DeepseekV3Attention(DynamicModule):
    def rotate_half(self, x: Tensor):
        """Rotates half the hidden dims of the input."""
        # x1 = x[..., : x.shape[-1] // 2]
        # x2 = x[..., x.shape[-1] // 2 :]
        # x1 = torch_ops_xh2a_slice(x, [0], [self.head_dim // 2], [3], [1])
        # x2 = torch_ops_xh2a_slice(x, [self.head_dim // 2], [sys.maxsize], [3], [1])
        x1 = self.rotate_half_slice_1(x)
        x2 = self.rotate_half_slice_2(x)
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(self, q: Tensor, k: Tensor, cos: Tensor, sin: Tensor, unsqueeze_dim: int = 1):
        # cos = cos.unsqueeze(unsqueeze_dim)
        # sin = sin.unsqueeze(unsqueeze_dim)
        # cos = self.cos_unsqueeze(cos)
        # sin = self.sin_unsqueeze(sin)

        b, h, s, d = q.shape
        q = q.view(b, h, s, 32, 2).transpose(4, 3).reshape(b, h, s, d)

        b, h, s, d = k.shape
        k = k.view(b, h, s, 32, 2).transpose(4, 3).reshape(b, h, s, d)

        if self.enable_rope:
            q_embed = self.rope(q, cos, sin)
            k_embed = self.rope(k, cos, sin)
        else:
            q_embed = (q * cos) + (self.rotate_half(q) * sin)
            k_embed = (k * cos) + (self.rotate_half(k) * sin)

        # q_embed = self.rope(q, cos, sin)
        # k_embed = self.rope(k, cos, sin)
        return q_embed, k_embed

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_seq_length: Optional[Tensor] = None,
        current_input_length: Optional[Tensor] = None,
        position_ids: torch.Tensor = None,
        past_k_cache: Optional[Tensor] = None,
        past_v_cache: Optional[Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()
        causal_mask = None
         
        q = self.q_proj(hidden_states)
        
        q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        q_nope, q_pe = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        compressed_kv, k_pe = torch.split(
            compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )

        k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
        kv = (
            self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
            .view(bsz, q_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
            .transpose(1, 2)
        )

        k_nope, value_states = torch.split(
            kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
        )
        kv_seq_len = value_states.shape[-2] + past_seq_length
        
        # if past_key_value is not None:
        #     if self.layer_idx is None:
        #         raise ValueError(
        #             f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
        #             "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
        #             "with a layer index."
        #         )
        #     kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        
        # cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        cos, sin = position_embeddings

        # q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids)
        q_pe, k_pe = self.apply_rotary_pos_emb(q_pe, k_pe, cos, sin, unsqueeze_dim=1) # 881 5932

        # query_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
        # query_states[:, :, :, : self.qk_nope_head_dim] = q_nope
        # query_states[:, :, :, self.qk_nope_head_dim :] = q_pe

        # key_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
        # key_states[:, :, :, : self.qk_nope_head_dim] = k_nope
        # key_states[:, :, :, self.qk_nope_head_dim :] = k_pe

        query_states = torch.concat([q_nope, q_pe], dim=-1) * self.softmax_scale # 90.25

        k_pe = k_pe.repeat(1, 16, 1, 1)
        key_states = torch.concat([k_nope, k_pe], dim=-1) # 135

        # if past_key_value is not None:
        #     cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
        #     key_states, value_states = past_key_value.update(
        #         key_states, value_states, self.layer_idx, cache_kwargs
        #     )
        
        if self.use_cache:
            key_states = self.k_cache(key_states, past_seq_length, current_input_length, past_k_cache)
            value_states = self.v_cache(value_states, past_seq_length, current_input_length, past_v_cache)

        # attn_weights = (
        #     torch.matmul(query_states, key_states.transpose(2, 3)) * self.softmax_scale
        # )
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3))


        # if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
        #     raise ValueError(
        #         f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
        #         f" {attn_weights.size()}"
        #     )
        # assert attention_mask is not None
        # if attention_mask is not None:
        #     if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
        #         raise ValueError(
        #             f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
        #         )
        #     attn_weights = attn_weights + attention_mask

        # # upcast attention to fp32
        # attn_weights = nn.functional.softmax(
        #     attn_weights, dim=-1, dtype=torch.float32
        # ).to(query_states.dtype)
        # attn_weights = nn.functional.dropout(
        #     attn_weights, p=self.attention_dropout, training=self.training
        # )
        attn_weights: Optional[Tensor] = self.masked_softmax(attn_weights, past_seq_length)

        attn_output = torch.matmul(attn_weights, value_states)

        # if attn_output.size() != (bsz, self.num_heads, q_len, self.v_head_dim):
        #     raise ValueError(
        #         f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.v_head_dim)}, but is"
        #         f" {attn_output.size()}"
        #     )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.v_head_dim)

        attn_output = self.o_proj(attn_output)

        # if not output_attentions:
        #     attn_weights = None

        return attn_output, None, None        


    def _setup(self, cfg: Optional[Dict] = None):
        self.enable_rope = cfg.get("enable_rope", True)
        if self.enable_rope:
            self.rope = xhnn.Rope()

        if not hasattr(self, "num_key_value_heads"):
            self.num_key_value_heads = self.config.num_key_value_heads

        if not hasattr(self, "num_heads"):
            self.num_heads = self.config.num_attention_heads  # 28

        # self.attn_hidden_dim = self.head_dim * self.num_heads
        self.head_dim = 64

        self.rotate_half_slice_1 = xhnn.Slice([0], [self.head_dim // 2], [3], [1])
        self.rotate_half_slice_2 = xhnn.Slice([self.head_dim // 2], [sys.maxsize], [3], [1])
        self.masked_softmax = MaskedSoftmax(dim=-1)
        use_cache = cfg.use_cache
        self.use_cache = use_cache
        max_sequence_length = cfg.max_sequence_length
        self.max_sequence_length = max_sequence_length
        input_seq_len = cfg.input_sequence_length
        # self.cos_gather = xhnn.Gather(0)
        # self.sin_gather = xhnn.Gather(0)
        # self.rope = Rope() 

        # self.qk_nope_head_dim, self.qk_rope_head_dim
        # self.q_nopre_slice = xhnn.Slice([0], [self.head_dim // 2], [3], [1])

        if use_cache:
            cache_axis = cfg.kv_cache.cache_axis
            self.k_cache = LLMCacheV2(
                axis=cache_axis,
            )
            self.v_cache = LLMCacheV2(
                axis=cache_axis,
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
        DeepseekV3DecoderLayer: "DeepseekV3DecoderLayer",
    }
)
class _DeepseekV3DecoderLayer(DynamicModule):
    def forward(
        self,
        hidden_states: torch.Tensor,
        past_seq_length: Optional[Tensor] = None,
        current_input_length: Optional[Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_k_cache: Optional[Tensor] = None,
        past_v_cache: Optional[Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence.
            position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """
        residual = hidden_states # 20.7344

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            position_ids=position_ids,
            past_seq_length=past_seq_length,
            current_input_length=current_input_length,
            past_k_cache=past_k_cache,
            past_v_cache=past_v_cache,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        if isinstance(hidden_states, tuple):
            hidden_states, router_logits = hidden_states
        else:
            router_logits = None
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        return outputs

    def _setup(self, cfg: Optional[Dict] = None):
        return self


@XHLLM_TRACEABLE_MODULES.register_module(
    {
        DeepseekV3MoE: "DeepseekV3MoE",
    }
)
class _DeepseekV3MoE(DynamicModule):
    def forward(self, hidden_states):
        ori_hidden_states = hidden_states
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        # hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate_linear(hidden_states)

        routing_weights = router_logits.sigmoid() + self.score_bias
        # routing_weights = F.softmax(router_logits, dim=-1)
        y = self.moeblock(
            ori_hidden_states,
            routing_weights,
            #  self.expert_gate_proj_weight.to(ori_hidden_states),
            #  self.expert_gate_proj_bias,
            #  self.expert_up_proj_weight.to(ori_hidden_states),
            #  self.expert_up_proj_bias,
            #  self.expert_down_proj_weight.to(ori_hidden_states),
            #  self.expert_down_proj_bias
        ) * self.gate.routed_scaling_factor

        y = y + self.shared_experts(hidden_states)
        return y

        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:  # only diff with mixtral sparse moe block!
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(torch.float16)
        # final_hidden_states = torch.zeros(
        #     (self.batch_size * self.input_seq_len, 2048), dtype=torch.float16, device=self.device
        # )
        final_hidden_states = torch.zeros_like(hidden_states)
        # final_hidden_states = torch.zeros(
        #     (self.batch_size * self.input_seq_len, 2048), dtype=hidden_states.dtype, device=hidden_states.device
        # )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])
            # original index add
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]
            final_hidden_states = final_hidden_states.index_add_(0, top_x, current_hidden_states.to(torch.float16))
        final_hidden_states = final_hidden_states.reshape(self.batch_size, sequence_length, 2048)
        # final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits

    def _setup(self, cfg: Optional[Dict] = None):
        self.input_seq_len = cfg.input_sequence_length
        self.batch_size = cfg.batch_size
        self.device = self.gate.weight.device
        self.num_experts = len(self.experts)

        self.score_bias = self.gate.e_score_correction_bias.unsqueeze(0).half().to(self.device)

        self.gate_linear = nn.Linear(self.gate.weight.shape[1], self.gate.weight.shape[0], bias=False)
        self.gate_linear.weight.data.copy_(self.gate.weight)

        self.moeblock = MoeBlock(self.experts[0].act_fn._get_name().lower(), self.gate.top_k, self.gate.norm_topk_prob)

        self.moeblock.expert_gate_proj_weight = torch.nn.Parameter(
            torch.zeros(
                self.num_experts,
                self.experts[0].gate_proj.weight.shape[0],
                self.experts[0].gate_proj.weight.shape[1],
                device=self.device,
                dtype=self.experts[0].gate_proj.weight.dtype,
            )
        )
        if hasattr(self.experts[0].gate_proj, "quant_weight") and self.experts[0].gate_proj.quant_weight is not None:
            self.moeblock.expert_gate_proj_quant_weight = torch.tensor(
                torch.zeros(
                    self.num_experts,
                    self.experts[0].gate_proj.quant_weight.shape[0],
                    self.experts[0].gate_proj.quant_weight.shape[1],
                    device=self.device,
                    dtype=self.experts[0].gate_proj.quant_weight.dtype,
                )
            )

        if self.experts[0].gate_proj.bias is not None:
            self.moeblock.expert_gate_proj_bias = torch.nn.Parameter(
                torch.zeros(self.num_experts, self.experts[0].gate_proj.bias.shape[0], device=self.device)
            )
        else:
            self.moeblock.expert_gate_proj_bias = None

        self.moeblock.expert_up_proj_weight = torch.nn.Parameter(
            torch.zeros(
                self.num_experts,
                self.experts[0].up_proj.weight.shape[0],
                self.experts[0].up_proj.weight.shape[1],
                device=self.device,
                dtype=self.experts[0].up_proj.weight.dtype,
            )
        )

        if hasattr(self.experts[0].up_proj, "quant_weight") and self.experts[0].up_proj.quant_weight is not None:
            self.moeblock.expert_up_proj_quant_weight = torch.tensor(
                torch.zeros(
                    self.num_experts,
                    self.experts[0].up_proj.quant_weight.shape[0],
                    self.experts[0].up_proj.quant_weight.shape[1],
                    device=self.device,
                    dtype=self.experts[0].up_proj.quant_weight.dtype,
                )
            )

        if self.experts[0].up_proj.bias is not None:
            self.moeblock.expert_up_proj_bias = torch.nn.Parameter(
                torch.zeros(self.num_experts, self.experts[0].up_proj.bias.shape[0], device=self.device)
            )
        else:
            self.moeblock.expert_up_proj_bias = None

        self.moeblock.expert_down_proj_weight = torch.nn.Parameter(
            torch.zeros(
                self.num_experts,
                self.experts[0].down_proj.weight.shape[0],
                self.experts[0].down_proj.weight.shape[1],
                device=self.device,
                dtype=self.experts[0].down_proj.weight.dtype,
            )
        )

        if hasattr(self.experts[0].down_proj, "quant_weight") and self.experts[0].down_proj.quant_weight is not None:
            self.moeblock.expert_down_proj_quant_weight = torch.tensor(
                torch.zeros(
                    self.num_experts,
                    self.experts[0].down_proj.quant_weight.shape[0],
                    self.experts[0].down_proj.quant_weight.shape[1],
                    device=self.device,
                    dtype=self.experts[0].down_proj.quant_weight.dtype,
                )
            )

        if self.experts[0].down_proj.bias is not None:
            self.moeblock.expert_down_proj_bias = torch.nn.Parameter(
                torch.zeros(self.num_experts, self.experts[0].down_proj.bias.shape[0], device=self.device)
            )
        else:
            self.moeblock.expert_down_proj_bias = None

        with torch.no_grad():
            self.moeblock.expert_gate_proj_weight.copy_(
                torch.cat([expert.gate_proj.weight.data.to(self.device).unsqueeze(0) for expert in self.experts], dim=0)
            )
            if (
                hasattr(self.experts[0].gate_proj, "quant_weight")
                and self.experts[0].gate_proj.quant_weight is not None
            ):
                self.moeblock.expert_gate_proj_quant_weight.copy_(
                    torch.cat(
                        [expert.gate_proj.quant_weight.data.to(self.device).unsqueeze(0) for expert in self.experts],
                        dim=0,
                    )
                )
            if self.moeblock.expert_gate_proj_bias is not None:
                self.moeblock.expert_gate_proj_bias.copy_(
                    torch.cat(
                        [expert.gate_proj.bias.data.to(self.device).unsqueeze(0) for expert in self.experts], dim=0
                    )
                )

            self.moeblock.expert_up_proj_weight.copy_(
                torch.cat([expert.up_proj.weight.data.to(self.device).unsqueeze(0) for expert in self.experts], dim=0)
            )
            if hasattr(self.experts[0].up_proj, "quant_weight") and self.experts[0].up_proj.quant_weight is not None:
                self.moeblock.expert_up_proj_quant_weight.copy_(
                    torch.cat(
                        [expert.up_proj.quant_weight.data.to(self.device).unsqueeze(0) for expert in self.experts],
                        dim=0,
                    )
                )
            if self.moeblock.expert_up_proj_bias is not None:
                self.moeblock.expert_up_proj_bias.copy_(
                    torch.cat([expert.up_proj.bias.data.to(self.device).unsqueeze(0) for expert in self.experts], dim=0)
                )

            self.moeblock.expert_down_proj_weight.copy_(
                torch.cat([expert.down_proj.weight.data.to(self.device).unsqueeze(0) for expert in self.experts], dim=0)
            )
            if (
                hasattr(self.experts[0].down_proj, "quant_weight")
                and self.experts[0].down_proj.quant_weight is not None
            ):
                self.moeblock.expert_down_proj_quant_weight.copy_(
                    torch.cat(
                        [expert.down_proj.quant_weight.data.to(self.device).unsqueeze(0) for expert in self.experts],
                        dim=0,
                    )
                )
            if self.moeblock.expert_down_proj_bias is not None:
                self.moeblock.expert_down_proj_bias.copy_(
                    torch.cat(
                        [expert.down_proj.bias.data.to(self.device).unsqueeze(0) for expert in self.experts], dim=0
                    )
                )

        del self.experts
        torch.cuda.empty_cache()

        return self

    # def create_zeros(self, shape, dtype, device):
    #     return torch.zeros(shape, dtype=dtype, device=device)


@XHLLM_TRACEABLE_MODULES.register_module(
    {
        DeepseekV3RMSNorm: "DeepseekV3RMSNorm",
    }
)
class _DeepseekV3RMSNorm(DynamicModule):
    def forward(self, hidden_states):
        return self.norm(hidden_states)

    def _setup(self, cfg: Optional[Dict] = None):
        hidden_size = self.weight.shape[0]
        self.norm = RMSNorm(hidden_size, self.variance_epsilon)
        self.norm.weight = nn.Parameter(deepcopy(self.weight.data))
        return self


@XHLLM_TRACEABLE_MODULES.register_module(
    {
        DeepseekV3Model: "DeepseekV3Model",
    }
)
class _DeepseekV3Model(DynamicModule):
    def forward(
        self,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_seq_length: Optional[Tensor] = None,
        current_input_length: Optional[Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_cache: Optional[List[Tensor]] = None,
        past_value_cache: Optional[List[Tensor]] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        causal_mask = None  # 在Qwen2Attention中处理
        hidden_states = inputs_embeds

        cos = self.cos_embeding(position_ids)
        sin = self.sin_embeding(position_ids)
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

        position_embeddings = (cos, sin)

        for idx, decoder_layer in enumerate(self.layers):
            print("processing: ", idx)
            if self.use_cache:
                _past_k_cache = past_key_cache[idx]
                _past_v_cache = past_value_cache[idx]
            else:
                _past_k_cache = None
                _past_v_cache = None

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                past_seq_length=past_seq_length,
                current_input_length=current_input_length,
                position_ids=position_ids,
                past_k_cache=_past_k_cache,
                past_v_cache=_past_v_cache,
                position_embeddings=position_embeddings,
            )

            hidden_states = layer_outputs[0]
            # break
            if self.only_first_block:
                break

        # hidden_states = hidden_states[
        #     :,
        #     -num_logits_to_keep:,
        # ]
        # hidden_states = hidden_states[:, :current_input_length, :]
        # hidden_states = self.llm_gather(hidden_states, current_input_length, num_logits_to_keep)
        if self.num_logits_to_keep == 0:  # for PPL task
            # hidden_states = torch_ops_xh2a_slice(hidden_states, [0], [current_input_length], [1], [1])
            # hidden_states = self.slice(
            #     hidden_states
            # )  # 此时返回的结果，含有padding,调用者需要根据current_input_length切片
            pass
        else:
            # 取最后一个token的输出
            hidden_states = self.llm_gather(hidden_states, current_input_length - 1)
        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
        )

    def _setup_cos_sin_embeding(self):
        cos = self.rotary_emb.cos_cached.squeeze(1)
        sin = self.rotary_emb.sin_cached.squeeze(1)
        num_embeddings, embedding_dim = cos.shape

        self.cos_embeding = nn.Embedding(num_embeddings, embedding_dim)
        self.cos_embeding.weight.data = cos

        self.sin_embeding = nn.Embedding(num_embeddings, embedding_dim)
        self.sin_embeding.weight.data = sin

    def _setup(self, cfg: Optional[Dict] = None):
        self.batch_size = cfg.get("batch_size", 1)
        self.only_first_block = cfg.get("only_first_block", False)
        # max_seq_len = cfg.max_sequence_length
        # self.rotary_matrix_cache = RotaryMatrixCache(self.rotary_emb, max_seq_len)

        self.num_logits_to_keep = cfg.num_logits_to_keep  # 1,取最后一个token的输出，0,取所有token的输出
        assert self.num_logits_to_keep in [0, 1]

        input_seq_len = cfg.input_sequence_length
        self.slice = xhnn.Slice([0], [input_seq_len], [1], [1])

        self.llm_gather = xhnn.BatchGather(1)
        self.llm_gather.update_offset_indices(self.batch_size, input_seq_len)

        def _llm_gather_update_cfg(self: xhnn.BatchGather, cfg: Optional[Dict] = None):
            self.num_logits_to_keep = cfg.num_logits_to_keep
            input_seq_len = cfg.input_sequence_length
            batch_size = cfg.get("batch_size", 1)
            self.update_offset_indices(batch_size, input_seq_len)

        self.llm_gather._update_cfg = types.MethodType(_llm_gather_update_cfg, self.llm_gather)

        def _slice_update_cfg(self, cfg: Optional[Dict] = None):
            input_seq_len = cfg.input_sequence_length
            self.ends = [input_seq_len]

        self.slice._update_cfg = types.MethodType(_slice_update_cfg, self.slice)
        self.use_cache = cfg.use_cache

        self.rotary_emb = self.layers[0].self_attn.rotary_emb
        if not hasattr(self.rotary_emb, "cos_cached"):
            self.rotary_emb.setup_after_callback = self._setup_cos_sin_embeding
        else:
            self._setup_cos_sin_embeding()

        return self


@XHLLM_TRACEABLE_MODULES.register_module(
    {
        DeepseekV3ForCausalLM: "DeepseekV3ForCausalLM",
    }
)
class _DeepseekV3ForCausalLM(DynamicModule):
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

        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)
        return logits

    def _setup(self, cfg: Optional[Dict] = None):
        # assert not self.config.use_sliding_window, f"{self.__class__.__name__} does not support sliding window."
        return self


def register_wrap_modules():
    pass
