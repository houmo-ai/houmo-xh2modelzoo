import math
import sys
import types
from typing import Any, Optional

import torch
from qwen_tts.core.models.modeling_qwen3_tts import (
    Qwen3TTSAttention,
    Qwen3TTSDecoderLayer,
    Qwen3TTSRotaryEmbedding,
    Qwen3TTSTalkerCodePredictorModel,
    Qwen3TTSTalkerCodePredictorOutputWithPast,
)
from torch import Tensor
from transformers.modeling_outputs import BaseModelOutputWithPast

import xhquant.nn as xhnn
from xhquant.nn import BfpFlashAttention, LLMCache, MaskedSoftmax
from xhquant.utils import ConfigDict
from xhquant.utils.registry import DynamicModule

from xh_model_zoo.api import get_root_logger

from ..builder import XHLLM_TRACEABLE_MODULES
from .qwen3_tts import (
    XHQwen3TTSTalkerCodePredictorModelForConditionalGeneration as Qwen3TTSTalkerCodePredictorModelForConditionalGeneration,
)


@XHLLM_TRACEABLE_MODULES.register_module({Qwen3TTSRotaryEmbedding: "Qwen3TTSRotaryEmbedding"})
class _Qwen3TTSRotaryEmbedding(DynamicModule):
    def _setup(self, cfg: dict | None = None):
        assert "dynamic" not in self.rope_type, f"{self.rope_type} is not supported in dynamic mode"
        max_seq_len = cfg.max_sequence_length
        max_seq_len = max(max_seq_len, self.config.max_position_embeddings)

        self.config.max_position_embeddings = max_seq_len
        self.max_seq_len_cached = max_seq_len
        self.original_max_seq_len = max_seq_len
        # self.max_position_embeddings = max_position_embeddings
        # Build here to make `torch.jit.trace` work.
        self._setup_cos_sin_cache(seq_len=self.max_seq_len_cached, dtype=self.inv_freq.dtype)

        if hasattr(self, "setup_after_callback"):
            self.setup_after_callback()

    def _setup_cos_sin_cache(self, seq_len, dtype):
        """
        4.45 版本实现
        """
        # self.inv_freq = self.inv_freq.to(torch.float16)
        inv_freq = self.inv_freq.to(torch.float32)
        device = self.inv_freq.device
        if torch.cuda.is_available() and inv_freq.device.type != "cuda":
            inv_freq = self.inv_freq.cuda()
        position_ids = torch.arange(0, seq_len, dtype=torch.long, device=inv_freq.device).unsqueeze(0)
        cos, sin = self.forward(inv_freq, position_ids)
        cos = cos.to(device)
        sin = sin.to(device)
        sin = sin.squeeze(0)
        cos = cos.squeeze(0)

        # TODO:临时处理
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

        sin = sin.contiguous().to(dtype=dtype)
        cos = cos.contiguous().to(dtype=dtype)

        self.register_buffer("sin_cached", sin, persistent=False)
        self.register_buffer("cos_cached", cos, persistent=False)
        # self.sin_cached = nn.Parameter(sin.to(device=device, dtype=dtype), requires_grad=False)
        # self.cos_cached = nn.Parameter(cos.to(device=device, dtype=dtype), requires_grad=False)

    def _set_dtype(self, dtype: torch.dtype) -> None:
        self.inv_freq = self.inv_freq.to(dtype)
        self._setup_cos_sin_cache(seq_len=self.max_seq_len_cached, dtype=self.inv_freq.dtype)


@XHLLM_TRACEABLE_MODULES.register_module({Qwen3TTSAttention: "Qwen3TTSAttention"})
class _Qwen3TTSAttention(DynamicModule):
    def _setup(self, cfg: ConfigDict | dict[str, Any]):
        if isinstance(cfg, dict):
            cfg = ConfigDict(cfg)
        if not hasattr(self, "num_key_value_heads"):
            self.num_key_value_heads = self.config.num_key_value_heads
        if not hasattr(self, "num_heads"):
            self.num_heads = self.config.num_attention_heads  # 28
        if not hasattr(self, "hidden_size"):
            self.hidden_size = self.config.hidden_size  # 3584

        self.enable_rope = cfg.get("enable_rope", True)
        if self.enable_rope:
            self.rope = xhnn.Rope()

        self.slice_1 = xhnn.Slice([0], [self.head_dim // 2], [3], [1])
        self.slice_2 = xhnn.Slice([self.head_dim // 2], [sys.maxsize], [3], [1])
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
        if self.use_bfp_flash_attention:
            self.bfp_attn = BfpFlashAttention(
                self.hidden_size,
                self.num_heads,
                True,
                self.sefp_manbit,
                self.out_fp_expbit,
                self.out_fp_manbit,
            )
        else:
            self.masked_softmax = MaskedSoftmax(dim=-1)
        use_cache = cfg.use_cache
        self.use_cache = use_cache
        self.key_extra_scale = 1.0 if "key_extra_scale" not in cfg else cfg.key_extra_scale
        self.query_extra_scale = 1.0 if "query_extra_scale" not in cfg else cfg.query_extra_scale
        max_sequence_length = cfg.max_sequence_length
        self.max_sequence_length = max_sequence_length
        # input_seq_len = cfg.input_sequence_length
        self.cos_gather = xhnn.Gather(0)
        self.sin_gather = xhnn.Gather(0)

        if use_cache:
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
        _kv_scale = 1 / math.sqrt(self.head_dim)
        self.kv_scale = _kv_scale

        return self

    def rotate_half(self, x: Tensor):
        """Rotates half the hidden dims of the input."""
        # x1 = x[..., : x.shape[-1] // 2]
        # x2 = x[..., x.shape[-1] // 2 :]
        # x1 = torch_ops_xh2a_slice(x, [0], [self.head_dim // 2], [3], [1])
        # x2 = torch_ops_xh2a_slice(x, [self.head_dim // 2], [sys.maxsize], [3], [1])
        x1 = self.slice_1(x)
        x2 = self.slice_2(x)
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(self, q: Tensor, k: Tensor, cos: Tensor, sin: Tensor, unsqueeze_dim: int = 1):
        # cos = cos.unsqueeze(unsqueeze_dim)
        # sin = sin.unsqueeze(unsqueeze_dim)
        # cos = self.cos_unsqueeze(cos)
        # sin = self.sin_unsqueeze(sin)
        if self.enable_rope:
            q_embed = self.rope(q, cos, sin)
            k_embed = self.rope(k, cos, sin)
        else:
            q_embed = (q * cos) + (self.rotate_half(q) * sin)
            k_embed = (k * cos) + (self.rotate_half(k) * sin)
        return q_embed, k_embed

    def forward(
        self,
        hidden_states: torch.Tensor,
        # position_ids: torch.Tensor,
        past_seq_length: Tensor | None = None,
        current_input_length: Tensor | None = None,
        past_k_cache: Tensor | None = None,
        past_v_cache: Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]:
        # input_shape = hidden_states.shape[:-1]
        # hidden_shape = (*input_shape, -1, self.head_dim)

        bsz, q_len, _ = hidden_states.size()
        # bsz = hidden_states.shape[0]
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = self.q_norm(query_states.view(bsz, q_len, self.num_heads, self.head_dim)).transpose(1, 2)
        key_states = self.k_norm(key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # cos = self.cos_slice(self.rotary_emb.cos_cached, past_seq_length)
        # sin = self.sin_slice(self.rotary_emb.sin_cached, past_seq_length)

        # cos = self.cos_unsqueeze(cos)
        # sin = self.sin_unsqueeze(sin)
        cos, sin = position_embeddings
        query_states, key_states = self.apply_rotary_pos_emb(query_states, key_states, cos, sin, unsqueeze_dim=0)

        if self.use_cache:
            key_states = self.k_cache(key_states, past_seq_length, current_input_length, past_k_cache)
            value_states = self.v_cache(value_states, past_seq_length, current_input_length, past_v_cache)
        if self.use_bfp_flash_attention:
            attn_output = self.bfp_attn(query_states, key_states, value_states)
        else:
            query_states = query_states * self.kv_scale  # [bsz, self.num_key_value_heads, seq_len, self.head_dim]
            key_states = key_states.transpose(2, 3)
            key_states = torch.repeat_interleave(
                key_states,
                self.num_key_value_groups,
                dim=1,
            )

            attn_weights = torch.matmul(query_states, key_states)  # [4, 28, 256, 128], [4, 28, 128, 32768]
            attn_weights: Optional[Tensor] = self.masked_softmax(attn_weights, past_seq_length)
            value_states = torch.repeat_interleave(
                value_states,
                self.num_key_value_groups,
                dim=1,
            )
            attn_output = torch.matmul(attn_weights, value_states)  # [4, 28, 256, 32768], [4, 28, 32768, 128]
            attn_output = attn_output.transpose(1, 2)

        attn_output = attn_output.reshape(bsz, q_len, self.config.num_attention_heads * self.head_dim)
        attn_output = self.o_proj(attn_output)

        # return attn_output, attn_weights, past_key_value
        return attn_output, None


@XHLLM_TRACEABLE_MODULES.register_module(
    {
        Qwen3TTSDecoderLayer: "Qwen3TTSDecoderLayer",
    }
)
class _Qwen3TTSDecoderLayer(DynamicModule):
    def forward(
        self,
        hidden_states: torch.Tensor,
        # position_ids: torch.LongTensor | None = None,
        # rotary_matrix: torch.Tensor | None = None,
        past_seq_length: Tensor | None = None,
        current_input_length: Tensor | None = None,
        past_k_cache: Tensor | None = None,
        past_v_cache: Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> tuple[torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor] | None]:
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
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            # rotary_matrix=rotary_matrix,
            # position_ids=position_ids,
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
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        return outputs

    def _setup(self, cfg: dict | None = None):
        return self


@XHLLM_TRACEABLE_MODULES.register_module({Qwen3TTSTalkerCodePredictorModel: "Qwen3TTSTalkerCodePredictorModel"})
class _Qwen3TTSTalkerCodePredictorModel(DynamicModule):
    def _setup_cos_sin_embeding(self):
        cos_cached = self.rotary_emb.cos_cached
        sin_cached = self.rotary_emb.sin_cached
        logger = get_root_logger()
        if cos_cached is None or sin_cached is None:
            logger.warning("Cosine and sine caches are not set up. This may lead to incorrect positional embeddings.")

    def _setup(self, cfg: dict | None = None):
        self.only_first_block = cfg.get("only_first_block", False)
        self.max_layers = -1
        if self.only_first_block:
            self.max_layers = 1
        if "max_layers" in cfg:
            self.max_layers = cfg.max_layers
        # max_seq_len = cfg.max_sequence_length
        # self.rotary_matrix_cache = RotaryMatrixCache(self.rotary_emb, max_seq_len)

        self.num_logits_to_keep = cfg.num_logits_to_keep  # 1,取最后一个token的输出，0,取所有token的输出
        assert self.num_logits_to_keep in [0, 1]

        input_seq_len = cfg.input_sequence_length
        self.slice = xhnn.Slice([0], [input_seq_len], [1], [1])

        self.llm_gather = xhnn.BatchGather(1)
        self.llm_gather.update_offset_indices(1, input_seq_len)

        def _llm_gather_update_cfg(self: xhnn.BatchGather, cfg: Optional[dict] = None):
            self.num_logits_to_keep = cfg.num_logits_to_keep
            input_seq_len = cfg.input_sequence_length
            batch_size = cfg.get("batch_size", 1)
            self.update_offset_indices(batch_size, input_seq_len)

        self.llm_gather._update_cfg = types.MethodType(_llm_gather_update_cfg, self.llm_gather)

        def _update_cfg(self, cfg: Optional[dict] = None):
            input_seq_len = cfg.input_sequence_length
            self.ends = [input_seq_len]

        self.slice._update_cfg = types.MethodType(_update_cfg, self.slice)
        self.use_cache = cfg.use_cache

        self.sin_slice = xhnn.DynamicSlice([input_seq_len], [2], [1])
        self.cos_slice = xhnn.DynamicSlice([input_seq_len], [2], [1])

        def _sin_cos_slice_update_cfg(self, cfg: Optional[dict] = None):
            input_seq_len = cfg.input_sequence_length
            self.valid_length = [input_seq_len]

        self.sin_slice._update_cfg = types.MethodType(_sin_cos_slice_update_cfg, self.sin_slice)
        self.cos_slice._update_cfg = types.MethodType(_sin_cos_slice_update_cfg, self.cos_slice)

        self.cos_unsqueeze = xhnn.Unsqueeze(0)
        self.sin_unsqueeze = xhnn.Unsqueeze(0)

        if not hasattr(self.rotary_emb, "cos_cached"):
            self.rotary_emb.setup_after_callback = self._setup_cos_sin_embeding
        else:
            self._setup_cos_sin_embeding()

        return self

    def forward(
        self,
        # position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        past_seq_length: Tensor | None = None,
        current_input_length: Tensor | None = None,
        past_key_cache: list[Tensor] | None = None,
        past_value_cache: list[Tensor] | None = None,
    ) -> tuple | BaseModelOutputWithPast:
        causal_mask = None  # 在Qwen3TTSAttention中处理
        hidden_states = inputs_embeds
        cos = self.cos_slice(self.rotary_emb.cos_cached, past_seq_length)
        sin = self.sin_slice(self.rotary_emb.sin_cached, past_seq_length)
        position_embeddings = (cos, sin)

        for idx, decoder_layer in enumerate(self.layers):
            if self.use_cache:
                _past_k_cache = past_key_cache[idx]
                _past_v_cache = past_value_cache[idx]
            else:
                _past_k_cache = None
                _past_v_cache = None
            # if idx == 0:
            #    import time
            #    time_start = time.time()
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                # # position_ids=position_ids,
                past_seq_length=past_seq_length,
                current_input_length=current_input_length,
                past_k_cache=_past_k_cache,
                past_v_cache=_past_v_cache,
                position_embeddings=position_embeddings,
            )

            hidden_states = layer_outputs[0]
            if self.max_layers > 0 and idx + 1 >= self.max_layers:
                break
        if self.num_logits_to_keep == 0:
            # hidden_states = torch_ops_xh2a_slice(hidden_states, [0], [current_input_length], [1], [1])
            # hidden_states = self.slice(
            #     hidden_states
            # )  # 此时返回的结果，含有padding,调用者需要根据current_input_length切片
            pass
        else:
            # 取最后一个token的输出
            hidden_states = self.llm_gather(hidden_states, current_input_length - 1)
        hidden_states = self.norm(hidden_states)
        # last_hidden_state = hidden_states

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
        )


@XHLLM_TRACEABLE_MODULES.register_module(
    {
        Qwen3TTSTalkerCodePredictorModelForConditionalGeneration: "Qwen3TTSTalkerCodePredictorModelForConditionalGeneration",
    }
)
class _Qwen3TTSTalkerCodePredictorModelForConditionalGeneration(DynamicModule):
    def _setup(self, cfg: dict | None = None):
        return self

    def forward(
        self,
        inputs_embeds: Tensor | None = None,
        past_seq_length: Tensor | None = None,
        current_input_length: Tensor | None = None,
        past_key_cache: list[Tensor] | None = None,
        past_value_cache: list[Tensor] | None = None,
        generation_steps: Tensor | None = None,
    ):
        inputs_embeds = self.small_to_mtp_projection(inputs_embeds)
        outputs: BaseModelOutputWithPast = self.model(
            # position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            past_seq_length=past_seq_length,
            current_input_length=current_input_length,
            past_key_cache=past_key_cache,
            past_value_cache=past_value_cache,
        )
        hidden_states = outputs.last_hidden_state
        weight_t_flat = self.weight_embedding(generation_steps)  # [1024*2048]
        weight_t = weight_t_flat.view(self.in_features, self.out_features)  # [1024, 2048]

        # 使用 matmul 计算
        # x: [batch, seq, 1024], weight_t: [1024, 2048]
        # result: [batch, seq, 2048]
        logits = torch.matmul(hidden_states, weight_t)

        return Qwen3TTSTalkerCodePredictorOutputWithPast(
            logits=logits,
            past_key_values=outputs.past_key_values,
        )


def register_wrap_modules(talker: Qwen3TTSTalkerCodePredictorModelForConditionalGeneration | None = None):
    pass
