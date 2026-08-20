import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from transformers.cache_utils import EncoderDecoderCache


try:
    from transformers.models.whisper.modeling_whisper import WhisperSdpaAttention
except ImportError:
    WhisperSdpaAttention = None
try:
    from transformers.models.whisper.modeling_whisper import WhisperAttention
except ImportError:
    WhisperAttention = None

from transformers.models.whisper.modeling_whisper import WhisperEncoder, WhisperEncoderLayer

from xhquant.nn import LLMCache

from ...register import DynamicRegister, XHLLM_TRACEABLE_MODULES
from ._audio_streaming import StreamingWhisperEncoderMixin


def _create_whisper_attention_wrapper(cls_to_wrap):
    """Factory function to create a wrapper class for Whisper attention classes."""

    @XHLLM_TRACEABLE_MODULES.register_module(
        {
            cls_to_wrap: cls_to_wrap.__name__,
        }
    )
    class _WhisperAttentionWrapper(DynamicRegister):
        def _setup(self, cfg: Optional[Dict] = None):
            _kv_scale = 1 / math.sqrt(self.head_dim)
            self.register_buffer("kv_scale", torch.tensor(_kv_scale, dtype=torch.float16), persistent=False)
            self.k_cache = LLMCache(axis=2)
            self.v_cache = LLMCache(axis=2)

        def _shape_states(self, states: torch.Tensor, bsz: int) -> torch.Tensor:
            if hasattr(states, "reshape"):
                shaped = states.reshape(bsz, -1, self.num_heads, self.head_dim)
            else:
                try:
                    shaped = torch.reshape(states, (bsz, -1, self.num_heads, self.head_dim))
                except Exception:
                    raw_states = getattr(states, "_data", getattr(states, "data", states))
                    shaped = raw_states.reshape(bsz, -1, self.num_heads, self.head_dim)
            shaped = shaped.transpose(1, 2)
            if hasattr(shaped, "contiguous"):
                return shaped.contiguous()
            return shaped

        def forward(
            self,
            hidden_states: torch.Tensor,
            key_value_states: Optional[torch.Tensor] = None,
            past_key_value: Optional[EncoderDecoderCache] = None,
            attention_mask: Optional[torch.Tensor] = None,
            layer_head_mask: Optional[torch.Tensor] = None,
            output_attentions: bool = False,
            cache_position: Optional[torch.LongTensor] = None,
            past_seq_length: Optional[Tensor] = None,
            current_input_length: Optional[Tensor] = None,
            past_k_cache: Optional[Tensor] = None,
            past_v_cache: Optional[Tensor] = None,
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
            """Input shape: Batch x Time x Channel"""
            if output_attentions or layer_head_mask is not None:
                raise NotImplementedError()

            is_cross_attention = key_value_states is not None
            if hasattr(hidden_states, "shape"):
                bsz, tgt_len, _ = hidden_states.shape
            elif hasattr(hidden_states, "size"):
                bsz, tgt_len, _ = hidden_states.size()
            else:
                data = getattr(hidden_states, "_data", getattr(hidden_states, "data", hidden_states))
                bsz, tgt_len, _ = data.shape if hasattr(data, "shape") else data.size()

            query_states = self._shape_states(self.q_proj(hidden_states), bsz)

            if past_key_value is not None:
                is_updated = past_key_value.is_updated.get(self.layer_idx)
                if is_cross_attention:
                    past_key_value.is_updated[self.layer_idx] = True
                    past_key_value = past_key_value.cross_attention_cache
                else:
                    past_key_value = past_key_value.self_attention_cache

            current_states = key_value_states if key_value_states is not None else hidden_states
            if is_cross_attention and past_key_value and is_updated:
                if hasattr(past_key_value, "layers"):
                    key_states = past_key_value.layers[self.layer_idx].keys
                    value_states = past_key_value.layers[self.layer_idx].values
                else:
                    key_states = past_key_value.key_cache[self.layer_idx]
                    value_states = past_key_value.value_cache[self.layer_idx]
            else:
                key_states = self._shape_states(self.k_proj(current_states), bsz)
                value_states = self._shape_states(self.v_proj(current_states), bsz)
                if past_k_cache is not None and past_v_cache is not None:
                    key_states = self.k_cache(key_states, past_seq_length, current_input_length, past_k_cache)
                    value_states = self.v_cache(value_states, past_seq_length, current_input_length, past_v_cache)
                elif past_key_value is not None:
                    cache_position = cache_position if not is_cross_attention else None
                    key_states, value_states = past_key_value.update(
                        key_states, value_states, self.layer_idx, {"cache_position": cache_position}
                    )

            present_key_states = key_states
            present_value_states = value_states
            attention_key_states = key_states.transpose(2, 3) * self.kv_scale
            attn_weights = torch.matmul(query_states, attention_key_states)
            mask = attention_mask if attention_mask is not None else 0
            attn_weights: Optional[Tensor] = torch.nn.functional.softmax(attn_weights + mask, dim=-1)
            attn_output = torch.matmul(attn_weights, value_states)

            attn_output = attn_output.transpose(1, 2)
            attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
            attn_output = self.out_proj(attn_output)

            return attn_output, None, past_key_value, present_key_states, present_value_states

    return _WhisperAttentionWrapper


if WhisperSdpaAttention is not None:
    _WhisperSdpaAttention = _create_whisper_attention_wrapper(WhisperSdpaAttention)

if WhisperAttention is not None:
    _WhisperAttention = _create_whisper_attention_wrapper(WhisperAttention)


@XHLLM_TRACEABLE_MODULES.register_module(
    {
        WhisperEncoderLayer: "minicpmo.audio.MiniCPMWhisperEncoderLayer",
    }
)
class _MiniCPMWhisperEncoderLayer(DynamicRegister):
    def _setup(self, cfg: Optional[Dict] = None):
        self.clamp_value = torch.finfo(torch.float16).max - 1000

    @staticmethod
    def _safe_dropout(x, p: float, training: bool):
        if isinstance(x, torch.Tensor):
            return nn.functional.dropout(x, p=p, training=training)
        return x

    @staticmethod
    def _safe_add(a, b):
        try:
            return a + b
        except TypeError:
            pass

        def get_raw_tensor(x):
            if hasattr(x, "_data"):
                return x._data
            if hasattr(x, "data"):
                return x.data
            return x

        raw_a = get_raw_tensor(a)
        raw_b = get_raw_tensor(b)
        try:
            return raw_a + raw_b
        except TypeError:
            pass
        return torch.add(raw_a, raw_b)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_head_mask: torch.Tensor,
        output_attentions: bool = False,
        past_key_values: Optional[EncoderDecoderCache] = None,
        use_cache: Optional[bool] = False,
        past_seq_length: Optional[Tensor] = None,
        current_input_length: Optional[Tensor] = None,
        past_k_cache: Optional[Tensor] = None,
        past_v_cache: Optional[Tensor] = None,
    ):
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states, attn_weights, past_key_values, present_key_cache, present_value_cache = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
            past_key_value=past_key_values,
            past_seq_length=past_seq_length,
            current_input_length=current_input_length,
            past_k_cache=past_k_cache,
            past_v_cache=past_v_cache,
        )
        hidden_states = self._safe_dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = self._safe_add(residual, hidden_states)

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = self._safe_dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = self._safe_dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = self._safe_add(residual, hidden_states)

        clamp_value = self.clamp_value
        hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        if use_cache:
            outputs += (past_key_values,)

        outputs += (present_key_cache, present_value_cache)

        return outputs


@XHLLM_TRACEABLE_MODULES.register_module(
    {
        WhisperEncoder: "minicpmo.audio.MiniCPMWhisperEncoder",
    }
)
class _MiniCPMWhisperEncoder(StreamingWhisperEncoderMixin, DynamicRegister):
    def _setup(self, cfg: Optional[Dict] = None):
        pass

    @torch.inference_mode()
    def forward(
        self,
        input_features: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
    ):
        inputs_embeds = nn.functional.gelu(self.conv1(input_features))
        inputs_embeds = nn.functional.gelu(self.conv2(inputs_embeds))
        hidden_states = inputs_embeds.permute(0, 2, 1)

        encoder_states = ()

        for layer in self.layers:
            encoder_states = encoder_states + (hidden_states,)

            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                layer_head_mask=None,
                output_attentions=False,
                past_key_values=None,
            )

            hidden_states = layer_outputs[0]

        hidden_states = self.layer_norm(hidden_states)
        encoder_states = encoder_states + (hidden_states,)

        audio_states = encoder_states[self.audio_encoder_layer]
        audio_embeds = self.audio_projection_layer(audio_states)
        audio_embeds = self.audio_avg_pooler(audio_embeds.transpose(1, 2))
        return audio_embeds.transpose(1, 2)

    def _wrap_to_hf_compatible(self):
        self.__class__ = WhisperEncoder
        self.embed_audio.__class__ = type(self.embed_audio).__bases__[0]
        self.layer_norm.__class__ = type(self.layer_norm).__bases__[0]

        for layer in self.layers:
            layer.__class__ = WhisperEncoderLayer
            layer.self_attn.__class__ = type(layer.self_attn).__bases__[0]
            layer.self_attn_layer_norm.__class__ = type(layer.self_attn_layer_norm).__bases__[0]
            layer.fc1.__class__ = type(layer.fc1).__bases__[0]
            layer.fc2.__class__ = type(layer.fc2).__bases__[0]
            layer.final_layer_norm.__class__ = type(layer.final_layer_norm).__bases__[0]


def register_wrap_cls(hf_model):
    encoder_cls = type(hf_model.apm)
    layer_cls = type(hf_model.apm.layers[0])
    if layer_cls not in XHLLM_TRACEABLE_MODULES:
        XHLLM_TRACEABLE_MODULES.register_module({layer_cls: layer_cls.__name__}, _MiniCPMWhisperEncoderLayer)
    if encoder_cls not in XHLLM_TRACEABLE_MODULES:
        XHLLM_TRACEABLE_MODULES.register_module({encoder_cls: encoder_cls.__name__}, _MiniCPMWhisperEncoder)
