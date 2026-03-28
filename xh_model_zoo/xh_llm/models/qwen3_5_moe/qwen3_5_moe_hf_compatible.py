# Copyright 2025 HOUMO AI
#
# File: qwen3_5_moe_hf_compatible.py
# Description:
#   Qwen3.5-MoE HF-compatible wrapper for lm_eval / generate().
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

from contextlib import contextmanager
from typing import Optional, Union

import torch
from accelerate import init_empty_weights
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, DynamicCache
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast
try:
    from transformers.modeling_utils import no_init_weights
except ImportError:
    import contextlib

    @contextlib.contextmanager
    def no_init_weights(_enable=True):
        yield
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeForConditionalGeneration,
)
from xhquant.core import CacheTensor

from .inference import Qwen3_5MoeInference


def get_empty_hf_model(hf_model_dir: str, **kwargs) -> Qwen3_5MoeForConditionalGeneration:
    """Load model structure only, without initialising weights.

    Qwen3.5-MoE ships as ``Qwen3_5MoeForConditionalGeneration`` whose outer
    config has no top-level ``vocab_size``.  ``AutoModelForCausalLM`` maps
    this architecture to ``Qwen3_5MoeForCausalLM`` but still passes the
    outer config, which lacks ``vocab_size`` → AttributeError.
    Work-around: extract ``text_config`` and use it directly.
    """
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeForCausalLM

    config = AutoConfig.from_pretrained(hf_model_dir)
    text_config = getattr(config, "text_config", config)
    with no_init_weights(), init_empty_weights():
        hf_model = Qwen3_5MoeForCausalLM(text_config)
    return hf_model


class Qwen3_5MoeHFCompatible(Qwen3_5MoeForConditionalGeneration):
    """HF-compatible wrapper around Qwen3_5MoeInference for lm_eval / generate()."""

    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "Qwen3_5MoeHFCompatible cannot be initialized directly; use to_hf_compatible() instead."
        )

    def setup(self, llm_model: Qwen3_5MoeInference, embed_tokens: torch.nn.Module):
        self._is_prefill_phase = True
        self._past_seq_length = 0
        self._embed_tokens = embed_tokens
        self._llm_model = llm_model

    def get_output_embeddings(self):
        return None

    def _get_language_model_module(self):
        if hasattr(self._llm_model, "_get_language_model_module"):
            return self._llm_model._get_language_model_module()
        wrap_model = getattr(self._llm_model, "wrap_model", None)
        model = getattr(wrap_model, "model", None)
        if model is None:
            return None
        return getattr(model, "language_model", model)

    def _prepare_linear_state_caches(self, device: torch.device):
        """Ensure past_conv_caches / past_recurrent_states exist on device."""
        if not hasattr(self._llm_model, "past_conv_caches"):
            self._llm_model.past_conv_caches = []
        if not hasattr(self._llm_model, "past_recurrent_states"):
            self._llm_model.past_recurrent_states = []

        past_conv_caches = self._llm_model.past_conv_caches
        past_recurrent_states = self._llm_model.past_recurrent_states

        # Lazily build if not yet created (fallback from meta.json info)
        if len(past_conv_caches) == 0 or len(past_recurrent_states) == 0:
            language_model = self._get_language_model_module()
            if language_model is not None and hasattr(language_model, "layers"):
                layer_indices = getattr(self._llm_model, "linear_attention_layer_indices", None)
                if layer_indices is None:
                    layer_indices = [
                        i
                        for i, layer in enumerate(language_model.layers)
                        if getattr(layer, "layer_type", None) == "linear_attention"
                    ]
                batch_size = getattr(self._llm_model, "batch_size", 1)
                past_conv_caches, past_recurrent_states = [], []
                for layer_idx in layer_indices:
                    layer = language_model.layers[layer_idx]
                    linear_attn = layer.linear_attn
                    cache_dtype = linear_attn.conv1d.weight.dtype
                    past_conv_caches.append(
                        CacheTensor(
                            torch.zeros(
                                [batch_size, linear_attn.conv_dim, linear_attn.conv_kernel_size],
                                dtype=cache_dtype,
                                device=device,
                            )
                        )
                    )
                    past_recurrent_states.append(
                        CacheTensor(
                            torch.zeros(
                                [batch_size, linear_attn.num_v_heads, linear_attn.head_k_dim, linear_attn.head_v_dim],
                                dtype=cache_dtype,
                                device=device,
                            )
                        )
                    )
                self._llm_model.past_conv_caches = past_conv_caches
                self._llm_model.past_recurrent_states = past_recurrent_states

        past_conv_caches = self._llm_model.past_conv_caches
        past_recurrent_states = self._llm_model.past_recurrent_states

        # Device placement is handled inside _forward() via HMONNXInference's
        # to_device() calls, so we do NOT pre-move the caches here.
        return self._llm_model.past_conv_caches, self._llm_model.past_recurrent_states

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        use_cache = use_cache if use_cache is not None else getattr(self.config, "use_cache", True)

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds.")

        if not isinstance(past_key_values, (type(None), Cache)):
            raise ValueError("`past_key_values` must be a Cache object or None.")

        if inputs_embeds is None:
            self._embed_tokens.weight = torch.nn.Parameter(
                self._embed_tokens.weight.to(input_ids.device)
            )
            inputs_embeds = self._embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        # Determine phase BEFORE any computation so we dispatch to the correct ONNX.
        is_prefill = self._is_prefill_phase
        self._llm_model.set_phase_prefill(is_prefill)

        past_seq_length = torch.tensor([self._past_seq_length], dtype=torch.int32, device=inputs_embeds.device)
        seq_length = inputs_embeds.shape[1] if input_ids is None else input_ids.shape[-1]
        current_input_length = torch.tensor([seq_length], dtype=torch.int32, device=inputs_embeds.device)

        past_key_caches = self._llm_model.past_key_caches
        past_value_caches = self._llm_model.past_value_caches
        past_conv_caches, past_recurrent_states = self._prepare_linear_state_caches(inputs_embeds.device)

        if not is_prefill:
            # --- Decode phase: ISL=1, no padding, use decode ONNX ---
            device = inputs_embeds.device
            mask_dtype = inputs_embeds.dtype
            time_pos = (torch.tensor([self._past_seq_length], dtype=torch.int32, device=device)
                        .view(1, 1).expand(inputs_embeds.shape[0], -1).contiguous())
            hight_pos = time_pos
            width_pos = time_pos
            lin_mask = torch.ones(inputs_embeds.shape[0], 1, dtype=mask_dtype, device=device)
            out = self._llm_model._forward(
                inputs_embeds,
                time_pos, hight_pos, width_pos,
                past_seq_length, current_input_length,
                lin_mask,
                past_key_caches, past_value_caches,
                past_conv_caches, past_recurrent_states,
            )
            logits = out if isinstance(out, torch.Tensor) else (out[0] if isinstance(out, tuple) else out.logits)
        else:
            # --- Prefill phase: pad to ISL, use prefill ONNX (chunked for long prompts) ---
            net_input_seq_len = self._llm_model.get_input_sequence_length()
            steps = (seq_length + net_input_seq_len - 1) // net_input_seq_len

            # Pad embeds to multiple of net_input_seq_len
            padding_len = steps * net_input_seq_len - seq_length
            if padding_len > 0:
                padding_embeds = self._embed_tokens(
                    torch.zeros(inputs_embeds.shape[0], padding_len, dtype=torch.long, device=inputs_embeds.device)
                )
                inputs_embeds = torch.cat([inputs_embeds, padding_embeds], dim=1)

            padded_seq_len = inputs_embeds.shape[1]
            device = inputs_embeds.device
            mask_dtype = inputs_embeds.dtype

            # Build position IDs: real positions + repeat last position for padding
            # (matches reference prepare_inputs: last_pos.expand(pad_len))
            real_pos = torch.arange(
                self._past_seq_length, self._past_seq_length + seq_length,
                dtype=torch.int32, device=device,
            )
            if padding_len > 0:
                last_pos = real_pos[-1:].expand(padding_len)
                full_pos = torch.cat([real_pos, last_pos])
            else:
                full_pos = real_pos
            position_ids = full_pos.view(1, -1).expand(inputs_embeds.shape[0], -1)

            if steps > 1:
                all_logits = []
                for i in tqdm(range(steps)):
                    start = i * net_input_seq_len
                    end = (i + 1) * net_input_seq_len
                    sub_embeds = inputs_embeds[:, start:end, :]
                    sub_past_seq_length = past_seq_length + start
                    actual_len = min(end, seq_length) - start
                    sub_current_input_length = torch.tensor([actual_len], dtype=torch.int32, device=device)

                    sub_pos = position_ids[:, start:end]
                    time_pos = sub_pos
                    hight_pos = sub_pos
                    width_pos = sub_pos

                    # Always use all-ones lin_mask (matches reference generate())
                    lin_mask = torch.ones(1, net_input_seq_len, dtype=mask_dtype, device=device)

                    self._llm_model.set_input_sequence_length(int(sub_current_input_length.item()))
                    out = self._llm_model._forward(
                        sub_embeds,
                        time_pos, hight_pos, width_pos,
                        sub_past_seq_length, sub_current_input_length,
                        lin_mask,
                        past_key_caches, past_value_caches,
                        past_conv_caches, past_recurrent_states,
                    )
                    all_logits.append(out if isinstance(out, torch.Tensor) else (out[0] if isinstance(out, tuple) else out.logits))

                logits = torch.cat(all_logits, dim=1)
            else:
                time_pos = position_ids
                hight_pos = position_ids
                width_pos = position_ids
                # Always use all-ones lin_mask (matches reference generate())
                lin_mask = torch.ones(1, net_input_seq_len, dtype=mask_dtype, device=device)

                out = self._llm_model._forward(
                    inputs_embeds,
                    time_pos, hight_pos, width_pos,
                    past_seq_length, current_input_length,
                    lin_mask,
                    past_key_caches, past_value_caches,
                    past_conv_caches, past_recurrent_states,
                )
                logits = out if isinstance(out, torch.Tensor) else (out[0] if isinstance(out, tuple) else out.logits)

            # Switch to decode for all subsequent calls
            self._is_prefill_phase = False

        logits = logits[:, :seq_length, :]

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return CausalLMOutputWithPast(logits=logits, past_key_values=past_key_values)

    @classmethod
    def to_hf_compatible(
        cls,
        hf_model_or_path: Union[Qwen3_5MoeForConditionalGeneration, str],
        llm_model: Optional[Qwen3_5MoeInference] = None,
    ) -> Qwen3_5MoeForConditionalGeneration:
        """Wrap hf model with HMONNX inference engine."""
        if isinstance(hf_model_or_path, str):
            hf_model = get_empty_hf_model(hf_model_or_path)
            hf_model.__class__ = cls
            hf_model._is_prefill_phase = True
            hf_model._past_seq_length = 0
            hf_model._llm_model = None
            hf_model._embed_tokens = None
        else:
            hf_model = hf_model_or_path
            hf_model.__class__ = cls
            hf_model._is_prefill_phase = True
            hf_model._past_seq_length = 0
            hf_model._llm_model = None
            hf_model._embed_tokens = None

        if llm_model is not None:
            hf_model._llm_model = llm_model
            hf_model._embed_tokens = llm_model.token_embedding
            # Remove heavyweight model weights
            for attr in ("model", "lm_head"):
                if hasattr(hf_model, attr):
                    try:
                        delattr(hf_model, attr)
                    except Exception:
                        pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return hf_model

    def _sample_forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        out = self._xh_orig_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )
        seq_length = input_ids.shape[-1] if input_ids is not None else inputs_embeds.shape[-2]
        self._past_seq_length += seq_length
        if self._is_prefill_phase:
            self._is_prefill_phase = False
        return out

    def _prepare_position_ids_for_generation(self, inputs_tensor, model_kwargs):
        """Override: skip the ConditionalGeneration version that requires self.model.

        Our forward() computes M-RoPE position IDs internally, so we only need
        to return standard sequential 2D position IDs here to satisfy
        GenerationMixin bookkeeping without crashing.
        """
        from transformers.generation.utils import GenerationMixin

        return GenerationMixin._prepare_position_ids_for_generation(self, inputs_tensor, model_kwargs)

    @contextmanager
    def _optimize_model_for_decode(self, *args, **kwargs):
        """No-op: HMONNX switches to decode session via _is_prefill_phase flag."""
        yield

    def generate(self, *args, **kwargs):
        self._is_prefill_phase = True
        self._past_seq_length = 0
        self._llm_model.set_phase_prefill(True)  # reset to prefill ONNX
        self._xh_orig_forward = self.forward
        self.forward = self._sample_forward
        try:
            out = super().generate(*args, **kwargs)
        finally:
            self.forward = self._xh_orig_forward
            del self._xh_orig_forward
        return out
