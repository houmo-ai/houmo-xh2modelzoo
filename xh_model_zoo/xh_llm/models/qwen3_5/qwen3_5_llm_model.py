# Copyright 2025 HOUMO AI
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

from copy import deepcopy
from typing import List, Optional, cast

import torch
import torch.nn as nn
from accelerate import init_empty_weights
from torch import Tensor
from transformers import AutoConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

try:
    from transformers.modeling_utils import no_init_weights
except ImportError:
    no_init_weights = init_empty_weights

from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5ForConditionalGeneration,
    Qwen3_5ForCausalLM,
)

from xhquant.core import CacheTensor
from xhquant.nn import MatMul

from ..base_llm_model import LLMBaseModel
from ..builder import MODELS


def _get_text_model(model):
    """Get the text model backbone from either ForConditionalGeneration or ForCausalLM."""
    if isinstance(model, Qwen3_5ForConditionalGeneration):
        return model.model.language_model
    elif isinstance(model, Qwen3_5ForCausalLM):
        return model.model
    else:
        raise ValueError(f"Unsupported model type: {type(model)}")


def _get_text_config(model):
    """Get text_config from the model config."""
    config = model.config
    if hasattr(config, "text_config"):
        return config.text_config
    return config


@MODELS.register_module()
class XHQwen3_5Model(LLMBaseModel):
    def __init__(
        self,
        hf_model: str,
        wrap_cfg,
        quant_config,
        frontend_type="TorchFX",
        allow_quant=True,
        export_cfg=None,
    ):
        super().__init__(
            hf_model,
            wrap_cfg,
            quant_config,
            frontend_type,
            allow_quant=allow_quant,
            export_cfg=export_cfg,
        )

    def get_hf_model(self, device_map="cpu", **kwargs):
        hf_model = super().get_hf_model(device_map, **kwargs)
        assert isinstance(hf_model, (Qwen3_5ForConditionalGeneration, Qwen3_5ForCausalLM))
        text_model = _get_text_model(hf_model)
        assert id(hf_model.get_output_embeddings().weight) != id(
            text_model.get_input_embeddings().weight
        )
        return hf_model

    def _load_hf_model(self, device_map="cpu", **kwargs):
        config = AutoConfig.from_pretrained(self.hf_model_dir, trust_remote_code=True)
        quantization_config = getattr(config, "quantization_config", None)
        quant_method = getattr(quantization_config, "quant_method", None)
        if isinstance(quantization_config, dict):
            quant_method = quantization_config.get("quant_method", quant_method)
        if str(quant_method).lower() == "gptq":
            return self._load_gptqmodel(device_map, **kwargs)

        if "torch_dtype" not in kwargs:
            kwargs["torch_dtype"] = torch.float16

        model_cls = Qwen3_5ForConditionalGeneration
        architectures = getattr(config, "architectures", [])
        if architectures and architectures[0] == "Qwen3_5ForCausalLM":
            model_cls = Qwen3_5ForCausalLM

        if device_map == "meta":
            with no_init_weights():
                hf_model = model_cls(config)
        else:
            hf_model = model_cls.from_pretrained(
                self.hf_model_dir,
                trust_remote_code=True,
                device_map=device_map,
                **kwargs,
            ).eval()
        return hf_model

    def _load_gptqmodel(self, device_map="cpu", **kwargs):
        from gptqmodel import GPTQModel

        trust_remote_code = bool(kwargs.pop("trust_remote_code", True))
        backend = kwargs.pop("backend", "torch")
        valid_string_device_maps = {"auto", "balanced", "balanced_low_0", "sequential"}
        load_kwargs = {
            "backend": backend,
            "trust_remote_code": trust_remote_code,
            **kwargs,
        }

        if isinstance(device_map, dict):
            load_kwargs["device_map"] = device_map
        elif isinstance(device_map, str):
            if device_map in valid_string_device_maps:
                load_kwargs["device_map"] = device_map
            elif device_map != "meta":
                load_kwargs["device"] = device_map
        elif device_map is not None:
            load_kwargs["device"] = device_map

        if "device_map" in load_kwargs and "device" not in load_kwargs:
            load_kwargs["device"] = "cuda:0" if torch.cuda.is_available() else "cpu"

        try:
            q_model = GPTQModel.load(self.hf_model_dir, **load_kwargs)
        except TypeError:
            load_kwargs.pop("backend", None)
            q_model = GPTQModel.load(self.hf_model_dir, **load_kwargs)

        hf_model = q_model.model.eval()
        qcfg = getattr(hf_model.config, "quantization_config", None)
        if isinstance(qcfg, dict):
            try:
                from transformers.utils.quantization_config import GPTQConfig

                hf_model.config.quantization_config = GPTQConfig.from_dict(qcfg)
            except Exception:
                pass
        return hf_model

    def _set_dtype(self, dtype):
        self.token_embedding = self.token_embedding.to(dtype)
        if hasattr(self, "past_conv_caches"):
            self.past_conv_caches = [t.to(dtype) for t in self.past_conv_caches]
        if hasattr(self, "past_recurrent_states"):
            self.past_recurrent_states = [t.to(dtype) for t in self.past_recurrent_states]
        return super()._set_dtype(dtype)

    def _set_device(self, device):
        device = torch.device(device)
        if device != torch.device("meta"):
            if hasattr(self, "past_conv_caches"):
                self.past_conv_caches = [t.to(device) for t in self.past_conv_caches]
            if hasattr(self, "past_recurrent_states"):
                self.past_recurrent_states = [t.to(device) for t in self.past_recurrent_states]
        return super()._set_device(device)

    def init_wrap_model(self, hf_model=None):
        from ._model import register_wrap_modules

        register_wrap_modules()
        super().init_wrap_model(hf_model)
        hf_model = self.wrap_model
        self.split_conv_cache = self.wrap_cfg.get("split_conv_cache", False)

        text_model = _get_text_model(hf_model)
        text_config = _get_text_config(hf_model)

        self.token_embedding = deepcopy(text_model.get_input_embeddings())
        self.generation_config = hf_model.generation_config
        self.config = hf_model.config
        self.num_hidden_layers = text_config.num_hidden_layers
        self.head_dim = text_config.head_dim
        self.pad_token_id = getattr(hf_model.config, "eos_token_id", None) or getattr(
            text_config, "eos_token_id", 151645
        )
        batch_size = self.wrap_cfg.batch_size

        self.layer_types = list(text_config.layer_types)
        self.full_attention_layer_indices = [
            i for i, layer_type in enumerate(self.layer_types) if layer_type == "full_attention"
        ]
        self.linear_attention_layer_indices = [
            i for i, layer_type in enumerate(self.layer_types) if layer_type == "linear_attention"
        ]
        self.num_full_attention_layers = len(self.full_attention_layer_indices)
        self.num_linear_attention_layers = len(self.linear_attention_layer_indices)

        base_input_names = [
            "inputs_embeds",
            "time_position_ids",
            "hight_position_ids",
            "width_position_ids",
            "past_seq_length",
            "current_input_length",
            "linear_attn_mask",
        ]
        if self.export_cfg is not None:
            self.export_cfg.input_names = base_input_names

        self.past_conv_caches = []
        self.past_recurrent_states = []

        if self.use_cache:
            only_first_block = self.wrap_cfg.get("only_first_block", False)
            max_layers = self.wrap_cfg.get("max_layers", None)
            if only_first_block:
                num_full_attn_layers = 1 if self.layer_types[0] == "full_attention" else 0
                num_linear_attn_layers = 1 if self.layer_types[0] == "linear_attention" else 0
            elif max_layers is not None and max_layers > 0:
                truncated_types = self.layer_types[:max_layers]
                num_full_attn_layers = sum(1 for t in truncated_types if t == "full_attention")
                num_linear_attn_layers = sum(1 for t in truncated_types if t == "linear_attention")
            else:
                num_full_attn_layers = self.num_full_attention_layers
                num_linear_attn_layers = self.num_linear_attention_layers

            self.prepare_kv_cache(
                num_full_attn_layers,
                [
                    batch_size,
                    text_config.num_key_value_heads,
                    self.cache_length,
                    self.head_dim,
                ],
            )
            self.prepare_linear_cache(num_linear_attn_layers)
        else:
            self.prepare_kv_cache(0, [batch_size, 1, 0, 1])
            self.prepare_linear_cache(self.num_linear_attention_layers)

        hf_model = None

    def _get_wrap_text_model(self):
        """Get text model from the wrapped model."""
        return _get_text_model(self.wrap_model)

    def prepare_linear_cache(self, num_linear_attention_layers: int):
        self.past_conv_caches = []
        self.past_recurrent_states = []
        linear_layer_indices = self.linear_attention_layer_indices[:num_linear_attention_layers]

        batch_size = self.wrap_cfg.batch_size
        split_conv_cache = getattr(self, "split_conv_cache", False)
        text_model = self._get_wrap_text_model()
        for layer_idx in linear_layer_indices:
            layer = text_model.layers[layer_idx]
            assert layer.layer_type == "linear_attention", f"Layer {layer_idx} should be linear_attention"
            linear_attn = layer.linear_attn
            recurrent_cache_shape = [
                batch_size,
                linear_attn.num_v_heads,
                linear_attn.head_k_dim,
                linear_attn.head_v_dim,
            ]

            if split_conv_cache:
                cache_dtype = (
                    linear_attn.conv1d_q.weight.dtype
                    if hasattr(linear_attn, "conv1d_q")
                    else linear_attn.conv1d.weight.dtype
                )
                # q: [batch_size, key_dim, conv_kernel_size]
                conv_cache_q_shape = [batch_size, linear_attn.key_dim, linear_attn.conv_kernel_size]
                # k: [batch_size, key_dim, conv_kernel_size]
                conv_cache_k_shape = [batch_size, linear_attn.key_dim, linear_attn.conv_kernel_size]
                # v: [batch_size, value_dim, conv_kernel_size]
                conv_cache_v_shape = [batch_size, linear_attn.value_dim, linear_attn.conv_kernel_size]
                self.past_conv_caches.append(CacheTensor(torch.zeros(conv_cache_q_shape, dtype=cache_dtype)))
                self.past_conv_caches.append(CacheTensor(torch.zeros(conv_cache_k_shape, dtype=cache_dtype)))
                self.past_conv_caches.append(CacheTensor(torch.zeros(conv_cache_v_shape, dtype=cache_dtype)))
            else:
                cache_dtype = linear_attn.conv1d.weight.dtype
                conv_cache_shape = [batch_size, linear_attn.conv_dim, linear_attn.conv_kernel_size]
                self.past_conv_caches.append(CacheTensor(torch.zeros(conv_cache_shape, dtype=cache_dtype)))

            self.past_recurrent_states.append(CacheTensor(torch.zeros(recurrent_cache_shape, dtype=cache_dtype)))

        if self.export_cfg is not None:
            if split_conv_cache:
                for cache_idx in range(num_linear_attention_layers):
                    for branch in ("q", "k", "v"):
                        self.export_cfg.input_names.append(f"past_conv_cache_{branch}_{cache_idx}")
            else:
                for cache_idx in range(num_linear_attention_layers):
                    self.export_cfg.input_names.append(f"past_conv_cache_{cache_idx}")
            for cache_idx in range(num_linear_attention_layers):
                self.export_cfg.input_names.append(f"past_recurrent_state_{cache_idx}")
            if self.use_cache:
                output_names = ["logits"]
                # Verify-intermediates path expands BOTH conv_cache and
                # recurrent_state into per-step snapshots so the runtime can
                # pick the snapshot for the committed accepted_steps by name
                # without slicing.
                verify_steps = 1
                if (
                    bool(self.wrap_cfg.get("verify_output_intermediates", False))
                    and int(self.wrap_cfg.get("input_sequence_length", 1)) > 1
                ):
                    verify_steps = int(self.wrap_cfg.input_sequence_length)
                if verify_steps > 1:
                    if split_conv_cache:
                        for cache_idx in range(num_linear_attention_layers):
                            for branch in ("q", "k", "v"):
                                for step_idx in range(verify_steps):
                                    output_names.append(
                                        f"conv_cache_out_{branch}_{cache_idx}_{step_idx}"
                                    )
                    else:
                        for cache_idx in range(num_linear_attention_layers):
                            for step_idx in range(verify_steps):
                                output_names.append(
                                    f"conv_cache_out_{cache_idx}_{step_idx}"
                                )
                    for cache_idx in range(num_linear_attention_layers):
                        for step_idx in range(verify_steps):
                            output_names.append(
                                f"recurrent_state_out_{cache_idx}_{step_idx}"
                            )
                else:
                    if split_conv_cache:
                        for cache_idx in range(num_linear_attention_layers):
                            for branch in ("q", "k", "v"):
                                output_names.append(f"conv_cache_out_{branch}_{cache_idx}")
                    else:
                        for cache_idx in range(num_linear_attention_layers):
                            output_names.append(f"conv_cache_out_{cache_idx}")
                    for cache_idx in range(num_linear_attention_layers):
                        output_names.append(f"recurrent_state_out_{cache_idx}")
                # Add spec_decode_hidden output if configured
                if self.wrap_cfg.get("output_hidden_state_indices") is not None:
                    output_names.append("target_hidden")
                elif self.wrap_cfg.get("output_post_norm_hidden", False):
                    output_names.append("post_norm_hidden")
                self.export_cfg.output_names = output_names

    def prepare_inputs(self, data):
        raw_input_ids = data.get("input_ids", None)
        if isinstance(raw_input_ids, torch.Tensor):
            raw_input_ids = raw_input_ids.tolist()
        assert raw_input_ids is not None, "input_ids is required"

        if raw_input_ids and isinstance(raw_input_ids[0], int):
            raw_input_ids = [raw_input_ids]

        past_seq_lengths = data.get("past_seq_length", 0)
        if isinstance(past_seq_lengths, int):
            past_seq_lengths = [past_seq_lengths for _ in range(len(raw_input_ids))]

        explicit_current_input_length = data.get("current_input_length", None)
        if explicit_current_input_length is not None and isinstance(explicit_current_input_length, int):
            explicit_current_input_length = [explicit_current_input_length for _ in range(len(raw_input_ids))]

        device = self.execution_device
        input_ids = []
        current_input_length = []
        linear_attn_mask = []

        for batch_idx, input_id in enumerate(raw_input_ids):
            input_id = torch.tensor(input_id, dtype=torch.long)
            seq_length = input_id.shape[0]
            past_seq_length = past_seq_lengths[batch_idx]

            if explicit_current_input_length is not None:
                actual_length = explicit_current_input_length[batch_idx]
            else:
                actual_length = seq_length
            current_input_length.append(actual_length)

            assert seq_length <= self.input_sequence_length, (
                f"Input sequence length is too long. max input sequence length is "
                f"{self.input_sequence_length} but got {seq_length}"
            )

            if self.input_sequence_length > actual_length:
                pad_len = self.input_sequence_length - actual_length
                mask = torch.cat(
                    [
                        torch.ones(actual_length, device=input_id.device),
                        torch.zeros(pad_len, device=input_id.device),
                    ],
                    dim=0,
                )
            else:
                mask = torch.ones(actual_length, device=input_id.device)

            input_ids.append(input_id.unsqueeze(0))
            linear_attn_mask.append(mask.unsqueeze(0))

        input_ids = torch.cat(input_ids, dim=0).to(device)
        mask_dtype = self.dtype if self.dtype is not None else torch.float32
        linear_attn_mask = torch.cat(linear_attn_mask, dim=0).to(device=device, dtype=mask_dtype)
        current_input_length = torch.tensor(current_input_length, dtype=torch.int32).to(device)

        self.token_embedding.to(device)
        inputs_embeds = self.token_embedding(input_ids)

        past_seq_length = torch.tensor(past_seq_lengths, dtype=torch.int32).to(device)
        assert torch.all(past_seq_length >= 0)

        # M-RoPE position IDs: for text-only, all three are the same sequential positions
        batch_size = len(raw_input_ids)
        seq_len = inputs_embeds.shape[1]
        position_ids_list = []
        for batch_idx in range(batch_size):
            start = past_seq_lengths[batch_idx]
            pos = torch.arange(start, start + seq_len, dtype=torch.long, device=device)
            position_ids_list.append(pos.unsqueeze(0))
        position_ids = torch.cat(position_ids_list, dim=0)
        time_position_ids = position_ids
        hight_position_ids = position_ids
        width_position_ids = position_ids

        past_key_caches = self.past_key_caches
        past_value_caches = self.past_value_caches
        past_conv_caches = self.past_conv_caches
        past_recurrent_states = self.past_recurrent_states
        if past_conv_caches is not None and len(past_conv_caches) > 0 and past_conv_caches[0].device != device:
            past_conv_caches = [t.to(device) for t in past_conv_caches]
            self.past_conv_caches = past_conv_caches
        if (
            past_recurrent_states is not None
            and len(past_recurrent_states) > 0
            and past_recurrent_states[0].device != device
        ):
            past_recurrent_states = [t.to(device) for t in past_recurrent_states]
            self.past_recurrent_states = past_recurrent_states

        return (
            inputs_embeds.to(device),
            time_position_ids,
            hight_position_ids,
            width_position_ids,
            past_seq_length.to(device),
            current_input_length,
            linear_attn_mask.to(device),
            past_key_caches,
            past_value_caches,
            past_conv_caches,
            past_recurrent_states,
        )

    def prepare_inputs_for_graph(self, data):
        return self.prepare_inputs(data)

    def _forward(
        self,
        inputs_embeds: Tensor,
        time_position_ids: Tensor,
        hight_position_ids: Tensor,
        width_position_ids: Tensor,
        past_seq_length: Tensor,
        current_input_length: Tensor,
        linear_attn_mask: Optional[Tensor],
        past_key_caches: List[Tensor],
        past_value_caches: List[Tensor],
        past_conv_caches: List[Tensor],
        past_recurrent_states: List[Tensor],
    ):
        out = self(
            inputs_embeds,
            time_position_ids,
            hight_position_ids,
            width_position_ids,
            past_seq_length,
            current_input_length,
            linear_attn_mask,
            past_key_caches,
            past_value_caches,
            past_conv_caches,
            past_recurrent_states,
        )
        logits = out[0]
        conv_cache_out_list = out[1]
        recurrent_state_out_list = out[2]

        verify_output_intermediates = bool(self.wrap_cfg.get("verify_output_intermediates", False))
        if self.use_cache and conv_cache_out_list is not None and not verify_output_intermediates:
            for idx, conv_cache_out in enumerate(conv_cache_out_list):
                if idx < len(self.past_conv_caches):
                    self.past_conv_caches[idx].copy_(conv_cache_out)
        if self.use_cache and recurrent_state_out_list is not None and not verify_output_intermediates:
            for idx, recurrent_state_out in enumerate(recurrent_state_out_list):
                if idx < len(self.past_recurrent_states):
                    self.past_recurrent_states[idx].copy_(recurrent_state_out)

        result = CausalLMOutputWithPast(logits=logits)
        # Propagate extra hidden state for speculative decoding
        if len(out) > 3:
            result.spec_decode_hidden = out[3]
        return result

    def set_linear_attention_mode(self, mode: str):
        self.wrap_cfg.linear_attention_mode = mode
        self.update_cfg(self.wrap_cfg)

    def convert_to_quant_graph(self, target_device: str = "xh2a"):
        if self._frontend_model is not None:
            cumsum_quant_cfg = self.wrap_cfg.get("cumsum_matmul_quant_config", None)
            if cumsum_quant_cfg is None:
                cumsum_quant_cfg = dict(
                    act_schema=dict(
                        fp_mode="sefp",
                        man_bit=16,
                    ),
                    act_schema_2=dict(
                        fp_mode="fp16",
                        man_bit=8,
                    ),
                )

            for node in self._frontend_model.graph.nodes:
                if node.op == "call_module":
                    module = self._frontend_model.get_submodule(node.target)
                    if isinstance(module, MatMul) and "cumsum_matmul" in node.target:
                        node.meta["quant_config"] = deepcopy(cumsum_quant_cfg)

        return super().convert_to_quant_graph(target_device)

    def reset_kvcache(self):
        super().reset_kvcache()
        self.reset_linear_cache()

    def reset_linear_cache(self):
        if hasattr(self, "past_conv_caches") and self.past_conv_caches is not None:
            for cache in self.past_conv_caches:
                if hasattr(cache, "reset"):
                    cache.reset()
                elif isinstance(cache, torch.Tensor):
                    cache.zero_()
        if hasattr(self, "past_recurrent_states") and self.past_recurrent_states is not None:
            for cache in self.past_recurrent_states:
                if hasattr(cache, "reset"):
                    cache.reset()
                elif isinstance(cache, torch.Tensor):
                    cache.zero_()
