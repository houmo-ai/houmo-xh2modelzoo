# Copyright 2025 HOUMO AI
#
# File: minicpmo_llm_model.py
# Description:
#   Minicpmo Llm Model model implementation.
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
from typing import Optional, Union

import torch
import torch.nn as nn
from transformers.models.qwen2 import Qwen2ForCausalLM
from xhquant import nn as xhnn
from xhquant.api import QuantGraph

from ..base_llm_model import LLMBaseModel
from ..builder import MODELS
from .minicpmo_base_model import XHMiniCPMOBaseModel


@MODELS.register_module()
class XHMiniCPMOLLMModel(LLMBaseModel, XHMiniCPMOBaseModel):
    def __init__(
        self,
        hf_model: str,
        wrap_cfg,
        quant_config,
        frontend_type,
        allow_quant=True,
        export_cfg=None,
    ):
        super().__init__(
            hf_model,
            wrap_cfg=wrap_cfg,
            quant_config=quant_config,
            frontend_type=frontend_type,
            allow_quant=allow_quant,
            export_cfg=export_cfg,
        )
        if "extra_cfg" in self.quant_cfg:
            self.extra_quant_cfg = self.quant_cfg.pop("extra_cfg")
        else:
            self.extra_quant_cfg = None

    def _set_dtype(self, dtype):
        self.token_embedding = self.token_embedding.to(dtype)
        return super()._set_dtype(dtype)

    def init_wrap_model(self, hf_model=None):
        if hf_model is None:
            hf_model = self.get_hf_model()

        from ._llm_model_impl import register_wrap_modules as qwen2_register_wrap_modules
        super().init_wrap_model(hf_model.llm)
        self._wrap_model.chat_tts_projector = hf_model.tts.projector
      
        self.token_embedding = deepcopy(self._wrap_model.model.get_input_embeddings())
        self.generation_config = self._wrap_model.generation_config
        self.config = self._wrap_model.config
        self.num_hidden_layers = self._wrap_model.model.config.num_hidden_layers
        head_dim = self._wrap_model.model.layers[0].self_attn.head_dim
        self.pad_token_id = self._wrap_model.config.eos_token_id
        self.head_dim = head_dim
        if self.use_cache:
            num_decoder_layers = self.num_hidden_layers
            only_first_block = self.wrap_cfg.get("only_first_block", False)
            if only_first_block:
                num_decoder_layers = 1
            self.prepare_kv_cache(
                num_decoder_layers,
                [1, self._wrap_model.model.config.num_key_value_heads, self.cache_length, head_dim],
            )
        del hf_model

    def get_hf_model(self, device_map="cpu", **kwargs):
        hf_model = super().get_hf_model(device_map=device_map, **kwargs)
        return hf_model

    def prepare_inputs(self, data: Union[dict, tuple, list]):
        inputs_embeds = data["inputs_embeds"]
        assert self.token_embedding is not None, "Token embedding is not available."
        assert inputs_embeds.shape[0] == 1, "Batch size should be 1 in inference mode."
        seq_length = inputs_embeds.shape[1]
        inputs_embeds = inputs_embeds.to(self.execution_device)

        past_seq_length = data["past_seq_length"]
        assert past_seq_length >= 0, "past_seq_length should be non-negative."
        past_key_caches = self.past_key_caches
        past_value_caches = self.past_value_caches

        return (
            # position_ids.to(self.device),
            inputs_embeds.to(self.execution_device),
            torch.tensor([past_seq_length], dtype=torch.int32).to(
                self.execution_device),
            torch.tensor([seq_length], dtype=torch.int32).to(
                self.execution_device),
            past_key_caches,
            past_value_caches,
        )

    def prepare_inputs_for_graph(self, data: Union[dict, tuple, list]):
        inputs_embeds, past_seq_length, seg_length, past_key_caches, past_value_caches = self.prepare_inputs(data)
        return (
            inputs_embeds[:, : self.input_sequence_length, :],
            past_seq_length,
            torch.tensor([self.input_sequence_length], dtype=torch.int32).to(
                self.execution_device),
            past_key_caches,
            past_value_caches,
        )
    
    def test_step(self, data: Union[dict, tuple, list]):
        input_embeds = data["inputs_embeds"]
        input_seq_len = input_embeds.shape[1]
        steps = (input_seq_len + self.input_sequence_length - 1) // self.input_sequence_length
        inputs = self.prepare_inputs(data)

        inputs_embeds, past_seq_length, seg_length, past_key_caches, past_value_caches = inputs

        input_sequence_length = self.input_sequence_length
        hidden_states = list()
        for i in range(steps):
            start = i * self.input_sequence_length
            end = (i + 1) * self.input_sequence_length
            current_input_length = min(end, input_seq_len) - start

            sub_inputs_embeds = inputs_embeds[:, start:end, :]
            if current_input_length != input_sequence_length:
                b, sl, hdim = sub_inputs_embeds.shape
                sub_padding_inputs_embeds = torch.zeros((b, input_sequence_length, hdim), dtype=sub_inputs_embeds.dtype, device=sub_inputs_embeds.device)
                sub_padding_inputs_embeds[:, :sl, :] = sub_inputs_embeds
                sub_inputs_embeds = sub_padding_inputs_embeds
            else:
                sl = current_input_length
            # self.set_input_sequence_length(current_input_length)
            output = self._forward(
                sub_inputs_embeds,
                past_seq_length,
                torch.tensor([current_input_length], dtype=torch.int32).to(inputs_embeds.device),
                past_key_caches,
                past_value_caches,
            )
            hidden_states.append(output.logits[1][:, :sl, :])
            past_seq_length += current_input_length

        # self.set_input_sequence_length(input_sequence_length)
        return output.logits[0], torch.cat(hidden_states, dim=1)



