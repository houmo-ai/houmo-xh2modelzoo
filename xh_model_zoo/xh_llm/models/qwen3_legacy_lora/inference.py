# Copyright 2025 HOUMO AI
#
# File: inference.py
# Description:
#   Qwen3 legacy LoRA ONNX inference wrapper with KV-cache support for xh2modelzoo.
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

import json
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from transformers import AutoTokenizer
from xhquant.api import CacheTensor, GoldenMixin, HMONNXInference

from ....utils import DeviceDtypeMixin
from ....xh_llm.utils import decode_next_token


class Qwen3LegacyLoRAInference(DeviceDtypeMixin):
    def __init__(
        self,
        model_config_file: str,
        fast_mode=True,
        device: str = "cuda",
        execution_device: str = "cuda",
    ):
        super().__init__()

        self.fast_mode = fast_mode
        self._device = torch.device(device)
        self.set_exec_device(torch.device(execution_device))

        model_dir = Path(model_config_file).parent
        meta_info = json.load(open(model_config_file, "r"))
        self.meta_info = meta_info

        # hmonnx files
        self.prefill_onnx_file = model_dir / meta_info["prefill_onnx"]
        self.decode_onnx_file = model_dir / meta_info["decode_onnx"]

        # kv cache
        kv_cache_shape = meta_info["kv_cache"]["shape"]
        num_decoder_layers = meta_info["kv_cache"]["num_decoder_layers"]

        # create kv cache
        past_key_caches: List[CacheTensor] = []
        past_value_caches: List[CacheTensor] = []
        for i in range(num_decoder_layers):
            past_k_cache = CacheTensor(torch.zeros(kv_cache_shape, dtype=torch.float16))
            past_v_cache = CacheTensor(torch.zeros(kv_cache_shape, dtype=torch.float16))
            past_key_caches.append(past_k_cache)
            past_value_caches.append(past_v_cache)

        self.past_key_caches = past_key_caches
        self.past_value_caches = past_value_caches

        # tokenizer
        hf_model_config_dir = str(model_dir / meta_info["hf_config"])
        self.tokenizer = AutoTokenizer.from_pretrained(hf_model_config_dir)

        # token embedding
        token_embedding_state_dict = torch.load(
            model_dir / meta_info["token_embedding_file"],
            map_location="cpu",
            weights_only=True,
        )
        self.token_embedding = nn.Embedding(
            token_embedding_state_dict["weight"].shape[0],
            token_embedding_state_dict["weight"].shape[1],
        ).to(torch.float16)
        self.token_embedding.load_state_dict(token_embedding_state_dict)

        self.batch_size = 1

        self.prefill_input_sequence_length = meta_info["wrap_cfg"]["input_sequence_length"]
        self.input_sequence_length = self.prefill_input_sequence_length
        self.pad_token_id = self.tokenizer.eos_token_id
        self._phase_prefill = True

        self.prefill_session: Optional[HMONNXInference] = None
        self.decode_session: Optional[HMONNXInference] = None

        self._enable_lora = True

    @property
    def enable_lora(self) -> bool:
        return self._enable_lora

    @enable_lora.setter
    def enable_lora(self, value: bool):
        self._enable_lora = value

    def set_phase_prefill(self, prefill: bool):
        self._phase_prefill = prefill
        if prefill:
            if self.prefill_session is None:
                self.init_prefill()
                self.input_sequence_length = self.prefill_input_sequence_length
        else:
            self.prefill_session = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if self.decode_session is None:
                self.init_decode()
            self.input_sequence_length = 1

    def init_prefill(self):
        if self.prefill_session is not None:
            return
        self.prefill_session = HMONNXInference(self.prefill_onnx_file)
        if self.fast_mode:
            self.prefill_session.to_fast_mode()
        self.prefill_session.exec_device = self.execution_device
        self.prefill_session.to(self._device)

    def init_decode(self):
        if self.decode_session is not None:
            return
        self.decode_session = HMONNXInference(self.decode_onnx_file)
        if self.fast_mode:
            self.decode_session.to_fast_mode()
        self.decode_session.exec_device = self.execution_device
        self.decode_session.to(self._device)

    def get_input_sequence_length(self):
        return self.input_sequence_length

    def set_input_sequence_length(self, input_sequence_length):
        self.input_sequence_length = input_sequence_length

    def prepare_inputs(
        self, data, input_sequence_length
    ) -> Tuple[Tensor, Tensor, Tensor, List[CacheTensor], List[CacheTensor]]:
        input_ids = data["input_ids"]
        assert self.token_embedding is not None, "Token embedding is not available."
        assert input_ids.shape[0] == 1, "Batch size should be 1 in inference mode."
        seq_length = input_ids.shape[1]
        input_ids = input_ids.to(self.execution_device)
        assert seq_length <= input_sequence_length, (
            f"Input sequence length is too long. max input sequence length is {input_sequence_length} but got {seq_length}"
        )
        if input_sequence_length > seq_length:
            padding_input_ids = torch.zeros((1, input_sequence_length - seq_length), dtype=torch.long).to(
                self.execution_device
            )
            padding_input_ids.fill_(self.pad_token_id)
            input_ids = torch.cat([input_ids, padding_input_ids], dim=-1)
        inputs_embeds = self.token_embedding.to(self.execution_device)(input_ids)

        past_seq_length = data["past_seq_length"]
        assert past_seq_length >= 0, "past_seq_length should be non-negative."
        past_key_caches = self.past_value_caches
        past_value_caches = self.past_key_caches

        return (
            inputs_embeds.to(self.execution_device),
            torch.tensor([past_seq_length], dtype=torch.int32).to(self.execution_device),
            torch.tensor([seq_length], dtype=torch.int32).to(self.execution_device),
            past_key_caches,
            past_value_caches,
        )

    def forward(
        self,
        inputs_embeds: Tensor,
        past_seq_length: Tensor,
        current_input_length: Tensor,
        past_key_caches: List[CacheTensor],
        past_value_caches: List[CacheTensor],
    ) -> torch.FloatTensor:
        if self.enable_lora:
            lora_mask = torch.tensor([1.0], dtype=torch.float16).to(self._device)
        else:
            lora_mask = torch.tensor([0.0], dtype=torch.float16).to(self._device)
        if self._phase_prefill:
            self.init_prefill()
            assert self.prefill_session is not None, "Prefill session is not initialized."

            out = self.prefill_session(
                inputs_embeds.to(self._device),
                past_seq_length.to(self._device),
                current_input_length.to(self._device),
                *past_key_caches,
                *past_value_caches,
                lora_mask,
            )
            if isinstance(self.prefill_session, GoldenMixin):
                self.prefill_session.update_step()
            return out
        else:
            self.init_decode()
            assert self.decode_session is not None, "Decode session is not initialized."
            out = self.decode_session(
                inputs_embeds.to(self._device),
                past_seq_length.to(self._device),
                current_input_length.to(self._device),
                *past_key_caches,
                *past_value_caches,
                lora_mask,
            )
            if isinstance(self.decode_session, GoldenMixin):
                self.decode_session.update_step()
            return out

    @torch.no_grad()
    def _forward(self, messages, enable_thinking=False):
        assert self.batch_size == 1, "Batch size should be 1 in inference mode."
        assert len(messages) == self.batch_size

        texts = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            enable_thinking=enable_thinking,
            add_generation_prompt=True,
        )

        batch_input_ids = []
        for text in texts:
            model_inputs = self.tokenizer([text], padding=False, return_tensors="pt")
            batch_input_ids.append(model_inputs.input_ids.cpu().numpy().tolist()[0])

        data_prefill = {
            "input_ids": torch.tensor(batch_input_ids),
            "past_seq_length": 0,
        }

        device = torch.device(self.device)
        execution_device = torch.device(self.execution_device)

        # prefill
        prefill_inputs = self.prepare_inputs(data_prefill, self.prefill_input_sequence_length)
        (
            inputs_embeds,
            past_seq_length,
            seq_length,
            past_key_caches,
            past_value_caches,
        ) = prefill_inputs

        prefill_session = HMONNXInference(str(self.prefill_onnx_file))
        prefill_session.to(device)
        prefill_session.exec_device = execution_device

        prefill_logits = prefill_session(
            inputs_embeds,
            past_seq_length,
            seq_length,
            *past_key_caches,
            *past_value_caches,
        )
        prefill_next_token_id, prefill_next_token_text = decode_next_token(self.tokenizer, prefill_logits)

        del prefill_session
        torch.cuda.empty_cache()

        # decode
        past_seq_len = [len(input_ids) for input_ids in batch_input_ids]
        batch_input_ids = prefill_next_token_id.cpu().tolist()

        data_decode = {
            "input_ids": torch.tensor(batch_input_ids),
            "past_seq_length": past_seq_len[0],
        }
        decode_inputs = self.prepare_inputs(data_decode, 1)
        (
            inputs_embeds,
            past_seq_length,
            seq_length,
            past_key_caches,
            past_value_caches,
        ) = decode_inputs

        decode_session = HMONNXInference(str(self.decode_onnx_file))
        decode_session.to(device)
        decode_session.exec_device = execution_device

        decode_logits = decode_session(
            inputs_embeds,
            past_seq_length,
            seq_length,
            *past_key_caches,
            *past_value_caches,
        )
        decode_next_token_id, decode_next_token_text = decode_next_token(self.tokenizer, decode_logits)
        # logger.info(f"Decode next token: {decode_next_token_id} {decode_next_token_text}")
        generate_ids = torch.cat([prefill_next_token_id, decode_next_token_id], dim=1)

        generate_text = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True)
        return (generate_ids, generate_text)
