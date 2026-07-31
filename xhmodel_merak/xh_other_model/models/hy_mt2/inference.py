# Copyright 2025 HOUMO AI
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


def decode_next_token(tokenizer, logits: torch.Tensor):
    next_token_id = torch.argmax(logits, dim=-1)
    next_token_str = tokenizer.batch_decode(next_token_id, skip_special_tokens=True)
    return next_token_id, next_token_str


class HyMT2Inference(DeviceDtypeMixin):
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
        self.prefill_onnx_file = model_dir / meta_info["prefill_onnx"]
        self.decode_onnx_file = model_dir / meta_info["decode_onnx"]

        kv_cache_shape = meta_info["kv_cache"]["shape"]
        num_decoder_layers = meta_info["kv_cache"]["num_decoder_layers"]
        self.past_key_caches: List[CacheTensor] = []
        self.past_value_caches: List[CacheTensor] = []
        for _ in range(num_decoder_layers):
            self.past_key_caches.append(CacheTensor(torch.zeros(kv_cache_shape, dtype=torch.float16)))
            self.past_value_caches.append(CacheTensor(torch.zeros(kv_cache_shape, dtype=torch.float16)))

        hf_model_config_dir = str(model_dir / meta_info["hf_config"])
        self.tokenizer = AutoTokenizer.from_pretrained(hf_model_config_dir, trust_remote_code=True)

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
        self.pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        self._phase_prefill = True
        self.prefill_session: Optional[HMONNXInference] = None
        self.decode_session: Optional[HMONNXInference] = None

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
        assert input_ids.shape[0] == 1, "Batch size should be 1 in inference mode."
        seq_length = input_ids.shape[1]
        input_ids = input_ids.to(self.execution_device)
        assert seq_length <= input_sequence_length, (
            f"Input sequence length is too long. max input sequence length is {input_sequence_length} but got {seq_length}"
        )
        if input_sequence_length > seq_length:
            padding_input_ids = torch.full(
                (1, input_sequence_length - seq_length), self.pad_token_id, dtype=torch.long, device=self.execution_device
            )
            input_ids = torch.cat([input_ids, padding_input_ids], dim=-1)
        inputs_embeds = self.token_embedding.to(self.execution_device)(input_ids)
        past_seq_length = data["past_seq_length"]
        return (
            inputs_embeds.to(self.execution_device),
            torch.tensor([past_seq_length], dtype=torch.int32).to(self.execution_device),
            torch.tensor([seq_length], dtype=torch.int32).to(self.execution_device),
            self.past_key_caches,
            self.past_value_caches,
        )

    def forward(
        self,
        inputs_embeds: Tensor,
        past_seq_length: Tensor,
        current_input_length: Tensor,
        past_key_caches: List[CacheTensor],
        past_value_caches: List[CacheTensor],
    ) -> torch.FloatTensor:
        if self._phase_prefill:
            self.init_prefill()
            assert self.prefill_session is not None
            out = self.prefill_session(
                inputs_embeds.to(self._device),
                past_seq_length.to(self._device),
                current_input_length.to(self._device),
                *past_key_caches,
                *past_value_caches,
            )
            if isinstance(self.prefill_session, GoldenMixin):
                self.prefill_session.update_step()
            return out
        self.init_decode()
        assert self.decode_session is not None
        out = self.decode_session(
            inputs_embeds.to(self._device),
            past_seq_length.to(self._device),
            current_input_length.to(self._device),
            *past_key_caches,
            *past_value_caches,
        )
        if isinstance(self.decode_session, GoldenMixin):
            self.decode_session.update_step()
        return out

    @torch.no_grad()
    def _forward(self, messages):
        assert self.batch_size == 1
        texts = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer(texts, padding=False, return_tensors="pt")
        data_prefill = {"input_ids": model_inputs.input_ids, "past_seq_length": 0}
        self.set_phase_prefill(True)
        prefill_inputs = self.prepare_inputs(data_prefill, self.prefill_input_sequence_length)
        prefill_logits = self.forward(*prefill_inputs)
        prefill_next_token_id, prefill_next_token_text = decode_next_token(self.tokenizer, prefill_logits)
        return prefill_next_token_id, prefill_next_token_text

    __call__ = forward
