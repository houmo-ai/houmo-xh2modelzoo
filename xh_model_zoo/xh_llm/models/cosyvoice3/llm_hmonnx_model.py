# Copyright 2025 HOUMO AI
#
# File: llm_hmonnx_model.py
# Description:
#   LLM HMONNX model wrapper for HOUMO AI xh2modelzoo.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
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
from typing import List, Union

import torch
import torch.nn as nn
from torch import Tensor
from xhquant.api import ConfigDict

from ..builder import MODELS
from ..llm_onnx_model import LLMONNXModel

from transformers.modeling_outputs import CausalLMOutputWithPast

from xhquant.api import HMONNXInference

@MODELS.register_module()
class XHQwen2HMONNXModel(LLMONNXModel):
    def __init__(self, model_dir: str):
        meta_info_file = Path(model_dir) / "meta_info.json"
        assert meta_info_file.exists(), f"meta_info.json not found in {model_dir}"
        meta_info = json.load(open(meta_info_file, "r"))
        meta_info = ConfigDict(meta_info)
        self.wrap_cfg = meta_info.wrap_cfg
        prefill = ConfigDict(
            dict(
                onnx=str(Path(model_dir) / meta_info.prefill_onnx_file),
                input_sequence_length=meta_info.wrap_cfg.input_sequence_length,
            )
        )

        decode = ConfigDict(
            dict(
                onnx=str(Path(model_dir) / meta_info.decode_onnx_file),
            )
        )
        
        kv_cache = ConfigDict(
            dict(
                num_hidden_layers=meta_info.num_hidden_layers,
                shape=meta_info.kv_cache_shape,
            )
        )
        super().__init__(prefill, decode, kv_cache)
        self.input_sequence_length = self.prefill_input_sequence_length

        # token embedding
        token_embedding_state_dict = torch.load(
            Path(model_dir) / meta_info["token_embedding_file"], map_location="cpu", weights_only=True
        )
        token_embedding = nn.Embedding(
            token_embedding_state_dict["weight"].shape[0],
            token_embedding_state_dict["weight"].shape[1],
        )
        
        token_embedding.load_state_dict(token_embedding_state_dict)
        
        speech_embedding = nn.Embedding(6761, 896)
        speech_embedding_param = torch.load(meta_info["speech_embedding_file"])
        speech_embedding_state_dict = {"weight": speech_embedding_param}
        speech_embedding.load_state_dict(speech_embedding_state_dict)

        self.llm_decoder_session = HMONNXInference(meta_info["llm_decoder_file"])

        self.set_input_embeddings(token_embedding)
        self.set_input_embeddings_speech(speech_embedding)
        self.pad_token_id = 151645
        self._phase_prefill = None

    def get_input_sequence_length(self):
        return self.input_sequence_length

    def set_input_sequence_length(self, input_sequence_length):
        self.input_sequence_length = input_sequence_length
    
    def prepare_inputs(self, data: Union[dict, tuple, list]):
        device = self._exec_device
        self.token_embedding.to(device)

        past_seq_length = data["past_seq_length"]
        assert isinstance(past_seq_length, list), f"past_seq_length must be list, but got {type(past_seq_length)}"

        if "input_ids" in data:
            input_ids = data["input_ids"]
            input_ids = torch.tensor(input_ids, dtype=torch.long)
            inputs_embeds = self.token_embedding(input_ids)
        else:
            inputs_embeds = data["inputs_embeds"]
        seq_length = inputs_embeds.shape[1]

        assert seq_length <= self.input_sequence_length, f"input_sequence_length must be greater than {seq_length}"

        inputs_embeds = inputs_embeds.to(device)
        position_ids = torch.arange(past_seq_length[0], past_seq_length[0] + seq_length, dtype=torch.long)
        position_ids = position_ids.unsqueeze(0).to(device)
        current_input_length = [seq_length]

        current_input_length = torch.tensor(current_input_length, dtype=torch.int32).to(device)
        past_seq_length = torch.tensor(past_seq_length, dtype=torch.int32).to(device)
        assert torch.all(past_seq_length >= 0)

        if seq_length < self.input_sequence_length:
            padding_input_ids = torch.zeros((self.input_sequence_length - seq_length), dtype=torch.long, device=device)
            padding_input_ids = padding_input_ids.unsqueeze(0)
            position_ids = torch.cat([position_ids, padding_input_ids], dim=1)

            padding_input_ids = torch.zeros((self.input_sequence_length - seq_length), dtype=torch.long, device=device)
            padding_input_ids.fill_(self.pad_token_id)
            padding_input_ids = padding_input_ids.unsqueeze(0)

            inputs_embeds = torch.cat([inputs_embeds, self.token_embedding(padding_input_ids)], dim=1)

        past_key_caches = []
        past_value_caches = []
        for i in range(self.num_hidden_layers):
            past_key_caches.append(getattr(self, f"past_k_cache_{i}"))
            past_value_caches.append(getattr(self, f"past_v_cache_{i}"))

        return (
            inputs_embeds.to(device),
            past_seq_length.to(device),
            current_input_length,
            position_ids.to(device),
            past_key_caches,
            past_value_caches,
        )

    def forward(
        self,
        inputs_embeds: Tensor,
        past_seq_length: Tensor,
        current_input_length: Tensor,
        position_ids: Tensor,
        past_key_caches: List,
        past_value_caches: List,
    ):
        if self._phase_prefill:
            # if self.prefill_session._step is None:
            #     self.prefill_session._step = 0
            # self.prefill_session.to_fast_mode()
            out = self.prefill_session(
                inputs_embeds.to(torch.float16).to(self.device),
                past_seq_length.to(self.device),
                current_input_length.to(self.device),
                *past_key_caches,
                *past_value_caches,
            )
            # self.prefill_session.update_step()
            return out
        else:
            # if self.decode_session._step is None:
            #     self.decode_session._step = 0
            # self.decode_session.to_fast_mode()
            out = self.decode_session(
                inputs_embeds.to(self.device),
                past_seq_length.to(self.device),
                current_input_length.to(self.device),
                *past_key_caches,
                *past_value_caches,
            )
            # self.decode_session.update_step()
            return out
    
    def set_phase_prefill(self, prefill: bool):
        self._phase_prefill = prefill
        if prefill:
            if self.prefill_session is None:
                self.init_prefill()
        else:
            del self.prefill_session
            self.prefill_session = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if self.decode_session is None:
                self.init_decode()

    @torch.no_grad()
    def prefill(self, data: Union[dict, tuple, list]):
        raise NotImplementedError("Prefill is not supported for XHQwen2HMONNXModel.")

    @torch.no_grad()
    def decode(self, data: Union[dict, tuple, list]):
        raise NotImplementedError("Decode is not supported for XHQwen2HMONNXModel.")

    def nucleus_sampling(self, weighted_scores, top_p=0.8, top_k=25):
        prob, indices = [], []
        cum_prob = 0.0
        sorted_value, sorted_idx = weighted_scores.softmax(dim=0).sort(descending=True, stable=True)
        for i in range(len(sorted_idx)):
            # sampling both top-p and numbers.
            if cum_prob < top_p and len(prob) < top_k:
                cum_prob += sorted_value[i]
                prob.append(sorted_value[i])
                indices.append(sorted_idx[i])
            else:
                break
        prob = torch.tensor(prob).to(weighted_scores)
        indices = torch.tensor(indices, dtype=torch.long).to(weighted_scores.device)
        top_ids = indices[prob.multinomial(1, replacement=True)]
        return top_ids

    def random_sampling(self, weighted_scores, decoded_tokens, sampling):
        top_ids = weighted_scores.softmax(dim=0).multinomial(1, replacement=True)
        return top_ids
    
    def ras_sampling(self, weighted_scores, decoded_tokens, sampling, top_p=0.8, top_k=25, win_size=10, tau_r=0.1):
        top_ids = self.nucleus_sampling(weighted_scores, top_p=top_p, top_k=top_k)
        rep_num = (torch.tensor(decoded_tokens[-win_size:]).to(weighted_scores.device) == top_ids).sum().item()
        if rep_num >= win_size * tau_r:
            top_ids = self.random_sampling(weighted_scores, decoded_tokens, sampling)
        return top_ids

    def sampling_ids(
            self,
            weighted_scores: torch.Tensor,
            decoded_tokens: List,
            sampling: int,
            ignore_eos: bool = True,
        ):
        num_trials, max_trials = 0, 100
        speech_token_size = 6561
        while True:
            top_ids = self.ras_sampling(weighted_scores, decoded_tokens, sampling)
            if (not ignore_eos) or (speech_token_size not in top_ids):
                break
            num_trials += 1
            if num_trials > max_trials:
                raise RuntimeError('sampling reaches max_trials {} and still get eos when ignore_eos is True, check your input!'.format(max_trials))
        return top_ids