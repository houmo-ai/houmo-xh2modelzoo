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

"""XH2a MTP draft model wrapper with explicit cache inputs/outputs."""

import json
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor
from transformers.modeling_outputs import CausalLMOutputWithPast

from ..base_model import BaseModel
from ..builder import MODELS

from ._mtp_model import MTPModelXH2a


def _pad_hidden_tensor(hidden: Tensor, target_seq_len: int) -> Tensor:
    if hidden.shape[1] >= target_seq_len:
        return hidden[:, :target_seq_len, :]
    pad = torch.zeros(
        hidden.shape[0],
        target_seq_len - hidden.shape[1],
        hidden.shape[2],
        dtype=hidden.dtype,
        device=hidden.device,
    )
    return torch.cat([hidden, pad], dim=1)


@MODELS.register_module()
class XHMTPDraftModel(BaseModel):
    def __init__(
        self,
        hf_model=None,
        wrap_cfg=None,
        quant_config=None,
        frontend_type="TorchFX",
        allow_quant=True,
        export_cfg=None,
        target_model_dir=None,
    ):
        super().__init__(
            hf_model=hf_model,
            wrap_cfg=wrap_cfg,
            quant_config=quant_config,
            frontend_type=frontend_type,
            allow_quant=allow_quant,
            export_cfg=export_cfg,
        )
        self.target_model_dir = target_model_dir
        self.hidden_size = 0
        self.vocab_size = 0
        self.num_key_value_heads = 0
        self.head_dim = 0
        self.cache_length = int(self.wrap_cfg.get("max_sequence_length", 256))
        self.max_pe_length = int(self.wrap_cfg.get("max_pe_length", 262144))
        self.token_embedding: Optional[nn.Embedding] = None
        if target_model_dir is not None:
            self._load_config(target_model_dir)

    def _load_config(self, target_model_dir: str):
        cfg_path = Path(target_model_dir) / "config.json"
        with open(cfg_path, encoding="utf-8") as f:
            cfg = json.load(f)
        text_cfg = cfg.get("text_config", cfg)
        self.hidden_size = text_cfg["hidden_size"]
        self.vocab_size = text_cfg["vocab_size"]
        self.num_key_value_heads = text_cfg["num_key_value_heads"]
        self.head_dim = text_cfg["head_dim"]

    @property
    def input_sequence_length(self):
        return self.wrap_cfg.input_sequence_length

    @input_sequence_length.setter
    def input_sequence_length(self, value):
        self.wrap_cfg.input_sequence_length = value

    def init_wrap_model(self, hf_model=None):
        dtype = getattr(torch, self.wrap_cfg.get("dtype", "float16"))
        model = MTPModelXH2a.from_pretrained(
            self.target_model_dir,
            dtype=dtype,
            input_sequence_length=self.input_sequence_length,
            max_pe_length=self.max_pe_length,
            use_cache=True,
        )

        from safetensors import safe_open

        for sf_path in sorted(Path(self.target_model_dir).glob("*.safetensors")):
            with safe_open(str(sf_path), framework="pt") as f:
                for key in f.keys():
                    if "embed_tokens.weight" in key:
                        embed_weight = f.get_tensor(key).to(dtype)
                        self.token_embedding = nn.Embedding(
                            embed_weight.shape[0], embed_weight.shape[1]
                        )
                        self.token_embedding.weight.data.copy_(embed_weight)
                        break
                if self.token_embedding is not None:
                    break

        self._wrap_model = model
        if self.export_cfg is not None:
            self.export_cfg.input_names = [
                "next_token_embedding",
                "pre_norm_hidden",
                "past_seq_length",
                "current_input_length",
                "past_key_cache",
                "past_value_cache",
            ]
            self.export_cfg.output_names = [
                "logits",
                "pre_norm_out",
                "present_key_cache",
                "present_value_cache",
            ]
        return self._wrap_model

    def prepare_inputs(self, data: Dict[str, Any] = None):
        dtype = torch.float16
        batch_size = int(self.wrap_cfg.get("batch_size", 1))
        seq_len = self.input_sequence_length

        if data is None:
            next_token_embedding = torch.randn(
                batch_size, seq_len, self.hidden_size, dtype=dtype
            )
            pre_norm_hidden = torch.randn(
                batch_size, seq_len, self.hidden_size, dtype=dtype
            )
            past_seq_length = torch.zeros(batch_size, dtype=torch.int32)
            current_input_length = torch.full(
                (batch_size,), seq_len, dtype=torch.int32
            )
        else:
            if "next_token_embedding" in data:
                next_token_embedding = data["next_token_embedding"].to(dtype=dtype)
            else:
                next_token_ids = data["next_token_ids"]
                if isinstance(next_token_ids, list):
                    next_token_ids = torch.tensor(next_token_ids, dtype=torch.long)
                if next_token_ids.dim() == 1:
                    next_token_ids = next_token_ids.unsqueeze(0)
                if self.token_embedding is None:
                    raise ValueError("token_embedding is not initialized")
                self.token_embedding.to(dtype=dtype)
                next_token_embedding = self.token_embedding(next_token_ids).to(dtype)
            pre_norm_hidden = data["pre_norm_hidden"].to(dtype=dtype)
            next_token_embedding = _pad_hidden_tensor(next_token_embedding, seq_len)
            pre_norm_hidden = _pad_hidden_tensor(pre_norm_hidden, seq_len)

            raw_past_seq_length = data.get("past_seq_length", 0)
            if isinstance(raw_past_seq_length, int):
                past_seq_length = torch.full(
                    (next_token_embedding.shape[0],),
                    raw_past_seq_length,
                    dtype=torch.int32,
                )
            else:
                past_seq_length = torch.tensor(raw_past_seq_length, dtype=torch.int32)

            raw_current_input_length = data.get(
                "current_input_length", next_token_embedding.shape[1]
            )
            if isinstance(raw_current_input_length, int):
                current_input_length = torch.full(
                    (next_token_embedding.shape[0],),
                    raw_current_input_length,
                    dtype=torch.int32,
                )
            else:
                current_input_length = torch.tensor(
                    raw_current_input_length, dtype=torch.int32
                )

        cache_shape = [
            next_token_embedding.shape[0],
            self.num_key_value_heads,
            self.cache_length,
            self.head_dim,
        ]
        past_key_cache = data.get("past_key_cache") if data else None
        if past_key_cache is None:
            past_key_cache = torch.zeros(cache_shape, dtype=dtype)
        else:
            past_key_cache = past_key_cache.to(dtype=dtype)
        past_value_cache = data.get("past_value_cache") if data else None
        if past_value_cache is None:
            past_value_cache = torch.zeros(cache_shape, dtype=dtype)
        else:
            past_value_cache = past_value_cache.to(dtype=dtype)

        return (
            next_token_embedding,
            pre_norm_hidden,
            past_seq_length,
            current_input_length,
            past_key_cache,
            past_value_cache,
        )

    def _forward(
        self,
        next_token_embedding: Tensor,
        pre_norm_hidden: Tensor,
        past_seq_length: Tensor,
        current_input_length: Tensor,
        past_key_cache: Tensor,
        past_value_cache: Tensor,
    ):
        out = self(
            next_token_embedding,
            pre_norm_hidden,
            past_seq_length,
            current_input_length,
            past_key_cache,
            past_value_cache,
        )
        return CausalLMOutputWithPast(logits=out[0])

    def _set_dtype(self, dtype):
        if self.token_embedding is not None:
            self.token_embedding = self.token_embedding.to(dtype)
        return super()._set_dtype(dtype)

    def _set_device(self, device):
        device = torch.device(device)
        if device != torch.device("meta") and self.token_embedding is not None:
            self.token_embedding = self.token_embedding.to(device)
        return super()._set_device(device)
