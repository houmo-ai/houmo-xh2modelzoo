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

"""XH2a DFlash draft wrappers with explicit context-cache and decode graphs."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch import Tensor
from transformers.modeling_outputs import CausalLMOutputWithPast

from ..base_model import BaseModel
from ..builder import MODELS

from ._dflash_model import DFlashModelXH2a


def _pad_seq_tensor(tensor: Tensor, target_seq_len: int) -> Tensor:
    if tensor.shape[1] >= target_seq_len:
        return tensor[:, :target_seq_len, :]
    pad = torch.zeros(
        tensor.shape[0],
        target_seq_len - tensor.shape[1],
        tensor.shape[2],
        dtype=tensor.dtype,
        device=tensor.device,
    )
    return torch.cat([tensor, pad], dim=1)


def _build_dflash_export_adapter(
    core_model: DFlashModelXH2a,
    *,
    mode: str,
    num_hidden_layers: int,
) -> nn.Module:
    if mode == "context":
        arg_names = [
            "target_hidden",
            "past_seq_length",
            "current_input_length",
            *[f"past_key_cache_{idx}" for idx in range(num_hidden_layers)],
            *[f"past_value_cache_{idx}" for idx in range(num_hidden_layers)],
        ]
        core_call = "self.core.forward_context"
    elif mode == "decode":
        arg_names = [
            "noise_embedding",
            "past_seq_length",
            "current_input_length",
            "attn_mask",
            *[f"past_key_cache_{idx}" for idx in range(num_hidden_layers)],
            *[f"past_value_cache_{idx}" for idx in range(num_hidden_layers)],
        ]
        core_call = "self.core.forward_decode"
    else:
        raise ValueError(f"Unsupported DFlash mode: {mode}")

    forward_src = (
        f"def forward(self, {', '.join(arg_names)}):\n"
        f"    return {core_call}({', '.join(arg_names)})\n"
    )
    namespace: dict[str, Any] = {}
    exec(forward_src, {}, namespace)
    forward_impl = namespace["forward"]

    class _DFlashExportAdapter(nn.Module):
        def __init__(self, core: DFlashModelXH2a):
            super().__init__()
            self.core = core

    _DFlashExportAdapter.forward = forward_impl
    return _DFlashExportAdapter(core_model)


@MODELS.register_module()
class XHDFlashDraftModel(BaseModel):
    def __init__(
        self,
        hf_model=None,
        wrap_cfg=None,
        quant_config=None,
        frontend_type="TorchFX",
        allow_quant=True,
        export_cfg=None,
        dflash_model_dir=None,
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
        self.dflash_model_dir = dflash_model_dir
        self.target_model_dir = target_model_dir
        self.mode = str(self.wrap_cfg.get("mode", "decode"))
        self.dflash_config: Optional[dict] = None
        self.hidden_size = 0
        self.target_layer_ids: List[int] = []
        self.num_target_layers = 0
        self.num_hidden_layers = 0
        self.num_key_value_heads = 0
        self.head_dim = 0
        self.vocab_size = 0
        self.cache_length = int(self.wrap_cfg.get("max_sequence_length", 256))
        self.max_pe_length = int(self.wrap_cfg.get("max_pe_length", 262144))
        self.token_embedding: Optional[nn.Embedding] = None
        if dflash_model_dir is not None:
            self._load_dflash_config(dflash_model_dir)

    def _load_dflash_config(self, dflash_model_dir: str):
        with open(Path(dflash_model_dir) / "config.json", encoding="utf-8") as f:
            self.dflash_config = json.load(f)
        cfg = self.dflash_config
        self.hidden_size = cfg["hidden_size"]
        self.target_layer_ids = cfg["dflash_config"]["target_layer_ids"]
        self.num_target_layers = len(self.target_layer_ids)
        self.num_hidden_layers = cfg["num_hidden_layers"]
        self.num_key_value_heads = cfg["num_key_value_heads"]
        self.head_dim = cfg.get("head_dim", self.hidden_size // cfg["num_attention_heads"])
        self.vocab_size = cfg["vocab_size"]

    @property
    def input_sequence_length(self):
        return self.wrap_cfg.input_sequence_length

    @input_sequence_length.setter
    def input_sequence_length(self, value):
        self.wrap_cfg.input_sequence_length = value

    def init_wrap_model(self, hf_model=None):
        dtype = getattr(torch, self.wrap_cfg.get("dtype", "float16"))
        core_model = DFlashModelXH2a.from_pretrained(
            self.dflash_model_dir,
            self.target_model_dir,
            mode=self.mode,
            dtype=dtype,
            input_sequence_length=self.input_sequence_length,
            max_pe_length=self.max_pe_length,
            max_sequence_length=self.cache_length,
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

        self._wrap_model = _build_dflash_export_adapter(
            core_model,
            mode=self.mode,
            num_hidden_layers=self.num_hidden_layers,
        )
        if self.export_cfg is not None:
            input_names = []
            if self.mode == "context":
                input_names.extend(
                    ["target_hidden", "past_seq_length", "current_input_length"]
                )
                output_names = [
                    f"present_key_cache_{idx}" for idx in range(self.num_hidden_layers)
                ] + [
                    f"present_value_cache_{idx}" for idx in range(self.num_hidden_layers)
                ]
            else:
                input_names.extend(
                    [
                        "noise_embedding",
                        "past_seq_length",
                        "current_input_length",
                        "attn_mask",
                    ]
                )
                output_names = ["logits"]
            input_names.extend(
                [f"past_key_cache_{idx}" for idx in range(self.num_hidden_layers)]
            )
            input_names.extend(
                [f"past_value_cache_{idx}" for idx in range(self.num_hidden_layers)]
            )
            self.export_cfg.input_names = input_names
            self.export_cfg.output_names = output_names
        return self._wrap_model

    def prepare_inputs(self, data: Dict[str, Any] = None):
        dtype = torch.float16
        batch_size = int(self.wrap_cfg.get("batch_size", 1))
        seq_len = self.input_sequence_length
        cache_shape = [
            batch_size,
            self.num_key_value_heads,
            self.cache_length,
            self.head_dim,
        ]

        if self.mode == "context":
            if data is None:
                target_hidden = torch.randn(
                    batch_size,
                    seq_len,
                    self.num_target_layers * self.hidden_size,
                    dtype=dtype,
                )
                past_seq_length = torch.zeros(batch_size, dtype=torch.int32)
                current_input_length = torch.full(
                    (batch_size,), seq_len, dtype=torch.int32
                )
            else:
                target_hidden = _pad_seq_tensor(
                    data["target_hidden"].to(dtype=dtype), seq_len
                )
                raw_past_seq_length = data.get("past_seq_length", 0)
                if isinstance(raw_past_seq_length, int):
                    past_seq_length = torch.full(
                        (target_hidden.shape[0],),
                        raw_past_seq_length,
                        dtype=torch.int32,
                    )
                else:
                    past_seq_length = torch.tensor(
                        raw_past_seq_length, dtype=torch.int32
                    )
                raw_current_input_length = data.get(
                    "current_input_length", target_hidden.shape[1]
                )
                if isinstance(raw_current_input_length, int):
                    current_input_length = torch.full(
                        (target_hidden.shape[0],),
                        raw_current_input_length,
                        dtype=torch.int32,
                    )
                else:
                    current_input_length = torch.tensor(
                        raw_current_input_length, dtype=torch.int32
                    )
            tensors: list[Tensor] = [target_hidden, past_seq_length, current_input_length]
        else:
            if data is None:
                noise_embedding = torch.randn(
                    batch_size, seq_len, self.hidden_size, dtype=dtype
                )
                past_seq_length = torch.zeros(batch_size, dtype=torch.int32)
                current_input_length = torch.full(
                    (batch_size,), seq_len, dtype=torch.int32
                )
                attn_mask = torch.zeros(batch_size, self.cache_length, dtype=dtype)
            else:
                if "noise_embedding" in data:
                    noise_embedding = data["noise_embedding"].to(dtype=dtype)
                else:
                    noise_token_ids = data["noise_token_ids"]
                    if isinstance(noise_token_ids, list):
                        noise_token_ids = torch.tensor(noise_token_ids, dtype=torch.long)
                    if noise_token_ids.dim() == 1:
                        noise_token_ids = noise_token_ids.unsqueeze(0)
                    if self.token_embedding is None:
                        raise ValueError("token_embedding is not initialized")
                    self.token_embedding.to(dtype=dtype)
                    noise_embedding = self.token_embedding(noise_token_ids).to(dtype)
                noise_embedding = _pad_seq_tensor(noise_embedding, seq_len)
                raw_past_seq_length = data.get("past_seq_length", 0)
                if isinstance(raw_past_seq_length, int):
                    past_seq_length = torch.full(
                        (noise_embedding.shape[0],),
                        raw_past_seq_length,
                        dtype=torch.int32,
                    )
                else:
                    past_seq_length = torch.tensor(
                        raw_past_seq_length, dtype=torch.int32
                    )
                raw_current_input_length = data.get(
                    "current_input_length", noise_embedding.shape[1]
                )
                if isinstance(raw_current_input_length, int):
                    current_input_length = torch.full(
                        (noise_embedding.shape[0],),
                        raw_current_input_length,
                        dtype=torch.int32,
                    )
                else:
                    current_input_length = torch.tensor(
                        raw_current_input_length, dtype=torch.int32
                    )
                attn_mask = data.get("attn_mask")
                if attn_mask is None:
                    attn_mask = torch.full(
                        (noise_embedding.shape[0], self.cache_length),
                        -10000.0,
                        dtype=dtype,
                    )
                    for batch_idx in range(noise_embedding.shape[0]):
                        valid_ctx = int(past_seq_length[batch_idx].item())
                        attn_mask[batch_idx, :valid_ctx] = 0
                else:
                    attn_mask = attn_mask.to(dtype=dtype)
            tensors = [noise_embedding, past_seq_length, current_input_length, attn_mask]

        for idx in range(self.num_hidden_layers):
            cache = data.get(f"past_key_cache_{idx}") if data else None
            if cache is None:
                cache = torch.zeros(cache_shape, dtype=dtype)
            else:
                cache = cache.to(dtype=dtype)
            tensors.append(cache)
        for idx in range(self.num_hidden_layers):
            cache = data.get(f"past_value_cache_{idx}") if data else None
            if cache is None:
                cache = torch.zeros(cache_shape, dtype=dtype)
            else:
                cache = cache.to(dtype=dtype)
            tensors.append(cache)

        return tuple(tensors)

    def _forward(self, *inputs: Tensor):
        out = self(*inputs)
        if isinstance(out, Tensor):
            return CausalLMOutputWithPast(logits=out)
        return CausalLMOutputWithPast(logits=torch.zeros(1, 1, self.vocab_size))

    def _set_dtype(self, dtype):
        if self.token_embedding is not None:
            self.token_embedding = self.token_embedding.to(dtype)
        return super()._set_dtype(dtype)

    def _set_device(self, device):
        device = torch.device(device)
        if device != torch.device("meta") and self.token_embedding is not None:
            self.token_embedding = self.token_embedding.to(device)
        return super()._set_device(device)
