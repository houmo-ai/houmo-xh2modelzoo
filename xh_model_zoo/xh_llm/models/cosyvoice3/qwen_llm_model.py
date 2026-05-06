# -*- coding: utf-8 -*-
# Copyright 2025 The Qwen Team, Alibaba Group and The HuggingFace Inc. team. All rights reserved.
# Copyright 2025 HOUMO AI. All rights reserved.
#
# Modifications:
# - Portions of this file have been modified by HOUMO AI.
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
# File: qwen_llm_model.py
# Description:
#   CosyVoice3 TTS LLM model adapted for the xh2 model zoo (xh2modelzoo).

from copy import deepcopy
from typing import Optional, Union

import torch
import torch.nn as nn
from transformers.models.qwen2 import Qwen2ForCausalLM
from xhquant import nn as xhnn
from xhquant.api import QuantGraph, get_root_logger

from ..base_llm_model import LLMBaseModel
from ..builder import MODELS


@MODELS.register_module()
class XHQwen2LegacyModel(LLMBaseModel):
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

    # def _set_device(self, device):
    #     self.token_embedding = self.token_embedding.to(device)
    #     return super()._set_device(device)

    def _set_dtype(self, dtype):
        self.token_embedding = self.token_embedding.to(dtype)
        return super()._set_dtype(dtype)

    def init_wrap_model(self, hf_model=None):
        from ._model import register_wrap_modules as qwen2_register_wrap_modules

        qwen2_register_wrap_modules()

        super().init_wrap_model(hf_model)
        hf_model = self.wrap_model

        # self.token_embedding.weight 和 lm_head.weight 是相同对象
        self.token_embedding = deepcopy(hf_model.model.get_input_embeddings())
        self.generation_config = hf_model.generation_config
        self.config = hf_model.config
        self.num_hidden_layers = hf_model.model.config.num_hidden_layers
        # assert self.num_hidden_layers == 28
        # head_dim = hf_model.model.config.hidden_size // hf_model.model.config.num_attention_heads
        head_dim = hf_model.model.layers[0].self_attn.head_dim
        self.pad_token_id = hf_model.config.eos_token_id
        self.head_dim = head_dim
        if self.use_cache:
            num_decoder_layers = self.num_hidden_layers
            only_first_block = self.wrap_cfg.get("only_first_block", False)
            if only_first_block:
                num_decoder_layers = 1
            self.prepare_kv_cache(
                num_decoder_layers,
                [1, hf_model.model.config.num_key_value_heads, self.cache_length, head_dim],
            )

        hf_model = None

    def get_hf_model(self, device_map="cpu", **kwargs) -> Qwen2ForCausalLM:
        hf_model: Qwen2ForCausalLM = super().get_hf_model(device_map)
        assert id(hf_model.get_output_embeddings().weight) != id(hf_model.get_input_embeddings().weight)
        return hf_model

    def load_wraped_model_state_dict_prefix(self, native_model, checkpoint: str):
        logger = get_root_logger()

        archive_file = checkpoint
        logger.info(f"Load previously saved checkpoint from: {archive_file}")
        is_safetensors = archive_file.endswith(".safetensors")

        if is_safetensors:
            state_dict: Dict[str, Tensor] = load_safetensors_file(archive_file, device="cpu")
        else:
            state_dict: Dict[str, Tensor] = torch.load(archive_file, weights_only=True, map_location="cpu")

        model_state_dict = native_model.state_dict()
        new_state_dict = {}

        # 自动匹配前缀并移除
        def strip_prefix_if_present(key, prefixes):
            """
            尝试从 key 中移除 prefixes 中的一个，
            如果 native_model 中存在移除后的 key，则返回它。
            """
            for p in prefixes:
                if key.startswith(p):
                    new_k = key[len(p):]
                    if new_k in model_state_dict:
                        return new_k
            return key

        prefixes = [
            "llm.model."
        ]

        # 处理 state_dict
        for k, v in state_dict.items():
            new_k = strip_prefix_if_present(k, prefixes)

            if new_k in model_state_dict:
                new_state_dict[new_k] = v
            else:
                if k.split(".")[-1] != "quant_weight":
                    logger.warning(f"ignore unexpect state dict: {k}")

        # 注册 quant_weight buffer
        for k, v in state_dict.items():
            if k.split(".")[-1] == "quant_weight":
                paths = k.split(".")
                submodule_name = ".".join(paths[:-1])
                try:
                    submodule = native_model.get_submodule(submodule_name)

                    if v.min().item() >= -128 and v.max().item() <= 127:
                        v = v.to(torch.int8)
                    elif v.min().item() >= -32768 and v.max().item() <= 32767:
                        v = v.to(torch.int16)
                    else:
                        v = v.to(torch.float32)

                    submodule.register_buffer("quant_weight", v, persistent=False)
                    logger.debug(f"add quant_weight to {submodule_name}")
                except Exception:
                    logger.warning(f"ignore quant_weight state dict: {k}")

        # 加载参数
        native_model.load_state_dict(new_state_dict, strict=False)
        del state_dict
