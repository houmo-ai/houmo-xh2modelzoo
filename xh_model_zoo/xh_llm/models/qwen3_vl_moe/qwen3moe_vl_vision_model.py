# Copyright 2025 HOUMO AI
#
# File: qwen3moe_vl_vision_model.py
# Description:
#   Qwen3moe Vl Vision Model model implementation.
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

from typing import List, Literal, Optional, Tuple

import torch
from torch import Tensor
from torchvision import transforms

# import transformers_modules
from transformers import AutoModelForCausalLM
from .modeling_qwen3moe_vl import Qwen3VLMoeForConditionalGeneration

from ..base_model import BaseModel
from ..builder import MODELS


@MODELS.register_module()
class XHQwen3Moe_VLVisionModel(BaseModel):
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
            hf_model=hf_model,
            wrap_cfg=wrap_cfg,
            quant_config=quant_config,
            allow_quant=allow_quant,
            export_cfg=export_cfg,
            frontend_type=frontend_type,
        )

    def get_hf_model(self, device_map="cpu", **kwargs) -> Qwen3VLMoeForConditionalGeneration:
        assert self.hf_model_dir is not None
        hf_model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            self.hf_model_dir,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="cpu",
        ).eval()
        return hf_model

    def init_wrap_model(self, hf_model=None):
        if hf_model is None:
            hf_model = self.get_hf_model()

        from ._vision_model_impl import register_wrap_cls as vision_register_wrap_cls

        vision_register_wrap_cls(hf_model)
        visual = hf_model.visual
        self.config = visual.config
        # self.token_embedding = hf_model.model.embed_tokens
        wraped_model = super().init_wrap_model(visual)
        wraped_model.to(torch.float16)
        return wraped_model
