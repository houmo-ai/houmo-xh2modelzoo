# Copyright 2025 HOUMO AI
#
# File: pipeline_cus.py
# Description:
#   Pipeline Cus implementation.
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

from typing import Any, Callable, Dict, List, Optional, Union
import torch
from .gr00t.model.gr00t_n1d6.gr00t_n1d6 import Gr00tN1d6

class cus_GROOT(Gr00tN1d6):
    def __init__(self, *args, **kwargs):
        pass
    
    def __setup__(self, backbone, transformer=None, vae=None):
        """"""
        """
        初始化模型
        """
        self.backbone.model = backbone
        return self

    @classmethod
    def to_hf_compatible(
        cls,
        hf_model,
        backbone = None,
        vae = None,
        transformers = None,
        meta_info = None,
    ):
        """
        将改写后的模型转换为兼容 Hugging Face 的模型
        """
        if backbone is not None:
            hf_model.__class__ = cls
            hf_model.__setup__(backbone, transformers,  vae)
            # hf_model.embed_tokens = hf_model.model.embed_tokens
            # del hf_model.text_encoder
            # del hf_model.lm_head
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return hf_model

    def get_action(self, inputs: dict):
        """
        Generate actions using the complete model.
        """
        # Prepare inputs for backbone and action head
        backbone_inputs, action_inputs = self.prepare_input(inputs)

        # Forward through backbone
        backbone_outputs = self.backbone(backbone_inputs)
        action_outputs = self.action_head.get_action(backbone_outputs, action_inputs)
        return action_outputs