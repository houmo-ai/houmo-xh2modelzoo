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

TEM_NUM = 8

class cus_GROOT(Gr00tN1d6):
    def __init__(self, *args, **kwargs):
        pass
    
    def __setup__(self, backbone, new_network_pre, new_network):
        """"""
        """
        初始化模型
        """
        self.backbone.model = backbone
        self.new_network_pre = new_network_pre
        self.new_network = new_network
        return self
    
    def head_get_action(self, backbone_output, action_input):
        """
        Generate actions using the complete model.
        """
        image_mask = backbone_output.image_mask
        backbone_attention_mask = backbone_output.backbone_attention_mask
        backbone_features = backbone_output.backbone_features
        state = action_input['state']

        # breakpoint()
        batch_size = image_mask.shape[0]

        if image_mask.shape[1] < 256:
            image_mask = torch.cat([image_mask, torch.zeros((batch_size, 256 - image_mask.shape[1]), dtype=torch.bool, device=image_mask.device)], dim=1)
            backbone_attention_mask = torch.cat([backbone_attention_mask, torch.zeros((batch_size, 256 - backbone_attention_mask.shape[1]), dtype=torch.bool, device=backbone_attention_mask.device)], dim=1)
            backbone_features = torch.cat([backbone_features, torch.zeros((batch_size, 256 - backbone_features.shape[1], backbone_features.shape[-1]), dtype=backbone_features.dtype, device=backbone_features.device)], dim=1)

        image_attention_mask= image_mask & backbone_attention_mask
        non_image_attention_mask= (~image_mask) & backbone_attention_mask

        image_attn_reshape = image_attention_mask.repeat(32,1).view(batch_size,32,1,256)
        non_image_attn_reshape = non_image_attention_mask.repeat(32,1).view(batch_size,32,1,256)

        image_attention_mask = torch.zeros_like(image_attn_reshape, dtype=torch.float16)
        image_attention_mask[image_attn_reshape == False] = -65504

        non_image_attention_mask = torch.zeros_like(non_image_attn_reshape, dtype=torch.float16)
        non_image_attention_mask[non_image_attn_reshape == False] = -65504

        # backbone_features = []
        # state_features = []
        actions_list = []
        for i in range(batch_size):
            backbone_feature, state_feature = self.new_network_pre(
                backbone_features[i:i+1],
                state[i:i+1],
            )
            # backbone_features.append(backbone_feature)
            # state_features.append(state_feature)
        
        # backbone_features = torch.cat(backbone_features, dim=0)
        # state_features = torch.cat(state_features, dim=0)

            actions = torch.randn( [1, 50, 128], device=backbone_features.device, dtype=torch.float16)

            for t in range(self.temb_num):
                timesteps_tensor = torch.tensor([t], device=self.device, dtype=torch.int32)
                actions = self.new_network(
                    backbone_feature,
                    state_feature,
                    timesteps_tensor,
                    actions,
                    image_attention_mask[i:i+1],
                    non_image_attention_mask[i:i+1]
                )
            
            actions_list.append(actions)

        actions = torch.cat(actions_list, dim=0)
        output = {}
        output["action_pred"] = actions
        return output
    @classmethod
    def to_hf_compatible(
        cls,
        hf_model,
        backbone = None,
        new_network_pre = None,
        new_network = None,
        meta_info = None,
        temb_num = 4,
    ):
        """
        将改写后的模型转换为兼容 Hugging Face 的模型
        """
        hf_model.__class__ = cls
        hf_model.__setup__(backbone, new_network_pre, new_network)
        # hf_model.embed_tokens = hf_model.model.embed_tokens
        # del hf_model.text_encoder
        # del hf_model.lm_head
        hf_model.temb_num = temb_num
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
        # action_outputs = self.action_head.get_action(backbone_outputs, action_inputs)

        hm_action = self.head_get_action(backbone_outputs, action_inputs)
        return hm_action
