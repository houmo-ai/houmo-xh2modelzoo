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
import os
import sys
import importlib
user_home = os.path.expanduser("~")
target_path = os.path.join(
    user_home,
    ".cache/huggingface/modules"
)
sys.path.append(target_path) # modeling.py
eagle3_full_module_name = "transformers_modules.Eagle-Block2A-2B-v2.modeling_eagle3_vl"
modeling_eagle3_vl = importlib.import_module(eagle3_full_module_name)

class cus_eagle3_inference(modeling_eagle3_vl.Eagle3_VLForConditionalGeneration):
    def __init__(self, *args, **kwargs):
        pass
    
    def __setup__(self, text_encoder, vision=None):
        """"""
        """
        初始化模型
        """
        return self

    @classmethod
    def to_hf_compatible(
        cls,
        hf_model,
        text_encoder = None,
        vision = None,
        meta_info = None,
    ):
        """
        将改写后的模型转换为兼容 Hugging Face 的模型
        """
        if text_encoder is not None:
            hf_model.__class__ = cls
            hf_model.__setup__(text_encoder, vision)
            # hf_model.embed_tokens = hf_model.model.embed_tokens
            # del hf_model.text_encoder
            # del hf_model.lm_head
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return hf_model

    def extract_feature(self, pixel_values, image_flags=None):
        vision_model_output = self.vision_model(
            pixel_values=pixel_values,
            output_hidden_states=False,
            return_dict=True)
        if hasattr(vision_model_output, 'last_hidden_state'):
            vit_embeds = vision_model_output.last_hidden_state
        if hasattr(vision_model_output, 'spatial_shapes'):
            spatial_shapes = vision_model_output.spatial_shapes


        vit_embeds, spatial_shapes = self.pixel_shuffle_back(vit_embeds, spatial_shapes)

        vit_embeds = self.mlp1(vit_embeds)
        
        B, N, C = vit_embeds.shape
        vit_embeds = vit_embeds.reshape(B * N, C)

        return vit_embeds

    def forward(
        self,
        pixel_values: List[torch.FloatTensor],
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        image_flags: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        input_embeds = self.language_model.get_input_embeddings()(input_ids)

        num_images = len(pixel_values)
        
        if image_flags is not None:
            image_flags = image_flags.view(-1)

        vit_embeds = self.extract_feature(pixel_values, image_flags) # [1, 3, 252, 252]


        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)

        input_ids = input_ids.reshape(B * N)
        selected = (input_ids == self.image_token_index)
        try:
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds
        except Exception as e:
            print(f'warning: {e}, input_embeds[selected].shape={input_embeds[selected].shape}, '
                  f'vit_embeds.shape={vit_embeds.shape}')
            n_token = selected.sum()
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds[:n_token]

        input_embeds = input_embeds.reshape(B, N, C)

        outputs = self.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        # logits = outputs.logits



        return outputs