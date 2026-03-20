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
from xhquant.api import CacheTensor
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
        self.pad_token_id = 151643
        pass
    
    def __setup__(self, text_encoder, vision=None):
        """"""
        """
        初始化模型
        """
        self.hm_vison = vision
        self.hm_text_encoder = text_encoder
        return self

    @classmethod
    def to_hf_compatible(
        cls,
        hf_model,
        text_encoder = None,
        vision = None,
        meta_info = None,
        fix_num_images = None,
    ):
        """
        将改写后的模型转换为兼容 Hugging Face 的模型
        """
        # if text_encoder is not None:
        hf_model.__class__ = cls
        hf_model.__setup__(text_encoder, vision)

        if fix_num_images is not None:
            hf_model.fix_num_images = fix_num_images
        # hf_model.embed_tokens = hf_model.model.embed_tokens
        # del hf_model.text_encoder
        # del hf_model.lm_head
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return hf_model

    def extract_feature(self, pixel_values, image_flags=None):
        # vision_model_output = self.vision_model(
        #     pixel_values=pixel_values,
        #     output_hidden_states=False,
        #     return_dict=True) 
        
        windows_tensor, win_meta_list, spatial_shapes, reverse_mapping = self.vision_model.vision_model.embeddings(pixel_values)
        print(windows_tensor.shape, spatial_shapes)
        # breakpoint()
        vision_model_output = self.hm_vison(windows_tensor.half())

        # if hasattr(vision_model_output, 'last_hidden_state'):
        #     vit_embeds = vision_model_output.last_hidden_state # [1, 324, 1152]
        # if hasattr(vision_model_output, 'spatial_shapes'):
        #     spatial_shapes = vision_model_output.spatial_shapes
        vision_model_output = vision_model_output[:, reverse_mapping, :]

        vit_embeds = vision_model_output 
        vit_embeds, spatial_shapes = self.pixel_shuffle_back(vit_embeds, spatial_shapes)

        vit_embeds = self.mlp1(vit_embeds)
        
        B, N, C = vit_embeds.shape
        vit_embeds = vit_embeds.reshape(B * N, C)

        return vit_embeds

    def extract_feature_ori(self, pixel_values, image_flags=None):

        if self.select_layer == -1:
            vision_model_output = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=False,
                return_dict=True)
            if hasattr(vision_model_output, 'last_hidden_state'):
                vit_embeds = vision_model_output.last_hidden_state
            if hasattr(vision_model_output, 'spatial_shapes'):
                spatial_shapes = vision_model_output.spatial_shapes
        else:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True).hidden_states[self.select_layer]

        vit_embeds, spatial_shapes = self.pixel_shuffle_back(vit_embeds, spatial_shapes)


        if self.mlp_checkpoint and vit_embeds.requires_grad:
            vit_embeds = cp.checkpoint(self.mlp1, vit_embeds)
        else:
            vit_embeds = self.mlp1(vit_embeds)
        
        B, N, C = vit_embeds.shape
        vit_embeds = vit_embeds.reshape(B * N, C)
        
        if image_flags is not None and any(image_flags==0):
            vit_embeds = self.mask_valid_tokens(vit_embeds, spatial_shapes, image_flags)
            
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

        input_len = input_ids.shape[1] if input_ids is not None else 0
        # if input_len < 256:
        #     pad_len = 256 - input_len
        #     padding_input_ids = torch.ones((1, pad_len), dtype=torch.long).to(
        #         self.device
        #     ) * 151643
        #     # padding_input_ids.fill_(151643)
        #     input_ids = torch.cat([input_ids, padding_input_ids], dim=-1)            

        input_embeds = self.language_model.get_input_embeddings()(input_ids)
        
        num_images = len(pixel_values)

        if image_flags is not None:
            image_flags = image_flags.view(-1)

        print(len(pixel_values))
        # for img in pixel_values:
        #     print(img.shape)
        if True:
            vit_embeds = []
            for i in range(num_images):
                vit_embeds.append( self.extract_feature([pixel_values[i]], image_flags)) # [1, 3, 252, 252] => 1,81,2048
            vit_embeds = torch.cat(vit_embeds, dim=0) # [num_images, 81, 2048]
        else:
            vit_embeds = self.extract_feature(pixel_values, image_flags) # [num_images, 81, 2048]

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

        # breakpoint() # [5,121,2048]

        if self.fix_num_images is not None:
            num_images = self.fix_num_images
        else:
            num_images = len(pixel_values)
        
        outputs = []
        for i in range(num_images):
            input_embeds_single = input_embeds[i:i+1, ...]
            # outputs_single = self.language_model(
            #     inputs_embeds=input_embeds_single,
            #     attention_mask=attention_mask,
            #     position_ids=position_ids,
            #     past_key_values=past_key_values,
            #     use_cache=use_cache,
            #     output_attentions=output_attentions,
            #     output_hidden_states=output_hidden_states,
            # ) # [1,
            input_len = input_embeds_single.shape[1]
            if input_len < 256:
                pad_len = 256 - input_len  
                padding_input_ids = torch.ones((1, pad_len, 2048), dtype=torch.long).to(
                    self.device
                ) * 151643
                # padding_input_ids.fill_(151643)
                input_em_pad = torch.cat([input_embeds_single, padding_input_ids], dim=1) 
            else:
                input_em_pad = input_embeds_single

            outputs_single = self.hm_text_encoder(input_em_pad)

            outputs_single = outputs_single[:, :input_len, :]

            outputs.append(outputs_single)
        outputs = torch.cat(outputs, dim=0)
        

        # kv_cache_shape = [1, 8, 2048, 128]
        # past_key_caches = []
        # past_value_caches = []

        # for _ in range(16):
        #     past_key_caches.append(CacheTensor(torch.zeros(kv_cache_shape, dtype=torch.float16, device=self.device)))
        #     past_value_caches.append(CacheTensor(torch.zeros(kv_cache_shape, dtype=torch.float16, device=self.device)))

        # outputs_hm = self.hm_text_encoder(
        #     input_embeds,
        #     torch.tensor([0], device=self.device, dtype=torch.int32),
        #     torch.tensor([256], device=self.device, dtype=torch.int32),
        #     # position_ids,
        #     past_key_caches,
        #     past_value_caches,            
        # )
        # logits = outputs.logits   [1, 109, 151680]  [1, 109, 2048]

        # hidden_states_l = list(outputs.hidden_states)
        # hidden_states_l[-1] = hidden_states_l[-1][:,:109,:]
        # outputs.hidden_states = tuple(hidden_states_l)
        
        output_dict = {
            "hidden_states": [outputs],
        }
        return output_dict
