# -*- coding: utf-8 -*-
# Copyright 2025 NVIDIA Corporation. All rights reserved.
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
# File: modeling_eagle3_vl.py
# Description:
#   GR00T (stub; license review pending) model definition adapted for the xh2 model zoo (xh2modelzoo).

from typing import Optional, Tuple, Union
from transformers.modeling_utils import PreTrainedModel
from transformers.generation import GenerationMixin
from .configuration_eagle3_vl import Eagle3_VLConfig


class Eagle3_VLPreTrainedModel(PreTrainedModel):
    config_class = Eagle3_VLConfig
    base_model_prefix = "model"
    main_input_name = "input_ids"
    supports_gradient_checkpointing = True
    _no_split_modules = []
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_cache_class = True
    _supports_static_cache = True
    _supports_quantized_cache = True
    _supports_sdpa = True

    def _init_weights(self, module):
        raise NotImplementedError(
            "Eagle3_VLPreTrainedModel: Implementation removed pending upstream starforce license clarification."
        )


class Eagle3_VLForConditionalGeneration(Eagle3_VLPreTrainedModel, GenerationMixin):
    config_class = Eagle3_VLConfig

    def __init__(self, config: Eagle3_VLConfig, vision_model=None, language_model=None):
        raise NotImplementedError(
            "Eagle3_VLForConditionalGeneration: Implementation removed pending upstream starforce license clarification. "
            "This model is not available in the Apache-2.0 release."
        )

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Eagle3_VLForConditionalGeneration removed.")

    def prepare_inputs_for_generation(self, *args, **kwargs):
        raise NotImplementedError("Eagle3_VLForConditionalGeneration removed.")

    def _reorder_cache(self, *args, **kwargs):
        raise NotImplementedError("Eagle3_VLForConditionalGeneration removed.")
