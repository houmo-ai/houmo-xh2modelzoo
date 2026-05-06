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
# File: configuration_eagle3_vl.py
# Description:
#   GR00T (stub; license review pending) model configuration adapted for the xh2 model zoo (xh2modelzoo).

from transformers.configuration_utils import PretrainedConfig


class Eagle3_VLConfig(PretrainedConfig):
    model_type = "eagle_3_vl"
    is_composition = True
    sub_configs = {}

    def __init__(
        self,
        vision_config=None,
        text_config=None,
        use_backbone_lora=0,
        use_llm_lora=0,
        pad2square=False,
        select_layer=-4,
        downsample_ratio=0.5,
        template=None,
        loss_version="v1",
        mlp_checkpoint=False,
        image_token_index=151667,
        **kwargs,
    ):
        raise NotImplementedError(
            "Eagle3_VLConfig: Implementation removed pending upstream starforce license clarification. "
            "This model is not available in the Apache-2.0 release."
        )

    def to_dict(self):
        raise NotImplementedError("Eagle3_VLConfig removed; to_dict() unavailable.")
