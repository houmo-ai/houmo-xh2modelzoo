# -*- coding: utf-8 -*-
# Copyright 2024 Alibaba Inc. (FunAudioLLM Team). All rights reserved.
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
# File: qwen2_05b_instruct_xh2a_2k.py
# Description:
#   CosyVoice3 TTS module support code adapted for the xh2 model zoo (xh2modelzoo).

quant_config = dict(inputs=dict())
target_device = "XH2a"
frontend_type = "TorchFX"
hf_model_dir = "/data01/home/she.gao/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B/CosyVoice-BlankEN"  # 模型路径

quant_config = dict(
    inputs=dict(
        inputs_embeds=dict(
            quantizer=dict(
                qspec=dict(fake_dtype="float16"),
            )
        ),
        past_seq_length=dict(
            quantizer=dict(
                qspec=dict(fake_dtype="int32"),
            )
        ),
        current_input_length=dict(
            quantizer=dict(
                qspec=dict(fake_dtype="int32"),
            )
        ),
    )
)

model = dict(
    type="XHQwen2LegacyModel",
    hf_model=hf_model_dir,
    wrap_cfg=dict(
        max_sequence_length=2048,
        input_sequence_length=256,
        use_cache=True,
        num_logits_to_keep=1,
        kv_cache=dict(
            cache_axis=2,
        ),
    ),  # wrap模型时，需要传入的配置参数
    quant_config=quant_config,
    frontend_type=frontend_type,
    export_cfg=dict(
        input_names=[
            "inputs_embeds",
            "past_seq_length",
            "current_input_length",
        ],
        output_names=["logits"],
    ),
)
