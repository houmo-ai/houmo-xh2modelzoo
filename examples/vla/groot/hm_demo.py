# Copyright 2025 HOUMO AI
#
# File: text_encoder_demo.py
# Description:
#   Example script: llm/zimage/text_encoder_demo.py
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

import torch
from pathlib import Path
import argparse
from xhquant.api import DeviceType, xhquant_init, QuantScheme, get_root_logger, HMONNXGoldenInference, Config
import numpy as np
from xh_model_zoo.xh_llm.models.groot.gr00t.policy.gr00t_policy import Gr00tPolicy
from xh_model_zoo.xh_llm.models.groot.gr00t.data.embodiment_tags import EmbodimentTag
import torch

from xh_model_zoo.xh_llm.models.groot.inference import Qwen3LegacyInference
from xh_model_zoo.xh_llm.models.groot.cus_egale3 import cus_eagle3_inference
from xh_model_zoo.xh_llm.models.groot.cus_groot import cus_GROOT
def main(args):
    model_name = "/data02/datasets/groot"
    device = "cuda"
    np.random.seed(42)
    policy = Gr00tPolicy(
        model_path='/data02/datasets/GROOT-N1.6-3B',
        embodiment_tag=EmbodimentTag('gr1'),
        device='cuda',
    )
    # policy.model.backbone.model.vision_model
    obs = {
        'video': {
            'ego_view_bg_crop_pad_res256_freq20': np.random.randint(0, 255, (1, 1, 256, 256, 3), dtype=np.uint8),
        },
        'state': {
            'left_arm': np.random.rand(1, 1, 7).astype(np.float32),
            'right_arm': np.random.rand(1, 1, 7).astype(np.float32),
            'left_hand': np.random.rand(1, 1, 6).astype(np.float32),
            'right_hand': np.random.rand(1, 1, 6).astype(np.float32),
            'waist': np.random.rand(1, 1, 3).astype(np.float32),
        },
        'language': {
            'task': [['pick up the red apple and bule ball']],
        },
    }


    vision_model_path = "/data01/home/xuchen/xh2/xh2_model_zoo/work_dirs/groot/hmonnx/groot_vision-XH2a-w8a8h1_sefp.onnx"
    vision_model = HMONNXGoldenInference(vision_model_path)
    vision_model.exec_device = torch.device("cuda:0")

    qwen3_inference_engine = Qwen3LegacyInference("work_dirs/groot/meta.json", fast_mode=True, tokenizer=policy.collate_fn.processor.tokenizer)
    
    backbone_engine = cus_eagle3_inference.to_hf_compatible(
        hf_model=policy.model.backbone.model,
        text_encoder=qwen3_inference_engine,
        vision=vision_model,
        meta_info="work_dirs/groot/meta.json",
    )

    head_pre_path= "/data01/home/xuchen/xh2/xh2_model_zoo/work_dirs/groot_head/hmonnx/groot_head_pre-XH2a-w8a8h1_sefp.onnx"
    head_pre_model = HMONNXGoldenInference(head_pre_path)
    head_pre_model.exec_device = torch.device("cuda:0")

    head_path= "/data01/home/xuchen/xh2/xh2_model_zoo/work_dirs/groot_head/hmonnx/groot_head-XH2a-w8a8h1_sefp.onnx"
    head_model = HMONNXGoldenInference(head_path)
    head_model.exec_device = torch.device("cuda:0")

    xhmodel = cus_GROOT.to_hf_compatible(
        hf_model=policy.model,
        backbone=backbone_engine,
        new_network_pre=head_pre_model,
        new_network=head_model,
        meta_info="work_dirs/groot/meta.json",
    )


    torch.cuda.empty_cache()
    policy.model = policy.model.to(torch.float16)
    # policy.model.backbone.model = backbone_engine


    work_dir = Path("work_dirs") / "groot"
    work_dir.mkdir(exist_ok=True, parents=True)

    policy.model = xhmodel
    action = policy.get_action(obs)
    print('Action output:')
    for k, v in action[0].items():
        print(f'  {k}: shape={v.shape}')
    print('GR00T Inference Success!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument("--model", type=str, default="weights/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--batch-size", type=int, default=1, help="batch size")
    parser.add_argument("--context-length", type=int, default=2048, help="max sequence length")
    parser.add_argument("--input-sequence-length", type=int, default=256, help="input sequence length")
    parser.add_argument("--max_pe_length", type=int, default=32768, help="max pe length")
    parser.add_argument("--quant-type", default="w8a8h1_sefp", help="quant type, default is w8a8")
    parser.add_argument("--image_max_size_h", type=int, default=448, help="image max size height")
    parser.add_argument("--image_max_size_w", type=int, default=448, help="image max size width")
    parser.add_argument("--image_max_size_t", type=int, default=2, help="if image, temporal max size is 2, if video, temporal max size is fps")
    parser.add_argument("--patch_size", type=int, default=14, help="patch size")
    parser.add_argument("--temporal_patch_size", type=int, default=2, help="temporal patch size")
    parser.add_argument("--sample_image_path", type=str, default="data/images/qwen2_vl_demo.jpeg", help="sample image path for generate golden")
    parser.add_argument("--use_gptqmodel", action="store_true", help="use gptqmodel quanted model")
    parser.add_argument(
        "--quant_weight",
        type=str,
        default=None,
        help="quant weight path, for example: gptq or quarot, if empty, use w8a8",
    )
    args = parser.parse_args()
    main(args)
