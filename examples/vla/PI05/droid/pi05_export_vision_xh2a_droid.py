# -*- coding: utf-8 -*-
# Copyright 2025 HOUMO AI
#
# File: pi05_export_vision_xh2a_droid.py
# Description:
#   Export utilities for droid in HOUMO AI xh2modelzoo.
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
# SPDX-License-Identifier: Apache-2.0

import os
import torch
import numpy as np
import torch.nn as nn
import random
import onnx
import os.path as osp
from copy import deepcopy
from onnxsim import simplify

import argparse
from lerobot.policies.pi05 import PI05Config, PI05Policy
from lerobot.policies.pi05.processor_pi05 import make_pi05_pre_post_processors

from xhquant.api import convert_onnx_to_hmonnx, QuantScheme, create_quant_config, DeviceType
os.environ["ENABLE_LAYERNORM2RMSNORM"] = "1"

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
workdir = os.path.join(ROOT_DIR, "workdir")
hmonnx_file = os.path.join(ROOT_DIR, "hmonnx")
os.makedirs(workdir, exist_ok=True)
os.makedirs(hmonnx_file, exist_ok=True)

class Siglip(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.vision_tower = model.vision_tower.eval()
        self.multi_modal_projector = model.multi_modal_projector.eval()

    def forward(self, pixel_values):
        image_outputs = self.vision_tower(pixel_values)
        selected_image_feature = image_outputs.last_hidden_state
        image_features = self.multi_modal_projector(selected_image_feature)
        return image_features

# ---------------------------------------------------------------
# 固定随机种子
# ---------------------------------------------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

set_seed(42)

DEVICE = "cpu"

# ---------------------------------------------------------------
# Dummy 配置（与官方测试一致）
# ---------------------------------------------------------------
DUMMY_ACTION_DIM = 8
DUMMY_STATE_DIM = 8
DUMMY_ACTION_HORIZON = 15

DUMMY_DATASET_STATS = {
    "observation.state": {
        "mean": torch.zeros(DUMMY_STATE_DIM),
        "std": torch.ones(DUMMY_STATE_DIM),
        "q01": torch.zeros(DUMMY_STATE_DIM),
        "q99": torch.ones(DUMMY_STATE_DIM),
    },
    "action": {
        "mean": torch.zeros(DUMMY_ACTION_DIM),
        "std": torch.ones(DUMMY_ACTION_DIM),
        "q01": torch.zeros(DUMMY_ACTION_DIM),
        "q99": torch.ones(DUMMY_ACTION_DIM),
    },
    "images": {
        "image": {"mean": torch.zeros(3,256,256), "std": torch.ones(3,256,256)},
        "image2": {"mean": torch.zeros(3,256,256), "std": torch.ones(3,256,256)},
        "empty_camera_0": {"mean": torch.zeros(3,224,224), "std": torch.ones(3,224,224)},
    },
}

# ---------------------------------------------------------------
# ① 载入 PI0.5（LeRobot）
# ---------------------------------------------------------------
def load_pi05(model_path):
    policy = PI05Policy.from_pretrained(model_path, strict=True)
    policy.to(DEVICE)
    policy.config.device = DEVICE

    preprocessor, postprocessor = make_pi05_pre_post_processors(
        config=policy.config,
        dataset_stats=DUMMY_DATASET_STATS,
    )
    return policy, preprocessor, postprocessor

# ---------------------------------------------------------------
# ② 构造输入 batch（PI0.5 需要 state + image + task）
# ---------------------------------------------------------------
def create_dummy_batch(batch_size=1):
    prompt = "Pick up the red block and place it in the bin"
    return {
        "observation.state": torch.randn(batch_size, DUMMY_STATE_DIM, device=DEVICE),
        "action": torch.randn(batch_size, DUMMY_ACTION_HORIZON, DUMMY_ACTION_DIM, device=DEVICE),
        "observation.images.image": torch.rand(batch_size, 3, 256, 256, device=DEVICE),
        "observation.images.image2": torch.rand(batch_size, 3, 256, 256, device=DEVICE),
        "observation.images.empty_camera_0": torch.rand(batch_size, 3, 224, 224, device=DEVICE),
        "task": [prompt] * batch_size,
    }

# ---------------------------------------------------------------
# ③ PI0.5 推理（与官方 test 中 LeRobot 路径一致）
# ---------------------------------------------------------------
def run_pi05_inference(args):
    policy, preprocessor, postprocessor = load_pi05(args.model_path)
    batch = create_dummy_batch(batch_size=1)
    policy.eval()
    with torch.no_grad():
        batch_proc = preprocessor(deepcopy(batch))
        actions = policy.predict_action_chunk(batch_proc)
    print(actions)
    print("actions shape:", actions.shape)
    print("mean:", actions.mean().item())
    print("std:", actions.std().item())
    
    #导出vision-tower + multi_modal_projector onnx
    if True:
        # vision-tower
        siglip_model = Siglip(policy.model.paligemma_with_expert.paligemma.model)
        siglip_model.eval()
        siglip_model = siglip_model.float()
        temp_onnx_file = "./workdir/pi0.5_siglip_droid.onnx"
        simplified_onnx_file = "./workdir/pi0.5_siglip_droid_simplified.onnx"
        input_features = torch.randn(1, 3, 224, 224, dtype=torch.float32)

        torch.onnx.export(
            siglip_model,
            input_features,
            temp_onnx_file,
            input_names=["pixel_values"],
            output_names=["output"],
            opset_version=17,
            verbose=False,
        )
        onnx_model = onnx.load(temp_onnx_file)
        model_simplified, check = simplify(
            onnx_model,
            test_input_shapes={"pixel_values": [1, 3, 224, 224]},
        )

        if check:
            onnx.save(model_simplified, simplified_onnx_file)
            print("Simplified:", simplified_onnx_file)
        else:
            print("Simplify failed")

        import onnxruntime as ort
        import numpy as np

        # ===== PyTorch 输出 =====
        with torch.no_grad():
            torch_out = siglip_model(input_features)
            if isinstance(torch_out, (tuple, list)):
                torch_out = torch_out[0]
        torch_out_np = torch_out.float().cpu().numpy()

        # ===== ONNX Runtime 输出 =====
        sess = ort.InferenceSession(
            simplified_onnx_file,
            providers=["CUDAExecutionProvider"],
        )
        onnx_out = sess.run(
            None,
            {"pixel_values": input_features.cpu().numpy()},
        )
        onnx_out_np = onnx_out[0]

        # ===== 对比 =====
        print("max abs diff:", np.max(np.abs(torch_out_np - onnx_out_np)))
        print("mean abs diff:", np.mean(np.abs(torch_out_np - onnx_out_np)))
        print(
            "allclose:",
            np.allclose(torch_out_np, onnx_out_np, rtol=1e-3, atol=1e-4),
        )
    
    #导出vision-tower + multi_modal_projector的hmonnx
    input = torch.randn(1, 3, 224, 224)
    quant_type = "w8a8h1_sefp"
    quant_scheme = QuantScheme(target_device=DeviceType.XH2a, quant_type=quant_type)
    quant_config = create_quant_config(quant_scheme)
    convert_onnx_to_hmonnx(simplified_onnx_file, (input,), out_hmonnx_file=osp.join(hmonnx_file, "vision.onnx"), device_type="XH2A", quant_config=quant_config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/data01/home/she.gao/.cache/huggingface/hub/pi05-droid")
    args = parser.parse_args()
    run_pi05_inference(args)
