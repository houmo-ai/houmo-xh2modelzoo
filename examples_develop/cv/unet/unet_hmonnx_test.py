# Copyright 2025 HOUMO AI
#
# File: unet_hmonnx_test.py
# Description:
#   UNet HMONNX example test with image preprocessing and segmentation postprocessing.
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

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from xhquant.api import HMONNXInference, get_root_logger, xhquant_init


def preprocess_image(image):
    h, w = image.shape[-2:]
    scale_factor_w = w / 2048
    scale_factor_h = h / 1024
    image = cv2.resize(image, (2048, 1024))
    mean = [123.675, 116.28, 103.53]
    std = [58.395, 57.12, 57.375]
    mean = np.array(mean, dtype=np.float32).reshape(1, 1, -1)
    std = np.array(std, dtype=np.float32).reshape(1, 1, -1)
    image = (image - mean) / std
    image = torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0).to(torch.float16)
    return {
        "metas": [
            {
                "ori_shape": image.shape[-2:],
                "scale_factor": (scale_factor_w, scale_factor_h),
            }
        ],
        "inputs": image,
    }


def postprocess_result(seg_logits: Tensor, image_metas: list[dict]):
    batch_size = seg_logits.shape[0]
    seg_results = []
    for i in range(batch_size):
        image_meta = image_metas[i]
        ori_shape = image_meta["ori_shape"]
        i_seg_logit = seg_logits[i : i + 1]
        i_seg_logit = F.interpolate(i_seg_logit, tuple(ori_shape), None, mode="bilinear", align_corners=False).squeeze(
            0
        )
        i_seg_pred = i_seg_logit.argmax(dim=0, keepdim=True)
        seg_results.append(
            {
                "seg_logit": i_seg_logit,
                "seg_pred": i_seg_pred,
            }
        )
    return seg_results


def main(args):
    xhquant_init(None, debug=args.debug)
    work_dirs = Path("work_dirs") / "fcn_unet_s5-d16_4x4_512x1024_160k_cityscapes"
    work_dirs.mkdir(exist_ok=True, parents=True)

    session = HMONNXInference(args.hmonnx)
    exec_device = torch.device("cuda")
    session.to(exec_device)
    logger = get_root_logger()
    logger.info("session is created successfully")
    image_file = args.image
    image_fname = Path(image_file).stem

    # 读取图像并进行预处理
    image = cv2.imread(image_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    inputs = preprocess_image(image)
    input = inputs["inputs"].to(exec_device)
    image_metas = inputs["metas"]

    input_name = session.get_input_names()[0]
    seg_logits = session.run({input_name: input})
    seg_results = postprocess_result(seg_logits, image_metas)
    seg_result = seg_results[0]  # batch 0
    sem_seg = seg_result["seg_pred"]

    # 可视化
    classes = (
        "road",
        "sidewalk",
        "building",
        "wall",
        "fence",
        "pole",
        "traffic light",
        "traffic sign",
        "vegetation",
        "terrain",
        "sky",
        "person",
        "rider",
        "car",
        "truck",
        "bus",
        "train",
        "motorcycle",
        "bicycle",
    )

    palette = [
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [70, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
    ]

    num_classes = len(classes)

    sem_seg = sem_seg.cpu().data
    ids = np.unique(sem_seg)[::-1]
    legal_indices = ids < num_classes
    ids = ids[legal_indices]
    labels = np.array(ids, dtype=np.int64)

    colors = [palette[label] for label in labels]
    alpha = 0.8
    mask = np.zeros_like(image, dtype=np.uint8)
    for label, color in zip(labels, colors):
        mask[sem_seg[0] == label, :] = color
    color_seg = (image * (1 - alpha) + mask * alpha).astype(np.uint8)

    out_dir = work_dirs
    out_file = out_dir / f"{image_fname}_seg_vis.jpg"
    cv2.imwrite(str(out_file), color_seg)
    logger.info(f"Seg result is saved to {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hmonnx",
        type=str,
        default="work_dirs/fcn_unet_s5-d16_4x4_512x1024_160k_cityscapes/fcn_unet_s5-d16_4x4_512x1024_160k_cityscapes_XH2a.onnx",
    )
    parser.add_argument("--image", type=str, default="data/images/berlin_000006_000019_leftImg8bit.png")
    parser.add_argument("--debug", action="store_true", help="debug mode")
    args = parser.parse_args()
    main(args)
