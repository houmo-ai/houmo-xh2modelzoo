# Copyright 2025 HOUMO AI
#
# File: evaluate_yolov7_on_coco.py
# Description:
#   Example script: cv/yolo6_yolo7_test/evaluate_yolov7_on_coco.py
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

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import yaml
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from xhquant.api import HMONNXInference, get_root_logger, xhquant_init


# ==============================================================================
# 数据集加载模块 (Dataset Loader Module)
# ==============================================================================
class COCODataset(Dataset):
    def __init__(self, img_dir, img_size=640):
        self.img_dir = img_dir
        self.img_size = (img_size, img_size)
        self.files = [
            os.path.join(self.img_dir, f) for f in os.listdir(self.img_dir) if f.endswith((".jpg", ".jpeg", ".png"))
        ]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img_path = self.files[index]
        image = cv2.imread(img_path)
        image_id = int(Path(img_path).stem)

        results = self._preprocess_image(image, self.img_size)

        return {"image_id": image_id, "inputs": results["inputs"], "metas": results["metas"]}

    def _letterbox(self, im, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleup=True, stride=32):
        shape = im.shape[:2]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:
            r = min(r, 1.0)
        ratio = r, r
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        dw /= 2
        dh /= 2
        if shape[::-1] != new_unpad:
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return im, ratio, (dw, dh)

    def _preprocess_image(self, image, input_img_size) -> Dict[str, Any]:
        img_resized, ratio, pad = self._letterbox(image, input_img_size, auto=False)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_chw = img_rgb.transpose((2, 0, 1))
        img_contiguous = np.ascontiguousarray(img_chw)
        img_tensor = torch.from_numpy(img_contiguous).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(torch.float16)
        return {
            "metas": {"ori_shape": image.shape, "resized_shape": img_resized.shape, "ratio": ratio, "pad": pad},
            "inputs": img_tensor,
        }


# ==============================================================================
# 后处理函数 (Post-processing Functions)
# ==============================================================================
def _xywh2xyxy(x):
    y = x.clone()
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def _non_max_suppression(prediction, conf_thres=0.001, iou_thres=0.6, max_det=300):
    nc = prediction.shape[2] - 5
    xc = prediction[..., 4] > conf_thres
    max_wh = 7680
    max_nms = 30000
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):
        x = x[xc[xi]]
        if not x.shape[0]:
            continue
        x[:, 5:] *= x[:, 4:5]
        box = _xywh2xyxy(x[:, :4])
        conf, j = x[:, 5:].max(1, keepdim=True)
        x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]
        n = x.shape[0]
        if not n:
            continue
        elif n > max_nms:
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]
        c = x[:, 5:6] * (0 if False else max_wh)
        boxes, scores = x[:, :4] + c, x[:, 4]
        i = torchvision.ops.nms(boxes, scores, iou_thres)
        if i.shape[0] > max_det:
            i = i[:max_det]
        output[xi] = x[i]
    return output


def _scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]
    coords[:, [0, 2]] -= pad[0]
    coords[:, [1, 3]] -= pad[1]
    coords[:, :4] /= gain
    coords[:, 0].clamp_(0, img0_shape[1])
    coords[:, 1].clamp_(0, img0_shape[0])
    coords[:, 2].clamp_(0, img0_shape[1])
    coords[:, 3].clamp_(0, img0_shape[0])
    return coords


def _make_grid(nx=20, ny=20):
    yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)], indexing="ij")
    return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


def postprocess_yolov7(outputs: Tuple[torch.Tensor], metas: Dict[str, Any]):
    device = outputs[0].device
    strides = torch.tensor([8.0, 16.0, 32.0]).to(device)
    anchors = torch.tensor(
        [
            [[12.0, 16.0], [19.0, 36.0], [40.0, 28.0]],
            [[36.0, 75.0], [76.0, 55.0], [72.0, 146.0]],
            [[142.0, 110.0], [192.0, 243.0], [459.0, 401.0]],
        ]
    ).to(device)
    predictions = []
    for i, (out, stride, anchor) in enumerate(zip(outputs, strides, anchors)):
        # --- 关键修复: 正确解包5维张量 ---
        bs, na, ny, nx, no = out.shape
        grid = _make_grid(nx, ny).to(device)
        anchor_grid = anchor.view(1, -1, 1, 1, 2)

        y = out.sigmoid()
        y[..., 0:2] = (y[..., 0:2] * 2.0 - 0.5 + grid) * stride
        y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * anchor_grid
        predictions.append(y.reshape(bs, -1, 85))

    prediction = torch.cat(predictions, 1)
    return _non_max_suppression(prediction, conf_thres=0.001, iou_thres=0.65)


# ==============================================================================
# 主评测逻辑 (Main Evaluation Logic)
# ==============================================================================
def evaluate(opt):
    logger = get_root_logger()

    with open(opt.data) as f:
        data_cfg = yaml.safe_load(f)

    logger.info(f"正在加载YOLOv7模型: {opt.hmonnx}")
    xhquant_init(None)
    session = HMONNXInference(opt.hmonnx)
    exec_device = torch.device("cuda")
    session.to(exec_device)
    input_name = session.get_input_names()[0]

    logger.info(f"正在从 '{data_cfg['val']}' 加载数据集...")
    dataset = COCODataset(img_dir=data_cfg["val"], img_size=opt.img_size)
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    all_results = []
    coco91class = [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        27,
        28,
        31,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        40,
        41,
        42,
        43,
        44,
        46,
        47,
        48,
        49,
        50,
        51,
        52,
        53,
        54,
        55,
        56,
        57,
        58,
        59,
        60,
        61,
        62,
        63,
        64,
        65,
        67,
        70,
        72,
        73,
        74,
        75,
        76,
        77,
        78,
        79,
        80,
        81,
        82,
        84,
        85,
        86,
        87,
        88,
        89,
        90,
    ]

    logger.info("开始在COCO val2017上进行评测...")
    for batch in tqdm(dataloader):
        inputs = batch["inputs"].squeeze(1).to(exec_device)
        metas = batch["metas"]
        image_ids = batch["image_id"]

        raw_outputs = session.run({input_name: inputs})

        outputs_cpu = [out.cpu() for out in raw_outputs]
        detections = postprocess_yolov7(outputs_cpu, metas)

        for i, pred in enumerate(detections):
            image_id = image_ids[i].item()

            ori_shape = (
                metas["ori_shape"][0][i].item(),
                metas["ori_shape"][1][i].item(),
                metas["ori_shape"][2][i].item(),
            )
            ratio_i = (metas["ratio"][0][i].item(), metas["ratio"][1][i].item())
            pad_i = (metas["pad"][0][i].item(), metas["pad"][1][i].item())
            ratio_pad_i = (ratio_i, pad_i)

            if len(pred) == 0:
                continue

            pred[:, :4] = _scale_coords(inputs[i].shape[1:], pred[:, :4], ori_shape[:2], ratio_pad=ratio_pad_i).round()

            for *xyxy, conf, cls in pred:
                xywh = [xyxy[0].item(), xyxy[1].item(), (xyxy[2] - xyxy[0]).item(), (xyxy[3] - xyxy[1]).item()]
                coco_result = {
                    "image_id": image_id,
                    "category_id": coco91class[int(cls)],
                    "bbox": [round(x, 3) for x in xywh],
                    "score": round(conf.item(), 5),
                }
                all_results.append(coco_result)

    if not all_results:
        logger.warning("模型未检测到任何物体，无法进行评测。")
        return

    results_path = opt.output_json
    logger.info(f"评测完成，正在将结果写入 {results_path}")
    with open(results_path, "w") as f:
        json.dump(all_results, f)

    coco_gt = COCO(data_cfg["annotations_path"])
    coco_dt = coco_gt.loadRes(results_path)

    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    logger.info(f"评测指标计算完成。mAP@.5:.95 = {coco_eval.stats[0]:.4f}, mAP@.5 = {coco_eval.stats[1]:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hmonnx", type=str, required=True, help="量化后的 .hmonnx 模型文件路径")
    parser.add_argument("--data", type=str, default="coco_eval.yaml", help="数据集配置文件路径")
    parser.add_argument("--batch-size", type=int, default=1, help="评测时的批大小")
    parser.add_argument("--img-size", type=int, default=640, help="图片推理尺寸")
    parser.add_argument(
        "--output-json", type=str, default="coco_results_yolov7.json", help="保存COCO格式JSON结果的路径"
    )
    opt = parser.parse_args()

    try:
        from pycocotools.coco import COCO
    except ImportError:
        print("正在安装 pycocotools... (可能需要几分钟)")
        os.system(
            "pip install cython; pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'"
        )

    evaluate(opt)
