# Copyright 2025 HOUMO AI
#
# File: yolov5_hmonnx_test.py
# Description:
#   Example script: cv/yolo/yolov5/yolov5_hmonnx_test.py
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

# yolov5_hmonnx_test.py
import argparse
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import yaml
from xhquant.api import HMONNXInference, get_root_logger, xhquant_init


# ==============================================================================
# 预处理部分
# ==============================================================================
def _scale_size(
    size: Tuple[int, int],
    scale: Union[float, int, Tuple[float, float], Tuple[int, int]],
) -> Tuple[int, int]:
    if isinstance(scale, (float, int)):
        scale = (scale, scale)
    w, h = size
    return int(w * float(scale[0]) + 0.5), int(h * float(scale[1]) + 0.5)


def rescale_size(
    old_size: tuple,
    scale: Union[float, int, Tuple[int, int]],
    return_scale: bool = False,
) -> tuple:
    w, h = old_size
    if isinstance(scale, (float, int)):
        if scale <= 0:
            raise ValueError(f"Invalid scale {scale}, must be positive.")
        scale_factor = scale
    elif isinstance(scale, tuple):
        max_long_edge = max(scale)
        max_short_edge = min(scale)
        scale_factor = min(max_long_edge / max(h, w), max_short_edge / min(h, w))
    else:
        raise TypeError(f"Scale must be a number or tuple of int, but got {type(scale)}")
    new_size = _scale_size((w, h), scale_factor)
    if return_scale:
        return new_size, scale_factor
    else:
        return new_size


def preprocess_image(image, input_img_size) -> Dict[str, Any]:
    h, w = image.shape[:2]
    new_size, scale_factor = rescale_size((w, h), input_img_size, return_scale=True)

    resized_img = cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)

    # 计算 padding
    target_w, target_h = input_img_size
    pad_w = target_w - new_size[0]
    pad_h = target_h - new_size[1]

    # 使用cv2.copyMakeBorder进行padding，更高效
    # BGR (114, 114, 114) for padding
    padded_img = cv2.copyMakeBorder(resized_img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(114, 114, 114))

    # BGR to RGB, HWC to CHW, to Tensor
    padded_img = padded_img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
    padded_img = np.ascontiguousarray(padded_img)

    img_tensor = torch.from_numpy(padded_img).unsqueeze(0).float() / 255.0

    return {
        "metas": [{"ori_shape": (h, w), "scale_factor": scale_factor, "pad_size": (pad_w, pad_h)}],
        "inputs": img_tensor.to(torch.float16),
    }


# ==============================================================================
# 后处理部分 (为YOLOv5重写)
# ==============================================================================
def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y


def postprocess_v5(prediction, conf_thres=0.25, iou_thres=0.45, max_det=300):
    """
    YOLOv5 Post-processing function.
    Args:
        prediction (torch.Tensor): Model output, shape [1, 25200, 85]
    Returns:
        A list of detections, each is a tensor with shape [N, 6] -> [x1, y1, x2, y2, conf, class]
    """
    # 85 = 4 (xywh) + 1 (obj_conf) + 80 (class_conf)
    nc = prediction.shape[2] - 5  # number of classes

    # Filter out detections with low object confidence
    xc = prediction[..., 4] > conf_thres

    output = [torch.zeros((0, 6))] * prediction.shape[0]

    for xi, x in enumerate(prediction):  # Iterate through batch
        x = x[xc[xi]]  # Apply confidence filter

        if not x.shape[0]:
            continue

        # Compute confidence: obj_conf * class_conf
        x[:, 5:] *= x[:, 4:5]

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Get best class score and its index
        conf, j = x[:, 5:].max(1, keepdim=True)

        # Final detections matrix: [x1, y1, x2, y2, conf, class_id]
        x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        n = x.shape[0]  # number of boxes
        if not n:
            continue

        # Apply Non-Maximum Suppression (NMS)
        boxes, scores = x[:, :4], x[:, 4]
        i = torchvision.ops.nms(boxes, scores, iou_thres)

        if i.shape[0] > max_det:
            i = i[:max_det]

        output[xi] = x[i]

    return output


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain

    # Clip coordinates
    coords[:, [0, 2]] = coords[:, [0, 2]].clamp(0, img0_shape[1])  # x1, x2
    coords[:, [1, 3]] = coords[:, [1, 3]].clamp(0, img0_shape[0])  # y1, y2
    return coords


def yaml_load(file="data.yaml"):
    with open(file, errors="ignore", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ==============================================================================
# Main Execution
# ==============================================================================
def main(args):
    xhquant_init(None, debug=args.debug)
    session = HMONNXInference(args.hmonnx)
    exec_device = torch.device("cuda")
    session.to(exec_device)
    logger = get_root_logger()
    logger.info("session is created successfully")

    input_shape = (640, 640)
    image_file = args.image
    image_bgr = cv2.imread(image_file)

    # Preprocess
    results = preprocess_image(image_bgr, input_shape)
    inputs = results["inputs"]
    batch_img_metas = results["metas"]

    input_name = session.get_input_names()[0]
    inputs = inputs.to(exec_device).to(torch.float16)

    # Inference
    batch_nn_out_list = session.run({input_name: inputs})
    batch_nn_out = batch_nn_out_list[0].to(exec_device)

    if batch_nn_out.ndim == 2:
        batch_nn_out = batch_nn_out.unsqueeze(0)

    # Postprocess
    det_results = postprocess_v5(batch_nn_out, conf_thres=0.3, iou_thres=0.5)

    # Visualization
    out_dir = Path("work_dirs/yolov5m")
    out_dir.mkdir(exist_ok=True, parents=True)
    image_fname = Path(image_file).stem

    # 加载类别名称
    coco_cfg_file = str(Path(__file__).parent / "coco_yolov5_eval.yaml")
    classes = yaml_load(coco_cfg_file)["names"]

    # Process detections for the first image in the batch
    det = det_results[0]  # Detections for batch item 0
    if det is not None and len(det):
        # Rescale boxes from img_size to im0 size
        img_meta = batch_img_metas[0]
        scale_ratio = img_meta["scale_factor"]

        # Our preprocessing pads only on the right and bottom, so the top-left padding offset is (0, 0).
        top_left_pad = (0, 0)
        det[:, :4] = scale_coords(
            input_shape, det[:, :4], img_meta["ori_shape"], ratio_pad=((scale_ratio, scale_ratio), top_left_pad)
        ).round()

        for *xyxy, conf, cls in reversed(det):
            if conf < 0.3:
                continue

            x1, y1, x2, y2 = [int(coord) for coord in xyxy]
            class_id = int(cls)

            cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{classes[class_id]}: {conf:.2f}"
            cv2.putText(image_bgr, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    out_file = out_dir / f"{image_fname}_det_vis_v5.jpg"
    cv2.imwrite(str(out_file), image_bgr)
    logger.info(f"Detect result is saved to {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--hmonnx", type=str, default="work_dirs/yolov5m/hmonnx/yolov5m_XH2a.onnx")
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument("--image", type=str, default="data/images/000000001490.jpg")
    args = parser.parse_args()
    main(args)
