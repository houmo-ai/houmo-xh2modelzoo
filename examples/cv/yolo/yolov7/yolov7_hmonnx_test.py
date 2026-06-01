# Copyright 2025 HOUMO AI
#
# File: yolov7_hmonnx_test.py
# Description:
#   Example script: cv/yolo/yolov7/yolov7_hmonnx_test.py
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

# 文件名: examples/cv/yolo/yolov7/yolov7_hmonnx_test.py
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


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)


def preprocess_image(image, input_img_size) -> Dict[str, Any]:
    # 图像预处理 (Letterbox)
    img_resized, ratio, pad = letterbox(image, input_img_size, auto=False)

    # BGR to RGB, HWC to CHW
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_chw = img_rgb.transpose((2, 0, 1))  # HWC to CHW
    img_contiguous = np.ascontiguousarray(img_chw)

    img_tensor = torch.from_numpy(img_contiguous)
    img_tensor = img_tensor.float() / 255.0  # 0 - 255 to 0.0 - 1.0
    img_tensor = img_tensor.unsqueeze(0).to(torch.float16)

    return {
        "metas": [
            {"ori_shape": image.shape, "resized_shape": img_resized.shape, "ratio": ratio, "pad": pad}  # h, w, c
        ],
        "inputs": img_tensor,
    }


def box_iou(box1, box2, eps=1e-7):
    (a1, a2), (b1, b2) = box1.chunk(2, 2), box2.chunk(2, 1)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)
    return inter / (box_area(box1.T)[:, None] + box_area(box2.T) - inter + eps)


def box_area(box):
    return (box[2] - box[0]) * (box[3] - box[1])


def xywh2xyxy(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def non_max_suppression(
    prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False, max_det=300
):
    nc = prediction.shape[2] - 5
    xc = prediction[..., 4] > conf_thres

    max_wh = 7680
    max_nms = 30000
    redundant = True
    multi_label &= nc > 1
    merge = False

    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):
        x = x[xc[xi]]

        if not x.shape[0]:
            continue

        x[:, 5:] *= x[:, 4:5]
        box = xywh2xyxy(x[:, :4])

        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        n = x.shape[0]
        if not n:
            continue
        elif n > max_nms:
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]

        c = x[:, 5:6] * (0 if agnostic else max_wh)
        boxes, scores = x[:, :4] + c, x[:, 4]
        i = torchvision.ops.nms(boxes, scores, iou_thres)
        if i.shape[0] > max_det:
            i = i[:max_det]
        if merge and (1 < n < 3e3):
            iou = box_iou(boxes[i], boxes) > iou_thres
            weights = iou * scores[None]
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)
            if redundant:
                i = i[iou.sum(1) > 1]

        output[xi] = x[i]

    return output


def _make_grid(nx=20, ny=20):
    yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)], indexing="ij")
    return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


def postprocess_yolov7(outputs: Tuple[torch.Tensor], batch_img_metas: List[Dict[str, Any]]):
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
        bs, na, ny, nx, no = out.shape
        grid = _make_grid(nx, ny).to(device)
        anchor_grid = anchor.view(1, -1, 1, 1, 2)
        y = out.sigmoid()
        y[..., 0:2] = (y[..., 0:2] * 2.0 - 0.5 + grid) * stride
        y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * anchor_grid
        predictions.append(y.reshape(bs, -1, 85))

    prediction = torch.cat(predictions, 1)

    # <<< START DEBUGGING BLOCK >>>
    logger = get_root_logger()
    logger.info("-" * 50)
    logger.info("DEBUGGING MODEL'S RAW OUTPUT (Pre-NMS)")

    pred_for_debug = prediction[0]

    objectness_scores = pred_for_debug[:, 4]
    max_obj_score, max_obj_idx = torch.max(objectness_scores, dim=0)
    best_prediction_vector = pred_for_debug[max_obj_idx]

    logger.info(f"Total candidate boxes: {pred_for_debug.shape[0]}")
    logger.info(f"Highest objectness score found: {max_obj_score:.4f}")

    class_scores = best_prediction_vector[5:]
    obj_x_class_scores = class_scores * max_obj_score

    max_class_prob, max_class_idx = torch.max(obj_x_class_scores, dim=0)

    coco_cfg_file = str(Path(__file__).parent / "coco8.yaml")
    classes = yaml_load(coco_cfg_file)["names"]

    logger.info(
        f"For the box with highest objectness, the most likely class is: '{classes[max_class_idx.item()]}' (index {max_class_idx.item()}) with final score {max_class_prob:.4f}"
    )

    surfboard_index = 38
    surfboard_score = obj_x_class_scores[surfboard_index]
    logger.info(f"Score for 'surfboard' (index {surfboard_index}) for this box is: {surfboard_score:.4f}")

    top5_probs, top5_indices = torch.topk(obj_x_class_scores, 5)
    logger.info("Top 5 likely classes for this best box:")
    for prob, idx in zip(top5_probs, top5_indices):
        logger.info(f"  - {classes[idx.item()]} (index {idx.item()}): {prob:.4f}")

    logger.info("-" * 50)
    # <<< END DEBUGGING BLOCK >>>

    detections = non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45)

    det = detections[0]
    img_meta = batch_img_metas[0]
    det_boxes = [{"boxes": [], "scores": [], "labels": []}]

    if det is not None and len(det):
        padded_shape = img_meta["resized_shape"]
        det[:, :4] = scale_coords(
            padded_shape, det[:, :4], img_meta["ori_shape"], ratio_pad=(img_meta["ratio"], img_meta["pad"])
        ).round()

        for *xyxy, conf, cls in reversed(det):
            det_boxes[0]["boxes"].append([int(c) for c in xyxy])
            det_boxes[0]["scores"].append(conf.item())
            det_boxes[0]["labels"].append(int(cls.item()))

    det_boxes[0]["boxes"] = torch.tensor(det_boxes[0]["boxes"])
    det_boxes[0]["scores"] = torch.tensor(det_boxes[0]["scores"])
    det_boxes[0]["labels"] = torch.tensor(det_boxes[0]["labels"])

    return det_boxes


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]
    coords[:, [1, 3]] -= pad[1]
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def clip_coords(boxes, img_shape):
    boxes[:, 0].clamp_(0, img_shape[1])
    boxes[:, 1].clamp_(0, img_shape[0])
    boxes[:, 2].clamp_(0, img_shape[1])
    boxes[:, 3].clamp_(0, img_shape[0])


def yaml_load(file="data.yaml", append_filename=False):
    assert Path(file).suffix in {".yaml", ".yml"}, f"Attempting to load non-YAML file {file} with yaml_load()"
    with open(file, errors="ignore", encoding="utf-8") as f:
        s = f.read()
        if not s.isprintable():
            s = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\x85\xA0-\uD7FF\uE000-\uFFFD\U00010000-\U0010ffff]+", "", s)
        data = yaml.safe_load(s) or {}
        if append_filename:
            data["yaml_file"] = str(file)
        return data


def main(args):
    xhquant_init(None, debug=args.debug)
    session = HMONNXInference(args.hmonnx)
    exec_device = torch.device("cuda")
    session.to(exec_device)
    logger = get_root_logger()
    logger.info("session is created successfully")

    input_shape = (640, 640)
    image_file = args.image
    image = cv2.imread(image_file)
    results = preprocess_image(image, input_shape)
    inputs = results["inputs"]
    batch_img_metas = results["metas"]

    input_name = session.get_input_names()[0]
    inputs = inputs.to(exec_device)

    batch_nn_out = session.run({input_name: inputs})

    batch_nn_out_cpu = [out.cpu() for out in batch_nn_out]
    det_boxes = postprocess_yolov7(batch_nn_out_cpu, batch_img_metas)

    det_boxes = det_boxes[0]

    out_dir = Path("work_dirs/yolov7")
    out_dir.mkdir(exist_ok=True, parents=True)
    image_fname = Path(image_file).stem
    image_to_draw = cv2.imread(image_file)

    coco_cfg_file = str(Path(__file__).parent / "coco8.yaml")
    classes = yaml_load(coco_cfg_file)["names"]

    for box, score, label in zip(det_boxes["boxes"], det_boxes["scores"], det_boxes["labels"]):
        if score < 0.25:
            continue
        box = box.int().tolist()
        x1, y1, x2, y2 = box
        class_id = int(label)
        cv2.rectangle(image_to_draw, (x1, y1), (x2, y2), (0, 255, 0), 2)

        label_text = f"{classes[class_id]}: {score:.2f}"
        (label_width, label_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_x = x1
        label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10
        cv2.putText(
            image_to_draw, label_text, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA
        )

    out_file = out_dir / f"{image_fname}_det_vis.jpg"
    cv2.imwrite(str(out_file), image_to_draw)
    logger.info(f"Detect result is saved to {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hmonnx", type=str, default="work_dirs/yolov7/hmonnx/yolov7_XH2a.onnx")
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument("--image", type=str, default="data/images/000000001490.jpg")
    args = parser.parse_args()
    main(args)
