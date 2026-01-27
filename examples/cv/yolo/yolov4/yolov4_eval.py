# Copyright 2025 HOUMO AI
#
# File: yolov4_eval.py
# Description:
#   Example script: cv/yolo/yolov4/yolov4_eval.py
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

# yolov4_eval.py
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime
import torch
import torchvision
import yaml
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm


# --- 1. 预处理函数 (保持不变) ---
def preprocess_image_v4(image, input_size):
    """
    为YOLOv4准备图像。
    - 将图像直接缩放到目标尺寸（不保持宽高比）。
    - 将像素值归一化到 [0, 1]。
    - 转换通道顺序 HWC -> CHW。
    """
    ih, iw = image.shape[:2]
    w, h = input_size
    resized_img = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
    img_rgb = resized_img[:, :, ::-1]
    img_chw = np.transpose(img_rgb, (2, 0, 1))
    img_contiguous = np.ascontiguousarray(img_chw, dtype=np.float32)
    img_tensor = torch.from_numpy(img_contiguous)
    img_tensor /= 255.0
    img_tensor = img_tensor.unsqueeze(0)
    metas = [{"ori_shape": (ih, iw), "input_shape": (h, w)}]
    return {"metas": metas, "inputs": img_tensor}


# --- 2. 新的后处理函数 (关键修改) ---
def postprocess_v4_withpp(
    outputs: list[torch.Tensor],
    conf_thres=0.001,
    iou_thres=0.65,
    num_classes=80,  # 保持接口一致性，但可能会被覆盖
    max_det=300,
):
    """
    处理YOLOv4带有内置后处理(withpp)的模型输出。

    Args:
        outputs (list[torch.Tensor]): 模型输出的列表，应包含两个张量:
                                       - outputs[0]: boxes, shape [B, N, 1, 4] or [B, N, 4]
                                       - outputs[1]: confs, shape [B, N, num_classes]
        conf_thres (float): 置信度阈值。
        iou_thres (float): NMS的IoU阈值。
        max_det (int): 每张图片最多保留的检测框数量。

    Returns:
        list[torch.Tensor]: 每个batch图像的最终检测结果列表。
                            每个tensor的形状是 [N, 6]，其中N是检测到的物体数，
                            6列分别是 [x1, y1, x2, y2, score, class_id]。
    """
    # 假设模型输出顺序是 boxes, confs
    # 根据Netron截图，boxes的shape为[B, N, 1, 4]，confs的shape为[B, N, num_classes]
    box_preds = outputs[0]
    conf_preds = outputs[1]

    # 如果boxes的维度是 [B, N, 1, 4]，需要压缩掉多余的维度
    if box_preds.dim() == 4 and box_preds.shape[2] == 1:
        box_preds = box_preds.squeeze(2)

    batch_size = box_preds.shape[0]
    num_classes = conf_preds.shape[2]  # 从confs张量动态获取类别数

    output = [torch.zeros((0, 6), device=box_preds.device)] * batch_size

    for i in range(batch_size):
        # 提取单个图像的预测
        boxes = box_preds[i]  # Shape: [N, 4]
        confs = conf_preds[i]  # Shape: [N, num_classes]

        # 1. 置信度过滤
        # 找到每个框最可能的类别和对应的置信度
        class_scores, class_ids = torch.max(confs, dim=1)

        # 应用置信度阈值
        conf_mask = class_scores > conf_thres

        # 如果没有满足条件的框，则跳到下一张图
        if not conf_mask.any():
            continue

        # 过滤掉低置信度的预测
        boxes = boxes[conf_mask]
        class_scores = class_scores[conf_mask]
        class_ids = class_ids[conf_mask]

        # 2. 按类别进行NMS (与YOLOv4OnnxModel逻辑保持一致)
        final_boxes = []
        final_scores = []
        final_labels = []

        for j in range(num_classes):
            # 筛选出当前类别的所有框
            cls_mask = class_ids == j
            if not cls_mask.any():
                continue

            cls_boxes = boxes[cls_mask]
            cls_scores = class_scores[cls_mask]

            # 对当前类别的框进行NMS
            keep_indices = torchvision.ops.nms(cls_boxes, cls_scores, iou_thres)

            # 保存保留下来的框
            final_boxes.append(cls_boxes[keep_indices])
            final_scores.append(cls_scores[keep_indices])
            final_labels.append(torch.full_like(cls_scores[keep_indices], fill_value=j, dtype=torch.long))

        if not final_boxes:
            continue

        # 将所有类别的NMS结果拼接起来
        final_boxes = torch.cat(final_boxes, dim=0)
        final_scores = torch.cat(final_scores, dim=0)
        final_labels = torch.cat(final_labels, dim=0)

        # 3. 按分数排序并限制最大检测数量
        sort_indices = torch.argsort(final_scores, descending=True)
        if len(sort_indices) > max_det:
            sort_indices = sort_indices[:max_det]

        final_boxes = final_boxes[sort_indices]
        final_scores = final_scores[sort_indices]
        final_labels = final_labels[sort_indices]

        # 4. 组合成 [x1, y1, x2, y2, score, class_id] 格式
        output[i] = torch.cat([final_boxes, final_scores.unsqueeze(1), final_labels.unsqueeze(1).float()], dim=1)

    return output


# --- 3. 推理引擎包装类 (保持不变) ---
class InferenceEngine:
    def __init__(self, model_path, model_type="onnx"):
        self.model_type = model_type
        if self.model_type == "hmonnx":
            try:
                from xhquant.api import HMONNXInference, xhquant_init

                xhquant_init()
                self.session = HMONNXInference(model_path)
                self.session.to(torch.device("cuda"))
            except ImportError:
                raise ImportError("Please install xhquant library to use 'hmonnx' model type.")
            self.input_names = self.session.get_input_names()
            self.output_names = self.session.get_output_names()
        else:
            self.session = onnxruntime.InferenceSession(
                model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            )
            self.input_names = [inp.name for inp in self.session.get_inputs()]
            self.output_names = [out.name for out in self.session.get_outputs()]

    def __call__(self, *inputs_tensors) -> list[torch.Tensor]:
        device = inputs_tensors[0].device
        if self.model_type == "hmonnx":
            cuda_inputs = [t.to(device) for t in inputs_tensors]
            outputs = self.session(*cuda_inputs)
            if not isinstance(outputs, (list, tuple)):
                outputs = [outputs]
            return list(outputs)
        else:
            input_feed = {name: tensor.cpu().numpy() for name, tensor in zip(self.input_names, inputs_tensors)}
            outputs_np = self.session.run(self.output_names, input_feed)
            return [torch.from_numpy(o).to(device) for o in outputs_np]


# --- 4. 主函数 (适配新的后处理函数) ---
def main(args):
    # 加载数据集配置
    with open(args.data) as f:
        data_cfg = yaml.safe_load(f)

    base_path = Path(data_cfg["path"])
    img_dir = base_path / data_cfg.get("val", "images/val2017")
    annotation_file = base_path / data_cfg.get("annotations", "annotations/instances_val2017.json")

    # 初始化推理引擎
    engine = InferenceEngine(args.model, args.model_type)

    # 加载COCO GT数据和类别映射
    coco_gt = COCO(str(annotation_file))
    try:
        class_names = data_cfg["names"]
        coco_cat_ids = coco_gt.getCatIds(catNms=class_names)
        model_cls_to_coco_cat = {i: coco_id for i, coco_id in enumerate(coco_cat_ids)}
    except Exception:
        print("Warning: Could not map class names from YAML. Using direct COCO category IDs.")
        cat_ids = sorted(coco_gt.getCatIds())
        model_cls_to_coco_cat = {i: cat_id for i, cat_id in enumerate(cat_ids)}

    results = []
    image_files = sorted(list(img_dir.glob("*.jpg")))
    print(f"Evaluating model: {args.model} with batch size {args.batch_size} on {len(image_files)} images...")

    for i in tqdm(range(0, len(image_files), args.batch_size), desc="Processing batches"):
        batch_paths = image_files[i : i + args.batch_size]

        if len(batch_paths) != args.batch_size:
            print(f"\nSkipping the last incomplete batch of size {len(batch_paths)}.")
            continue

        batch_inputs, batch_metas = [], []
        for img_path in batch_paths:
            image = cv2.imread(str(img_path))
            if image is None:
                continue

            proc_results = preprocess_image_v4(image, (args.imgsz, args.imgsz))
            batch_inputs.append(proc_results["inputs"])
            meta = proc_results["metas"][0]
            meta["image_id"] = int(img_path.stem.lstrip("0"))
            batch_metas.append(meta)

        inputs_tensor = torch.cat(batch_inputs, dim=0)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        input_tensor_final = inputs_tensor.to(device)
        if args.model_type == "hmonnx":
            input_tensor_final = input_tensor_final.to(torch.float16)

        # 推理
        outputs_list = engine(input_tensor_final)

        # --- 关键修改：调用新的后处理函数 ---
        det_results = postprocess_v4_withpp(
            outputs_list,
            conf_thres=0.001,
            iou_thres=0.65,
            # num_classes参数将从模型输出动态获取，这里可以不传或传一个占位符
        )

        for j, detections in enumerate(det_results):
            img_meta = batch_metas[j]
            ori_h, ori_w = img_meta["ori_shape"]

            if len(detections) == 0:
                continue

            # --- 坐标恢复逻辑修改 ---
            # withpp模型的输出坐标是相对于模型输入尺寸归一化到[0, 1]的
            # 我们需要乘以input_size来得到像素坐标，然后再进行缩放
            boxes = detections[:, :4]
            # 步骤 1: 将归一化的坐标 [0,1] 转换为模型输入尺寸的像素坐标
            boxes[:, [0, 2]] *= args.imgsz  # 乘以宽度
            boxes[:, [1, 3]] *= args.imgsz  # 乘以高度

            # 步骤 2: 将模型输入尺寸的像素坐标，缩放到原始图像尺寸
            scale_w = ori_w / args.imgsz
            scale_h = ori_h / args.imgsz
            boxes[:, [0, 2]] *= scale_w
            boxes[:, [1, 3]] *= scale_h

            boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, ori_w)
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, ori_h)

            scores = detections[:, 4]
            classes = detections[:, 5]

            for k in range(len(detections)):
                box_tensor = boxes[k]
                score = scores[k].item()
                cls_id = int(classes[k].item())

                x1, y1, x2, y2 = box_tensor.tolist()

                coco_bbox = [x1, y1, x2 - x1, y2 - y1]

                results.append(
                    {
                        "image_id": img_meta["image_id"],
                        "category_id": model_cls_to_coco_cat.get(cls_id, -1),
                        "bbox": coco_bbox,
                        "score": round(score, 5),
                    }
                )

    result_json_path = Path(f"detections_yolov4_{Path(args.model).stem}_b{args.batch_size}.json")
    with open(result_json_path, "w") as f:
        json.dump(results, f)
    print(f"\nDetection results saved to {result_json_path}")
    print("Running COCO-style evaluation...")

    coco_dt = coco_gt.loadRes(str(result_json_path))
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate YOLOv4 ONNX/HMONNX models on COCO dataset.")
    SCRIPT_DIR = Path(__file__).parent.resolve()
    DEFAULT_DATA_CONFIG = SCRIPT_DIR / "coco_yolov4_eval.yaml"

    parser.add_argument("--model", type=str, required=True, help="Path to the YOLOv4 model file (.onnx or .hmonnx)")
    parser.add_argument("--model-type", type=str, default="onnx", choices=["onnx", "hmonnx"], help="Model type")
    parser.add_argument(
        "--data", type=str, default=str(DEFAULT_DATA_CONFIG), help="Path to the dataset config file (e.g., coco.yaml)"
    )
    parser.add_argument("--imgsz", type=int, default=416, help="Image size for model input (e.g., 416, 608)")
    parser.add_argument(
        "--batch-size", type=int, default=1, help="Batch size for evaluation (must match static model's batch size)"
    )

    args = parser.parse_args()
    main(args)
