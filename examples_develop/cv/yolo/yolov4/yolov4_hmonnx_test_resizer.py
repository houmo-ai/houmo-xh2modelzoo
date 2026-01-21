# yolov4_hmonnx_test_FIXED.py
import argparse
from locale import normalize
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
import yaml
from xhquant.api import HMONNXInference, get_root_logger, xhquant_init


# --- 1. 修正后的预处理函数 ---
def preprocess_image_stretch(image, input_size, do_normlize=True, output_dtype=torch.float16):
    """
    为YOLOv4 withpp模型准备图像。
    - 直接将图像缩放到目标尺寸（不保持宽高比）。
    - 将像素值归一化到 [0, 1]。
    - 转换通道顺序 HWC -> CHW, BGR -> RGB。
    """
    h, w = image.shape[:2]
    target_w, target_h = input_size

    # 直接、非等比缩放
    resized_img = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    # 转换格式
    img_rgb = resized_img[:, :, ::-1]  # BGR -> RGB
    img_chw = np.transpose(img_rgb, (2, 0, 1))  # HWC -> CHW
    img_contiguous = np.ascontiguousarray(img_chw)

    if do_normlize:
        img_tensor = torch.from_numpy(img_contiguous).unsqueeze(0).float() / 255.0
    else:
        img_tensor = torch.from_numpy(img_contiguous).unsqueeze(0).float()

    # meta信息只需要原始形状
    return {
        "metas": [{"ori_shape": (h, w)}],
        "inputs": img_tensor.to(output_dtype),
    }


# --- 2. 移除复杂的坐标恢复函数，后处理部分保持不变 ---
def postprocess_v4_withpp(outputs, conf_thres, iou_thres, max_det=300):
    # (这个函数本身逻辑是正确的，不需要修改)
    box_preds = outputs[0].float()
    conf_preds = outputs[1].float()
    if box_preds.dim() == 4 and box_preds.shape[2] == 1:
        box_preds = box_preds.squeeze(2)

    boxes = box_preds[0]
    confs = conf_preds[0]

    class_scores, class_ids = torch.max(confs, dim=1)
    conf_mask = class_scores > conf_thres

    if not conf_mask.any():
        return [None]

    boxes = boxes[conf_mask]
    class_scores = class_scores[conf_mask]
    class_ids = class_ids[conf_mask]

    final_boxes_list = []
    final_scores_list = []
    final_labels_list = []

    for j in range(confs.shape[1]):
        cls_mask = class_ids == j
        if not cls_mask.any():
            continue

        cls_boxes = boxes[cls_mask]
        cls_scores = class_scores[cls_mask]

        keep_indices = torchvision.ops.nms(cls_boxes, cls_scores, iou_thres)

        final_boxes_list.append(cls_boxes[keep_indices])
        final_scores_list.append(cls_scores[keep_indices])
        final_labels_list.append(torch.full_like(cls_scores[keep_indices], fill_value=j, dtype=torch.long))

    if not final_boxes_list:
        return [None]

    final_boxes = torch.cat(final_boxes_list, dim=0)
    final_scores = torch.cat(final_scores_list, dim=0)
    final_labels = torch.cat(final_labels_list, dim=0)

    sort_indices = torch.argsort(final_scores, descending=True)
    if len(sort_indices) > max_det:
        sort_indices = sort_indices[:max_det]

    output = torch.cat(
        [
            final_boxes[sort_indices],
            final_scores[sort_indices].unsqueeze(1),
            final_labels[sort_indices].unsqueeze(1).float(),
        ],
        dim=1,
    )

    return [output]


def yaml_load(file="data.yaml"):
    with open(file, errors="ignore", encoding="utf-8") as f:
        return yaml.safe_load(f)


# --- 3. 修正后的主函数 ---
def main(args):
    xhquant_init(None, debug=args.debug)
    session = HMONNXInference(args.hmonnx)
    exec_device = torch.device("cuda")
    session.to(exec_device)
    logger = get_root_logger()
    logger.info("Session is created successfully")

    input_shape = (640, 640)
    image_file = args.image
    image_bgr = cv2.imread(image_file)
    ori_h, ori_w = image_bgr.shape[:2]  # 在预处理前获取原始尺寸

    # --- 使用修正后的预处理 ---
    results = preprocess_image_stretch(image_bgr, input_shape, do_normlize=False, output_dtype=torch.uint8)
    inputs = results["inputs"].to(exec_device)

    # 推理
    batch_nn_out = session(inputs)

    # 后处理
    det_results = postprocess_v4_withpp(batch_nn_out, conf_thres=0.3, iou_thres=0.5)
    det = det_results[0]

    out_dir = Path("work_dirs/yolov4_withpp_test")
    out_dir.mkdir(exist_ok=True, parents=True)
    image_fname = Path(image_file).stem

    coco_cfg_file = str(Path(__file__).parent / "coco_yolov4_eval.yaml")
    classes = yaml_load(coco_cfg_file)["names"]

    if det is not None and len(det):
        # --- 使用修正后的坐标恢复逻辑 ---
        # 输出的坐标是[0,1]归一化的，直接乘以原始图像宽高即可恢复
        det[:, [0, 2]] *= ori_w  # x1, x2 -> 乘以原始宽度
        det[:, [1, 3]] *= ori_h  # y1, y2 -> 乘以原始高度

        # 裁剪到边界
        det[:, [0, 2]] = det[:, [0, 2]].clamp(0, ori_w)
        det[:, [1, 3]] = det[:, [1, 3]].clamp(0, ori_h)

        for *xyxy, conf, cls in reversed(det):
            x1, y1, x2, y2 = [int(coord) for coord in xyxy]
            class_id = int(cls)
            cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{classes[class_id]}: {conf:.2f}"
            cv2.putText(image_bgr, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    out_file = out_dir / f"{image_fname}_det_vis_FIXED.jpg"
    cv2.imwrite(str(out_file), image_bgr)
    logger.info(f"Detect result is saved to {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hmonnx", type=str, default="work_dirs/yolov4_1_3_416_416_static_withpp/hmonnx/yolov4_1_3_416_416_static_withpp_w8a8_sefp_XH2a.onnx"
    )  # 建议先用w8a8测试
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument("--image", type=str, default="data/images/ILSVRC2012_val_00002031.JPEG")
    args = parser.parse_args()
    main(args)
